# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Running a connection on another node, without a second class to write.

A connection with a ``node_rank`` is rebuilt inside a scheduler worker on that
node, and the object you are holding *becomes* a view onto it: same class, same
methods, same ``isinstance``, but every public call now travels. Nothing about
a driver changes to make that work, and there is no remote counterpart to keep
in step with it -- both halves are derived from the driver class itself.

The worker side already worked that way. ``WorkerGroup`` binds every public
method of the class it hosts, so a hosted part exposes its whole surface --
``get_observation`` and ``send_action`` alongside ``is_robot_up`` or
``reset_joint`` -- with no delegation written by hand. What used to be missing
was the other side: a table named one hand-written proxy class per device
category, so a new category meant a new proxy, and a category the table did not
know arrived as a plain part.

:func:`remote_view_of` closes that by deriving the proxy from the class rather
than from a name for it. Two things do not survive the trip:

* **Properties.** ``WorkerGroup`` binds callables, and a property is not one,
  so ``observation_features`` and friends would simply be missing. The worker
  gets an :meth:`attribute` call and the view's properties use it.
* **Anything that owns the worker.** ``connect``, ``disconnect`` and the
  composition surface stay local, or a view would ask the worker to shut itself
  down. :data:`_STAYS_LOCAL` is that list, and it is short on purpose.

This is the only module in ``rlinf.robotics`` that imports the scheduler.
:meth:`Connection.connect` reaches it lazily.
"""

import inspect
from typing import Any, Callable, ClassVar, Optional
from uuid import uuid4

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.scheduler.worker.worker import WorkerMeta

from ..parts.base import Connection, _ConnectionMeta

#: Names a remote view answers itself instead of forwarding.
#:
#: Lifecycle, because a view that forwarded ``disconnect`` would ask the worker
#: to end itself and then still be holding it. Composition, because the parts a
#: connection backs are local view objects wrapping *this* object -- sending
#: them over the wire would hand back copies bound to the far side. And the
#: placement surface, which is what got the object here to begin with.
_STAYS_LOCAL: frozenset[str] = frozenset(
    {
        "connect",
        "disconnect",
        "is_connected",
        "owner",
        "node_rank",
        "part",
        "parts",
        "children",
        "child",
    }
)

#: The RPC a view reads properties through. Public so ``WorkerGroup`` binds it.
_ATTRIBUTE_RPC = "attribute"


def _first(result: Any) -> Any:
    """Unwrap a one-worker result, rejecting groups sized for more."""
    values = result.wait()
    if len(values) != 1:
        raise RuntimeError(
            f"A part must be hosted by exactly one worker, got {len(values)} results."
        )
    return values[0]


def _forwarding_method(name: str) -> Callable[..., Any]:
    """A method that runs ``name`` on the worker and returns what it returned."""

    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _first(getattr(self._group, name)(*args, **kwargs))

    call.__name__ = name
    call.__qualname__ = name
    call.__doc__ = f"Run ``{name}`` on the node holding this connection."
    return call


def _forwarding_property(name: str) -> property:
    """A property that reads ``name`` off the hosted object."""

    def read(self: Any) -> Any:
        return _first(getattr(self._group, _ATTRIBUTE_RPC)(name))

    read.__name__ = name
    read.__doc__ = f"``{name}`` as the node holding this connection reports it."
    return property(read)


def _public_surface(part_cls: type) -> tuple[list[str], list[str]]:
    """The methods and properties of ``part_cls`` that should travel.

    Everything public that the class or its bases define, minus what
    :data:`_STAYS_LOCAL` keeps here. Taken from the class rather than from an
    instance so it is known before anything is built.

    Class methods are left out. ``register``, ``backends`` and ``discover``
    answer questions about the driver registry rather than about a device, and
    the answer is the same on either machine -- forwarding them would send a
    question about this process to a worker that would answer for its own.
    """
    methods: list[str] = []
    properties: list[str] = []
    for name in dir(part_cls):
        if name.startswith("_") or name in _STAYS_LOCAL:
            continue
        attribute = inspect.getattr_static(part_cls, name, None)
        if isinstance(attribute, property):
            properties.append(name)
        elif isinstance(attribute, (classmethod, staticmethod)):
            continue
        elif callable(getattr(part_cls, name, None)):
            methods.append(name)
    return methods, properties


class _RemoteViewMeta(WorkerMeta, _ConnectionMeta):
    """Reconcile the metaclasses a synthesized view inherits."""


#: One synthesized view per part class, reused after the first placement.
_VIEWS: dict[type, type] = {}


def remote_view_of(part_cls: type) -> type:
    """Return (and cache) the class a ``part_cls`` becomes when placed.

    A subclass of ``part_cls`` whose public methods and properties forward to
    the worker holding the real one. Being a subclass is the point: a placed
    camera still satisfies ``isinstance(part, Camera)``, so
    ``robot.parts_of_type(Camera)`` and the ``ControllablePart`` filter that
    builds an action tree keep working with nothing registered anywhere.
    """
    cached = _VIEWS.get(part_cls)
    if cached is not None:
        return cached

    methods, properties = _public_surface(part_cls)
    namespace: dict[str, Any] = {
        "__module__": part_cls.__module__,
        "__qualname__": f"Remote{part_cls.__name__}",
        "__doc__": (
            f"{part_cls.__name__} running on another node.\n\n"
            "Synthesized: every public method and property forwards to the "
            "worker holding the real one."
        ),
    }
    for name in methods:
        namespace[name] = _forwarding_method(name)
    for name in properties:
        namespace[name] = _forwarding_property(name)

    view = _RemoteViewMeta(f"Remote{part_cls.__name__}", (part_cls,), namespace)
    _VIEWS[part_cls] = view
    return view


def _refuse_collisions(part_cls: type, methods: dict[str, Any]) -> None:
    """Refuse to host a part whose method names the worker already uses.

    The part's methods are re-declared in the worker's class body, so one
    sharing a name with :class:`Worker` -- or with :data:`_ATTRIBUTE_RPC`, the
    call a view reads properties through -- would replace it. That breaks the
    worker rather than the part, at some later point and with no mention of the
    method that did it, so it is refused here where the name is still in hand.
    """
    taken = {name for name in dir(Worker) if not name.startswith("_")}
    taken.add(_ATTRIBUTE_RPC)
    clashing = sorted(set(methods) & taken)
    if clashing:
        raise TypeError(
            f"{part_cls.__name__} cannot be placed on a node: its methods "
            f"{clashing} share a name with the worker that would host it. "
            "Rename them, or keep them private."
        )


class PartWorkerHost:
    """Hosts one connection in a scheduler worker and hands back its group.

    Synthesising the worker class, naming the group and launching it are one
    job with one set of inputs, so they are one object rather than three
    functions passing the same three arguments around.

    Args:
        part_cls: The class to construct on the target node.
        args: Positional arguments for its constructor.
        kwargs: Keyword arguments for its constructor.
        node_rank: Cluster node rank that is physically wired to the device.
        worker_name: Worker-group name. Defaults to :meth:`default_name`.
    """

    #: One synthesised ``Worker`` subclass per part class, reused after the
    #: first placement of that class.
    _worker_classes: ClassVar[dict[type, type]] = {}

    def __init__(
        self,
        part_cls: type,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
        *,
        node_rank: int,
        worker_name: Optional[str] = None,
    ) -> None:
        self.part_cls = part_cls
        self.args = args
        self.kwargs = kwargs or {}
        self.node_rank = node_rank
        self.worker_name = worker_name or self.default_name(part_cls, node_rank)

    @staticmethod
    def default_name(part_cls: type, node_rank: int) -> str:
        """Name a hosted part so no two of them collide.

        The class and the node are not enough on their own: a robot with a
        camera on each wrist places two of the same class on the same node,
        and the same robot built in two env workers places the whole tree
        twice. Ray refuses a duplicate name rather than placing the second
        part, so the fresh suffix is what lets either happen at all.
        """
        return f"{part_cls.__name__}-node{node_rank}-{uuid4().hex[:8]}"

    @classmethod
    def worker_class(cls, part_cls: type) -> type:
        """Return (and cache) the ``Worker`` subclass that hosts ``part_cls``."""
        cached = cls._worker_classes.get(part_cls)
        if cached is not None:
            return cached

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            Worker.__init__(self)
            part_cls.__init__(self, *args, **kwargs)
            self.connect()

        def attribute(self: Any, name: str) -> Any:
            """Read one attribute of the hosted part.

            ``WorkerGroup`` binds callables, and a property is not one, so
            without this a remote view could not answer
            ``observation_features`` -- the one thing an env needs before it
            can build a space.
            """
            return getattr(self, name)

        namespace: dict[str, Any] = {
            "__init__": __init__,
            attribute.__name__: attribute,
            "__module__": part_cls.__module__,
            "__qualname__": f"{part_cls.__name__}Worker",
            "__doc__": f"Scheduler host for :class:`{part_cls.__name__}`.",
        }

        # Re-declare the part's public methods in the new class body so
        # ``WorkerMeta`` wraps them for failure capture; inherited attributes
        # are invisible to it. This is a loop, not hand-written delegation, and
        # the bodies stay in the part.
        methods = {
            name: func
            for name, func in inspect.getmembers(part_cls, inspect.isfunction)
            if not name.startswith("_")
        }
        _refuse_collisions(part_cls, methods)
        namespace.update(methods)

        worker_cls = _RemoteViewMeta(
            f"{part_cls.__name__}Worker", (Worker, part_cls), namespace
        )
        cls._worker_classes[part_cls] = worker_cls
        return worker_cls

    def launch(self) -> Any:
        """Start the worker and return the group that reaches it."""
        return (
            self.worker_class(self.part_cls)
            .create_group(*self.args, **self.kwargs)
            .launch(
                cluster=Cluster(),
                placement_strategy=NodePlacementStrategy(node_ranks=[self.node_rank]),
                name=self.worker_name,
            )
        )


def host(connection: Connection) -> tuple[Any, type]:
    """Rebuild ``connection`` on the node it named, and say what it becomes.

    Returns the worker group and the class the caller should take on, so the
    object the robot already holds turns into a view of the hosted one rather
    than being replaced by something else.
    """
    recipe = connection._recipe
    group = PartWorkerHost(
        recipe.part_cls,
        recipe.args,
        recipe.kwargs,
        node_rank=recipe.node_rank,
        worker_name=recipe.worker_name,
    ).launch()
    return group, remote_view_of(recipe.part_cls)


def shutdown(group: Any) -> None:
    """Close the hosted connection, then terminate the worker holding it.

    In that order: killing the actor first would leave the device open with
    nothing left to close it -- a camera still streaming, a gripper still
    powered -- until something else on that node claimed the handle.
    """
    try:
        group.disconnect().wait()
    finally:
        close = getattr(group, "_close", None)
        if callable(close):
            close()
