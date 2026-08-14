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

"""Placing a part on a node, and reaching it once it is there.

Any :class:`~rlinf.robotics.parts.base.RobotPart` can be hosted in a scheduler
worker: an arm on the machine wired to it, a camera on the machine it is
plugged into. :meth:`RobotPart.spawn` is the entry point; everything here is
what it returns.

There is no per-hardware worker class. For any part, :func:`part_worker_cls`
synthesizes ``type(name, (Worker, PartCls), ...)`` once and caches it.
``WorkerGroup`` then binds every public method of that class as an RPC, so a
hosted part exposes its whole surface -- the subpart-addressed calls *and*
hardware methods like ``is_robot_up`` or ``reset_joint`` -- with no delegation
written by hand.

A handle answers two questions the same way whether the part runs here or in a
worker: *what subparts does it expose*, and *how do I call a method that is not
on the part interface*. Off-interface calls always return a result object with
``wait() -> list``, so call sites read identically::

    handle.is_robot_up().wait()[0]

This is the only module in ``rlinf.robotics`` that imports the scheduler.
``RobotPart.spawn`` imports it lazily, so importing a part never loads Ray.
"""

import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Optional

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.scheduler.worker.worker import WorkerMeta

from .parts.base import Camera, ControllablePart, EndEffector, RobotPart


class LocalResult:
    """A value computed in this process, shaped like a worker-group result."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def wait(self) -> list[Any]:
        """Return the value in the one-element list callers expect."""
        return [self._value]


class PartHandle(ABC):
    """Common surface over a local part or a hosted one."""

    @property
    @abstractmethod
    def subparts(self) -> dict[str, RobotPart]:
        """Return the subparts of the hosted part, keyed by its local names."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the connection and, when hosted, its worker."""

    def subpart(self, name: str) -> RobotPart:
        """Return one named subpart, or raise a clear configuration error."""
        if name not in self.subparts:
            raise KeyError(
                f"Hosted part exposes no subpart {name!r}. "
                f"Available: {sorted(self.subparts)}."
            )
        return self.subparts[name]


class LocalPartHandle(PartHandle):
    """Handle for a part constructed in this process."""

    def __init__(self, part: Any) -> None:
        self._part = part
        self._parts = dict(part.subparts())

    @property
    def subparts(self) -> dict[str, RobotPart]:
        """Return the local part's own subpart objects."""
        return self._parts

    @property
    def part(self) -> Any:
        """Return the underlying part, for code that legitimately needs it."""
        return self._part

    def disconnect(self) -> None:
        """Disconnect the part if it is still connected."""
        if self._part.is_connected:
            self._part.disconnect()

    def __getattr__(self, name: str) -> Any:
        """Forward off-interface hardware methods, wrapping results for symmetry."""
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._part, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> LocalResult:
            return LocalResult(attr(*args, **kwargs))

        return call


class RemotePartHandle(PartHandle):
    """Handle for a part hosted in a one-worker scheduler group."""

    def __init__(self, worker_group: Any, described: dict[str, dict[str, Any]]) -> None:
        self._worker_group = worker_group
        self._parts: dict[str, RobotPart] = {
            name: _make_remote_part(worker_group, name, entry)
            for name, entry in described.items()
        }
        self._connected = True

    @property
    def subparts(self) -> dict[str, RobotPart]:
        """Return proxies for the hosted part's subparts."""
        return self._parts

    @property
    def worker_group(self) -> Any:
        """Return the underlying worker group."""
        return self._worker_group

    def disconnect(self) -> None:
        """Shut the hosted part down and terminate its worker."""
        if not self._connected:
            return
        self._connected = False
        try:
            self._worker_group.shutdown().wait()
        finally:
            close = getattr(self._worker_group, "_close", None)
            if callable(close):
                close()

    def __getattr__(self, name: str) -> Any:
        """Forward off-interface hardware methods to the worker group.

        ``WorkerGroup`` binds every public method of the hosted class, so this
        reaches ``is_robot_up``, ``reset_joint``, ``clear_errors`` and friends
        with no per-hardware declaration.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._worker_group, name)


def _first(result: Any) -> Any:
    """Unwrap a one-worker result, rejecting groups sized for more."""
    values = result.wait()
    if len(values) != 1:
        raise RuntimeError(
            f"A part must be hosted by exactly one worker, got {len(values)} results."
        )
    return values[0]


class RemotePart(RobotPart):
    """One subpart of a hosted part, addressed by name through its worker."""

    def __init__(
        self,
        worker_group: Any,
        part_name: str,
        observation_features: dict[str, Any],
    ) -> None:
        self._worker_group = worker_group
        self._part_name = part_name
        self._observation_features = observation_features

    @property
    def is_connected(self) -> bool:
        """Subparts of a hosted part are live for as long as it is."""
        return True

    @property
    def observation_features(self) -> dict[str, Any]:
        """Return the features captured when the host was described."""
        return self._observation_features

    def connect(self) -> None:
        """No-op: the hosted part connects when its worker starts."""

    def reset(self) -> None:
        """Reset this subpart through its host."""
        self._worker_group.subpart_reset(self._part_name).wait()

    def get_observation(self) -> dict[str, Any]:
        """Read this subpart's observation through its host."""
        return _first(self._worker_group.subpart_observation(self._part_name))

    def disconnect(self) -> None:
        """No-op: the handle owns the hosted part's lifetime."""


class RemoteControllablePart(RemotePart, ControllablePart):
    """Hosted part that also accepts actions."""

    def __init__(
        self,
        worker_group: Any,
        part_name: str,
        observation_features: dict[str, Any],
        action_features: dict[str, Any],
    ) -> None:
        super().__init__(worker_group, part_name, observation_features)
        self._action_features = action_features

    @property
    def action_features(self) -> dict[str, Any]:
        """Return the action features captured at describe time."""
        return self._action_features

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action to this subpart through its host."""
        return _first(self._worker_group.subpart_action(self._part_name, action))


class RemoteEndEffector(RemoteControllablePart, EndEffector):
    """Hosted end effector."""


class RemoteCamera(RemotePart, Camera):
    """Hosted camera."""


_REMOTE_PART_BY_KIND: dict[str, Callable[..., RemotePart]] = {
    "end_effector": RemoteEndEffector,
    "camera": RemoteCamera,
    "controllable": RemoteControllablePart,
    "part": RemotePart,
}


def _make_remote_part(
    worker_group: Any,
    name: str,
    described: dict[str, Any],
) -> RemotePart:
    """Build a proxy mirroring the interface of the hosted part."""
    kind = described["kind"]
    part_cls = _REMOTE_PART_BY_KIND[kind]
    if kind in ("end_effector", "controllable"):
        return part_cls(
            worker_group,
            name,
            described["observation"],
            described["action"],
        )
    return part_cls(worker_group, name, described["observation"])


class WorkerPartMeta(WorkerMeta, ABCMeta):
    """Reconcile ``Worker``'s metaclass with the ``ABCMeta`` drivers carry.

    ``Worker`` uses ``WorkerMeta(type)`` and every part is an ABC, so a class
    deriving from both needs a metaclass deriving from both.
    """


_WORKER_CLS_CACHE: dict[type, type] = {}


def part_worker_cls(part_cls: type) -> type:
    """Return (and cache) the ``Worker`` subclass that hosts ``part_cls``."""
    cached = _WORKER_CLS_CACHE.get(part_cls)
    if cached is not None:
        return cached

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        Worker.__init__(self)
        part_cls.__init__(self, *args, **kwargs)
        self.connect()

    namespace: dict[str, Any] = {
        "__init__": __init__,
        "__module__": part_cls.__module__,
        "__qualname__": f"{part_cls.__name__}Worker",
        "__doc__": f"Scheduler host for :class:`{part_cls.__name__}`.",
    }

    # Re-declare the part's public methods in the new class body so
    # ``WorkerMeta`` wraps them for failure capture; inherited attributes are
    # invisible to it. This is a loop, not hand-written delegation, and the
    # bodies stay in the part.
    for name, func in inspect.getmembers(part_cls, inspect.isfunction):
        if not name.startswith("_") and name not in namespace:
            namespace[name] = func

    worker_cls = WorkerPartMeta(
        f"{part_cls.__name__}Worker",
        (Worker, part_cls),
        namespace,
    )
    _WORKER_CLS_CACHE[part_cls] = worker_cls
    return worker_cls


def spawn_part_worker(
    part_cls: type,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    node_rank: int,
    name: Optional[str] = None,
) -> RemotePartHandle:
    """Host one part on ``node_rank`` and return a handle to it.

    Args:
        part_cls: The part class to construct on the target node.
        args: Positional arguments for the part constructor.
        kwargs: Keyword arguments for the part constructor.
        node_rank: Cluster node rank that is physically wired to the device.
        name: Worker-group name. Must be unique across concurrently running
            parts; callers that spawn one part per environment should
            include the environment index.

    Returns:
        RemotePartHandle: Handle whose parts proxy to the hosted part.
    """
    worker_cls = part_worker_cls(part_cls)
    group = worker_cls.create_group(*args, **kwargs).launch(
        cluster=Cluster(),
        placement_strategy=NodePlacementStrategy(node_ranks=[node_rank]),
        name=name or f"{part_cls.__name__}-node{node_rank}",
    )
    return RemotePartHandle(group, _first(group.describe_subparts()))
