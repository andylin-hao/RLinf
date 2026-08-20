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

"""What a robot is built from: connections, the parts they back, and the tree.

The whole file turns on one question -- *does reading this thing mean
anything?* -- asked of one idea:

* :class:`Connection` -- one link to hardware. It knows the machine it runs on,
  and it opens and closes. Nothing else. Subclass it directly when a single
  link backs several components without being any of them: a ROS node driving
  two arms, a CAN bus, a vendor SDK session.
* :class:`RobotPart` -- a connection you *can* read, and therefore a component
  of the robot: an arm, a camera, a gripper. :class:`ControllablePart` adds
  commands.
* :class:`PartGroup` -- a part made of named parts, which is how a robot is
  composed. :class:`~rlinf.robotics.robot.Robot` is the outermost one, and a
  group holds parts only: hand it a bare connection and it says so.

An arm is both a connection and a part, and that is not a contradiction:
``FrankaROSArm`` *is* the ROS link to the arm, so it is a link that happens to
be readable. A link driving four components is one that is not.

Device categories are *not* here. ``Camera`` lives in
:mod:`rlinf.robotics.parts.cameras`, ``EndEffector`` in
:mod:`rlinf.robotics.parts.end_effectors`, and ``MobileBase`` in
:mod:`rlinf.robotics.parts.mobility`, each beside the drivers that implement it.
Categories with a specialized remote surface announce it with
:func:`register_kind`; a mobile base adds no methods beyond
``ControllablePart`` and uses that standard proxy. This module is the taxonomy,
not the catalogue, so importing it pulls in nothing about hardware a node does
not have.

Composing is inert. Constructing a connection records its arguments and the
node it belongs to; :meth:`~rlinf.robotics.robot.Robot.connect` is the only
thing that opens anything. That is what lets a robot be composed and described
on a laptop, then run on the machine wired to the hardware.

No module here imports ``rlinf.scheduler`` or Gymnasium. The dependency runs
one way: the scheduler is a general framework that reaches robotics by name
from a config and through a registry, never by importing it, and Gymnasium
belongs to the env layer that consumes a robot. Only the composition layer
imports either back, and :meth:`Connection.place` reaches the placement layer
lazily so that stays true of parts.

Ray is not on that list -- it is a base dependency of the package, so the name
is allowed. Nor is this a promise that importing a part loads nothing: a part
may use ``rlinf.utils`` helpers such as ``get_logger``, which reach further.
"""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, TypeVar

if TYPE_CHECKING:
    from ..placement import PartHandle

KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


def run_parallel(
    jobs: Mapping[KeyType, Callable[[], ValueType]],
) -> dict[KeyType, ValueType]:
    """Run independent part operations concurrently, keyed by component name.

    Parts on separate connections do not contend, so reading or commanding
    several of them costs one round trip rather than the sum. A single job runs
    inline to keep the common one-arm case free of thread overhead.
    """
    if len(jobs) <= 1:
        return {key: job() for key, job in jobs.items()}
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {key: executor.submit(job) for key, job in jobs.items()}
        return {key: future.result() for key, future in futures.items()}


@dataclass(frozen=True)
class _Recipe:
    """The arguments a connection was built from, and the machine it named.

    :class:`_ConnectionMeta` records one on every connection. A part whose vendor
    SDK exists only on the machine holding the hardware cannot be built here
    and moved there, so what travels is the recipe: the class and its
    arguments, rebuilt on the far side.
    """

    part_cls: type
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    node_rank: Optional[int] = None
    worker_name: Optional[str] = None


@dataclass(frozen=True, eq=False)
class _ExportRef:
    """A part picked out of a connection that is not open yet.

    ``connection.part("arm")`` evaluates to one of these.
    :meth:`PartGroup.resolve` swaps it for the real part once the connection
    behind it has been opened, so this only exists between composing a robot
    and connecting it.

    Internal, and deliberately unnamed in the public API: a robot author writes
    ``part("arm")`` and composes the result, and never has a reason to say what
    type that is.
    """

    connection: "Connection"
    name: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self.connection).__name__}.{self.name}, not open yet>"


class _ConnectionMeta(ABCMeta):
    """Take ``node_rank`` and ``worker_name`` before ``__init__`` sees them.

    Where a part runs is the base class's business, not the part's. Removing
    the two keywords here is what lets every connection accept them without
    writing a line: an author declares the constructor their hardware needs,
    and ``ExampleArm("10.0.0.2", node_rank=1)`` still says which machine it
    runs on.

    The alternative is a pair of parameters in every constructor, forwarded up
    through ``super().__init__`` -- which is both noise in a signature that
    should describe hardware, and one more thing for a new driver to get wrong
    silently.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Build the connection, and remember how, so it can be built elsewhere."""
        node_rank = worker_name = None
        if cls._TAKES_PLACEMENT:
            node_rank = kwargs.pop("node_rank", None)
            worker_name = kwargs.pop("worker_name", None)
        connection = super().__call__(*args, **kwargs)
        connection._recipe = _Recipe(cls, args, dict(kwargs), node_rank, worker_name)
        return connection


#: Device categories, by the name a remote proxy is rebuilt from. Registered
#: rather than listed, because this module is the taxonomy and not the
#: catalogue: cameras live in ``parts/cameras``, end effectors in
#: ``parts/end_effectors``, and neither is imported here.
_PART_KINDS: dict[str, type] = {}


def register_kind(kind: str) -> Callable[[type], type]:
    """Name a device category, so a hosted part comes back as one.

    Placement sends a part's :attr:`Connection.kind` across the process
    boundary, and the handle on the far side builds a proxy of the matching
    class. Registering is what lets a camera hosted on another machine arrive
    with ``get_frame`` on it rather than as a part in general::

        @register_kind("camera")
        class Camera(RobotPart): ...

    Decorate the category, not a driver: one name covers every camera. A
    category nobody registers still works, and simply arrives as the nearest
    ancestor that is registered.
    """

    def register(part_cls: type) -> type:
        _PART_KINDS[kind] = part_cls
        return part_cls

    return register


@register_kind("connection")
class Connection(ABC, metaclass=_ConnectionMeta):
    """One link to hardware: which machine it runs on, and when it is open.

    A camera's USB handle, a ROS session driving two arms, a gripper on a
    serial port. What these have in common, and all a connection is, is those
    two things -- a machine the link belongs to, and a lifecycle, where
    :meth:`_open` reaches the hardware and :meth:`_release` lets it go.
    Everything else about a device belongs to the subclasses.

    Constructing a connection declares it. It does not open it::

        arm = ExampleArm("10.0.0.2", node_rank=1)

    ``node_rank`` names the machine the device is wired to. Every connection
    accepts it and none declares it: :class:`_ConnectionMeta` takes it out of
    the keywords before the constructor runs. Leave it out and the connection
    runs in this process. Opening waits for :meth:`Robot.connect`, which for a
    remote connection also rebuilds it on the node it named.

    A constructor therefore stores its settings and does nothing else: no SDK
    import, no socket, no thread. That was already the rule, since a part has to
    be importable on a machine without its vendor library. Placement now depends
    on it too.

    Subclass this directly when one link backs several components without being
    any of them -- a ROS node, a CAN bus, an SDK session. Reading such a link
    would mean nothing, so it lists what rides on it in :attr:`parts` and a
    robot composes those::

        session = Turtle2Connection(50, node_rank=0)
        robot = Turtle2Robot(
            left=PartGroup(
                arm=session.part("left"),
                end_effector=session.part("left_end_effector"),
            ),
        )

    When reading the link *does* mean something -- a ROS session for one arm --
    subclass :class:`RobotPart` instead and list yourself in :attr:`parts`. That
    is the only question worth asking: does an observation of the whole thing
    say anything a policy can use.

    A connection whose lifecycle is more than opening a device -- an arm that
    must home before it is usable -- overrides :meth:`connect` and
    :meth:`disconnect` rather than the two hooks.
    """

    #: The vendor object this part talks to, or ``None`` before it is opened.
    _device: Any = None

    #: Whether ``node_rank`` and ``worker_name`` mean anything to this class.
    #: A :class:`PartGroup` is composed rather than placed, and it names its
    #: parts with arbitrary keywords, so it must not swallow either of them.
    _TAKES_PLACEMENT: ClassVar[bool] = True

    #: Set by the metaclass on every instance. The class default keeps code
    #: that reads it safe for a connection built some other way.
    _recipe: Optional[_Recipe] = None

    def _open(self) -> Any:
        """Reach the hardware and return whatever speaks to it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not say how to open its hardware. "
            "Implement _open(), or override connect() for a part whose "
            "lifecycle is more than opening a device."
        )

    def _release(self, device: Any) -> None:
        """Let ``device`` go. The default has nothing to release.

        The handle is passed in rather than read back off ``self``, so an
        implementation cannot be defeated by the order ``disconnect`` does
        things in. That is not hypothetical: clearing ``_device`` first left
        every teleop device closing nothing.
        """

    @property
    def is_connected(self) -> bool:
        """Whether the part is ready for observations."""
        return self._device is not None

    def connect(self) -> None:
        """Connect to the physical part, once.

        A part that opens in place rather than returning a handle still counts
        as connected, so ``_open`` may return nothing.
        """
        if self._device is None:
            self._device = self._open() or self

    def disconnect(self) -> None:
        """Release resources owned by the connection, once.

        ``_release`` runs while the handle is still there, because that is what
        it releases. Clearing it first left every implementation that reaches
        for ``self._device`` -- which is where the vendor object is documented
        to live -- closing nothing, and the connection reported itself
        disconnected while the reader, its thread and its serial port stayed
        open.
        """
        device = self._device
        if device is None:
            return
        try:
            self._release(device)
        finally:
            self._device = None

    def reset(self) -> None:
        """Reset this connection when it has resettable state."""

    # -- Composition ------------------------------------------------------

    @property
    def kind(self) -> str:
        """This connection's narrowest registered category.

        Named rather than derived at each call site, because the proxy on the
        other side of a placement has to rebuild the same interface from a
        description that crossed a process boundary.

        Narrowest wins, so a gripper is an ``end_effector`` rather than the
        ``controllable`` it also is. That is settled by comparing the matching
        classes, not by the order categories happened to register in -- they
        register as their modules are imported, and a node only imports the
        hardware it has.
        """
        found: Optional[str] = None
        narrowest: Optional[type] = None
        for kind, part_cls in _PART_KINDS.items():
            if not isinstance(self, part_cls):
                continue
            if narrowest is None or issubclass(part_cls, narrowest):
                found, narrowest = kind, part_cls
        if found is None:
            raise TypeError(f"{type(self).__name__} is not a Connection.")
        return found

    # -- Drivers, by the name a config selects them with --------------------

    @classmethod
    def register(cls, *names: str) -> Callable[[type], type]:
        """Register a driver under the names a config spells it with.

        A category owns the registry; the drivers put themselves in it::

            @BaseCamera.register("realsense", "rs")
            class RealSenseCamera(BaseCamera): ...

        Then ``BaseCamera.backend("realsense")`` finds it. Adding a camera is
        one decorator in the file that implements it, rather than an edit to a
        table somewhere else that has to be kept in step -- which is the whole
        point, because the table is what people forget.

        Names are matched case-insensitively, since they arrive from YAML.
        Registering the same name for two drivers is refused rather than
        letting import order decide which one a config gets.
        """

        def add(driver_cls: type) -> type:
            # Stored on the class this was called on, so BaseCamera and
            # BaseGripper keep separate registries rather than one shared by
            # everything descending from Connection.
            if "_BACKENDS" not in cls.__dict__:
                cls._BACKENDS = {}
            for name in names:
                key = name.lower()
                taken = cls._BACKENDS.get(key)
                if taken is not None and taken is not driver_cls:
                    raise ValueError(
                        f"{cls.__name__} backend {name!r} is already registered "
                        f"to {taken.__name__}; {driver_cls.__name__} cannot take it."
                    )
                cls._BACKENDS[key] = driver_cls
            return driver_cls

        return add

    @classmethod
    def backends(cls) -> dict[str, type]:
        """Every driver registered for this category, by name."""
        merged: dict[str, type] = {}
        for base in reversed(cls.__mro__):
            merged.update(base.__dict__.get("_BACKENDS", {}))
        return merged

    @classmethod
    def backend(cls, name: str) -> type:
        """The driver a config name selects, or a list of what is available."""
        registered = cls.backends()
        driver_cls = registered.get(str(name).lower())
        if driver_cls is None:
            raise ValueError(
                f"Unsupported {cls.__name__} backend {name!r}. "
                f"Registered: {sorted(registered)}."
            )
        return driver_cls

    # -- What is actually plugged into this machine -------------------------

    #: The vendor library this driver imports, as ``(module, install name)``.
    #: Naming it lets :meth:`require_sdk` report a missing one without every
    #: caller knowing which library goes with which device.
    SDK: ClassVar[Optional[tuple[str, str]]] = None

    @classmethod
    def discover(cls) -> set[str]:
        """Identify the devices of this kind attached to this machine.

        Serial numbers for a camera, stable ``by-id`` names for a V4L2 device
        -- whatever a config would name one by. Enumeration belongs to the
        driver because only the driver knows the SDK call that answers, so a
        robot asks its parts rather than carrying a copy of that call per robot
        type.

        Returns an empty set when the vendor library is absent, so a node
        without the hardware enumerates to nothing rather than failing. Use
        :meth:`require_sdk` when a missing library should be an error.
        """
        return set()

    @classmethod
    def require_sdk(cls, where: str = "this machine") -> None:
        """Raise unless the vendor library this driver needs is installed.

        Args:
            where: What to call the machine in the error, so a scheduler can
                say which node rank is missing the library.
        """
        if cls.SDK is None:
            return
        module, install_name = cls.SDK
        try:
            import_module(module)
        except ModuleNotFoundError as missing:
            raise ModuleNotFoundError(
                f"{install_name} is required for {cls.__name__}, "
                f"but it is not installed on {where}."
            ) from missing

    # -- Composition -------------------------------------------------------

    @property
    def parts(self) -> dict[str, "RobotPart"]:
        """The parts this link backs, by the names it knows them under.

        One ROS link drives two arms and their grippers; one arm carries its
        gripper on the same connection. Either way the connection says what rides
        on it, and a robot picks the ones it wants and gives them public names.
        A leaf -- a camera, a gripper on its own port -- backs nothing and
        leaves this empty.

        These are not the robot's parts. What a robot is *made of* is
        :attr:`PartGroup.children`, and the two answered to one name for long
        enough to be worth separating: this is what a hardware session offers,
        ``children`` is a tree of names a policy sees.
        """
        return {}

    def part(self, name: str) -> "_ExportRef":
        """Pick one of the parts this connection backs, to compose under a name.

        Use it when one session backs several parts::

            session = Turtle2Connection(50, node_rank=0)
            left = PartGroup(
                arm=session.part("left"),
                end_effector=session.part("left_end_effector"),
            )

        Which parts a session backs is only settled once it is open, and for a
        session open on another machine only that machine can say. So this
        records the choice and :meth:`Robot.connect` acts on it. The connection
        is opened once however many parts are picked out of it.
        """
        return _ExportRef(self, name)

    def _live_part(self, name: str) -> "RobotPart":
        """Return one part of an open connection, or say what is on offer.

        The lookup :attr:`parts` supports once the connection is open, which the
        part-addressed calls below need. :meth:`part` is the composing verb and
        answers before anything has been opened.
        """
        available = self.parts
        if name not in available:
            raise KeyError(
                f"{type(self).__name__} backs no part {name!r}. "
                f"Available: {sorted(available)}."
            )
        return available[name]

    # -- Subpart-addressed surface ----------------------------------------
    # Public, so a hosted part exposes these as RPCs automatically and one
    # generic proxy can reach any subpart.

    def describe_self(self) -> dict[str, Any]:
        """Describe this part itself: its kind and its feature dictionaries.

        A leaf part -- a camera, a gripper on its own port -- exposes no
        subparts, so this is what a remote handle proxies.
        """
        described: dict[str, Any] = {"kind": self.kind}
        if isinstance(self, RobotPart):
            described["observation"] = self.observation_features
        if isinstance(self, ControllablePart):
            described["action"] = self.action_features
        return described

    def describe_parts(self) -> dict[str, dict[str, Any]]:
        """Describe every part: its kind and its feature dictionaries.

        One call carries everything a remote handle needs to build correctly
        typed proxies, so placement costs a single round trip rather than one
        per subpart per property.
        """
        described: dict[str, dict[str, Any]] = {}
        for name, part in self.parts.items():
            entry: dict[str, Any] = {
                "kind": part.kind,
                "observation": part.observation_features,
            }
            if isinstance(part, ControllablePart):
                entry["action"] = part.action_features
            described[name] = entry
        return described

    def part_observation(self, name: str) -> dict[str, Any]:
        """Read one part's observation."""
        return self._live_part(name).get_observation()

    def part_action(self, name: str, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action to one controllable part."""
        part = self._live_part(name)
        if not isinstance(part, ControllablePart):
            raise TypeError(
                f"Part {name!r} of {type(self).__name__} is not controllable."
            )
        return part.send_action(action)

    def part_reset(self, name: str) -> None:
        """Reset one part."""
        self._live_part(name).reset()

    def part_reopen(self, name: str) -> None:
        """Reopen one part, for a camera that stalled behind a proxy."""
        part = self._live_part(name)
        reopen = getattr(part, "reopen", None)
        if not callable(reopen):
            raise TypeError(
                f"Part {name!r} of {type(self).__name__} cannot be reopened."
            )
        reopen()

    def shutdown(self) -> None:
        """Disconnect during worker teardown."""
        if self.is_connected:
            self.disconnect()

    # -- Placement --------------------------------------------------------

    @property
    def node_rank(self) -> Optional[int]:
        """The node this connection runs on, or ``None`` for this process."""
        return self._recipe.node_rank if self._recipe else None

    def place(self) -> "PartHandle":
        """Open this connection where it belongs, and return a handle to it.

        With no ``node_rank`` the connection is opened in this process and the
        handle wraps the object you already have. With one, it is rebuilt from
        its recipe inside a scheduler worker on that node and the handle proxies
        to it -- which is why a constructor must only store its settings.

        Both handles expose the same API, so callers never branch on placement.
        Any connection can be placed, not only arms: a camera can run on the
        machine it is plugged into while the policy runs elsewhere.

        :meth:`Robot.connect` calls this, once per connection however many parts
        were picked out of it, and owns the handle it gets back. Call it
        yourself only outside a robot, on a bench script, where nothing else
        will release it.

        The placement layer is imported here rather than at module scope, so a
        part's own source never names the scheduler.
        """
        from ..placement import LocalPartHandle, PartWorkerHost

        recipe = self._recipe
        if recipe is None or recipe.node_rank is None:
            self.connect()
            return LocalPartHandle(self)

        return PartWorkerHost(
            recipe.part_cls,
            recipe.args,
            recipe.kwargs,
            node_rank=recipe.node_rank,
            worker_name=recipe.worker_name,
        ).spawn()

    @classmethod
    def spawn(cls, *args: Any, **kwargs: Any) -> "PartHandle":
        """Declare this connection and place it in one step.

        The eager form, for a bench script or a test that wants one device and
        no robot around it. ``ExampleArm.spawn(ip, node_rank=1)`` is
        ``ExampleArm(ip, node_rank=1).place()``, and the caller owns the handle.
        """
        return cls(*args, **kwargs).place()


@register_kind("part")
class RobotPart(Connection):
    """A connection you can read: a component of the robot itself.

    An arm, a camera, a gripper. What separates a part from a bare
    :class:`Connection` is that reading it means something, so a part declares
    what it observes and answers with exactly that. Where it runs, and when it
    opens and closes, it gets from being a connection.

    Parts are what a robot is composed of, and the only thing a
    :class:`PartGroup` accepts.
    """

    @property
    @abstractmethod
    def observation_features(self) -> dict[str, Any]:
        """Describe the values returned by :meth:`get_observation`."""

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Read the current part observation."""


@register_kind("controllable")
class ControllablePart(RobotPart):
    """A part a policy can command, not only read.

    An arm and a gripper are controllable; a camera is not. The action contract
    is stated the same way the observation contract is, so an env builds its
    action space from :attr:`action_features` rather than from a hard-coded
    width.
    """

    @property
    @abstractmethod
    def action_features(self) -> dict[str, Any]:
        """Describe the values accepted by :meth:`send_action`."""

    @abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply an action and return the action actually sent."""


class PartGroup(ControllablePart):
    """A part made of named parts: an arm assembly, a torso, a whole robot.

    Names are the composition. They become the keys of the observation and the
    action, and the path a policy sees, so a group with a lift, a head, or a
    third arm needs no new concept -- only another name.

    What may be composed is a part, and only a part::

        PartGroup(
            arm=connection.part("arm"),  # a part of a session
            end_effector=ExampleGripper("/dev/ttyUSB0"),  # a part of its own
            wrist=PartGroup(camera=...),  # a subtree of parts
        )

    A :class:`Connection` is refused: it backs parts without being one, so
    reading it would mean nothing. Pick what rides on it with
    :meth:`Connection.part` instead.

    Reads fan out across parts that sit on different connections. Parts sharing
    one connection are read and commanded in their declared order, because a
    vendor SDK behind a single link is rarely safe to call concurrently.
    """

    #: Composed, not placed. Its keywords name parts, so ``node_rank`` and
    #: ``worker_name`` are ordinary part names here and must reach ``__init__``.
    _TAKES_PLACEMENT: ClassVar[bool] = False

    def __init__(self, parts: Optional[Mapping[str, Any]] = None, **named: Any) -> None:
        combined = {**(parts or {}), **named}
        if any(not name or not isinstance(name, str) for name in combined):
            raise ValueError(
                f"{type(self).__name__} part names must be non-empty strings."
            )
        for name, value in combined.items():
            self._check_composable(name, value)
        self._children: dict[str, Any] = combined
        self._handle_of: dict[str, int] = {}
        """Which connection each part came from, so sharing is respected."""

    def _check_composable(self, name: str, value: Any) -> None:
        """Refuse anything that is not a part, and say what to do instead.

        The tree holds parts. Catching this here rather than at the first read
        is the difference between a message naming the keyword that is wrong
        and an ``AttributeError`` from inside a fan-out three calls later.

        A bare :class:`Connection` gets its own message, because it is the
        mistake worth explaining: the value is a link that backs parts, and the
        fix is to pick one rather than to find a different object.
        """
        if isinstance(value, (RobotPart, _ExportRef)):
            return
        if isinstance(value, Connection):
            # Name a real one where the connection can say: a message that
            # shows the line to write beats one that describes it.
            backed = sorted(value.parts)
            example = repr(backed[0]) if backed else '"arm"'
            offers = f" It backs {backed}." if backed else ""
            raise TypeError(
                f"{type(self).__name__} cannot compose {name}="
                f"{type(value).__name__}: it backs parts without being one of "
                f"them, so there is nothing to read.{offers} Pick one instead, "
                f"as in {name}=<{type(value).__name__.lower()}>.part({example})."
            )
        raise TypeError(
            f"{type(self).__name__} cannot compose {name}="
            f"{type(value).__name__}: a robot is made of parts. Pass a "
            "RobotPart, another PartGroup, or one part picked out of a "
            "connection with .part(...)."
        )

    @property
    def children(self) -> dict[str, Any]:
        """The parts this group is composed of, by the names it gave them."""
        return self._children

    def child(self, name: str) -> Any:
        """Return one composed part, or say which names exist."""
        if name not in self._children:
            raise KeyError(
                f"{type(self).__name__} has no part {name!r}. "
                f"Available: {sorted(self._children)}."
            )
        return self._children[name]

    @property
    def is_connected(self) -> bool:
        """Whether every part is resolved and connected."""
        values = list(self._children.values())
        if any(isinstance(value, _ExportRef) for value in values):
            return False
        return all(part.is_connected for part in values)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe each part's observation under its name."""
        return {
            name: part.observation_features for name, part in self._children.items()
        }

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe each controllable part's action under its name."""
        return {
            name: part.action_features
            for name, part in self._children.items()
            if isinstance(part, ControllablePart)
        }

    def _batches(self) -> list[list[str]]:
        """Group part names by connection: distinct ones may run together."""
        order: list[list[str]] = []
        index: dict[int, int] = {}
        for position, name in enumerate(self._children):
            key = self._handle_of.get(name, -position - 1)
            if key in index:
                order[index[key]].append(name)
            else:
                index[key] = len(order)
                order.append([name])
        return order

    def _fan_out(self, call) -> dict[str, Any]:
        """Run *call* over every part, concurrently where connections differ."""

        def run(names: list[str]) -> dict[str, Any]:
            return {name: call(self._children[name]) for name in names}

        batches = self._batches()
        results = run_parallel(
            {position: partial(run, names) for position, names in enumerate(batches)}
        )
        merged: dict[str, Any] = {}
        for position in range(len(batches)):
            merged.update(results[position])
        return merged

    def connect(self) -> None:
        """Connect every part, rolling back the ones already connected."""
        connected: list[RobotPart] = []
        try:
            for part in self._children.values():
                if not part.is_connected:
                    part.connect()
                    connected.append(part)
        except Exception:
            for part in reversed(connected):
                part.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect every connected part, in reverse order.

        Disconnecting puts the tree back to what it was composed with, so a
        second call walks one holding unresolved picks rather than live parts.
        A pick has nothing to disconnect, and asking it used to raise -- from
        inside the ``finally`` a caller runs when it is not sure the robot ever
        came up, replacing the error it was actually handling.
        """
        for part in reversed(list(self._children.values())):
            if isinstance(part, _ExportRef):
                continue
            if isinstance(part, PartGroup) or part.is_connected:
                part.disconnect()

    def reset(self) -> None:
        """Reset every part."""
        self._fan_out(lambda part: part.reset())

    def get_observation(self) -> dict[str, Any]:
        """Read every part, namespaced by name."""
        return self._fan_out(lambda part: part.get_observation())

    def send_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch each named action to the part that owns it."""
        unknown = set(action) - set(self._children)
        if unknown:
            raise KeyError(
                f"{type(self).__name__} has no parts {sorted(unknown)}; "
                f"available: {sorted(self._children)}."
            )
        not_controllable = [
            name
            for name in action
            if not isinstance(self._children[name], ControllablePart)
        ]
        if not_controllable:
            raise TypeError(f"Parts {sorted(not_controllable)} are not controllable.")

        requested = dict(action)
        batches = [[n for n in names if n in requested] for names in self._batches()]

        def run(names: list[str]) -> dict[str, Any]:
            return {
                name: self._children[name].send_action(requested[name])
                for name in names
            }

        results = run_parallel(
            {
                position: partial(run, names)
                for position, names in enumerate(batches)
                if names
            }
        )
        applied: dict[str, Any] = {}
        for value in results.values():
            applied.update(value)
        return applied

    def resolve(self, placement: Any) -> dict[str, list[Any]]:
        """Open each composed connection and put what it resolves to in the tree.

        Returns the handles used, keyed by the part name that needed them, so a
        robot can publish them and so sharing is visible to :meth:`_batches`.
        """
        used: dict[str, list[Any]] = {}
        for name, value in list(self._children.items()):
            if isinstance(value, PartGroup):
                nested = value.resolve(placement)
                flat = [h for handles in nested.values() for h in handles]
                if flat:
                    used[name] = flat
                    self._handle_of[name] = id(flat[0])
                continue
            if not isinstance(value, (Connection, _ExportRef)):
                continue
            connection = value.connection if isinstance(value, _ExportRef) else value
            handle = placement.handle_for(connection)
            used[name] = [handle]
            self._handle_of[name] = id(handle)
            self._children[name] = placement.resolve(value)
        return used

    def declarations(self) -> dict[str, Any]:
        """Snapshot the tree as composed, so opening it can be undone."""
        return {
            name: value.declarations() if isinstance(value, PartGroup) else value
            for name, value in self._children.items()
        }

    def restore(self, declared: Mapping[str, Any]) -> None:
        """Put every part back to what it was composed with."""
        for name, value in declared.items():
            current = self._children.get(name)
            if isinstance(value, dict) and isinstance(current, PartGroup):
                current.restore(value)
            else:
                self._children[name] = value
        self._handle_of.clear()
