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
This module is the taxonomy, not the catalogue, so importing it pulls in
nothing about hardware a node does not have.

Composing is inert. Constructing a connection records its arguments and the
node it belongs to; :meth:`~rlinf.robotics.robot.Robot.connect` is the only
thing that opens anything. That is what lets a robot be composed and described
on a laptop, then run on the machine wired to the hardware.

Where a part runs changes nothing about what it is. Given a ``node_rank``, a
connection is rebuilt inside a worker on that node when it opens, and the
object already in the tree becomes a view of it -- same class, same
``isinstance``, every public call now travelling. So a robot is composed from
real parts and nothing is swapped, and a driver never writes a remote
counterpart to itself.

No module here imports ``rlinf.scheduler`` or Gymnasium. The dependency runs
one way: the scheduler is a general framework that reaches robotics by name
from a config and through a registry, never by importing it, and Gymnasium
belongs to the env layer that consumes a robot. Only the composition layer
imports either back, and :meth:`Connection.connect` reaches the placement layer
lazily so that stays true of parts.

Ray is not on that list -- it is a base dependency of the package, so the name
is allowed. Nor is this a promise that importing a part loads nothing: a part
may use ``rlinf.utils`` helpers such as ``get_logger``, which reach further.
"""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, TypeVar

from rlinf.utils.logging import get_logger

if TYPE_CHECKING:
    pass


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

    Opening and closing are :meth:`_open` and :meth:`_release`, and a driver
    implements those rather than :meth:`connect` and :meth:`disconnect`. The
    two public ones decide *where* the device lives and must run whole for a
    placed connection to work; a subclass that overrode ``connect`` and started
    a thread after it would start that thread on the wrong machine. A category
    that wraps its drivers -- a camera running a capture loop around whatever
    the driver opened -- uses :meth:`_opened` and :meth:`_closing`, which run
    beside the device wherever it ended up.
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

    #: The worker group holding this connection, while it is placed on a node.
    _group: Any = None

    #: What to become again when a placed connection is let go.
    _local_cls: Optional[type] = None

    #: The connection this one rides, when it is not its own link. Set by
    #: :meth:`part`, or by a view's constructor when it is handed its host.
    _owner: Optional["Connection"] = None

    #: What rides on this part, once :attr:`RobotPart.children` has built it.
    _beneath: "Optional[dict[str, RobotPart]]" = None

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

    def _opened(self) -> None:
        """Run once the device is open, on the machine holding it.

        For a category that adds something around its drivers rather than
        instead of them: :class:`~rlinf.robotics.parts.cameras.base.BaseCamera`
        starts the loop that reads frames out of whatever the driver opened.
        A driver itself has no use for this -- ``_open`` is already that place.
        """

    def _closing(self) -> None:
        """Undo :meth:`_opened`, while the device is still open."""

    @property
    def is_connected(self) -> bool:
        """Whether the part is ready for observations.

        A part riding a session has no device of its own, so it answers for the
        session: a gripper on an arm's bus is readable exactly when the arm is.
        """
        if self._owner is not None:
            return self._owner.is_connected
        return self._device is not None

    @property
    def node_rank(self) -> Optional[int]:
        """The node this connection runs on, or ``None`` for this process.

        Read off the recipe, so it answers before anything is opened and keeps
        answering the same afterwards -- including once the object has become a
        view of a hosted connection, which is why it never travels.
        """
        return self._recipe.node_rank if self._recipe else None

    @property
    def owner(self) -> "Connection":
        """The connection that is opened and closed on this part's behalf.

        A part holding its own link answers itself, which is the ordinary case
        for an arm or a camera. A part riding a shared session -- one arm of a
        dual-arm controller, a gripper on the arm's bus -- answers that
        session, so a robot opens it once however many of its parts it
        composed.

        Nothing declares this. :meth:`part` binds it when it hands a part out,
        which is the only way a part riding a session reaches a robot. A driver
        that returns a fresh helper from :attr:`parts` therefore cannot forget
        to say what opens it -- and forgetting used to mean the session was
        never opened at all, with the failure surfacing at the first read.

        Followed through, so a part on a session that itself rides another link
        answers the one at the bottom.
        """
        return self._owner.owner if self._owner is not None else self

    def connect(self) -> None:
        """Open this connection, once, on the machine it belongs to.

        With no ``node_rank`` that is here: :meth:`_open` reaches the hardware
        and what it returns is held until :meth:`disconnect`. A part that opens
        in place rather than returning a handle still counts as connected, so
        ``_open`` may return nothing.

        With one, the connection is rebuilt inside a worker on that node and
        *this object becomes a view of it* -- same class, same ``isinstance``,
        every public call now travelling. Nothing is swapped in the robot's
        tree, so a part placed on another machine and one sitting on this
        bench are the same thing to everything holding them.

        A part riding a session opens that session instead of anything of its
        own. That is what lets ``connect`` stay a method nothing overrides: a
        gripper on an arm's bus has no ``_open`` to write, and calling this on
        it does the one thing that makes it readable.
        """
        if self._owner is not None:
            self._owner.connect()
            return
        if self._device is not None:
            return
        if self._recipe is None or self._recipe.node_rank is None:
            device = self._open()
            # ``is None``, not falsiness: a file descriptor of 0 and an empty
            # buffer are real handles, and ``or self`` threw them away, leaving
            # _release nothing to close.
            self._device = self if device is None else device
            try:
                self._opened()
            except BaseException:
                # Whatever a category starts around the device did not start,
                # so the part is not usable; hand the device back rather than
                # leaving it open under a part that reports itself connected.
                try:
                    self._release(self._device)
                finally:
                    self._device = None
                raise
            return

        # Imported here rather than at module scope: this is the one bridge to
        # the scheduler, and a driver's source may not name it.
        from ..placement import host

        group, view = host(self)
        self._group = group
        self._local_cls = type(self)
        self._device = group
        self.__class__ = view

    def disconnect(self) -> None:
        """Let this connection go, once, wherever it was opened.

        ``_release`` runs while the handle is still there, because that is what
        it releases. Clearing it first left every implementation that reaches
        for ``self._device`` -- which is where the vendor object is documented
        to live -- closing nothing, and the connection reported itself
        disconnected while the reader, its thread and its serial port stayed
        open.

        A placed connection stops its worker instead, and becomes an ordinary
        unopened object again, so the same tree can be connected a second time.

        A part riding a session does nothing here. The session backs parts this
        one knows nothing about, so closing it would take them down too; it is
        closed once, by whoever opened it.
        """
        if self._owner is not None:
            return
        device = self._device
        if device is None:
            return
        try:
            if self._group is None:
                # Nested so the device is released even when the category's
                # teardown throws: the finalizer below clears ``_device``
                # either way, so a skipped release could never be retried.
                try:
                    self._closing()
                finally:
                    self._release(device)
            else:
                from ..placement import shutdown

                self.__class__ = self._local_cls
                try:
                    shutdown(self._group)
                finally:
                    self._group = None
                    self._local_cls = None
        finally:
            self._device = None

    def reset(self) -> None:
        """Reset this connection when it has resettable state."""

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

    def part(self, name: str) -> "RobotPart":
        """Pick one of the parts this connection backs, to compose under a name.

        Use it when one session backs several parts::

            session = Turtle2Connection(50, node_rank=0)
            left = PartGroup(
                arm=session.part("left"),
                end_effector=session.part("left_end_effector"),
            )

        What comes back is the part itself, unopened, riding this connection --
        so a robot holds the same object whether the session ends up here or on
        another node. Picking several costs one open: this is where each of
        them is told that this connection is its :attr:`owner`.

        Only a part with no way to open anything is adopted like that, which is
        what a view onto this session is. A part that implements :meth:`_open`
        holds a link of its own -- a wrist camera on its own USB bus, listed
        here because it is bolted to this arm -- and keeps it, along with the
        node it named. Adopting it too would open it on this connection's
        machine instead, or not at all.
        """
        available = self.parts
        if name not in available:
            raise KeyError(
                f"{type(self).__name__} backs no part {name!r}. "
                f"Available: {sorted(available)}."
            )
        return self._adopt(available[name])

    def _adopt(self, part: "RobotPart") -> "RobotPart":
        """Say that this connection is what opens ``part``, when it is."""
        if part is not self and part._owner is None and not part._opens_itself():
            part._owner = self
        return part

    @classmethod
    def _opens_itself(cls) -> bool:
        """Whether this class reaches hardware of its own."""
        return cls._open is not Connection._open


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

    @property
    def children(self) -> "dict[str, RobotPart]":
        """What sits beneath this in the robot's tree, by name.

        For a part, that is what rides on it: an arm's gripper is under the
        arm, because that is where it is. For a :class:`PartGroup` it is what
        the group was composed of. One question with one answer, so walking the
        tree -- to describe it, to find every camera, to read it -- never has to
        ask which kind of thing it is holding.

        A bare :class:`Connection` has no place in the tree and so does not
        answer this at all, however many parts it backs; those are composed one
        at a time with :meth:`Connection.part`.

        Built once and held. :attr:`parts` makes a fresh view on every read, so
        without this the tree handed out a different object each time it was
        walked: what :meth:`_adopt` recorded went to a throwaway, and a robot
        opened one object while the reading came from another.
        """
        if self._beneath is not None:
            return self._beneath
        beneath: "dict[str, RobotPart]" = {}
        for name, part in self.parts.items():
            if part is self:
                raise TypeError(
                    f"{type(self).__name__} lists itself in parts as {name!r}. "
                    "That mapping says what *rides* on this part, and a part "
                    "does not ride itself -- composing it already puts it in "
                    "the tree, with everything here beneath it. Drop the entry."
                )
            beneath[name] = self._adopt(part)
        self._beneath = beneath
        return beneath

    def child(self, name: str) -> "RobotPart":
        """Return one part from beneath this one, or say which names exist."""
        available = self.children
        if name not in available:
            raise KeyError(
                f"{type(self).__name__} has no part {name!r}. "
                f"Available: {sorted(available)}."
            )
        return available[name]

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Read the current part observation."""


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


def _close_all(owners: "Sequence[Connection]", doing: str) -> None:
    """Close every one of these, newest first, whatever any of them does.

    A rollback that stopped at the first connection to raise left the ones
    opened before it open, with nothing holding them -- which is the state the
    rollback exists to avoid. Every failure is reported and the last is raised,
    so one stuck device cannot hide the others or swallow itself.
    """
    failures: list[BaseException] = []
    for owner in reversed(list(owners)):
        try:
            owner.disconnect()
        except BaseException as error:  # noqa: BLE001 - reported below
            failures.append(error)
            get_logger().exception(
                "%s: %s failed to close; continuing with the rest",
                doing,
                type(owner).__name__,
            )
    if failures:
        raise failures[-1]


class PartGroup(ControllablePart):
    """A part made of named parts: an arm assembly, a torso, a whole robot.

    Names are the composition. They become the keys of the observation and the
    action, and the path a policy sees, so a group with a lift, a head, or a
    third arm needs no new concept -- only another name.

    What may be composed is a part, and only a part::

        PartGroup(
            arm=ExampleArm("10.0.0.2"),  # and whatever rides on it
            end_effector=ExampleGripper("/dev/ttyUSB0"),  # a part of its own
            left=PartGroup(camera=...),  # a subtree of parts
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

    def __init__(
        self,
        parts: "Optional[Mapping[str, RobotPart]]" = None,
        **named: "RobotPart",
    ) -> None:
        combined: "dict[str, RobotPart]" = {**(parts or {}), **named}
        if any(not name or not isinstance(name, str) for name in combined):
            raise ValueError(
                f"{type(self).__name__} part names must be non-empty strings."
            )
        for name, value in combined.items():
            self._check_composable(name, value)
        self._children: "dict[str, RobotPart]" = combined

    def _check_composable(self, name: str, value: Any) -> None:
        """Refuse anything that is not a part, and say what to do instead.

        The tree holds parts. Catching this here rather than at the first read
        is the difference between a message naming the keyword that is wrong
        and an ``AttributeError`` from inside a fan-out three calls later.

        A bare :class:`Connection` gets its own message, because it is the
        mistake worth explaining: the value is a link that backs parts, and the
        fix is to pick one rather than to find a different object.
        """
        if isinstance(value, RobotPart):
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
    def children(self) -> "dict[str, RobotPart]":
        """The parts this group was composed of, by the names it gave them.

        A group carries nothing of its own, so unlike a part its children are
        what it was handed rather than what rides on it.
        """
        return self._children

    @property
    def owner(self) -> "Connection":
        """Refused: a group rides no connection of its own.

        Every part beneath it may ride a different one, so there is no single
        answer and inventing one hides a mistake rather than reporting it.
        Answering ``self`` is what let an env ask a group for the arm session
        behind it and get an object with no ``get_state`` on it, three calls
        later and nowhere near the line at fault.
        """
        raise TypeError(
            f"{type(self).__name__} is composed of parts and rides no "
            "connection itself, so it has no owner. Ask the part you mean "
            "-- group.child(<name>).owner -- or owners() for all of them."
        )

    @property
    def is_connected(self) -> bool:
        """Whether every connection this group's parts ride on is open."""
        return all(owner.is_connected for owner in self.owners())

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe each part's observation under its name."""
        return {
            name: PartGroup._read_part(part, lambda p: p.observation_features)
            for name, part in self._children.items()
        }

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe each controllable part's action under its name."""
        return {
            name: PartGroup._read_part(
                part,
                lambda p: p.action_features,
                include=lambda p: isinstance(p, ControllablePart),
            )
            for name, part in self._children.items()
            if isinstance(part, ControllablePart)
        }

    def owners(self) -> list["Connection"]:
        """Every distinct connection this group's parts ride on, in order.

        One session backing four parts appears once, which is what makes it
        open once and close once. Identity, not equality: two arms of the same
        model built from equal arguments are still two devices.
        """
        seen: dict[int, Connection] = {}
        for part in self._children.values():
            for owner in self._owners_of(part):
                seen.setdefault(id(owner), owner)
        return list(seen.values())

    @staticmethod
    def _read_part(
        part: "RobotPart",
        call: "Callable[[RobotPart], Mapping[str, Any]]",
        include: "Optional[Callable[[RobotPart], bool]]" = None,
    ) -> dict[str, Any]:
        """Read one part, with each part riding it under its own name.

        The gripper on an arm's bus is beneath the arm in the tree, so it is
        beneath the arm in the reading. A part answers for its own device and
        knows nothing about this; assembling a carrier and its riders into one
        value is the tree's job.

        ``include`` drops riders that have nothing to contribute -- a camera on
        an arm takes no action, so it is absent from the action rather than
        present and empty.
        """
        reading = dict(call(part))
        for name, rider in part.children.items():
            if include is None or include(rider):
                reading[name] = PartGroup._read_part(rider, call, include)
        return reading

    @staticmethod
    def _command_part(part: "RobotPart", action: "Mapping[str, Any]") -> dict[str, Any]:
        """Apply one part's action, handing each rider's share to the rider.

        The keys of a carrier's action are its own fields plus the names of what
        rides on it, so this splits them by that: a name the carrier backs goes
        down, anything else is the carrier's own.
        """
        riders = part.children
        own = {name: value for name, value in action.items() if name not in riders}
        applied: dict[str, Any] = dict(part.send_action(own)) if own else {}
        for name, value in action.items():
            if name not in riders:
                continue
            rider = riders[name]
            if not isinstance(rider, ControllablePart):
                raise TypeError(f"Part {name!r} is not controllable.")
            applied[name] = PartGroup._command_part(rider, value)
        return applied

    def _batches(self) -> list[list[str]]:
        """Group part names so no two batches touch the same connection.

        :meth:`_fan_out` runs the batches concurrently and each batch in
        declaration order, so this is what keeps two threads off one vendor
        session -- few of which are safe for that. Grouping by the connection a
        part rides is not enough on its own: a nested group has several, and
        two sibling groups can share one. DOSW1 is exactly that, with both arms
        and both grippers on a single SDK session, so its two sides have to end
        up in one batch. Children are therefore merged whenever the sets of
        connections they touch overlap at all.
        """
        names = list(self._children)
        touched = [
            frozenset(id(owner) for owner in self._owners_of(self._children[name]))
            for name in names
        ]
        # Union-find over "shares at least one connection". Quadratic in the
        # number of children, which is the number of parts one group names.
        parent = list(range(len(names)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                if touched[left] & touched[right]:
                    parent[root(right)] = root(left)

        order: list[list[str]] = []
        index_of: dict[int, int] = {}
        for position, name in enumerate(names):
            group = root(position)
            if group not in index_of:
                index_of[group] = len(order)
                order.append([])
            order[index_of[group]].append(name)
        return order

    @staticmethod
    def _owners_of(part: "RobotPart") -> list["Connection"]:
        """Every connection this child needs open, itself and beneath it.

        Walking beneath is what a rider holding its own link depends on. A
        wrist camera on its own USB bus is composed with the arm it is bolted
        to, and keeps that bus rather than being adopted -- so if this stopped
        at the arm, the camera would sit in the tree and in the observation
        while nothing ever opened it.
        """
        if isinstance(part, PartGroup):
            return part.owners()
        found = [part.owner]
        for rider in part.children.values():
            found.extend(PartGroup._owners_of(rider))
        return found

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
        """Open every connection this tree rides on, or none of them.

        A session backing several parts is opened once, and if a later one
        fails, whatever already opened is closed again -- so a half-built robot
        is never handed back and the same tree can be connected again once the
        cause is fixed.
        """
        opened: list[Connection] = []
        try:
            for owner in self.owners():
                if not owner.is_connected:
                    owner.connect()
                    opened.append(owner)
        except BaseException:
            _close_all(opened, "rolling back a failed connect")
            raise

    def disconnect(self) -> None:
        """Close every connection this tree rides on, newest first.

        Idempotent, because teardown runs this from a ``finally`` block that
        does not know whether the robot ever came up.
        """
        _close_all(
            [owner for owner in self.owners() if owner.is_connected], "disconnecting"
        )

    def reset(self) -> None:
        """Reset every part."""
        self._fan_out(lambda part: part.reset())

    def get_observation(self) -> dict[str, Any]:
        """Read every part, namespaced by name, riders under their carrier."""
        return self._fan_out(
            lambda part: self._read_part(part, lambda p: p.get_observation())
        )

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
                name: self._command_part(self._children[name], requested[name])
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
