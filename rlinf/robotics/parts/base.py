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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from ..placement import PartHandle
    from ..specs import PartSpec

KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


def run_parallel(jobs: Mapping[KeyType, Callable[[], ValueType]]) -> dict[KeyType, ValueType]:
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


class RobotPart(ABC):
    """Observable part of a physical robot, such as an arm or camera."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the part is ready for observations."""

    @property
    @abstractmethod
    def observation_features(self) -> dict[str, Any]:
        """Describe the values returned by :meth:`get_observation`."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the physical part."""

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Read the current part observation."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources owned by the part."""

    def reset(self) -> None:
        """Reset the part when it has resettable state."""

    # -- Composition ------------------------------------------------------

    def subparts(self) -> dict[str, "RobotPart"]:
        """Return the named parts this one exposes. Leaves expose none.

        One piece of hardware often presents several components over a single
        connection: a dual-arm controller exposes two arms, two end effectors,
        and wrist cameras. Those are subparts of it. A part that also stands on
        its own includes itself, conventionally under ``"arm"``.
        """
        return {}

    def subpart(self, name: str) -> "RobotPart":
        """Return one named subpart, or raise a clear configuration error."""
        subparts = self.subparts()
        if name not in subparts:
            raise KeyError(
                f"{type(self).__name__} has no subpart {name!r}. "
                f"Available: {sorted(subparts)}."
            )
        return subparts[name]

    # -- Subpart-addressed surface ----------------------------------------
    # Public, so a hosted part exposes these as RPCs automatically and one
    # generic proxy can reach any subpart.

    def describe_subparts(self) -> dict[str, dict[str, Any]]:
        """Describe every subpart: its kind and its feature dictionaries.

        One call carries everything a remote handle needs to build correctly
        typed proxies, so placement costs a single round trip rather than one
        per subpart per property.
        """
        described: dict[str, dict[str, Any]] = {}
        for name, part in self.subparts().items():
            entry: dict[str, Any] = {
                "kind": part_kind(part),
                "observation": part.observation_features,
            }
            if isinstance(part, ControllablePart):
                entry["action"] = part.action_features
            described[name] = entry
        return described

    def subpart_observation(self, name: str) -> dict[str, Any]:
        """Read one subpart's observation."""
        return self.subpart(name).get_observation()

    def subpart_action(self, name: str, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action to one controllable subpart."""
        part = self.subpart(name)
        if not isinstance(part, ControllablePart):
            raise TypeError(f"Subpart {name!r} of {type(self).__name__} is not controllable.")
        return part.send_action(action)

    def subpart_reset(self, name: str) -> None:
        """Reset one subpart."""
        self.subpart(name).reset()

    def shutdown(self) -> None:
        """Disconnect during worker teardown."""
        if self.is_connected:
            self.disconnect()

    # -- Placement --------------------------------------------------------

    @classmethod
    def at(
        cls,
        *args: Any,
        node_rank: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "PartSpec":
        """Declare this part and the node it runs on, without building it.

        Compose the result exactly where you would compose a part;
        :meth:`Robot.connect` places it. Prefer this over :meth:`spawn`: a part
        whose SDK lives only on the target machine cannot be built here first,
        and declaring lets the robot own teardown and roll back cleanly.
        """
        from ..specs import PartSpec

        return PartSpec(cls, args, kwargs, node_rank=node_rank, name=name)

    @classmethod
    def spawn(
        cls,
        *args: Any,
        node_rank: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "PartHandle":
        """Construct and connect this part, here or on a chosen node.

        This is the eager, low-level form and it hands you the handle to manage.
        Prefer :meth:`at` inside a robot, which defers placement to
        :meth:`Robot.connect` and lets the robot own the handle.

        With ``node_rank`` unset the part is built in this process. Otherwise it
        is hosted in a scheduler worker on that node and the returned handle
        proxies to it. Both handles expose the same API, so callers never branch
        on placement.

        Any part can be placed, not only arms: a camera can run on the machine
        it is plugged into while the policy runs elsewhere.

        The scheduler is imported here rather than at module scope, so importing
        a part never pulls Ray into the process.
        """
        from ..placement import LocalPartHandle, spawn_part_worker

        if node_rank is None:
            part = cls(*args, **kwargs)
            part.connect()
            return LocalPartHandle(part)

        return spawn_part_worker(cls, args, kwargs, node_rank=node_rank, name=name)


class ControllablePart(RobotPart):
    """Robot part that accepts commands in addition to observations."""

    @property
    @abstractmethod
    def action_features(self) -> dict[str, Any]:
        """Describe the values accepted by :meth:`send_action`."""

    @abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply an action and return the action actually sent."""


class EndEffector(ControllablePart):
    """Controllable tool attached to an arm, such as a gripper or hand."""


class Camera(RobotPart):
    """Observation-only camera part."""


class Arm(ControllablePart):
    """Compose a manipulator with an end effector and wrist cameras.

    The three slots line up with the namespaces the policy sees: the
    manipulator's own action appears under ``"arm"``, its observation under
    ``"state"``, then ``"end_effector"`` and ``"cameras"``.
    """

    def __init__(
        self,
        manipulator: ControllablePart,
        end_effector: Optional[EndEffector] = None,
        cameras: Optional[Mapping[str, Camera]] = None,
    ) -> None:
        self.manipulator = manipulator
        self.end_effector = end_effector
        self.cameras = dict(cameras or {})
        if any(not name for name in self.cameras):
            raise ValueError("Arm camera names must be non-empty strings.")

    def resolve(self, placement: Any) -> list[Any]:
        """Replace any declared slots with the parts they resolve to.

        Called by :meth:`Robot.connect` before anything is connected. An arm
        built from live parts is unaffected.

        Returns:
            The handles this arm's declarations were placed on, in order, so the
            robot can publish them under the arm's name.
        """
        from ..specs import PartSpec, SubpartRef

        used: list[Any] = []

        def resolve(value: Any) -> Any:
            if isinstance(value, (PartSpec, SubpartRef)):
                spec = value.spec if isinstance(value, SubpartRef) else value
                used.append(placement.resolve_handle(spec))
                return placement.resolve(value)
            return value

        manipulator = self.manipulator
        self.manipulator = resolve(manipulator)
        # A declaration that also exposes an end effector fills an empty slot,
        # so the common one-connection arm needs no second declaration.
        if self.end_effector is None and isinstance(manipulator, PartSpec):
            subparts = placement.resolve_handle(manipulator).subparts
            if "end_effector" in subparts:
                self.end_effector = subparts["end_effector"]
        self.end_effector = resolve(self.end_effector)
        self.cameras = {
            name: resolve(camera) for name, camera in self.cameras.items()
        }
        return used

    @property
    def is_connected(self) -> bool:
        """Whether the manipulator and every attached part are connected.

        An arm with slots still declared but not yet placed is not connected.
        """
        from ..specs import PartSpec, SubpartRef

        parts: list[Any] = [self.manipulator, *self.cameras.values()]
        if self.end_effector is not None:
            parts.append(self.end_effector)
        if any(isinstance(part, (PartSpec, SubpartRef)) for part in parts):
            return False
        return all(part.is_connected for part in parts)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the arm, end-effector, and camera observations."""
        features: dict[str, Any] = {"state": self.manipulator.observation_features}
        if self.end_effector is not None:
            features["end_effector"] = self.end_effector.observation_features
        if self.cameras:
            features["cameras"] = {
                name: camera.observation_features
                for name, camera in self.cameras.items()
            }
        return features

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe arm and optional end-effector actions."""
        features: dict[str, Any] = {"arm": self.manipulator.action_features}
        if self.end_effector is not None:
            features["end_effector"] = self.end_effector.action_features
        return features

    def connect(self) -> None:
        """Connect the manipulator and attached parts with rollback on failure."""
        connected: list[RobotPart] = []
        parts: list[RobotPart] = [self.manipulator]
        if self.end_effector is not None:
            parts.append(self.end_effector)
        parts.extend(self.cameras.values())
        try:
            for part in parts:
                if not part.is_connected:
                    part.connect()
                    connected.append(part)
        except Exception:
            for part in reversed(connected):
                part.disconnect()
            raise

    def reset(self) -> None:
        """Reset the manipulator and attached end effector."""
        self.manipulator.reset()
        if self.end_effector is not None:
            self.end_effector.reset()

    def get_observation(self) -> dict[str, Any]:
        """Read a namespaced observation from the complete arm."""
        observation: dict[str, Any] = {"state": self.manipulator.get_observation()}
        if self.end_effector is not None:
            observation["end_effector"] = self.end_effector.get_observation()
        if self.cameras:
            observation["cameras"] = {
                name: camera.get_observation() for name, camera in self.cameras.items()
            }
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Dispatch arm and end-effector actions independently."""
        unknown = set(action) - {"arm", "end_effector"}
        if unknown:
            raise KeyError(f"Unknown arm action fields: {sorted(unknown)}")
        applied: dict[str, Any] = {}
        if "arm" in action:
            applied["arm"] = self.manipulator.send_action(action["arm"])
        if "end_effector" in action:
            if self.end_effector is None:
                raise ValueError("Arm has no end effector.")
            applied["end_effector"] = self.end_effector.send_action(
                action["end_effector"]
            )
        return applied

    def disconnect(self) -> None:
        """Disconnect cameras, the end effector, and the manipulator."""
        parts: list[RobotPart] = [self.manipulator]
        if self.end_effector is not None:
            parts.append(self.end_effector)
        parts.extend(self.cameras.values())
        for part in reversed(parts):
            if part.is_connected:
                part.disconnect()


class MobileBase(ControllablePart):
    """Controllable wheeled or tracked base."""


class LeggedBase(ControllablePart):
    """Controllable legged base."""


#: Ordered most specific first, so a part matches its narrowest kind.
_PART_KINDS: tuple[tuple[str, type], ...] = (
    ("end_effector", EndEffector),
    ("camera", Camera),
    ("controllable", ControllablePart),
    ("part", RobotPart),
)


def part_kind(part: RobotPart) -> str:
    """Classify a part so a remote proxy can mirror its interface."""
    for kind, part_type in _PART_KINDS:
        if isinstance(part, part_type):
            return kind
    raise TypeError(f"{type(part).__name__} is not a RobotPart.")
