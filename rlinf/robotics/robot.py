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

from collections.abc import Mapping
from functools import partial
from typing import Any, ClassVar, Optional, TypeVar

from .parts.base import Arm, Camera, ControllablePart, RobotPart, run_parallel

RobotPartType = TypeVar("RobotPartType", bound=RobotPart)
RobotType = TypeVar("RobotType", bound="Robot")


class Robot:
    """Compose named arms, robot-level cameras, and additional parts."""

    ROBOT_TYPE: ClassVar[str] = ""

    @classmethod
    def build(cls, **kwargs: Any) -> "Robot":
        """Place this robot's parts and compose them into an instance.

        Subclasses implement this. It is what ``register`` hands to the
        registry, so :func:`~rlinf.robotics.discovery.build_robot` can compose a
        robot from its type name alone.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement build().")

    @classmethod
    def register(cls, config_cls: type, discovery_cls: type) -> type:
        """Register this robot's config, discovery, and builder in one call.

        Call it at the end of the robot's own module, once the config and
        discovery classes exist. Nothing central needs editing.
        """
        from .discovery import register_robot

        return register_robot(config_cls, cls, build=cls.build)(discovery_cls)

    def __init__(
        self,
        arms: Optional[Mapping[str, Arm]] = None,
        cameras: Optional[Mapping[str, Camera]] = None,
        parts: Optional[Mapping[str, RobotPart]] = None,
        handles: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.arms = self._validate_named_parts("arm", arms, Arm)
        self.cameras = self._validate_named_parts("camera", cameras, Camera)
        self.parts = self._validate_named_parts("part", parts, RobotPart)
        self.handles = dict(handles or {})
        """Part handles this robot owns, keyed by name. Composed parts borrow
        their connections, so the robot releases the handles once every part is
        disconnected."""

        self._placement: Any = None
        """Set by :meth:`connect` once declared parts have been placed."""

    @classmethod
    def single_arm(
        cls: type[RobotType],
        arm: Arm,
        cameras: Optional[Mapping[str, Camera]] = None,
        parts: Optional[Mapping[str, RobotPart]] = None,
        handles: Optional[Mapping[str, Any]] = None,
    ) -> RobotType:
        """Compose a single-arm robot with stable part names."""
        return cls(arms={"arm": arm}, cameras=cameras, parts=parts, handles=handles)

    @classmethod
    def dual_arm(
        cls: type[RobotType],
        left_arm: Arm,
        right_arm: Arm,
        cameras: Optional[Mapping[str, Camera]] = None,
        parts: Optional[Mapping[str, RobotPart]] = None,
        handles: Optional[Mapping[str, Any]] = None,
    ) -> RobotType:
        """Compose a dual-arm robot with stable left and right part names."""
        return cls(
            arms={"left": left_arm, "right": right_arm},
            cameras=cameras,
            parts=parts,
            handles=handles,
        )

    @staticmethod
    def _validate_named_parts(
        kind: str,
        parts: Optional[Mapping[str, RobotPartType]],
        part_type: type[RobotPartType],
    ) -> dict[str, RobotPartType]:
        from .specs import PartSpec, SubpartRef

        validated = dict(parts or {})
        invalid_names = [
            name for name in validated if not isinstance(name, str) or not name
        ]
        if invalid_names:
            raise ValueError(f"Robot {kind} names must be non-empty strings.")
        # Declarations are accepted here and become parts in connect().
        invalid_parts = [
            name
            for name, part in validated.items()
            if not isinstance(part, (part_type, PartSpec, SubpartRef))
        ]
        if invalid_parts:
            raise TypeError(f"Invalid robot {kind}s: {sorted(invalid_parts)}")
        return validated

    @property
    def is_connected(self) -> bool:
        """Whether every configured part is placed and connected."""
        from .specs import PartSpec, SubpartRef

        parts = self._top_level_parts()
        if any(isinstance(part, (PartSpec, SubpartRef)) for part in parts):
            return False
        return all(part.is_connected for part in parts)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Return canonical features grouped by robot component."""
        features: dict[str, Any] = {
            "arms": {name: arm.observation_features for name, arm in self.arms.items()}
        }
        if self.cameras:
            features["cameras"] = {
                name: camera.observation_features
                for name, camera in self.cameras.items()
            }
        if self.parts:
            features["parts"] = {
                name: part.observation_features for name, part in self.parts.items()
            }
        return features

    @property
    def action_features(self) -> dict[str, Any]:
        """Return canonical action features for controllable components."""
        features: dict[str, Any] = {
            "arms": {name: arm.action_features for name, arm in self.arms.items()}
        }
        controllable_parts = {
            name: part.action_features
            for name, part in self.parts.items()
            if isinstance(part, ControllablePart)
        }
        if controllable_parts:
            features["parts"] = controllable_parts
        return features

    def parts_of_type(self, part_type: type[RobotPartType]) -> dict[str, RobotPartType]:
        """Return all named parts implementing ``part_type``."""
        matches: dict[str, RobotPartType] = {}
        for name, part in self.named_parts.items():
            if isinstance(part, part_type):
                matches[name] = part
        return matches

    @property
    def named_parts(self) -> dict[str, RobotPart]:
        """Return every part keyed by its canonical dotted path."""
        named: dict[str, RobotPart] = {}
        for arm_name, arm in self.arms.items():
            prefix = f"arms.{arm_name}"
            named[prefix] = arm
            named[f"{prefix}.arm"] = arm.manipulator
            if arm.end_effector is not None:
                named[f"{prefix}.end_effector"] = arm.end_effector
            for camera_name, camera in arm.cameras.items():
                named[f"{prefix}.cameras.{camera_name}"] = camera
        named.update(
            {f"cameras.{name}": camera for name, camera in self.cameras.items()}
        )
        named.update({f"parts.{name}": part for name, part in self.parts.items()})
        return named

    def connect(self) -> None:
        """Place any declared parts, then connect everything.

        Parts declared with :meth:`~.parts.base.RobotPart.at` are built on their
        node here, each distinct declaration exactly once. If anything fails,
        whatever was already placed or connected is torn down, so a half-built
        robot is never left behind.
        """
        from .specs import PartSpec, Placement, SubpartRef

        placement = self._placement or Placement()
        connected: list[RobotPart] = []
        try:
            for name, arm in self.arms.items():
                used = arm.resolve(placement)
                if used:
                    # Named so callers can reach off-interface hardware methods
                    # as robot.handles[<arm name>].
                    self.handles.setdefault(name, used[0])
            for named in (self.cameras, self.parts):
                for name, value in list(named.items()):
                    if isinstance(value, (PartSpec, SubpartRef)):
                        spec = value.spec if isinstance(value, SubpartRef) else value
                        self.handles.setdefault(name, placement.resolve_handle(spec))
                        named[name] = placement.resolve(value)
            self._placement = placement

            for part in self._top_level_parts():
                if not part.is_connected:
                    part.connect()
                    connected.append(part)
        except Exception:
            for part in reversed(connected):
                part.disconnect()
            placement.release()
            self._placement = None
            raise

    def attach_camera(
        self,
        name: str,
        camera: Camera,
        *,
        arm: Optional[str] = None,
    ) -> None:
        """Attach an already constructed camera to the robot or to one arm."""
        if not name:
            raise ValueError("Camera name must be a non-empty string.")
        cameras = self.cameras if arm is None else self.arms[arm].cameras
        if name in cameras:
            raise ValueError(f"Camera {name!r} is already attached.")
        cameras[name] = camera

    def reset(self) -> None:
        """Reset all arms in parallel, then additional controllable parts."""
        run_parallel({name: arm.reset for name, arm in self.arms.items()})
        for part in self.parts.values():
            part.reset()

    def get_observation(self) -> dict[str, Any]:
        """Read a canonical namespaced robot observation.

        Arms, robot-level cameras, and extra parts sit on independent
        connections, so they are read concurrently.
        """
        jobs: dict[tuple[str, str], Any] = {}
        for name, arm in self.arms.items():
            jobs[("arms", name)] = arm.get_observation
        for name, camera in self.cameras.items():
            jobs[("cameras", name)] = camera.get_observation
        for name, part in self.parts.items():
            jobs[("parts", name)] = part.get_observation

        observation: dict[str, Any] = {"arms": {}}
        for (section, name), value in run_parallel(jobs).items():
            observation.setdefault(section, {})[name] = value
        return observation

    def send_action(
        self, action: Mapping[str, Mapping[str, dict[str, Any]]]
    ) -> dict[str, Any]:
        """Dispatch canonical namespaced actions and return applied actions."""
        unknown_sections = set(action) - {"arms", "parts"}
        if unknown_sections:
            raise KeyError(f"Unknown robot action sections: {sorted(unknown_sections)}")

        applied: dict[str, Any] = {}
        arm_actions = action.get("arms", {})
        unknown_arms = set(arm_actions) - set(self.arms)
        if unknown_arms:
            raise KeyError(f"Unknown robot arms: {sorted(unknown_arms)}")
        if arm_actions:
            # Arms are independent, so their commands are dispatched together;
            # within one arm the manipulator and end effector stay ordered.
            applied["arms"] = run_parallel(
                {
                    name: partial(self.arms[name].send_action, dict(arm_action))
                    for name, arm_action in arm_actions.items()
                }
            )

        part_actions = action.get("parts", {})
        unknown_parts = set(part_actions) - set(self.parts)
        if unknown_parts:
            raise KeyError(f"Unknown robot parts: {sorted(unknown_parts)}")
        if part_actions:
            applied_parts: dict[str, Any] = {}
            for name, part_action in part_actions.items():
                part = self.parts[name]
                if not isinstance(part, ControllablePart):
                    raise TypeError(f"Robot part {name!r} is not controllable.")
                applied_parts[name] = part.send_action(part_action)
            applied["parts"] = applied_parts
        return applied

    def disconnect(self) -> None:
        """Disconnect every part, then release the handles behind them."""
        for part in reversed(self._top_level_parts()):
            if isinstance(part, Arm) or part.is_connected:
                part.disconnect()
        placed = set()
        if self._placement is not None:
            placed = {id(handle) for handle in self._placement.handles}
        for handle in reversed(list(self.handles.values())):
            if id(handle) not in placed:
                handle.disconnect()
        if self._placement is not None:
            self._placement.release()
            self._placement = None
        self.handles = {
            name: handle
            for name, handle in self.handles.items()
            if id(handle) not in placed
        }

    def _top_level_parts(self) -> list[RobotPart]:
        return [*self.arms.values(), *self.cameras.values(), *self.parts.values()]
