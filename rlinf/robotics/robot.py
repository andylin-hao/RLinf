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
from typing import Any, ClassVar, Optional, TypeVar

from .part import Arm, Camera, ControllablePart, RobotPart

RobotPartType = TypeVar("RobotPartType", bound=RobotPart)
RobotType = TypeVar("RobotType", bound="Robot")


class Robot:
    """Compose named arms, robot-level cameras, and additional parts."""

    ROBOT_TYPE: ClassVar[str] = ""

    def __init__(
        self,
        arms: Optional[Mapping[str, Arm]] = None,
        cameras: Optional[Mapping[str, Camera]] = None,
        parts: Optional[Mapping[str, RobotPart]] = None,
    ) -> None:
        self.arms = self._validate_named_parts("arm", arms, Arm)
        self.cameras = self._validate_named_parts("camera", cameras, Camera)
        self.parts = self._validate_named_parts("part", parts, RobotPart)

    @classmethod
    def single_arm(
        cls: type[RobotType],
        arm: Arm,
        cameras: Optional[Mapping[str, Camera]] = None,
        parts: Optional[Mapping[str, RobotPart]] = None,
    ) -> RobotType:
        """Compose a single-arm robot with stable part names."""
        return cls(arms={"arm": arm}, cameras=cameras, parts=parts)

    @classmethod
    def dual_arm(
        cls: type[RobotType],
        left_arm: Arm,
        right_arm: Arm,
        cameras: Optional[Mapping[str, Camera]] = None,
        parts: Optional[Mapping[str, RobotPart]] = None,
    ) -> RobotType:
        """Compose a dual-arm robot with stable left and right part names."""
        return cls(
            arms={"left": left_arm, "right": right_arm},
            cameras=cameras,
            parts=parts,
        )

    @staticmethod
    def _validate_named_parts(
        kind: str,
        parts: Optional[Mapping[str, RobotPartType]],
        part_type: type[RobotPartType],
    ) -> dict[str, RobotPartType]:
        validated = dict(parts or {})
        invalid_names = [
            name for name in validated if not isinstance(name, str) or not name
        ]
        if invalid_names:
            raise ValueError(f"Robot {kind} names must be non-empty strings.")
        invalid_parts = [
            name for name, part in validated.items() if not isinstance(part, part_type)
        ]
        if invalid_parts:
            raise TypeError(f"Invalid robot {kind}s: {sorted(invalid_parts)}")
        return validated

    @property
    def is_connected(self) -> bool:
        """Whether every configured part is connected."""
        return all(part.is_connected for part in self._top_level_parts())

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
            named[f"{prefix}.driver"] = arm.driver
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
        """Connect every part in configuration order."""
        connected: list[RobotPart] = []
        try:
            for part in self._top_level_parts():
                if not part.is_connected:
                    part.connect()
                    connected.append(part)
        except Exception:
            for part in reversed(connected):
                part.disconnect()
            raise

    def reset(self) -> None:
        """Reset all arms and additional controllable parts."""
        for arm in self.arms.values():
            arm.reset()
        for part in self.parts.values():
            part.reset()

    def get_observation(self) -> dict[str, Any]:
        """Read a canonical namespaced robot observation."""
        observation: dict[str, Any] = {
            "arms": {name: arm.get_observation() for name, arm in self.arms.items()}
        }
        if self.cameras:
            observation["cameras"] = {
                name: camera.get_observation() for name, camera in self.cameras.items()
            }
        if self.parts:
            observation["parts"] = {
                name: part.get_observation() for name, part in self.parts.items()
            }
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
            applied["arms"] = {
                name: self.arms[name].send_action(dict(arm_action))
                for name, arm_action in arm_actions.items()
            }

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
        """Disconnect every connected part in reverse order."""
        for part in reversed(self._top_level_parts()):
            if isinstance(part, Arm) or part.is_connected:
                part.disconnect()

    def _top_level_parts(self) -> list[RobotPart]:
        return [*self.arms.values(), *self.cameras.values(), *self.parts.values()]
