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
from typing import Any, Optional


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
    """Compose an arm driver with an end effector and wrist cameras."""

    def __init__(
        self,
        driver: ControllablePart,
        end_effector: Optional[EndEffector] = None,
        cameras: Optional[Mapping[str, Camera]] = None,
    ) -> None:
        self.driver = driver
        self.end_effector = end_effector
        self.cameras = dict(cameras or {})
        if any(not name for name in self.cameras):
            raise ValueError("Arm camera names must be non-empty strings.")

    @property
    def is_connected(self) -> bool:
        """Whether the driver and every attached part are connected."""
        parts: list[RobotPart] = [self.driver, *self.cameras.values()]
        if self.end_effector is not None:
            parts.append(self.end_effector)
        return all(part.is_connected for part in parts)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the arm, end-effector, and camera observations."""
        features: dict[str, Any] = {"state": self.driver.observation_features}
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
        features: dict[str, Any] = {"arm": self.driver.action_features}
        if self.end_effector is not None:
            features["end_effector"] = self.end_effector.action_features
        return features

    def connect(self) -> None:
        """Connect the driver and attached parts with rollback on failure."""
        connected: list[RobotPart] = []
        parts: list[RobotPart] = [self.driver]
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
        """Reset the driver and attached end effector."""
        self.driver.reset()
        if self.end_effector is not None:
            self.end_effector.reset()

    def get_observation(self) -> dict[str, Any]:
        """Read a namespaced observation from the complete arm."""
        observation: dict[str, Any] = {"state": self.driver.get_observation()}
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
            applied["arm"] = self.driver.send_action(action["arm"])
        if "end_effector" in action:
            if self.end_effector is None:
                raise ValueError("Arm has no end effector.")
            applied["end_effector"] = self.end_effector.send_action(
                action["end_effector"]
            )
        return applied

    def disconnect(self) -> None:
        """Disconnect cameras, the end effector, and the driver."""
        parts: list[RobotPart] = [self.driver]
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
