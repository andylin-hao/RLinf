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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Generic, Optional, TypeVar

import numpy as np

from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    NodeHardwareConfig,
)

from .part import ControllablePart, RobotPart

RobotConfigType = TypeVar("RobotConfigType", bound="RobotConfig", covariant=True)


@dataclass
class RobotConfig(HardwareConfig):
    """Base configuration for a registered physical robot."""


@dataclass
class RobotInfo(HardwareInfo, Generic[RobotConfigType]):
    """Scheduler resource information for a registered robot."""

    config: RobotConfigType


class Robot(Hardware):
    """Compose named robot parts and register robot discovery policies.

    A concrete robot integration sets ``HW_TYPE``, implements
    :meth:`Hardware.enumerate`, and uses ``@Robot.register_robot(Config)``.
    Runtime robot instances may compose arms, end effectors, cameras, or mobile
    bases through the ``parts`` mapping.
    """

    CONFIG_CLS: ClassVar[type[RobotConfig]]
    registry: ClassVar[dict[str, type["Robot"]]] = {}

    def __init__(self, parts: Optional[Mapping[str, RobotPart]] = None) -> None:
        self.parts = dict(parts or {})
        if any(not isinstance(name, str) or not name for name in self.parts):
            raise ValueError("Robot part names must be non-empty strings.")

    @classmethod
    def register_robot(
        cls, config_cls: type[RobotConfig]
    ) -> Callable[[type["Robot"]], type["Robot"]]:
        """Register a concrete robot and its configuration with the scheduler."""

        def decorator(robot_cls: type["Robot"]) -> type["Robot"]:
            if not issubclass(robot_cls, cls):
                raise TypeError(f"{robot_cls.__name__} must inherit from Robot.")
            if not robot_cls.HW_TYPE:
                raise ValueError(f"{robot_cls.__name__}.HW_TYPE must be set.")
            if not issubclass(config_cls, RobotConfig):
                raise TypeError(f"{config_cls.__name__} must inherit from RobotConfig.")
            if (
                robot_cls.HW_TYPE in cls.registry
                or robot_cls.HW_TYPE in Hardware.hw_types
                or robot_cls.HW_TYPE in NodeHardwareConfig._hardware_config_registry
            ):
                raise ValueError(
                    f"Robot type {robot_cls.HW_TYPE!r} is already registered."
                )

            robot_cls.CONFIG_CLS = config_cls
            Hardware.register()(robot_cls)
            NodeHardwareConfig.register_hardware_config(robot_cls.HW_TYPE)(config_cls)
            cls.registry[robot_cls.HW_TYPE] = robot_cls
            return robot_cls

        return decorator

    @property
    def is_connected(self) -> bool:
        """Whether every configured part is connected."""
        return all(part.is_connected for part in self.parts.values())

    @property
    def observation_features(self) -> dict[str, dict]:
        """Return observation features grouped by part name."""
        return {name: part.observation_features for name, part in self.parts.items()}

    @property
    def action_features(self) -> dict[str, dict]:
        """Return action features for controllable parts."""
        return {
            name: part.action_features
            for name, part in self.parts.items()
            if isinstance(part, ControllablePart)
        }

    def connect(self) -> None:
        """Connect every part in configuration order."""
        connected: list[RobotPart] = []
        try:
            for part in self.parts.values():
                if not part.is_connected:
                    part.connect()
                    connected.append(part)
        except Exception:
            for part in reversed(connected):
                part.disconnect()
            raise

    def get_observation(self) -> dict[str, dict[str, np.ndarray]]:
        """Read observations grouped by part name."""
        return {name: part.get_observation() for name, part in self.parts.items()}

    def send_action(
        self, action: Mapping[str, dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Dispatch actions by part name and return the applied actions."""
        unknown_parts = set(action) - set(self.parts)
        if unknown_parts:
            raise KeyError(f"Unknown robot parts: {sorted(unknown_parts)}")

        applied = {}
        for name, part_action in action.items():
            part = self.parts[name]
            if not isinstance(part, ControllablePart):
                raise TypeError(f"Robot part {name!r} is not controllable.")
            applied[name] = part.send_action(part_action)
        return applied

    def disconnect(self) -> None:
        """Disconnect every connected part in reverse order."""
        for part in reversed(list(self.parts.values())):
            if part.is_connected:
                part.disconnect()
