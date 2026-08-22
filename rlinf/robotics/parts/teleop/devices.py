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

"""Teleoperation devices represented as robot parts.

Each part reports device-native readings. Bindings in
:mod:`rlinf.robotics.teleop` translate those readings into robot actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from ..base import RobotPart


class TeleopPart(RobotPart, ABC):
    """Base class for operator input devices."""

    @abstractmethod
    def _open(self) -> Any:
        """Open the device and return its vendor reader."""

    def _release(self, device: Any) -> None:
        """Release the reader by whichever name its vendor gave the method."""
        for method in ("close", "stop"):
            release = getattr(device, method, None)
            if callable(release):
                release()
                return

    @property
    def ready(self) -> bool:
        """Return whether the device has produced a usable reading."""
        return bool(getattr(self._device, "ready", self.is_connected))


class SpaceMouse(TeleopPart):
    """A 6-DoF mouse: a twist from the puck, and two buttons.

    Args:
        device_index: Which HID device to open when several are attached.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index

    def _open(self) -> Any:
        from .readers.spacemouse import SpaceMouseExpert

        return SpaceMouseExpert(device_index=self._device_index)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Return the twist and button-state feature declarations."""
        return {
            "twist": {"shape": (6,), "dtype": "float32"},
            "buttons": {"shape": (2,), "dtype": "bool"},
        }

    def get_observation(self) -> dict[str, Any]:
        """Read the puck and the buttons."""
        twist, buttons = self._device.get_action()
        return {
            "twist": np.asarray(twist, dtype=np.float32),
            "buttons": np.asarray(buttons, dtype=bool),
        }


class TeleopLeaderArm(TeleopPart):
    """Leader arm that reports Cartesian or joint-space input.

    Args:
        port: Serial port of the leader arm.
        joint_space: Report joint positions rather than a Cartesian target.
    """

    def __init__(self, port: str, joint_space: bool = False) -> None:
        self._port = port
        self._joint_space = joint_space

    def _open(self) -> Any:
        if self._joint_space:
            from .readers.gello_joint import GelloJointExpert

            return GelloJointExpert(port=self._port)
        from .readers.gello import GelloExpert

        return GelloExpert(port=self._port)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Joint positions, or a Cartesian target, plus the grip."""
        if self._joint_space:
            return {
                "joint_position": {"shape": (7,), "dtype": "float32"},
                "grip": {"shape": (1,), "dtype": "float32"},
            }
        return {
            "position": {"shape": (3,), "dtype": "float32"},
            "orientation": {"shape": (4,), "dtype": "float32"},
            "grip": {"shape": (1,), "dtype": "float32"},
        }

    def get_observation(self) -> dict[str, Any]:
        """Read the arm the operator is holding."""
        if self._joint_space:
            joints, grip = self._device.get_action()
            return {
                "joint_position": np.asarray(joints, dtype=np.float32),
                "grip": np.asarray(grip, dtype=np.float32).reshape(1),
            }
        position, orientation, grip = self._device.get_action()
        return {
            "position": np.asarray(position, dtype=np.float32),
            "orientation": np.asarray(orientation, dtype=np.float32),
            "grip": np.asarray(grip, dtype=np.float32).reshape(1),
        }


class Glove(TeleopPart):
    """A data glove reporting finger angles.

    Args:
        left_port: Serial port of the left glove.
        right_port: Serial port of the right glove.
        frequency: Polling rate in Hz.
        config_file: Optional calibration file.
    """

    def __init__(
        self,
        left_port: Optional[str] = "/dev/ttyACM0",
        right_port: Optional[str] = None,
        frequency: int = 60,
        config_file: Optional[str] = None,
    ) -> None:
        self._left_port = left_port
        self._right_port = right_port
        self._frequency = frequency
        self._config_file = config_file

    def _open(self) -> Any:
        from .readers.glove import GloveExpert

        return GloveExpert(
            left_port=self._left_port,
            right_port=self._right_port,
            frequency=self._frequency,
            config_file=self._config_file,
        )

    @property
    def observation_features(self) -> dict[str, Any]:
        """Return one angle feature per finger joint."""
        return {"angles": {"shape": (6,), "dtype": "float32"}}

    def get_observation(self) -> dict[str, Any]:
        """Read the operator's finger angles."""
        return {"angles": np.asarray(self._device.get_angles(), dtype=np.float32)}


class PicoController(TeleopPart):
    """VR controller that reports motion from its grip point.

    Args:
        pico_config: Forwarded to the vendor reader; ``hand`` selects a side.
    """

    def __init__(self, **pico_config: Any) -> None:
        self._config = pico_config

    def _open(self) -> Any:
        from .readers.pico import PicoExpert

        return PicoExpert(**self._config)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Whether the operator is driving, and how far they have moved."""
        return {
            "held": {"dtype": "bool", "shape": ()},
            "position_delta": {"dtype": "float64", "shape": (3,)},
            "rotation_delta": {"dtype": "float64", "shape": (3,)},
            "grip_close": {"dtype": "bool", "shape": ()},
            "grip_open": {"dtype": "bool", "shape": ()},
        }

    def get_observation(self) -> dict[str, Any]:
        """Read the controller."""
        return self._device.get_reading()
