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

"""Turning one device's reading into part of a robot's action."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

#: Keys a binding may read from its context. The caller supplies whichever the
#: bindings in play need; a missing key that a binding requires is an error at
#: the point of use rather than a silent zero.
#:
#: tcp_pose        the arm's measured pose, xyz + quat
#: action_scale    the env's [position, rotation, gripper] divisors
#: joint_positions measured joint positions, one row per arm
#: gripper_open    whether the gripper is currently open
CONTEXT_KEYS = ("tcp_pose", "action_scale", "joint_positions", "gripper_open")


class TeleopBinding(ABC):
    """What one device means for the robot it drives.

    A device reports what the operator did. A binding says which parts of the
    action that fills, and computes them. Keeping the two apart is what lets the
    same spacemouse drive a Cartesian arm here and something else elsewhere.
    """

    #: Action parts this binding can fill. Matched against what the robot
    #: actually has, so a binding that offers a gripper to an arm carrying a
    #: hand simply does not fill it.
    PRODUCES: tuple[str, ...] = ()

    #: Below this, motion is the device resting rather than the operator
    #: driving. Devices jitter; a person moving does not.
    MOVEMENT_EPSILON: float = 0.001

    @abstractmethod
    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Return the action parts this device fills, by name."""

    def is_driving(self, reading: Mapping[str, Any]) -> bool:
        """Whether the operator is actually moving this device."""
        return True

    def publish(self, reading: Mapping[str, Any]) -> dict[str, Any]:
        """Context this device offers the bindings listed after it.

        Devices in one rig are not independent: on the dex-hand setup the
        spacemouse's left button is what puts the glove in control. Saying so
        here keeps that coupling visible and ordered, rather than hidden in a
        class that reads both devices.
        """
        return {}

    def reset(self) -> None:
        """Forget anything held from the previous episode."""
