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

from .kinds import ActionKind

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

    #: Action parts this binding can fill, and what each command means.
    #: Matched against what the robot actually has, so a binding that offers a
    #: gripper to an arm carrying a hand simply does not fill it -- and one that
    #: offers a twist to a joint-space arm is refused rather than obeyed.
    #:
    #: A binding whose meaning depends on how it was built sets this on the
    #: instance instead, as a leader arm does for deltas versus targets.
    PRODUCES: Mapping[str, "ActionKind"] = {}

    #: Below this, motion is the device resting rather than the operator
    #: driving. Devices jitter; a person moving does not.
    MOVEMENT_EPSILON: float = 0.001

    #: How long the operator keeps control after their last active reading.
    #: ``None`` accepts the shared default; ``0`` suits a device held down to
    #: take over, where the button already says exactly when they are driving.
    HOLD_WINDOW: float | None = None

    #: Whether the parts this binding fills are clipped into the env's action
    #: space. Absolute commands can leave it; normalised deltas cannot.
    CLIPS_TO_ACTION_SPACE: bool = False

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

    def info(self) -> dict[str, Any]:
        """Device state worth recording alongside the step it produced."""
        return {}

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """The parts that keep this device's share of the robot where it is.

        Asked for when a chunk of policy actions is skipped and something still
        has to be commanded. A binding that fills a part with a delta has
        nothing to say here: zero motion is already the policy's own action.
        """
        return {}

    def on_action_chunk_begin(self) -> None:
        """Let go of anything held only until the next chunk of actions."""

    def reset(self) -> None:
        """Forget anything held from the previous episode."""
