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

"""What each device means for the robot it drives."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .binding import TeleopBinding


def jittered_grip(is_open: bool) -> np.ndarray:
    """An open or close command, jittered.

    The jitter is deliberate: a dataset of identical +/-1.0 gripper commands
    trains a policy that only ever emits those two values.
    """
    if is_open:
        return np.random.uniform(0.9, 1.0, size=(1,))
    return np.random.uniform(-1.0, -0.9, size=(1,))


class SpaceMouseBinding(TeleopBinding):
    """The puck drives the arm; the buttons latch the gripper.

    The gripper is a latch rather than a level: the left button closes it and
    the right opens it, and it stays there until the other button is pressed.
    """

    PRODUCES = ("arm", "end_effector")

    def __init__(self) -> None:
        self._grip: Optional[np.ndarray] = None
        self.left = False
        self.right = False

    def reset(self) -> None:
        """Start each episode with the buttons released."""
        self.left = False
        self.right = False

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Map the twist onto the arm, and the buttons onto the gripper."""
        buttons = reading["buttons"]
        self.left, self.right = bool(buttons[0]), bool(buttons[1])

        parts: dict[str, np.ndarray] = {"arm": np.asarray(reading["twist"])}

        if self.left:
            self._grip = jittered_grip(is_open=False)
        elif self.right:
            self._grip = jittered_grip(is_open=True)
        elif self._grip is None:
            self._grip = jittered_grip(is_open=bool(context.get("gripper_open", True)))
        parts["end_effector"] = self._grip.copy()
        return parts

    def is_driving(self, reading: Mapping[str, Any]) -> bool:
        """Moving the puck, or pressing either button."""
        moved = float(np.linalg.norm(reading["twist"])) > self.MOVEMENT_EPSILON
        return moved or bool(reading["buttons"][0]) or bool(reading["buttons"][1])


class LeaderArmBinding(TeleopBinding):
    """A leader arm posed by hand, as a Cartesian delta.

    The device reports where the operator put it. The env takes a delta, so the
    reading is differenced against the follower's measured pose and divided by
    the env's own scale.
    """

    PRODUCES = ("arm", "end_effector")

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Difference the leader's pose against the follower's."""
        tcp_pose = np.asarray(context["tcp_pose"])
        scale = np.asarray(context["action_scale"])

        delta_position = (np.asarray(reading["position"]) - tcp_pose[:3]) / scale[0]
        rotation = (
            R.from_quat(np.asarray(reading["orientation"]).copy())
            * R.from_quat(tcp_pose[3:].copy()).inv()
        )
        delta_rotation = rotation.as_euler("xyz") / scale[1]

        parts = {
            "arm": np.clip(
                np.concatenate((delta_position, delta_rotation), axis=0), -1.0, 1.0
            )
        }
        grip = np.asarray(reading["grip"]) / scale[2]
        parts["end_effector"] = np.clip(-(2 * grip - 1.0), -1.0, 1.0)
        return parts

    def is_driving(self, reading: Mapping[str, Any]) -> bool:
        """The leader arm is always driving once it is streaming."""
        return True


class LeaderJointBinding(TeleopBinding):
    """A leader arm reported in joint space, for one side of the robot.

    Args:
        side: Index of the arm this leader drives, for reading the follower's
            joint positions.
        use_delta: Whether the env takes joint deltas or absolute targets.
        action_scale: Divisor turning a joint delta into a normalized action.
    """

    PRODUCES = ("arm", "end_effector")

    def __init__(
        self, side: int = 0, use_delta: bool = False, action_scale: float = 0.1
    ) -> None:
        self.side = side
        self.use_delta = use_delta
        self.action_scale = action_scale

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Difference the leader's joints against the follower's."""
        target = np.asarray(reading["joint_position"])
        if self.use_delta:
            current = np.asarray(context["joint_positions"])[self.side]
            arm = np.clip((target - current) / self.action_scale, -1.0, 1.0)
        else:
            arm = target.copy()

        grip = np.asarray(reading["grip"])
        return {
            "arm": arm,
            "end_effector": np.clip(-(2 * grip - 1.0), -1.0, 1.0),
        }

    def is_driving(self, reading: Mapping[str, Any]) -> bool:
        """Streaming leader arms are always driving."""
        return True


class GloveBinding(TeleopBinding):
    """A data glove driving a dexterous hand, relative to a baseline.

    Pressing the arm device's button re-baselines the glove against the hand's
    current pose, so the operator can reposition their hand without the robot's
    hand following.

    The hold is explicit: once the operator lets go, the hand stays where they
    posed it rather than returning to whatever the policy commands. That is the
    behavior the dex-hand setup has always had, and it is stated here rather
    than left to fall out of the wrapper.
    """

    PRODUCES = ("hand",)

    def __init__(self, hold: bool = True) -> None:
        self.hold = hold
        self._baseline: Optional[np.ndarray] = None
        self._commanded = np.zeros(6, dtype=np.float64)
        self._base = np.zeros(6, dtype=np.float64)
        self._rebaseline = False

    def reset(self, start: Optional[np.ndarray] = None) -> None:
        """Start the episode from the task's configured hand pose."""
        self._commanded = (
            np.zeros(6, dtype=np.float64)
            if start is None
            else np.asarray(start, dtype=np.float64)
        )
        self._base = self._commanded.copy()
        self._baseline = None

    def rebaseline(self) -> None:
        """Re-zero the glove against the hand's current pose."""
        self._rebaseline = True

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Track the operator's fingers, or hold where they left them."""
        angles = np.asarray(reading["angles"], dtype=np.float64)
        if context.get("hand_driving", True):
            if self._rebaseline or self._baseline is None:
                self._baseline = angles.copy()
                self._base = self._commanded.copy()
                self._rebaseline = False
            self._commanded = np.clip(self._base + (angles - self._baseline), 0.0, 1.0)
        return {"hand": self._commanded.copy()}

    def is_driving(self, reading: Mapping[str, Any]) -> bool:
        """The glove follows the arm device's button rather than deciding."""
        return False
