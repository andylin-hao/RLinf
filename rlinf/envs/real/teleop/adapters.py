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

"""Turning each teleop device into actions for an environment.

One class per device, each implementing
:meth:`~.intervention.TeleopDevice.read`. The hold window, the fallback, and the
``info`` plumbing live in :class:`~.intervention.TeleopIntervention`, so what is
left here is genuinely device-specific: which axes the hardware reports, how a
target pose becomes a delta, and what counts as the operator actually moving.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from .intervention import TeleopDevice, TeleopSample


def sample_gripper_action(is_open: bool) -> np.ndarray:
    """A jittered open/close command.

    The jitter is deliberate: a dataset of identical +/-1.0 gripper commands
    trains a policy that only ever emits those two values.
    """
    if is_open:
        return np.random.uniform(0.9, 1.0, size=(1,))
    return np.random.uniform(-1.0, -0.9, size=(1,))


class SpaceMouseTeleop(TeleopDevice):
    """A 6-DoF mouse driving Cartesian deltas, with buttons for the gripper.

    Args:
        gripper_enabled: Whether the action space has a gripper channel.
    """

    def __init__(self, gripper_enabled: bool = True) -> None:
        from .devices.spacemouse import SpaceMouseExpert

        self.expert = SpaceMouseExpert()
        self.gripper_enabled = gripper_enabled
        self.left = False
        self.right = False
        self.gripper_action: Optional[np.ndarray] = None

    def reset(self, env: gym.Env) -> None:
        """Match the cached gripper command to where the gripper actually is."""
        self.left, self.right = False, False
        if not self.gripper_enabled:
            return
        state = env.get_wrapper_attr("_franka_state")
        self.gripper_action = sample_gripper_action(
            is_open=bool(getattr(state, "gripper_open", True))
        )

    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Read the mouse; buttons latch the gripper until pressed again."""
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)

        active = bool(
            np.linalg.norm(expert_a) > 0.001 or (self.left + self.right) > 0.5
        )

        if self.gripper_enabled:
            if self.left:
                self.gripper_action = sample_gripper_action(is_open=False)
                active = True
            elif self.right:
                self.gripper_action = sample_gripper_action(is_open=True)
                active = True
            if self.gripper_action is None:
                self.gripper_action = sample_gripper_action(is_open=True)
            expert_a = np.concatenate((expert_a, self.gripper_action.copy()), axis=0)

        return TeleopSample(
            action=expert_a,
            active=active,
            info={"left": self.left, "right": self.right},
        )


class GelloTeleop(TeleopDevice):
    """A leader arm posed by hand, reported as a Cartesian target.

    The device gives an absolute target, but the env takes deltas, so the
    reading is differenced against the robot's current TCP pose and scaled by
    the env's own action scale.

    Args:
        port: Serial port of the GELLO device.
        gripper_enabled: Whether the action space has a gripper channel.
    """

    def __init__(self, port: str, gripper_enabled: bool = True) -> None:
        from .devices.gello import GelloExpert

        self.expert = GelloExpert(port=port)
        self.gripper_enabled = gripper_enabled

    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Difference the leader pose against the follower's current pose."""
        if not self.expert.ready:
            return TeleopSample(action=None, active=False)

        target_pos, target_quat, target_gripper = self.expert.get_action()
        tcp_pose = env.get_wrapper_attr("get_tcp_pose")()
        action_scale = env.get_wrapper_attr("get_action_scale")()

        delta_pos = (target_pos - tcp_pose[:3]) / action_scale[0]
        r_delta = (
            R.from_quat(target_quat.copy()) * R.from_quat(tcp_pose[3:].copy()).inv()
        )
        delta_euler = r_delta.as_euler("xyz") / action_scale[1]

        expert_a = np.clip(np.concatenate((delta_pos, delta_euler), axis=0), -1.0, 1.0)

        gripper_active = False
        if self.gripper_enabled:
            grip = np.clip(-(2 * (target_gripper / action_scale[2]) - 1.0), -1.0, 1.0)
            gripper_active = bool(np.abs(grip).item() > 0.5)
            expert_a = np.concatenate((expert_a, grip), axis=0)

        active = bool(np.linalg.norm(expert_a[:6]) > 0.001 or gripper_active)
        return TeleopSample(action=expert_a, active=active)


class DexHandTeleop(TeleopDevice):
    """A SpaceMouse for the arm and a glove for the hand.

    The glove is relative: pressing the left button re-baselines it against the
    current hand pose, so the operator can reposition their hand without the
    robot's hand following. That posed hand state persists after they let go,
    which is why :meth:`fallback` overrides the hand channels.

    Args:
        left_port: Serial port of the left glove.
        right_port: Serial port of the right glove.
        glove_frequency: Glove polling rate in Hz.
        glove_config_file: Optional glove calibration file.
    """

    def __init__(
        self,
        left_port: Optional[str] = "/dev/ttyACM0",
        right_port: Optional[str] = None,
        glove_frequency: int = 60,
        glove_config_file: Optional[str] = None,
    ) -> None:
        from .devices.glove import GloveExpert
        from .devices.spacemouse import SpaceMouseExpert

        self._spacemouse = SpaceMouseExpert()
        self._glove = GloveExpert(
            left_port=left_port,
            right_port=right_port,
            frequency=glove_frequency,
            config_file=glove_config_file,
        )
        self.left = False
        self.right = False
        self._prev_left = False
        self._glove_baseline: Optional[np.ndarray] = None
        self._hand_base = np.zeros(6, dtype=np.float64)
        self._hand_current = np.zeros(6, dtype=np.float64)

    def reset(self, env: gym.Env) -> None:
        """Start each episode from the task's configured hand pose."""
        config = getattr(env.unwrapped, "config", None)
        hand_reset = getattr(config, "hand_reset_state", None)
        self._hand_current = (
            np.array(hand_reset, dtype=np.float64)
            if hand_reset is not None
            else np.zeros(6, dtype=np.float64)
        )
        self._hand_base = self._hand_current.copy()
        self._glove_baseline = None
        self._prev_left = False

    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Combine mouse deltas for the arm with glove angles for the hand."""
        arm_expert, buttons = self._spacemouse.get_action()
        self.left, self.right = bool(buttons[1]), bool(buttons[0])

        active = bool(np.linalg.norm(arm_expert) > 0.001 or self.left or self.right)

        glove_raw = self._glove.get_angles()
        if self.left:
            if not self._prev_left:
                self._glove_baseline = glove_raw.copy()
                self._hand_base = self._hand_current.copy()
            delta = glove_raw - self._glove_baseline
            self._hand_current = np.clip(self._hand_base + delta, 0.0, 1.0)
            active = True
        self._prev_left = self.left

        return TeleopSample(
            action=np.concatenate([arm_expert, self._hand_current.copy()]),
            active=active,
            info={"left": self.left, "right": self.right},
        )

    def fallback(self, env: gym.Env, policy_action: np.ndarray) -> np.ndarray:
        """Keep the hand where the operator posed it; the policy drives the arm."""
        held = np.array(policy_action, dtype=np.float64)
        held[6:] = self._hand_current
        return held

    def close(self) -> None:
        """Close the glove; the SpaceMouse has no handle to release."""
        self._glove.close()


class GelloJointTeleop(TeleopDevice):
    """A pair of leader arms driving both arms in joint space.

    Args:
        left: The left leader arm reader.
        right: The right leader arm reader.
        action_scale: Divisor turning a joint delta into a normalized action.
        use_delta: Whether the env takes joint deltas or absolute targets.
        gripper_enabled: Whether each arm's action carries a gripper channel.
    """

    def __init__(
        self,
        left: Any,
        right: Any,
        action_scale: float = 1.0,
        use_delta: bool = True,
        gripper_enabled: bool = True,
    ) -> None:
        self.left_expert = left
        self.right_expert = right
        self.action_scale = action_scale
        self.use_delta = use_delta
        self.gripper_enabled = gripper_enabled

    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Difference both leader arms against the followers' joint positions."""
        if not (self.left_expert.ready and self.right_expert.ready):
            return TeleopSample(action=None, active=False)

        left_q, left_g = self.left_expert.get_action()
        right_q, right_g = self.right_expert.get_action()
        current = env.get_wrapper_attr("get_joint_positions")()  # (2, 7)

        per_arm = []
        for target_q, current_q in zip((left_q, right_q), (current[0], current[1])):
            if self.use_delta:
                per_arm.append(
                    np.clip((target_q - current_q) / self.action_scale, -1.0, 1.0)
                )
            else:
                per_arm.append(target_q.copy())

        gripper_active = False
        if self.gripper_enabled:
            grippers = []
            for grip in (left_g, right_g):
                g = np.clip(-(2 * grip - 1.0), -1.0, 1.0)
                grippers.append(g)
                gripper_active = gripper_active or bool(np.abs(g).item() > 0.5)
            expert_a = np.concatenate(
                [per_arm[0], grippers[0], per_arm[1], grippers[1]], axis=0
            )
        else:
            expert_a = np.concatenate(per_arm, axis=0)

        movement = float(
            np.linalg.norm(
                np.concatenate([left_q, right_q])
                - np.concatenate([current[0], current[1]])
            )
        )
        return TeleopSample(
            action=expert_a,
            active=movement > 0.01 or gripper_active,
        )

    def close(self) -> None:
        """Release both leader arms."""
        self.left_expert.close()
        self.right_expert.close()
