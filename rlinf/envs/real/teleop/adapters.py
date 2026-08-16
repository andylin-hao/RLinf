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

"""Devices that are more than a reading mapped onto an action.

Composition covers the rest: a device reports, a binding maps, and a group
merges, in :mod:`rlinf.robotics.teleop`. What is left here needs something
composition does not describe -- in this case a thread that pushes joint targets
to the controllers at roughly 1 kHz, outside ``env.step`` entirely.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from .streaming import TeleopStreamer


class DualGelloJointStream(TeleopStreamer):
    """The 1 kHz loop that pushes a pair of leader arms to the controllers.

    The action itself comes from composition, one leader per side. This exists
    because a leader arm tracked at the policy's step rate feels laggy to the
    operator, so its targets go straight to each controller; ``env.step`` then
    reads state and grippers but stops forwarding motion, and only one writer
    touches the motion queue.

    Args:
        left_port: Serial port of the left leader arm.
        right_port: Serial port of the right leader arm.
        gripper_enabled: Whether each arm's action carries a gripper channel.
        use_delta: Whether the env takes joint deltas or absolute targets.
        action_scale: Divisor turning a joint delta into a normalized action.
        direct_stream: Push targets from the device's own thread.
        stream_period: Seconds between pushes when streaming.
    """

    def __init__(
        self,
        left_port: str,
        right_port: str,
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
        direct_stream: bool = False,
        stream_period: float = 0.001,
    ) -> None:
        from rlinf.robotics.parts.teleop.readers.gello_joint import GelloJointExpert

        super().__init__(period=stream_period, enabled=direct_stream)
        self.left_expert = GelloJointExpert(port=left_port)
        self.right_expert = GelloJointExpert(port=right_port)
        self.gripper_enabled = gripper_enabled
        self.use_delta = use_delta
        self.action_scale = action_scale
        # Gripper commands are edge-triggered: an open/close RPC takes ~100 ms,
        # and repeating it every tick would starve the serial channel.
        self._last_open: list[Optional[bool]] = [None, None]

    def before_reset(self, env: gym.Env, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Skip the env's slew home; aligning to the leader arms replaces it.

        Homing first and then aligning would move the robot twice, and the
        second move starts from wherever the first one left it.
        """
        kwargs = super().before_reset(env, kwargs)
        options = dict(kwargs.get("options") or {})
        options.setdefault("skip_reset_to_home", True)
        kwargs["options"] = options
        return kwargs

    def _controllers(self, env: gym.Env):
        inner = env.unwrapped
        return getattr(inner, "_left_ctrl", None), getattr(inner, "_right_ctrl", None)

    def ready_to_stream(self, env: gym.Env) -> bool:
        """Both controllers must exist before a tick can command them."""
        return self._controllers(env) != (None, None)

    def align(self, env: gym.Env) -> bool:
        """Move each follower onto its leader's current joint pose."""
        if not (self.left_expert.ready and self.right_expert.ready):
            return False
        left_ctrl, right_ctrl = self._controllers(env)
        if left_ctrl is None or right_ctrl is None:
            return False
        left_q, _ = self.left_expert.get_action()
        right_q, _ = self.right_expert.get_action()
        left_ctrl.reset_joint(np.asarray(left_q, dtype=np.float64).tolist()).wait()
        right_ctrl.reset_joint(np.asarray(right_q, dtype=np.float64).tolist()).wait()
        inner = env.unwrapped
        inner._left_state = left_ctrl.get_state().wait()[0]
        inner._right_state = right_ctrl.get_state().wait()[0]
        return True

    def stream_once(self, env: gym.Env) -> None:
        """Push one set of joint targets, and any gripper edge, to both arms."""
        if not (self.left_expert.ready and self.right_expert.ready):
            time.sleep(self._period)
            return
        left_ctrl, right_ctrl = self._controllers(env)
        if left_ctrl is None or right_ctrl is None:
            return

        left_q, left_g = self.left_expert.get_action()
        right_q, right_g = self.right_expert.get_action()
        left_ctrl.move_joints(left_q.astype(np.float32)).wait()
        right_ctrl.move_joints(right_q.astype(np.float32)).wait()

        if not self.gripper_enabled:
            return
        for index, (ctrl, grip) in enumerate(
            zip((left_ctrl, right_ctrl), (left_g, right_g))
        ):
            is_open = grip.item() < 0.5
            if self._last_open[index] is None:
                self._last_open[index] = is_open
            elif is_open != self._last_open[index]:
                ctrl.open_gripper() if is_open else ctrl.close_gripper()
                self._last_open[index] = is_open

    def close(self) -> None:
        """Stop the loop, then release both leader arms."""
        super().close()
        self.left_expert.close()
        self.right_expert.close()
