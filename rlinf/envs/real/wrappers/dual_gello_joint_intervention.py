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

"""Dual-arm GELLO intervention, kept as a name for existing configs."""

from __future__ import annotations

import gymnasium as gym

from rlinf.envs.real.teleop.adapters import DualGelloJointTeleop
from rlinf.envs.real.teleop.intervention import TeleopIntervention


class DualGelloJointIntervention(TeleopIntervention):
    """Drive both arms in joint space from a pair of GELLO leader arms.

    Args:
        env: The environment to wrap.
        left_port: Serial port of the left leader arm.
        right_port: Serial port of the right leader arm.
        gripper_enabled: Whether each arm's action carries a gripper channel.
        use_delta: Whether the env takes joint deltas or absolute targets.
        action_scale: Divisor turning a joint delta into a normalized action.
        direct_stream: Push targets from the device's own thread rather than
            through ``env.step``.
        stream_period: Seconds between pushes when streaming.
    """

    def __init__(
        self,
        env: gym.Env,
        left_port: str,
        right_port: str,
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
        direct_stream: bool = False,
        stream_period: float = 0.001,
    ) -> None:
        super().__init__(
            env,
            DualGelloJointTeleop(
                left_port=left_port,
                right_port=right_port,
                gripper_enabled=gripper_enabled,
                use_delta=use_delta,
                action_scale=action_scale,
                direct_stream=direct_stream,
                stream_period=stream_period,
            ),
            mark_flag=True,
        )
