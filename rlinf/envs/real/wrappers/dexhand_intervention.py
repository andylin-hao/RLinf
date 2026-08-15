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

"""Dexterous-hand intervention, kept as a name for existing configs."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym

from rlinf.envs.real.teleop.adapters import DexHandTeleop
from rlinf.envs.real.teleop.intervention import TeleopIntervention


class DexHandIntervention(TeleopIntervention):
    """Drive the arm with a SpaceMouse and the hand with a glove.

    Args:
        env: The environment to wrap. Its action space must be 12-D: six arm
            channels followed by six hand channels.
        left_port: Serial port of the left glove.
        right_port: Serial port of the right glove.
        glove_frequency: Glove polling rate in Hz.
        glove_config_file: Optional glove calibration file.
        timeout: How long the operator keeps control after their last motion.
    """

    def __init__(
        self,
        env: gym.Env,
        left_port: Optional[str] = "/dev/ttyACM0",
        right_port: Optional[str] = None,
        glove_frequency: int = 60,
        glove_config_file: Optional[str] = None,
        timeout: float = 0.5,
    ) -> None:
        assert env.action_space.shape == (12,), (
            f"DexHandIntervention expects a 12-D action space, "
            f"got {env.action_space.shape}"
        )
        device = DexHandTeleop(
            left_port=left_port,
            right_port=right_port,
            glove_frequency=glove_frequency,
            glove_config_file=glove_config_file,
        )
        device.timeout = timeout
        super().__init__(env, device)

    @property
    def left(self) -> bool:
        """Whether the left button is held."""
        return self.device.left

    @property
    def right(self) -> bool:
        """Whether the right button is held."""
        return self.device.right
