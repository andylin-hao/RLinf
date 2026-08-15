# Copyright 2025 The RLinf Authors.
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

"""SpaceMouse intervention, kept as a name for existing configs."""

from __future__ import annotations

import gymnasium as gym

from rlinf.envs.real.teleop.adapters import (
    SpaceMouseTeleop,
)
from rlinf.envs.real.teleop.adapters import (
    sample_gripper_action as sample_gripper_action,
)
from rlinf.envs.real.teleop.intervention import TeleopIntervention


class SpacemouseIntervention(TeleopIntervention):
    """Drive the arm with a SpaceMouse, overriding the policy while it moves.

    Args:
        env: The environment to wrap.
        gripper_enabled: Whether the action space has a gripper channel.
    """

    def __init__(self, env: gym.Env, gripper_enabled: bool = True) -> None:
        super().__init__(env, SpaceMouseTeleop(gripper_enabled=gripper_enabled))

    @property
    def left(self) -> bool:
        """Whether the left button is held."""
        return self.device.left

    @property
    def right(self) -> bool:
        """Whether the right button is held."""
        return self.device.right
