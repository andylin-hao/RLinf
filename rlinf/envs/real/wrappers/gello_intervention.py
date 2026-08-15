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

"""GELLO intervention, kept as a name for existing configs."""

from __future__ import annotations

import gymnasium as gym

from rlinf.envs.real.teleop.adapters import GelloTeleop
from rlinf.envs.real.teleop.intervention import TeleopIntervention


class GelloIntervention(TeleopIntervention):
    """Drive the arm with a GELLO leader arm.

    Args:
        env: The environment to wrap.
        port: Serial port of the GELLO device, typically the ``gello_port``
            field of the env YAML config.
        gripper_enabled: Whether the action space has a gripper channel,
            the inverse of the ``no_gripper`` env config field.
    """

    def __init__(self, env: gym.Env, port: str, gripper_enabled: bool = True) -> None:
        super().__init__(env, GelloTeleop(port=port, gripper_enabled=gripper_enabled))
