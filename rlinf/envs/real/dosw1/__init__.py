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

"""DOSW1 tasks, and the dual-arm env they are built on."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym

from rlinf.envs.real.registry import register_tasks
from rlinf.envs.real.wrappers import LeaderFollowerKeyboardIntervention

from .base import ControlMode, DOSW1Config, DOSW1Env
from .pick import PickEnv


def apply_dosw1_wrappers(env: gym.Env, env_cfg: Mapping[str, Any]) -> gym.Env:
    """Hand the leader arms to the operator when the task asks for it."""
    if (
        env_cfg.get("keyboard_intervention_wrapper", False)
        and getattr(env.config, "enable_human_in_loop", False)
        and not getattr(env.config, "is_dummy", False)
    ):
        env = LeaderFollowerKeyboardIntervention(env)
    return env


#: Gym id -> the env class behind it and the wrapper stack it takes.
TASKS = {"DOSW1PickEnv-v1": (PickEnv, apply_dosw1_wrappers)}

_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = [
    "TASKS",
    "ControlMode",
    "DOSW1Config",
    "DOSW1Env",
    "PickEnv",
    "apply_dosw1_wrappers",
    *_ENTRY_POINTS,
]
