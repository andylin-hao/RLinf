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

"""Franka tasks, and the two envs they are built on.

Each module beside this one is a task: a config dataclass saying where the
target is and how the arm should comply, plus whatever reset behavior that task
needs. The machinery every task shares lives in :mod:`.base` for one arm and
:mod:`.dual_base` for two.

Registering a task is one line, because the only things that differ are the env
class and whether it takes the single- or dual-arm wrapper stack.
"""

from __future__ import annotations

from rlinf.envs.real.registry import WrapperStack, register_tasks
from rlinf.envs.real.wrappers import (
    apply_dual_franka_joint_wrappers,
    apply_single_arm_wrappers,
)
from rlinf.robotics.parts.end_effectors.base import EndEffectorType

from .base import FrankaEnv, FrankaRobotConfig, FrankaRobotState
from .bin_relocation import FrankaBinRelocationEnv
from .bottle import BottleEnv
from .dex_pnp import DexpnpEnv
from .dual_base import DualFrankaEnv, DualFrankaRobotConfig
from .dual_franka_joint import DualFrankaJointEnv, DualFrankaJointRobotConfig
from .dual_franka_tcp import DualFrankaTCPEnv, DualFrankaTCPRobotConfig
from .peg_insertion import PegInsertionEnv

#: Gym id -> the env class behind it and the wrapper stack it takes. A new task
#: is an entry here plus the module implementing it.
TASKS: dict[str, tuple[type, WrapperStack]] = {
    "FrankaEnv-v1": (FrankaEnv, apply_single_arm_wrappers),
    "PegInsertionEnv-v1": (PegInsertionEnv, apply_single_arm_wrappers),
    "FrankaBinRelocationEnv-v1": (FrankaBinRelocationEnv, apply_single_arm_wrappers),
    "BottleEnv-v1": (BottleEnv, apply_single_arm_wrappers),
    "DexpnpEnv-v1": (DexpnpEnv, apply_single_arm_wrappers),
    "DualFrankaJointEnv-v1": (DualFrankaJointEnv, apply_dual_franka_joint_wrappers),
    "DualFrankaTCPEnv-v1": (DualFrankaTCPEnv, apply_dual_franka_joint_wrappers),
}

_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = [
    "TASKS",
    "BottleEnv",
    "DexpnpEnv",
    "DualFrankaEnv",
    "DualFrankaJointEnv",
    "DualFrankaJointRobotConfig",
    "DualFrankaRobotConfig",
    "DualFrankaTCPEnv",
    "DualFrankaTCPRobotConfig",
    "EndEffectorType",
    "FrankaBinRelocationEnv",
    "FrankaEnv",
    "FrankaRobotConfig",
    "FrankaRobotState",
    "PegInsertionEnv",
    *_ENTRY_POINTS,
]
