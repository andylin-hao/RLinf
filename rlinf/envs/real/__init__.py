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

from .dosw1 import DOSW1Config, DOSW1Env, PickEnv
from .franka import (
    BottleEnv,
    DexpnpEnv,
    DualFrankaEnv,
    DualFrankaJointEnv,
    DualFrankaJointRobotConfig,
    DualFrankaRobotConfig,
    DualFrankaTCPEnv,
    DualFrankaTCPRobotConfig,
    FrankaBinRelocationEnv,
    FrankaEnv,
    FrankaRobotConfig,
    FrankaRobotState,
    PegInsertionEnv,
)
from .gim_arm import (
    GimArmEnv,
    GimArmPegInsertionEnv,
    GimArmRobotConfig,
    GimArmRobotState,
)
from .realworld_env import RealWorldEnv
from .robot_task_env import RobotTask, RobotTaskEnv
from .xsquare import ButtonEnv, Turtle2Env, Turtle2RobotConfig, Turtle2RobotState

RealWorldEnv.realworld_setup()

__all__ = [
    "DualFrankaEnv",
    "DualFrankaJointEnv",
    "DualFrankaJointRobotConfig",
    "DualFrankaTCPEnv",
    "DualFrankaTCPRobotConfig",
    "DualFrankaRobotConfig",
    "BottleEnv",
    "ButtonEnv",
    "DOSW1Config",
    "DOSW1Env",
    "DexpnpEnv",
    "FrankaBinRelocationEnv",
    "FrankaEnv",
    "FrankaRobotConfig",
    "FrankaRobotState",
    "GimArmEnv",
    "GimArmPegInsertionEnv",
    "GimArmRobotConfig",
    "GimArmRobotState",
    "Turtle2Env",
    "Turtle2RobotConfig",
    "Turtle2RobotState",
    "RealWorldEnv",
    "RobotTask",
    "RobotTaskEnv",
    "PegInsertionEnv",
    "PickEnv",
]
