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

"""Supported robots: one module each, holding its config, discovery, and builder."""

from .dosw1 import DOSW1Robot, DOSW1RobotConfig, build_dosw1_robot
from .dual_franka import DualFrankaConfig, DualFrankaRobot, build_dual_franka_robot
from .franka import (
    FrankaArmConfig,
    FrankaConfig,
    FrankaRobot,
    build_franka_robot,
    place_franka_arms,
)
from .gim_arm import GimArmConfig, GimArmRobot, build_gim_arm_robot
from .turtle2 import Turtle2Config, Turtle2Robot, build_turtle2_robot

__all__ = [
    "DOSW1Robot",
    "DOSW1RobotConfig",
    "DualFrankaConfig",
    "DualFrankaRobot",
    "FrankaArmConfig",
    "FrankaConfig",
    "FrankaRobot",
    "GimArmConfig",
    "GimArmRobot",
    "Turtle2Config",
    "Turtle2Robot",
    "build_dosw1_robot",
    "build_dual_franka_robot",
    "build_franka_robot",
    "build_gim_arm_robot",
    "build_turtle2_robot",
    "place_franka_arms",
]
