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

"""Real-world robot environments, tasks, and wrapper integration.

Public symbols load lazily to avoid importing optional robotics and vision
dependencies until an environment is requested.
"""

import importlib
import typing

#: Public symbol mapped to its defining module.
_EXPORTS: dict[str, str] = {
    name: module
    for module, names in {
        ".dosw1": ("DOSW1Config", "DOSW1Env", "PickEnv"),
        ".franka": (
            "BottleEnv",
            "DexpnpEnv",
            "DualFrankaEnv",
            "DualFrankaJointEnv",
            "DualFrankaJointRobotConfig",
            "DualFrankaRobotConfig",
            "DualFrankaTCPEnv",
            "DualFrankaTCPRobotConfig",
            "FrankaBinRelocationEnv",
            "FrankaEnv",
            "FrankaRobotConfig",
            "FrankaRobotState",
            "PegInsertionEnv",
        ),
        ".gim_arm": (
            "GimArmEnv",
            "GimArmPegInsertionEnv",
            "GimArmRobotConfig",
            "GimArmRobotState",
        ),
        ".so101": (
            "SO101Env",
            "SO101ReachConfig",
            "SO101ReachEnv",
            "SO101RobotConfig",
            "SO101RobotState",
        ),
        ".xsquare": (
            "ButtonEnv",
            "Turtle2Env",
            "Turtle2RobotConfig",
            "Turtle2RobotState",
        ),
        ".env": ("RealWorldEnv",),
        ".task_env": ("RobotTask", "RobotTaskEnv"),
    }.items()
    for name in names
}

__all__ = sorted(_EXPORTS)

_loaded = False


def _load_all() -> None:
    """Import all robot packages and register their Gymnasium tasks."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in (
        ".dosw1",
        ".franka",
        ".gim_arm",
        ".so101",
        ".xsquare",
        ".task_env",
    ):
        importlib.import_module(module, __name__)
    env_module = importlib.import_module(".env", __name__)
    env_module.RealWorldEnv.realworld_setup()


def __getattr__(name: str) -> "typing.Any":
    """Load the real-world envs the first time one of their names is used."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    _load_all()
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value
