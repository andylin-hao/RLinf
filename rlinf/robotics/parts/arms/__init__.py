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

"""Arm interfaces and registered hardware backends.

Driver modules load lazily so importing the package does not require every arm
SDK.
"""

# ruff: noqa: F822

from importlib import import_module
from typing import Any

_MODULE_BY_NAME: dict[str, str] = {
    "ARM_STATE_FIELDS": ".base",
    "Arm": ".base",
    "ArmState": ".base",
    "BaseArm": ".base",
    "DOSW1Arm": ".dosw1",
    "DOSW1RobotState": ".dosw1",
    "DOSW1ConnectionConfig": ".dosw1",
    "DOSW1EndEffector": ".dosw1",
    "DOSW1Connection": ".dosw1",
    "FrankaROSArm": ".franka_ros",
    "FrankaRobotState": ".franka",
    "FrankyArm": ".franky",
    "GimArm": ".gim_arm",
    "GimArmRobotState": ".gim_arm",
    "Turtle2Connection": ".turtle2",
    "Turtle2RobotState": ".turtle2",
}

__all__ = sorted(_MODULE_BY_NAME)


def load_drivers() -> None:
    """Import all arm modules to populate the backend registry."""
    for module_name in sorted(set(_MODULE_BY_NAME.values())):
        import_module(module_name, __name__)


def __getattr__(name: str) -> Any:
    """Load an arm module only when one of its symbols is requested."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
