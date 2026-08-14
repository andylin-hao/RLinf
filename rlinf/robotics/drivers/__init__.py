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

# ruff: noqa: F822

from importlib import import_module
from typing import Any

__all__ = [
    "DOSW1ArmDriver",
    "DOSW1ConnectionConfig",
    "DOSW1EndEffector",
    "DOSW1SDKAdapter",
    "FrankyDriver",
    "FrankaROSDriver",
    "GimArmDriver",
    "Turtle2Driver",
]

_MODULE_BY_NAME = {
    "DOSW1ArmDriver": ".dosw1",
    "DOSW1ConnectionConfig": ".dosw1",
    "DOSW1EndEffector": ".dosw1",
    "DOSW1SDKAdapter": ".dosw1",
    "FrankaROSDriver": ".franka_ros",
    "FrankyDriver": ".franky",
    "GimArmDriver": ".gim_arm",
    "Turtle2Driver": ".turtle2",
}


def __getattr__(name: str) -> Any:
    """Load optional driver modules only when their symbols are requested."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
