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

"""Arms, and the hardware sessions behind them.

Each module owns one vendor connection and the parts it exposes through
:attr:`~rlinf.robotics.parts.base.Connection.parts` -- typically the arm
itself plus its end effector, and for coupled hardware several arms at once.

Symbols load lazily so a node without a given vendor SDK can still import this
package.
"""

# ruff: noqa: F822

from importlib import import_module
from typing import Any

_MODULE_BY_NAME: dict[str, str] = {
    "ARM_STATE_FIELDS": ".base",
    "Arm": ".base",
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
    """Import every arm module, so the drivers in them have registered.

    Registration is a decorator in the file that implements a driver, which
    only runs once that file is imported -- and these are imported lazily, so
    that a node missing one vendor SDK can still use the others. Asking the
    :class:`~.base.Arm` registry what exists is the moment they all have to
    have run, and this is where that happens.
    """
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
