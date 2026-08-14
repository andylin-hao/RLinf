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

"""RLinf's robotics layer: parts, drivers, and the robots composed from them.

Three concepts, and the boundary between them is what keeps this layer small:

* **Part** -- a robot-semantic view with a policy-facing contract: an arm, an
  end effector, a camera. See :mod:`rlinf.robotics.parts.base`.
* **Driver** -- a connection to one physical device, backing one or more parts.
  It is the unit of placement. See :mod:`rlinf.robotics.drivers.base`.
* **Robot** -- a named composition of parts. See :mod:`rlinf.robotics.robot`.

The scheduler never imports this package; drivers never import the scheduler
except through :meth:`~rlinf.robotics.drivers.base.Driver.spawn`.

Symbols load lazily so a node without a given robot's SDK can still import
``rlinf.robotics``.
"""

# ruff: noqa: F822

from importlib import import_module
from typing import Any

__all__ = [
    # Parts
    "Arm",
    "Camera",
    "ControllablePart",
    "EndEffector",
    "LeggedBase",
    "MobileBase",
    "RobotPart",
    "run_parallel",
    # Drivers
    "ARM_STATE_FIELDS",
    "Driver",
    "DriverArm",
    "DriverCamera",
    "DriverGripper",
    "DriverHandle",
    "LocalDriverHandle",
    "RemoteCamera",
    "RemoteControllablePart",
    "RemoteDriverHandle",
    "RemoteEndEffector",
    "RemotePart",
    "SinglePartDriver",
    # Composition
    "Robot",
    # Robots
    "DOSW1Robot",
    "DOSW1RobotConfig",
    "DualFrankaConfig",
    "DualFrankaRobot",
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
    # Configuration and discovery
    "LegacyObservationAdapter",
    "RobotAutoConfig",
    "RobotConfig",
    "RobotDiscovery",
    "RobotInfo",
    "RobotRegistration",
    "build_robot",
    "VectorActionAdapter",
    "VectorActionBinding",
    "register_robot",
]

#: Symbols are grouped by the module that defines them, so adding one is a
#: single-line change in the group it belongs to.
_MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    ".parts": (
        "Arm",
        "Camera",
        "ControllablePart",
        "EndEffector",
        "LeggedBase",
        "MobileBase",
        "RobotPart",
        "run_parallel",
    ),
    ".robot": ("Robot",),
    ".drivers.base": (
        "ARM_STATE_FIELDS",
        "Driver",
        "SinglePartDriver",
    ),
    ".drivers.views": ("DriverArm", "DriverCamera", "DriverGripper"),
    ".drivers.handle": (
        "DriverHandle",
        "LocalDriverHandle",
        "RemoteCamera",
        "RemoteControllablePart",
        "RemoteDriverHandle",
        "RemoteEndEffector",
        "RemotePart",
    ),
    ".robots": (
        "DOSW1Robot",
        "DOSW1RobotConfig",
        "DualFrankaConfig",
        "DualFrankaRobot",
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
    ),
    ".adapters": (
        "LegacyObservationAdapter",
        "VectorActionAdapter",
        "VectorActionBinding",
    ),
    ".config": ("RobotAutoConfig",),
    ".discovery": (
        "build_robot",
        "RobotConfig",
        "RobotDiscovery",
        "RobotInfo",
        "RobotRegistration",
        "register_robot",
    ),
    ".layout": (
                        ),
}

_MODULE_BY_NAME: dict[str, str] = {
    name: module for module, names in _MODULE_GROUPS.items() for name in names
}

#: Discovery classes only exist once every robot module has registered itself.
_DISCOVERY_NAMES = frozenset(_MODULE_GROUPS[".discovery"])


def __getattr__(name: str) -> Any:
    """Import the module owning ``name`` on first access."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    if name in _DISCOVERY_NAMES:
        import_module(".robots", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
