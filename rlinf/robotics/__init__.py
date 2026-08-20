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

"""RLinf's robotics layer: parts, and the robots composed from them.

Three concepts:

* **Connection** -- one link to hardware. It knows the machine it runs on, it
  opens, and it closes. Subclass it directly when a single link backs several
  components without being any of them -- a coupled dual-arm controller, a
  two-armed SDK session. Reading such a link would mean nothing, so
  :attr:`~rlinf.robotics.parts.base.Connection.parts` says what rides on it and
  the robot composes those.
* **Part** -- a connection you *can* read, which is what makes it a component
  of the robot: an arm, an end effector, a camera, a mobile base. It has a
  policy-facing observation contract, and an action contract when it is
  controllable. An arm is both a part and its own link, and that is the ordinary
  case.
* **Robot** -- a named composition of parts, and only parts. Hand it a bare
  connection and it says which of the parts to pick instead. See
  :mod:`rlinf.robotics.robot`.

Every connection takes an optional ``node_rank``, so a camera can run on the
machine it is plugged into while the policy runs elsewhere. Nothing else is
needed to say where hardware lives, and no part writes a line for it:
constructing a connection declares it, and
:meth:`~rlinf.robotics.robot.Robot.connect` opens it on the machine it named.

The scheduler never imports this package, and parts never import the scheduler
except through :meth:`~rlinf.robotics.parts.base.Connection.place`, which loads
:mod:`rlinf.robotics.placement` lazily.

Symbols load lazily so a node without a given robot's SDK can still import
``rlinf.robotics``.
"""

# ruff: noqa: F822

from importlib import import_module
from typing import Any

#: Symbols grouped by the module that defines them, so adding one is a
#: single-line change in the group it belongs to.
_MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    ".parts": (
        "Connection",
        "ControllablePart",
        "PartGroup",
        "RobotPart",
        "register_kind",
        "run_parallel",
    ),
    ".parts.arms": ("ARM_STATE_FIELDS",),
    # Each device category lives with the drivers that implement it, and is
    # reached lazily so a node loads only the hardware it has.
    ".parts.cameras": ("BaseCamera", "Camera", "CameraInfo"),
    ".parts.end_effectors": ("EndEffector",),
    ".parts.mobility": ("MobileBase",),
    ".robot": ("Robot",),
    ".parts.views": ("MethodArm", "MethodCamera", "MethodGripper"),
    ".placement": (
        "LocalPartHandle",
        "PartHandle",
        "Placement",
        "RemoteCamera",
        "RemoteControllablePart",
        "RemoteEndEffector",
        "RemotePart",
        "RemotePartHandle",
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
        "FRANKA_BACKENDS",
    ),
    ".adapters": (
        "LegacyObservationAdapter",
        "VectorActionAdapter",
        "VectorActionBinding",
    ),
    ".discovery": (
        "RobotAutoConfig",
        "RobotConfig",
        "RobotDiscovery",
        "RobotInfo",
        "RobotRegistration",
        "build_robot",
        "register_robot",
    ),
}

_MODULE_BY_NAME: dict[str, str] = {
    name: module for module, names in _MODULE_GROUPS.items() for name in names
}

__all__ = sorted(_MODULE_BY_NAME)

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
