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
    "Arm",
    "ArmRuntime",
    "ArmSpec",
    "Camera",
    "CameraSpec",
    "ControllablePart",
    "DOSW1Robot",
    "DOSW1RobotConfig",
    "DualFrankaConfig",
    "DualFrankaRobot",
    "EndEffector",
    "EndEffectorSpec",
    "FrankaConfig",
    "FrankaRobot",
    "GimArmConfig",
    "GimArmRobot",
    "LeggedBase",
    "LegacyObservationAdapter",
    "MobileBase",
    "PartRuntime",
    "PartSpec",
    "RemoteCamera",
    "RemoteControllablePart",
    "RemoteControllerArm",
    "RemoteControllerEndEffector",
    "RemoteEndEffector",
    "RemoteMethodCamera",
    "RemotePart",
    "Robot",
    "RobotAutoConfig",
    "RobotConfig",
    "RobotDiscovery",
    "RobotInfo",
    "RobotPart",
    "RobotRegistration",
    "RobotRuntime",
    "RobotSpec",
    "Turtle2Config",
    "Turtle2Robot",
    "VectorActionAdapter",
    "VectorActionBinding",
    "build_dosw1_runtime",
    "launch_dual_franka_runtime",
    "launch_franka_runtime",
    "launch_gim_arm_runtime",
    "launch_turtle2_runtime",
    "register_robot",
]

_MODULE_BY_NAME = {
    "LegacyObservationAdapter": ".adapters",
    "VectorActionAdapter": ".adapters",
    "VectorActionBinding": ".adapters",
    "RobotAutoConfig": ".config",
    "RobotConfig": ".discovery",
    "RobotDiscovery": ".discovery",
    "RobotInfo": ".discovery",
    "RobotRegistration": ".discovery",
    "register_robot": ".discovery",
    "ArmSpec": ".layout",
    "CameraSpec": ".layout",
    "EndEffectorSpec": ".layout",
    "PartSpec": ".layout",
    "RobotSpec": ".layout",
    "Arm": ".part",
    "Camera": ".part",
    "ControllablePart": ".part",
    "EndEffector": ".part",
    "LeggedBase": ".part",
    "MobileBase": ".part",
    "RobotPart": ".part",
    "Robot": ".robot",
    "DOSW1Robot": ".robots",
    "DOSW1RobotConfig": ".robots",
    "DualFrankaConfig": ".robots",
    "DualFrankaRobot": ".robots",
    "FrankaConfig": ".robots",
    "FrankaRobot": ".robots",
    "GimArmConfig": ".robots",
    "GimArmRobot": ".robots",
    "Turtle2Config": ".robots",
    "Turtle2Robot": ".robots",
    "ArmRuntime": ".runtime",
    "PartRuntime": ".runtime",
    "RemoteCamera": ".runtime",
    "RemoteControllablePart": ".runtime",
    "RemoteControllerArm": ".runtime",
    "RemoteControllerEndEffector": ".runtime",
    "RemoteEndEffector": ".runtime",
    "RemoteMethodCamera": ".runtime",
    "RemotePart": ".runtime",
    "RobotRuntime": ".runtime",
    "build_dosw1_runtime": ".runtime",
    "launch_dual_franka_runtime": ".runtime",
    "launch_franka_runtime": ".runtime",
    "launch_gim_arm_runtime": ".runtime",
    "launch_turtle2_runtime": ".runtime",
}

_DISCOVERY_NAMES = {
    "RobotConfig",
    "RobotDiscovery",
    "RobotInfo",
    "RobotRegistration",
    "register_robot",
}


def __getattr__(name: str) -> Any:
    """Load scheduler-dependent robotics layers only when requested."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    if name in _DISCOVERY_NAMES:
        import_module(".robots", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
