# Copyright 2025 The RLinf Authors.
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

"""State vocabulary shared by the Franka arm backends.

Both the ROS-backed and libfranka-backed arms report the same fields, so the
dataclass lives beside them rather than inside either one.
"""

import ipaddress
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FrankaRobotState:
    """Full state of the Franka robot arm and its end-effector.

    The state covers the arm kinematics (pose, velocity, force/torque)
    as well as the end-effector.  For the built-in Franka gripper the
    scalar ``gripper_position`` / ``gripper_open`` fields are used.  For
    dexterous hands such as Ruiyan, the 6-D ``hand_position`` array is
    used instead.
    """

    # https://docs.ros.org/en/kinetic/api/libfranka/html/structfranka_1_1RobotState.html
    tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.O_T_EE
    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    arm_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.q
    arm_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # FrankaState.dq
    tcp_force: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # FrankaState.K_F_ext_hat_K[0:3]
    tcp_torque: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # FrankaState.K_F_ext_hat_K[3:6]
    arm_jacobian: np.ndarray = field(
        default_factory=lambda: np.zeros((6, 7))
    )  # ZeroJacobian.zero_jacobian

    # -- Franka built-in gripper -----------------------------------------
    gripper_position: float = 0.0  # Sum(JointState.position)
    gripper_open: bool = False

    # -- Dexterous hand --------------------------------------------------
    hand_position: Optional[np.ndarray] = None  # 6-D normalised [0, 1]

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


def validated_robot_ip(robot_ip: str, part_name: str) -> str:
    """Return ``robot_ip`` if it is an address an arm could actually dial.

    The check belongs to the part rather than to a robot config: the part is
    what opens the connection, so it is the thing that cannot proceed without a
    real address, and a config placeholder left in a YAML should fail where it
    is used rather than where it is parsed.

    Format only. Whether anything answers at that address is not something a
    constructor can find out without touching the network, and a driver that
    reached for the network before ``connect`` would break placement.
    """
    if not robot_ip:
        raise ValueError(
            f"{part_name} needs a 'robot_ip'; none was given and none could be "
            "resolved from the node's hardware infos."
        )
    try:
        ipaddress.ip_address(robot_ip)
    except ValueError:
        raise ValueError(
            f"{part_name} needs 'robot_ip' to be an IP address, but got "
            f"{robot_ip!r}. A placeholder left in a config reaches the arm as "
            "this."
        ) from None
    return robot_ip
