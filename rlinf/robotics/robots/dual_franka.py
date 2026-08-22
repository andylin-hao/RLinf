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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.base import PartGroup
from ..parts.cameras import Camera
from .franka import FrankaRobot


class DualFrankaRobot(FrankaRobot):
    """Composable dual-arm Franka robot."""

    ROBOT_TYPE = "DualFranka"

    BACKEND = "franky"

    @classmethod
    def build_arms(
        cls,
        *,
        left_robot_ip: Optional[str] = None,
        right_robot_ip: Optional[str] = None,
        node_rank: Optional[int] = None,
        left_node_rank: Optional[int] = None,
        right_node_rank: Optional[int] = None,
        worker_rank: int = 0,
        env_idx: int = 0,
        left_gripper_type: str = "robotiq",
        right_gripper_type: str = "robotiq",
        left_gripper_connection: Optional[str] = None,
        right_gripper_connection: Optional[str] = None,
        arm_cameras: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Two arms instead of one, each with whatever rides on its wrist.

        A wrist camera belongs to the arm it is bolted to, so it is named
        inside that arm's group rather than at the top of the robot.
        """
        if not left_robot_ip or not right_robot_ip:
            raise ValueError("Both Franka robot IPs are required for a dual-arm robot.")

        # One node for both arms unless a side names its own, which is what
        # puts a controller next to the arm it drives.
        shared = 0 if node_rank is None else node_rank
        sides = {
            "left": (
                left_robot_ip,
                left_gripper_type,
                left_gripper_connection,
                shared if left_node_rank is None else left_node_rank,
            ),
            "right": (
                right_robot_ip,
                right_gripper_type,
                right_gripper_connection,
                shared if right_node_rank is None else right_node_rank,
            ),
        }
        arms = {}
        for side, (robot_ip, gripper_type, connection, node_rank) in sides.items():
            declared = cls.declare_arm(
                robot_ip,
                node_rank=node_rank,
                name=f"{cls.ROBOT_TYPE}Arm-{side}-{worker_rank}-{env_idx}",
                gripper_type=gripper_type,
                gripper_connection=connection,
            )
            arms[side] = PartGroup(
                arm=declared,
                **Camera.declare((arm_cameras or {}).get(side), node_rank=node_rank),
            )
        return arms


@dataclass
class DualFrankaConfig(RobotConfig):
    """Configuration for a dual-arm Franka robotic system.

    The env process (cameras + teleop) always runs on the node indicated
    by :attr:`node_rank`.  Each arm's low-level controller can be placed
    on a separate node via the ``*_controller_node_rank`` fields — this
    is the key mechanism for *Option D* (main controller + remote arm).
    """

    left_robot_ip: Optional[str] = None
    """IP address of the left Franka arm.
    When unset in YAML it is auto-detected from the ``LEFT_ROBOT_IP``
    environment variable on the node where the arm is enumerated."""

    right_robot_ip: Optional[str] = None
    """IP address of the right Franka arm.
    When unset in YAML it is auto-detected from the ``RIGHT_ROBOT_IP``
    environment variable on the node where the arm is enumerated."""

    left_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the left arm's wrist camera(s)."""

    right_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the right arm's wrist camera(s)."""

    base_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the base (third-person) camera(s)."""

    camera_type: str = "realsense"
    """Default camera backend when a per-slot type is not set.
    Supported: ``"realsense"``, ``"zed"``, ``"lumos"``."""

    base_camera_type: Optional[str] = None
    """Camera backend for the base (third-person) camera(s).
    Falls back to :attr:`camera_type` when ``None``."""

    left_camera_type: Optional[str] = None
    """Camera backend for the left wrist camera(s).
    Falls back to :attr:`camera_type` when ``None``."""

    right_camera_type: Optional[str] = None
    """Camera backend for the right wrist camera(s).
    Falls back to :attr:`camera_type` when ``None``."""

    left_gripper_type: str = "franka"
    """Gripper backend for the left arm."""

    right_gripper_type: str = "franka"
    """Gripper backend for the right arm."""

    left_gripper_connection: Optional[str] = None
    """Serial port for the left arm's Robotiq gripper."""

    right_gripper_connection: Optional[str] = None
    """Serial port for the right arm's Robotiq gripper."""

    left_controller_node_rank: Optional[int] = None
    """Node rank for the left arm part.
    ``None`` means co-located with the env worker."""

    right_controller_node_rank: Optional[int] = None
    """Node rank for the right arm part.
    ``None`` means co-located with the env worker."""

    disable_validate: bool = False
    """Whether to skip the enumeration checks on the hardware this config names.

    Set it for an offline run, or a bench check against faked SDKs. The arm
    addresses are not checked at enumeration either way: whether each is an
    address at all is settled by the arm part that dials it."""

    def __post_init__(self) -> None:  # noqa: D105
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in DualFranka config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.left_camera_serials:
            self.left_camera_serials = list(self.left_camera_serials)
        if self.right_camera_serials:
            self.right_camera_serials = list(self.right_camera_serials)
        if self.base_camera_serials:
            self.base_camera_serials = list(self.base_camera_serials)


DualFrankaRobot.register_type(DualFrankaConfig)
