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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.arms.base import Arm
from ..parts.cameras import BaseCamera
from ..robot import Robot


class FrankaRobot(Robot):
    """Composable Franka robot.

    Single-arm by default. :class:`~..dual_franka.DualFrankaRobot` inherits the
    declaration logic and only changes the backend and the arm count.
    """

    ROBOT_TYPE = "Franka"

    BACKEND: str = "franka_ros"
    """Name of the arm backend this robot drives.

    Any name registered on :class:`~..parts.arms.base.Arm`, so a variant that
    drives the same hardware through a different stack sets this and inherits
    everything else. ``Arm.backends()`` lists them.
    """

    @classmethod
    def declare_arm(
        cls,
        robot_ip: Optional[str],
        *,
        node_rank: int,
        name: str,
        backend: Optional[str] = None,
        gripper_type: str = "franka",
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
    ):
        """Declare one whole arm: its motion and the end effector it carries.

        The end effector rides the arm's own connection, so it comes with the
        arm rather than being composed beside it. Nothing is opened here --
        constructing the arm records where it runs, and ``connect`` opens it
        there.
        """
        resolved_ip = robot_ip or resolve_robot_ip(node_rank)
        if not resolved_ip:
            raise ValueError(
                f"Franka arm {name!r} has no 'robot_ip' and none could be "
                f"resolved from node rank {node_rank}'s hardware infos."
            )
        return Arm.backend(backend or cls.BACKEND).declare(
            resolved_ip,
            gripper_type=gripper_type,
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
            node_rank=node_rank,
            worker_name=name,
        )

    @classmethod
    def build_arms(
        cls,
        *,
        robot_ip: Optional[str],
        node_rank: int,
        worker_rank: int = 0,
        env_idx: int = 0,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        gripper_connection: Optional[str] = None,
    ) -> dict[str, Any]:
        """The arms this robot carries, by name.

        Override this to give a robot a different number of arms; everything
        else about building stays the same.
        """
        connection = cls.declare_arm(
            robot_ip,
            node_rank=node_rank,
            name=f"{cls.ROBOT_TYPE}Arm-{worker_rank}-{env_idx}",
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        # The arm, and with it whatever rides on it: a gripper, or a hand when
        # one is fitted. Naming the end effector here too would put it beside
        # the arm rather than on it, and would have to be kept in step with
        # what the driver actually carries.
        return {"arm": connection}

    @classmethod
    def build_cameras(
        cls,
        cameras: Optional[Mapping[str, Any]] = None,
        *,
        node_rank: Optional[int] = None,
    ) -> dict[str, Any]:
        """The cameras this robot carries, each placed where it is plugged in."""
        return BaseCamera.declare(cameras, node_rank=node_rank)

    @classmethod
    def build(
        cls,
        *,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
        **config: Any,
    ) -> "FrankaRobot":
        """Compose this robot from the parts it is made of.

        Everything that varies between Franka robots lives in ``build_arms``, so
        a variant with a different number of arms overrides that alone and
        inherits this.
        """
        return cls(
            **cls.build_arms(**config),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


@dataclass
class FrankaConfig(RobotConfig):
    """Configuration for a robotic system."""

    REQUIRES_CAMERA = True

    robot_ip: Optional[str] = None
    """IP address of the robotic system.
    When unset in YAML it is auto-detected from the ``ROBOT_IP`` environment
    variable on the node where the arm is enumerated. For a remote
    ``controller_node_rank`` it may stay unset here and be resolved by the
    controller from its node's hardware infos at launch."""

    camera_serials: Optional[list[str]] = None
    """List of camera serial numbers associated with the robot."""

    camera_type: str = "realsense"
    """Camera backend: ``"realsense"``, ``"zed"``, or ``"lumos"``."""

    gripper_type: str = "franka"
    """Gripper backend: ``"franka"`` (ROS-based) or ``"robotiq"`` (Modbus RTU)."""

    gripper_connection: Optional[str] = None
    """Serial port for Robotiq grippers (e.g. ``"/dev/ttyUSB0"``).
    Ignored when *gripper_type* is ``"franka"``."""

    camera_node_rank: Optional[int] = None
    """Node the cameras are plugged into.
    ``None`` (default) co-locates them with the env worker. Set this when the
    cameras hang off a different machine than the one running the policy."""

    controller_node_rank: Optional[int] = None
    """Node rank where the arm part should run.
    When ``None`` (default), the arm is co-located with the env
    worker.  Set this when the arm/gripper and cameras are on different
    machines (e.g. cameras on a GPU server, arm on a NUC)."""

    disable_validate: bool = False
    """Whether to skip the enumeration checks on the cameras this config names.

    Set it for an offline run, or a bench check against faked SDKs, where the
    hardware named here is not attached. The arm's address is not checked at
    enumeration either way: whether it is an address at all is settled by the
    arm part that dials it."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in franka config must be an integer. But got {type(self.node_rank)}."
        )

        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)


def resolve_robot_ip(node_rank: int) -> Optional[str]:
    """Read a robot IP off a node's enumerated hardware.

    A remote arm may leave ``robot_ip`` unset in YAML because only the node
    wired to it knows the address. Any process in the cluster can ask, so this
    resolves before placement rather than inside the hosted part.
    """
    from rlinf.scheduler import Cluster

    try:
        node_info = Cluster().get_node_info(node_rank)
    except Exception:
        return None
    for resource in node_info.hardware_resources:
        for info in resource.infos:
            robot_ip = getattr(getattr(info, "config", None), "robot_ip", None)
            if robot_ip:
                return robot_ip
    return None


FrankaRobot.register_type(FrankaConfig)
