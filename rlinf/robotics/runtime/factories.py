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

from typing import Any, Optional

from ..drivers import (
    DOSW1ArmDriver,
    DOSW1ConnectionConfig,
    DOSW1EndEffector,
    DOSW1SDKAdapter,
)
from ..part import Arm
from ..robots import (
    DOSW1Robot,
    DualFrankaRobot,
    FrankaRobot,
    GimArmRobot,
    Turtle2Robot,
)
from .controller_proxy import (
    RemoteControllerArm,
    RemoteControllerEndEffector,
    RemoteMethodCamera,
)
from .robot_runtime import RobotRuntime

_ARM_STATE_FIELDS = (
    "tcp_pose",
    "tcp_vel",
    "arm_joint_position",
    "arm_joint_velocity",
    "tcp_force",
    "tcp_torque",
    "arm_jacobian",
)


def build_dosw1_runtime(config: DOSW1ConnectionConfig) -> RobotRuntime:
    """Connect and compose a local DOSW1 dual-arm runtime."""
    sdk = DOSW1SDKAdapter(config)
    left_arm = Arm(
        DOSW1ArmDriver(sdk, "left", owns_connection=True),
        DOSW1EndEffector(sdk, "left"),
    )
    right_arm = Arm(
        DOSW1ArmDriver(sdk, "right"),
        DOSW1EndEffector(sdk, "right"),
    )
    runtime = RobotRuntime(DOSW1Robot.dual_arm(left_arm, right_arm))
    runtime.connect()
    return runtime


def launch_franka_runtime(
    *,
    robot_ip: Optional[str],
    env_idx: int,
    node_rank: int,
    worker_rank: int,
    end_effector_type: str,
    end_effector_config: Optional[dict[str, Any]] = None,
    gripper_connection: Optional[str] = None,
) -> RobotRuntime:
    """Launch and compose one ROS-controlled Franka robot."""
    from .franka_ros import FrankaController

    controller = FrankaController.launch_controller(
        robot_ip=robot_ip or "",
        env_idx=env_idx,
        node_rank=node_rank,
        worker_rank=worker_rank,
        end_effector_type=end_effector_type,
        end_effector_config=end_effector_config,
        gripper_connection=gripper_connection,
    )
    driver = RemoteControllerArm(
        controller,
        command_methods={"tcp_pose": "move_arm"},
        state_fields=_ARM_STATE_FIELDS,
        owns_controller=True,
    )
    if end_effector_type == "ruiyan_hand":
        end_effector = RemoteControllerEndEffector(
            controller,
            state_field="hand_position",
            action_dim=6,
            command_method="command_end_effector",
        )
    else:
        end_effector = RemoteControllerEndEffector(
            controller,
            state_field="gripper_position",
        )
    return RobotRuntime(FrankaRobot.single_arm(Arm(driver, end_effector)))


def launch_dual_franka_runtime(
    *,
    left_robot_ip: Optional[str],
    right_robot_ip: Optional[str],
    left_env_idx: int,
    right_env_idx: int,
    left_node_rank: int,
    right_node_rank: int,
    worker_rank: int,
    left_gripper_type: str,
    right_gripper_type: str,
    left_gripper_connection: Optional[str] = None,
    right_gripper_connection: Optional[str] = None,
) -> RobotRuntime:
    """Launch and compose two independently placed Franka controllers."""
    from .franky import FrankyController

    if not left_robot_ip or not right_robot_ip:
        raise ValueError("Both Franka robot IPs are required for a dual-arm runtime.")

    left_controller: Any = FrankyController.launch_controller(
        robot_ip=left_robot_ip,
        env_idx=left_env_idx,
        node_rank=left_node_rank,
        worker_rank=worker_rank,
        gripper_type=left_gripper_type,
        gripper_connection=left_gripper_connection,
    )
    try:
        right_controller = FrankyController.launch_controller(
            robot_ip=right_robot_ip,
            env_idx=right_env_idx,
            node_rank=right_node_rank,
            worker_rank=worker_rank,
            gripper_type=right_gripper_type,
            gripper_connection=right_gripper_connection,
        )
    except Exception:
        try:
            left_controller.cleanup().wait()
        finally:
            left_controller._close()
        raise

    def make_arm(controller: Any) -> Arm:
        driver = RemoteControllerArm(
            controller,
            command_methods={
                "joint_position": "move_joints",
                "tcp_pose": "move_tcp_pose",
            },
            state_fields=_ARM_STATE_FIELDS,
            owns_controller=True,
        )
        end_effector = RemoteControllerEndEffector(
            controller,
            state_field="gripper_position",
        )
        return Arm(driver, end_effector)

    return RobotRuntime(
        DualFrankaRobot.dual_arm(
            make_arm(left_controller),
            make_arm(right_controller),
        )
    )


def launch_gim_arm_runtime(
    *,
    can_interface: str,
    arm_variant: str,
    enable_gripper: bool,
    gripper_type: str,
    control_mode: str,
    env_idx: int,
    node_rank: int,
    worker_rank: int,
) -> RobotRuntime:
    """Launch and compose one CAN-controlled GimArm robot."""
    from .gim_arm import GimArmController

    controller = GimArmController.launch_controller(
        can_interface=can_interface,
        arm_variant=arm_variant,
        enable_gripper=enable_gripper,
        gripper_type=gripper_type,
        control_mode=control_mode,
        env_idx=env_idx,
        node_rank=node_rank,
        worker_rank=worker_rank,
    )
    driver = RemoteControllerArm(
        controller,
        command_methods={"joint_position": "move_joints"},
        state_fields=_ARM_STATE_FIELDS,
        owns_controller=True,
    )
    end_effector = None
    if enable_gripper:
        end_effector = RemoteControllerEndEffector(
            controller,
            state_field="gripper_position",
        )
    return RobotRuntime(GimArmRobot.single_arm(Arm(driver, end_effector)))


def launch_turtle2_runtime(
    *,
    frequency: int,
    camera_ids: list[int],
    env_idx: int,
    node_rank: int,
    worker_rank: int,
) -> RobotRuntime:
    """Launch and compose the coupled Turtle2 dual-arm controller."""
    from .turtle2 import Turtle2SmoothController

    controller = Turtle2SmoothController.launch_controller(
        freq=frequency,
        env_idx=env_idx,
        node_rank=node_rank,
        worker_rank=worker_rank,
    )

    def make_arm(side: str, state_prefix: str) -> Arm:
        driver = RemoteControllerArm(
            controller,
            command_methods={"tcp_pose": f"move_{side}_arm"},
            state_fields={
                "tcp_pose": f"{state_prefix}_pos",
                "joint_position": f"{state_prefix}_joints",
                "joint_current": f"{state_prefix}_cur_data",
            },
            owns_controller=side == "left",
        )
        end_effector = RemoteControllerEndEffector(
            controller,
            state_field=f"{state_prefix}_pos",
            command_method=f"move_{side}_gripper",
            state_index=6,
        )
        return Arm(driver, end_effector)

    cameras = {
        f"wrist_{index + 1}": RemoteMethodCamera(
            controller,
            "get_camera",
            camera_id,
        )
        for index, camera_id in enumerate(camera_ids)
    }
    return RobotRuntime(
        Turtle2Robot.dual_arm(
            make_arm("left", "follow1"),
            make_arm("right", "follow2"),
            cameras=cameras,
        )
    )
