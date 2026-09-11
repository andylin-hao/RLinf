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

"""How a GimArm is driven and what its policy sees, for any task on it.

A GimArm takes absolute joint targets over CAN, with an optional gripper on
the same bus that is either open or closed. It reports its joints and the
tool pose and wrench its controller solves for, so a task may score either.
"""

from dataclasses import dataclass
from typing import Any

from rlinf.envs.real.control import (
    BinaryGripper,
    JointControlConfig,
    JointPositionControl,
)
from rlinf.envs.real.task_env import ObservationSpec, RegisteredTaskEnv, StateField
from rlinf.robotics import GimArmConfig, GimArmRobot
from rlinf.robotics.parts.arms.gim_arm import GimArm
from rlinf.robotics.parts.cameras import CameraInfo

# GIM_ARM_XL joint limits from the gim_arm_control SDK, in radians.
# Override them when using the standard GIM_ARM variant.
_DEFAULT_JOINT_LIMIT_LOW = (-1.4, -3.0, 0.0, -1.5, -1.5, -1.88)
_DEFAULT_JOINT_LIMIT_HIGH = (1.4, 0.0, 3.0, 1.5, 1.5, 1.90)


@dataclass
class GimArmOptions:
    """Settings a GimArm run passes to the arm and its gripper."""

    control_mode: str = "momentum_observer"
    """Arm control mode: ``"idle"``, ``"gravity_comp"``,
    ``"momentum_observer"``, ``"position"``, or ``"torque"``."""

    binary_gripper_threshold: float = 0.5
    """Gripper channel magnitude at which the gripper opens or closes."""


class GimArmEnv(RegisteredTaskEnv):
    """A task on a GimArm, driven by absolute joint targets."""

    ROBOT = GimArmRobot
    CONTROL = JointPositionControl
    OPTIONS = GimArmOptions
    DEFAULTS = {
        "joint_limit_low": _DEFAULT_JOINT_LIMIT_LOW,
        "joint_limit_high": _DEFAULT_JOINT_LIMIT_HIGH,
        "joint_reset_qpos": (0.0,) * 6,
    }
    RETIRED = {
        "save_video_path": "Record episodes with the env's video_cfg instead.",
        "add_gripper_penalty": "Use enable_gripper_penalty.",
    }

    @classmethod
    def make_control(
        cls, hardware: GimArmConfig, config: JointControlConfig, options: GimArmOptions
    ) -> JointPositionControl:
        """Six joints, then a binary gripper channel, kept without a gripper."""
        return JointPositionControl(
            config,
            dof=GimArm.DOF,
            gripper=BinaryGripper(threshold=options.binary_gripper_threshold),
            gripper_fitted=hardware.enable_gripper,
        )

    @classmethod
    def make_observation(
        cls, hardware: GimArmConfig, cameras: tuple[CameraInfo, ...]
    ) -> ObservationSpec:
        """Tool pose, twist and wrench, joints, the gripper, and the cameras."""
        state = (
            StateField("tcp_pose", "tcp_pose", (7,)),
            StateField("tcp_vel", "tcp_vel", (6,)),
            StateField("arm_joint_position", "arm_joint_position", (GimArm.DOF,)),
            StateField(
                "gripper_position",
                "state",
                (1,),
                end_effector=True,
                low=-1.0,
                high=1.0,
                absent=0.0,
            ),
            StateField("tcp_force", "tcp_force", (3,)),
            StateField("tcp_torque", "tcp_torque", (3,)),
        )
        return ObservationSpec(state, cameras=cameras)

    @classmethod
    def robot_options(cls, options: GimArmOptions) -> dict[str, Any]:
        """The arm's control mode."""
        return {"control_mode": options.control_mode}
