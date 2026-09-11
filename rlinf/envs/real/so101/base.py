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

"""How an SO-101 is driven and what its policy sees, for any task on it.

The SO-101 reports joint positions and nothing else: it has no pose, force, or
torque sensing, and no kinematic model ships with it. Its five joints and its
gripper share one servo bus, so both go out in one joint-space command.
"""

import numpy as np

from rlinf.envs.real.control import (
    ContinuousGripper,
    JointControlConfig,
    JointPositionControl,
)
from rlinf.envs.real.task_env import ObservationSpec, RegisteredTaskEnv, StateField
from rlinf.robotics import SO101Config, SO101Robot
from rlinf.robotics.parts.arms.so101 import SO101Arm
from rlinf.robotics.parts.cameras import CameraInfo

#: Joint travel of the SO-101, in radians. The servos turn further than this;
#: these are the limits the arm can hold without the links colliding.
_DEFAULT_JOINT_LIMIT_LOW = np.array([-1.91, -1.75, -1.69, -1.66, -2.79])
_DEFAULT_JOINT_LIMIT_HIGH = np.array([1.91, 1.75, 1.69, 1.66, 2.79])


class SO101Env(RegisteredTaskEnv):
    """A task on an SO-101, driven by absolute joint targets."""

    ROBOT = SO101Robot
    CONTROL = JointPositionControl
    # The leader arm is the same five joints and gripper as this follower.
    TELEOP = ("so101_leader",)
    DEFAULTS = {
        "joint_limit_low": tuple(_DEFAULT_JOINT_LIMIT_LOW),
        "joint_limit_high": tuple(_DEFAULT_JOINT_LIMIT_HIGH),
    }

    @classmethod
    def make_control(
        cls, hardware: SO101Config, config: JointControlConfig
    ) -> JointPositionControl:
        """Five joints, then the gripper's opening."""
        return JointPositionControl(
            config, dof=SO101Arm.DOF, gripper=ContinuousGripper()
        )

    @classmethod
    def make_observation(
        cls, hardware: SO101Config, cameras: tuple[CameraInfo, ...]
    ) -> ObservationSpec:
        """Joints, the gripper's opening, and the cameras."""
        state = (
            StateField("arm_joint_position", "arm_joint_position", (SO101Arm.DOF,)),
            StateField(
                "gripper_position",
                "state",
                (1,),
                end_effector=True,
                low=0.0,
                high=1.0,
            ),
        )
        return ObservationSpec(state, cameras=cameras)
