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

"""How a Piper is driven and what its policy sees, for any task on it.

A Piper takes absolute joint targets and, when a gripper is fitted, an opening
in ``0..1`` on the same CAN session. It reports its six joints and the tool pose
its own controller solves for, so a task may score either.
"""

from rlinf.envs.real.control import (
    ContinuousGripper,
    JointControlConfig,
    JointPositionControl,
)
from rlinf.envs.real.task_env import ObservationSpec, RegisteredTaskEnv, StateField
from rlinf.robotics import PiperConfig, PiperRobot
from rlinf.robotics.parts.arms.piper import PiperArm
from rlinf.robotics.parts.cameras import CameraInfo


class PiperEnv(RegisteredTaskEnv):
    """A task on a Piper, driven by absolute joint targets."""

    ROBOT = PiperRobot
    CONTROL = JointPositionControl
    DEFAULTS = {
        "joint_limit_low": tuple(PiperArm.JOINT_LIMITS_LOWER),
        "joint_limit_high": tuple(PiperArm.JOINT_LIMITS_UPPER),
    }

    @classmethod
    def make_control(
        cls, hardware: PiperConfig, config: JointControlConfig
    ) -> JointPositionControl:
        """Six joints, then the gripper's opening when one is fitted."""
        gripper = ContinuousGripper() if hardware.with_gripper else None
        return JointPositionControl(config, dof=PiperArm.DOF, gripper=gripper)

    @classmethod
    def make_observation(
        cls, hardware: PiperConfig, cameras: tuple[CameraInfo, ...]
    ) -> ObservationSpec:
        """Joints and tool pose, the gripper's opening, and the cameras."""
        state = [
            StateField("arm_joint_position", "arm_joint_position", (PiperArm.DOF,)),
            StateField("tcp_pose", "tcp_pose", (7,)),
        ]
        if hardware.with_gripper:
            state.append(
                StateField(
                    "gripper_position",
                    "state",
                    (1,),
                    end_effector=True,
                    low=0.0,
                    high=1.0,
                )
            )
        return ObservationSpec(tuple(state), cameras=cameras)
