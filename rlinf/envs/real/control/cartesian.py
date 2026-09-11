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

"""Tool-frame deltas for one arm, with a gripper or a hand."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional, Union

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.tasks.requirements import Needs, Parts, Reading, nest
from rlinf.envs.real.tasks.workspace import Workspace
from rlinf.robotics.actions import ActionKind, ActionPart

from .base import Applied, Control
from .end_effectors import BinaryGripper, HandCommand


@dataclass
class CartesianControlConfig:
    """How far one action moves the tool, and how its end effector is driven."""

    action_scale: Sequence[float] = (1.0, 1.0, 1.0)
    """Metres per unit of position action, radians per unit of rotation
    action, and the multiplier on the gripper channel."""

    compliance_param: Mapping[str, float] = field(default_factory=dict)
    """Impedance gains applied at the start of every episode. Empty leaves the
    arm's controller as it is."""

    binary_gripper_threshold: float = 0.5
    """Gripper channel magnitude at which the gripper opens or closes."""

    hand_action_scale: float = 1.0
    """Multiplier from a hand channel to a finger target."""

    hand_max_delta_per_step: float = float("inf")
    """Largest change of one finger target between two steps."""

    hand_reset_state: Sequence[float] = (0.0,) * 6
    """Finger pose a hand rests at between episodes."""

    def __post_init__(self) -> None:
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)
        self.hand_reset_state = np.asarray(self.hand_reset_state, dtype=np.float64)
        self.compliance_param = dict(self.compliance_param)


class CartesianDeltaControl(Control):
    """Move one arm's tool by a delta each step, and drive its end effector.

    The action is ``[dx, dy, dz, rx, ry, rz]`` followed by the end-effector
    channels. The position delta is scaled and added to the tool's position;
    the rotation delta is scaled and composed onto its orientation. The pose
    is then brought inside the task's workspace and sent as a ``tcp_pose``
    target. The end effector is commanded first, so a grasp closes before the
    arm moves away.

    Args:
        config: Scales, gains, and end-effector settings.
        end_effector: How the end-effector channels are read: one binary
            channel for a gripper, one channel per finger for a hand.
        role: The role this control drives.
    """

    CONFIG = CartesianControlConfig
    ACTION_WRAPPERS = ("GripperCloseEnv",)
    TRANSFORMS = ("RelativeFrame", "Quat2EulerWrapper")

    def __init__(
        self,
        config: CartesianControlConfig,
        *,
        end_effector: Union[BinaryGripper, HandCommand],
        role: str = "arm",
    ) -> None:
        super().__init__(config)
        self.end_effector = end_effector
        self.role = role
        self.workspace: Optional[Workspace] = None

    @property
    def _is_hand(self) -> bool:
        return isinstance(self.end_effector, HandCommand)

    @property
    def _ee_width(self) -> int:
        return self.end_effector.dim if self._is_hand else 1

    def requirements(self) -> Mapping[str, Needs]:
        """A pose to move from and a pose target, with a gripper or a hand."""
        return {
            self.role: Needs(
                observes=frozenset({"tcp_pose"}),
                commands=frozenset({"tcp_pose"}),
                end_effector="hand" if self._is_hand else "gripper",
            )
        }

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Six tool deltas, then the gripper or the fingers."""
        if self._is_hand:
            effector = ActionPart("hand", self._ee_width, ActionKind.HAND)
        else:
            effector = ActionPart("end_effector", 1, ActionKind.GRIPPER)
        return (ActionPart(self.role, 6, ActionKind.CARTESIAN_DELTA), effector)

    def action_space(self) -> gym.spaces.Box:
        """Every channel in ``[-1, 1]``."""
        width = 6 + self._ee_width
        return gym.spaces.Box(
            -np.ones(width, dtype=np.float32), np.ones(width, dtype=np.float32)
        )

    def confine(self, workspace: Optional[Workspace]) -> None:
        """Keep every commanded pose inside ``workspace``."""
        self.workspace = workspace

    def reset(self, parts: Optional[Parts] = None) -> None:
        """Apply the episode's impedance gains."""
        if parts is not None:
            parts.arm(self.role).reconfigure_compliance_params(
                self.config.compliance_param
            )

    def apply(self, parts: Parts, action: np.ndarray, reading: Reading) -> Applied:
        """Command the end effector, then the tool pose one delta away."""
        scale = self.config.action_scale
        current = np.asarray(reading.arm(self.role)["tcp_pose"], dtype=np.float64)
        target = current.copy()
        target[:3] += action[:3] * scale[0]
        target[3:] = (
            R.from_euler("xyz", action[3:6] * scale[1]) * R.from_quat(current[3:])
        ).as_quat()

        applied = self._command_end_effector(parts, action[6:])
        if self.workspace is not None:
            target = self.workspace.clip(target, current)
        parts.arm(self.role).clear_errors()
        parts.robot.send_action(
            nest(parts.bound[self.role].part, {"tcp_pose": target.astype(np.float32)})
        )
        return applied

    def grasp(self, parts: Parts) -> bool:
        """Close the gripper as a ``-1`` on its channel would."""
        return self._command_end_effector(parts, np.array([-1.0])).ee_effective

    def release(self, parts: Parts) -> bool:
        """Open the gripper, or bring a hand's fingers to their resting pose."""
        channel = self.end_effector.reset_state if self._is_hand else np.array([1.0])
        return self._command_end_effector(parts, channel).ee_effective

    def rest_end_effector(self, parts: Parts) -> None:
        """Put a hand at its resting pose; a gripper keeps its state."""
        if self._is_hand:
            self.end_effector.rest(parts.end_effector(self.role))

    def action_scale(self) -> Optional[np.ndarray]:
        """The position, rotation, and gripper scales."""
        return self.config.action_scale

    def gripper_open(self, parts: Parts) -> Optional[bool]:
        """Whether the end effector reports itself open."""
        return bool(parts.end_effector(self.role).is_open)

    def hand_reset_pose(self) -> Optional[np.ndarray]:
        """The hand's resting finger pose; ``None`` for a gripper."""
        return self.end_effector.reset_state if self._is_hand else None

    def _command_end_effector(self, parts: Parts, channels: np.ndarray) -> Applied:
        effector = parts.end_effector(self.role)
        if self._is_hand:
            return Applied(
                ee_effective=self.end_effector.command(effector, channels),
                is_hand=True,
            )
        value = float(channels[0]) * self.config.action_scale[2]
        return Applied(ee_effective=self.end_effector.command(effector, value))
