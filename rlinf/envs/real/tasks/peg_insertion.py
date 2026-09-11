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

"""Peg insertion: bring a grasped peg down into its hole.

The target is the peg seated in the hole. Between episodes the gripper closes
on the peg and the arm takes it clear of the hole before returning to rest, so
the peg is never dragged sideways out of the slot.

An arm driven by tool poses lifts the peg straight up and moves the tool to
its rest pose. An arm driven by joint targets cannot be sent a pose, so in
``reset_mode="joint"`` it retracts through a joint configuration known to
clear the hole and rests at a joint configuration instead. The reward is the
same either way: the tool's distance to the seated pose.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ResetContext
from .cartesian import CartesianTarget, FixtureConfig, hold, lift
from .requirements import Parts

RESET_MODES = ("cartesian", "joint")


@dataclass
class PegInsertionConfig(FixtureConfig):
    """Settings for :class:`PegInsertion`."""

    random_xy_range: float = 0.05
    random_rz_range: float = np.pi / 6
    clip_x_range: float = 0.05
    clip_y_range: float = 0.05
    clip_z_range_low: float = 0.0
    clip_z_range_high: float = 0.1
    clip_rz_range: float = np.pi / 6

    reset_mode: str = "cartesian"
    """``"cartesian"`` lifts the tool and moves it to ``reset_ee_pose``;
    ``"joint"`` retracts and rests through joint configurations, for an arm
    driven by joint targets."""

    reset_joint_qpos: Optional[Sequence[float]] = None
    """Joint mode: rest configuration in radians. ``None`` rests at zero."""

    safe_retract_qpos: Optional[Sequence[float]] = None
    """Joint mode: configuration that clears the hole on the way to rest.
    ``None`` goes straight to rest."""

    random_joint_noise: float = 0.02
    """Joint mode: largest perturbation of each rest joint when randomising,
    in radians."""

    def __post_init__(self) -> None:
        if self.reset_mode not in RESET_MODES:
            raise ValueError(
                f"reset_mode must be one of {RESET_MODES}, got {self.reset_mode!r}."
            )
        super().__post_init__()


class PegInsertion(CartesianTarget):
    """Insert a grasped peg into a hole at the target pose."""

    CONFIG = PegInsertionConfig
    DESCRIPTION = "peg and insertion"

    config: PegInsertionConfig

    @property
    def _joint_mode(self) -> bool:
        return self.config.reset_mode == "joint"

    def validate(self, dof: Mapping[str, Optional[int]]) -> None:
        """In joint mode, size the rest configuration to the arm."""
        joints = dof.get("arm")
        if not self._joint_mode or joints is None:
            return
        if self.config.reset_joint_qpos is None:
            self.config.reset_joint_qpos = [0.0] * joints
        given = len(self.config.reset_joint_qpos)
        if given != joints:
            raise ValueError(
                f"The arm has {joints} arm joints, so 'reset_joint_qpos' needs "
                f"{joints} values, got {given}."
            )

    def home(self, parts: Parts, context: ResetContext) -> None:
        """Move to rest and let the arm settle."""
        if not self._joint_mode:
            super().home(parts, context)
            return
        parts.arm().reset_joint(list(self.config.reset_joint_qpos))
        time.sleep(1.0)

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Grip the peg, take it clear of the hole, then return to rest."""
        if self._joint_mode:
            self._reset_through_joints(parts, context)
            return
        context.control.grasp(parts)
        hold(parts)
        lift(parts, context, 0.10)
        self.go_to_rest(parts, context)

    def _reset_through_joints(self, parts: Parts, context: ResetContext) -> None:
        rest = np.asarray(self.config.reset_joint_qpos, dtype=np.float64)
        if self.config.enable_random_reset:
            noise = self.config.random_joint_noise
            rest = rest + context.rng.uniform(-noise, noise, size=rest.shape)
            limits = context.control.joint_limits()
            if limits is not None:
                rest = np.clip(rest, *limits)

        arm = parts.arm()
        context.control.grasp(parts)
        if self.config.safe_retract_qpos is not None:
            arm.reset_joint(list(self.config.safe_retract_qpos))
            time.sleep(0.5)
        self.reset_joints_if_due(parts, context)
        arm.reset_joint(list(rest))
