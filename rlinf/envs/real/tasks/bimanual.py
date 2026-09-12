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

"""Bring several tools to their own targets and hold them all there.

A task with more than one arm differs from a one-armed one in what counts as
success: every tool has its own target, and the step scores only when all of
them are in their zones at once. Everything else -- the workspace around each
target, what happens between episodes -- is stated once per role.

Poses in configs are ``[x, y, z, rx, ry, rz]`` with xyz Euler angles, one row
per role in the order :attr:`MultiArmTargetConfig.roles` names them.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import Evaluation, ResetContext, Task, TaskConfig
from .requirements import Needs, Parts, Reading
from .workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.envs.real.policy import Applied


def _rows(value, roles: Sequence[str], name: str, width: int) -> np.ndarray:
    """Read a per-role table, accepting one row shared by every role."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = np.tile(array, (len(roles), 1))
    if array.shape != (len(roles), width):
        raise ValueError(
            f"{name!r} needs {len(roles)} rows of {width} for roles "
            f"{list(roles)}, got shape {array.shape}."
        )
    return array


@dataclass
class MultiArmTargetConfig(TaskConfig):
    """Settings for :class:`MultiArmTarget`.

    Every table below takes one row per role, in the order ``roles`` names
    them, or a single row applied to all of them.
    """

    roles: tuple[str, ...] = ("left", "right")
    """The arms this task drives, in the order its tables are indexed."""

    target_ee_pose: Sequence = field(
        default_factory=lambda: np.zeros((2, 6), dtype=np.float64)
    )
    """Where each tool must be, ``xyz`` plus xyz Euler angles."""

    reward_threshold: Sequence = field(
        default_factory=lambda: np.full((2, 6), 0.01, dtype=np.float64)
    )
    """How close each tool must be on every position axis to be in its zone."""

    ee_pose_limit_min: Sequence = field(
        default_factory=lambda: np.full((2, 6), -np.inf, dtype=np.float64)
    )
    """Lowest pose each arm may be commanded to."""

    ee_pose_limit_max: Sequence = field(
        default_factory=lambda: np.full((2, 6), np.inf, dtype=np.float64)
    )
    """Highest pose each arm may be commanded to."""

    reset_joint_qpos: Sequence = field(
        default_factory=lambda: np.zeros((2, 7), dtype=np.float64)
    )
    """Configuration each arm returns to between episodes."""

    reset_settle_s: float = 0.5
    """Seconds the arms are given to reach that configuration."""

    dense_gain: float = 500.0
    """Steepness of the dense reward in the tools' combined position error."""

    def __post_init__(self) -> None:
        roles = tuple(self.roles)
        if len(set(roles)) != len(roles) or not roles:
            raise ValueError(f"'roles' must name distinct arms, got {list(roles)}.")
        self.roles = roles
        self.target_ee_pose = _rows(self.target_ee_pose, roles, "target_ee_pose", 6)
        self.reward_threshold = _rows(
            self.reward_threshold, roles, "reward_threshold", 6
        )
        self.ee_pose_limit_min = _rows(
            self.ee_pose_limit_min, roles, "ee_pose_limit_min", 6
        )
        self.ee_pose_limit_max = _rows(
            self.ee_pose_limit_max, roles, "ee_pose_limit_max", 6
        )
        qpos = np.asarray(self.reset_joint_qpos, dtype=np.float64)
        if qpos.ndim == 1:
            qpos = np.tile(qpos, (len(roles), 1))
        if qpos.shape[0] != len(roles):
            raise ValueError(
                f"'reset_joint_qpos' needs {len(roles)} rows for roles "
                f"{list(roles)}, got shape {qpos.shape}."
            )
        self.reset_joint_qpos = qpos


class MultiArmTarget(Task):
    """Bring every tool to its own target and hold them all there.

    The reward is 1 only while every tool is within its
    :attr:`MultiArmTargetConfig.reward_threshold` of its target on each
    position axis. Short of that, the dense reward falls off with the combined
    position error of the tools that are out of their zones, so an arm already
    in place neither helps nor hurts.
    """

    CONFIG = MultiArmTargetConfig

    config: MultiArmTargetConfig

    def requirements(self) -> Mapping[str, Needs]:
        """One arm per role, each reporting its tool pose."""
        return {
            role: Needs(observes=frozenset({"tcp_pose"}))
            for role in self.config.roles
        }

    @property
    def workspace(self) -> Mapping[str, Workspace]:
        """Each role's own pose limits, centred on that role's target."""
        return {
            role: Workspace(
                low=self.config.ee_pose_limit_min[index],
                high=self.config.ee_pose_limit_max[index],
                target_euler=self.config.target_ee_pose[index][3:],
            )
            for index, role in enumerate(self.config.roles)
        }

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Open the end effectors and return every arm to its configuration.

        A teleoperation device that aligns itself to wherever the arms ended
        up asks for ``skip_reset_to_home``, which leaves them there.
        """
        if context.options.get("skip_reset_to_home", False):
            return
        context.action.release(parts)
        for index, role in enumerate(self.config.roles):
            parts.arm(role).reset_joint(self.config.reset_joint_qpos[index])
        time.sleep(self.config.reset_settle_s)

    def evaluate(self, reading: Reading, applied: "Applied") -> Evaluation:
        """Score the tools together: all in their zones, or none of the credit."""
        squared = 0.0
        in_zone = True
        for index, role in enumerate(self.config.roles):
            pose = np.asarray(reading.arm(role)["tcp_pose"], dtype=np.float64)
            delta = np.abs(pose[:3] - self.config.target_ee_pose[index][:3])
            if not np.all(delta <= self.config.reward_threshold[index][:3]):
                in_zone = False
                squared += float(np.sum(np.square(delta)))
        if in_zone:
            return Evaluation(reward=1.0, in_zone=True)
        reward = (
            float(np.exp(-self.config.dense_gain * squared))
            if self.config.use_dense_reward
            else 0.0
        )
        return Evaluation(reward=reward, in_zone=False)
