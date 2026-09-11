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

"""Absolute joint targets, one arm, with an optional gripper channel."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.tasks.requirements import Needs, Parts, Reading, merge, nest
from rlinf.robotics.actions import ActionKind, ActionPart

from .base import Applied, Control
from .end_effectors import ContinuousGripper


@dataclass
class JointControlConfig:
    """Joint bounds for an absolute joint-position action, in radians."""

    joint_limit_low: Optional[Sequence[float]] = None
    """Lowest joint target a policy may send. A robot preset fills this in."""

    joint_limit_high: Optional[Sequence[float]] = None
    """Highest joint target a policy may send. A robot preset fills this in."""


class JointPositionControl(Control):
    """Drive one arm with absolute joint targets and a gripper opening.

    The action is ``dof`` joint positions in radians, clipped to the joint
    bounds, followed by one gripper value when a gripper is driven. When the
    gripper rides the arm, as on a shared servo bus, both go out in one
    command.

    Args:
        config: The joint bounds, which must be ``dof`` wide.
        dof: Joints the arm drives.
        gripper: How the gripper channel is read, or ``None`` for no channel.
        role: The role this control drives.
    """

    CONFIG = JointControlConfig

    def __init__(
        self,
        config: JointControlConfig,
        *,
        dof: int,
        gripper: Optional[ContinuousGripper] = None,
        role: str = "arm",
    ) -> None:
        super().__init__(config)
        if config.joint_limit_low is None or config.joint_limit_high is None:
            raise ValueError(
                "Joint control needs 'joint_limit_low' and 'joint_limit_high'."
            )
        self._low = np.asarray(config.joint_limit_low, dtype=np.float64)
        self._high = np.asarray(config.joint_limit_high, dtype=np.float64)
        for name, bound in (
            ("joint_limit_low", self._low),
            ("joint_limit_high", self._high),
        ):
            if bound.shape != (dof,):
                raise ValueError(
                    f"The arm has {dof} joints, so {name!r} needs {dof} values, "
                    f"got {bound.size}."
                )
        self._dof = dof
        self._gripper = gripper
        self._role = role

    def requirements(self) -> Mapping[str, Needs]:
        """Joint targets, and a gripper when this control opens one."""
        return {
            self._role: Needs(
                commands=frozenset({"joint_position"}),
                end_effector="gripper" if self._gripper is not None else None,
            )
        }

    def action_parts(self) -> tuple[ActionPart, ...]:
        """The joints, then the gripper when there is one."""
        parts = [ActionPart(self._role, self._dof, ActionKind.JOINT_POSITION)]
        if self._gripper is not None:
            parts.append(ActionPart("end_effector", 1, ActionKind.GRIPPER))
        return tuple(parts)

    def action_space(self) -> gym.spaces.Box:
        """Joint bounds, then the gripper's range."""
        low = self._low.astype(np.float32)
        high = self._high.astype(np.float32)
        if self._gripper is not None:
            low = np.append(low, np.float32(self._gripper.low))
            high = np.append(high, np.float32(self._gripper.high))
        return gym.spaces.Box(low, high)

    def dof(self) -> Mapping[str, Optional[int]]:
        """The driven arm's joints."""
        return {self._role: self._dof}

    def joint_limits(
        self, role: str = "arm"
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """The joint bounds in radians."""
        return (self._low, self._high) if role == self._role else None

    def reset(self, parts: Optional[Parts] = None) -> None:
        """Forget the gripper's last opening."""
        if self._gripper is not None:
            self._gripper.reset()

    def apply(self, parts: Parts, action: np.ndarray, reading: Reading) -> Applied:
        """Send the joint targets and the gripper opening together."""
        bound = parts.bound[self._role]
        command = nest(bound.part, {"joint_position": action[: self._dof]})
        moved = False
        if self._gripper is not None:
            target, moved = self._gripper.target(action[self._dof])
            merge(command, nest(bound.end_effector, {"target": target}))
        parts.robot.send_action(command)
        return Applied(ee_effective=moved)
