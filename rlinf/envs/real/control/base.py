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

"""How a policy's action vector becomes commands to a robot's parts.

A control owns the policy side of the action: its width, bounds and meaning,
and the wrappers that make sense over it. It does not own the robot, which
stays free of anything a policy decides, or the task, which never sees an
action at all.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.tasks.requirements import Needs, Parts, Reading
from rlinf.robotics.actions import ActionPart


@dataclass
class Applied:
    """What one action did beyond moving the arm.

    Attributes:
        ee_effective: The end effector changed state this step, which a task
            may charge a penalty for.
        is_hand: The end effector is a hand, which is never charged.
    """

    ee_effective: bool = False
    is_hand: bool = False


class Control(ABC):
    """Map a flat policy action onto the parts a task was bound to."""

    #: The config dataclass built from a run's ``override_cfg``.
    CONFIG: ClassVar[type]

    #: Registered action wrappers that fit this action layout.
    ACTION_WRAPPERS: ClassVar[tuple[str, ...]] = ()

    #: Registered observation and action transforms that fit it.
    TRANSFORMS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def requirements(self) -> Mapping[str, Needs]:
        """What each role's parts must accept for this control to drive them."""

    @abstractmethod
    def action_parts(self) -> tuple[ActionPart, ...]:
        """Named slices of the action, in order, tiling its whole width."""

    @abstractmethod
    def action_space(self) -> gym.spaces.Box:
        """Bounds of the policy action."""

    def dof(self) -> Mapping[str, Optional[int]]:
        """Joints of the arm filling each role, where the control knows it."""
        return {}

    def joint_limits(
        self, role: str = "arm"
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Joint bounds for ``role`` in radians, when the control keeps them."""
        return None

    def reset(self) -> None:
        """Forget anything carried between steps of the previous episode."""

    @abstractmethod
    def apply(self, parts: Parts, action: np.ndarray, reading: Reading) -> Applied:
        """Command the parts from one action already clipped to the space.

        Args:
            parts: The parts filling each role.
            action: One policy action.
            reading: The robot as read at the end of the previous step.
        """
