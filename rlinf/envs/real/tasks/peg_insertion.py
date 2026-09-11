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
on the peg, and the arm lifts it clear of the hole before returning to rest,
so the peg is never dragged sideways out of the slot.
"""

from dataclasses import dataclass

import numpy as np

from .base import ResetContext
from .cartesian import CartesianTarget, FixtureConfig, hold, lift
from .requirements import Parts


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


class PegInsertion(CartesianTarget):
    """Insert a grasped peg into a hole at the target pose."""

    CONFIG = PegInsertionConfig
    DESCRIPTION = "peg and insertion"

    config: PegInsertionConfig

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Grip the peg, lift it 10 cm clear of the hole, then return to rest."""
        context.control.grasp(parts)
        hold(parts)
        lift(parts, context, 0.10)
        self.go_to_rest(parts, context)
