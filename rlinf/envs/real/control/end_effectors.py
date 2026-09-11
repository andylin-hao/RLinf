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

"""How one action channel commands an end effector."""

from typing import Optional

import numpy as np


class ContinuousGripper:
    """A gripper opened to any fraction of its stroke.

    Args:
        low: Opening at the bottom of the action channel.
        high: Opening at the top of it.
        moved_tolerance: Change in opening between two steps that counts as
            the gripper moving, for a task's gripper penalty.
    """

    def __init__(
        self, low: float = 0.0, high: float = 1.0, moved_tolerance: float = 0.05
    ) -> None:
        self.low = low
        self.high = high
        self.moved_tolerance = moved_tolerance
        self._last: Optional[float] = None

    def reset(self) -> None:
        """Forget the previous episode's opening."""
        self._last = None

    def target(self, value: float) -> tuple[np.ndarray, bool]:
        """Return the command for one action value, and whether it moved."""
        opening = float(np.clip(value, self.low, self.high))
        moved = (
            self._last is not None and abs(opening - self._last) > self.moved_tolerance
        )
        self._last = opening
        return np.array([opening]), moved
