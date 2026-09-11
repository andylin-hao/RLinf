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

import time
from collections.abc import Sequence
from typing import Optional

import numpy as np

from rlinf.robotics import EndEffector


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


class BinaryGripper:
    """A gripper that is either open or closed.

    The channel closes the gripper at ``-threshold`` or below and opens it at
    ``threshold`` or above; anything between leaves it as it is. A command
    that would not change the gripper's state is not sent, so a policy holding
    the channel down does not re-grasp every step.

    Args:
        threshold: Magnitude at which the channel acts.
        settle_s: Seconds to wait after a change for the fingers to finish
            moving, so the next step starts from a settled grasp.
    """

    def __init__(self, threshold: float = 0.5, settle_s: float = 0.6) -> None:
        self.threshold = threshold
        self.settle_s = settle_s

    def command(self, effector: EndEffector, value: float) -> bool:
        """Open or close ``effector`` for one channel value.

        Returns:
            Whether the gripper changed state.
        """
        if value <= -self.threshold and effector.is_open:
            effector.close()
        elif value >= self.threshold and not effector.is_open:
            effector.open()
        else:
            return False
        time.sleep(self.settle_s)
        return True


class HandCommand:
    """Finger targets for a dexterous hand, scaled and rate limited.

    Args:
        dim: Fingers the hand drives, one channel each.
        scale: Multiplier from a channel value to a finger target.
        max_delta: Largest change of one finger target between two commands.
        reset_state: Finger pose the hand rests at between episodes.
    """

    def __init__(
        self,
        dim: int,
        *,
        scale: float = 1.0,
        max_delta: float = float("inf"),
        reset_state: Optional[Sequence[float]] = None,
    ) -> None:
        self.dim = dim
        self.scale = scale
        self.max_delta = max_delta
        self.reset_state = np.asarray(
            np.zeros(dim) if reset_state is None else reset_state, dtype=np.float64
        )
        self._last: Optional[np.ndarray] = None

    def command(self, effector: EndEffector, values: np.ndarray) -> bool:
        """Send finger targets, moving each at most ``max_delta`` from the last.

        Returns:
            ``True``: a hand is commanded every step.
        """
        target = np.asarray(values, dtype=np.float64) * self.scale
        if self._last is not None:
            step = np.clip(target - self._last, -self.max_delta, self.max_delta)
            target = self._last + step
        self._last = target.copy()
        effector.command(target)
        return True

    def rest(self, effector: EndEffector) -> None:
        """Put the hand at :attr:`reset_state` and limit the next command from it."""
        effector.reset(self.reset_state)
        self._last = self.reset_state * self.scale
