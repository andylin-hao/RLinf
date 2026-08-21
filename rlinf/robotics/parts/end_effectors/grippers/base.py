# Copyright 2025 The RLinf Authors.
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

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from rlinf.robotics.parts.end_effectors.base import BaseEndEffector


class BaseGripper(BaseEndEffector, ABC):
    """A one-degree-of-freedom end effector: two fingers on one axis.

    All gripper implementations (Franka parallel gripper, Robotiq 2F, …)
    implement this interface so that the arm part can use them
    interchangeably.

    A gripper is a :class:`~rlinf.robotics.parts.end_effectors.base.BaseEndEffector`
    and not a sibling of one, so everything a caller can assume about an end
    effector holds here too: it opens through :meth:`_open` and closes through
    :meth:`_release`, its observation is a ``state`` vector, and ``reset``
    takes an optional target. What a gripper adds is the vocabulary its axis
    deserves -- :meth:`open`, :meth:`close`, :meth:`move` -- and the generic
    surface below is written once in terms of those, so a driver implements
    only the three.

    The constructor stores settings and nothing else. It used to open the
    serial port, with ``connect()`` merely checking that it had worked, which
    meant a gripper reported itself connected before anything connected it and
    stayed that way after disconnecting.

    :meth:`is_ready` is a separate question from being connected: a gripper can
    hold its link and still be mid-activation.
    """

    @classmethod
    def declare(cls, *, ros=None, port=None, **settings) -> "BaseGripper":
        """Take whichever attachment this gripper is reached through.

        A Franka Hand rides the arm's ROS session and a Robotiq its own serial
        port, so the arm offers both and each takes one. Overriding here rather
        than branching in the arm is what lets a third gripper arrive without
        the arm changing.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not say which attachment it is reached "
            "through. Override declare() to take the one it uses."
        )

    @abstractmethod
    def open(self, speed: float = 0.3) -> None:
        """Fully open the gripper.

        Args:
            speed: Opening speed, normalized to [0, 1].
        """
        raise NotImplementedError

    @abstractmethod
    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        """Fully close the gripper (or grasp).

        Args:
            speed: Closing speed, normalized to [0, 1].
            force: Grasping force (unit depends on implementation).
        """
        raise NotImplementedError

    @abstractmethod
    def move(self, width: float, speed: float = 0.3) -> None:
        """Move the fingers to an absolute opening width, in metres.

        Zero is closed and :pyattr:`max_width` is fully open, whatever counts
        the hardware underneath actually takes. That conversion belongs to the
        driver, because the driver is the only thing that knows it: a Robotiq
        speaks in 0-255 running the other way, and reading the raw number back
        would tell a policy nothing about the world.

        Args:
            width: Target opening in metres, clamped to ``[0, max_width]``.
            speed: Movement speed, normalized to [0, 1].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def position(self) -> float:
        """Current opening width, in metres."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_width(self) -> float:
        """Opening width of the fully-open gripper, in metres.

        The far end of the axis :meth:`move` and :pyattr:`position` share, so
        it is what bounds an action space built from this part -- and what
        :meth:`open` travels to.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the gripper is currently in the *open* state."""
        raise NotImplementedError

    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the gripper is activated and ready to accept commands."""
        raise NotImplementedError

    # -- The end-effector contract, in terms of the three above -----------

    @property
    def state_dim(self) -> int:
        """One: the position of the single axis the fingers ride."""
        return 1

    @property
    def action_dim(self) -> int:
        """One: a position on that axis."""
        return 1

    @property
    def control_mode(self) -> str:
        """``"continuous"``: :meth:`move` accepts every point on the axis.

        A driver whose hardware only travels to its two ends should say
        ``"binary"`` instead, so a policy is not handed a range it cannot use.
        """
        return "continuous"

    def get_state(self) -> np.ndarray:
        """The opening width, in metres, as the one-element state vector."""
        return np.asarray([self.position], dtype=np.float32)

    def command(self, action: np.ndarray) -> bool:
        """Move to an absolute width, and say whether that changed the grip.

        The target is in the same units :meth:`get_state` reports -- metres --
        which is what lets a policy read a width and write one back. The return
        value follows the end-effector contract: ``True`` when the command
        opened or closed the gripper, which is what a task counts.
        """
        target = np.asarray(action, dtype=np.float32).reshape(-1)
        if target.size != self.action_dim:
            raise ValueError(f"Gripper target must have one value, got {target.size}.")
        was_open = self.is_open
        self.move(float(target[0]))
        return self.is_open != was_open

    def reset(self, target_state: Optional[np.ndarray] = None) -> None:
        """Open the gripper, or move it to ``target_state`` when one is given."""
        if target_state is None:
            self.open()
            return
        self.command(target_state)
