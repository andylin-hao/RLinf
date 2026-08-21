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
    def move(self, position: float, speed: float = 0.3) -> None:
        """Move gripper to an absolute position.

        Args:
            position: Target position. Semantics are implementation-specific
                (e.g. Franka uses width in metres, Robotiq uses 0–255).
            speed: Movement speed.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def position(self) -> float:
        """Current gripper opening width / position."""
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
        """The position of the axis, as the one-element state vector."""
        return np.asarray([self.position], dtype=np.float32)

    def command(self, action: np.ndarray) -> bool:
        """Move to an absolute position, and say whether that changed the grip.

        The return value follows the end-effector contract: ``True`` when the
        command opened or closed the gripper, which is what a task counts.
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
