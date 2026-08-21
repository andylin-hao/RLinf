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

"""Abstract base class for robot end-effectors."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np

from rlinf.robotics.parts.base import ControllablePart


class EndEffector(ControllablePart, ABC):
    """A controllable tool at the end of an arm: a gripper, a dexterous hand.

    What every end effector answers, whoever opens it. A driver holding its own
    serial port and a view riding an arm's bus are both one of these, and a
    task holding one can read it and command it without knowing which it got.

    The lifecycle is deliberately not here. :class:`BaseEndEffector` adds it
    for a device with a link of its own; a view has none and so subclasses this
    directly.
    """

    @classmethod
    def of(cls, end_effector_type: Any, **settings: Any) -> "EndEffector":
        """Build the end effector a config names, without opening it.

        The same shape as
        :meth:`~rlinf.robotics.parts.cameras.base.BaseCamera.of`: a name from
        the registry, and the settings that driver takes. What each one needs
        -- a serial port for a Robotiq, the arm's ROS session for a Franka Hand
        -- it checks for itself, in its own constructor, so a caller building
        one directly is told the same thing as one going through here.

        Args:
            end_effector_type: A registered name, or an
                :class:`EndEffectorType` carrying one.
            **settings: Offered to that driver's :meth:`declare`.
        """
        name = getattr(end_effector_type, "value", end_effector_type)
        return cls.backend(name).declare(**settings)

    @classmethod
    def declare(cls, **settings: Any) -> "EndEffector":
        """Build this driver from what the arm carrying it can offer.

        An arm fitting an end effector offers everything it has to attach one
        with -- its ROS session, a serial port -- because it does not know
        which the configured backend uses. A driver takes what it needs and
        leaves the rest, which is why these are offers rather than settings a
        config asked for: dropping one is the normal case, not a mistake.
        """
        return cls(**settings)

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the end-effector action vector."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the end-effector state vector."""

    @property
    @abstractmethod
    def control_mode(self) -> str:
        """Control mode: ``"binary"`` (open/close) or ``"continuous"``."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current end-effector state as a 1-D array.

        The length of the returned array must equal :pyattr:`state_dim`.
        """

    @abstractmethod
    def command(self, action: np.ndarray) -> bool:
        """Send a command to the end-effector.

        Args:
            action: Action vector whose length equals :pyattr:`action_dim`.

        Returns:
            ``True`` if the command caused a meaningful state change
            (e.g. gripper opened/closed), ``False`` otherwise.
        """

    @property
    def observation_features(self) -> dict:
        """Describe the canonical end-effector state."""
        return {"state": {"shape": (self.state_dim,), "dtype": "float32"}}

    @property
    def action_features(self) -> dict:
        """Describe the canonical end-effector command."""
        return {"target": {"shape": (self.action_dim,), "dtype": "float32"}}

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return the end-effector state under its canonical key."""
        return {"state": self.get_state()}

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the canonical target command."""
        if set(action) != {"target"}:
            raise KeyError("End-effector action must contain only 'target'.")
        self.command(action["target"])
        return {"target": action["target"]}


class EndEffectorType(str, Enum):
    """Supported end-effector types for the Franka robot arm."""

    FRANKA_GRIPPER = "franka_gripper"
    ROBOTIQ_GRIPPER = "robotiq_gripper"
    RUIYAN_HAND = "ruiyan_hand"

    @property
    def is_gripper(self) -> bool:
        return self in (
            type(self).FRANKA_GRIPPER,
            type(self).ROBOTIQ_GRIPPER,
        )

    @property
    def is_hand(self) -> bool:
        return self == type(self).RUIYAN_HAND

    @property
    def gripper_backend(self) -> str:
        if self == type(self).FRANKA_GRIPPER:
            return "franka"
        if self == type(self).ROBOTIQ_GRIPPER:
            return "robotiq"
        raise ValueError(f"{self.value!r} is not a gripper type")


def normalize_end_effector_type(
    end_effector_type: str | EndEffectorType,
    gripper_type: str | None = None,
) -> EndEffectorType:
    if isinstance(end_effector_type, str):
        end_effector_type = EndEffectorType(end_effector_type)

    if end_effector_type.is_hand or gripper_type is None:
        return end_effector_type
    if end_effector_type == EndEffectorType.ROBOTIQ_GRIPPER:
        return end_effector_type

    gt = gripper_type.lower()
    if gt == "franka":
        return EndEffectorType.FRANKA_GRIPPER
    if gt == "robotiq":
        return EndEffectorType.ROBOTIQ_GRIPPER
    raise ValueError(
        f"Unsupported gripper_type={gripper_type!r}. "
        "Supported types: 'franka', 'robotiq'."
    )


class BaseEndEffector(EndEffector, ABC):
    """An end effector that holds a link of its own.

    What every driver in this package subclasses, grippers included: a
    :class:`~rlinf.robotics.parts.end_effectors.grippers.base.BaseGripper` is
    one of these with a single degree of freedom.

    All this adds to :class:`EndEffector` is the lifecycle. What an end
    effector *is* -- its dimensions, its state, the command it takes -- belongs
    to the category, so a view riding an arm's bus answers the same questions
    without pretending to own a device.
    """

    @abstractmethod
    def _open(self) -> Any:
        """Reach the hardware -- open the serial port, claim the bus -- and
        return whatever speaks to it.

        The same contract as every other part, restated as ``abstractmethod``
        so an end-effector driver that never wrote one is refused at class
        definition rather than at the first ``connect``. Return nothing when
        the device is opened in place and there is no handle to keep.
        """

    @abstractmethod
    def _release(self, device: Any) -> None:
        """Let go of exactly what :meth:`_open` returned.

        The handle arrives as an argument rather than off ``self`` so teardown
        cannot be defeated by the order ``disconnect`` does things in.
        """

    @property
    def finger_names(self) -> list[str]:
        """Human-readable names for each DOF.

        Subclasses may override this to provide meaningful labels.
        The default returns generic names ``["dof_0", "dof_1", ...]``.
        """
        return [f"dof_{i}" for i in range(self.state_dim)]

    def get_detailed_state(self) -> dict:
        """Return a detailed status dictionary for diagnostic purposes.

        The default implementation wraps :meth:`get_state` into a dict.
        Subclasses (e.g. dexterous hands) should override this to expose
        per-motor velocity, current, error status, etc.
        """
        state = self.get_state()
        return {
            "positions": state.tolist(),
            "finger_names": self.finger_names,
        }

    @abstractmethod
    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Reset the end-effector to a default or specified state.

        Args:
            target_state: Optional target state. If ``None``, reset to the
                implementation-defined default.
        """
