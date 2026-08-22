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
    """Common observation and action interface for end effectors.

    Implementations may own a connection or borrow the connection of the arm
    that carries them.
    """

    @classmethod
    def of(
        cls, end_effector_type: "str | EndEffectorType", **settings: Any
    ) -> "EndEffector":
        """Declare an end effector from a registered backend name.

        Args:
            end_effector_type: A registered name, or an
                :class:`EndEffectorType` carrying one.
            **settings: Offered to that driver's :meth:`declare`.
        """
        name = getattr(end_effector_type, "value", end_effector_type)
        return cls.backend(name).declare(**settings)

    @classmethod
    def declare(cls, **settings: Any) -> "EndEffector":
        """Declare this backend from attachment settings offered by an arm."""
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
    """Base class for end effectors with an independent connection."""

    @abstractmethod
    def _open(self) -> Any:
        """Open the end effector and return its device handle."""

    @abstractmethod
    def _release(self, device: Any) -> None:
        """Release the handle returned by :meth:`_open`."""

    @property
    def finger_names(self) -> list[str]:
        """Human-readable names for each DOF.

        Subclasses may override this to provide meaningful labels.
        The default returns generic names ``["dof_0", "dof_1", ...]``.
        """
        return [f"dof_{i}" for i in range(self.state_dim)]

    def get_detailed_state(self) -> dict:
        """Return diagnostic state using the generic position representation."""
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
