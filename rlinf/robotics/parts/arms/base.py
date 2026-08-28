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

"""Arm interfaces and the canonical observation schema."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Optional, Protocol

from rlinf.robotics.parts.base import ControllablePart, Features, Observation
from rlinf.utils.logging import get_logger

#: Canonical arm fields; mounted devices expose their own observations.
ARM_STATE_FIELDS: tuple[str, ...] = (
    "tcp_pose",
    "tcp_vel",
    "arm_joint_position",
    "arm_joint_velocity",
    "tcp_force",
    "tcp_torque",
    "arm_jacobian",
)


class ArmState(Protocol):
    """What an arm's own state object has to offer.

    A driver returns whatever dataclass it already keeps -- ``FrankaRobotState``,
    ``GimArmRobotState`` -- and :meth:`BaseArm.get_observation` selects the
    canonical fields out of it. Naming the one method it uses says that without
    naming any one driver's class, and gives ``get_state()`` somewhere to jump
    to instead of ``Any``.
    """

    def to_dict(self) -> dict[str, Any]:
        """This state as a mapping, keyed by the canonical field names."""
        ...


class Arm(ControllablePart):
    """Base category for registered arm backends.

    Drivers register configuration names on this class::

        @Arm.register("franky")
        class FrankyArm(BaseArm): ...
    """

    @classmethod
    def backends(cls) -> dict[str, type]:
        """Return all registered arm backends by configuration name."""
        if cls is Arm:
            from rlinf.robotics.parts.arms import load_drivers

            load_drivers()
        return super().backends()

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        **placement: Any,
    ) -> "Arm":
        """Declare an unconnected arm from standard robot settings.

        Args:
            address: Arm endpoint, such as an IP address, bus, or device path.
            gripper_type: Gripper backend, when the arm builds its own.
            gripper_connection: Where that gripper is attached.
            end_effector_type: End effector fitted, when the arm builds it.
            end_effector_config: Settings for that end effector.
            **placement: Placement arguments forwarded to the connection.
        """
        offered = {
            "gripper_type": gripper_type,
            "gripper_connection": gripper_connection,
            "end_effector_type": end_effector_type,
            "end_effector_config": end_effector_config,
        }
        unused = sorted(name for name, value in offered.items() if value is not None)
        if unused:
            raise TypeError(
                f"{cls.__name__} was given {unused}, which it does not take. "
                "Silently dropping them would leave a robot configured one way "
                f"and running another. Override {cls.__name__}.declare() to "
                "map them onto its constructor, or drop them from the config."
            )
        return cls(address, **placement)


class BaseArm(Arm, ABC):
    """Arm base class that exposes fields from a state object."""

    #: The fields this arm reports. A driver may narrow it.
    STATE_FIELDS: ClassVar[tuple[str, ...]] = ARM_STATE_FIELDS

    @abstractmethod
    def _open(self) -> Any:
        """Open the arm connection and return its device handle."""

    @abstractmethod
    def _release(self, device: Any) -> None:
        """Release the handle returned by :meth:`_open`."""

    @abstractmethod
    def get_state(self) -> ArmState:
        """The arm's whole state, as something with ``to_dict()``."""

    @property
    def observation_features(self) -> Features:
        """Describe the canonical arm observation fields."""
        return {name: {} for name in self.STATE_FIELDS}

    def get_observation(self) -> Observation:
        """Select the canonical fields out of this arm's state."""
        state = self.get_state().to_dict()
        return {name: state[name] for name in self.STATE_FIELDS}

    def reconfigure_compliance_params(self, params: "Mapping[str, float]") -> None:
        """Apply controller compliance settings while the arm is running.

        Backends that fix their gains when the controller starts keep this
        default. It reports the settings it could not apply rather than
        dropping them, so a task tuned against one backend does not appear to
        run with those gains on another.
        """
        if params:
            get_logger().warning(
                "%s cannot change compliance while running; %s were not "
                "applied. Set them where this backend configures its "
                "controller instead.",
                type(self).__name__,
                sorted(params),
            )
