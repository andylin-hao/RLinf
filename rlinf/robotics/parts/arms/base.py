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

"""The arm category, and the canonical arm every Franka-shaped driver is."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from rlinf.robotics.parts.base import ControllablePart

#: Canonical arm observation fields. An arm reports these and nothing else:
#: end-effector values belong to the end-effector part riding it, and camera
#: frames to camera parts.
ARM_STATE_FIELDS: tuple[str, ...] = (
    "tcp_pose",
    "tcp_vel",
    "arm_joint_position",
    "arm_joint_velocity",
    "tcp_force",
    "tcp_torque",
    "arm_jacobian",
)


class Arm(ControllablePart):
    """A manipulator that moves a tool through space.

    The category, in the same sense as
    :class:`~rlinf.robotics.parts.cameras.base.Camera` and
    :class:`~rlinf.robotics.parts.end_effectors.base.EndEffector`: what a robot
    asks for when it wants an arm, and what the drivers register themselves
    with so a config can name one::

        @Arm.register("franky")
        class FrankyArm(BaseArm): ...


        Arm.backend("franky")  # the class
        Arm.backends()  # every name a config may use

    Two arms driving the same hardware through different stacks -- libfranka
    and ROS both drive a Franka -- are two backends of this one category, and a
    robot swaps them by naming one. Before this existed, that swap lived in a
    table inside the robot that mapped a name to a class *name*, then compared
    that string to decide which module to import.
    """

    @classmethod
    def backends(cls) -> dict[str, type]:
        """Every registered arm, by the name a config selects it with.

        Arm modules are imported lazily, and a driver registers itself when its
        module is imported, so this is where they all have to have been.
        """
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
        """Build this arm from the settings a robot config carries.

        Every backend takes the same settings here and maps them onto its own
        constructor, which is what makes them interchangeable. The mapping
        belongs to the driver because the constructor does: one stack wants a
        ROS package and an end-effector type, another a gripper port, and the
        robot naming a backend should not have to know which.

        Args:
            address: How to reach the arm -- an IP, a bus, a device path.
            gripper_type: Gripper backend, when the arm builds its own.
            gripper_connection: Where that gripper is attached.
            end_effector_type: End effector fitted, when the arm builds it.
            end_effector_config: Settings for that end effector.
            **placement: ``node_rank`` and ``worker_name``, untouched.

        Nothing is opened: this returns a declared arm, like any constructor.
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
    """An arm reporting the canonical fields, from one state object.

    What the Franka and GimArm families are. The three of them had the same two
    methods written out three times, over a tuple that lived beside them rather
    than on anything; an arm that reports something else subclasses
    :class:`Arm` directly and says what it reports.
    """

    #: The fields this arm reports. A driver may narrow it.
    STATE_FIELDS: ClassVar[tuple[str, ...]] = ARM_STATE_FIELDS

    @abstractmethod
    def get_state(self) -> Any:
        """The arm's whole state, as an object with ``to_dict()``."""

    @property
    def observation_features(self) -> dict:
        """Describe the canonical arm fields.

        End-effector values are deliberately absent: they belong to the part
        riding this arm, which the robot composes beneath it.
        """
        return {name: {} for name in self.STATE_FIELDS}

    def get_observation(self) -> dict:
        """Select the canonical fields out of this arm's state."""
        state = self.get_state().to_dict()
        return {name: state[name] for name in self.STATE_FIELDS}
