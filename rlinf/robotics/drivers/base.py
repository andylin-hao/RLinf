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

"""Drivers: connections to physical devices.

A :class:`Driver` owns a *session* -- a socket, a CAN bus, a ROS node, a vendor
SDK handle. Two properties follow from that and shape the whole layer:

* **A driver is the unit of placement.** It must run on the machine physically
  wired to the hardware, so it is what gets hosted in a scheduler worker.
* **A driver may back several parts.** A coupled dual-arm controller exposes two
  arms, two end effectors, and wrist cameras through one connection. Parts are
  robot-semantic *views*; the driver is the connection they share.

:meth:`Driver.spawn` is the only scheduler-aware entry point and imports the
scheduler lazily, so driver modules stay importable from plain scripts such as
``toolkits/realworld_check`` with no Ray in the process.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from ..parts.base import Camera, ControllablePart, EndEffector, RobotPart

if TYPE_CHECKING:
    from .handle import DriverHandle

#: Canonical arm observation fields shared by the Franka and GimArm families.
#: An arm part reports these and nothing else; end-effector values belong to the
#: end-effector part, and cameras to camera parts.
ARM_STATE_FIELDS: tuple[str, ...] = (
    "tcp_pose",
    "tcp_vel",
    "arm_joint_position",
    "arm_joint_velocity",
    "tcp_force",
    "tcp_torque",
    "arm_jacobian",
)

#: Ordered most specific first, so a part matches its narrowest kind.
_PART_KINDS: tuple[tuple[str, type], ...] = (
    ("end_effector", EndEffector),
    ("camera", Camera),
    ("controllable", ControllablePart),
    ("part", RobotPart),
)


def _part_kind(part: RobotPart) -> str:
    """Classify a part so a remote proxy can mirror its interface."""
    for kind, part_type in _PART_KINDS:
        if isinstance(part, part_type):
            return kind
    raise TypeError(f"{type(part).__name__} is not a RobotPart.")


class Driver(ABC):
    """A connection to one physical device, backing one or more parts."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the underlying session is live."""

    @abstractmethod
    def connect(self) -> None:
        """Open the session to the device."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the session and release its resources."""

    @abstractmethod
    def parts(self) -> dict[str, RobotPart]:
        """Return the robot-semantic views backed by this connection.

        Keys are driver-local names (``"arm"``, ``"end_effector"``,
        ``"left"``, ``"wrist_1"``); a robot config maps them onto the names the
        policy sees.
        """

    def part(self, name: str) -> RobotPart:
        """Return one named part, or raise a clear configuration error."""
        parts = self.parts()
        if name not in parts:
            raise KeyError(
                f"{type(self).__name__} has no part {name!r}. "
                f"Available parts: {sorted(parts)}."
            )
        return parts[name]

    # -- Part-addressed surface -------------------------------------------
    # These are public, so a hosted driver exposes them as RPCs automatically
    # and ``RemotePart`` can proxy any part through one generic path.

    def describe_parts(self) -> dict[str, dict[str, Any]]:
        """Describe every part: its kind and its feature dictionaries.

        One call carries everything a remote handle needs to build correctly
        typed proxies, so spawning costs a single round trip rather than one
        per part per property.
        """
        described: dict[str, dict[str, Any]] = {}
        for name, part in self.parts().items():
            entry: dict[str, Any] = {
                "kind": _part_kind(part),
                "observation": part.observation_features,
            }
            if isinstance(part, ControllablePart):
                entry["action"] = part.action_features
            described[name] = entry
        return described

    def part_observation(self, name: str) -> dict[str, Any]:
        """Read one part's observation."""
        return self.part(name).get_observation()

    def part_action(self, name: str, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action to one controllable part."""
        part = self.part(name)
        if not isinstance(part, ControllablePart):
            raise TypeError(f"Part {name!r} of {type(self).__name__} is not controllable.")
        return part.send_action(action)

    def part_reset(self, name: str) -> None:
        """Reset one part."""
        self.part(name).reset()

    def shutdown(self) -> None:
        """Disconnect during worker teardown."""
        if self.is_connected:
            self.disconnect()

    # -- Placement --------------------------------------------------------

    @classmethod
    def spawn(
        cls,
        *args: Any,
        node_rank: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "DriverHandle":
        """Construct and connect this driver, locally or on a chosen node.

        With ``node_rank`` unset the driver is built in this process. Otherwise
        it is hosted in a scheduler worker on that node and the returned handle
        proxies to it. Both handles expose the same API, so callers never
        branch on placement.

        Named ``spawn`` rather than ``launch`` deliberately: ``WorkerGroup``
        already defines ``launch``, and it rejects hosted classes that would
        shadow it.
        """
        from .handle import LocalDriverHandle

        if node_rank is None:
            driver = cls(*args, **kwargs)
            driver.connect()
            return LocalDriverHandle(driver)

        from .worker import spawn_driver_worker

        return spawn_driver_worker(cls, args, kwargs, node_rank=node_rank, name=name)


class SinglePartDriver(Driver, ControllablePart):
    """Driver whose connection exposes exactly one controllable part: itself.

    The common case for an arm controller that owns its own socket. Drivers with
    an attached end effector or cameras override :meth:`parts` to add views.
    """

    PART_NAME: str = "arm"

    def parts(self) -> dict[str, RobotPart]:
        """Expose the driver itself as its single part."""
        return {self.PART_NAME: self}
