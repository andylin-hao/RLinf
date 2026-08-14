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

"""A robot: the outermost group of parts, and what places them."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional, TypeVar

from .parts.base import Group, RobotPart

RobotPartType = TypeVar("RobotPartType", bound=RobotPart)


class Robot(Group):
    """A named group of parts that owns their placement.

    A robot is a :class:`~rlinf.robotics.parts.base.Group` like any other, with
    three additions: it knows its registered type, it builds itself from a
    hardware config, and :meth:`connect` places the parts that were declared
    rather than constructed::

        robot = FrankaRobot(
            arm=FrankaROSArm.at(robot_ip, node_rank=1).part("arm"),
            wrist=RealSenseCamera.at(info, node_rank=3),
        )
        robot.connect()

    There are no arm, camera, or base slots. Names carry the meaning, so a robot
    with a lift, a head, or a third arm needs no new concept.
    """

    ROBOT_TYPE: ClassVar[str] = ""

    def __init__(
        self, parts: Optional[Mapping[str, Any]] = None, **named: Any
    ) -> None:
        super().__init__(parts, **named)
        self.handles: dict[str, Any] = {}
        """Connections this robot placed, keyed by the part that needed them.
        Reach hardware methods outside the part interface through these."""

        self._placement: Any = None
        self._declared: Optional[dict[str, Any]] = None

    @classmethod
    def build(cls, **kwargs: Any) -> "Robot":
        """Compose this robot from its hardware config.

        Subclasses implement this. It is what ``register`` hands to the
        registry, so :func:`~rlinf.robotics.discovery.build_robot` can compose a
        robot from its type name alone. Composing does not connect.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement build().")

    @classmethod
    def register(cls, config_cls: type, discovery_cls: type) -> type:
        """Register this robot's config, discovery, and builder in one call."""
        from .discovery import register_robot

        return register_robot(config_cls, cls, build=cls.build)(discovery_cls)

    def parts_of_type(self, part_type: type[RobotPartType]) -> dict[str, RobotPartType]:
        """Return every part implementing ``part_type``, by its dotted path."""
        matches: dict[str, RobotPartType] = {}

        def walk(group: Group, prefix: str) -> None:
            for name, part in group.parts.items():
                path = f"{prefix}{name}"
                if isinstance(part, part_type):
                    matches[path] = part
                if isinstance(part, Group):
                    walk(part, f"{path}.")

        walk(self, "")
        return matches

    @property
    def named_parts(self) -> dict[str, RobotPart]:
        """Every part keyed by its dotted path."""
        return self.parts_of_type(RobotPart)

    def connect(self) -> None:
        """Place any declared parts, then connect everything.

        Each distinct declaration is built once, however many parts refer to it.
        If anything fails, whatever was already placed or connected is torn down
        and the parts go back to their declarations, so you can fix the cause and
        call ``connect`` again.
        """
        from .specs import Placement

        if self._declared is None:
            self._declared = self.declarations()

        placement = self._placement or Placement()
        try:
            for name, handles in self.resolve(placement).items():
                self.handles.setdefault(name, handles[0])
            self._placement = placement
            super().connect()
        except Exception:
            aborted = {id(handle) for handle in placement.handles}
            self.handles = {
                name: handle
                for name, handle in self.handles.items()
                if id(handle) not in aborted
            }
            placement.release()
            self._placement = None
            self.restore(self._declared)
            raise

    def disconnect(self) -> None:
        """Disconnect every part, release the connections, restore declarations.

        A disconnected robot can be connected again: the parts go back to the
        declarations they were composed with, so ``connect`` places fresh ones
        rather than reusing any whose connection is gone.
        """
        super().disconnect()

        placed = set()
        if self._placement is not None:
            placed = {id(handle) for handle in self._placement.handles}
        for handle in reversed(list(self.handles.values())):
            if id(handle) not in placed:
                handle.disconnect()
        if self._placement is not None:
            self._placement.release()
            self._placement = None

        self.handles.clear()
        if self._declared is not None:
            self.restore(self._declared)
