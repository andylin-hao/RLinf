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

"""Opening each composed connection once, and resolving the tree against it.

Composing a robot builds its connections but opens none of them::

    connection = FrankaROSArm("10.0.0.1", node_rank=1)
    robot = FrankaRobot(
        arm=connection.part("arm"),
        end_effector=connection.part("end_effector"),
        scene=RealSenseCamera(info, node_rank=2),
    )
    robot.connect()

:meth:`Robot.connect` owns a :class:`Placement` and hands the tree to it. One
connection is opened once however many parts were picked out of it, so a driver
whose single session backs an arm and its gripper is not opened twice, and the
robot releases it once.

Nobody writes a placement call. This module exists so that ``connect`` and
``disconnect`` can be all-or-nothing.
"""

from typing import Any


class Placement:
    """Opens each distinct connection once and resolves references against it.

    :meth:`Robot.connect` owns one of these. It keeps the handles so the robot
    can release them, and tears down what it already opened if a later part
    fails, so a half-built robot is never handed back.
    """

    def __init__(self) -> None:
        self._handles: dict[int, Any] = {}
        self._order: list[Any] = []

    @property
    def handles(self) -> list[Any]:
        """Handles for every connection opened so far, in the order opened."""
        return list(self._order)

    def resolve(self, value: Any) -> Any:
        """Return the live part a composed value stands for.

        A connection resolves to what it turned out to be once open, and a
        reference from :meth:`Connection.part` to the one capability it names.
        Anything else passes through untouched.
        """
        from ..parts.base import Connection, PartGroup, _ExportRef

        if isinstance(value, _ExportRef):
            return self._handle_for(value.connection).part_named(value.name)
        if isinstance(value, Connection):
            handle = self._handle_for(value)
            parts = handle.parts
            # A leaf -- a camera, a gripper on its own port -- is its own part.
            if not parts:
                return handle.part
            # One part means the hardware is that part.
            if len(parts) == 1:
                return next(iter(parts.values()))
            # Several means the hardware is a subtree. Hand back the whole
            # thing: an arm that carries a gripper resolves to both, so a robot
            # composing the arm never has to reach inside for the gripper.
            return PartGroup(parts)
        return value

    def handle_for(self, connection: Any) -> Any:
        """Return the handle for a connection, opening it on first use."""
        return self._handle_for(connection)

    def _handle_for(self, connection: Any) -> Any:
        # Keyed by identity rather than equality: two arms of the same model on
        # the same node are built from equal arguments and are not the same
        # device, and a constructor's arguments are routinely unhashable.
        key = id(connection)
        handle = self._handles.get(key)
        if handle is None:
            handle = connection.place()
            self._handles[key] = handle
            self._order.append(handle)
        return handle

    def release(self) -> None:
        """Release every open handle, newest first."""
        while self._order:
            handle = self._order.pop()
            handle.disconnect()
        self._handles.clear()
