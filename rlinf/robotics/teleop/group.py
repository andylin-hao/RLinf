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

"""Several devices, each driving a named part of one action.

A dexterous-hand setup is a spacemouse on the arm and a glove on the hand. A
dual-arm GELLO setup is one leader per side. Each of those used to be a class
that read its devices and spliced their outputs together by hand; here they are
the same object with a different list.

:class:`TeleopGroup` is the teleop counterpart of
:class:`~rlinf.robotics.parts.base.Group`: it fans out over what it holds and
merges the results by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from ..parts.teleop.devices import TeleopPart
from .binding import TeleopBinding


@dataclass
class TeleopEntry:
    """One device, what it means, and which branch it drives.

    Attributes:
        device: The hardware the operator drives.
        binding: What its reading means for the robot.
        drives: Branch of a multi-arm action this entry fills. ``None`` on a
            robot with one of each part, where the binding's own names suffice.
    """

    device: TeleopPart
    binding: TeleopBinding
    drives: Optional[str] = None

    @property
    def parts(self) -> tuple[str, ...]:
        """Action parts this entry fills, qualified by its branch."""
        if self.drives is None:
            return self.binding.PRODUCES
        return tuple(f"{self.drives}.{part}" for part in self.binding.PRODUCES)


class TeleopGroup:
    """Devices and their bindings, producing one action.

    Args:
        entries: The devices in play.
        available: Action parts the robot actually has. A binding offering a
            part outside this set does not fill it, which is how a spacemouse
            drives the arm of a robot whose end effector is a hand rather than a
            gripper.

    Raises:
        ValueError: If two entries fill the same part, or an entry fills
            nothing. Both are mistakes worth reporting when the group is built
            rather than at the first step with a robot moving.
    """

    def __init__(
        self,
        entries: Iterable[TeleopEntry],
        available: Optional[Iterable[str]] = None,
    ) -> None:
        self.entries = list(entries)
        self.available = None if available is None else set(available)
        self._filled = self._resolve()

    def _resolve(self) -> dict[str, TeleopEntry]:
        """Decide which entry fills which part, and refuse a conflict."""
        filled: dict[str, TeleopEntry] = {}
        for entry in self.entries:
            claimed = [
                part
                for part in entry.parts
                if self.available is None or part in self.available
            ]
            if not claimed:
                raise ValueError(
                    f"{type(entry.binding).__name__} fills none of this robot's "
                    f"action parts. It offers {list(entry.parts)}; the robot has "
                    f"{sorted(self.available or [])}."
                )
            for part in claimed:
                if part in filled:
                    raise ValueError(
                        f"{type(entry.binding).__name__} and "
                        f"{type(filled[part].binding).__name__} both drive "
                        f"{part!r}. Give one of them a different 'drives'."
                    )
                filled[part] = entry
        return filled

    @property
    def parts(self) -> tuple[str, ...]:
        """Every action part this group fills."""
        return tuple(self._filled)

    @property
    def devices(self) -> tuple[TeleopPart, ...]:
        """Each distinct device, in the order it was given.

        Distinct by identity: one device listed under two parts is read once,
        the same way one connection is opened once.
        """
        seen: list[TeleopPart] = []
        for entry in self.entries:
            if not any(entry.device is device for device in seen):
                seen.append(entry.device)
        return tuple(seen)

    def connect(self) -> None:
        """Open every device."""
        for device in self.devices:
            if not device.is_connected:
                device.connect()

    def disconnect(self) -> None:
        """Close every device, newest first."""
        for device in reversed(self.devices):
            if device.is_connected:
                device.disconnect()

    def reset(self) -> None:
        """Drop anything the bindings held from the previous episode."""
        for entry in self.entries:
            entry.binding.reset()

    def action(
        self, context: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray], bool, dict[str, Any]]:
        """Read every device and merge what its binding fills.

        Returns:
            The action parts by name, whether any operator is driving, and any
            info the bindings want recorded.
        """
        parts: dict[str, np.ndarray] = {}
        driving = False
        info: dict[str, Any] = {}

        readings = {id(device): device.get_observation() for device in self.devices}
        # Entries are processed in order so a binding can gate the ones after
        # it, which is how a spacemouse button puts a glove in control.
        running = dict(context)
        for entry in self.entries:
            reading = readings[id(entry.device)]
            running.update(entry.binding.publish(reading))
            produced = entry.binding.action(reading, running)
            for name, value in produced.items():
                qualified = name if entry.drives is None else f"{entry.drives}.{name}"
                if self.available is not None and qualified not in self.available:
                    continue
                parts[qualified] = value
            driving |= entry.binding.is_driving(reading)
        return parts, driving, info
