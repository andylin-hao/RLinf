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

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class CameraSpec:
    """Physical camera placement in a robot layout."""

    name: str
    camera_type: str
    serial_number: str
    node_rank: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CameraSpec.name must be non-empty.")
        if not self.camera_type:
            raise ValueError("CameraSpec.camera_type must be non-empty.")
        if not self.serial_number:
            raise ValueError("CameraSpec.serial_number must be non-empty.")


@dataclass(frozen=True)
class EndEffectorSpec:
    """Physical end-effector placement and connection settings."""

    kind: str
    connection: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmSpec:
    """One named arm in a physical robot layout."""

    name: str
    driver: str
    node_rank: int
    connection: dict[str, Any] = field(default_factory=dict)
    end_effector: Optional[EndEffectorSpec] = None
    cameras: tuple[CameraSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ArmSpec.name must be non-empty.")
        if not self.driver:
            raise ValueError("ArmSpec.driver must be non-empty.")
        camera_names = [camera.name for camera in self.cameras]
        if len(camera_names) != len(set(camera_names)):
            raise ValueError(f"Arm {self.name!r} contains duplicate camera names.")


@dataclass(frozen=True)
class PartSpec:
    """Additional physical component in a robot layout."""

    name: str
    kind: str
    driver: str
    node_rank: int
    connection: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.kind or not self.driver:
            raise ValueError("PartSpec name, kind, and driver must be non-empty.")


@dataclass(frozen=True)
class RobotSpec:
    """Scheduler-independent physical layout produced from a robot config."""

    robot_type: str
    node_rank: int
    arms: tuple[ArmSpec, ...]
    cameras: tuple[CameraSpec, ...] = ()
    parts: tuple[PartSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.robot_type:
            raise ValueError("RobotSpec.robot_type must be non-empty.")
        self._require_unique_names("arm", [arm.name for arm in self.arms])
        self._require_unique_names("camera", [camera.name for camera in self.cameras])
        self._require_unique_names("part", [part.name for part in self.parts])

    def arm(self, name: str) -> ArmSpec:
        """Return a named arm or raise a clear configuration error."""
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(f"Robot {self.robot_type!r} has no arm named {name!r}.")

    @staticmethod
    def _require_unique_names(kind: str, names: list[str]) -> None:
        if len(names) != len(set(names)):
            raise ValueError(f"RobotSpec contains duplicate {kind} names.")
