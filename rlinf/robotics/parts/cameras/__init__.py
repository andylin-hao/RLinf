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

from dataclasses import dataclass
from typing import Any, Optional

from ...specs import PartConfig
from .base import BaseCamera, CameraInfo
from .realsense import RealSenseCamera

__all__ = [
    "BaseCamera",
    "CameraConfig",
    "CameraInfo",
    "RealSenseCamera",
    "camera_cls",
    "create_camera",
]


def camera_cls(camera_type: str) -> type[BaseCamera]:
    """Return the camera class for a backend, imported lazily.

    Supported ``camera_type`` values:

    * ``"realsense"`` / ``"rs"`` -- Intel RealSense (requires ``pyrealsense2``)
    * ``"zed"`` -- Stereolabs ZED (requires the ZED SDK / ``pyzed``)
    * ``"lumos"`` -- LUMOS V4L2 USB camera (requires ``opencv-python``)
    """
    kind = camera_type.lower()
    if kind == "zed":
        from .zed import ZEDCamera

        return ZEDCamera
    if kind in ("realsense", "rs"):
        return RealSenseCamera
    if kind == "lumos":
        from .lumos import LumosCamera

        return LumosCamera
    raise ValueError(
        f"Unsupported camera_type={camera_type!r}. "
        "Supported types: 'realsense', 'zed', 'lumos'."
    )


@dataclass
class CameraConfig(PartConfig):
    """One camera and the node it is plugged into.

    Declaring a camera is what makes it placeable: it can run on the machine
    holding the USB or GigE link while the policy runs elsewhere.
    """

    info: Optional[CameraInfo] = None

    def part_cls(self) -> type:
        """Return the backend class named by the camera info."""
        if self.info is None:
            raise ValueError("CameraConfig needs a CameraInfo.")
        return camera_cls(self.info.camera_type)

    def part_args(self) -> tuple[Any, ...]:
        """The camera constructor takes its descriptor."""
        return (self.info,)


def create_camera(camera_info: CameraInfo) -> BaseCamera:
    """Build a camera of the backend named by *camera_info*, in this process.

    Prefer :class:`CameraConfig` inside a robot, which declares the camera and
    lets :meth:`Robot.connect` place it on the node it is plugged into.
    """
    return camera_cls(camera_info.camera_type)(camera_info)
