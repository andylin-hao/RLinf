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

"""Cameras: the category, and the drivers that implement it.

:class:`Camera` is what a robot composes and what a policy reads.
:class:`BaseCamera` is what a driver subclasses -- it owns the capture thread,
so an implementation supplies :meth:`~BaseCamera._open`,
:meth:`~BaseCamera._read_frame` and :meth:`~BaseCamera._release` and nothing
else.

Each driver registers the names a config selects it by, so building one from a
:class:`CameraInfo` needs no table here::

    camera = Camera.of(info, node_rank=2)

Importing this module registers every driver and imports no vendor SDK: each
reaches for pyrealsense2, pyzed, or cv2 inside ``_open``, on the machine the
camera is actually plugged into.
"""

from .base import BaseCamera, Camera, CameraInfo

# Imported for the side effect of registering: a driver announces itself, and
# nothing here lists them. Adding a camera is one file plus one decorator.
from .lumos import LumosCamera
from .realsense import RealSenseCamera
from .zed import ZEDCamera

__all__ = [
    "BaseCamera",
    "Camera",
    "CameraInfo",
    "LumosCamera",
    "RealSenseCamera",
    "ZEDCamera",
]
