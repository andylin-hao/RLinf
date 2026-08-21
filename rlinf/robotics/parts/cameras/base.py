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

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from rlinf.robotics.parts.base import RobotPart
from rlinf.utils.logging import get_logger

_logger = get_logger()


class Camera(RobotPart):
    """A part that is read and never commanded.

    The category, with no lifecycle of its own -- :class:`BaseCamera` below is
    what a driver subclasses. Naming it separately is what lets a robot ask for
    every camera it carries, and what lets a camera hosted on another machine
    come back as a camera rather than as a part in general.
    """


@dataclass
class CameraInfo:
    """Descriptor for a single camera device."""

    name: str
    serial_number: str
    camera_type: str = "realsense"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    enable_depth: bool = False
    crop_region: Optional[tuple[float, float, float, float]] = None


class BaseCamera(Camera, ABC):
    """Abstract base class for threaded camera capture.

    A camera is a part like any other: :meth:`_open` reaches the hardware,
    :meth:`_read_frame` reads one frame from it, and :meth:`_release` lets it go.
    The capture thread and its queue are handled here, started and stopped
    around those by :meth:`connect` and :meth:`disconnect`.
    """

    def __init__(self, camera_info: CameraInfo):
        self._camera_info = camera_info
        self._frame_queue: queue.Queue = queue.Queue()
        self._frame_capturing_thread: Optional[threading.Thread] = None
        self._frame_capturing_start = False

    @classmethod
    def of(cls, camera_info: CameraInfo, **placement: Any) -> "BaseCamera":
        """Build the camera the descriptor names, without opening it.

        The backend is chosen by :attr:`CameraInfo.camera_type`, so a config
        naming ``"zed"`` reaches the ZED driver without the caller importing
        it. Pass ``node_rank`` to say which machine the camera is plugged
        into; the robot opens it there.
        """
        return cls.backend(camera_info.camera_type)(camera_info, **placement)

    @classmethod
    def declare(
        cls,
        cameras: "Optional[Mapping[str, CameraInfo]]" = None,
        *,
        node_rank: Optional[int] = None,
    ) -> dict[str, "BaseCamera"]:
        """Build one camera per descriptor, ready to compose into a robot.

        The names are the public paths the robot will publish, so this returns
        exactly what a builder splats into its composition::

            return cls(**cls.declare(cameras, node_rank=camera_node_rank))

        Every camera lands on ``node_rank`` -- the machine they are plugged
        into -- and none is opened until the robot connects.
        """
        return {
            name: cls.of(info, node_rank=node_rank)
            for name, info in (cameras or {}).items()
        }

    @property
    def name(self) -> str:
        return self._camera_info.name

    @property
    def camera_info(self) -> CameraInfo:
        """Return the immutable camera connection descriptor."""
        return self._camera_info

    @property
    def is_connected(self) -> bool:
        """Whether the camera is opened and its capture thread is running."""
        return self._device is not None

    @property
    def observation_features(self) -> dict:
        """Describe the raw BGR frame returned by this camera."""
        width, height = self._camera_info.resolution
        channels = 4 if self._camera_info.enable_depth else 3
        return {
            "frame": {
                "shape": (height, width, channels),
                "dtype": "uint16" if self._camera_info.enable_depth else "uint8",
            }
        }

    def _opened(self) -> None:
        """Start capturing frames from the camera the driver just opened.

        The queue and the thread are made here rather than in ``__init__``: a
        thread runs once and cannot be restarted, so a camera built once and
        connected twice -- which is what recovering from a stall does -- would
        raise ``RuntimeError`` on the second connect. A fresh queue also drops
        the frames buffered before the stall, which are the stale ones.

        This runs beside the camera, so a camera placed on another node runs
        its capture loop there and this process only asks for the last frame.
        """
        if self._frame_capturing_start:
            return
        self._frame_queue = queue.Queue()
        self._frame_capturing_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self._frame_capturing_start = True
        self._frame_capturing_thread.start()

    def _closing(self) -> None:
        """Stop capturing, while the camera is still open to be read."""
        self._frame_capturing_start = False
        thread = self._frame_capturing_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._frame_capturing_thread = None

    def reopen(self) -> None:
        """Close this camera and open it again.

        What a stalled camera needs. It is a method on the camera rather than
        a disconnect/connect pair at the call site because a placed camera is
        reached through a proxy, where those two are no-ops and the reopen has
        to happen on the node holding the device.
        """
        self.disconnect()
        self.connect()

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return the latest raw frame under the canonical camera key."""
        return {"frame": self.get_frame()}

    def get_frame(self, timeout: float = 5) -> np.ndarray:
        """Return the most recent frame (blocks up to *timeout* seconds).

        Args:
            timeout: Maximum seconds to wait for a frame.
        """
        assert self._frame_capturing_start, (
            "Frame capturing is not started. Call connect() first."
        )
        return self._frame_queue.get(timeout=timeout)

    # ── internal ──────────────────────────────────────────────────────

    def _capture_frames(self):
        while self._frame_capturing_start:
            time.sleep(1 / self._camera_info.fps)
            try:
                has_frame, frame = self._read_frame()
            except Exception as e:
                _logger.error(
                    "[%s] _read_frame raised %s: %s — exiting capture thread.",
                    self._camera_info.name,
                    type(e).__name__,
                    e,
                )
                break
            if not has_frame:
                _logger.error(
                    "[%s] _read_frame returned (False, None) — exiting capture thread.",
                    self._camera_info.name,
                )
                break
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)

    @abstractmethod
    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera hardware.

        Returns:
            ``(success, frame)`` where *frame* is a BGR ``uint8`` numpy array,
            or ``(False, None)`` on failure.
        """
        raise NotImplementedError

    @abstractmethod
    def _release(self, device: Any) -> None:
        """Release hardware-specific resources (pipeline, SDK handle, …)."""
        raise NotImplementedError
