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

from unittest.mock import MagicMock

import numpy as np

import rlinf.envs.realworld.common.camera as camera_module
from rlinf.envs.realworld.common.camera import CameraInfo
from rlinf.envs.realworld.common.camera.embodied_runtime_camera import (
    EmbodiedRuntimeCamera,
)


def test_runtime_camera_opens_and_decodes_watch_stream(monkeypatch):
    jpeg = b"\xff\xd8payload\xff\xd9"
    cli = MagicMock()
    cli.executable = "/opt/rlinf/bin/camctr"
    cli.resolve_camera_id.return_value = "video0"
    cli.run_json.side_effect = [
        {"camera": {"state": "closed"}},
        {"encoding": "jpeg"},
    ]
    process = MagicMock()
    process.stdout.fileno.return_value = 10

    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.EmbodiedRuntimeCLI",
        lambda _: cli,
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.os.read",
        lambda fd, size: jpeg,
    )
    decoded = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr("cv2.imdecode", lambda data, flags: decoded)
    camera = EmbodiedRuntimeCamera(
        CameraInfo(name="wrist", serial_number="cam-123", fps=15)
    )
    monkeypatch.setattr(camera, "_capture_frames", lambda: None)

    camera.open()
    ok, frame = camera._read_frame()
    camera.close()

    assert ok
    assert frame is decoded
    cli.run_json.assert_called_with(
        "open",
        "video0",
        "--width",
        "640",
        "--height",
        "480",
        "--fps",
        "15",
        "--encoding",
        "jpeg",
    )
    cli.run.assert_called_once_with("close", "video0")


def test_runtime_camera_rejects_incompatible_existing_stream(monkeypatch):
    cli = MagicMock()
    cli.resolve_camera_id.return_value = "video0"
    cli.run_json.side_effect = [
        {"camera": {"state": "closed"}},
        {"encoding": "h264"},
    ]
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.EmbodiedRuntimeCLI",
        lambda _: cli,
    )
    camera = EmbodiedRuntimeCamera(CameraInfo(name="wrist", serial_number="cam-123"))

    try:
        camera.open()
    except RuntimeError as exc:
        assert "incompatible encoding" in str(exc)
    else:
        raise AssertionError("Expected incompatible stream to fail")
    cli.run.assert_called_once_with("close", "video0")


def test_camera_factory_prefers_enabled_runtime(monkeypatch):
    runtime_camera = object()
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.EmbodiedRuntimeCLI.is_enabled",
        lambda _: True,
    )
    constructor = MagicMock(return_value=runtime_camera)
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.EmbodiedRuntimeCamera", constructor
    )
    info = CameraInfo(name="wrist", serial_number="cam-123")

    camera = camera_module.create_camera(info)

    assert camera is runtime_camera
    constructor.assert_called_once_with(info)


def test_runtime_camera_handles_jpeg_marker_bytes_in_scan_data(monkeypatch):
    cli = MagicMock()
    cli.resolve_camera_id.return_value = "video0"
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.EmbodiedRuntimeCLI",
        lambda _: cli,
    )
    camera = EmbodiedRuntimeCamera(CameraInfo(name="wrist", serial_number="cam-123"))
    camera._buffer.extend(
        b"\xff\xd8\xff\xda\x00\x08\x00\x00\x00\x00\x00\x00abc\xff\x00\xd9def\xff\xd9"
    )

    assert camera._find_jpeg_end() == len(camera._buffer)


def test_runtime_camera_closes_existing_runtime_state(monkeypatch):
    cli = MagicMock()
    cli.executable = "/opt/rlinf/bin/camctr"
    cli.resolve_camera_id.return_value = "video0"
    cli.run_json.side_effect = [
        {"camera": {"state": "error"}},
        {"encoding": "jpeg"},
    ]
    process = MagicMock()
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.EmbodiedRuntimeCLI",
        lambda _: cli,
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.camera.embodied_runtime_camera.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    camera = EmbodiedRuntimeCamera(CameraInfo(name="wrist", serial_number="cam-123"))
    monkeypatch.setattr(camera, "_capture_frames", lambda: None)

    camera.open()

    cli.run.assert_called_once_with("close", "video0")
    camera.close()
