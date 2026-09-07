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

"""Camera toolkit coverage and resource cleanup with a small SDK stand-in."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "toolkits/realworld_check/test_franka_camera.py"
)


def camera_sdk(monkeypatch, serials, *, fail_read=False):
    events = []

    class Config:
        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, *args):
            pass

    class Pipeline:
        def start(self, config):
            self.serial = config.serial
            events.append((self.serial, "start"))

        def wait_for_frames(self):
            if fail_read:
                raise RuntimeError("frame timeout")
            events.append((self.serial, "frame"))

        def stop(self):
            events.append((self.serial, "stop"))

    devices = [
        SimpleNamespace(get_info=lambda key, serial=serial: serial)
        for serial in serials
    ]
    sdk = SimpleNamespace(
        context=lambda: SimpleNamespace(devices=devices),
        camera_info=SimpleNamespace(serial_number="serial"),
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
        config=Config,
        pipeline=Pipeline,
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", sdk)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return events


def test_camera_tool_reads_every_camera_and_releases_each(monkeypatch):
    events = camera_sdk(monkeypatch, ["scene", "wrist"])

    runpy.run_path(str(SCRIPT), run_name="__main__")

    for serial in ("scene", "wrist"):
        assert events.count((serial, "frame")) == 20
        assert events.count((serial, "stop")) == 1
    assert events.index(("scene", "stop")) < events.index(("wrist", "start"))


def test_camera_tool_releases_stream_when_a_read_fails(monkeypatch):
    events = camera_sdk(monkeypatch, ["wrist"], fail_read=True)

    with pytest.raises(RuntimeError, match="frame timeout"):
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert events[-1] == ("wrist", "stop")


def test_camera_tool_reports_no_devices(monkeypatch):
    events = camera_sdk(monkeypatch, [])

    with pytest.raises(RuntimeError, match="No RealSense cameras"):
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert events == []
