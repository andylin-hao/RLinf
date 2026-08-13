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

import pytest

from rlinf.scheduler.hardware.robots.franka import FrankaConfig, FrankaRobot


class _RuntimeCLI:
    def __init__(self, tool: str):
        self.tool = tool

    @staticmethod
    def is_enabled(tool: str) -> bool:
        return tool in ("camctr", "rosctr")

    def list_robots(self):
        return [
            {
                "robotId": "franka-0",
                "params": {"robot_ip": "172.16.0.2"},
            }
        ]

    def list_cameras(self):
        return [
            {
                "cameraId": "video0",
                "serialNumber": "cam-123",
            }
        ]

    def resolve_robot_id(self, robot_ip):
        assert robot_ip == "172.16.0.2"
        return "franka-0"

    def resolve_camera_id(self, identifier):
        assert identifier == "cam-123"
        return "video0"


def test_franka_hardware_is_discovered_from_runtime(monkeypatch):
    monkeypatch.delenv("ROBOT_IP", raising=False)
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI",
        _RuntimeCLI,
    )

    resource = FrankaRobot.enumerate(node_rank=0)

    assert resource is not None
    assert resource.count == 1
    config = resource.infos[0].config
    assert config.robot_ip == "172.16.0.2"
    assert config.embodied_runtime_robot_id == "franka-0"
    assert config.camera_serials == ["cam-123"]


def test_franka_hardware_uses_explicit_runtime_robot_id(monkeypatch):
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI",
        _RuntimeCLI,
    )
    config = FrankaConfig(
        node_rank=0,
        embodied_runtime_robot_id="franka-0",
        camera_serials=["cam-123"],
    )

    resource = FrankaRobot.enumerate(node_rank=0, configs=[config])

    assert resource is not None
    assert resource.infos[0].config.robot_ip == "172.16.0.2"


def test_franka_hardware_rejects_empty_runtime_inventory(monkeypatch):
    monkeypatch.setattr(_RuntimeCLI, "list_robots", lambda _: [])
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI",
        _RuntimeCLI,
    )

    with pytest.raises(ValueError, match="No Franka robots"):
        FrankaRobot.enumerate(node_rank=0)


def test_franka_hardware_is_absent_without_config_or_runtime(monkeypatch):
    monkeypatch.delenv("ROBOT_IP", raising=False)
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI.is_enabled",
        lambda _: False,
    )

    assert FrankaRobot.enumerate(node_rank=0, configs=[]) is None


def test_franka_hardware_requires_runtime_robot_ip(monkeypatch):
    monkeypatch.setattr(
        _RuntimeCLI,
        "list_robots",
        lambda _: [{"robotId": "franka-0", "params": {}}],
    )
    monkeypatch.setattr(
        "rlinf.scheduler.hardware.robots.franka.EmbodiedRuntimeCLI",
        _RuntimeCLI,
    )

    with pytest.raises(ValueError, match="params.robot_ip"):
        FrankaRobot.enumerate(node_rank=0)
