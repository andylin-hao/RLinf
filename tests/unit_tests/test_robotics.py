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

from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import pytest

from rlinf.robotics import (
    ControllablePart,
    PartRuntime,
    Robot,
    RobotAutoConfig,
    RobotConfig,
    RobotPart,
)
from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareResource,
    NodeHardwareConfig,
)


class FakePart(RobotPart):
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def observation_features(self) -> dict[str, dict]:
        return {"state": {"shape": (1,)}}

    def connect(self) -> None:
        self.events.append(f"connect:{self.name}")
        self._connected = True

    def get_observation(self) -> dict[str, np.ndarray]:
        return {"state": np.array([1.0])}

    def disconnect(self) -> None:
        self.events.append(f"disconnect:{self.name}")
        self._connected = False


class FakeControllablePart(FakePart, ControllablePart):
    @property
    def action_features(self) -> dict[str, dict]:
        return {"target": {"shape": (1,)}}

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return action


def test_robot_composes_and_namespaces_parts():
    events: list[str] = []
    arm = FakeControllablePart("arm", events)
    camera = FakePart("camera", events)
    robot = Robot({"left_arm": arm, "front_camera": camera})

    robot.connect()

    assert robot.is_connected
    assert events == ["connect:arm", "connect:camera"]
    assert set(robot.observation_features) == {"left_arm", "front_camera"}
    assert set(robot.action_features) == {"left_arm"}
    action = {"left_arm": {"target": np.array([0.5])}}
    assert robot.send_action(action) == action

    robot.disconnect()
    assert events[-2:] == ["disconnect:camera", "disconnect:arm"]


def test_robot_rejects_actions_for_observation_only_parts():
    robot = Robot({"camera": FakePart("camera", [])})

    with pytest.raises(TypeError, match="not controllable"):
        robot.send_action({"camera": {"target": np.array([0.5])}})


def test_robot_requires_non_empty_string_part_names():
    with pytest.raises(ValueError, match="non-empty strings"):
        Robot({0: FakePart("camera", [])})  # type: ignore[dict-item]


def test_register_robot_registers_policy_and_config(monkeypatch):
    monkeypatch.setattr(Robot, "registry", Robot.registry.copy())
    monkeypatch.setattr(Hardware, "hw_types", Hardware.hw_types.copy())
    monkeypatch.setattr(Hardware, "policy_registry", Hardware.policy_registry.copy())
    monkeypatch.setattr(
        NodeHardwareConfig,
        "_hardware_config_registry",
        NodeHardwareConfig._hardware_config_registry.copy(),
    )

    @dataclass
    class TestRobotConfig(RobotConfig):
        endpoint: str = "loopback"

    class TestRobot(Robot):
        HW_TYPE = "TestRobot"

        @classmethod
        def enumerate(
            cls,
            node_rank: int,
            configs: Optional[list[HardwareConfig]] = None,
        ) -> Optional[HardwareResource]:
            return None

    registered = Robot.register_robot(TestRobotConfig)(TestRobot)
    parsed = NodeHardwareConfig(
        type="TestRobot",
        configs=cast(Any, [{"node_rank": 3, "endpoint": "robot.local"}]),
    )

    assert registered is TestRobot
    assert Robot.registry["TestRobot"] is TestRobot
    assert TestRobot in Hardware.policy_registry
    assert isinstance(parsed.configs[0], TestRobotConfig)
    assert parsed.configs[0].endpoint == "robot.local"

    with pytest.raises(ValueError, match="already registered"):
        Robot.register_robot(TestRobotConfig)(TestRobot)
    assert NodeHardwareConfig._hardware_config_registry["TestRobot"] is TestRobotConfig


def test_scheduler_does_not_export_concrete_robot_types():
    import rlinf.scheduler as scheduler

    assert not hasattr(scheduler, "FrankaConfig")
    assert not hasattr(scheduler, "FrankaHWInfo")


def test_robot_auto_config_supports_pep604_optional(monkeypatch):
    @dataclass
    class TestConfig(RobotConfig):
        port: int | None = None

    config = TestConfig(node_rank=0)
    monkeypatch.setenv("PORT", "5000")

    assert RobotAutoConfig.resolve([config])[0].port == 5000


def test_part_runtime_hosts_controllable_part():
    runtime = PartRuntime.__new__(PartRuntime)
    runtime._part_cls = FakeControllablePart
    runtime._part_kwargs = {"name": "arm", "events": []}
    runtime._part = None

    runtime.initialize()

    assert runtime.is_connected()
    action = {"target": np.array([0.5])}
    assert runtime.send_action(action) == action
    runtime.shutdown()
    assert runtime._part is None
