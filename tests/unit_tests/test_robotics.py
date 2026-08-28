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

"""Tests for robot parts, composition, placement, and layer boundaries."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pytest

import rlinf.robotics.robots.franka as franka_module
from rlinf.robotics import (
    Arm,
    Camera,
    Connection,
    ControllablePart,
    DOSW1Robot,
    DOSW1RobotConfig,
    DualFrankaRobot,
    EndEffector,
    FrankaRobot,
    GimArmConfig,
    LegacyObservationAdapter,
    MethodArm,
    MethodEndEffector,
    PartGroup,
    Robot,
    RobotAutoConfig,
    RobotConfig,
    RobotDiscovery,
    RobotPart,
    Turtle2Config,
    VectorActionAdapter,
    VectorActionBinding,
    register_robot,
)
from rlinf.robotics.parts.arms import (
    FrankaROSArm,
    FrankyArm,
    GimArm,
    Turtle2Connection,
)
from rlinf.robotics.parts.arms.franka import FrankaRobotState
from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareResource,
    NodeHardwareConfig,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class FakePart(RobotPart):
    """Record lifecycle calls through the standard driver hooks."""

    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    @property
    def observation_features(self) -> dict[str, dict]:
        return {"state": {"shape": (1,)}}

    def _open(self) -> Any:
        self.events.append(f"connect:{self.name}")
        return f"device:{self.name}"

    def get_observation(self) -> dict[str, np.ndarray]:
        return {"state": np.array([1.0])}

    def _release(self, device: Any) -> None:
        self.events.append(f"disconnect:{self.name}")


class FakeControllablePart(FakePart, ControllablePart):
    @property
    def action_features(self) -> dict[str, dict]:
        return {"target": {"shape": (1,)}}

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return action


class FakeEndEffector(FakePart, EndEffector):
    """Minimal end effector with the shared category interface."""

    state_dim = 1
    action_dim = 1
    control_mode = "binary"

    def get_state(self) -> np.ndarray:
        return np.array([1.0])

    def command(self, action: np.ndarray) -> bool:
        return True


class FakeCamera(FakePart, Camera):
    pass


class FakeRemoteResult:
    def __init__(self, value: Any):
        self.value = value

    def wait(self) -> list[Any]:
        return [self.value]


class FakeMethodDriver:
    """Expose arm and gripper operations as named methods."""

    def __init__(self):
        self.state = FrankaRobotState(gripper_position=0, gripper_open=False)
        self.calls: list[tuple[str, Any]] = []
        self.is_connected = True

    def get_state(self) -> FrankaRobotState:
        return self.state

    def move_arm(self, target: np.ndarray) -> None:
        self.calls.append(("move_arm", target))

    def open_gripper(self) -> None:
        self.calls.append(("open_gripper", None))

    def close_gripper(self) -> None:
        self.calls.append(("close_gripper", None))

    def is_robot_up(self) -> bool:
        """Return a driver-specific status used by real environments."""
        return True


class FakeWorkerGroup:
    """Record calls forwarded to a single hosted connection."""

    def __init__(self, values: Optional[dict[str, Any]] = None):
        self.calls: list[tuple[str, Any]] = []
        self.values = values or {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args: Any, **kwargs: Any) -> FakeRemoteResult:
            self.calls.append((name, args[0] if len(args) == 1 else args or None))
            if name == "attribute":
                return FakeRemoteResult(self.values.get(args[0]))
            return FakeRemoteResult(self.values.get(name))

        return call

    def _close(self) -> None:
        self.calls.append(("_close", None))


def test_robot_composes_and_namespaces_parts():
    events: list[str] = []
    arm = PartGroup(
        arm=FakeControllablePart("arm", events),
        gripper=FakeEndEffector("gripper", events),
        wrist=FakeCamera("wrist", events),
    )
    robot = Robot(left=arm, front=FakeCamera("front", events))

    robot.connect()

    assert robot.is_connected
    assert events == [
        "connect:arm",
        "connect:gripper",
        "connect:wrist",
        "connect:front",
    ]
    # Observation and action namespaces follow the composed part names.
    assert set(robot.observation_features) == {"left", "front"}
    assert set(robot.action_features) == {"left"}, "a camera takes no action"
    action = {
        "left": {
            "arm": {"target": np.array([0.5])},
            "gripper": {"target": np.array([1.0])},
        }
    }
    assert robot.send_action(action) == action
    assert set(robot.parts_of_type(Camera)) == {"left.wrist", "front"}

    robot.disconnect()
    assert events[-4:] == [
        "disconnect:front",
        "disconnect:wrist",
        "disconnect:gripper",
        "disconnect:arm",
    ]


def test_robot_rejects_actions_for_observation_only_parts():
    robot = Robot(camera=FakeCamera("camera", []))

    with pytest.raises(TypeError, match="not controllable"):
        robot.send_action({"camera": {"target": np.array([0.5])}})


def test_robot_disconnects_remaining_arm_parts_after_camera_failure():
    events: list[str] = []
    camera = FakeCamera("wrist", events)
    arm = PartGroup(arm=FakeControllablePart("driver", events), wrist=camera)
    robot = Robot(arm=arm)
    robot.connect()
    camera.disconnect()

    robot.disconnect()

    assert "disconnect:driver" in events


def test_driver_views_expose_composed_part_api():
    driver = FakeMethodDriver()
    arm = MethodArm(
        driver,
        commands={"tcp_pose": "move_arm"},
        state_fields=("tcp_pose", "arm_joint_position"),
    )
    end_effector = MethodEndEffector(driver, state_field="gripper_position")
    target = np.ones(7)

    assert set(arm.get_observation()) == {"tcp_pose", "arm_joint_position"}
    assert end_effector.get_observation()["state"].tolist() == [0]
    assert arm.send_action({"tcp_pose": target})["tcp_pose"] is target
    end_effector.send_action({"target": np.array([1.0])})
    end_effector.send_action({"target": np.array([-1.0])})

    assert [name for name, _ in driver.calls] == [
        "move_arm",
        "open_gripper",
        "close_gripper",
    ]


def test_letting_a_placed_connection_go_closes_it_before_killing_it():
    from rlinf.robotics.placement import shutdown

    group = FakeWorkerGroup()
    shutdown(group)

    assert [name for name, _ in group.calls] == ["disconnect", "_close"]


def test_a_placed_connection_forwards_off_interface_driver_methods():
    from rlinf.robotics.placement import remote_view_of

    view = remote_view_of(FakeMethodDriver)
    placed = object.__new__(view)
    placed._group = FakeWorkerGroup({"is_robot_up": True})

    assert placed.is_robot_up() is True


def test_robot_requires_non_empty_string_part_names():
    with pytest.raises(ValueError, match="non-empty strings"):
        Robot(parts={0: FakePart("camera", [])})  # type: ignore[dict-item]


def test_builtin_robots_expose_standard_composition_layouts():
    events: list[str] = []
    left_arm = PartGroup(
        arm=FakeControllablePart("left_arm", events),
        gripper=FakeEndEffector("left_gripper", events),
    )
    right_arm = PartGroup(
        arm=FakeControllablePart("right_arm", events),
        gripper=FakeEndEffector("right_gripper", events),
    )
    third_arm = PartGroup(arm=FakeControllablePart("third_arm", events))

    single = FrankaRobot(arm=left_arm, front_camera=FakeCamera("front", events))
    dual = DualFrankaRobot(
        left=left_arm, right=right_arm, base_camera=FakeCamera("base", events)
    )
    # Composition supports arbitrary names and part counts.
    triple = FrankaRobot(left=left_arm, right=right_arm, third=third_arm)

    assert set(single.children) == {"arm", "front_camera"}
    assert set(single.parts_of_type(PartGroup)) == {"arm"}
    assert set(single.parts_of_type(EndEffector)) == {"arm.gripper"}
    assert set(single.parts_of_type(Camera)) == {"front_camera"}
    assert set(dual.children) == {"left", "right", "base_camera"}
    assert set(dual.parts_of_type(PartGroup)) == {"left", "right"}
    assert set(triple.children) == {"left", "right", "third"}
    assert set(triple.parts_of_type(PartGroup)) == {"left", "right", "third"}


def test_register_robot_registers_policy_and_config(monkeypatch):
    monkeypatch.setattr(RobotDiscovery, "registry", RobotDiscovery.registry.copy())
    monkeypatch.setattr(Hardware, "hw_types", Hardware.hw_types.copy())
    monkeypatch.setattr(Hardware, "policy_registry", Hardware.policy_registry.copy())
    monkeypatch.setattr(
        NodeHardwareConfig,
        "_hardware_config_registry",
        NodeHardwareConfig._hardware_config_registry.copy(),
    )

    @dataclass
    class TestRobotConfig(RobotConfig):
        connection: str = "loopback"

    class TestRobot(Robot):
        ROBOT_TYPE = "TestRobot"

    class TestRobotDiscovery(RobotDiscovery):
        HW_TYPE = "TestRobot"

        @classmethod
        def enumerate(
            cls,
            node_rank: int,
            configs: Optional[list[HardwareConfig]] = None,
        ) -> Optional[HardwareResource]:
            return None

    registered = register_robot(TestRobotConfig, TestRobot)(TestRobotDiscovery)
    parsed = NodeHardwareConfig(
        type="TestRobot",
        configs=cast(Any, [{"node_rank": 3, "connection": "robot.local"}]),
    )

    assert registered is TestRobotDiscovery
    assert RobotDiscovery.registry["TestRobot"].robot_cls is TestRobot
    assert TestRobotDiscovery in Hardware.policy_registry
    assert isinstance(parsed.configs[0], TestRobotConfig)
    assert parsed.configs[0].connection == "robot.local"

    with pytest.raises(ValueError, match="already registered"):
        register_robot(TestRobotConfig, TestRobot)(TestRobotDiscovery)
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


def test_namespaces_follow_the_composition():
    robot = Robot(arm=PartGroup(arm=FakeControllablePart("arm", [])))
    robot.connect()

    observation = robot.get_observation()
    action = {"arm": {"arm": {"target": np.array([0.25])}}}

    assert observation["arm"]["arm"]["state"].shape == (1,)
    assert robot.send_action(action) == action
    robot.disconnect()


def test_a_connection_hands_out_the_part_it_backs_not_a_controllable_one():
    class CameraOnlyHost(FakePart):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"wrist": FakeCamera("wrist", [])}

    wrist = CameraOnlyHost("host", []).part("wrist")

    assert isinstance(wrist, Camera)
    assert not isinstance(wrist, ControllablePart)
    with pytest.raises(TypeError, match="not controllable"):
        Robot(wrist=wrist).send_action({"wrist": {}})


def test_legacy_adapters_preserve_policy_facing_layouts():
    canonical_observation = {
        "arms": {
            "left": {"state": {"joint_position": np.arange(6)}},
            "right": {"state": {"joint_position": np.arange(6, 12)}},
        },
        "cameras": {
            "front": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        },
    }
    observation_adapter = LegacyObservationAdapter(
        state_fields={
            "left_joint_position": ("arms", "left", "state", "joint_position"),
            "right_joint_position": (
                "arms",
                "right",
                "state",
                "joint_position",
            ),
        },
        frame_fields={"front": ("cameras", "front", "rgb")},
    )
    action_adapter = VectorActionAdapter(
        action_dim=12,
        bindings=[
            VectorActionBinding(("arms", "left", "arm", "target"), 0, 6),
            VectorActionBinding(("arms", "right", "arm", "target"), 6, 12),
        ],
    )

    legacy_observation = observation_adapter.adapt(canonical_observation)
    canonical_action = action_adapter.adapt(np.arange(12))

    assert set(legacy_observation) == {"state", "frames"}
    assert legacy_observation["state"]["left_joint_position"].tolist() == list(range(6))
    assert canonical_action["arms"]["right"]["arm"]["target"].tolist() == list(
        range(6, 12)
    )


def test_all_builtin_configs_construct_from_a_node_rank_alone():
    configs = [
        GimArmConfig(node_rank=0),
        DOSW1RobotConfig(node_rank=0),
        Turtle2Config(node_rank=0),
    ]

    assert all(config.node_rank == 0 for config in configs)


def test_every_registered_robot_can_skip_the_enumeration_probe():
    """Validation is uniform across robots, so the opt-out must be too.

    Enumeration checks the cameras a config names, which needs the camera SDK
    on the enumerating node. Every robot config has to offer the same way out,
    or a node without that SDK cannot declare the robot at all.
    """
    registry = RobotDiscovery.registry

    without = sorted(
        name
        for name, reg in registry.items()
        if "disable_validate" not in {f.name for f in fields(reg.config_cls)}
    )
    assert without == []

    # The flag has to reach enumeration, not merely exist on the config.
    discovery = registry["DOSW1"].discovery_cls
    probed = []

    class Probing(discovery):
        @classmethod
        def validate(cls, config, node_rank):
            probed.append(config)

    Probing.enumerate(0, [DOSW1RobotConfig(node_rank=0, camera_serials=["cam"])])
    assert len(probed) == 1

    Probing.enumerate(
        0,
        [DOSW1RobotConfig(node_rank=0, camera_serials=["cam"], disable_validate=True)],
    )
    assert len(probed) == 1


def test_every_registered_robot_carries_a_builder():
    registry = RobotDiscovery.registry

    assert set(registry) >= {"Franka", "DualFranka", "GimArm", "Turtle2", "DOSW1"}
    missing = sorted(name for name, reg in registry.items() if reg.build is None)
    assert missing == []


def test_dosw1_dummy_runtime_uses_composed_dual_arm_interface():
    robot = DOSW1Robot.build(is_dummy=True)
    robot.connect()

    assert set(robot.children) == {"left", "right"}
    observation = robot.get_observation()
    assert observation["left"]["arm"]["joint_position"].shape == (6,)
    robot.disconnect()
    assert not robot.is_connected


def test_pure_drivers_construct_without_scheduler_or_vendor_sdks():
    from rlinf.robotics.parts.base import Connection

    # Single-arm connections are themselves controllable parts.
    arms = [
        FrankaROSArm("10.0.0.1"),
        FrankyArm("10.0.0.1"),
        GimArm("can0", "gim_arm_xl", True, "parallel"),
    ]
    # Multi-part buses are connections, not parts.
    buses = [Turtle2Connection()]

    assert all(isinstance(driver, ControllablePart) for driver in arms)
    # Both forms participate in the connection lifecycle; only arms are readable.
    assert all(isinstance(driver, Connection) for driver in arms + buses)
    assert all(isinstance(driver, RobotPart) for driver in arms)
    assert not any(isinstance(driver, RobotPart) for driver in buses)
    assert all(not driver.is_connected for driver in arms + buses)
    # A connection exports the parts it genuinely backs. A Franka arm backs
    # none: its end effector answers on its own endpoint and is composed
    # beside it. A GimArm gripper shares the arm's bus, so the arm exports it.
    assert not FrankaROSArm("10.0.0.1").parts
    assert not FrankyArm("10.0.0.1").parts
    assert all(driver.parts for driver in buses)
    assert GimArm("can0", "gim_arm_xl", True, "parallel").parts


class _BareArm(Arm):
    """Minimal arm used by placement and backend-selection tests."""

    @classmethod
    def declare(cls, address, **settings):
        """Declare the arm while preserving placement settings."""
        placement = {
            name: settings.pop(name)
            for name in ("node_rank", "worker_name")
            if name in settings
        }
        return cls(address, **placement)

    @property
    def observation_features(self) -> dict:
        return {}

    @property
    def action_features(self) -> dict:
        return {}

    def get_observation(self) -> dict:
        return {}

    def send_action(self, action):
        return action


def _opens_here(monkeypatch):
    """Force declared connections to open in the current process."""
    from dataclasses import replace

    from rlinf.robotics.parts.base import Connection

    connect = Connection.connect

    def connect_here(self):
        if self._remote_info is not None and self._remote_info.node_rank is not None:
            self._remote_info = replace(self._remote_info, node_rank=None)
        connect(self)

    monkeypatch.setattr(Connection, "connect", connect_here)


def _fake_arm_backend(monkeypatch, *, failing_ip=None, disconnected=None):
    """Register a fake arm backend and select it for Franka robots."""
    from rlinf.robotics.parts.arms.base import Arm

    _opens_here(monkeypatch)

    class FakeArm(_BareArm):
        def __init__(self, robot_ip, *_args, **_kwargs):
            self.robot_ip = robot_ip

        @property
        def parts(self):
            return {}

        @property
        def observation_features(self):
            return {"state": {"shape": (1,)}}

        @property
        def action_features(self):
            return {"target": {"shape": (1,)}}

        def get_observation(self):
            return {"state": np.array([1.0])}

        def send_action(self, action):
            return action

        def _open(self):
            if failing_ip is not None and self.robot_ip == failing_ip:
                raise RuntimeError("right arm is unreachable")
            return f"arm:{self.robot_ip}"

        def _release(self, device):
            if disconnected is not None:
                disconnected.append(self.robot_ip)

    # Populate the registry before monkeypatch installs the temporary backend.
    Arm.backends()
    monkeypatch.setitem(Arm.__dict__["_BACKENDS"], "bench", FakeArm)
    monkeypatch.setattr(franka_module.FrankaRobot, "BACKEND", "bench")
    return FakeArm


def test_an_arm_backend_is_selected_from_the_registry_like_any_driver():
    from rlinf.robotics.parts.arms.base import Arm
    from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robots import DualFrankaRobot, FrankaRobot

    assert Arm.backend("franka_ros") is FrankaROSArm
    assert Arm.backend("franky") is FrankyArm
    assert {"franka_ros", "franky"} <= set(Arm.backends())

    # The robot selects the backend by its registry name.
    assert FrankaRobot.BACKEND == "franka_ros"
    assert DualFrankaRobot.BACKEND == "franky"
    for robot in (FrankaRobot, DualFrankaRobot):
        assert Arm.backend(robot.BACKEND) is not None

    with pytest.raises(ValueError, match="Unsupported Arm backend"):
        Arm.backend("no_such_stack")


def test_a_backend_maps_the_robot_settings_onto_its_own_constructor():
    from rlinf.robotics.parts.arms.base import Arm
    from rlinf.robotics.robots import FrankaRobot

    ros = FrankaRobot.declare_arm(
        "10.0.0.2", node_rank=1, name="arm", backend="franka_ros"
    )
    assert type(ros).__name__ == "FrankaROSArm"
    assert ros.node_rank == 1, "placement must survive the mapping"

    franky = FrankaRobot.declare_arm(
        "10.0.0.3", node_rank=2, name="arm", backend="franky"
    )
    assert type(franky).__name__ == "FrankyArm"
    assert franky.node_rank == 2

    # An end effector is declared on its own, and takes its own placement.
    hand = FrankaRobot.declare_end_effector(
        "10.0.0.3", node_rank=4, name="hand", end_effector_type="ruiyan_hand"
    )
    assert type(hand).__name__ == "RuiyanHand"
    assert hand.node_rank == 4, "an end effector is placed independently"

    # End-effector settings reach the end effector, not the arm.
    with pytest.raises(TypeError, match="does not take"):
        FrankaRobot.declare_arm(
            "10.0.0.3",
            node_rank=0,
            name="arm",
            backend="franky",
            gripper_type="robotiq",
        )

    # A backend with no matching options also rejects them.
    class Plain(Arm):
        def __init__(self, address):
            self.address = address

        @property
        def observation_features(self):
            return {}

        @property
        def action_features(self):
            return {}

        def _open(self):
            return "device"

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

    assert Plain.declare("addr", node_rank=3).node_rank == 3
    with pytest.raises(TypeError, match="does not take"):
        Plain.declare("addr", gripper_connection="/dev/ttyUSB0")


def test_every_canonical_arm_reports_the_same_fields_from_one_place():
    import inspect

    from rlinf.robotics.parts.arms.base import ARM_STATE_FIELDS, BaseArm
    from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.parts.arms.gim_arm import GimArm

    for driver in (FrankaROSArm, FrankyArm, GimArm):
        assert issubclass(driver, BaseArm)
        for inherited in ("observation_features", "get_observation"):
            assert inherited not in vars(driver), (
                f"{driver.__name__} writes its own {inherited}; the three of "
                "them had the same body three times"
            )
        assert set(driver.STATE_FIELDS) == set(ARM_STATE_FIELDS)
        assert "get_state" in vars(driver), f"{driver.__name__} must supply state"

    # Robot builders depend on the category interface.
    assert inspect.isabstract(BaseArm)


def test_declaring_arms_opens_nothing_until_connect(monkeypatch):
    from rlinf.robotics.parts.arms.base import Arm

    class NeverOpens(_BareArm):
        def __init__(self, *_args, **_kwargs):
            pass

        def _open(self):
            raise AssertionError("nothing may be opened while composing")

    Arm.backends()
    monkeypatch.setitem(Arm.__dict__["_BACKENDS"], "bench", NeverOpens)
    monkeypatch.setattr(franka_module.FrankaRobot, "BACKEND", "bench")

    robot = FrankaRobot(
        arm=FrankaRobot.declare_arm("10.0.0.1", node_rank=0, name="left")
    )

    assert not robot.is_connected


def test_connect_tears_down_parts_already_opened(monkeypatch):
    disconnected: list[str] = []
    _fake_arm_backend(monkeypatch, failing_ip="10.0.0.2", disconnected=disconnected)

    robot = FrankaRobot(
        left=FrankaRobot.declare_arm("10.0.0.1", node_rank=0, name="left"),
        right=FrankaRobot.declare_arm("10.0.0.2", node_rank=0, name="right"),
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        robot.connect()

    assert disconnected == ["10.0.0.1"]


def test_declaring_arms_scales_past_two(monkeypatch):
    _fake_arm_backend(monkeypatch)

    robot = FrankaRobot(
        **{
            name: FrankaRobot.declare_arm(f"10.0.0.{index}", node_rank=0, name=name)
            for index, name in enumerate(("left", "right", "third"), start=1)
        }
    )
    robot.connect()

    assert list(robot.children) == ["left", "right", "third"]
    assert robot.is_connected


def test_one_connection_is_opened_once_however_often_it_is_named():
    opens: list[str] = []

    class Riding(ControllablePart):
        """Borrow the connection opened by its host."""

        @property
        def observation_features(self) -> dict:
            return {}

        @property
        def action_features(self) -> dict:
            return {}

        def get_observation(self) -> dict:
            return {}

        def send_action(self, action):
            return action

    class RidingCamera(Riding, Camera):
        pass

    class CoupledHardware(Connection):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"left": Riding(), "right": Riding(), "wrist": RidingCamera()}

        def _open(self):
            opens.append("open")
            return "link"

    hardware = CoupledHardware()
    robot = Robot(
        left=hardware.part("left"),
        right=hardware.part("right"),
        wrist=hardware.part("wrist"),
    )
    robot.connect()

    assert opens == ["open"], "the shared connection was opened more than once"
    assert isinstance(robot.child("wrist"), Camera)
    assert robot.is_connected


def test_a_robot_composes_an_arm_and_gets_what_rides_on_it():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.robots import FrankaRobot, GimArmRobot

        arm = FrankaRobot.declare_arm("10.0.0.2", node_rank=0, name="arm")
        assert list(arm.parts) == [], (
            "a Franka arm backs no end effector: the hand answers on its own "
            "endpoint, so it is composed beside the arm rather than under it"
        )
        hand = FrankaRobot.declare_end_effector(
            "10.0.0.2",
            node_rank=0,
            name="hand",
            gripper_type="robotiq",
            gripper_connection="/dev/ttyUSB0",
        )

        robot = FrankaRobot(arm=arm, end_effector=hand)
        assert list(robot.children) == ["arm", "end_effector"]
        assert robot.child("arm") is arm
        assert robot.child("end_effector") is hand
        # Neither owns the other, so either can be placed on its own node.
        assert hand.owner is hand and arm.owner is arm

        # Readings preserve the composed tree structure.
        assert set(robot.observation_features["arm"]) >= {"tcp_pose"}
        assert set(robot.observation_features["end_effector"]) == {"state"}
        assert set(robot.action_features["arm"]) == {"tcp_pose"}
        assert set(robot.action_features["end_effector"]) == {"target"}

        config = {
            "node_rank": 0,
            "can_interface": "can0",
            "arm_variant": "arm6",
            "gripper_type": "default",
            "control_mode": "position",
            "env_idx": 0,
            "worker_rank": 0,
        }
        fitted = GimArmRobot.build(enable_gripper=True, **config)
        bare = GimArmRobot.build(enable_gripper=False, **config)

    assert list(fitted.child("arm").children) == ["end_effector"]
    assert list(bare.child("arm").children) == [], (
        "the arm knows whether a gripper is fitted; the robot must not decide "
        "that a second time"
    )


def test_every_env_reaches_a_real_connection_through_the_tree():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.robots import (
            DualFrankaRobot,
            FrankaRobot,
            GimArmRobot,
            Turtle2Robot,
        )

        # Exercise each driver path used by the shipped environments.
        reached = [
            (
                FrankaRobot.build(
                    robot_ip="10.0.0.2", node_rank=0, env_idx=0, worker_rank=0
                )
                .child("arm")
                .owner,
                "get_state",
            ),
            (
                GimArmRobot.build(
                    node_rank=0,
                    can_interface="can0",
                    arm_variant="arm6",
                    enable_gripper=True,
                    gripper_type="default",
                    control_mode="position",
                    env_idx=0,
                    worker_rank=0,
                )
                .child("arm")
                .owner,
                "get_state",
            ),
            (
                Turtle2Robot.build(
                    frequency=50,
                    camera_ids=(1, 2),
                    env_idx=0,
                    node_rank=0,
                    worker_rank=0,
                )
                .child("left")
                .child("arm")
                .owner,
                "get_cams",
            ),
        ]
        dual = DualFrankaRobot.build(
            left_robot_ip="1.2.3.4",
            right_robot_ip="1.2.3.5",
            left_gripper_connection="/dev/a",
            right_gripper_connection="/dev/b",
            env_idx=0,
            worker_rank=0,
            node_rank=0,
        )
        reached.append((dual.child("left").child("arm").owner, "clear_errors"))

        for connection, method in reached:
            assert not isinstance(connection, PartGroup), (
                f"the path reached a {type(connection).__name__}, not a driver"
            )
            assert hasattr(connection, method), (
                f"{type(connection).__name__} has no {method}(), which the env calls"
            )

        # Groups do not claim ownership of a connection.
        with pytest.raises(TypeError, match="rides no connection"):
            dual.child("left").owner


def _arm_with_a_camera_of_its_own(log):
    """Build an arm with an independently connected wrist camera."""

    class WristCamera(Camera):
        def _open(self):
            log.append("open:camera")
            return "usb"

        def _release(self, device):
            log.append("close:camera")

        @property
        def observation_features(self):
            return {"frame": {}}

        def get_observation(self):
            return {"frame": "IMAGE"}

    class ArmWithCamera(ControllablePart):
        def __init__(self, *args, **kwargs):
            self._camera = WristCamera()

        def _open(self):
            log.append("open:arm")
            return "arm"

        def _release(self, device):
            log.append("close:arm")

        @property
        def observation_features(self):
            return {"q": {}}

        @property
        def action_features(self):
            return {}

        def get_observation(self):
            return {"q": 0}

        def send_action(self, action):
            return action

        @property
        def parts(self):
            return {"wrist": self._camera}

    return ArmWithCamera()


def test_a_rider_may_not_shadow_one_of_its_carriers_own_fields():
    class Rider(RobotPart):
        @property
        def observation_features(self):
            return {"v": {}}

        def get_observation(self):
            return {"v": "RIDER"}

    class Shadowed(ControllablePart):
        def _open(self):
            return "arm"

        @property
        def observation_features(self):
            return {"tcp_pose": {}}

        @property
        def action_features(self):
            return {"tcp_pose": {}}

        def get_observation(self):
            return {"tcp_pose": "ARM"}

        def send_action(self, action):
            return action

        @property
        def parts(self):
            return {"tcp_pose": Rider()}

    with pytest.raises(ValueError, match="also its own observation or action"):
        Shadowed().children


def test_a_constructor_reaches_no_hardware_and_no_vendor_library():
    import sys
    import warnings

    from rlinf.robotics.parts.arms.gim_arm import GimArm
    from rlinf.robotics.parts.end_effectors.hands.ruiyan import RuiyanHand

    sys.modules.pop("rlinf_dexhand", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hand = RuiyanHand(port="/dev/ttyUSB9", node_rank=1)
        arm = GimArm("can-not-here", "arm6", True, "default", "position", node_rank=2)

    assert "rlinf_dexhand" not in sys.modules, (
        "RuiyanHand imported its vendor SDK while composing, so a hand bound "
        "for another node needs that package on this one"
    )
    assert not [w for w in caught if "CAN interface" in str(w.message)], (
        "GimArm warned about the composing machine's CAN bus for an arm that "
        "is going to node 2"
    )
    assert not hand.is_connected and not arm.is_connected


def test_a_rider_holding_its_own_link_is_opened_by_the_robot():
    log: list[str] = []
    arm = _arm_with_a_camera_of_its_own(log)
    robot = Robot(arm=arm)

    assert len(robot.owners()) == 2, (
        "the camera rides no connection but its own, so it is a second one to "
        f"open; owners() found {[type(o).__name__ for o in robot.owners()]}"
    )

    robot.connect()
    assert log == ["open:arm", "open:camera"]
    assert arm.child("wrist").is_connected

    robot.disconnect()
    assert log[-2:] == ["close:camera", "close:arm"], "closed newest first"


def test_the_tree_holds_the_same_objects_every_time_it_is_walked():
    log: list[str] = []
    robot = Robot(arm=_arm_with_a_camera_of_its_own(log))

    once = robot.child("arm").child("wrist")
    assert robot.child("arm").child("wrist") is once
    assert robot.named_parts["arm.wrist"] is once
    assert robot.parts_of_type(Camera)["arm.wrist"] is once

    robot.connect()
    try:
        assert once.is_connected, "the object the tree hands out is the one opened"
    finally:
        robot.disconnect()


def test_a_failed_connect_leaves_nothing_open():
    released: list[Any] = []

    class Balky(RobotPart):
        def _open(self):
            return "device"

        def _opened(self):
            raise RuntimeError("the capture loop would not start")

        def _release(self, device):
            released.append(device)

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    part = Balky()
    with pytest.raises(RuntimeError, match="capture loop"):
        part.connect()

    assert released == ["device"], "the device was not handed back"
    assert not part.is_connected


def test_a_device_is_released_even_when_teardown_throws():
    released: list[Any] = []

    class Balky(RobotPart):
        def _open(self):
            return "device"

        def _closing(self):
            raise RuntimeError("the capture loop would not stop")

        def _release(self, device):
            released.append(device)

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    part = Balky()
    part.connect()
    with pytest.raises(RuntimeError, match="capture loop"):
        part.disconnect()

    assert released == ["device"]
    assert not part.is_connected


def test_a_handle_that_is_falsy_is_still_a_handle():
    for handle in (0, "", 0.0, []):

        class Zero(RobotPart):
            def _open(self):
                return handle

            @property
            def observation_features(self):
                return {}

            def get_observation(self):
                return {}

        part = Zero()
        part.connect()
        assert part._device is handle, f"{handle!r} was replaced by the part"
        assert part.is_connected
        part.disconnect()


def test_rollback_closes_every_connection_even_if_one_will_not():
    log: list[str] = []

    def part(tag, fail_open=False, fail_close=False):
        class Flaky(RobotPart):
            def _open(self):
                if fail_open:
                    raise RuntimeError("hardware unreachable")
                log.append(f"open:{tag}")
                return tag

            def _release(self, device):
                if fail_close:
                    raise RuntimeError("this one will not close")
                log.append(f"close:{tag}")

            @property
            def observation_features(self):
                return {}

            def get_observation(self):
                return {}

        return Flaky()

    group = PartGroup(
        first=part("first"),
        second=part("second", fail_close=True),
        third=part("third", fail_open=True),
    )

    with pytest.raises(RuntimeError):
        group.connect()

    assert "close:first" in log, (
        f"the rollback stopped at the connection that would not close: {log}"
    )


def test_a_device_with_its_own_link_keeps_it_when_a_connection_lists_it():
    events: list[str] = []

    class WristCamera(FakeCamera):
        pass

    class ArmWithCamera(FakeControllablePart):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"arm": self, "wrist": WristCamera("wrist", events, node_rank=3)}

    arm = ArmWithCamera("arm", events)
    wrist = arm.part("wrist")

    assert wrist.owner is wrist, "the camera was adopted by the arm"
    assert wrist.node_rank == 3, "the camera lost the node it named"

    arm.connect()
    assert arm.is_connected
    assert not wrist.is_connected, (
        "the camera reported itself connected off the back of the arm's link"
    )

    # Views without their own connection are adopted by the host.
    gripper = MethodEndEffector(arm, state_field="gripper_position")
    assert gripper.owner is arm


def test_a_connection_answers_its_parts_before_it_is_opened():
    events: list[str] = []

    class HostWithSubparts(FakeControllablePart):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"arm": self, "end_effector": FakeEndEffector("ee", events)}

    host = HostWithSubparts("host", events)

    assert not host.is_connected
    assert set(host.parts) == {"arm", "end_effector"}
    assert isinstance(host.part("arm"), RobotPart)
    assert isinstance(host.part("end_effector"), EndEffector)
    assert events == [], "asking what a connection backs must not open it"


def test_any_connection_can_be_placed_not_only_arms():
    events: list[str] = []
    camera = FakeCamera("wrist", events, node_rank=2)

    assert camera.node_rank == 2
    assert FakeCamera("bench", events).node_rank is None
    assert events == [], "declaring where a camera runs must not open it"


def test_every_robot_owns_its_construction():
    registry = RobotDiscovery.registry

    for name, registration in registry.items():
        build = registration.build
        assert build is not None, f"{name} registered no builder"
        assert getattr(build, "__self__", None) is registration.robot_cls, (
            f"{name}'s builder is not bound to {registration.robot_cls.__name__}"
        )


def test_dual_franka_inherits_declaration_from_franka():
    assert issubclass(DualFrankaRobot, FrankaRobot)
    # DualFranka inherits arm declaration and changes only arm construction.
    assert DualFrankaRobot.declare_arm.__func__ is FrankaRobot.declare_arm.__func__
    assert DualFrankaRobot.build_arms.__func__ is not FrankaRobot.build_arms.__func__, (
        "only the arm count differs, and that is what build_arms says"
    )
    # The backend selection applies independently of arm count.
    assert (FrankaRobot.BACKEND, DualFrankaRobot.BACKEND) == ("franka_ros", "franky")
    # Arm construction contains the remaining single/dual distinction.
    overridden = [
        name
        for name in ("declare_arm", "build_arms", "build_cameras", "build")
        if getattr(DualFrankaRobot, name).__func__
        is not getattr(FrankaRobot, name).__func__
    ]
    assert overridden == ["build_arms"]


def test_every_part_places_independently_whatever_it_is():
    def fake(base):
        class Fake(base):
            state_dim = action_dim = 1
            control_mode = "binary"

            def __init__(self, *args, **kwargs):
                pass

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            def _open(self):
                return "device"

            def get_observation(self):
                return {}

            def get_state(self):
                return np.array([0.0])

            def send_action(self, action):
                return action

            def command(self, action):
                return True

        return Fake

    arm = fake(ControllablePart)("10.0.0.1", node_rank=1)
    gripper = fake(EndEffector)(port="/dev/ttyUSB0", node_rank=2)
    wrist = fake(Camera)(node_rank=3)
    robot = Robot(arm=PartGroup(arm=arm, gripper=gripper, wrist=wrist))

    assert [part.node_rank for part in (arm, gripper, wrist)] == [1, 2, 3]
    # Three independent connections require three opens.
    assert len(robot.owners()) == 3
    assert [part.owner for part in (arm, gripper, wrist)] == [arm, gripper, wrist]


def test_a_leaf_part_placed_remotely_is_still_the_part_it_was():
    from rlinf.robotics.placement import remote_view_of

    class Leaf(Camera):
        @property
        def observation_features(self):
            return {"frame": {}}

        def _open(self):
            return object()

        def get_observation(self):
            return {"frame": None}

    view = remote_view_of(Leaf)

    assert issubclass(view, Leaf) and issubclass(view, Camera)
    assert "get_observation" in view.__dict__, "the reading call must travel"
    assert isinstance(view.__dict__["observation_features"], property), (
        "a property is not callable, so a worker group would not bind it; the "
        "view has to read it through the attribute call instead"
    )
    assert "parts" not in view.__dict__, "composition is answered here"
    assert Leaf().parts == {}


def test_declaring_cameras_needs_no_config_class():
    from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo

    info = CameraInfo(name="scene", serial_number="123", camera_type="realsense")
    declared = Camera.declare({"scene": info}, node_rank=4)

    assert set(declared) == {"scene"}
    assert declared["scene"].node_rank == 4
    assert type(declared["scene"]).__name__ == "RealSenseCamera"
    assert not declared["scene"].is_connected, "declaring a camera must not open it"
    assert Camera.declare(None) == {}
    # Backend resolution uses the category registry.
    assert BaseCamera.backend("rs") is BaseCamera.backend("realsense")
    assert set(BaseCamera.backends()) >= {"realsense", "rs", "zed", "lumos"}
    with pytest.raises(ValueError, match="Unsupported BaseCamera backend"):
        BaseCamera.backend("no-such-camera")


def test_failed_connect_can_be_retried():
    opened: list[str] = []
    state = {"failing": True}

    def maker(tag, flaky=False):
        class Made(ControllablePart):
            def __init__(self, *args, **kwargs):
                pass

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            def _open(self):
                if flaky and state["failing"]:
                    raise RuntimeError("hardware unreachable")
                opened.append(tag)
                return object()

            def get_observation(self):
                return {}

            def send_action(self, action):
                return action

        return Made

    left = maker("left")()
    robot = Robot(left=left, right=maker("right", flaky=True)())

    with pytest.raises(RuntimeError, match="unreachable"):
        robot.connect()

    assert robot.child("left") is left, "the tree did not go back to what it held"
    assert not left.is_connected, (
        "the arm that opened was left connected with nobody holding it"
    )

    state["failing"] = False
    robot.connect()

    assert opened == ["left", "left", "right"], "the retry did not re-open"
    assert robot.is_connected

    robot.disconnect()
    assert not left.is_connected
    robot.connect()
    assert robot.is_connected, "a disconnected robot must be connectable again"


def test_a_robot_owned_camera_is_opened_and_closed_like_any_other_part(monkeypatch):
    events: list[str] = []

    class Cam(Camera):
        def __init__(self, *args, **kwargs):
            pass

        @property
        def observation_features(self):
            return {"frame": {}}

        def _open(self):
            events.append(f"open@{self.node_rank}")
            return "camera"

        def _release(self, device):
            events.append("close")

        def get_observation(self):
            return {"frame": None}

    camera = Cam(node_rank=5)
    robot = Robot(arm=PartGroup(arm=FakeControllablePart("arm", []), wrist=camera))

    assert camera.node_rank == 5, "the camera did not keep the node it named"

    _opens_here(monkeypatch)
    robot.connect()
    robot.disconnect()

    assert events == ["open@None", "close"]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_the_lazy_package_still_types_what_it_exports():
    source = (_ROOT / "rlinf" / "robotics" / "__init__.py").read_text()
    tree = ast.parse(source)

    typed: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)):
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.ImportFrom):
                module = "." * statement.level + (statement.module or "")
                for alias in statement.names:
                    typed[alias.name] = module

    from rlinf.robotics import _MODULE_BY_NAME

    missing = sorted(set(_MODULE_BY_NAME) - set(typed))
    assert missing == [], (
        f"lazily exported but invisible to a type checker: {missing}. "
        "Add them to the TYPE_CHECKING block in rlinf/robotics/__init__.py."
    )

    extra = sorted(set(typed) - set(_MODULE_BY_NAME))
    assert extra == [], f"typed but not exported at run time: {extra}"

    wrong = {
        name: (module, _MODULE_BY_NAME[name])
        for name, module in typed.items()
        if module != _MODULE_BY_NAME[name]
    }
    assert wrong == {}, f"typed from a different module than it loads from: {wrong}"


def test_scheduler_has_no_robotics_dependency():
    scheduler_dir = _ROOT / "rlinf" / "scheduler"
    offenders = {
        path.relative_to(_ROOT): module
        for path in scheduler_dir.rglob("*.py")
        for module in _imports(path)
        if module == "rlinf.robotics" or module.startswith("rlinf.robotics.")
    }

    assert offenders == {}


_SCHEDULER_BRIDGE = Path("rlinf") / "robotics" / "placement" / "handles.py"


def test_robotics_devices_do_not_depend_on_the_scheduler_or_gym():
    robotics_dir = _ROOT / "rlinf" / "robotics"
    device_paths = [
        robotics_dir / "robot.py",
        robotics_dir / "adapters.py",
        *robotics_dir.joinpath("parts").rglob("*.py"),
    ]
    forbidden = ("gymnasium", "rlinf.scheduler")
    offenders = {
        path.relative_to(_ROOT): module
        for path in device_paths
        if path.relative_to(_ROOT) != _SCHEDULER_BRIDGE
        for module in _imports(path)
        if module == forbidden or module.startswith(forbidden)
    }

    assert offenders == {}


def test_scheduler_use_is_confined_to_the_composition_layer():
    robotics_dir = _ROOT / "rlinf" / "robotics"
    allowed = {
        _SCHEDULER_BRIDGE,
        Path("rlinf") / "robotics" / "discovery" / "registry.py",
    }
    importers = {
        path.relative_to(_ROOT)
        for path in robotics_dir.rglob("*.py")
        for module in _imports(path)
        if module == "rlinf.scheduler" or module.startswith("rlinf.scheduler.")
    }
    leaks = {
        path
        for path in importers
        if path not in allowed and path.parent.name != "robots"
    }

    assert leaks == set()


def test_realworld_environments_do_not_own_controller_workers():
    realworld_dir = _ROOT / "rlinf" / "envs" / "realworld"
    legacy_controller_files = {
        "franka/franka_controller.py",
        "franka/franky_controller.py",
        "gim_arm/gim_arm_controller.py",
        "xsquare/turtle2_smooth_controller.py",
    }

    assert not any(
        realworld_dir.joinpath(path).exists() for path in legacy_controller_files
    )


def test_env_packages_live_under_sim_or_real():
    envs = _ROOT / "rlinf" / "envs"
    # Ignore stale bytecode left by a moved package.
    stray = sorted(
        path.name
        for path in envs.iterdir()
        if path.is_dir()
        and path.name not in {"sim", "real", "venv", "wrappers", "__pycache__"}
        and any(path.rglob("*.py"))
    )

    assert stray == []
    assert "_MovedEnvFinder" not in (envs / "__init__.py").read_text()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib\n"
            "for name in ('rlinf.envs.realworld', 'rlinf.envs.maniskill'):\n"
            "    try:\n"
            "        importlib.import_module(name)\n"
            "    except ModuleNotFoundError:\n"
            "        continue\n"
            "    raise AssertionError(name + ' still resolves')\n"
            "import rlinf.envs.real, rlinf.envs.sim.maniskill, rlinf.envs.utils\n",
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_teleop_devices_are_parts_like_any_other_hardware():
    from rlinf.robotics.parts.base import RobotPart
    from rlinf.robotics.parts.teleop import (
        Glove,
        PicoController,
        SpaceMouse,
        TeleopLeaderArm,
    )

    devices = (SpaceMouse, TeleopLeaderArm, Glove, PicoController)

    assert all(issubclass(device, RobotPart) for device in devices)
    assert not (_ROOT / "rlinf" / "envs" / "real" / "teleop" / "devices").exists()

    # Declaration remains hardware-free and supports offline description.
    mouse = SpaceMouse()
    assert not mouse.is_connected
    assert sorted(mouse.observation_features) == ["buttons", "twist"]


def test_teleop_readers_do_not_import_gymnasium():
    readers = _ROOT / "rlinf" / "robotics" / "parts" / "teleop" / "readers"
    offenders = {
        path.name
        for path in readers.glob("*.py")
        if re.search(r"^\s*(import|from)\s+gymnasium\b", path.read_text(), re.M)
    }

    assert offenders == set()


# DOSW1 production composition through its hardware-free dummy mode


def _dosw1_robot():
    from rlinf.robotics.robots import DOSW1Robot

    return DOSW1Robot.build(is_dummy=True)


def test_building_a_real_robot_touches_no_hardware():
    from rlinf.robotics.parts.base import PartGroup, RobotPart

    robot = _dosw1_robot()

    assert not robot.is_connected
    # Every exported part shares the same DOSW1 SDK session.
    leaves = [
        leaf
        for part in robot.children.values()
        for leaf in (part.children.values() if isinstance(part, PartGroup) else [part])
    ]
    assert leaves, "the robot composed no parts"
    assert all(isinstance(leaf, RobotPart) for leaf in leaves)
    # Ownership is based on connection identity, not constructor equality.
    sessions = [leaf.owner for leaf in leaves]
    assert all(session is sessions[0] for session in sessions)
    assert not sessions[0].is_connected, "composing the robot opened the session"


def test_parts_sharing_a_session_are_never_read_concurrently():
    robot = _dosw1_robot()

    assert len(robot.owners()) == 1, "DOSW1 should present exactly one session"
    assert robot._batches() == [["left", "right"]], (
        f"both sides ride one session but were batched as {robot._batches()}"
    )


def test_parts_on_separate_connections_still_run_together():
    from rlinf.robotics.robots.dual_franka import DualFrankaRobot

    robot = DualFrankaRobot.build(
        left_robot_ip="1.2.3.4",
        right_robot_ip="1.2.3.5",
        left_gripper_connection="/dev/ttyUSB0",
        right_gripper_connection="/dev/ttyUSB1",
        env_idx=0,
        worker_rank=0,
        node_rank=0,
    )

    # Four connections now: an arm and a hand on each side.
    assert len(robot.owners()) == 4
    assert robot._batches() == [["left"], ["right"]]
    # The hand no longer shares the arm's connection, so the two run at once.
    assert robot.child("left")._batches() == [["arm"], ["end_effector"]]
    assert set(robot.child("left").children) >= {"arm", "end_effector"}


def test_a_group_spanning_two_sessions_pulls_both_into_one_batch():
    class Riding(RobotPart):
        @property
        def observation_features(self) -> dict:
            return {}

        def get_observation(self) -> dict:
            return {}

    class Session(RobotPart):
        @property
        def observation_features(self) -> dict:
            return {}

        def get_observation(self) -> dict:
            return {}

        def _open(self):
            return "link"

        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"a": Riding(), "b": Riding()}

    first, second = Session(), Session()
    tree = PartGroup(
        x=first.part("a"),
        bridge=PartGroup(p=first.part("b"), q=second.part("a")),
        y=second.part("b"),
    )

    assert len(tree.owners()) == 2
    assert tree._batches() == [["x", "bridge", "y"]]


def test_real_robot_lifecycle_without_hardware():
    robot = _dosw1_robot()

    assert not robot.is_connected
    robot.connect()
    assert robot.is_connected

    observation = robot.get_observation()
    assert set(observation) == {"left", "right"}
    assert set(observation["left"]) == {"arm", "gripper"}
    assert observation["left"]["arm"]["joint_position"].shape == (6,)

    applied = robot.send_action(
        {
            "left": {
                "arm": {"joint_position": observation["left"]["arm"]["joint_position"]}
            }
        }
    )
    assert set(applied) == {"left"}

    robot.reset()
    robot.disconnect()
    assert not robot.is_connected

    # Disconnect restores declarations for a later reconnect.
    robot.connect()
    assert robot.is_connected
    robot.disconnect()


def test_one_connection_is_opened_once_for_every_part_it_drives():
    robot = _dosw1_robot()

    owners = robot.owners()
    assert len(owners) == 1, (
        f"four parts on one session produced {len(owners)} connections to open"
    )
    left = robot.child("left").child("arm")
    right = robot.child("right").child("arm")
    assert left.owner is right.owner is owners[0]

    robot.connect()
    assert left.is_connected and right.is_connected
    robot.disconnect()
    assert not left.is_connected


def test_coupled_hardware_exposes_its_components_as_parts():
    from rlinf.robotics import EndEffector

    robot = _dosw1_robot()
    robot.connect()

    assert sorted(robot.named_parts) == [
        "left",
        "left.arm",
        "left.gripper",
        "right",
        "right.arm",
        "right.gripper",
    ]
    assert sorted(robot.parts_of_type(EndEffector)) == ["left.gripper", "right.gripper"]
    assert type(robot.child("left").child("arm")).__name__ == "DOSW1Arm"

    robot.disconnect()


def test_observation_tree_follows_the_composition_on_real_parts():
    robot = _dosw1_robot()
    robot.connect()

    observation = robot.get_observation()
    paths = {
        f"{group}.{part}" for group, parts in observation.items() for part in parts
    }

    assert paths == set(robot.named_parts) - set(robot.children)

    robot.disconnect()


# Teleoperation composition


def _scripted_device(reading):
    """Build a teleoperation part that returns a fixed reading."""
    from rlinf.robotics.parts.teleop.devices import TeleopPart

    class Scripted(TeleopPart):
        def _open(self):
            return object()

        @property
        def observation_features(self):
            return {key: {} for key in reading}

        def get_observation(self):
            return dict(reading)

    device = Scripted()
    device.connect()
    return device


def _kinds(*names):
    """Build the action-kind mapping expected by a robot."""
    from rlinf.robotics.teleop import ActionKind

    per_name = {
        "hand": ActionKind.HAND,
        "end_effector": ActionKind.GRIPPER,
    }
    return {
        name: per_name.get(name.rsplit(".", 1)[-1], ActionKind.CARTESIAN_DELTA)
        for name in names
    }


def _binding(produces, value, driving=True):
    from rlinf.robotics.teleop import TeleopBinding

    class Fixed(TeleopBinding):
        PRODUCES = _kinds(*produces)

        def action(self, reading, context):
            from rlinf.robotics.teleop import TeleopAction

            return TeleopAction(parts=dict.fromkeys(produces, value), driving=driving)

        def is_driving(self, reading):
            return driving

    return Fixed()


def test_two_devices_merge_into_one_action():
    import numpy as np

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(_scripted_device({"twist": 1}), _binding(("arm",), np.ones(6))),
            TeleopEntry(
                _scripted_device({"angles": 2}), _binding(("hand",), np.ones(6) * 2)
            ),
        ],
        available=_kinds(*("arm", "hand")),
    )

    parts, driving, _ = group.action({})

    assert sorted(parts) == ["arm", "hand"]
    assert driving
    assert np.allclose(parts["hand"], 2.0)


def test_a_part_the_robot_lacks_is_not_filled():
    import numpy as np

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(
                _scripted_device({"twist": 1}),
                _binding(("arm", "end_effector"), np.ones(6)),
            )
        ],
        available=_kinds(*("arm", "hand")),
    )

    assert group.parts == ("arm",)


def test_two_devices_claiming_one_part_is_refused():
    import numpy as np
    import pytest

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    with pytest.raises(ValueError, match="both drive"):
        TeleopGroup(
            [
                TeleopEntry(_scripted_device({}), _binding(("arm",), np.ones(6))),
                TeleopEntry(_scripted_device({}), _binding(("arm",), np.ones(6))),
            ],
            available=_kinds(*("arm",)),
        )


def test_a_device_that_fills_nothing_is_refused():
    import numpy as np
    import pytest

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    with pytest.raises(ValueError, match="fills none"):
        TeleopGroup(
            [TeleopEntry(_scripted_device({}), _binding(("hand",), np.ones(6)))],
            available=_kinds(*("arm", "end_effector")),
        )


def test_drives_separates_two_identical_leaders():
    import numpy as np

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    group = TeleopGroup(
        [
            TeleopEntry(
                _scripted_device({}), _binding(("arm",), np.ones(7)), drives="left"
            ),
            TeleopEntry(
                _scripted_device({}), _binding(("arm",), np.ones(7) * 2), drives="right"
            ),
        ],
        available=_kinds(*("left.arm", "right.arm")),
    )

    parts, _, _ = group.action({})

    assert sorted(parts) == ["left.arm", "right.arm"]
    assert np.allclose(parts["right.arm"], 2.0)


def test_one_device_listed_twice_is_read_once():
    import numpy as np

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    device = _scripted_device({"twist": 1})
    group = TeleopGroup(
        [
            TeleopEntry(device, _binding(("arm",), np.ones(6))),
            TeleopEntry(device, _binding(("end_effector",), np.ones(1))),
        ],
        available=_kinds(*("arm", "end_effector")),
    )

    assert len(group.devices) == 1
    assert group.parts == ("arm", "end_effector")


def test_a_teleop_rig_waits_until_every_reader_is_ready():
    from rlinf.robotics.parts.teleop.devices import TeleopPart
    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    class Starting(TeleopPart):
        def _open(self):
            return object()

        @property
        def ready(self):
            return False

        @property
        def observation_features(self):
            return {"twist": {}}

        def get_observation(self):
            raise AssertionError("an unready reader must not be sampled")

    device = Starting()
    device.connect()
    group = TeleopGroup(
        [TeleopEntry(device, _binding(("arm",), np.ones(6)))],
        available=_kinds("arm"),
    )

    assert group.action({}) == ({}, False, {})


def test_spacemouse_reset_resyncs_the_gripper_and_reports_its_buttons():
    from rlinf.robotics.teleop import SpaceMouseBinding

    binding = SpaceMouseBinding()
    opened = binding.action(
        {"twist": np.zeros(6), "buttons": [False, False]},
        {"gripper_open": True},
    )
    binding.reset()
    closed = binding.action(
        {"twist": np.zeros(6), "buttons": [True, False]},
        {"gripper_open": False},
    )

    assert opened.parts["end_effector"].item() > 0
    assert closed.parts["end_effector"].item() < 0
    assert closed.info == {"left": True, "right": False}
    dex = SpaceMouseBinding(dexterous_hand=True).action(
        {"twist": np.zeros(6), "buttons": [True, False]},
        {"gripper_open": True},
    )
    assert dex.info == {"left": False, "right": True}


def test_leader_arm_only_takes_control_for_motion_or_an_active_gripper():
    from rlinf.robotics.teleop import LeaderArmBinding

    context = {
        "tcp_pose": np.array([0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0]),
        "action_scale": np.array([0.05, 0.3, 1.0]),
    }
    idle = {
        "position": context["tcp_pose"][:3],
        "orientation": context["tcp_pose"][3:],
        "grip": np.array([0.5]),
    }

    assert not LeaderArmBinding().action(idle, context).driving
    assert (
        not LeaderArmBinding(gripper=False)
        .action({**idle, "grip": np.array([0.0])}, context)
        .driving
    )
    moved = {**idle, "position": idle["position"] + np.array([0.01, 0.0, 0.0])}
    assert LeaderArmBinding().action(moved, context).driving


def test_leader_joint_uses_the_legacy_motion_and_gripper_thresholds():
    from rlinf.robotics.teleop import LeaderJointBinding

    binding = LeaderJointBinding(side=0)
    current = np.zeros((2, 7))
    idle = {"joint_position": np.zeros(7), "grip": np.array([0.5])}

    assert not binding.action(idle, {"joint_positions": current}).driving
    moved = {**idle, "joint_position": np.full(7, 0.01)}
    assert binding.action(moved, {"joint_positions": current}).driving
    gripped = {**idle, "grip": np.array([0.0])}
    assert binding.action(gripped, {"joint_positions": current}).driving


def test_the_glove_holds_what_the_operator_posed():
    import numpy as np

    from rlinf.robotics.teleop import GloveBinding

    glove = GloveBinding()
    glove.reset({"hand_reset_pose": np.zeros(6)})

    posed = glove.action({"angles": np.full(6, 0.4)}, {"hand_driving": True}).parts[
        "hand"
    ]
    released = glove.action({"angles": np.full(6, 0.9)}, {"hand_driving": False}).parts[
        "hand"
    ]

    assert np.allclose(posed, released)


def test_the_glove_tracks_only_while_its_gate_is_held():
    import numpy as np

    from rlinf.robotics.teleop import GloveBinding, SpaceMouseBinding

    mouse, glove = SpaceMouseBinding(), GloveBinding()
    glove.reset({"hand_reset_pose": np.zeros(6)})

    released = mouse.publish({"twist": np.zeros(6), "buttons": [False, False]})
    held = mouse.publish({"twist": np.zeros(6), "buttons": [False, True]})

    assert released == {"hand_driving": False}
    assert held == {"hand_driving": True}

    # The first active reading sets the reference; the next applies motion.
    glove.action({"angles": np.zeros(6)}, held)
    moved = glove.action({"angles": np.full(6, 0.3)}, held).parts["hand"]
    assert np.allclose(moved, 0.3)

    # Releasing control preserves the last commanded hand pose.
    assert np.allclose(
        glove.action({"angles": np.full(6, 0.9)}, released).parts["hand"], 0.3
    )


# PICO device and binding


class _ScriptedController:
    """Replay a fixed sequence of controller readings."""

    def __init__(self, steps):
        self._steps = list(steps)
        self._index = 0
        self.is_connected = True

    def get_observation(self):
        import numpy as np

        moved, turned, held, grip = self._steps[min(self._index, len(self._steps) - 1)]
        self._index += 1
        return {
            "held": held,
            "ready": True,
            "hand": "right",
            "calibrated": True,
            "control_value": 1.0 if held else 0.0,
            "position_delta": np.asarray(moved, dtype=np.float64),
            "rotation_delta": np.asarray(turned, dtype=np.float64),
            "grip_close": grip < 0,
            "grip_open": grip > 0,
        }

    def connect(self):
        pass

    def disconnect(self):
        pass


def _pico_context():
    import numpy as np

    return {
        "tcp_pose": np.array([0.3, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "action_scale": np.array([0.05, 0.3, 1.0]),
    }


def test_a_controller_drives_the_arm_to_a_pose_it_can_reach():
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0)
    device = _ScriptedController([((0.025, 0.0, 0.0), (0.0, 0.0, 0.0), True, -1)])

    parts = binding.action(device.get_observation(), _pico_context()).parts

    # Operator motion is relative to the pose captured on takeover.
    assert parts["arm"].size == 9
    assert np.isclose(parts["arm"][0], 0.3 + 0.025)
    assert np.isclose(parts["end_effector"][0], -1.0)
    assert binding.action(device.get_observation(), _pico_context()).driving


def test_releasing_the_grip_mid_chunk_holds_the_arm_where_it_was_left():
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0, hold_current_when_inactive=False)
    device = _ScriptedController(
        [
            ((0.025, 0.0, 0.0), (0.0, 0.0, 0.0), True, -1),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 0),
        ]
    )
    context = _pico_context()

    driven = binding.action(device.get_observation(), context).parts["arm"].copy()
    held = binding.action(device.get_observation(), context).parts["arm"]
    assert np.allclose(held, driven)

    # Policy-only input releases the intervention.
    binding.on_action_chunk_begin()
    assert binding.action(device.get_observation(), context).parts == {}


def test_holding_the_current_pose_leaves_the_gripper_to_the_policy():
    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0, hold_current_when_inactive=True)
    device = _ScriptedController([((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 0)])

    parts = binding.action(device.get_observation(), _pico_context()).parts

    assert set(parts) == {"arm"}


def test_a_delta_binding_has_no_pose_to_hold():
    from rlinf.robotics.teleop import PicoBinding, PicoTcpBinding

    assert PicoBinding(gripper=True).hold(_pico_context()) == {}
    assert "arm" in PicoTcpBinding(gripper=True).hold(_pico_context())


def test_absolute_commands_are_clipped_but_deltas_are_not():
    from rlinf.robotics.teleop import PicoBinding, PicoTcpBinding, SpaceMouseBinding

    assert PicoTcpBinding.CLIPS_TO_ACTION_SPACE
    assert not PicoBinding.CLIPS_TO_ACTION_SPACE
    assert not SpaceMouseBinding.CLIPS_TO_ACTION_SPACE


def test_each_side_reports_its_own_state():
    from rlinf.robotics.teleop import (
        ActionKind,
        PicoTcpBinding,
        TeleopEntry,
        TeleopGroup,
    )

    layout = {
        "left.arm": ActionKind.CARTESIAN_POSE,
        "left.end_effector": ActionKind.GRIPPER,
        "right.arm": ActionKind.CARTESIAN_POSE,
        "right.end_effector": ActionKind.GRIPPER,
    }
    entries = [
        TeleopEntry(
            _ScriptedController(
                [((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), side == "left", 1)]
            ),
            PicoTcpBinding(gripper=True, side=index),
            drives=side,
        )
        for index, side in enumerate(("left", "right"))
    ]
    group = TeleopGroup(entries, available=layout)

    _, driving, info = group.action(_pico_context())

    assert driving
    assert info["left_pico_active"] is True
    assert info["right_pico_active"] is False


def test_reading_a_controller_does_not_need_the_robot():
    import inspect

    from rlinf.robotics.parts.teleop.devices import PicoController
    from rlinf.robotics.parts.teleop.readers import pico

    source = inspect.getsource(pico.PicoExpert)
    for name in ("tcp_pose", "_ref_tcp"):
        assert name not in source, f"the reader still refers to {name!r}"

    observation = PicoController.get_observation
    assert "get_reading" in inspect.getsource(observation)


def test_a_controller_reading_is_data_not_a_handle():
    from rlinf.robotics.parts.teleop.devices import PicoController

    features = PicoController(hand="right").observation_features

    assert set(features) == {
        "held",
        "position_delta",
        "rotation_delta",
        "grip_close",
        "grip_open",
    }


def test_the_arm_anchors_where_it_was_when_the_operator_took_hold():
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0)
    device = _ScriptedController(
        [
            ((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), True, 0),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 0),
            ((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), True, 0),
        ]
    )
    context = _pico_context()

    first = binding.action(device.get_observation(), context).parts["arm"][0]
    binding.action(device.get_observation(), context)
    again = binding.action(device.get_observation(), context).parts["arm"][0]

    # Repeated motion from the same measured pose produces the same command.
    assert np.isclose(first, again)


# Controller packet reader


def _pico_packet(x, y, z, yaw, grip, close=False, open_=False):
    """Build one controller frame in the headset wire format."""
    from scipy.spatial.transform import Rotation as R

    return {
        "right_controller": {
            "position": [x, y, z],
            "orientation": list(R.from_euler("xyz", [0.0, 0.0, yaw]).as_quat()),
            "grip": grip,
            "trigger": 0.0,
        },
        "buttons": {"A": close, "B": open_},
    }


def _pico_reader(**overrides):
    """Build a PICO reader backed by scripted packets."""
    from rlinf.robotics.parts.teleop.readers.pico import PicoExpert

    config = {
        "hand": "right",
        "control_trigger": "grip",
        "control_threshold": 0.85,
        "gripper_close_button": "A",
        "gripper_open_button": "B",
        "position_scale": 1.0,
        "rotation_scale": 0.5,
        "calibration": {"enabled": False, "required": False},
    }
    config.update(overrides)
    PicoExpert.start = lambda self: None
    return PicoExpert(**config)


def test_a_reading_is_empty_until_the_operator_takes_hold():
    import numpy as np

    reader = _pico_reader()
    reader._snapshot = lambda: _pico_packet(0.1, 0.2, 0.3, 0.4, grip=0.0)

    reading = reader.get_reading()

    assert reading["held"] is False
    assert np.allclose(reading["position_delta"], 0.0)


def test_motion_is_measured_from_where_the_operator_took_hold():
    import numpy as np

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    reader._snapshot = lambda: _pico_packet(0.5, 0.5, 0.5, 0.0, grip=0.95)
    reader.get_reading()  # Establish the control anchor at this pose.

    reader._snapshot = lambda: _pico_packet(0.5, 0.5, 0.47, 0.0, grip=0.95)
    reading = reader.get_reading()

    assert reading["held"] is True
    assert np.allclose(reading["position_delta"], [0.03, 0.0, 0.0], atol=1e-9)


def test_letting_go_and_grabbing_again_re_anchors():
    import numpy as np

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    for position, grip in (
        ((0.0, 0.0, 0.0), 0.95),
        ((0.30, 0.0, 0.0), 0.95),
        ((0.30, 0.0, 0.0), 0.0),
        ((0.30, 0.0, 0.0), 0.95),
    ):
        reader._snapshot = lambda p=position, g=grip: _pico_packet(*p, 0.0, grip=g)
        reading = reader.get_reading()

    # Retaking control resets the anchor without applying motion.
    assert reading["held"] is True
    assert np.allclose(reading["position_delta"], 0.0, atol=1e-9)


def test_the_gripper_buttons_are_reported_separately():
    reader = _pico_reader()
    reader._snapshot = lambda: _pico_packet(0, 0, 0, 0, grip=0.95, close=True)

    reading = reader.get_reading()

    assert reading["grip_close"] is True
    assert reading["grip_open"] is False


def test_a_dropped_link_reports_stale_rather_than_stale_motion():
    reader = _pico_reader()
    reader._snapshot = lambda: None

    reading = reader.get_reading()

    assert reading["held"] is False
    assert reading["ready"] is False
    assert reading["stale"] is True


def test_the_binding_turns_a_reading_into_a_command_for_this_arm():
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    binding = PicoTcpBinding(gripper=True, side=0)
    context = _pico_context()

    reader._snapshot = lambda: _pico_packet(0.0, 0.0, 0.0, 0.0, grip=0.95)
    binding.action(reader.get_reading(), context)

    reader._snapshot = lambda: _pico_packet(0.0, 0.0, -0.04, 0.0, grip=0.95)
    reading = reader.get_reading()
    parts = binding.action(reading, context).parts

    # The target combines the takeover pose with operator motion.
    expected = np.asarray(context["tcp_pose"][:3]) + reading["position_delta"]
    assert np.allclose(parts["arm"][:3], expected, atol=1e-6)
    assert np.isclose(reading["position_delta"][0], 0.04, atol=1e-9)


# Shared lifecycle across part categories


def test_every_device_family_is_shaped_the_same_way():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        import rlinf.robotics.parts.cameras  # noqa: F401  - registers drivers
        import rlinf.robotics.parts.end_effectors  # noqa: F401
        import rlinf.robotics.parts.teleop.devices  # noqa: F401
        from rlinf.robotics.parts.arms.base import Arm, BaseArm
        from rlinf.robotics.parts.cameras.base import BaseCamera, Camera
        from rlinf.robotics.parts.end_effectors.base import BaseEndEffector, EndEffector
        from rlinf.robotics.parts.teleop.devices import TeleopPart

        # Configured backend names resolve through their category registry.
        for category in (Arm, Camera, EndEffector):
            assert category.backends(), (
                f"{category.__name__} has no registered backends; a config "
                "naming one would have nothing to resolve against"
            )

        # Driver classes implement both lifecycle hooks unless the category
        # provides one shared release implementation.
        for base in (BaseArm, BaseCamera, BaseEndEffector, TeleopPart):
            assert base._open.__isabstractmethod__, (
                f"{base.__name__} lets a driver inherit _open, so one that "
                "never wrote it fails at the first connect instead of at "
                "class definition"
            )
        for base in (BaseArm, BaseCamera, BaseEndEffector):
            assert base._release.__isabstractmethod__, base.__name__
        assert not getattr(TeleopPart._release, "__isabstractmethod__", False), (
            "TeleopPart releases every reader the same way, so it does it once"
        )


def test_every_part_presents_a_device_category():
    import inspect

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        import rlinf.robotics.parts.teleop.devices  # noqa: F401
        import rlinf.robotics.parts.views  # noqa: F401
        import rlinf.robotics.robots  # noqa: F401
        from rlinf.robotics.parts.arms.base import Arm
        from rlinf.robotics.parts.base import PartGroup, RobotPart
        from rlinf.robotics.parts.cameras.base import Camera
        from rlinf.robotics.parts.end_effectors.base import EndEffector
        from rlinf.robotics.parts.mobility.base import MobileBase
        from rlinf.robotics.parts.teleop.devices import TeleopPart

        Arm.backends()  # Load every arm backend module.

        def descendants(cls):
            for child in cls.__subclasses__():
                yield child
                yield from descendants(child)

        categories = (Arm, Camera, EndEffector, MobileBase, TeleopPart)
        homeless = sorted(
            cls.__name__
            for cls in descendants(RobotPart)
            if cls.__module__.startswith("rlinf.")  # Exclude local fakes.
            and not inspect.isabstract(cls)
            and not issubclass(cls, PartGroup)
            and not any(issubclass(cls, category) for category in categories)
        )

    assert homeless == [], f"parts belonging to no device category: {homeless}"


def test_every_part_family_opens_and_closes_the_same_way():
    import inspect

    from rlinf.robotics.parts.cameras.base import BaseCamera
    from rlinf.robotics.parts.end_effectors.base import BaseEndEffector
    from rlinf.robotics.parts.end_effectors.grippers.base import BaseGripper
    from rlinf.robotics.parts.teleop.devices import TeleopPart

    for family in (TeleopPart, BaseCamera, BaseEndEffector, BaseGripper):
        assert hasattr(family, "_open"), f"{family.__name__} has no _open"
        assert hasattr(family, "_release"), f"{family.__name__} has no _release"

    # Retired lifecycle hook names are no longer accepted.
    for family in (BaseCamera, BaseEndEffector):
        source = inspect.getsource(family)
        for retired in ("_close_device", "def initialize", "def shutdown"):
            assert retired not in source, f"{family.__name__} still has {retired}"

    # Placement-aware connect/disconnect remain owned by Connection.
    for family in (TeleopPart, BaseCamera, BaseEndEffector, BaseGripper):
        for public in ("connect", "disconnect"):
            assert public not in vars(family), (
                f"{family.__name__} overrides {public}; a part placed on another "
                "node would then never be rebuilt there"
            )

    # Categories extend drivers through local lifecycle hooks.
    assert "_opened" in vars(BaseCamera), (
        "BaseCamera starts its capture loop, and that has to run beside the "
        "camera rather than beside whoever is holding it"
    )


def test_a_gripper_is_an_end_effector_rather_than_a_second_kind_of_one():
    import inspect

    from rlinf.robotics.parts.end_effectors.base import BaseEndEffector, EndEffector
    from rlinf.robotics.parts.end_effectors.grippers.base import BaseGripper
    from rlinf.robotics.parts.views import MethodEndEffector

    assert issubclass(BaseGripper, BaseEndEffector)

    # Grippers and other end effectors share one lifecycle declaration.
    for hook in ("_open", "_release"):
        assert BaseGripper.__dict__.get(hook) is None, (
            f"BaseGripper re-declares {hook}; it should inherit the one contract"
        )
        assert getattr(BaseEndEffector, hook).__isabstractmethod__, (
            f"{hook} must be required, or a driver that never wrote one fails "
            "at the first connect instead of at class definition"
        )

    # ``reset`` has one signature across the category.
    assert list(inspect.signature(BaseGripper.reset).parameters) == list(
        inspect.signature(BaseEndEffector.reset).parameters
    )

    # A hosted view borrows its connection and therefore needs no ``_open``.
    assert issubclass(MethodEndEffector, EndEffector)
    assert not issubclass(MethodEndEffector, BaseEndEffector)


def test_a_gripper_is_commanded_in_the_units_it_reports():
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.end_effectors import EndEffector

    with mocked_sdks():
        gripper = EndEffector.of("robotiq", port="/dev/mock-gripper")
        gripper.connect()
        try:
            assert gripper.max_width > 0

            gripper.open()
            assert gripper.position == pytest.approx(gripper.max_width, abs=1e-3)
            gripper.close()
            assert gripper.position == pytest.approx(0.0, abs=1e-3)

            # Round-trip width may differ by one quantized register count.
            quantum = gripper.max_width / 255
            for width in (0.0, gripper.max_width / 2, gripper.max_width):
                gripper.move(width)
                assert gripper.position == pytest.approx(width, abs=quantum)

            # The canonical part API uses the same physical unit.
            gripper.move(gripper.max_width / 3)
            state = gripper.get_state()
            gripper.open()
            gripper.command(state)
            assert gripper.position == pytest.approx(state[0], abs=quantum)

            # Values beyond the stroke clamp instead of wrapping.
            gripper.move(gripper.max_width * 10)
            assert gripper.position == pytest.approx(gripper.max_width, abs=quantum)
        finally:
            gripper.disconnect()


def test_a_franka_hand_is_commanded_in_metres_on_the_wire():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franka_gripper")
        gripper.connect()
        try:
            widths: list[float] = []
            put = gripper._ros.put_channel

            def record(channel, message):
                widths.append(float(message.goal.width))
                return put(channel, message)

            gripper._ros.put_channel = record

            gripper.open()
            gripper.move(0.05)
            gripper.move(gripper.max_width * 10)
        finally:
            gripper.disconnect()

    assert widths == pytest.approx([gripper.max_width, 0.05, gripper.max_width])


def test_every_end_effector_answers_the_same_questions():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors.base import EndEffector
        from rlinf.robotics.robots import DOSW1Robot, FrankaRobot

        def franka(end_effector_type=None):
            robot = FrankaRobot.build(
                robot_ip="10.0.0.2",
                node_rank=0,
                env_idx=0,
                worker_rank=0,
                end_effector_type=end_effector_type,
            )
            return robot.child("end_effector")

        every = {
            "hand on a bus": franka("ruiyan_hand"),
            "gripper on a bus": franka(),
            "gripper on a session": DOSW1Robot.build(is_dummy=True)
            .child("left")
            .child("gripper"),
            "gripper on its own port": EndEffector.of(
                "robotiq", port="/dev/mock-gripper"
            ),
        }

        for label, part in every.items():
            assert isinstance(part, EndEffector), label
            assert part.state_dim >= 1 and part.action_dim >= 1, label
            assert part.control_mode in {"binary", "continuous"}, label
            assert callable(part.get_state) and callable(part.command), label
            assert set(part.observation_features) == {"state"}, label
            assert set(part.action_features) == {"target"}, label

        # A six-finger hand composes exactly where a gripper would.
        assert every["hand on a bus"].action_dim == 6


def test_every_end_effector_reports_its_state_under_the_same_name():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors.base import EndEffectorType

        gripper = EndEffector.of("robotiq", port="/dev/mock-gripper")
        hand = EndEffector.of(EndEffectorType.RUIYAN_HAND, port="/dev/mock-hand")

        for part in (gripper, hand):
            name = type(part).__name__
            assert set(part.observation_features) == {"state"}, (
                f"{name} reports {sorted(part.observation_features)}, not 'state'"
            )
            assert set(part.action_features) == {"target"}, name
            assert part.state_dim >= 1 and part.action_dim >= 1, name
            assert part.control_mode in {"binary", "continuous"}, name

            part.connect()
            try:
                observation = part.get_observation()
                assert set(observation) == {"state"}, name
                assert observation["state"].shape == (part.state_dim,), name
                target = np.zeros(part.action_dim, dtype=np.float32)
                assert set(part.send_action({"target": target})) == {"target"}, name
                part.reset()
                part.reset(target)
            finally:
                part.disconnect()
            assert not part.is_connected, name


class _Part(RobotPart):
    """Record lifecycle events in its device handle."""

    def __init__(self):
        self.log = []

    def _open(self):
        self.log.append("open")
        return "device"

    def _release(self, device):
        self.log.append("close")

    @property
    def observation_features(self):
        return {}

    def get_observation(self):
        return {}


def test_connecting_twice_opens_once():
    part = _Part()
    part.connect()
    part.connect()

    assert part.log == ["open"]
    assert part.is_connected


def test_disconnecting_what_was_never_opened_does_nothing():
    part = _Part()
    part.disconnect()

    assert part.log == []
    assert not part.is_connected


def test_a_part_is_released_with_the_device_it_opened():
    part = _Part()
    part.connect()

    assert part.is_connected
    part.disconnect()

    assert part.log == ["open", "close"]
    assert not part.is_connected
    # Repeated disconnect does not release a device twice.
    part.disconnect()
    assert part.log == ["open", "close"]


def test_a_part_that_says_nothing_about_its_hardware_fails_clearly():
    class Bare(RobotPart):
        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    with pytest.raises(NotImplementedError, match="does not say how to open"):
        Bare().connect()


# Minimal robot composition


def test_a_robot_is_named_parts_and_nothing_else():
    from rlinf.robotics import Robot

    class Gripper(_Part):
        @property
        def observation_features(self):
            return {"width": {"shape": (1,), "dtype": "float32"}}

        def get_observation(self):
            return {"width": 0.5}

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    robot = Bench(arm=_Part(), hand=Gripper())
    robot.connect()

    assert robot.is_connected
    assert set(robot.get_observation()) == {"arm", "hand"}
    assert robot.get_observation()["hand"] == {"width": 0.5}

    robot.disconnect()
    assert not robot.is_connected


def test_build_is_only_for_robots_reached_by_name():
    from rlinf.robotics import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    with pytest.raises(NotImplementedError, match="Construct Bench"):
        Bench.build()


# Connection and part separation


def test_a_connection_backing_several_parts_is_not_observable():
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, ControllablePart

    assert issubclass(Turtle2Connection, Connection)
    assert not issubclass(Turtle2Connection, ControllablePart)

    hardware = Turtle2Connection()
    assert set(hardware.parts) == {
        "left",
        "left_end_effector",
        "right",
        "right_end_effector",
    }
    # Connections omit the observation interface and cannot enter the part tree.
    assert not isinstance(hardware, RobotPart)
    assert not hasattr(hardware, "get_observation")
    assert not hasattr(hardware, "observation_features")


def test_every_robot_composes_from_named_parts():
    from rlinf.robotics.parts.base import Connection, RobotPart
    from rlinf.robotics.robots.dual_franka import DualFrankaRobot
    from rlinf.robotics.robots.franka import FrankaRobot
    from rlinf.robotics.robots.gim_arm import GimArmRobot

    built = [
        FrankaRobot.build_arms(robot_ip="1.2.3.4", node_rank=0),
        GimArmRobot.build_arms(
            node_rank=0,
            can_interface="can0",
            arm_variant="xl",
            enable_gripper=True,
            gripper_type="default",
            control_mode="joint",
        ),
        DualFrankaRobot.build_arms(
            left_robot_ip="1.2.3.4",
            right_robot_ip="1.2.3.5",
            left_gripper_connection="/dev/ttyUSB0",
            right_gripper_connection="/dev/ttyUSB1",
        ),
    ]
    for arms in built:
        for name, value in arms.items():
            values = (
                list(value.children.values()) if hasattr(value, "children") else [value]
            )
            # Each leaf must be a readable capability, not its shared session.
            assert all(isinstance(v, RobotPart) for v in values), (
                f"{name} holds {[type(v).__name__ for v in values]} rather than "
                "parts picked out of its connection"
            )
            assert not any(
                isinstance(v, Connection) and not isinstance(v, RobotPart)
                for v in values
            )


def test_one_connection_backs_every_part_that_names_it():
    from rlinf.robotics.robots.dual_franka import DualFrankaRobot
    from rlinf.robotics.robots.franka import FrankaRobot

    single = FrankaRobot.build_arms(
        robot_ip="1.2.3.4", node_rank=0, gripper_connection="/dev/ttyUSB0"
    )
    # The arm and the hand each answer on their own endpoint, so each owns its
    # own connection rather than borrowing one.
    assert len({id(part.owner) for part in single.values()}) == len(single)

    dual = DualFrankaRobot.build_arms(
        left_robot_ip="1.2.3.4",
        right_robot_ip="1.2.3.5",
        left_gripper_connection="/dev/ttyUSB0",
        right_gripper_connection="/dev/ttyUSB1",
    )
    sides = {
        side: {id(part.owner) for part in group.children.values()}
        for side, group in dual.items()
    }
    # No connection is shared between the two sides.
    assert not (sides["left"] & sides["right"])


def test_a_connection_with_several_parts_hands_each_of_them_out_by_name():
    class Pair(RobotPart):
        def _open(self):
            return "link"

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

        @property
        def parts(self):
            return {"a": _Part(), "b": _Part()}

    pair = Pair()

    assert set(pair.parts) == {"a", "b"}
    assert isinstance(pair.part("a"), RobotPart)
    with pytest.raises(KeyError):
        pair.part("c")


# Bench-check integration


def _fake_robot_type(broken=None):
    """Register a robot made of fakes, and yield its type name."""
    import numpy as np

    from rlinf.robotics.discovery import RobotConfig, RobotDiscovery, register_robot
    from rlinf.robotics.parts.base import Connection, ControllablePart, PartGroup
    from rlinf.robotics.parts.views import MethodArm, MethodEndEffector
    from rlinf.robotics.robot import Robot

    class State:
        def to_dict(self):
            return {"tcp_pose": np.zeros(7), "gripper_position": np.zeros(1)}

    class Bus(Connection):
        def _open(self):
            return "bus"

        def get_state(self):
            return State()

        def move(self, target):
            return target

        @property
        def parts(self):
            return {
                "left": MethodArm(
                    self, commands={"tcp_pose": "move"}, state_fields=("tcp_pose",)
                ),
                "left_end_effector": MethodEndEffector(
                    self, state_field="gripper_position"
                ),
            }

    class Liar(ControllablePart):
        def _open(self):
            return "liar"

        @property
        def observation_features(self):
            return {"tcp_pose": {}}

        @property
        def action_features(self):
            return {}

        def get_observation(self):
            return {"joint_position": np.zeros(7)}

        def send_action(self, action):
            return action

    class Bench(Robot):
        ROBOT_TYPE = f"BenchFake{broken or ''}"

        @classmethod
        def build(cls, **_):
            if broken == "Mismatch":
                return cls(arm=Liar())
            if broken == "Connection":
                return cls(bus=Bus())
            bus = Bus()
            return cls(
                left=PartGroup(
                    arm=bus.part("left"),
                    gripper=bus.part("left_end_effector"),
                )
            )

    class Config(RobotConfig):
        pass

    class Discovery(RobotDiscovery):
        HW_TYPE = Bench.ROBOT_TYPE

        @classmethod
        def enumerate(cls, node_rank, configs=None):
            return None

    register_robot(Config, Bench, build=Bench.build)(Discovery)
    return Bench.ROBOT_TYPE


def _run_bench(robot_type):
    from toolkits.realworld_check.check_robot_parts import check

    return check(robot_type, {})


def test_the_bench_check_passes_a_healthy_robot():
    assert _run_bench(_fake_robot_type()) == 0


def test_the_bench_check_catches_an_observation_that_was_never_declared():
    assert _run_bench(_fake_robot_type("Mismatch")) == 1


def test_a_connection_left_in_the_tree_is_refused_at_composition():
    with pytest.raises(TypeError, match="backs parts without being one"):
        _run_bench(_fake_robot_type("Connection"))


# Production parts with fake SDKs


def test_no_part_hook_collides_with_the_worker_group():
    import inspect

    from rlinf.robotics.parts.base import RobotPart
    from rlinf.scheduler.worker.worker_group import WorkerGroup

    taken = {
        name
        for name in dir(WorkerGroup)
        if name.startswith("_") and not name.startswith("__")
    }
    for family in (RobotPart, *RobotPart.__subclasses__()):
        ours = {
            name
            for name, value in inspect.getmembers(family, callable)
            if name.startswith("_") and not name.startswith("__")
        }
        assert not (ours & taken), (
            f"{family.__name__} defines {sorted(ours & taken)}, which the "
            "scheduler already attaches to a WorkerGroup"
        )


def test_a_real_camera_runs_against_a_faked_sdk():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.cameras.base import CameraInfo
        from rlinf.robotics.parts.cameras.realsense import RealSenseCamera

        camera = RealSenseCamera(
            CameraInfo(
                name="wrist",
                serial_number="MOCK0001",
                camera_type="realsense",
                fps=30,
                resolution=(64, 48),
            )
        )
        assert not camera.is_connected
        camera.connect()
        assert camera.is_connected

        frame = camera.get_observation()["frame"]
        assert frame.shape == (48, 64, 3)

        camera.disconnect()
        assert not camera.is_connected


def test_a_real_arm_runs_against_a_faked_sdk():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.franky import FrankyArm

        arm = FrankyArm("10.0.0.1")
        arm.connect()

        assert arm.is_connected
        # The arm backs nothing else: its end effector is composed beside it.
        assert not arm.parts and not arm.children

        observation = arm.get_observation()
        assert observation["tcp_pose"].shape == (7,)

        arm.send_action({"joint_position": [0.0] * 7})

        arm.disconnect()
        assert not arm.is_connected


def test_a_franky_arm_takes_no_end_effector_settings():
    from rlinf.robotics.parts.arms.franky import FrankyArm

    arm = FrankyArm.declare("10.0.0.1")
    assert not arm.parts

    # Settings the arm cannot honour are refused rather than dropped: the hand
    # they describe is composed beside the arm, so they belong to it.
    for unsupported in ("gripper_type", "gripper_connection", "end_effector_type"):
        with pytest.raises(TypeError, match="does not take"):
            FrankyArm.declare("10.0.0.1", **{unsupported: "whatever"})


def test_the_bench_check_runs_a_whole_robot_on_fakes():
    from robot_mocks import mocked_sdks

    from toolkits.realworld_check.check_robot_parts import check

    with mocked_sdks():
        import rlinf.robotics.robots  # noqa: F401

        assert (
            check(
                "DualFranka",
                {
                    "left_robot_ip": "10.0.0.1",
                    "right_robot_ip": "10.0.0.2",
                    "left_gripper_connection": "/dev/ttyUSB0",
                    "right_gripper_connection": "/dev/ttyUSB1",
                },
            )
            == 0
        )


def test_every_shipped_robot_runs_on_faked_sdks():
    from robot_mocks import mocked_sdks

    from toolkits.realworld_check.check_robot_parts import check

    robots = {
        "Franka": {"robot_ip": "10.0.0.1", "node_rank": 0},
        "DualFranka": {
            "left_robot_ip": "10.0.0.1",
            "right_robot_ip": "10.0.0.2",
            "left_gripper_connection": "/dev/ttyUSB0",
            "right_gripper_connection": "/dev/ttyUSB1",
        },
        "GimArm": {
            "node_rank": 0,
            "can_interface": "can0",
            "arm_variant": "gim_arm_xl",
            "enable_gripper": True,
            "gripper_type": "parallel",
            "control_mode": "momentum_observer",
            "env_idx": 0,
            "worker_rank": 0,
        },
        "Turtle2": {
            "frequency": 50,
            "camera_ids": [1],
            "env_idx": 0,
            "node_rank": 0,
            "worker_rank": 0,
        },
        "DOSW1": {
            "node_rank": 0,
            "robot_url": "localhost",
            "left_arm_port": 1,
            "right_arm_port": 2,
            "left_lead_port": 3,
            "right_lead_port": 4,
            "enable_human_in_loop": False,
            "is_dummy": False,
            "gripper_width_max": 0.08,
        },
    }
    with mocked_sdks():
        import rlinf.robotics.robots  # noqa: F401

        failed = [name for name, args in robots.items() if check(name, args) != 0]

    assert failed == []


def test_a_worker_installs_the_fakes_for_itself():
    import subprocess
    import sys

    from robot_mocks import _reach_worker_processes

    environment = {**os.environ, **_reach_worker_processes()}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, psutil;"
            "print('pyrealsense2' in sys.modules, type(psutil).__name__)",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd="/tmp",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True _Psutil"


def test_a_wrapper_that_narrows_the_action_declares_it():
    from types import SimpleNamespace

    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.wrappers.transforms import GripperCloseEnv

    inner = SimpleNamespace(
        action_parts=lambda: FrankaEnv.action_parts(SimpleNamespace(_is_hand=False))
    )
    wrapper = GripperCloseEnv.__new__(GripperCloseEnv)
    wrapper.env = SimpleNamespace(get_wrapper_attr=lambda name: getattr(inner, name))

    assert [part.name for part in wrapper.action_parts()] == ["arm"]
    assert sum(part.width for part in wrapper.action_parts()) == 6


def test_a_dual_arm_robot_is_given_its_wrist_cameras():
    import inspect

    from rlinf.robotics.robots.dual_franka import DualFrankaRobot

    accepted = set()
    for method in (DualFrankaRobot.build, DualFrankaRobot.build_arms):
        accepted |= set(inspect.signature(method).parameters)

    assert "arm_cameras" in accepted, (
        "DualFrankaEnv passes arm_cameras to build(); nothing accepts it, so "
        "wrist cameras never reach the robot"
    )


def test_two_parts_of_one_class_on_one_node_get_different_names():
    from rlinf.robotics.parts.cameras.realsense import RealSenseCamera
    from rlinf.robotics.placement import PartWorkerHost

    names = {PartWorkerHost.default_name(RealSenseCamera, 0) for _ in range(8)}

    assert len(names) == 8, f"names repeat: {sorted(names)}"
    assert all(name.startswith("RealSenseCamera-node0-") for name in names), (
        f"a name should still say what it is and where: {sorted(names)}"
    )


def test_every_real_task_is_registered_through_the_shared_factory():
    import re

    offenders = []
    for path in (_ROOT / "rlinf" / "envs" / "real").rglob("__init__.py"):
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), 1):
            if re.match(r"\s*register\(", line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == [], f"these register a task outside register_tasks: {offenders}"


def test_an_address_is_checked_by_the_arm_that_dials_it():
    from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robots.dual_franka import DualFrankaConfig
    from rlinf.robotics.robots.franka import FrankaConfig

    # Enumeration may still resolve the address from the node environment.
    assert (
        DualFrankaConfig(node_rank=0, left_robot_ip="LEFT_ROBOT_IP").left_robot_ip
        == "LEFT_ROBOT_IP"
    )
    assert FrankaConfig(node_rank=0, robot_ip="ROBOT_IP").robot_ip == "ROBOT_IP"

    for arm_cls in (FrankaROSArm, FrankyArm):
        with pytest.raises(ValueError, match="to be an IP address"):
            arm_cls("LEFT_ROBOT_IP")
        with pytest.raises(ValueError, match="needs a 'robot_ip'"):
            arm_cls("")
        assert arm_cls("10.0.0.1")._robot_ip == "10.0.0.1"


def test_a_connection_cannot_be_composed_into_a_robot():
    from rlinf.robotics.parts.arms.dosw1 import DOSW1Connection
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, RobotPart

    for cls in (Turtle2Connection, DOSW1Connection):
        assert issubclass(cls, Connection), f"{cls.__name__} must still be placeable"
        assert not issubclass(cls, RobotPart), (
            f"{cls.__name__} backs parts without being one; it must not be "
            "composable as a part"
        )
        for absent in ("get_observation", "observation_features"):
            assert not hasattr(cls, absent), (
                f"{cls.__name__}.{absent} exists, so a robot can compose it"
            )


def test_no_robot_builder_absorbs_a_setting_it_does_not_use():
    import inspect

    from rlinf.robotics.robots import (
        DOSW1Robot,
        DualFrankaRobot,
        FrankaRobot,
        GimArmRobot,
        Turtle2Robot,
    )

    # Limit the check to shipped robots because other tests register fakes.
    for robot in (
        FrankaRobot,
        DualFrankaRobot,
        GimArmRobot,
        Turtle2Robot,
        DOSW1Robot,
    ):
        # Forwarding catch-alls are valid only when a downstream API validates keys.
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            robot.build(no_such_setting=True)

        # Retired configuration objects are rejected as well.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        DOSW1Robot.build(config=object())

    # DOSW1 session placement follows the configured node.
    assert "node_rank" in inspect.signature(DOSW1Robot.build).parameters


def test_disconnect_releases_before_it_forgets_the_handle():
    from rlinf.robotics.parts.teleop.devices import TeleopPart

    class Reader:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Device(TeleopPart):
        def __init__(self):
            self.reader = Reader()

        def _open(self):
            return self.reader

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    device = Device()
    device.connect()
    device.disconnect()

    assert device.reader.closed, "the reader was left open"
    assert not device.is_connected


def test_a_camera_can_be_opened_again_after_it_is_closed():
    import numpy as np

    from rlinf.robotics.parts.cameras.base import BaseCamera, CameraInfo

    class Fake(BaseCamera):
        def __init__(self, info):
            super().__init__(info)
            self.opens = 0
            self.releases = 0

        def _open(self):
            self.opens += 1
            return object()

        def _read_frame(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def _release(self, device):
            self.releases += 1

    camera = Fake(CameraInfo(name="c", serial_number="X", camera_type="realsense"))
    camera.connect()
    camera.reopen()

    assert camera.opens == 2
    assert camera.releases == 1
    assert camera.is_connected
    assert camera.get_frame(timeout=2.0).shape == (4, 4, 3)

    camera.disconnect()
    assert camera.releases == 2


def test_a_part_that_would_break_its_worker_is_refused_before_placement():
    from rlinf.robotics.placement.handles import PartWorkerHost

    class Clashing(ControllablePart):
        @property
        def observation_features(self):
            return {}

        @property
        def action_features(self):
            return {}

        def _open(self):
            return "device"

        def get_observation(self):
            return {}

        def send_action(self, action):
            return action

        def attribute(self, name):
            """Conflict with the worker RPC reserved for property access."""

    with pytest.raises(TypeError, match="share a name with the worker"):
        PartWorkerHost.worker_class(Clashing)

        # Shipped drivers avoid worker-method name collisions.
    for driver in (FrankyArm, FrankaROSArm, GimArm):
        assert PartWorkerHost.worker_class(driver) is not None


def test_a_hosted_camera_reopens_on_the_node_that_holds_it():
    from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo
    from rlinf.robotics.placement import remote_view_of

    class Fake(BaseCamera):
        def _open(self):
            return object()

        def _read_frame(self):
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def _release(self, device):
            pass

    view = remote_view_of(Fake)
    placed = object.__new__(view)
    group = FakeWorkerGroup()
    placed._group = group
    placed._device = group

    placed.reopen()

    assert [name for name, _ in group.calls] == ["reopen"], (
        "reopen has to travel, or a stalled hosted camera can never recover"
    )
    assert "connect" not in view.__dict__ and "disconnect" not in view.__dict__

    # Local cameras use the same recovery operation directly.
    info = CameraInfo(name="c", serial_number="X", camera_type="realsense")
    assert callable(Fake(info).reopen)


def test_a_held_hand_resumes_from_the_pose_the_env_reset_it_to():
    import numpy as np

    from rlinf.robotics.teleop import GloveBinding

    configured = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    glove = GloveBinding()
    glove.reset({"hand_reset_pose": configured})

    first = glove.action({"angles": np.zeros(6)}, {}).parts["hand"]

    assert np.allclose(first, configured), (
        f"the hand starts at {first}, not the configured {configured}"
    )


def test_backed_parts_and_children_are_different_questions():
    from rlinf.robotics.parts.arms.turtle2 import Turtle2Connection
    from rlinf.robotics.parts.base import Connection, PartGroup, RobotPart

    connection = Turtle2Connection()

    assert set(connection.parts) >= {"left", "left_end_effector"}
    assert not hasattr(connection, "children"), (
        "a connection composes nothing; it only offers"
    )

    robot = PartGroup(arm=connection.part("left"))

    assert set(robot.children) == {"arm"}
    assert robot.parts == {}, "a group backs nothing of its own"
    assert isinstance(connection, Connection)
    assert not isinstance(connection, RobotPart)


def test_describe_says_where_a_part_runs_before_anything_is_opened():
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    from rlinf.robotics.parts.end_effectors import EndEffector

    robot = Bench(
        arm=FrankyArm("10.0.0.1", node_rank=2),
        end_effector=EndEffector.of("franky", robot_ip="10.0.0.1", node_rank=5),
    )

    described = robot.describe()

    assert "Bench" in described
    assert "arm" in described and "end_effector" in described
    # Each part reports the node it was placed on, before anything opens.
    assert "node=2" in described and "node=5" in described, described
    # Separate connections, so neither is listed as owned by the other.
    assert described.count("FrankyArm#1") == 1, described


def test_a_robot_can_be_disconnected_twice():
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    from rlinf.robotics.parts.end_effectors import EndEffector

    robot = Bench(
        arm=FrankyArm("10.0.0.1"),
        end_effector=EndEffector.of("robotiq", port="/dev/mock-gripper"),
    )

    with mocked_sdks():
        robot.connect()
        robot.disconnect()
        robot.disconnect()

        assert not robot.is_connected


def test_the_franka_hand_is_reachable_over_libfranka():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector, FrankyGripper

        assert EndEffector.backend("franky") is FrankyGripper
        assert EndEffector.backend("franky_gripper") is FrankyGripper
        # The ROS backend keeps the plain name; the two are different transports
        # to the same hand.
        assert EndEffector.backend("franka") is not FrankyGripper

        gripper = EndEffector.of("franky", robot_ip="10.0.0.1")
        assert isinstance(gripper, FrankyGripper)
        assert not gripper.is_connected

        # Before connecting, the nominal stroke stands in.
        assert gripper.max_width == pytest.approx(0.08)

        gripper.connect()
        assert gripper.is_connected
        assert gripper.is_ready()
        # Connecting adopts the stroke this hand reports for itself.
        assert gripper.max_width == pytest.approx(gripper._gripper.max_width)

        gripper.disconnect()
        assert not gripper.is_connected
        assert not gripper.is_ready()


def test_the_franka_hand_needs_the_arm_ip_it_hangs_from():
    from rlinf.robotics.parts.end_effectors import FrankyGripper

    with pytest.raises(ValueError, match="arm's own IP"):
        FrankyGripper.declare(port="/dev/ttyUSB0")


def test_the_franka_hand_commands_widths_in_metres():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franky", robot_ip="10.0.0.1")
        gripper.connect()
        sdk = gripper._gripper

        gripper.move(0.03)
        assert sdk.commands[-1][:2] == ("move", 0.03)
        assert gripper.position == pytest.approx(0.03)
        assert not gripper.is_open

        # Beyond the stroke clamps rather than raising.
        gripper.move(0.5)
        assert sdk.commands[-1][1] == pytest.approx(gripper.max_width)
        assert gripper.is_open

        # The end-effector interface reads and writes the same metre axis.
        assert gripper.get_state() == pytest.approx([gripper.max_width])
        gripper.command(np.asarray([0.02], dtype=np.float32))
        assert gripper.position == pytest.approx(0.02)


def test_the_franka_hand_grasps_within_the_force_it_can_apply():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franky", robot_ip="10.0.0.1")
        gripper.connect()
        sdk = gripper._gripper

        # A force written on the Robotiq scale is served at the hand's own.
        gripper.close()
        assert sdk.commands[-1][0] == "grasp"
        assert sdk.commands[-1][3] == pytest.approx(40.0)

        # A force the hand can apply is passed through.
        gripper.close(force=25.0)
        assert sdk.commands[-1][3] == pytest.approx(25.0)

        # Normalized speed maps into the hand's m/s band, never past it.
        gripper.open(speed=1.0)
        assert sdk.commands[-1] == ("open", pytest.approx(0.1))
        gripper.open(speed=0.0)
        assert sdk.commands[-1] == ("open", pytest.approx(0.01))


def test_a_refused_grasp_does_not_end_the_episode():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franky", robot_ip="10.0.0.1")
        gripper.connect()
        sdk = gripper._gripper

        def refuse(*_args, **_kwargs):
            raise RuntimeError("libfranka: command rejected")

        sdk.grasp = refuse
        # Closing on air raises in libfranka; the hand stops and moves instead.
        gripper.close()
        assert [c[0] for c in sdk.commands[-2:]] == ["stop", "move"]
        assert not gripper.is_open

        sdk.open = refuse
        gripper.open()
        assert [c[0] for c in sdk.commands[-2:]] == ["stop", "move"]
        assert gripper.is_open


def test_a_robot_composes_the_hand_its_config_names():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import (
            FrankaGripper,
            FrankyGripper,
            RobotiqGripper,
        )
        from rlinf.robotics.robots import FrankaRobot

        def hand_of(**settings):
            return FrankaRobot.declare_end_effector(
                "10.0.0.1", node_rank=0, name="hand", **settings
            )

        # The built-in hand is one device with two drivers, and the arm backend
        # the robot is built on decides which of them reaches it.
        assert isinstance(hand_of(gripper_type="franka"), FrankaGripper)
        assert isinstance(
            hand_of(gripper_type="franka", backend="franky"), FrankyGripper
        )
        # A config that names a driver outright is taken at its word.
        assert isinstance(hand_of(end_effector_type="franky_gripper"), FrankyGripper)
        assert isinstance(
            hand_of(gripper_type="robotiq", gripper_connection="/dev/ttyUSB0"),
            RobotiqGripper,
        )


def test_a_franka_robot_composes_the_backend_and_hand_it_is_given():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import FrankyGripper
        from rlinf.robotics.robots import FrankaRobot

        robot = FrankaRobot.build(
            robot_ip="10.0.0.1",
            node_rank=0,
            backend="franky",
            gripper_type="franka",
        )
        assert type(robot.child("arm")).__name__ == "FrankyArm"
        assert isinstance(robot.child("end_effector"), FrankyGripper)

        robot.connect()
        assert set(robot.get_observation()) == {"arm", "end_effector"}
        assert robot.child("end_effector").is_connected
        robot.disconnect()
        assert not robot.child("end_effector").is_connected


def test_reading_the_hand_twice_costs_one_round_trip():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector

        gripper = EndEffector.of("franky", robot_ip="10.0.0.1")
        gripper.connect()

        reads = []
        sdk = gripper._gripper
        sdk._width = sdk.width
        type(sdk).width = property(
            lambda self: (reads.append(1), self._width)[1],
            lambda self, value: setattr(self, "_width", value),
        )

        # An observation asks for both, and libfranka cannot produce a newer
        # width within one poll period.
        _, _ = gripper.position, gripper.is_open
        assert len(reads) == 1

        # Commanding the fingers means the cached width no longer describes them.
        gripper.move(0.02)
        _ = gripper.position
        assert len(reads) == 2


def test_one_part_at_a_time_holds_a_hardware_endpoint(tmp_path, monkeypatch):
    from rlinf.robotics.parts import claims
    from rlinf.robotics.parts.claims import DeviceClaim

    monkeypatch.setattr(claims, "_CLAIM_DIR", str(tmp_path))

    held = DeviceClaim("franky-arm:10.0.0.1", "FrankyArm")
    held.acquire()

    # A second part reaching for the same endpoint is told who has it, rather
    # than being left to fail later inside a vendor SDK.
    with pytest.raises(RuntimeError, match="FrankyArm"):
        DeviceClaim("franky-arm:10.0.0.1", "OtherArm").acquire()

    # An arm and the hand mounted on it answer on different endpoints, so
    # holding one says nothing about the other.
    beside = DeviceClaim("franky-hand:10.0.0.1", "FrankyGripper")
    beside.acquire()
    beside.release()

    held.release()
    # Releasing hands the endpoint on.
    DeviceClaim("franky-arm:10.0.0.1", "OtherArm").acquire()


def test_a_claim_survives_a_part_that_fails_to_open(tmp_path, monkeypatch):
    from rlinf.robotics.parts import claims
    from rlinf.robotics.parts.claims import DeviceClaim

    monkeypatch.setattr(claims, "_CLAIM_DIR", str(tmp_path))

    claim = DeviceClaim("robotiq:/dev/ttyUSB0", "RobotiqGripper")
    try:
        with claim:
            raise RuntimeError("the port was there but the gripper was not")
    except RuntimeError:
        pass

    # A failed open must not strand the endpoint for the rest of the session.
    DeviceClaim("robotiq:/dev/ttyUSB0", "RobotiqGripper").acquire()


def test_ros_parts_share_one_session_per_process():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.transports.ros import ROSController

        ROSController._shared = None
        try:
            first = ROSController.shared()
            # ROS 1 gives a process one node, so asking twice has to answer
            # with the session that node already belongs to.
            assert ROSController.shared() is first
        finally:
            ROSController._shared = None


def test_a_ros_hand_opens_without_an_arm_to_hand_it_a_session():
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors import EndEffector, FrankaGripper

        hand = EndEffector.of("franka_gripper")
        assert isinstance(hand, FrankaGripper)

        # No arm involved: the hand joins the session itself.
        hand.connect()
        try:
            assert hand.is_connected
            assert hand._ros is not None
        finally:
            hand.disconnect()
        assert not hand.is_connected
