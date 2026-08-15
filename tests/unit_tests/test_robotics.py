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

"""The robotics layer: parts, composition, placement, and its boundaries."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pytest

import rlinf.robotics.robots.franka as franka_module
from rlinf.robotics import (
    Camera,
    ControllablePart,
    DOSW1Robot,
    DOSW1RobotConfig,
    DualFrankaRobot,
    EndEffector,
    FrankaRobot,
    GimArmConfig,
    Group,
    LegacyObservationAdapter,
    MethodArm,
    MethodGripper,
    RemotePartHandle,
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
    Turtle2Hardware,
)
from rlinf.robotics.parts.arms.franka import FrankaRobotState
from rlinf.robotics.specs import PartSpec
from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareResource,
    NodeHardwareConfig,
)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --- from test_robotics.py --------------------------------------------


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


class FakeEndEffector(FakePart, EndEffector):
    @property
    def action_features(self) -> dict[str, dict]:
        return {"target": {"shape": (1,)}}

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return action


class FakeCamera(FakePart, Camera):
    pass


class FakeRemoteResult:
    def __init__(self, value: Any):
        self.value = value

    def wait(self) -> list[Any]:
        return [self.value]


class FakeMethodDriver:
    """A driver whose components are reachable only through named methods."""

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


class FakeWorkerGroup:
    """Stand-in for the one-worker group behind a RemotePartHandle."""

    def __init__(self):
        self.calls: list[tuple[str, Any]] = []

    def shutdown(self) -> FakeRemoteResult:
        self.calls.append(("shutdown", None))
        return FakeRemoteResult(None)

    def part_observation(self, name: str) -> FakeRemoteResult:
        self.calls.append(("part_observation", name))
        return FakeRemoteResult({"tcp_pose": np.zeros(7)})

    def part_action(self, name: str, action: Any) -> FakeRemoteResult:
        self.calls.append(("part_action", (name, action)))
        return FakeRemoteResult(action)

    def is_robot_up(self) -> FakeRemoteResult:
        self.calls.append(("is_robot_up", None))
        return FakeRemoteResult(True)

    def _close(self) -> None:
        self.calls.append(("_close", None))


def test_robot_composes_and_namespaces_parts():
    events: list[str] = []
    arm = Group(
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
    # The observation mirrors the composition: names, not fixed categories.
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
    arm = Group(arm=FakeControllablePart("driver", events), wrist=camera)
    robot = Robot(arm=arm)
    robot.connect()
    camera.disconnect()

    robot.disconnect()

    assert "disconnect:driver" in events


def test_driver_views_expose_composed_part_api():
    """A method-shaped driver decomposes into parts without any remoting."""
    driver = FakeMethodDriver()
    arm = MethodArm(
        driver,
        commands={"tcp_pose": "move_arm"},
        state_fields=("tcp_pose", "arm_joint_position"),
    )
    end_effector = MethodGripper(driver, state_field="gripper_position")
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


def test_remote_handle_releases_its_worker_group():
    group = FakeWorkerGroup()
    handle = RemotePartHandle(
        group,
        {"arm": {"kind": "controllable", "observation": {}, "action": {}}},
    )

    handle.disconnect()
    handle.disconnect()  # idempotent

    assert [name for name, _ in group.calls] == ["shutdown", "_close"]


def test_remote_handle_forwards_off_interface_driver_methods():
    """Methods outside the part interface reach the driver unchanged."""
    group = FakeWorkerGroup()
    handle = RemotePartHandle(group, {})

    assert handle.is_robot_up().wait()[0] is True


def test_robot_requires_non_empty_string_part_names():
    with pytest.raises(ValueError, match="non-empty strings"):
        Robot(parts={0: FakePart("camera", [])})  # type: ignore[dict-item]


def test_builtin_robots_expose_standard_composition_layouts():
    events: list[str] = []
    left_arm = Group(
        arm=FakeControllablePart("left_arm", events),
        gripper=FakeEndEffector("left_gripper", events),
    )
    right_arm = Group(
        arm=FakeControllablePart("right_arm", events),
        gripper=FakeEndEffector("right_gripper", events),
    )
    third_arm = Group(arm=FakeControllablePart("third_arm", events))

    single = FrankaRobot(arm=left_arm, front_camera=FakeCamera("front", events))
    dual = DualFrankaRobot(
        left=left_arm, right=right_arm, base_camera=FakeCamera("base", events)
    )
    # Names are the composition, so nothing caps the count or the kind.
    triple = FrankaRobot(left=left_arm, right=right_arm, third=third_arm)

    assert set(single.parts) == {"arm", "front_camera"}
    assert set(single.parts_of_type(Group)) == {"arm"}
    assert set(single.parts_of_type(EndEffector)) == {"arm.gripper"}
    assert set(single.parts_of_type(Camera)) == {"front_camera"}
    assert set(dual.parts) == {"left", "right", "base_camera"}
    assert set(dual.parts_of_type(Group)) == {"left", "right"}
    assert set(triple.parts) == {"left", "right", "third"}
    assert set(triple.parts_of_type(Group)) == {"left", "right", "third"}


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
        endpoint: str = "loopback"

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
        configs=cast(Any, [{"node_rank": 3, "endpoint": "robot.local"}]),
    )

    assert registered is TestRobotDiscovery
    assert RobotDiscovery.registry["TestRobot"].robot_cls is TestRobot
    assert TestRobotDiscovery in Hardware.policy_registry
    assert isinstance(parsed.configs[0], TestRobotConfig)
    assert parsed.configs[0].endpoint == "robot.local"

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
    """Observation and action keys are the names the robot was composed with."""
    robot = Robot(arm=Group(arm=FakeControllablePart("arm", [])))
    robot.connect()

    observation = robot.get_observation()
    action = {"arm": {"arm": {"target": np.array([0.25])}}}

    assert observation["arm"]["arm"]["state"].shape == (1,)
    assert robot.send_action(action) == action
    robot.disconnect()


def test_robot_releases_driver_handles_after_parts():
    """Parts borrow a connection; the robot closes it once they are done."""
    events: list[str] = []

    class FakeHandle:
        def disconnect(self) -> None:
            events.append("handle")

    robot = Robot(arm=Group(arm=FakeControllablePart("driver", events)))
    robot.handles["arm"] = FakeHandle()
    robot.connect()
    robot.disconnect()

    assert events[-1] == "handle"
    assert "disconnect:driver" in events


def test_driver_rejects_actions_for_observation_only_parts():
    class CameraOnlyHost(FakePart):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"wrist": FakeCamera("wrist", [])}

    with pytest.raises(TypeError, match="not controllable"):
        CameraOnlyHost("host", []).part_action("wrist", {})


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
    """Every built-in robot config is usable with only its placement set."""
    configs = [
        GimArmConfig(node_rank=0),
        DOSW1RobotConfig(node_rank=0),
        Turtle2Config(node_rank=0),
    ]

    assert all(config.node_rank == 0 for config in configs)


def test_every_registered_robot_carries_a_builder():
    """Registration is the single source of truth for a robot type.

    A robot module registers its config, robot, discovery, and builder in one
    call, so composing by type name never needs a central dispatch table.
    """
    registry = RobotDiscovery.registry

    assert set(registry) >= {"Franka", "DualFranka", "GimArm", "Turtle2", "DOSW1"}
    missing = sorted(name for name, reg in registry.items() if reg.build is None)
    assert missing == []


def test_dosw1_dummy_runtime_uses_composed_dual_arm_interface():
    @dataclass
    class DummyDOSW1Config:
        robot_url: str = "localhost"
        left_arm_port: int = 50051
        right_arm_port: int = 50053
        left_lead_port: int = 50050
        right_lead_port: int = 50052
        enable_human_in_loop: bool = False
        is_dummy: bool = True
        gripper_width_max: float = 0.07

    robot = DOSW1Robot.build(config=DummyDOSW1Config())
    robot.connect()

    assert set(robot.parts) == {"left", "right"}
    observation = robot.get_observation()
    assert observation["left"]["arm"]["joint_position"].shape == (6,)
    robot.disconnect()
    assert not robot.is_connected


def test_pure_drivers_construct_without_scheduler_or_vendor_sdks():
    drivers = [
        FrankaROSArm("10.0.0.1"),
        FrankyArm("10.0.0.1"),
        GimArm("can0", "gim_arm_xl", True, "parallel"),
        Turtle2Hardware(),
    ]

    assert all(isinstance(driver, RobotPart) for driver in drivers)
    assert all(isinstance(driver, ControllablePart) for driver in drivers)
    assert all(not driver.is_connected for driver in drivers)
    # Each declares the parts riding on its connection.
    assert all(driver.parts for driver in drivers)


def _fake_arm_backend(monkeypatch, *, failing_ip=None, disconnected=None):
    """Point FrankaRobot at a fake arm class that records what it places."""

    class FakeHandle:
        def __init__(self, name: str):
            self.name = name
            self.parts = {"arm": FakeControllablePart(name, [])}

        def part_named(self, _name: str):
            return self.parts["arm"]

        def disconnect(self) -> None:
            if disconnected is not None:
                disconnected.append(self.name)

    class FakeArm:
        @classmethod
        def at(cls, *args, node_rank=None, name=None, **kwargs):
            return PartSpec(cls, args, kwargs, node_rank=node_rank, name=name)

        @staticmethod
        def spawn(robot_ip, *args, node_rank=None, name=None, **kwargs):
            if failing_ip is not None and robot_ip == failing_ip:
                raise RuntimeError("right arm is unreachable")
            return FakeHandle(robot_ip)

    monkeypatch.setattr(franka_module, "franka_arm_cls", lambda backend: FakeArm)
    return FakeArm


def test_declaring_arms_places_nothing_until_connect(monkeypatch):
    """Declarations are inert. ``connect`` is what touches hardware."""

    class NeverSpawns:
        @classmethod
        def at(cls, *args, node_rank=None, name=None, **kwargs):
            return PartSpec(cls, args, kwargs, node_rank=node_rank, name=name)

        @staticmethod
        def spawn(*args, **kwargs):
            raise AssertionError("spawn must not run while declaring")

    monkeypatch.setattr(franka_module, "franka_arm_cls", lambda backend: NeverSpawns)

    robot = FrankaRobot(
        arm=FrankaRobot.declare_arm("10.0.0.1", node_rank=0, name="left")
    )

    assert not robot.is_connected


def test_connect_tears_down_parts_already_placed(monkeypatch):
    """A half-placed robot is never left behind when a later part fails."""
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
    """Nothing in declaration or placement is specific to one or two arms."""
    _fake_arm_backend(monkeypatch)

    robot = FrankaRobot(
        **{
            name: FrankaRobot.declare_arm(f"10.0.0.{index}", node_rank=0, name=name)
            for index, name in enumerate(("left", "right", "third"), start=1)
        }
    )
    robot.connect()

    assert list(robot.parts) == ["left", "right", "third"]
    assert robot.is_connected


def test_one_declaration_is_placed_once_however_often_referenced():
    """A connection backing several components opens exactly once."""
    placements: list[str] = []

    class FakeHandle:
        def __init__(self):
            self.parts = {
                "left": FakeControllablePart("left", []),
                "right": FakeControllablePart("right", []),
                "wrist": FakeCamera("wrist", []),
            }

        def part_named(self, name):
            return self.parts[name]

        def disconnect(self):
            pass

    class CoupledHardware:
        @classmethod
        def at(cls, *args, node_rank=None, name=None, **kwargs):
            return PartSpec(cls, args, kwargs, node_rank=node_rank, name=name)

        @staticmethod
        def spawn(*args, **kwargs):
            placements.append("placed")
            return FakeHandle()

    hardware = CoupledHardware.at(node_rank=0)
    robot = Robot(
        left=hardware.part("left"),
        right=hardware.part("right"),
        wrist=hardware.part("wrist"),
    )
    robot.connect()

    assert placements == ["placed"], "the shared connection was opened more than once"
    assert isinstance(robot.part("wrist"), Camera)
    assert robot.is_connected


def test_local_handle_subpart_returns_a_part_not_a_forwarded_call():
    """The handle's accessors must win over its catch-all forwarding.

    ``LocalPartHandle.__getattr__`` forwards unknown names to the hosted part
    and wraps the result for call-shape symmetry. If an accessor like
    ``subpart`` were missing from the handle, that forwarding would silently
    return a result wrapper instead of a part, and the failure would surface
    far away as a missing attribute on the composed arm.
    """
    events: list[str] = []

    class HostWithSubparts(FakeControllablePart):
        @property
        def parts(self) -> dict[str, RobotPart]:
            return {"arm": self, "end_effector": FakeEndEffector("ee", events)}

    handle = HostWithSubparts.spawn("host", events)

    assert isinstance(handle.part_named("arm"), RobotPart)
    assert isinstance(handle.part_named("end_effector"), EndEffector)
    assert set(handle.parts) == {"arm", "end_effector"}
    # Off-interface names still forward, and still wrap.
    assert handle.get_observation().wait()[0]["state"].shape == (1,)


def test_any_part_can_be_placed_not_only_arms():
    """Placement is a property of parts, so a camera can be spawned alone."""
    events: list[str] = []
    handle = FakeCamera.spawn("wrist", events)

    assert handle.part.is_connected
    assert "connect:wrist" in events
    handle.disconnect()
    assert "disconnect:wrist" in events


def test_every_robot_owns_its_construction():
    """Construction and registration are the robot class's own behaviour.

    ``build`` must be a classmethod bound to the registered class, not a loose
    module function handed to the registry, so a subclass overriding it is what
    ``build_robot`` dispatches to.
    """
    registry = RobotDiscovery.registry

    for name, registration in registry.items():
        build = registration.build
        assert build is not None, f"{name} registered no builder"
        assert getattr(build, "__self__", None) is registration.robot_cls, (
            f"{name}'s builder is not bound to {registration.robot_cls.__name__}"
        )


def test_dual_franka_inherits_declaration_from_franka():
    """Arm count and backend are the only differences between the two."""
    assert issubclass(DualFrankaRobot, FrankaRobot)
    # declare_arm is inherited; only build_arms differs.
    assert DualFrankaRobot.declare_arm.__func__ is FrankaRobot.declare_arm.__func__
    assert DualFrankaRobot.build_arms.__func__ is not FrankaRobot.build_arms.__func__, (
        "only the arm count differs, and that is what build_arms says"
    )
    # Switching the control backend is one class attribute, and it serves any
    # arm count because declare_arm is shared.
    assert (FrankaRobot.BACKEND, DualFrankaRobot.BACKEND) == ("franka_ros", "franky")
    # build_arms carries the entire difference; everything else is inherited.
    overridden = [
        name
        for name in ("declare_arm", "build_arms", "build_cameras", "build")
        if getattr(DualFrankaRobot, name).__func__
        is not getattr(FrankaRobot, name).__func__
    ]
    assert overridden == ["build_arms"]


def test_every_part_kind_places_independently():
    """Arm, end effector, and camera are separate parts, each with its own node.

    A Robotiq gripper is a serial device of its own and a camera holds its own
    USB link, so neither has to ride the arm's connection or its node.
    """
    placed: dict[str, int] = {}

    class Handle:
        parts: dict[str, RobotPart] = {}

        def __init__(self, part):
            self._part = part

        @property
        def part(self):
            return self._part

        def part_named(self, name):
            raise KeyError(name)

        def disconnect(self):
            pass

    def fake(kind, base):
        class Fake(base):
            def __init__(self, *args, **kwargs):
                self._connected = False

            @property
            def is_connected(self):
                return self._connected

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            def connect(self):
                self._connected = True

            def disconnect(self):
                self._connected = False

            def get_observation(self):
                return {}

            def send_action(self, action):
                return action

            @classmethod
            def spawn(cls, *args, node_rank=None, name=None, **kwargs):
                placed[kind] = node_rank
                part = cls()
                part.connect()
                return Handle(part)

        return Fake

    robot = Robot(
        **{
            "arm": Group(
                arm=fake("arm", ControllablePart).at("10.0.0.1", node_rank=1),
                gripper=fake("gripper", EndEffector).at(
                    port="/dev/ttyUSB0", node_rank=2
                ),
                wrist=fake("camera", Camera).at(node_rank=3),
            )
        }
    )
    robot.connect()

    assert placed == {"arm": 1, "gripper": 2, "camera": 3}
    arm = robot.part("arm")
    assert isinstance(arm.part("gripper"), EndEffector)
    assert isinstance(arm.part("wrist"), Camera)
    assert robot.is_connected


def test_a_leaf_part_placed_remotely_still_exposes_itself():
    """A camera has no parts, so its handle must proxy the part itself.

    Without this, declaring a camera on its own node would resolve to nothing.
    """

    class Leaf(Camera):
        @property
        def is_connected(self):
            return True

        @property
        def observation_features(self):
            return {"frame": {}}

        def connect(self):
            pass

        def disconnect(self):
            pass

        def get_observation(self):
            return {"frame": None}

    described = Leaf().describe_self()

    assert described["kind"] == "camera"
    assert described["observation"] == {"frame": {}}
    assert Leaf().parts == {}


def test_declaring_cameras_needs_no_config_class():
    """A camera descriptor plus a node is all a declaration needs."""
    from rlinf.robotics.parts.cameras import CameraInfo, declare_cameras

    info = CameraInfo(name="scene", serial_number="123", camera_type="realsense")
    declared = declare_cameras({"scene": info}, node_rank=4)

    assert set(declared) == {"scene"}
    assert declared["scene"].node_rank == 4
    assert declared["scene"].part_cls.__name__ == "RealSenseCamera"
    assert declare_cameras(None) == {}


def test_failed_connect_can_be_retried():
    """A failed connect restores declarations instead of poisoning the robot.

    Without this, an arm that placed successfully would keep a part whose
    handle was released during rollback, and a retry would place nothing.
    """
    placed: list[str] = []
    state = {"failing": True}

    class Handle:
        parts: dict = {}

        def __init__(self, part):
            self._part = part

        @property
        def part(self):
            return self._part

        def part_named(self, name):
            raise KeyError(name)

        def disconnect(self):
            pass

    def maker(tag, flaky=False):
        class Made(ControllablePart):
            def __init__(self, *args, **kwargs):
                self._connected = False

            @property
            def is_connected(self):
                return self._connected

            @property
            def observation_features(self):
                return {}

            @property
            def action_features(self):
                return {}

            def connect(self):
                self._connected = True

            def disconnect(self):
                self._connected = False

            def get_observation(self):
                return {}

            def send_action(self, action):
                return action

            @classmethod
            def spawn(cls, *args, node_rank=None, name=None, **kwargs):
                if flaky and state["failing"]:
                    raise RuntimeError("hardware unreachable")
                placed.append(tag)
                part = cls()
                part.connect()
                return Handle(part)

        return Made

    robot = Robot(
        **{
            "left": maker("left").at(node_rank=1),
            "right": maker("right", flaky=True).at(node_rank=2),
        }
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        robot.connect()

    assert robot.handles == {}, "handles from the aborted attempt were kept"
    assert isinstance(robot.part("left"), PartSpec), (
        "the successful arm kept a part whose handle was released"
    )

    state["failing"] = False
    robot.connect()

    assert placed == ["left", "left", "right"], "the retry did not re-place"
    assert robot.is_connected

    robot.disconnect()
    assert isinstance(robot.part("left"), PartSpec)
    robot.connect()
    assert robot.is_connected, "a disconnected robot must be connectable again"


def test_a_robot_owned_camera_is_placed_opened_and_closed():
    """Cameras follow the same lifecycle as every other part.

    They used to be built by the environment and bolted on after connect, which
    meant a camera could never sit on the node it was plugged into.
    """
    events: list[str] = []

    class Cam(Camera):
        def __init__(self, *args, **kwargs):
            self._connected = False

        @property
        def is_connected(self):
            return self._connected

        @property
        def observation_features(self):
            return {"frame": {}}

        def connect(self):
            self._connected = True
            events.append("open")

        def disconnect(self):
            self._connected = False
            events.append("close")

        def get_observation(self):
            return {"frame": None}

        @classmethod
        def spawn(cls, *args, node_rank=None, name=None, **kwargs):
            events.append(f"placed@{node_rank}")
            part = cls()

            class Handle:
                parts: dict = {}

                @property
                def part(self):
                    return part

                def part_named(self, name):
                    raise KeyError(name)

                def disconnect(self):
                    events.append("released")

            return Handle()

    robot = Robot(
        arm=Group(arm=FakeControllablePart("arm", []), wrist=Cam.at(node_rank=5))
    )
    robot.connect()
    robot.disconnect()

    assert events == ["placed@5", "open", "close", "released"]


# --- from test_robotics_boundaries.py ---------------------------------


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_scheduler_has_no_robotics_dependency():
    scheduler_dir = _ROOT / "rlinf" / "scheduler"
    offenders = {
        path.relative_to(_ROOT): module
        for path in scheduler_dir.rglob("*.py")
        for module in _imports(path)
        if module == "rlinf.robotics" or module.startswith("rlinf.robotics.")
    }

    assert offenders == {}


def test_pure_driver_import_does_not_load_scheduler():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from rlinf.robotics.parts.arms.franky import FrankyArm; "
                "assert 'rlinf.scheduler' not in sys.modules; "
                "assert not FrankyArm('10.0.0.1').is_connected"
            ),
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


_SCHEDULER_BRIDGE = Path("rlinf") / "robotics" / "placement.py"


def test_robotics_devices_do_not_depend_on_scheduler_ray_or_gym():
    robotics_dir = _ROOT / "rlinf" / "robotics"
    device_paths = [
        robotics_dir / "robot.py",
        robotics_dir / "adapters.py",
        robotics_dir / "views.py",
        *robotics_dir.joinpath("parts").rglob("*.py"),
    ]
    forbidden = ("ray", "gymnasium", "rlinf.scheduler")
    offenders = {
        path.relative_to(_ROOT): module
        for path in device_paths
        if path.relative_to(_ROOT) != _SCHEDULER_BRIDGE
        for module in _imports(path)
        if module == forbidden or module.startswith(forbidden)
    }

    assert offenders == {}


def test_scheduler_use_is_confined_to_the_composition_layer():
    """Only composition code may see the scheduler; hardware code never does.

    ``rlinf.robotics`` is two layers. Hardware -- parts: arms, cameras, end
    effectors, teleop devices -- is scheduler-free so it runs from plain
    scripts.
    Composition -- the placement bridge, robot builders, and hardware discovery
    -- is allowed to use the scheduler. This pins the boundary between them, so
    a new driver cannot quietly reach for ``Cluster``.
    """
    robotics_dir = _ROOT / "rlinf" / "robotics"
    allowed = {
        _SCHEDULER_BRIDGE,
        Path("rlinf") / "robotics" / "discovery.py",
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


def test_moved_env_modules_still_import_under_their_old_paths():
    """The pre-split module paths keep working, and resolve to one module.

    The alias has to precede the path finder on ``sys.meta_path``: an aliased
    package's ``__path__`` points into the new directory, so a later finder
    would load a second copy of each submodule under the old name and identity
    checks would silently fail.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore::DeprecationWarning",
            "-c",
            (
                "from rlinf.envs.realworld.robot_task_env import RobotTaskEnv as old; "
                "from rlinf.envs.real.robot_task_env import RobotTaskEnv as new; "
                "assert old is new, 'alias loaded a second copy'; "
                "import rlinf.envs.realworld as p, rlinf.envs.real as n; "
                "assert p is n; "
                "import rlinf.envs.utils"  # a name that did not move
            ),
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_teleop_devices_live_with_the_envs_that_read_them():
    """A teleop device reads the operator, not the robot, so it is not a part.

    No policy ever observes a leader arm or a glove, and no ``Robot`` composes
    one, so these modules belong beside the wrappers that turn their output into
    an intervention rather than under ``robotics/parts``.
    """
    devices_dir = _ROOT / "rlinf" / "envs" / "real" / "teleop" / "devices"
    modules = {path.stem for path in devices_dir.glob("*.py")} - {"__init__"}

    assert modules == {
        "gello",
        "gello_joint",
        "glove",
        "keyboard",
        "pico",
        "spacemouse",
    }
    assert not (_ROOT / "rlinf" / "robotics" / "parts" / "teleop").exists()


def test_teleop_device_readers_do_not_import_gymnasium():
    """A reader only talks to hardware, so a bench script can drive one directly.

    Turning a reading into an action is the adapters' job, and those may import
    Gymnasium freely; this holds the line one level down, at the readers.
    """
    devices_dir = _ROOT / "rlinf" / "envs" / "real" / "teleop" / "devices"
    offenders = {
        path.name
        for path in devices_dir.glob("*.py")
        if re.search(r"^\s*(import|from)\s+gymnasium\b", path.read_text(), re.M)
    }

    assert offenders == set()


def test_importing_a_teleop_device_does_not_load_the_env_stack():
    """A bench script driving one serial device should not need the env stack.

    ``toolkits/realworld_check`` runs on the machine the device is plugged into,
    which may have no Gymnasium, no OpenCV, and no cluster. Python executes every
    parent package on the way to a submodule, so this holds
    ``rlinf.envs.real.__init__`` to lazy loading.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import rlinf.envs.real.teleop.devices.gello\n"
            "leaked = {m.split('.')[0] for m in sys.modules} & {'gymnasium', 'cv2', 'ray'}\n"
            "assert not leaked, sorted(leaked)\n",
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
