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

import rlinf.robotics.robots.franka as franka_module
from rlinf.robotics import (
    Arm,
    Camera,
    ControllablePart,
    DOSW1RobotConfig,
    DriverArm,
    DriverGripper,
    DualFrankaConfig,
    DualFrankaRobot,
    EndEffector,
    FrankaArmConfig,
    FrankaConfig,
    FrankaRobot,
    GimArmConfig,
    LegacyObservationAdapter,
    RemoteDriverHandle,
    Robot,
    RobotAutoConfig,
    RobotConfig,
    RobotDiscovery,
    RobotPart,
    Turtle2Config,
    VectorActionAdapter,
    VectorActionBinding,
    build_dosw1_robot,
    register_robot,
)
from rlinf.robotics.drivers import (
    FrankaROSDriver,
    FrankyDriver,
    GimArmDriver,
    Turtle2Driver,
)
from rlinf.robotics.drivers.base import Driver
from rlinf.robotics.robots.franka import place_franka_arms
from rlinf.robotics.states import FrankaRobotState
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
    """Stand-in for the one-worker group behind a RemoteDriverHandle."""

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
    arm = Arm(
        driver=FakeControllablePart("arm", events),
        end_effector=FakeEndEffector("gripper", events),
        cameras={"wrist": FakeCamera("wrist", events)},
    )
    robot = Robot(
        arms={"left": arm},
        cameras={"front": FakeCamera("front", events)},
    )

    robot.connect()

    assert robot.is_connected
    assert events == [
        "connect:arm",
        "connect:gripper",
        "connect:wrist",
        "connect:front",
    ]
    assert set(robot.observation_features) == {"arms", "cameras"}
    assert set(robot.action_features) == {"arms"}
    action = {
        "arms": {
            "left": {
                "arm": {"target": np.array([0.5])},
                "end_effector": {"target": np.array([1.0])},
            }
        }
    }
    assert robot.send_action(action) == action
    assert set(robot.parts_of_type(Camera)) == {
        "arms.left.cameras.wrist",
        "cameras.front",
    }

    robot.disconnect()
    assert events[-4:] == [
        "disconnect:front",
        "disconnect:wrist",
        "disconnect:gripper",
        "disconnect:arm",
    ]


def test_robot_rejects_actions_for_observation_only_parts():
    robot = Robot(parts={"camera": FakeCamera("camera", [])})

    with pytest.raises(TypeError, match="not controllable"):
        robot.send_action({"parts": {"camera": {"target": np.array([0.5])}}})


def test_robot_disconnects_remaining_arm_parts_after_camera_failure():
    events: list[str] = []
    camera = FakeCamera("wrist", events)
    arm = Arm(FakeControllablePart("driver", events), cameras={"wrist": camera})
    robot = Robot.single_arm(arm)
    robot.connect()
    camera.disconnect()

    robot.disconnect()

    assert "disconnect:driver" in events


def test_driver_views_expose_composed_part_api():
    """A method-shaped driver decomposes into parts without any remoting."""
    driver = FakeMethodDriver()
    arm = DriverArm(
        driver,
        commands={"tcp_pose": "move_arm"},
        state_fields=("tcp_pose", "arm_joint_position"),
    )
    end_effector = DriverGripper(driver, state_field="gripper_position")
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
    handle = RemoteDriverHandle(
        group,
        {"arm": {"kind": "controllable", "observation": {}, "action": {}}},
    )

    handle.disconnect()
    handle.disconnect()  # idempotent

    assert [name for name, _ in group.calls] == ["shutdown", "_close"]


def test_remote_handle_forwards_off_interface_driver_methods():
    """Methods outside the part interface reach the driver unchanged."""
    group = FakeWorkerGroup()
    handle = RemoteDriverHandle(group, {})

    assert handle.is_robot_up().wait()[0] is True


def test_robot_requires_non_empty_string_part_names():
    with pytest.raises(ValueError, match="non-empty strings"):
        Robot(parts={0: FakePart("camera", [])})  # type: ignore[dict-item]


def test_builtin_robots_expose_standard_composition_layouts():
    events: list[str] = []
    left_arm = Arm(
        driver=FakeControllablePart("left_arm", events),
        end_effector=FakeEndEffector("left_gripper", events),
    )
    right_arm = Arm(
        driver=FakeControllablePart("right_arm", events),
        end_effector=FakeEndEffector("right_gripper", events),
    )
    single = FrankaRobot.single_arm(
        arm=left_arm,
        cameras={"front_camera": FakeCamera("front_camera", events)},
    )
    dual = DualFrankaRobot.dual_arm(
        left_arm=left_arm,
        right_arm=right_arm,
        cameras={"base_camera": FakeCamera("base_camera", events)},
    )

    assert set(single.arms) == {"arm"}
    assert set(single.parts_of_type(Arm)) == {"arms.arm"}
    assert set(single.parts_of_type(EndEffector)) == {"arms.arm.end_effector"}
    assert set(single.parts_of_type(Camera)) == {"cameras.front_camera"}
    assert set(dual.arms) == {"left", "right"}
    assert set(dual.parts_of_type(Arm)) == {"arms.left", "arms.right"}


def test_standard_layout_rejects_non_arm_driver():
    with pytest.raises(TypeError, match="Invalid robot arms"):
        Robot(arms={"arm": FakeCamera("camera", [])})  # type: ignore[dict-item]


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


def test_robot_preserves_canonical_namespaces():
    """Robot alone carries the namespaces the old RobotRuntime wrapped."""
    arm = Arm(driver=FakeControllablePart("arm", []))
    robot = Robot.single_arm(arm)
    robot.connect()

    observation = robot.get_observation()
    action = {"arms": {"arm": {"arm": {"target": np.array([0.25])}}}}

    assert observation["arms"]["arm"]["state"]["state"].shape == (1,)
    assert robot.send_action(action) == action
    robot.disconnect()


def test_robot_releases_driver_handles_after_parts():
    """Parts borrow a connection; the robot closes it once they are done."""
    events: list[str] = []

    class FakeHandle:
        def disconnect(self) -> None:
            events.append("handle")

    arm = Arm(driver=FakeControllablePart("driver", events))
    robot = Robot.single_arm(arm, drivers={"arm": FakeHandle()})
    robot.connect()
    robot.disconnect()

    assert events[-1] == "handle"
    assert "disconnect:driver" in events


def test_driver_rejects_actions_for_observation_only_parts():
    class CameraOnlyDriver(Driver):
        is_connected = True

        def connect(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

        def parts(self) -> dict[str, RobotPart]:
            return {"wrist": FakeCamera("wrist", [])}

    with pytest.raises(TypeError, match="not controllable"):
        CameraOnlyDriver().part_action("wrist", {})


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

    robot = build_dosw1_robot(DummyDOSW1Config())

    assert set(robot.arms) == {"left", "right"}
    observation = robot.get_observation()
    assert observation["arms"]["left"]["state"]["joint_position"].shape == (6,)
    robot.disconnect()
    assert not robot.is_connected


def test_pure_drivers_construct_without_scheduler_or_vendor_sdks():
    drivers = [
        FrankaROSDriver("10.0.0.1"),
        FrankyDriver("10.0.0.1"),
        GimArmDriver("can0", "gim_arm_xl", True, "parallel"),
        Turtle2Driver(),
    ]

    assert all(isinstance(driver, Driver) for driver in drivers)
    assert all(isinstance(driver, ControllablePart) for driver in drivers)
    assert all(not driver.is_connected for driver in drivers)
    # Every driver declares the parts it backs.
    assert all(driver.parts() for driver in drivers)


def test_single_and_dual_franka_configs_project_onto_one_arm_shape():
    """Arm count is the size of a mapping, not a difference in robot type.

    Both configs keep their existing flat YAML keys; ``arms()`` is where those
    keys stop being hand-written halves and become a uniform mapping that one
    builder iterates.
    """
    single = FrankaConfig(node_rank=0, robot_ip="10.0.0.1", disable_validate=True)
    dual = DualFrankaConfig(
        node_rank=0,
        left_robot_ip="10.0.0.1",
        right_robot_ip="10.0.0.2",
        right_controller_node_rank=3,
    )

    single_arms = single.arms()
    dual_arms = dual.arms()

    assert list(single_arms) == ["arm"]
    assert list(dual_arms) == ["left", "right"]
    assert all(
        isinstance(arm, FrankaArmConfig)
        for arm in (*single_arms.values(), *dual_arms.values())
    )
    # Per-arm placement is expressed identically in both.
    assert dual_arms["left"].node_rank == 0
    assert dual_arms["right"].node_rank == 3
    assert single_arms["arm"].node_rank == 0


def test_place_franka_arms_tears_down_arms_already_placed(monkeypatch):
    """A partial robot is never returned when a later arm fails to come up."""
    disconnected: list[str] = []

    class FakeHandle:
        def __init__(self, name: str):
            self.name = name

        def part(self, _name: str):
            return FakeControllablePart(self.name, [])

        def disconnect(self) -> None:
            disconnected.append(self.name)

    class FakeDriver:
        @staticmethod
        def spawn(robot_ip, *args, node_rank=None, name=None):
            if robot_ip == "10.0.0.2":
                raise RuntimeError("right arm is unreachable")
            return FakeHandle(robot_ip)

    monkeypatch.setattr(franka_module, "_franka_driver_cls", lambda backend: FakeDriver)

    with pytest.raises(RuntimeError, match="unreachable"):
        place_franka_arms(
            {
                "left": FrankaArmConfig(robot_ip="10.0.0.1"),
                "right": FrankaArmConfig(robot_ip="10.0.0.2"),
            },
            backend="franky",
            default_node_rank=0,
            worker_rank=0,
            env_idx=0,
        )

    assert disconnected == ["10.0.0.1"]


def test_place_franka_arms_scales_past_two(monkeypatch):
    """Nothing in placement is specific to one or two arms."""

    class FakeHandle:
        def __init__(self, name: str):
            self.name = name

        def part(self, _name: str):
            return FakeControllablePart(self.name, [])

        def disconnect(self) -> None:
            pass

    monkeypatch.setattr(
        franka_module,
        "_franka_driver_cls",
        lambda backend: type(
            "FakeDriver",
            (),
            {"spawn": staticmethod(lambda ip, *a, **k: FakeHandle(ip))},
        ),
    )

    arms, handles = place_franka_arms(
        {
            name: FrankaArmConfig(robot_ip=f"10.0.0.{index}")
            for index, name in enumerate(("left", "right", "third"), start=1)
        },
        backend="franky",
        default_node_rank=0,
        worker_rank=0,
        env_idx=0,
    )

    assert list(arms) == ["left", "right", "third"]
    assert list(handles) == ["left", "right", "third"]
    assert isinstance(FrankaRobot(arms=arms, drivers=handles).arms["third"], Arm)
