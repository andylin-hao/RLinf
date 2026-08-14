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
    Arm,
    Camera,
    ControllablePart,
    DOSW1Robot,
    DOSW1RobotConfig,
    DualFrankaConfig,
    DualFrankaRobot,
    EndEffector,
    FrankaArmConfig,
    FrankaConfig,
    FrankaRobot,
    GimArmConfig,
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

    def subpart_observation(self, name: str) -> FakeRemoteResult:
        self.calls.append(("subpart_observation", name))
        return FakeRemoteResult({"tcp_pose": np.zeros(7)})

    def subpart_action(self, name: str, action: Any) -> FakeRemoteResult:
        self.calls.append(("subpart_action", (name, action)))
        return FakeRemoteResult(action)

    def is_robot_up(self) -> FakeRemoteResult:
        self.calls.append(("is_robot_up", None))
        return FakeRemoteResult(True)

    def _close(self) -> None:
        self.calls.append(("_close", None))


def test_robot_composes_and_namespaces_parts():
    events: list[str] = []
    arm = Arm(
        manipulator=FakeControllablePart("arm", events),
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
    left_arm = Arm(
        manipulator=FakeControllablePart("left_arm", events),
        end_effector=FakeEndEffector("left_gripper", events),
    )
    right_arm = Arm(
        manipulator=FakeControllablePart("right_arm", events),
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
    arm = Arm(manipulator=FakeControllablePart("arm", []))
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

    arm = Arm(manipulator=FakeControllablePart("driver", events))
    robot = Robot.single_arm(arm, handles={"arm": FakeHandle()})
    robot.connect()
    robot.disconnect()

    assert events[-1] == "handle"
    assert "disconnect:driver" in events


def test_driver_rejects_actions_for_observation_only_parts():
    class CameraOnlyHost(FakePart):
        def subparts(self) -> dict[str, RobotPart]:
            return {"wrist": FakeCamera("wrist", [])}

    with pytest.raises(TypeError, match="not controllable"):
        CameraOnlyHost("host", []).subpart_action("wrist", {})


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

    assert set(robot.arms) == {"left", "right"}
    observation = robot.get_observation()
    assert observation["arms"]["left"]["state"]["joint_position"].shape == (6,)
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
    # Each declares the subparts riding on its connection.
    assert all(driver.subparts() for driver in drivers)


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


def _fake_arm_backend(monkeypatch, *, failing_ip=None, disconnected=None):
    """Point FrankaRobot at a fake arm class that records what it places."""

    class FakeHandle:
        def __init__(self, name: str):
            self.name = name
            self.subparts = {"arm": FakeControllablePart(name, [])}

        def subpart(self, _name: str):
            return self.subparts["arm"]

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

    monkeypatch.setattr(
        FrankaRobot, "arm_part_cls", classmethod(lambda cls, backend=None: FakeArm)
    )
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

    monkeypatch.setattr(
        FrankaRobot, "arm_part_cls", classmethod(lambda cls, backend=None: NeverSpawns)
    )

    robot = FrankaRobot(
        arms=FrankaRobot.declare_arms(
            {"left": FrankaArmConfig(robot_ip="10.0.0.1")},
            default_node_rank=0,
            worker_rank=0,
            env_idx=0,
        )
    )

    assert not robot.is_connected


def test_connect_tears_down_parts_already_placed(monkeypatch):
    """A half-placed robot is never left behind when a later part fails."""
    disconnected: list[str] = []
    _fake_arm_backend(monkeypatch, failing_ip="10.0.0.2", disconnected=disconnected)

    robot = FrankaRobot(
        arms=FrankaRobot.declare_arms(
            {
                "left": FrankaArmConfig(robot_ip="10.0.0.1"),
                "right": FrankaArmConfig(robot_ip="10.0.0.2"),
            },
            default_node_rank=0,
            worker_rank=0,
            env_idx=0,
        )
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        robot.connect()

    assert disconnected == ["10.0.0.1"]


def test_declaring_arms_scales_past_two(monkeypatch):
    """Nothing in declaration or placement is specific to one or two arms."""
    _fake_arm_backend(monkeypatch)

    robot = FrankaRobot(
        arms=FrankaRobot.declare_arms(
            {
                name: FrankaArmConfig(robot_ip=f"10.0.0.{index}")
                for index, name in enumerate(("left", "right", "third"), start=1)
            },
            default_node_rank=0,
            worker_rank=0,
            env_idx=0,
        )
    )
    robot.connect()

    assert list(robot.arms) == ["left", "right", "third"]
    assert robot.is_connected


def test_one_declaration_is_placed_once_however_often_referenced():
    """A connection backing several components opens exactly once."""
    placements: list[str] = []

    class FakeHandle:
        def __init__(self):
            self.subparts = {
                "left": FakeControllablePart("left", []),
                "right": FakeControllablePart("right", []),
                "wrist": FakeCamera("wrist", []),
            }

        def subpart(self, name):
            return self.subparts[name]

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
        arms={
            "left": Arm(hardware.subpart("left")),
            "right": Arm(hardware.subpart("right")),
        },
        cameras={"wrist": hardware.subpart("wrist")},
    )
    robot.connect()

    assert placements == ["placed"], "the shared connection was opened more than once"
    assert isinstance(robot.cameras["wrist"], Camera)
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
        def subparts(self) -> dict[str, RobotPart]:
            return {"arm": self, "end_effector": FakeEndEffector("ee", events)}

    handle = HostWithSubparts.spawn("host", events)

    assert isinstance(handle.subpart("arm"), RobotPart)
    assert isinstance(handle.subpart("end_effector"), EndEffector)
    assert set(handle.subparts) == {"arm", "end_effector"}
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
    # declare_arms is inherited, not duplicated.
    assert (
        DualFrankaRobot.declare_arms.__func__
        is FrankaRobot.declare_arms.__func__
    )
    assert (FrankaRobot.BACKEND, DualFrankaRobot.BACKEND) == ("franka_ros", "franky")
    # build is specialised per robot.
    assert DualFrankaRobot.build.__func__ is not FrankaRobot.build.__func__
