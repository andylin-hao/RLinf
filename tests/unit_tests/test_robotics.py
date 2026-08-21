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


# --- from test_robotics.py --------------------------------------------


class FakePart(RobotPart):
    """A part that writes down what was done to it.

    Written the way a driver is written: ``_open`` and ``_release``, never
    ``connect`` and ``disconnect``. Those two decide *where* the device runs,
    so a part that overrode them would opt itself out of ever being placed.
    """

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
    """An end effector answers the category's questions, whoever opens it."""

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

    def is_robot_up(self) -> bool:
        """Off-interface, and exactly the kind of call an env makes."""
        return True


class FakeWorkerGroup:
    """Stand-in for the one-worker group a placed connection talks through.

    Every public method of the hosted class arrives here as an attribute, and
    properties arrive as ``attribute(name)``, so a fake only has to record the
    call and hand back a result that ``wait()`` unwraps.
    """

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
    arm = PartGroup(arm=FakeControllablePart("driver", events), wrist=camera)
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
    """Order matters: a killed actor cannot close the device it was holding."""
    from rlinf.robotics.placement import shutdown

    group = FakeWorkerGroup()
    shutdown(group)

    assert [name for name, _ in group.calls] == ["disconnect", "_close"]


def test_a_placed_connection_forwards_off_interface_driver_methods():
    """Methods outside the part interface travel with everything else.

    ``is_robot_up`` is not part of any contract -- it is one driver's own -- and
    the env calls it. A view is derived from the driver class, so it has that
    method for the same reason the driver does, with nothing registered.
    """
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
    # Names are the composition, so nothing caps the count or the kind.
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
    """Observation and action keys are the names the robot was composed with."""
    robot = Robot(arm=PartGroup(arm=FakeControllablePart("arm", [])))
    robot.connect()

    observation = robot.get_observation()
    action = {"arm": {"arm": {"target": np.array([0.25])}}}

    assert observation["arm"]["arm"]["state"].shape == (1,)
    assert robot.send_action(action) == action
    robot.disconnect()


def test_a_connection_hands_out_the_part_it_backs_not_a_controllable_one():
    """Picking a part out of a connection does not make it commandable.

    ``part(name)`` returns the object the connection listed, so a camera stays
    a camera and the robot composing it refuses an action for it -- rather than
    accepting one and failing at the first step.
    """

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
    robot = DOSW1Robot.build(is_dummy=True)
    robot.connect()

    assert set(robot.children) == {"left", "right"}
    observation = robot.get_observation()
    assert observation["left"]["arm"]["joint_position"].shape == (6,)
    robot.disconnect()
    assert not robot.is_connected


def test_pure_drivers_construct_without_scheduler_or_vendor_sdks():
    from rlinf.robotics.parts.base import Connection

    # A link to one arm is that arm, and commands it.
    arms = [
        FrankaROSArm("10.0.0.1"),
        FrankyArm("10.0.0.1"),
        GimArm("can0", "gim_arm_xl", True, "parallel"),
    ]
    # A bus driving several components is no one of them.
    buses = [Turtle2Connection()]

    assert all(isinstance(driver, ControllablePart) for driver in arms)
    # Every one of them is a connection -- declared, placed, opened, closed.
    # Only the arms are parts, because only they mean something when read, and
    # that one difference is the whole taxonomy.
    assert all(isinstance(driver, Connection) for driver in arms + buses)
    assert all(isinstance(driver, RobotPart) for driver in arms)
    assert not any(isinstance(driver, RobotPart) for driver in buses)
    assert all(not driver.is_connected for driver in arms + buses)
    # Each declares the parts riding on its connection.
    assert all(driver.parts for driver in arms + buses)


class _BareArm(Arm):
    """An arm that says nothing about itself, for tests that only place it.

    A robot composes parts, so a stand-in for one has to be a part even when
    every check in the test is about where it was opened rather than what it
    reads. It is an ``Arm`` so that a robot can select it by backend name, the
    way it selects a real one.
    """

    @classmethod
    def declare(cls, address, **settings):
        """Take whatever a robot offers; these tests are not about the wiring."""
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
    """Make every connection open in this process, whatever node it named.

    Placement itself is covered against a real cluster; these tests are about
    what a robot does around it, and a class defined inside a test function
    cannot be rebuilt inside a worker anyway.
    """
    from dataclasses import replace

    from rlinf.robotics.parts.base import Connection

    connect = Connection.connect

    def connect_here(self):
        if self._recipe is not None and self._recipe.node_rank is not None:
            self._recipe = replace(self._recipe, node_rank=None)
        connect(self)

    monkeypatch.setattr(Connection, "connect", connect_here)


def _fake_arm_backend(monkeypatch, *, failing_ip=None, disconnected=None):
    """Register a fake arm backend and point FrankaRobot at it by name.

    Through the registry rather than around it, so what these tests exercise is
    the lookup a config actually takes.
    """
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

    # setitem rather than the decorator, so the entry is removed again when the
    # test ends; it is the same mapping ``Arm.backends()`` reads.
    Arm.backends()  # load the real drivers first, so the mapping exists
    monkeypatch.setitem(Arm.__dict__["_BACKENDS"], "bench", FakeArm)
    monkeypatch.setattr(franka_module.FrankaRobot, "BACKEND", "bench")
    return FakeArm


def test_an_arm_backend_is_selected_from_the_registry_like_any_driver():
    """Swapping the stack that drives a Franka is naming a registered arm.

    It used to be a table inside the robot mapping a name to a class *name*,
    then a string comparison on that name to decide which module to import, and
    a per-backend function assembling positional constructor arguments -- three
    places to keep in step for two backends, and none of them the pattern every
    other device family already used.
    """
    from rlinf.robotics.parts.arms.base import Arm
    from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robots import DualFrankaRobot, FrankaRobot

    assert Arm.backend("franka_ros") is FrankaROSArm
    assert Arm.backend("franky") is FrankyArm
    assert {"franka_ros", "franky"} <= set(Arm.backends())

    # A robot names one; that is the whole of the swap.
    assert FrankaRobot.BACKEND == "franka_ros"
    assert DualFrankaRobot.BACKEND == "franky"
    for robot in (FrankaRobot, DualFrankaRobot):
        assert Arm.backend(robot.BACKEND) is not None

    with pytest.raises(ValueError, match="Unsupported Arm backend"):
        Arm.backend("no_such_stack")


def test_a_backend_maps_the_robot_settings_onto_its_own_constructor():
    """Each backend knows its constructor, so nothing else has to.

    The two Franka stacks take different arguments -- one a named end effector
    it opens on its ROS session, the other a gripper backend it builds beside
    itself -- and a robot naming one should not have to know which.
    """
    from rlinf.robotics.parts.arms.base import Arm
    from rlinf.robotics.robots import FrankaRobot

    ros = FrankaRobot.declare_arm(
        "10.0.0.2",
        node_rank=1,
        name="arm",
        backend="franka_ros",
        end_effector_type="ruiyan_hand",
    )
    assert type(ros).__name__ == "FrankaROSArm"
    assert ros.node_rank == 1, "placement must survive the mapping"

    franky = FrankaRobot.declare_arm(
        "10.0.0.3",
        node_rank=2,
        name="arm",
        backend="franky",
        gripper_type="robotiq",
        gripper_connection="/dev/ttyUSB0",
    )
    assert type(franky).__name__ == "FrankyArm"
    assert franky.node_rank == 2

    # A setting a backend cannot honour is refused, not dropped: the arm would
    # otherwise run with an end effector the config did not ask for.
    with pytest.raises(TypeError, match="cannot fit a named end effector"):
        FrankaRobot.declare_arm(
            "10.0.0.3",
            node_rank=0,
            name="arm",
            backend="franky",
            end_effector_type="ruiyan_hand",
        )

    # And an arm that takes none of them says so rather than ignoring them.
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
    """One category, so the contract is written once rather than per driver."""
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

    # The category is what a robot can ask for.
    assert inspect.isabstract(BaseArm)


def test_declaring_arms_opens_nothing_until_connect(monkeypatch):
    """Composing is inert. ``connect`` is what touches hardware."""

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
    """A half-connected robot is never left behind when a later part fails."""
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

    assert list(robot.children) == ["left", "right", "third"]
    assert robot.is_connected


def test_one_connection_is_opened_once_however_often_it_is_named():
    """A link backing several components opens exactly once.

    Three parts are picked out of one connection. Each answers ``owner`` with
    that connection, and the robot connects owners, so the link is opened once
    rather than once per part -- which is what hardware refuses.
    """
    opens: list[str] = []

    class Riding(ControllablePart):
        """A part with no ``_open``: it reads whatever the session opened."""

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
    """A robot names the arm; the gripper comes with it, because it is on it.

    Handing the robot both meant the robot knew what the arm carried, and had
    to be edited when that changed. GimArm is the case that shows it: whether a
    gripper is fitted is the arm's own answer, and the robot used to decide it a
    second time.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.robots import FrankaRobot, GimArmRobot

        arm = FrankaRobot.declare_arm("10.0.0.2", node_rank=0, name="arm")
        assert list(arm.parts) == ["end_effector"], (
            "an arm backs what rides on it, and is not one of them"
        )

        robot = FrankaRobot(arm=arm)
        assert list(robot.children) == ["arm"]
        assert robot.child("arm") is arm
        assert list(robot.child("arm").children) == ["end_effector"]

        # The reading has the same shape as the tree.
        assert set(robot.observation_features["arm"]) >= {"tcp_pose", "end_effector"}
        assert set(robot.action_features["arm"]) == {"tcp_pose", "end_effector"}

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
    """The path an env walks to reach a driver must land on the driver.

    Envs call driver methods that are not part of any contract -- ``get_state``,
    ``is_robot_up``, ``move_arm`` -- by asking a part which connection it rides.
    A group answering that question with itself made one of those paths return
    an object with none of those methods, and the failure surfaced at the first
    read rather than at the line that was wrong.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.robots import (
            DualFrankaRobot,
            FrankaRobot,
            GimArmRobot,
            Turtle2Robot,
        )

        # The path each shipped env walks, and the method it then calls.
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

        # And a group says so rather than answering for a connection it has not
        # got, which is what hid the mistake.
        with pytest.raises(TypeError, match="rides no connection"):
            dual.child("left").owner


def test_a_device_with_its_own_link_keeps_it_when_a_connection_lists_it():
    """``part()`` adopts a view, not a device that opens itself.

    An arm may list the camera bolted to its wrist, but that camera holds its
    own USB bus and may name its own node. Adopting it would mean the arm's
    connect opened it -- on the arm's machine, or not at all -- while the
    camera reported itself connected the whole time.
    """
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

    # A part that opens nothing is adopted, which is what a view is.
    gripper = MethodEndEffector(arm, state_field="gripper_position")
    assert gripper.owner is arm


def test_a_connection_answers_its_parts_before_it_is_opened():
    """Composition happens before ``connect``, so ``parts`` must not need it.

    This is what lets a robot be composed on a machine with no hardware, and
    what lets a placed connection keep answering ``parts`` locally: the answer
    is a fact about the class, not a reading from the device.
    """
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
    """Placement is a property of every connection, so a camera can name a node.

    ``node_rank`` is taken by the metaclass, not declared by the part, so a
    camera accepts it without a line of its own and answers where it will run
    before anything opens.
    """
    events: list[str] = []
    camera = FakeCamera("wrist", events, node_rank=2)

    assert camera.node_rank == 2
    assert FakeCamera("bench", events).node_rank is None
    assert events == [], "declaring where a camera runs must not open it"


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


def test_every_part_places_independently_whatever_it_is():
    """Arm, end effector, and camera each name their own node.

    A Robotiq gripper is a serial device of its own and a camera holds its own
    USB link, so neither has to ride the arm's connection or its node. Nothing
    about a part's category enters into it: ``node_rank`` is taken by the
    metaclass, so every connection accepts one for the same reason.
    """

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
    # Three separate links, so three opens: none of them rides another.
    assert len(robot.owners()) == 3
    assert [part.owner for part in (arm, gripper, wrist)] == [arm, gripper, wrist]


def test_a_leaf_part_placed_remotely_is_still_the_part_it_was():
    """A camera backs nothing but itself, and placing it must not change that.

    The view is derived from the camera class, so it *is* a ``Camera`` to
    ``isinstance`` and to ``parts_of_type``. Nothing has to be registered for a
    new device category to survive the trip, which is what the old table of
    hand-written proxies got wrong: a category it did not list came back a
    plain part.
    """
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
    """A camera descriptor plus a node is all a declaration needs."""
    from rlinf.robotics.parts.cameras import BaseCamera, CameraInfo

    info = CameraInfo(name="scene", serial_number="123", camera_type="realsense")
    declared = Camera.declare({"scene": info}, node_rank=4)

    assert set(declared) == {"scene"}
    assert declared["scene"].node_rank == 4
    assert type(declared["scene"]).__name__ == "RealSenseCamera"
    assert not declared["scene"].is_connected, "declaring a camera must not open it"
    assert Camera.declare(None) == {}
    # The backend comes from the registry, not from a table in the package.
    assert BaseCamera.backend("rs") is BaseCamera.backend("realsense")
    assert set(BaseCamera.backends()) >= {"realsense", "rs", "zed", "lumos"}
    with pytest.raises(ValueError, match="Unsupported BaseCamera backend"):
        BaseCamera.backend("no-such-camera")


def test_failed_connect_can_be_retried():
    """A failed connect restores the composition instead of poisoning the robot.

    Without this, the arm that did open would be left open with nothing holding
    it, and a retry would skip it as already connected.
    """
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
    """Cameras follow the same lifecycle as every other part.

    They used to be built by the environment and bolted on after connect, which
    meant a camera could never sit on the node it was plugged into. Now a
    camera says where it runs the same way an arm does, and the robot opens it
    the same way too.
    """
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


def test_the_lazy_package_still_types_what_it_exports():
    """Every lazily exported name is also imported for a type checker.

    ``rlinf.robotics`` loads its symbols through a module ``__getattr__`` so a
    node without a vendor SDK can still import the package. A type checker
    cannot see that, and resolves every name to ``Any`` -- which is what made
    ``Robot(arm=...)`` uncheckable and left an editor unable to say what a
    robot accepts. A ``TYPE_CHECKING`` block imports the same names statically.

    The two lists have to agree, and nothing at run time would notice if they
    drifted, so this compares them.
    """
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
    """Hardware code names neither the layer above it nor the one beside it.

    ``rlinf.scheduler`` is what robotics is an extension of, and Gymnasium
    belongs to the env layer that consumes a robot. A driver that reached for
    either would invert the dependency.

    Ray is not on the list. It is a base dependency of the package, so every
    machine running RLinf has it, and forbidding the name bought nothing.
    """
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
    """Every environment sits in one of the two halves, with no alias behind it.

    The pre-split paths were kept working by a ``sys.meta_path`` alias while
    callers moved over. Nothing imports them now, so the alias is gone and the
    old names resolve to nothing.
    """
    envs = _ROOT / "rlinf" / "envs"
    # Only directories carrying source count. A package that moved leaves its
    # __pycache__ behind, and stale bytecode on one machine is not a stray env
    # package -- it would fail this for whoever had checked out the old tree.
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
    """A leader arm is an arm with encoders, so it is a part.

    What makes something robotics here is that it talks to hardware without
    importing Ray or Gymnasium, not whether a policy observes it. As parts these
    devices get placement, the connect and disconnect lifecycle, and one
    connection opened once -- none of which they had as env-side helpers.
    """
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

    # Declaring one opens nothing, so a device can be described anywhere.
    mouse = SpaceMouse()
    assert not mouse.is_connected
    assert sorted(mouse.observation_features) == ["buttons", "twist"]


def test_teleop_readers_do_not_import_gymnasium():
    """A reader only talks to its vendor SDK, so a bench script can drive it."""
    readers = _ROOT / "rlinf" / "robotics" / "parts" / "teleop" / "readers"
    offenders = {
        path.name
        for path in readers.glob("*.py")
        if re.search(r"^\s*(import|from)\s+gymnasium\b", path.read_text(), re.M)
    }

    assert offenders == set()


# --- the real composition, exercised through DOSW1's dummy SDK ---------------
#
# Everything above this line drives fakes, which pins the contracts but cannot
# catch a robot whose own parts, placement, or lifecycle is wrong. DOSW1 ships a
# dummy mode that runs the real arm, gripper, and SDK-adapter code without
# hardware, so these tests go through production paths end to end.


def _dosw1_robot():
    from rlinf.robotics.robots import DOSW1Robot

    return DOSW1Robot.build(is_dummy=True)


def test_building_a_real_robot_touches_no_hardware():
    """``build`` leaves declarations, so a robot can be described anywhere.

    The machine assembling a robot config is often not the machine holding the
    hardware, so nothing may be opened until ``connect``.
    """
    from rlinf.robotics.parts.base import PartGroup, RobotPart

    robot = _dosw1_robot()

    assert not robot.is_connected
    # Every part came out of one session, because one SDK session drives them
    # all. They are real parts from the moment they are composed -- the tree
    # holds the same objects before and after connect.
    leaves = [
        leaf
        for part in robot.children.values()
        for leaf in (part.children.values() if isinstance(part, PartGroup) else [part])
    ]
    assert leaves, "the robot composed no parts"
    assert all(isinstance(leaf, RobotPart) for leaf in leaves)
    # Compared by identity, not equality: two sessions built from equal
    # arguments are still two devices, so owners() keys on id() too.
    sessions = [leaf.owner for leaf in leaves]
    assert all(session is sessions[0] for session in sessions)
    assert not sessions[0].is_connected, "composing the robot opened the session"


def test_parts_sharing_a_session_are_never_read_concurrently():
    """DOSW1's two sides ride one SDK session, so they run in one batch.

    ``_fan_out`` runs batches concurrently and each batch in declaration order.
    Grouping only by the connection a leaf rides is not enough, because the two
    sides are nested groups rather than leaves: keying each group separately put
    two threads on one vendor session, which few of them survive.
    """
    robot = _dosw1_robot()

    assert len(robot.owners()) == 1, "DOSW1 should present exactly one session"
    assert robot._batches() == [["left", "right"]], (
        f"both sides ride one session but were batched as {robot._batches()}"
    )


def test_parts_on_separate_connections_still_run_together():
    """The grouping must not serialize what is genuinely independent.

    Two Franka arms are two devices on two links, so reading them at the same
    time is the point. Only overlap forces parts into one batch.
    """
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

    assert len(robot.owners()) == 2
    assert robot._batches() == [["left"], ["right"]]
    # Within one arm there is one part to batch: the gripper rides the arm and
    # is read with it, not beside it.
    assert robot.child("left")._batches() == [["arm"]]
    assert set(robot.child("left").child("arm").children) == {"end_effector"}


def test_a_group_spanning_two_sessions_pulls_both_into_one_batch():
    """Overlap is transitive, so grouping cannot key on one connection each.

    Three children: one on session A, one on session B, and between them a
    group holding a part from each. Keying every child by a single connection
    would leave the two outer children in separate batches while the middle one
    ran against both -- three threads on two sessions. They all belong together.
    """

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
    """Connect, read, command, reset, disconnect, and connect again."""
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

    # A disconnected robot goes back to its declarations and can run again.
    robot.connect()
    assert robot.is_connected
    robot.disconnect()


def test_one_connection_is_opened_once_for_every_part_it_drives():
    """DOSW1 drives both arms and both grippers over a single SDK session.

    Opening it once per part would open that session four times, which the
    hardware does not allow. Every part answers :attr:`owner` with the session,
    and the robot connects owners rather than parts, so there is one of them.
    """
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
    """One connection, four components, named by the robot that composed them."""
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
    """Every path in the observation is a name the robot was composed with."""
    robot = _dosw1_robot()
    robot.connect()

    observation = robot.get_observation()
    paths = {
        f"{group}.{part}" for group, parts in observation.items() for part in parts
    }

    assert paths == set(robot.named_parts) - set(robot.children)

    robot.disconnect()


# --- teleop composition ------------------------------------------------------


def _scripted_device(reading):
    """A teleop part that always reports `reading`."""
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
    """What a robot with these parts expects, for a group's `available`."""
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
    """A dex-hand rig is a spacemouse on the arm and a glove on the hand."""
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
    """A spacemouse offers a gripper; an arm carrying a hand has none."""
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
    """Silently letting one win would move the robot from the wrong device."""
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
    """A rig that cannot drive anything is a configuration mistake."""
    import numpy as np
    import pytest

    from rlinf.robotics.teleop import TeleopEntry, TeleopGroup

    with pytest.raises(ValueError, match="fills none"):
        TeleopGroup(
            [TeleopEntry(_scripted_device({}), _binding(("hand",), np.ones(6)))],
            available=_kinds(*("arm", "end_effector")),
        )


def test_drives_separates_two_identical_leaders():
    """Both leaders produce an arm; `drives` says which branch each fills."""
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
    """A spacemouse drives an arm and a gripper over one HID handle."""
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


def test_the_glove_holds_what_the_operator_posed():
    """Releasing the arm device leaves the hand where it was put.

    Under composition an unbound part would keep the policy's value, so the
    hold the dex-hand setup has always had is stated in the binding.
    """
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
    """A spacemouse button is what puts the glove in control.

    Devices in one rig are not independent. Stating the gate as published
    context keeps that ordered and visible, rather than hidden inside a class
    that reads both devices.
    """
    import numpy as np

    from rlinf.robotics.teleop import GloveBinding, SpaceMouseBinding

    mouse, glove = SpaceMouseBinding(), GloveBinding()
    glove.reset({"hand_reset_pose": np.zeros(6)})

    released = mouse.publish({"twist": np.zeros(6), "buttons": [False, False]})
    held = mouse.publish({"twist": np.zeros(6), "buttons": [False, True]})

    assert released == {"hand_driving": False}
    assert held == {"hand_driving": True}

    # Held: the first reading re-zeros, a later one moves the hand.
    glove.action({"angles": np.zeros(6)}, held)
    moved = glove.action({"angles": np.full(6, 0.3)}, held).parts["hand"]
    assert np.allclose(moved, 0.3)

    # Released: the hand stays where it was posed.
    assert np.allclose(
        glove.action({"angles": np.full(6, 0.9)}, released).parts["hand"], 0.3
    )


# --- PICO, as a device and a binding ----------------------------------


class _ScriptedController:
    """A controller part replaying a fixed sequence of readings."""

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
    """An absolute binding turns a delta into where the arm should end up."""
    import numpy as np

    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0)
    device = _ScriptedController([((0.025, 0.0, 0.0), (0.0, 0.0, 0.0), True, -1)])

    parts = binding.action(device.get_observation(), _pico_context()).parts

    # The arm goes where the operator moved from where it was when they grabbed.
    assert parts["arm"].size == 9
    assert np.isclose(parts["arm"][0], 0.3 + 0.025)
    assert np.isclose(parts["end_effector"][0], -1.0)
    assert binding.action(device.get_observation(), _pico_context()).driving


def test_releasing_the_grip_mid_chunk_holds_the_arm_where_it_was_left():
    """Snapping back to the policy's pose would jerk the arm mid-motion."""
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

    # The next chunk of policy actions releases it.
    binding.on_action_chunk_begin()
    assert binding.action(device.get_observation(), context).parts == {}


def test_holding_the_current_pose_leaves_the_gripper_to_the_policy():
    """The operator is not touching the gripper, so the policy still owns it."""
    from rlinf.robotics.teleop import PicoTcpBinding

    binding = PicoTcpBinding(gripper=True, side=0, hold_current_when_inactive=True)
    device = _ScriptedController([((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 0)])

    parts = binding.action(device.get_observation(), _pico_context()).parts

    assert set(parts) == {"arm"}


def test_a_delta_binding_has_no_pose_to_hold():
    """Zero motion already holds a robot that takes deltas."""
    from rlinf.robotics.teleop import PicoBinding, PicoTcpBinding

    assert PicoBinding(gripper=True).hold(_pico_context()) == {}
    assert "arm" in PicoTcpBinding(gripper=True).hold(_pico_context())


def test_absolute_commands_are_clipped_but_deltas_are_not():
    """A pose can leave the action space; a normalised delta cannot."""
    from rlinf.robotics.teleop import PicoBinding, PicoTcpBinding, SpaceMouseBinding

    assert PicoTcpBinding.CLIPS_TO_ACTION_SPACE
    assert not PicoBinding.CLIPS_TO_ACTION_SPACE
    assert not SpaceMouseBinding.CLIPS_TO_ACTION_SPACE


def test_each_side_reports_its_own_state():
    """Two controllers report separately, so a collector can tell them apart."""
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
    """The part reports data, so nothing about the robot reaches the reader."""
    import inspect

    from rlinf.robotics.parts.teleop.devices import PicoController
    from rlinf.robotics.parts.teleop.readers import pico

    source = inspect.getsource(pico.PicoExpert)
    for name in ("tcp_pose", "_ref_tcp"):
        assert name not in source, f"the reader still refers to {name!r}"

    observation = PicoController.get_observation
    assert "get_reading" in inspect.getsource(observation)


def test_a_controller_reading_is_data_not_a_handle():
    """An observation another part could record, rather than an object."""
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
    """Re-taking hold re-anchors, so the arm does not jump."""
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

    # Same motion from the same measured pose, so the same command both times.
    assert np.isclose(first, again)


# --- the controller reader, from packets ------------------------------


def _pico_packet(x, y, z, yaw, grip, close=False, open_=False):
    """One frame as the headset publishes it."""
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
    """A reader wired to scripted packets rather than to a headset."""
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
    """Not from the origin: grabbing at an arbitrary pose must not jump.

    The headset's axes are not the robot's. Its ``-z`` is the robot's ``+x``,
    so this also pins down the mapping the reader applies.
    """
    import numpy as np

    reader = _pico_reader(operator_to_robot_yaw=0.0)
    reader._snapshot = lambda: _pico_packet(0.5, 0.5, 0.5, 0.0, grip=0.95)
    reader.get_reading()  # takes hold here, at (0.5, 0.5, 0.5)

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

    # The last frame re-took hold where it stands, so nothing has moved yet.
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
    """The whole path, from packet to the numbers the env receives."""
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

    # The arm is asked for where it was when they grabbed, plus their motion.
    expected = np.asarray(context["tcp_pose"][:3]) + reading["position_delta"]
    assert np.allclose(parts["arm"][:3], expected, atol=1e-6)
    assert np.isclose(reading["position_delta"][0], 0.04, atol=1e-9)


# --- one lifecycle, whatever kind of part it is -----------------------


def test_every_device_family_is_shaped_the_same_way():
    """One category per family, the registry on it, one lifecycle rule.

    Four families drifted into four answers. Arms had no category at all, so
    their backends lived in a table inside a robot; cameras registered on their
    driver base, so asking the *category* what backends existed returned
    nothing; and whether a driver had to write ``_open`` depended on which
    family it belonged to.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        import rlinf.robotics.parts.cameras  # noqa: F401  - registers drivers
        import rlinf.robotics.parts.end_effectors  # noqa: F401
        import rlinf.robotics.parts.teleop.devices  # noqa: F401
        from rlinf.robotics.parts.arms.base import Arm, BaseArm
        from rlinf.robotics.parts.cameras.base import BaseCamera, Camera
        from rlinf.robotics.parts.end_effectors.base import BaseEndEffector, EndEffector
        from rlinf.robotics.parts.teleop.devices import TeleopPart

        # The registry answers on the category, which is what a config names.
        for category in (Arm, Camera, EndEffector):
            assert category.backends(), (
                f"{category.__name__} has no registered backends; a config "
                "naming one would have nothing to resolve against"
            )

        # A driver base requires opening. Releasing too, unless the family
        # releases the same way for all of its drivers and does it once.
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
    """Nothing readable in a robot is 'a part' with no kind.

    A category is what ``parts_of_type`` asks for and what a placed part comes
    back as, so a part outside all of them is invisible to both. The two arm
    views were exactly that until ``Arm`` existed to belong to.
    """
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

        Arm.backends()  # loads every arm module

        def descendants(cls):
            for child in cls.__subclasses__():
                yield child
                yield from descendants(child)

        categories = (Arm, Camera, EndEffector, MobileBase, TeleopPart)
        homeless = sorted(
            cls.__name__
            for cls in descendants(RobotPart)
            if cls.__module__.startswith("rlinf.")  # not this file's own fakes
            and not inspect.isabstract(cls)
            and not issubclass(cls, PartGroup)
            and not any(issubclass(cls, category) for category in categories)
        )

    assert homeless == [], f"parts belonging to no device category: {homeless}"


def test_every_part_family_opens_and_closes_the_same_way():
    """A part author learns _open/_release once, not once per device family."""
    import inspect

    from rlinf.robotics.parts.cameras.base import BaseCamera
    from rlinf.robotics.parts.end_effectors.base import BaseEndEffector
    from rlinf.robotics.parts.end_effectors.grippers.base import BaseGripper
    from rlinf.robotics.parts.teleop.devices import TeleopPart

    for family in (TeleopPart, BaseCamera, BaseEndEffector, BaseGripper):
        assert hasattr(family, "_open"), f"{family.__name__} has no _open"
        assert hasattr(family, "_release"), f"{family.__name__} has no _release"

    # The names the families used to use are gone.
    for family in (BaseCamera, BaseEndEffector):
        source = inspect.getsource(family)
        for retired in ("_close_device", "def initialize", "def shutdown"):
            assert retired not in source, f"{family.__name__} still has {retired}"

    # Opening and closing belong to the connection, and only to it. A family
    # that overrode connect or disconnect would decide where its devices run,
    # which is the one thing those two are for.
    for family in (TeleopPart, BaseCamera, BaseEndEffector, BaseGripper):
        for public in ("connect", "disconnect"):
            assert public not in vars(family), (
                f"{family.__name__} overrides {public}; a part placed on another "
                "node would then never be rebuilt there"
            )

    # What a category adds around its drivers goes in the local-only hooks.
    assert "_opened" in vars(BaseCamera), (
        "BaseCamera starts its capture loop, and that has to run beside the "
        "camera rather than beside whoever is holding it"
    )


def test_a_gripper_is_an_end_effector_rather_than_a_second_kind_of_one():
    """One driver base, so a caller holding an ``EndEffector`` can assume things.

    ``BaseGripper`` used to be a sibling of ``BaseEndEffector``, and the two
    disagreed about exactly what generic code needs: whether ``_open`` was
    required, what the observation was called, and whether ``reset`` took a
    target. A gripper is an end effector with one degree of freedom, so it says
    so, and writes the shared surface in terms of open/close/move.
    """
    import inspect

    from rlinf.robotics.parts.end_effectors.base import BaseEndEffector, EndEffector
    from rlinf.robotics.parts.end_effectors.grippers.base import BaseGripper
    from rlinf.robotics.parts.views import MethodEndEffector

    assert issubclass(BaseGripper, BaseEndEffector)

    # The lifecycle is declared once, and required of both.
    for hook in ("_open", "_release"):
        assert BaseGripper.__dict__.get(hook) is None, (
            f"BaseGripper re-declares {hook}; it should inherit the one contract"
        )
        assert getattr(BaseEndEffector, hook).__isabstractmethod__, (
            f"{hook} must be required, or a driver that never wrote one fails "
            "at the first connect instead of at class definition"
        )

    # ``reset`` has one signature across the family.
    assert list(inspect.signature(BaseGripper.reset).parameters) == list(
        inspect.signature(BaseEndEffector.reset).parameters
    )

    # A view is an end effector without being a driver, so the lifecycle above
    # does not apply to it and it needs no ``_open``.
    assert issubclass(MethodEndEffector, EndEffector)
    assert not issubclass(MethodEndEffector, BaseEndEffector)


def test_a_gripper_is_commanded_in_the_units_it_reports():
    """One axis, in metres, so a width read back can be commanded again.

    ``move`` used to take a Robotiq's 0-255 register counts while ``position``
    reported metres, which made ``command(get_state())`` meaningless -- and the
    two drivers ran that raw number in opposite directions, so the same call
    opened one gripper and closed the other. Whatever counts the hardware takes
    are now the driver's business.
    """
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

            # A width survives the trip to the hardware and back, to within the
            # one register count the device quantises to.
            quantum = gripper.max_width / 255
            for width in (0.0, gripper.max_width / 2, gripper.max_width):
                gripper.move(width)
                assert gripper.position == pytest.approx(width, abs=quantum)

            # And that is exactly what the end-effector contract rides on.
            gripper.move(gripper.max_width / 3)
            state = gripper.get_state()
            gripper.open()
            gripper.command(state)
            assert gripper.position == pytest.approx(state[0], abs=quantum)

            # Past the stroke clamps rather than wrapping round the register.
            gripper.move(gripper.max_width * 10)
            assert gripper.position == pytest.approx(gripper.max_width, abs=quantum)
        finally:
            gripper.disconnect()


def test_a_franka_hand_is_commanded_in_metres_on_the_wire():
    """The width a caller asks for is the width the topic carries.

    It used to be divided by ``255 * 10`` first, so a 5 cm command reached the
    hand as 20 micrometres -- fully closed. Nothing caught it because the
    number was still a plausible width.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm

        arm = FrankaROSArm("10.0.0.2", end_effector_type="franka_gripper")
        arm.connect()
        try:
            gripper = arm._gripper
            widths: list[float] = []
            put = gripper._ros.put_channel

            def record(channel, message):
                widths.append(float(message.goal.width))
                return put(channel, message)

            gripper._ros.put_channel = record

            arm.open_gripper()
            arm.move_gripper(0.05)
            arm.move_gripper(gripper.max_width * 10)
        finally:
            arm.disconnect()

    assert widths == pytest.approx([gripper.max_width, 0.05, gripper.max_width])


def test_every_end_effector_answers_the_same_questions():
    """A task holding an ``EndEffector`` must not have to ask which kind it got.

    A hand on an arm's bus, a gripper on an arm's bus, a gripper on a shared
    SDK session, and a gripper on its own serial port are four different things
    to the hardware and one thing to a policy. The contract used to live on the
    driver base, so only the ones with a link of their own answered it and the
    views -- which are what a robot actually composes -- answered none of it.
    """
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
            return robot.child("arm").child("end_effector")

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

        # The hand is six-fingered, which is why its class is not named for a
        # gripper: one view serves every end effector reached through a host.
        assert every["hand on a bus"].action_dim == 6
        assert type(every["hand on a bus"]) is type(every["gripper on a bus"])


def test_every_end_effector_reports_its_state_under_the_same_name():
    """A gripper and a hand answer the same keys, so one env reads both.

    The gripper used to report ``position`` where everything else reported
    ``state``, which an env building an observation space from the part could
    only handle by knowing which driver it had.
    """
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
    """A part whose hardware is a list of what happened to it."""

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
    """``_release`` gets the handle back, rather than reading it off ``self``.

    Clearing ``_device`` first left every implementation that reaches for it --
    which is where the vendor object is documented to live -- closing nothing,
    while the part reported itself disconnected and its thread and serial port
    stayed open.
    """
    part = _Part()
    part.connect()

    assert part.is_connected
    part.disconnect()

    assert part.log == ["open", "close"]
    assert not part.is_connected
    # Idempotent: nothing is released twice.
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


# --- the smallest robot that works ------------------------------------


def test_a_robot_is_named_parts_and_nothing_else():
    """The whole of a working robot, with no registry and no hardware config.

    The shipped robots all declare discovery and a hardware config because they
    are reached by type name from a config file. A robot you construct yourself
    needs neither, and this is what that costs.
    """
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
    """A robot you construct yourself never needs build(), and is told so."""
    from rlinf.robotics import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    with pytest.raises(NotImplementedError, match="Construct Bench"):
        Bench.build()


# --- a connection is not a part ---------------------------------------


def test_a_connection_backing_several_parts_is_not_observable():
    """One ROS node driving two arms is no part, and says so.

    It used to satisfy the part interface with an observation of the whole
    robot and a coupled two-arm action that nothing called.
    """
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
    # Not "refuses to be read" but "is not the kind of thing you read": the
    # methods are absent, so a robot cannot compose it into the tree and find
    # out at the first step.
    assert not isinstance(hardware, RobotPart)
    assert not hasattr(hardware, "get_observation")
    assert not hasattr(hardware, "observation_features")


def test_every_robot_composes_from_named_parts():
    """No robot puts a bare connection in its tree.

    Everything a robot holds is readable, because the tree is what an
    observation is built from. A connection that is also a part -- a ROS link
    to one arm -- lands under its own name, which is fine; a link that reads as
    nothing must be picked apart into the parts it backs.
    """
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
        DualFrankaRobot.build_arms(left_robot_ip="1.2.3.4", right_robot_ip="1.2.3.5"),
    ]
    for arms in built:
        for name, value in arms.items():
            values = (
                list(value.children.values()) if hasattr(value, "children") else [value]
            )
            # Every leaf is one capability picked out of a session. A leaf
            # that is a Connection and not a RobotPart is the session itself,
            # and reading it would mean nothing.
            assert all(isinstance(v, RobotPart) for v in values), (
                f"{name} holds {[type(v).__name__ for v in values]} rather than "
                "parts picked out of its connection"
            )
            assert not any(
                isinstance(v, Connection) and not isinstance(v, RobotPart)
                for v in values
            )


def test_one_connection_backs_every_part_that_names_it():
    """Both halves of an arm share a link; two arms do not share one."""
    from rlinf.robotics.robots.dual_franka import DualFrankaRobot
    from rlinf.robotics.robots.franka import FrankaRobot

    single = FrankaRobot.build_arms(robot_ip="1.2.3.4", node_rank=0)
    assert len({id(part.owner) for part in single.values()}) == 1

    dual = DualFrankaRobot.build_arms(left_robot_ip="1.2.3.4", right_robot_ip="1.2.3.5")
    sides = {
        side: {id(part.owner) for part in group.children.values()}
        for side, group in dual.items()
    }
    assert len(sides["left"]) == 1 and len(sides["right"]) == 1
    assert sides["left"] != sides["right"]


def test_a_connection_with_several_parts_hands_each_of_them_out_by_name():
    """A link backing several components is composed one name at a time.

    ``part(name)`` is the whole of it -- there is no resolution step and no
    intermediate reference, so what the robot holds is the part itself.
    """

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


# --- the bench check, checked ------------------------------------------


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
    """The script the bench runs, run here against fakes."""
    assert _run_bench(_fake_robot_type()) == 0


def test_the_bench_check_catches_an_observation_that_was_never_declared():
    assert _run_bench(_fake_robot_type("Mismatch")) == 1


def test_a_connection_left_in_the_tree_is_refused_at_composition():
    """The tree holds parts, and the message says what to do instead.

    The bench check still refuses a connection it finds among the leaves, but
    it can no longer be handed one: composing a robot is where this is caught
    now, which is before any hardware is reached and before a name for the
    mistake has to be guessed from a failed read.
    """
    with pytest.raises(TypeError, match="backs parts without being one"):
        _run_bench(_fake_robot_type("Connection"))


# --- real parts, faked SDKs -------------------------------------------


def test_no_part_hook_collides_with_the_worker_group():
    """A placed part becomes a worker, and its private methods are attached.

    `_close` was the obvious name for the lifecycle hook and it is taken:
    `WorkerGroup._close` already exists, so every remotely placed part failed
    to launch. Only placement caught it, which no unit test did.
    """
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
    """The camera class itself, from connect to frame to disconnect."""
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
    """The arm class itself: connect, the parts it backs, state, commands."""
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.arms.franky import FrankyArm

        arm = FrankyArm("10.0.0.1", gripper_connection="/dev/ttyUSB0")
        arm.connect()

        assert arm.is_connected
        # What the arm backs is what rides on it, and it is not one of them.
        assert set(arm.parts) == {"end_effector"}
        assert set(arm.children) == {"end_effector"}

        observation = arm.get_observation()
        assert observation["tcp_pose"].shape == (7,)
        assert arm.child("end_effector").get_observation()["state"].shape == (1,)

        arm.send_action({"joint_position": [0.0] * 7})

        arm.disconnect()
        assert not arm.is_connected


def test_the_bench_check_runs_a_whole_robot_on_fakes():
    """The command a bench runs, run here on the same code path."""
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
    """All five, through compose, connect, observe and disconnect.

    Building the fakes is itself the check: each field a part reads has to be
    there, with the shape the part expects, or this fails.
    """
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
    """A part placed on a node is built in a process that never saw them.

    ``sitecustomize`` in the mock package is what closes that gap, so a mocked
    run can exercise real placement rather than only in-process construction.
    """
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
    """The teleop layout is read from the wrapped env, not the bare one.

    ``GripperCloseEnv`` holds the gripper shut and drops that channel, so an
    env declaring seven numbers presents six. Composing teleop above it raised
    "the declaration and step() disagree" until the wrapper said what it took
    away -- which only a full stack shows, because that is where wrappers are
    applied before teleop.
    """
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
    """``arm_cameras`` is declared, passed, and consumed by nobody.

    ``DualFrankaEnv`` splits its cameras into per-arm and robot-level, then
    hands the per-arm ones to ``DualFrankaRobot.build(arm_cameras=...)``. That
    builder takes ``**config`` and forwards it to ``build_arms``, which ignores
    what it does not recognise, so the wrist cameras are dropped in silence and
    the env's observation space asks for frames no part produces.

    Found by an end-to-end run on faked SDKs, where reset returned no
    observation at all.
    """
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
    """A dual-arm robot places a camera per wrist, both on the same node.

    A name built from the class and the node alone is the same string twice,
    and Ray refuses the second actor rather than placing it, so the robot comes
    up with one wrist camera and an observation space asking for two.
    """
    from rlinf.robotics.parts.cameras.realsense import RealSenseCamera
    from rlinf.robotics.placement import PartWorkerHost

    names = {PartWorkerHost.default_name(RealSenseCamera, 0) for _ in range(8)}

    assert len(names) == 8, f"names repeat: {sorted(names)}"
    assert all(name.startswith("RealSenseCamera-node0-") for name in names), (
        f"a name should still say what it is and where: {sorted(names)}"
    )


def test_every_real_task_is_registered_through_the_shared_factory():
    """A task registered by hand does not fit the call the env worker makes.

    ``task_factory`` is what passes ``env_cfg`` to the wrapper stack and keeps
    it out of the env's constructor. A package that calls ``register`` itself
    builds an entry point taking different arguments, so the task can be
    constructed by hand and never by the runner.
    """
    import re

    offenders = []
    for path in (_ROOT / "rlinf" / "envs" / "real").rglob("__init__.py"):
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), 1):
            if re.match(r"\s*register\(", line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == [], f"these register a task outside register_tasks: {offenders}"


def test_an_address_is_checked_by_the_arm_that_dials_it():
    """The part refuses a placeholder; the config it came from does not.

    Whether a string is an address is one question with one answer, and two
    robot configs used to answer it separately -- with different messages, and
    with a ``disable_validate`` switch on one of them to turn it back off. The
    arm is what opens the connection, so the arm is what checks, and a
    placeholder left in a YAML now fails where it is used.
    """
    from rlinf.robotics.parts.arms.franka_ros import FrankaROSArm
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robots.dual_franka import DualFrankaConfig
    from rlinf.robotics.robots.franka import FrankaConfig

    # Parsing a config no longer decides this; enumeration may still fill the
    # field in from the node's environment.
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
    """The tree holds parts. A connection backs them and is not one.

    Before the split this was a policy: ``Connection`` inherited
    ``get_observation`` and overrode it to raise, so a connection composed into
    a robot by mistake type-checked fine and failed at the first read. Now the
    method does not exist on it, and the two shapes that used to disagree --
    a connection that refused to be read, and one that answered by repeating
    its own parts -- are the same shape.
    """
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


def test_a_retired_dosw1_config_object_is_refused_not_ignored():
    """``build`` absorbs unknown keywords, which is how a caller goes quiet.

    It used to take one config object. A call still passing ``config=`` would
    land in ``**_`` and leave every setting at its default -- ``is_dummy``
    included, so a session meant to skip the SDK would reach for it and fail
    somewhere else entirely.
    """
    from rlinf.robotics.robots import DOSW1Robot

    with pytest.raises(TypeError, match="no longer takes a config object"):
        DOSW1Robot.build(config=object())


def test_disconnect_releases_before_it_forgets_the_handle():
    """``_release`` is what closes the vendor object, so it needs it.

    Clearing ``_device`` first left every implementation reading it -- which is
    where the handle is documented to live, and what ``TeleopPart`` uses to
    close a reader by whichever name its vendor gave the method -- closing
    nothing. The part reported itself disconnected while the reader, its
    thread and its serial port stayed open, for GELLO, gloves, PICO and the
    spacemouse alike.
    """
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
    """Stall recovery closes a camera and opens it again.

    A thread runs once, so a camera holding one from ``__init__`` raised
    ``RuntimeError: threads can only be started once`` the second time -- on
    the recovery path both Franka envs take when a camera stops producing
    frames.
    """
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
    """A part method named like the worker's own would replace it.

    The part's methods are re-declared in the worker's class body, so the
    collision breaks the worker rather than the part -- later, somewhere else,
    and with no mention of the method responsible. Refuse it while the name is
    still in hand.
    """
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
            """Collides with the call a view reads properties through."""

    with pytest.raises(TypeError, match="share a name with the worker"):
        PartWorkerHost.worker_class(Clashing)

    # Every shipped driver is clear of it.
    for driver in (FrankyArm, FrankaROSArm, GimArm):
        assert PartWorkerHost.worker_class(driver) is not None


def test_a_hosted_camera_reopens_on_the_node_that_holds_it():
    """A stalled camera is recovered where it is, not where it is held.

    ``connect`` and ``disconnect`` stay local -- a view that forwarded them
    would ask the worker to end itself and then still be holding it -- so
    ``reopen`` is the call that crosses the boundary and does the work beside
    the device.
    """
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

    # And the same call on a local camera does the real thing.
    info = CameraInfo(name="c", serial_number="X", camera_type="realsense")
    assert callable(Fake(info).reopen)


def test_a_held_hand_resumes_from_the_pose_the_env_reset_it_to():
    """The env homes the hand, then the binding has to agree with it.

    Resetting with no context zeroed the commanded pose, so the first takeover
    command moved the hand away from where the env had just put it. On hardware
    that is an abrupt command.
    """
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
    """One word used to answer both, and they are not the same question.

    ``parts`` is what a hardware session backs; ``children`` is what a robot
    was composed of. Composition is where the two meet -- ``part(name)`` picks
    one and the keyword names it -- and nowhere else. Sharing a name meant a
    reader could not tell which of the two a given call site meant, and the
    docs got it wrong.
    """
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
    """The point of describing a robot is to check it without one present.

    Placement and ownership live in the declaration, so both are answerable
    before connecting -- and a connected part no longer carries a declaration,
    which is why the robot keeps the snapshot and reads it either side.
    """
    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    robot = Bench(arm=FrankyArm("10.0.0.1", node_rank=2))

    described = robot.describe()

    assert "Bench" in described
    # The arm, with what rides on it drawn beneath it.
    assert "arm" in described and "end_effector" in described
    assert "node=2" in described, described
    # Both rows come from one connection, so both say the same one.
    assert described.count("FrankyArm#1") == 2, described


def test_a_robot_can_be_disconnected_twice():
    """Teardown calls disconnect when it is not sure the robot came up.

    Disconnecting restores the composition, so the second call walked a tree
    holding unresolved picks and asked one whether it was connected. The
    ``AttributeError`` that produced would surface from inside a ``finally``,
    replacing whatever error the caller was actually handling.

    Found by the robot conformance suite, against every shipped robot at once.
    """
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.arms.franky import FrankyArm
    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "Bench"

    robot = Bench(arm=FrankyArm("10.0.0.1", gripper_connection="/dev/mock-gripper"))

    with mocked_sdks():
        robot.connect()
        robot.disconnect()
        robot.disconnect()

        assert not robot.is_connected
