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

"""Registered drivers own selection, dimensions, and device capabilities."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.robotics.parts.end_effectors import BaseHand, EndEffector


class PoseTool(BaseHand):
    """Three-axis test device whose name says nothing about its capabilities."""

    action_dim = 3
    state_dim = 3
    control_mode = "continuous"

    def __init__(self, gain=1.0):
        self.gain = gain
        self.position = np.zeros(3, dtype=np.float32)

    def _open(self):
        return object()

    def _release(self, device):
        pass

    def get_state(self):
        return self.position.copy()

    def command(self, action):
        self.position = self.gain * np.asarray(action, dtype=np.float32)
        return True

    def reset(self, target_state=None):
        self.position = (
            np.zeros(3) if target_state is None else np.asarray(target_state)
        )


@pytest.fixture
def registered_tool(monkeypatch):
    monkeypatch.setattr(EndEffector, "_BACKENDS", EndEffector.backends().copy())
    monkeypatch.setattr(EndEffector, "_ARM_BACKENDS", EndEffector._ARM_BACKENDS.copy())
    EndEffector.register("pose_tool")(PoseTool)
    return PoseTool


@pytest.mark.parametrize("backend", ["franka_ros", "franky"])
def test_custom_driver_composes_with_each_franka_arm(registered_tool, backend):
    from robot_mocks import mocked_sdks

    from rlinf.robotics import FrankaRobot

    with mocked_sdks():
        robot = FrankaRobot.build(
            robot_ip="10.0.0.1",
            node_rank=0,
            backend=backend,
            gripper_type="robotiq",
            end_effector_type="pose_tool",
            end_effector_config={"gain": 2.0},
        )
        tool = robot.child("end_effector", EndEffector)
        assert isinstance(tool, registered_tool)
        assert not tool.is_connected
        try:
            robot.connect()
            robot.send_action({"end_effector": {"target": np.ones(3)}})
            np.testing.assert_array_equal(
                robot.get_observation()["end_effector"]["state"], [2, 2, 2]
            )
        finally:
            robot.disconnect()
        assert not tool.is_connected


def test_dummy_env_and_wrapper_use_custom_driver_contract(registered_tool, monkeypatch):
    from rlinf.envs.real.franka import FrankaEnv
    from rlinf.envs.real.wrappers import build_stack
    from rlinf.robotics import FrankaConfig, RobotInfo

    def forbidden(*args, **kwargs):
        pytest.fail("A dummy environment must not construct or connect a driver")

    monkeypatch.setattr(registered_tool, "__init__", forbidden)
    info = RobotInfo(
        type="Franka",
        model="Franka",
        config=FrankaConfig(
            node_rank=0,
            camera_serials=["dummy"],
            end_effector_type="pose_tool",
        ),
    )
    env = FrankaEnv({"is_dummy": True, "hand_reset_state": [0.0] * 3}, robot_info=info)
    wrapped = build_stack(
        env, {"teleop": "none", "no_gripper": True, "use_relative_frame": False}
    )
    try:
        observation, _ = wrapped.reset()
        assert wrapped.action_space.shape == (9,)
        assert observation["state"]["hand_position"].shape == (3,)
        assert env.action_parts()[-1].width == 3
    finally:
        wrapped.close()


def test_connected_env_uses_part_dimensions(registered_tool):
    from rlinf.envs.real.franka import FrankaEnv

    tool = registered_tool(gain=1.0)
    env = FrankaEnv.__new__(FrankaEnv)
    env._end_effector = tool
    env._last_hand_command = None
    env.config = SimpleNamespace(hand_action_scale=1.0)
    assert env.action_parts()[-1].width == 3
    env._end_effector_action(np.array([0.2, 0.4, 0.6]))
    np.testing.assert_allclose(tool.get_state(), [0.2, 0.4, 0.6])


def test_driver_registers_its_own_arm_alias(registered_tool):
    from rlinf.robotics import FrankaRobot
    from rlinf.robotics.parts.end_effectors import FrankaGripper

    EndEffector.register("franka", arm_backend="custom_arm")(registered_tool)
    assert FrankaRobot.end_effector_class(backend="custom_arm") is registered_tool
    assert FrankaRobot.end_effector_class(backend="franka_ros") is FrankaGripper
    assert EndEffector.backend("POSE_TOOL") is registered_tool
    with pytest.raises(ValueError, match="already registered"):
        EndEffector.register("franka", arm_backend="custom_arm")(FrankaGripper)


@pytest.mark.parametrize("backend", ["franka_ros", "franky"])
def test_explicit_driver_takes_precedence_over_gripper_alias(backend):
    from rlinf.robotics import FrankaRobot
    from rlinf.robotics.parts.end_effectors import FrankaGripper

    assert (
        FrankaRobot.end_effector_class(
            backend=backend,
            gripper_type="robotiq",
            end_effector_type="franka_gripper",
        )
        is FrankaGripper
    )


def test_unknown_driver_reports_registry_names():
    from rlinf.robotics import FrankaRobot

    with pytest.raises(ValueError, match="Registered:"):
        FrankaRobot.end_effector_class(end_effector_type="missing")


def test_remote_view_retains_driver_metadata(registered_tool):
    from rlinf.robotics.placement import remote_view_of

    view = remote_view_of(registered_tool)
    assert view.action_dim == view.state_dim == 3
    assert view.is_gripper is False
    assert view.is_hand is True
    assert "command" in view.__dict__
    assert "get_state" in view.__dict__


def test_shared_gripper_views_report_their_capability():
    from rlinf.robotics import MethodEndEffector
    from rlinf.robotics.parts.arms.dosw1 import DOSW1EndEffector

    for tool in (
        MethodEndEffector(None, "width", is_gripper=True),
        DOSW1EndEffector(None, "left"),
    ):
        assert tool.is_gripper
        assert not tool.is_hand

    class HandView(MethodEndEffector, BaseHand):
        pass

    fingers = HandView(None, "fingers", dims=3)
    assert fingers.is_hand
    assert not fingers.is_gripper


def test_controller_cli_accepts_registered_driver_and_settings(
    registered_tool, monkeypatch
):
    from toolkits.realworld_check.test_franka_controller import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check",
            "--end-effector-type",
            "pose_tool",
            "--end-effector-config",
            '{"gain": 2}',
        ],
    )
    args = _parse_args()
    tool = EndEffector.of(args.end_effector_type, **args.end_effector_config)
    assert tool.gain == 2
    assert args.hand_baudrate is None
    assert args.hand_motor_ids is None


def test_teleop_device_owns_its_retired_flag(monkeypatch):
    from rlinf.envs.real.wrappers.teleop.config import resolve_teleop_devices
    from rlinf.robotics.parts.teleop import TeleopDevice

    monkeypatch.setattr(TeleopDevice, "_REGISTRY", TeleopDevice._REGISTRY.copy())

    @TeleopDevice.register("test_operator")
    class Operator(TeleopDevice):
        LEGACY_FLAGS = {"enable_test_operator": "test_operator"}

    with pytest.warns(DeprecationWarning, match="enable_test_operator"):
        assert resolve_teleop_devices(
            {"enable_test_operator": True}, supported=["test_operator"]
        ) == ["test_operator"]
    with pytest.warns(DeprecationWarning, match="supersedes"):
        assert (
            resolve_teleop_devices(
                {"teleop": "none", "enable_test_operator": True},
                supported=["test_operator"],
            )
            == []
        )


@pytest.mark.parametrize("dims", [1, 3, 6])
def test_generic_end_effector_has_no_inferred_capability(dims):
    from rlinf.robotics import MethodEndEffector

    tool = MethodEndEffector(None, "state", dims=dims)
    assert not tool.is_hand
    assert not tool.is_gripper


def test_hand_specialization_owns_finger_diagnostics():
    tool = PoseTool()
    assert tool.is_hand
    assert not tool.is_gripper
    assert tool.get_detailed_state() == {
        "positions": [0.0, 0.0, 0.0],
        "finger_names": ["dof_0", "dof_1", "dof_2"],
    }
    assert not hasattr(EndEffector, "finger_names")


@pytest.mark.parametrize("is_dummy", [False, True])
def test_franka_rejects_unclassified_tool_before_opening_hardware(
    monkeypatch, is_dummy
):
    from unittest.mock import Mock

    from rlinf.envs.real.franka import FrankaEnv
    from rlinf.robotics import FrankaConfig, FrankaRobot, RobotInfo

    monkeypatch.setattr(EndEffector, "_BACKENDS", EndEffector.backends().copy())

    @EndEffector.register("probe")
    class Probe(EndEffector):
        action_dim = state_dim = 3
        control_mode = "continuous"

        def get_state(self):
            return np.zeros(3)

        def command(self, action):
            return True

    build = Mock()
    monkeypatch.setattr(FrankaRobot, "build", build)
    config = FrankaConfig(
        node_rank=0, camera_serials=["dummy"], end_effector_type="probe"
    )
    info = RobotInfo(type="Franka", model="Franka", config=config)
    with pytest.raises(ValueError, match="exactly one of is_hand or is_gripper"):
        FrankaEnv({"is_dummy": is_dummy}, robot_info=info)
    build.assert_not_called()
    assert isinstance(EndEffector.of("probe"), Probe)


def test_shared_hand_uses_its_owners_lifecycle_and_reset():
    from rlinf.robotics import Connection, MethodEndEffector, Robot

    class Session(Connection):
        def __init__(self):
            self.positions = np.zeros(3)
            self.events = []

        def _open(self):
            self.events.append("open")
            return object()

        def _release(self, device):
            self.events.append("release")

        def get_state(self):
            return {"fingers": self.positions}

        def move(self, target):
            self.positions = np.asarray(target)

    class HandView(MethodEndEffector, BaseHand):
        pass

    session = Session()
    tool = HandView(session, "fingers", dims=3, command="move")
    robot = Robot(hand=tool)
    try:
        robot.connect()
        robot.connect()
        tool.reset(np.array([0.2, 0.4, 0.6]))
        tool.disconnect()
        assert session.is_connected
        np.testing.assert_allclose(tool.get_state(), [0.2, 0.4, 0.6])
        assert tool.is_hand
    finally:
        robot.disconnect()
        robot.disconnect()
    assert session.events == ["open", "release"]
