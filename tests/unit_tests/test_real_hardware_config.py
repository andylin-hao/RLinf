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

"""Hardware enumeration and task configuration must have one boundary."""

import dataclasses
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import pytest

from rlinf.envs.real import load_tasks
from rlinf.robotics.discovery import RobotDiscovery


@pytest.mark.parametrize(
    "path",
    [
        "evaluations/realworld/realworld_pnp_eval.yaml",
        "evaluations/realworld/realworld_pnp_eval_dreamzero.yaml",
        "evaluations/realworld/realworld_pnp_eval_pi05_sft_RTC.yaml",
        "examples/embodiment/config/realworld_pnp_dagger_openpi.yaml",
        "examples/embodiment/config/realworld_pnp_rlpd_cnn_async.yaml",
        "examples/reward/config/realworld_teleop.yaml",
    ],
)
def test_pnp_examples_discover_cameras_without_serial_placeholders(path):
    import yaml
    from robot_mocks import mocked_sdks

    from rlinf.robotics import FrankaConfig
    from rlinf.robotics.parts.cameras import BaseCamera

    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / path).read_text())
    entries = [
        config
        for group in doc["cluster"]["node_groups"]
        if group.get("hardware", {}).get("type") == "Franka"
        for config in group["hardware"]["configs"]
    ]
    assert entries
    with mocked_sdks():
        discovered = sorted(BaseCamera.backend("realsense").discover())
        assert discovered
        for entry in entries:
            assert "camera_serials" not in entry
            config = FrankaConfig(**entry)
            resources = RobotDiscovery.registry["Franka"].discovery_cls.enumerate(
                config.node_rank, [config]
            )
            assert resources.infos[0].config.camera_serials == discovered


def test_shipped_realworld_task_overrides_contain_no_hardware_fields():
    import yaml

    load_tasks()
    hardware_fields = {
        field.name
        for registration in RobotDiscovery.registry.values()
        for field in dataclasses.fields(registration.config_cls)
    }
    root = Path(__file__).resolve().parents[2]
    offenders = []

    def check(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "override_cfg" and isinstance(value, dict):
                    overlap = hardware_fields.intersection(value)
                    if overlap:
                        offenders.append((str(path.relative_to(root)), sorted(overlap)))
                check(value, path)
        elif isinstance(node, list):
            for value in node:
                check(value, path)

    for directory in (
        "examples",
        "evaluations",
        "tests",
    ):
        for path in (root / directory).rglob("*.yaml"):
            check(yaml.safe_load(path.read_text()), path)
    assert offenders == []


@pytest.mark.parametrize(
    "robot_type,env_id,hardware,frames,action_width",
    [
        (
            "Franka",
            "FrankaEnv-v1",
            {"camera_serials": ["MOCK0001"], "end_effector_type": "ruiyan_hand"},
            ["wrist_1"],
            12,
        ),
        (
            "DualFranka",
            "DualFrankaJointEnv-v1",
            {
                "base_camera_serials": ["MOCK0001"],
                "left_camera_serials": ["MOCK0002"],
                "right_camera_serials": [],
            },
            ["base_0_rgb", "left_wrist_0_rgb"],
            16,
        ),
        (
            "SO101",
            "SO101ReachEnv-v1",
            {
                "serial_port": "/dev/bench",
                "calibration_id": "bench",
                "camera_serials": ["MOCK0001", "MOCK0002"],
            },
            ["wrist_1", "wrist_2"],
            6,
        ),
        (
            "Piper",
            "PiperReachEnv-v1",
            {"model": "piper_h", "with_gripper": False, "camera_serials": []},
            [],
            6,
        ),
        (
            "GimArm",
            "GimArmPegInsertionEnv-v1",
            {
                "arm_variant": "gim_arm",
                "enable_gripper": False,
                "camera_serials": ["MOCK0001"],
            },
            ["wrist_1"],
            7,
        ),
        (
            "DOSW1",
            "DOSW1PickEnv-v1",
            {"robot_url": "bench", "camera_serials": ["MOCK0001"]},
            ["cam_front"],
            14,
        ),
        ("Turtle2", "ButtonEnv-v1", {"camera_ids": [0, 2]}, ["wrist_1", "wrist_2"], 7),
    ],
)
def test_task_schema_uses_enumerated_hardware_without_changing_it(
    robot_type, env_id, hardware, frames, action_width
):
    from robot_mocks import mocked_sdks

    load_tasks()
    registration = RobotDiscovery.registry[robot_type]
    config = registration.config_cls(node_rank=0, **hardware)
    with mocked_sdks():
        resources = registration.discovery_cls.enumerate(0, [config])
        info = resources.infos[0]
        before = pickle.dumps(info)
        env = gym.make(
            env_id,
            override_cfg={"is_dummy": True, "enable_camera_player": False}
            if robot_type != "Turtle2"
            else {"is_dummy": True},
            worker_info=None,
            robot_info=info,
            env_idx=0,
            env_cfg={
                "teleop": "none",
                "no_gripper": False,
                "use_relative_frame": False,
            },
        )
        try:
            observation, _ = env.reset(seed=3)
            assert sorted(observation.get("frames", {})) == sorted(frames)
            assert env.action_space.shape == (action_width,)
            assert pickle.dumps(info) == before
            assert not set(hardware) & {
                field.name for field in dataclasses.fields(env.unwrapped.config)
            }
            if robot_type == "Piper":
                assert info.model == "Piper"
                assert info.config.model == "piper_h"
        finally:
            env.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", "/dev/wrong-arm"),
        ("serial_port", "/dev/wrong-arm"),
        ("camera_serials", ["wrong-camera"]),
        ("calibration_id", "wrong-id"),
    ],
)
def test_task_overrides_cannot_redirect_the_allocated_robot(field, value):
    from rlinf.envs.real.so101 import SO101ReachEnv

    with pytest.raises(TypeError, match=field):
        SO101ReachEnv({"is_dummy": True, field: value})


def test_env_rejects_wrong_hardware_before_opening_an_arm(monkeypatch):
    from rlinf.envs.real.so101 import SO101ReachEnv
    from rlinf.robotics import PiperConfig, RobotInfo, SO101Robot

    build = Mock()
    monkeypatch.setattr(SO101Robot, "build", build)
    info = RobotInfo(type="Piper", model="Piper", config=PiperConfig(node_rank=0))
    with pytest.raises(TypeError, match="Expected SO101Config"):
        SO101ReachEnv({}, robot_info=info)
    build.assert_not_called()


def test_real_env_requires_a_robot_descriptor(monkeypatch):
    from rlinf.envs.real.so101 import SO101ReachEnv
    from rlinf.robotics import SO101Robot

    build = Mock()
    monkeypatch.setattr(SO101Robot, "build", build)
    with pytest.raises(ValueError, match="Supply robot_info"):
        SO101ReachEnv({})
    build.assert_not_called()


@pytest.mark.parametrize("controller_node_rank", [None, 7])
def test_franka_preserves_hardware_and_placement_at_construction(
    monkeypatch, controller_node_rank
):
    from rlinf.envs.real.franka import FrankaEnv
    from rlinf.robotics import FrankaConfig, FrankaRobot, RobotInfo

    config = FrankaConfig(
        node_rank=3,
        robot_ip="bench",
        controller_node_rank=controller_node_rank,
        camera_node_rank=5,
        camera_type="zed",
        camera_serials=["camera"],
        end_effector_type="ruiyan_hand",
        end_effector_config={"port": "/dev/hand"},
    )
    info = RobotInfo(type="Franka", model="Franka", config=config)
    before = pickle.dumps(info)
    build = Mock(side_effect=RuntimeError("stop before opening hardware"))
    monkeypatch.setattr(FrankaRobot, "build", build)
    with pytest.raises(RuntimeError, match="stop before opening hardware"):
        FrankaEnv(
            {},
            robot_info=info,
            worker_info=SimpleNamespace(cluster_node_rank=3, rank=2),
            env_idx=4,
        )
    kwargs = build.call_args.kwargs
    assert kwargs["robot_ip"] == "bench"
    assert kwargs["node_rank"] == (3 if controller_node_rank is None else 7)
    assert kwargs["camera_node_rank"] == 5
    assert kwargs["worker_rank"] == 2
    assert kwargs["env_idx"] == 4
    assert kwargs["end_effector_type"] == "ruiyan_hand"
    assert kwargs["end_effector_config"] == {"port": "/dev/hand"}
    camera = kwargs["cameras"]["wrist_1"]
    assert camera.serial_number == "camera"
    assert camera.camera_type == "zed"
    assert pickle.dumps(info) == before


@pytest.mark.parametrize("has_robot", [False, True])
def test_worker_passes_robot_descriptors_and_allows_dummy_cpu_placement(has_robot):
    from omegaconf import OmegaConf

    from rlinf.envs.real import RealWorldEnv
    from rlinf.robotics import RobotInfo, SO101Config
    from rlinf.scheduler.hardware import HardwareInfo

    allocated = (
        RobotInfo(
            type="SO101",
            model="SO101",
            config=SO101Config(node_rank=0, camera_serials=["camera"]),
        )
        if has_robot
        else HardwareInfo(type="CPU", model="CPU")
    )
    wrapper = RealWorldEnv.__new__(RealWorldEnv)
    wrapper.worker_info = SimpleNamespace(
        hardware_infos=[allocated], cluster_node_rank=0, rank=0
    )
    wrapper.override_cfg = {"is_dummy": True, "enable_camera_player": False}
    wrapper.cfg = OmegaConf.create(
        {"init_params": {"id": "SO101ReachEnv-v1"}, "teleop": "none"}
    )
    env = wrapper._create_env(0)
    try:
        observation, _ = env.reset()
        assert bool(observation.get("frames")) is has_robot
        assert env.unwrapped.robot_info is (allocated if has_robot else None)
    finally:
        env.close()


@pytest.fixture
def so101_tool(monkeypatch):
    from robot_mocks import mocked_sdks

    from rlinf.envs.real import so101
    from rlinf.robotics import SO101Config
    from toolkits.realworld_check import test_so101_env as tool

    for field in dataclasses.fields(SO101Config):
        monkeypatch.delenv(field.name.upper(), raising=False)
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(so101, "SO101ReachEnv", constructor)
    monkeypatch.setattr(tool, "drive", Mock())

    def run(*flags):
        monkeypatch.setattr(sys, "argv", ["test_so101_env", *flags])
        with mocked_sdks():
            tool.main()

    return run, constructor


@pytest.mark.parametrize(
    "flags,port,calibration,player",
    [
        ([], "/dev/from-env", "from-env", False),
        (
            ["--port", "/dev/from-env", "--id", "from-cli", "--enable-camera-player"],
            "/dev/from-env",
            "from-cli",
            True,
        ),
    ],
)
def test_so101_tool_enumerates_environment_and_cli_settings(
    monkeypatch, so101_tool, flags, port, calibration, player
):
    monkeypatch.setenv("SERIAL_PORT", "/dev/from-env")
    monkeypatch.setenv("CALIBRATION_ID", "from-env")
    monkeypatch.setenv("CAMERA_SERIALS", "MOCK0001,MOCK0002")
    monkeypatch.setenv("MAX_RELATIVE_TARGET", "12")
    run, constructor = so101_tool
    run(*flags)
    config = constructor.call_args.kwargs["robot_info"].config
    assert config.serial_port == port
    assert config.calibration_id == calibration
    assert config.camera_serials == ["MOCK0001", "MOCK0002"]
    assert config.max_relative_target == 12
    assert constructor.call_args.args[0]["enable_camera_player"] is player
    constructor.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("explicit_port", [False, True])
def test_so101_tool_ignores_unrelated_process_environment(
    monkeypatch, so101_tool, explicit_port
):
    from rlinf.robotics import SO101Config

    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("ID", "unrelated-service-id")
    run, constructor = so101_tool
    run(*(["--port", "/dev/ttyACM1", "--id", "follower"] if explicit_port else []))
    config = constructor.call_args.kwargs["robot_info"].config
    assert config.serial_port == (
        "/dev/ttyACM1" if explicit_port else SO101Config(node_rank=0).serial_port
    )
    assert config.calibration_id == ("follower" if explicit_port else None)


@pytest.mark.parametrize("ports", [None, "/dev/ttyACM0", "/dev/ttyACM0,/dev/ttyACM1"])
def test_so101_tool_rejects_a_port_not_in_the_configured_rig(
    monkeypatch, so101_tool, ports, capsys
):
    monkeypatch.setenv(
        "CALIBRATION_ID",
        "leader,follower" if ports and "," in ports else "leader",
    )
    if ports is not None:
        monkeypatch.setenv("SERIAL_PORT", ports)
    run, constructor = so101_tool
    with pytest.raises(SystemExit) as error:
        run("--port", "/dev/typo")
    assert error.value.code == 2
    assert "No configured SO-101 matches --port" in capsys.readouterr().err
    constructor.assert_not_called()


def test_so101_tool_selects_a_port_with_its_own_calibration(monkeypatch, so101_tool):
    monkeypatch.setenv("SERIAL_PORT", "/dev/ttyACM0,/dev/ttyACM1")
    monkeypatch.setenv("CALIBRATION_ID", "leader,follower")
    monkeypatch.setenv("MAX_RELATIVE_TARGET", "5,12")
    run, constructor = so101_tool
    run("--port", "/dev/ttyACM1")
    config = constructor.call_args.kwargs["robot_info"].config
    assert (config.serial_port, config.calibration_id, config.max_relative_target) == (
        "/dev/ttyACM1",
        "follower",
        12,
    )


def test_so101_tool_requires_selection_with_multiple_arms(monkeypatch, so101_tool):
    monkeypatch.setenv("SERIAL_PORT", "/dev/ttyACM0,/dev/ttyACM1")
    monkeypatch.setenv("CALIBRATION_ID", "leader,follower")
    run, constructor = so101_tool
    with pytest.raises(SystemExit):
        run()
    constructor.assert_not_called()


def test_dummy_franka_requires_an_explicit_camera_layout():
    from rlinf.envs.real.franka import FrankaEnv
    from rlinf.robotics import FrankaConfig, RobotInfo

    with pytest.raises(ValueError, match="including in dummy mode"):
        FrankaEnv({"is_dummy": True})
    info = RobotInfo(
        type="Franka",
        model="Franka",
        config=FrankaConfig(node_rank=0, camera_serials=["dummy"]),
    )
    env = FrankaEnv({"is_dummy": True}, robot_info=info)
    try:
        observation, _ = env.reset()
        assert observation in env.observation_space
        assert list(observation["frames"]) == ["wrist_1"]
        assert env.robot is None
    finally:
        env.close()


def test_dual_franka_hardware_defaults_build_supported_franky_grippers():
    from rlinf.robotics import DualFrankaConfig, DualFrankaRobot
    from rlinf.robotics.parts.end_effectors import EndEffector
    from rlinf.robotics.parts.end_effectors.grippers.franky import FrankyGripper

    config = DualFrankaConfig(node_rank=0)
    assert config.left_gripper_type == config.right_gripper_type == "franka"
    robot = DualFrankaRobot.build(
        left_robot_ip="192.0.2.1",
        right_robot_ip="192.0.2.2",
        left_gripper_type=config.left_gripper_type,
        right_gripper_type=config.right_gripper_type,
    )
    grippers = robot.parts_of_type(EndEffector)
    assert len(grippers) == 2
    assert all(isinstance(gripper, FrankyGripper) for gripper in grippers.values())


def test_scheduler_hardware_passes_plain_nested_settings_to_the_hand(monkeypatch):
    from omegaconf import OmegaConf
    from robot_mocks import mocked_sdks

    from rlinf.robotics import FrankaRobot
    from rlinf.scheduler.hardware import NodeHardwareConfig

    source = OmegaConf.create(
        {
            "type": "Franka",
            "configs": [
                {
                    "node_rank": 0,
                    "end_effector_type": "ruiyan_hand",
                    "end_effector_config": {
                        "port": "/dev/hand",
                        "motor_ids": [1, 2, 3, 4, 5, 6],
                        "default_state": [0.0] * 6,
                    },
                }
            ],
        }
    )
    hardware = NodeHardwareConfig(**source).configs[0]
    settings = hardware.end_effector_config
    assert type(settings) is dict
    assert type(settings["motor_ids"]) is list
    assert type(settings["default_state"]) is list
    with mocked_sdks():
        import rlinf_dexhand.ruiyan as sdk

        driver = Mock()
        monkeypatch.setattr(sdk, "RuiyanHandDriver", driver)
        hand = FrankaRobot.declare_end_effector(
            robot_ip="192.0.2.1",
            node_rank=None,
            name="test-hand",
            end_effector_type=hardware.end_effector_type,
            end_effector_config=settings,
        )
        hand.connect()
        try:
            for name, value in settings.items():
                assert driver.call_args.kwargs[name] == value
            assert type(driver.call_args.kwargs["motor_ids"]) is list
            assert type(driver.call_args.kwargs["default_state"]) is list
        finally:
            hand.disconnect()
