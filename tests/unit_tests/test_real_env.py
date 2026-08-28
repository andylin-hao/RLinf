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

"""Tests for real-world tasks, teleoperation, configuration, and layout."""

from __future__ import annotations

import importlib
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.real.dosw1.base import DOSW1Config, DOSW1Env
from rlinf.envs.real.franka.base import FrankaEnv
from rlinf.envs.real.franka.dual_franka_joint import (
    DualFrankaJointEnv,
)
from rlinf.envs.real.gim_arm.base import GimArmEnv, GimArmRobotConfig
from rlinf.envs.real.task_env import RobotTask, RobotTaskEnv
from rlinf.envs.real.wrappers.teleop.config import (  # noqa: E402
    NO_DEVICE,
    resolve_teleop_device,
    resolve_teleop_devices,
)
from rlinf.envs.real.wrappers.teleop.intervention import (  # noqa: E402
    TeleopDevice,
    TeleopIntervention,
    TeleopSample,
)
from rlinf.envs.real.xsquare.base import Turtle2Env, Turtle2RobotConfig
from rlinf.robotics import ControllablePart, PartGroup, Robot

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class DummyDriver(ControllablePart):
    def __init__(self) -> None:
        self.connected = False
        self.last_action: dict[str, Any] | None = None

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def observation_features(self) -> dict[str, Any]:
        return {"position": {"shape": (1,)}}

    @property
    def action_features(self) -> dict[str, Any]:
        return {"target": {"shape": (1,)}}

    def connect(self) -> None:
        self.connected = True

    def reset(self) -> None:
        self.last_action = None

    def get_observation(self) -> dict[str, Any]:
        return {"position": np.zeros(1)}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.last_action = action
        return action

    def disconnect(self) -> None:
        self.connected = False


class DummyTask(RobotTask):
    @property
    def description(self) -> str:
        return "Move the test arm."

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {"position": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)}
        )

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "arms": gym.spaces.Dict(
                    {
                        "arm": gym.spaces.Dict(
                            {
                                "arm": gym.spaces.Dict(
                                    {
                                        "target": gym.spaces.Box(
                                            -1.0,
                                            1.0,
                                            shape=(1,),
                                            dtype=np.float32,
                                        )
                                    }
                                )
                            }
                        )
                    }
                )
            }
        )

    def reset(
        self,
        robot: Robot,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed, options
        robot.reset()
        return {"position": np.zeros(1, dtype=np.float32)}, {}

    def step(
        self,
        robot: Robot,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        robot.send_action(action)
        return {"position": np.ones(1, dtype=np.float32)}, 1.0, True, False, {}


def test_robot_task_env_composes_task_and_robot_lifecycles():
    driver = DummyDriver()
    robot = Robot(arm=PartGroup(arm=driver))
    env = RobotTaskEnv(robot, DummyTask())
    action = {"arm": {"arm": {"target": np.array([0.5])}}}

    observation, _ = env.reset(seed=3)
    transition = env.step(action)

    assert env.task_description == "Move the test arm."
    assert observation["position"].tolist() == [0.0]
    assert transition[0]["position"].tolist() == [1.0]
    assert driver.last_action is not None
    assert driver.last_action["target"].tolist() == [0.5]
    env.close()
    assert not driver.is_connected


def _assert_legacy_transition(env) -> None:
    observation, _ = env.reset()
    transition = env.step(env.action_space.sample())

    assert set(observation) == {"state", "frames"}
    assert set(transition[0]) == {"state", "frames"}
    assert len(transition) == 5
    env.close()


def test_franka_dummy_preserves_legacy_policy_schema():
    env = FrankaEnv(
        override_cfg={
            "is_dummy": True,
            "camera_serials": ["dummy"],
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_dual_franka_dummy_preserves_legacy_policy_schema():
    env = DualFrankaJointEnv(
        override_cfg={
            "is_dummy": True,
            "base_camera_serials": ["dummy"],
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (16,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_gim_arm_dummy_preserves_legacy_policy_schema():
    env = GimArmEnv(
        config=GimArmRobotConfig(
            is_dummy=True,
            camera_serials=[],
            enable_camera_player=False,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_dosw1_dummy_preserves_legacy_policy_schema():
    env = DOSW1Env(
        config=DOSW1Config(
            is_dummy=True,
            camera_serials=[],
            camera_names=[],
            enable_camera_player=False,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (14,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_turtle2_dummy_preserves_legacy_policy_schema():
    env = Turtle2Env(
        config=Turtle2RobotConfig(
            is_dummy=True,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_franka_builds_cameras_after_applying_hardware_info(monkeypatch):
    from rlinf.envs.real.franka.base import FrankaRobotConfig
    from rlinf.robotics import FrankaConfig, RobotInfo
    from rlinf.robotics.robots.franka import FrankaRobot

    captured = {}

    class BuiltRobot:
        def connect(self):
            pass

        def child(self, name):
            # The env reaches for the arm and, beside it, the end effector.
            assert name in ("arm", "end_effector")
            return SimpleNamespace(owner=object())

    def build(**kwargs):
        captured.update(kwargs)
        return BuiltRobot()

    monkeypatch.setattr(FrankaRobot, "build", build)
    env = FrankaEnv.__new__(FrankaEnv)
    env.config = FrankaRobotConfig(camera_serials=None, camera_type=None)
    env.robot_info = RobotInfo(
        type="Robot",
        model="Franka",
        config=FrankaConfig(
            node_rank=0,
            robot_ip="10.0.0.1",
            camera_serials=["hardware-camera"],
        ),
    )
    env.env_idx = 0
    env.node_rank = 0
    env.env_worker_rank = 3

    env._setup_hardware()

    assert [info.serial_number for info in env._camera_infos] == ["hardware-camera"]
    assert list(captured["cameras"]) == ["wrist_1"]


def test_gim_arm_reopens_the_existing_camera_after_a_stall(monkeypatch):
    from rlinf.robotics.parts.cameras import CameraInfo

    class Camera:
        def __init__(self):
            self._camera_info = CameraInfo("wrist_1", "camera")
            self.reads = 0
            self.reopens = 0

        def get_frame(self):
            self.reads += 1
            if self.reads == 1:
                raise __import__("queue").Empty
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def reopen(self):
            self.reopens += 1

    env = GimArmEnv.__new__(GimArmEnv)
    camera = Camera()
    env._cameras = [camera]
    env._logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
    env.observation_space = gym.spaces.Dict(
        {
            "frames": gym.spaces.Dict(
                {"wrist_1": gym.spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8)}
            )
        }
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    frames = env._get_camera_frames()

    assert camera.reopens == 1
    assert frames["wrist_1"].shape == (4, 4, 3)


def test_dual_franka_runs_independent_arm_calls_concurrently():
    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    rendezvous = threading.Barrier(2)

    def call(side):
        rendezvous.wait(timeout=1.0)
        return side

    try:
        assert env._run_arm_calls(lambda: call("left"), lambda: call("right")) == (
            "left",
            "right",
        )
    finally:
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


def test_dual_franka_does_not_wait_for_gripper_motion():
    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env.config = SimpleNamespace(binary_gripper_threshold=0.5)
    env._logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    entered = threading.Event()
    release = threading.Event()

    class Hand:
        is_open = True

        def close(self):
            entered.set()
            assert release.wait(timeout=1.0)

    try:
        changed = env._gripper_action(0, Hand(), -1.0)
        assert changed
        assert entered.wait(timeout=1.0)
        assert not release.is_set(), "the control loop must not wait for the gripper"
    finally:
        release.set()
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


def test_franka_reward_model_waits_for_the_worker_result():
    class Work:
        def wait(self):
            return [np.array([0.75], dtype=np.float32)]

    env = FrankaEnv.__new__(FrankaEnv)
    env.config = SimpleNamespace(reward_image_key=None)
    env._reward_worker = SimpleNamespace(compute_reward=lambda _batch: Work())

    reward = env._compute_reward_model(
        {"frames": {"wrist_1": np.zeros((4, 4, 3), dtype=np.uint8)}}
    )

    assert reward == pytest.approx(0.75)


def test_direct_gello_stream_keeps_both_arm_commands_concurrent():
    from rlinf.envs.real.wrappers.teleop.adapters import DualGelloJointStream

    rendezvous = threading.Barrier(2)

    class Controller:
        def move_joints(self, _target):
            rendezvous.wait(timeout=1.0)

    class Leader:
        ready = True

        def get_observation(self):
            return {"joint_position": np.zeros(7), "grip": np.zeros(1)}

    env = DualFrankaJointEnv.__new__(DualFrankaJointEnv)
    env._left_ctrl = Controller()
    env._right_ctrl = Controller()
    env._arm_executors = (
        ThreadPoolExecutor(max_workers=1),
        ThreadPoolExecutor(max_workers=1),
    )
    streamer = DualGelloJointStream(
        Leader(), Leader(), gripper_enabled=False, direct_stream=False
    )

    try:
        streamer.stream_once(env)
    finally:
        for executor in env._arm_executors:
            executor.shutdown(wait=True)


class FakeEnv:
    """Record actions passed to ``step``."""

    def __init__(self) -> None:
        self.stepped: list[np.ndarray] = []
        self.reset_calls = 0
        self.closed = False

    def step(self, action):
        self.stepped.append(np.asarray(action))
        return {"obs": 1}, 0.0, False, False, {}

    def reset(self, **kwargs):
        self.reset_calls += 1
        return {"obs": 1}, {}

    def close(self):
        self.closed = True


class ScriptedDevice(TeleopDevice):
    """Return one scripted sample per read."""

    def __init__(self, samples: list[TeleopSample]) -> None:
        self.samples = list(samples)
        self.reads = 0
        self.resets = 0
        self.closed = False
        self.before_steps = 0
        self.fallback_action: Optional[np.ndarray] = None

    def read(self, env: Any, policy_action: np.ndarray) -> TeleopSample:
        sample = self.samples[min(self.reads, len(self.samples) - 1)]
        self.reads += 1
        return sample

    def reset(self, env: Any) -> None:
        self.resets += 1

    def before_step(self, env: Any) -> None:
        self.before_steps += 1

    def fallback(self, env: Any, policy_action: np.ndarray) -> np.ndarray:
        if self.fallback_action is not None:
            return self.fallback_action
        return policy_action

    def close(self) -> None:
        self.closed = True


POLICY = np.array([0.0, 0.0, 0.0])


EXPERT = np.array([1.0, 1.0, 1.0])


def test_active_sample_replaces_the_policy_action():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=EXPERT, active=True)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], EXPERT)
    assert np.array_equal(info["intervene_action"], EXPERT)


def test_inactive_device_leaves_the_policy_action_alone():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=None, active=False)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], POLICY)
    assert "intervene_action" not in info


def test_control_is_held_between_samples_then_released():
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)  # Operator moves.
    wrapper.step(POLICY)  # Quiet sample within the hold window.
    assert np.array_equal(env.stepped[1], EXPERT)

    device.timeout = 0.0  # Hold window expires.
    wrapper.step(POLICY)
    assert np.array_equal(env.stepped[2], POLICY)


def test_an_unfilled_part_keeps_the_policy_action():
    import numpy as np

    from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
    from rlinf.robotics.teleop import (
        ActionKind,
        TeleopBinding,
        TeleopEntry,
        TeleopGroup,
    )

    class Fixed(TeleopBinding):
        PRODUCES = {"hand": ActionKind.HAND}

        def action(self, reading, context):
            from rlinf.robotics.teleop import TeleopAction

            return TeleopAction(parts={"hand": np.full(6, 0.5)}, driving=True)

        def reset(self):
            pass

    class Device:
        is_connected = True

        def get_observation(self):
            return {}

        def connect(self):
            pass

        def disconnect(self):
            pass

    layout = {"arm": slice(0, 6), "hand": slice(6, 12)}
    group = TeleopGroup(
        [TeleopEntry(Device(), Fixed())],
        available={"arm": ActionKind.CARTESIAN_DELTA, "hand": ActionKind.HAND},
    )
    device = ComposedTeleop(group, layout)

    policy = np.arange(12, dtype=np.float64)
    sample = device.read(_FakeLayoutEnv(), policy)

    assert np.allclose(sample.action[:6], policy[:6])  # Arm remains policy-driven.
    assert np.allclose(sample.action[6:], 0.5)  # Glove controls the hand.


def test_an_idle_glove_keeps_its_hand_pose_without_claiming_the_arm():
    from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
    from rlinf.robotics.teleop import (
        ActionKind,
        TeleopAction,
        TeleopBinding,
        TeleopEntry,
        TeleopGroup,
    )

    class HeldHand(TeleopBinding):
        PRODUCES = {"hand": ActionKind.HAND}
        APPLIES_WHILE_IDLE = True

        def action(self, reading, context):
            return TeleopAction(parts={"hand": np.full(6, 0.5)}, driving=False)

    class Device:
        is_connected = True

        def get_observation(self):
            return {}

    layout = {"arm": slice(0, 6), "hand": slice(6, 12)}
    group = TeleopGroup(
        [TeleopEntry(Device(), HeldHand())],
        available={"arm": ActionKind.CARTESIAN_DELTA, "hand": ActionKind.HAND},
    )
    policy = np.arange(12, dtype=np.float64)
    sample = ComposedTeleop(group, layout).read(_FakeLayoutEnv(), policy)
    env = FakeEnv()

    _, _, _, _, info = TeleopIntervention(env, ScriptedDevice([sample])).step(policy)

    assert np.allclose(env.stepped[0][:6], policy[:6])
    assert np.allclose(env.stepped[0][6:], 0.5)
    assert "intervene_action" not in info


class _FakeLayoutEnv:
    """Provide the attributes required by teleoperation layout tests."""

    unwrapped = None

    def get_wrapper_attr(self, name):
        raise AttributeError(name)


def test_mark_flag_is_opt_in():
    sample = TeleopSample(action=EXPERT, active=True)

    plain = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]))
    flagged = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]), mark_flag=True)

    assert "intervene_flag" not in plain.step(POLICY)[4]
    assert flagged.step(POLICY)[4]["intervene_flag"] == np.ones(1)


def test_device_info_reaches_the_step_info():
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env,
        ScriptedDevice([TeleopSample(action=EXPERT, active=True, info={"left": True})]),
    )

    assert wrapper.step(POLICY)[4]["left"] is True


def test_reset_resyncs_the_device_and_drops_the_hold():
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)
    assert wrapper.intervening

    wrapper.reset()

    assert device.resets == 1
    assert env.reset_calls == 1
    assert not wrapper.intervening


def test_close_releases_the_device_before_the_env():
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.close()

    assert device.closed
    assert env.closed


def test_before_step_runs_ahead_of_the_env():
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)

    assert device.before_steps == 1


def test_read_is_abstract():
    with pytest.raises(TypeError):
        TeleopDevice()


SINGLE_ARM = ("spacemouse", "gello", "pico")


def test_retired_single_key_still_selects_its_device_and_warns():
    with pytest.warns(DeprecationWarning, match="'teleop_device' is retired"):
        device = resolve_teleop_device({"teleop_device": "gello"}, supported=SINGLE_ARM)

    assert device == "gello"


def test_missing_config_falls_back_to_the_default():
    assert resolve_teleop_device({}, supported=SINGLE_ARM) == NO_DEVICE
    assert (
        resolve_teleop_device({}, supported=SINGLE_ARM, default="spacemouse")
        == "spacemouse"
    )


def test_retired_boolean_still_selects_its_device_and_warns():
    with pytest.warns(DeprecationWarning, match="use_pico"):
        device = resolve_teleop_device({"use_pico": True}, supported=SINGLE_ARM)

    assert device == "pico"


def test_all_retired_booleans_off_means_no_device():
    with pytest.warns(DeprecationWarning):
        device = resolve_teleop_device(
            {"use_spacemouse": False, "use_gello": False},
            supported=SINGLE_ARM,
            default="spacemouse",
        )

    assert device == NO_DEVICE


def test_two_retired_booleans_on_is_an_error():
    with pytest.raises(ValueError, match="Only one teleop device"):
        resolve_teleop_device(
            {"use_spacemouse": True, "use_pico": True}, supported=SINGLE_ARM
        )


def test_disagreeing_old_and_new_keys_are_refused():
    with pytest.raises(ValueError, match="cannot be reconciled"):
        resolve_teleop_device(
            {"teleop_device": "pico", "use_spacemouse": True},
            supported=SINGLE_ARM,
        )


def test_agreeing_old_and_new_keys_only_warn():
    with pytest.warns(DeprecationWarning, match="redundant"):
        device = resolve_teleop_device(
            {"teleop_device": "pico", "use_pico": True}, supported=SINGLE_ARM
        )

    assert device == "pico"


def test_device_the_env_cannot_drive_is_refused():
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_device(
            {"teleop_device": "spacemouse"}, supported=("gello_joint", "pico")
        )


def test_none_is_always_allowed():
    with pytest.warns(DeprecationWarning):
        device = resolve_teleop_device({"teleop_device": "none"}, supported=("pico",))

    assert device == NO_DEVICE


def test_shipped_configs_use_the_new_key():
    roots = [_ROOT / "examples", _ROOT / "evaluations", _ROOT / "tests"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.yaml"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if any(
                    stripped.startswith(f"{flag}:")
                    for flag in (
                        "teleop_device",
                        "use_spacemouse",
                        "use_gello",
                        "use_gello_joint",
                        "use_pico",
                    )
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


_REAL = _ROOT / "rlinf" / "envs" / "real"


_ROBOTS = ("franka", "dosw1", "gim_arm", "xsquare")


EXPECTED_IDS = {
    "FrankaEnv-v1",
    "PegInsertionEnv-v1",
    "FrankaBinRelocationEnv-v1",
    "BottleEnv-v1",
    "DexpnpEnv-v1",
    "DualFrankaJointEnv-v1",
    "DualFrankaTCPEnv-v1",
    "DOSW1PickEnv-v1",
    "ButtonEnv-v1",
    "GimArmPegInsertionEnv-v1",
}


def test_no_robot_keeps_a_tasks_subpackage():
    leftovers = [name for name in _ROBOTS if (_REAL / name / "tasks").exists()]

    assert leftovers == []


def test_every_robot_folder_has_a_base():
    missing = [name for name in _ROBOTS if not (_REAL / name / "base.py").exists()]

    assert missing == []


def test_all_task_ids_are_registered():
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    assert EXPECTED_IDS <= set(registry)


def test_every_entry_point_resolves():
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    unresolved = []
    for env_id in sorted(EXPECTED_IDS):
        entry_point = registry[env_id].entry_point
        module_name, _, attribute = str(entry_point).partition(":")
        module = importlib.import_module(module_name)
        if getattr(module, attribute, None) is None:
            unresolved.append(f"{env_id} -> {entry_point}")

    assert unresolved == []


def test_task_tables_cover_the_wrapped_robots():
    from rlinf.envs.real import dosw1, franka, xsquare

    declared = set(franka.TASKS) | set(dosw1.TASKS) | set(xsquare.TASKS)

    # GimArm registers its environment class without a wrapper factory.
    assert declared == EXPECTED_IDS - {"GimArmPegInsertionEnv-v1"}


def test_pose_math_is_not_filed_under_a_robot():
    from rlinf.envs.real.utils import pose

    assert hasattr(pose, "construct_adjoint_matrix")
    assert not (_REAL / "franka" / "utils.py").exists()


def test_task_configs_state_only_their_compliance_deltas():
    from rlinf.envs.real.franka.base import COMPLIANCE_DEFAULTS
    from rlinf.envs.real.franka.bin_relocation import BinEnvConfig
    from rlinf.envs.real.franka.bottle import BottleConfig
    from rlinf.envs.real.franka.dex_pnp import DexpnpConfig
    from rlinf.envs.real.franka.peg_insertion import PegInsertionConfig

    deltas = {
        cls.__name__: {
            key
            for key, value in cls().compliance_param.items()
            if COMPLIANCE_DEFAULTS[key] != value
        }
        for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig)
    }

    # Every task receives the complete gain set after defaults are applied.
    for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig):
        assert set(cls().compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert {name: len(keys) for name, keys in deltas.items()} == {
        "PegInsertionConfig": 1,
        "BottleConfig": 8,
        "BinEnvConfig": 11,
        "DexpnpConfig": 6,
    }


def test_unknown_compliance_gain_is_refused():
    import pytest

    from rlinf.envs.real.franka.base import compliance

    with pytest.raises(KeyError, match="Unknown compliance gains"):
        compliance(translational_stifness=1000)


# Wrapper families


def test_wrappers_are_split_by_what_they_change():
    real = _ROOT / "rlinf" / "envs" / "real"
    wrappers = real / "wrappers"

    assert wrappers.is_dir(), "the three families live under one parent"
    for family in ("teleop", "transforms", "episode"):
        assert (wrappers / family / "__init__.py").exists(), family

    # Top-level modules contain robot packages and shared environment machinery.
    loose = sorted(
        path.stem for path in real.glob("*.py") if path.name != "__init__.py"
    )
    assert loose == ["env", "registry", "task_env", "venv"], loose


def test_no_teleop_wrapper_is_left_outside_teleop():
    real = _ROOT / "rlinf" / "envs" / "real"
    strays = sorted(
        path.name
        for family in ("transforms", "episode")
        for path in (real / "wrappers" / family).glob("*.py")
        if "intervention" in path.name and "leader_follower" not in path.name
    )

    assert strays == []


def test_a_held_button_device_does_not_keep_control_after_release():
    from rlinf.robotics.teleop import PicoBinding, PicoTcpBinding

    assert PicoBinding.HOLD_WINDOW == 0.0
    assert PicoTcpBinding.HOLD_WINDOW == 0.0


def test_streaming_device_lifecycle_without_hardware():
    from rlinf.envs.real.wrappers.teleop.intervention import TeleopSample
    from rlinf.envs.real.wrappers.teleop.streaming import TeleopStreamer

    ticks = []

    class Fake(TeleopStreamer):
        def read(self, env, policy_action):
            return TeleopSample(action=None, active=False)

        def stream_once(self, env):
            ticks.append(1)

    device = Fake(period=0.001, enabled=True)
    device.before_reset(None, {})
    device.reset(None)
    device.after_reset(None)
    deadline = time.monotonic() + 1.0
    while not ticks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ticks, "stream thread never ran"
    assert device.streaming

    device.close()

    assert not device.streaming


# Keyboard sessions


def _keyboard_session(monkeypatch, queued):
    """Build a keyboard session that replays queued key batches."""
    from rlinf.envs.real.wrappers.episode import session as session_module

    class FakeListener:
        def __init__(self):
            self.batches = list(queued)

        def pop_pressed_keys(self):
            return self.batches.pop(0) if self.batches else []

        def get_key(self):
            batch = self.pop_pressed_keys()
            return batch[0] if batch else None

    monkeypatch.setattr(session_module, "KeyboardListener", FakeListener)

    class Env:
        def __init__(self):
            self.resets = 0

        def reset(self, seed=None, options=None):
            self.resets += 1
            return {}, {}

        def step(self, action):
            return {}, 0.0, False, False, {}

    return session_module.KeyboardSession(Env())


def test_repeat_presses_within_the_debounce_window_are_dropped(monkeypatch):
    session = _keyboard_session(monkeypatch, [["a"], ["a"], ["b"]])

    assert list(session.presses()) == ["a"]
    assert list(session.presses()) == []  # Same key within the debounce window.
    assert list(session.presses()) == ["b"]  # A different key is accepted.


def test_presses_queued_between_episodes_do_not_leak(monkeypatch):
    session = _keyboard_session(monkeypatch, [["c"], ["a"]])

    session.reset()

    assert session.env.resets == 1
    assert list(session.presses()) == ["a"]  # The queued key was drained.


def test_every_keyboard_wrapper_shares_the_session(monkeypatch):
    from rlinf.envs.real.wrappers.episode import (
        KeyboardEvalControlWrapper,
        KeyboardRewardDoneMultiStageWrapper,
        KeyboardRewardDoneWrapper,
        KeyboardRLTPolicySwitchWrapper,
        KeyboardStartEndWrapper,
    )
    from rlinf.envs.real.wrappers.episode.session import KeyboardSession

    for wrapper in (
        KeyboardEvalControlWrapper,
        KeyboardRLTPolicySwitchWrapper,
        KeyboardStartEndWrapper,
        KeyboardRewardDoneWrapper,
        KeyboardRewardDoneMultiStageWrapper,
    ):
        assert issubclass(wrapper, KeyboardSession), wrapper.__name__


def test_episode_wrappers_report_through_the_logger():
    episode_dir = _ROOT / "rlinf" / "envs" / "real" / "episode"
    offenders = sorted(
        path.name
        for path in episode_dir.glob("*.py")
        if re.search(r"^\s*print\(", path.read_text(), re.M)
    )

    assert offenders == []


def test_euler_conversion_is_one_wrapper_for_any_arm_count():
    import numpy as np
    from gymnasium import spaces

    from rlinf.envs.real.wrappers.transforms import (
        DualQuat2EulerWrapper,
        Quat2EulerWrapper,
    )

    class Env:
        def __init__(self, dim):
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {"tcp_pose": spaces.Box(-np.inf, np.inf, (dim,))}
                    )
                }
            )

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    one = np.concatenate([np.array([1.0, 2.0, 3.0]), identity_quat])

    single = Quat2EulerWrapper(Env(7))
    dual = DualQuat2EulerWrapper(Env(14))

    assert single.observation_space["state"]["tcp_pose"].shape == (6,)
    assert dual.observation_space["state"]["tcp_pose"].shape == (12,)

    got = single.observation({"state": {"tcp_pose": one.copy()}})["state"]["tcp_pose"]
    assert np.allclose(got, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

    both = dual.observation({"state": {"tcp_pose": np.concatenate([one, one])}})
    assert np.allclose(both["state"]["tcp_pose"], [1.0, 2.0, 3.0, 0, 0, 0] * 2)


# Full wrapper stacks built through the production builders


def _dummy_franka(env_cls=None, **overrides):
    from rlinf.envs.real.franka.base import FrankaEnv

    cfg = {
        "is_dummy": True,
        "camera_serials": ["dummy"],
        "enable_camera_player": False,
        "step_frequency": 10000.0,
    }
    cfg.update(overrides)
    return (env_cls or FrankaEnv)(
        override_cfg=cfg, worker_info=None, robot_info=None, env_idx=0
    )


def _chain(env):
    """Return wrapper class names from outermost to innermost."""
    names = []
    while hasattr(env, "env"):
        names.append(type(env).__name__)
        env = env.env
    return names


def test_wrapper_stack_converts_the_pose_it_hands_the_policy():
    from rlinf.envs.real.wrappers import build_stack

    env = _dummy_franka()
    raw, _ = env.reset()
    assert raw["state"]["tcp_pose"].shape == (7,)

    wrapped = build_stack(
        env, {"teleop": "none", "no_gripper": False, "use_relative_frame": True}
    )
    observation, _ = wrapped.reset()

    assert _chain(wrapped) == ["Quat2EulerWrapper", "RelativeFrame"]
    assert observation["state"]["tcp_pose"].shape == (6,)
    wrapped.close()


def test_no_teleop_device_leaves_no_intervention_in_the_stack():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(),
        {"teleop": "none", "no_gripper": False, "use_relative_frame": False},
    )

    assert not any("Intervention" in name for name in _chain(wrapped))
    wrapped.close()


def test_no_gripper_narrows_the_action_the_policy_must_produce():
    from rlinf.envs.real.wrappers import build_stack

    env = _dummy_franka()
    assert env.action_space.shape == (7,)

    wrapped = build_stack(
        env,
        {"teleop": "none", "no_gripper": True, "use_relative_frame": False},
    )

    assert "GripperCloseEnv" in _chain(wrapped)
    assert wrapped.action_space.shape == (6,)
    wrapped.close()


def test_the_no_gripper_default_does_not_wrap_a_dexterous_hand():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(
            end_effector_type="ruiyan_hand",
            hand_target_state=np.zeros(6),
            hand_reset_state=np.zeros(6),
        ),
        {"teleop": "none", "use_relative_frame": False},
    )

    assert "GripperCloseEnv" not in _chain(wrapped)
    assert wrapped.action_space.shape == (12,)
    wrapped.close()


def test_gim_arm_keeps_the_unwrapped_legacy_action_and_observation_schema():
    from rlinf.envs.real.gim_arm.base import GimArmEnv, GimArmRobotConfig
    from rlinf.envs.real.wrappers import build_stack

    env = GimArmEnv(
        config=GimArmRobotConfig(is_dummy=True),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )
    wrapped = build_stack(env, {})

    assert wrapped is env
    assert wrapped.action_space.shape == (7,)
    assert wrapped.observation_space["state"]["tcp_pose"].shape == (7,)
    wrapped.close()


def test_dual_franka_keeps_the_legacy_no_gripper_default():
    from rlinf.envs.real.franka.dual_franka_joint import DualFrankaJointEnv
    from rlinf.envs.real.wrappers import build_stack

    env = DualFrankaJointEnv(
        override_cfg={
            "is_dummy": True,
            "base_camera_serials": ["dummy"],
            "left_camera_serials": [],
            "right_camera_serials": [],
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    with pytest.raises(NotImplementedError, match="no_gripper"):
        build_stack(env, {"teleop": "none"})
    env.close()


def test_a_task_env_runs_with_its_own_config():
    from rlinf.envs.real.franka.base import COMPLIANCE_DEFAULTS
    from rlinf.envs.real.franka.peg_insertion import PegInsertionEnv

    env = _dummy_franka(
        PegInsertionEnv, target_ee_pose=[0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
    )

    assert env.config.task_description == "peg and insertion"
    # Task-specific gains override shared defaults.
    assert set(env.config.compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert env.config.compliance_param["translational_stiffness"] == 2000

    env.reset()
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    assert set(observation) == {"state", "frames"}
    assert isinstance(bool(terminated), bool)
    env.close()


def test_every_registered_task_builds_through_its_entry_point():
    import gymnasium as gym

    from rlinf.envs.real import RealWorldEnv  # noqa: F401  (registers the tasks)

    cfg = {
        "teleop": "none",
        "no_gripper": False,
        "use_relative_frame": False,
    }
    built = []
    for env_id in ("FrankaEnv-v1", "PegInsertionEnv-v1", "BottleEnv-v1"):
        env = gym.make(
            env_id,
            override_cfg={
                "is_dummy": True,
                "camera_serials": ["dummy"],
                "enable_camera_player": False,
                "step_frequency": 10000.0,
            },
            worker_info=None,
            robot_info=None,
            env_idx=0,
            env_cfg=cfg,
        )
        env.reset()
        env.close()
        built.append(env_id)

    assert built == ["FrankaEnv-v1", "PegInsertionEnv-v1", "BottleEnv-v1"]


def test_converted_pose_stays_inside_the_observation_space():
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(),
        {"teleop": "none", "no_gripper": False, "use_relative_frame": False},
    )
    observation, _ = wrapped.reset()

    assert (
        observation["state"]["tcp_pose"].dtype
        == wrapped.observation_space["state"]["tcp_pose"].dtype
    )
    assert wrapped.observation_space.contains(observation)
    wrapped.close()


# Multiple teleoperation devices


class _FakeInner:
    """Provide the environment attributes read by teleoperation builders."""

    def __init__(self, **config: Any) -> None:
        self.config = SimpleNamespace(**config)


def test_one_named_device_resolves_to_one_entry():
    assert resolve_teleop_devices({"teleop": "gello"}, supported=SINGLE_ARM) == [
        "gello"
    ]


def test_saying_nothing_resolves_to_no_entries():
    assert resolve_teleop_devices({}, supported=SINGLE_ARM) == []
    assert resolve_teleop_devices({"teleop": "none"}, supported=SINGLE_ARM) == []


def test_a_list_keeps_every_device_it_names():
    assert resolve_teleop_devices(
        {"teleop": ["spacemouse", "glove"]}, supported=("spacemouse", "glove")
    ) == ["spacemouse", "glove"]


def test_an_entry_carries_its_own_options():
    entries = resolve_teleop_devices(
        {"teleop": [{"gello_joint": {"port": "/dev/left", "drives": "left"}}]},
        supported=("gello_joint",),
    )

    assert entries == [{"gello_joint": {"port": "/dev/left", "drives": "left"}}]


def test_one_device_may_appear_twice_on_different_branches():
    entries = resolve_teleop_devices(
        {
            "teleop": [
                {"gello_joint": {"drives": "left"}},
                {"gello_joint": {"drives": "right"}},
            ]
        },
        supported=("gello_joint",),
    )

    assert [entry["gello_joint"]["drives"] for entry in entries] == ["left", "right"]


def test_a_listed_device_the_env_cannot_drive_is_refused():
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_devices(
            {"teleop": ["spacemouse", "glove"]}, supported=("gello_joint", "pico")
        )


def test_a_list_supersedes_a_retired_key_underneath_it():
    with pytest.warns(DeprecationWarning, match="supersedes"):
        entries = resolve_teleop_devices(
            {"teleop": ["spacemouse"], "teleop_device": "none"},
            supported=SINGLE_ARM,
        )

    assert entries == ["spacemouse"]


def test_a_list_supersedes_a_retired_boolean():
    with pytest.warns(DeprecationWarning, match="supersedes"):
        entries = resolve_teleop_devices(
            {"teleop": ["spacemouse"], "use_pico": True}, supported=SINGLE_ARM
        )

    assert entries == ["spacemouse"]


def test_an_empty_list_is_refused():
    with pytest.raises(ValueError, match="'teleop' is empty"):
        resolve_teleop_devices({"teleop": []}, supported=SINGLE_ARM)


def test_none_cannot_share_the_list():
    with pytest.raises(ValueError, match="cannot share the list"):
        resolve_teleop_devices({"teleop": ["none", "spacemouse"]}, supported=SINGLE_ARM)


def test_none_alone_in_a_list_means_nobody_takes_over():
    assert resolve_teleop_devices({"teleop": ["none"]}, supported=SINGLE_ARM) == []


def test_a_two_key_entry_is_refused():
    with pytest.raises(ValueError, match="mapping of one name"):
        resolve_teleop_devices(
            {"teleop": [{"spacemouse": {}, "glove": {}}]},
            supported=("spacemouse", "glove"),
        )


def test_no_device_is_named_in_the_wrapper_stack():
    import inspect

    from rlinf.envs.real import wrappers

    source = inspect.getsource(wrappers)
    for device in ("spacemouse", "glove", "gello_joint", "pico"):
        assert device not in source, f"the wrapper stack still names {device!r}"


def test_a_leader_arm_reads_the_joint_convention_from_the_env():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts, TeleopBackend
    from rlinf.robotics.teleop import ActionKind

    facts = EnvFacts(
        layout={"left.arm": slice(0, 7), "left.end_effector": slice(7, 8)},
        kinds={
            "left.arm": ActionKind.JOINT_DELTA,
            "left.end_effector": ActionKind.GRIPPER,
        },
        joint_action_scale=0.25,
    )
    entry = TeleopBackend.named("gello_joint").entry(
        {"left_gello_port": "/dev/left"}, {"drives": "left"}, facts
    )

    assert entry.binding.use_delta is True
    assert entry.binding.action_scale == 0.25


def test_an_entry_option_wins_over_the_env_default():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts, TeleopBackend
    from rlinf.robotics.teleop import ActionKind

    facts = EnvFacts(
        layout={"left.arm": slice(0, 7), "left.end_effector": slice(7, 8)},
        kinds={
            "left.arm": ActionKind.JOINT_DELTA,
            "left.end_effector": ActionKind.GRIPPER,
        },
        joint_action_scale=0.25,
    )
    entry = TeleopBackend.named("gello_joint").entry(
        {"left_gello_port": "/dev/left"},
        {"drives": "left", "action_scale": 0.5},
        facts,
    )

    assert entry.binding.action_scale == 0.5
    assert entry.binding.use_delta is True


def test_a_failed_teleop_connect_leaves_no_device_open():
    from rlinf.robotics.parts.teleop.devices import TeleopPart
    from rlinf.robotics.teleop import ActionKind, SpaceMouseBinding, TeleopEntry
    from rlinf.robotics.teleop.group import TeleopGroup

    log: list[str] = []

    class Device(TeleopPart):
        def __init__(self, tag, fail=False):
            self.tag, self.fail = tag, fail

        def _open(self):
            if self.fail:
                raise RuntimeError("cable unplugged")
            log.append(f"open:{self.tag}")
            return self.tag

        def _release(self, device):
            log.append(f"close:{self.tag}")

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    kinds = {
        f"{side}.{part}": kind
        for side in ("left", "right")
        for part, kind in (
            ("arm", ActionKind.CARTESIAN_DELTA),
            ("end_effector", ActionKind.GRIPPER),
        )
    }
    group = TeleopGroup(
        [
            TeleopEntry(Device("first"), SpaceMouseBinding(), drives="left"),
            TeleopEntry(
                Device("second", fail=True), SpaceMouseBinding(), drives="right"
            ),
        ],
        available=kinds,
    )

    with pytest.raises(RuntimeError, match="cable unplugged"):
        group.connect()

    assert log == ["open:first", "close:first"], (
        f"the device opened before the failure was left open: {log}"
    )


def test_the_glove_reads_the_key_the_shipped_configs_set():
    from rlinf.envs.real.wrappers.teleop.backends import TeleopBackend

    glove = TeleopBackend.named("glove")
    cfg = {
        "glove_config": {
            "left_port": "/dev/ttyACM7",
            "right_port": "/dev/ttyACM8",
            "frequency": 90,
            "config_file": "/etc/glove.json",
        }
    }

    device = glove.entry(cfg, {}, None).device
    assert device._left_port == "/dev/ttyACM7"
    assert device._right_port == "/dev/ttyACM8"
    assert device._frequency == 90
    assert device._config_file == "/etc/glove.json"

    # Per-entry options override shared device configuration.
    overridden = glove.entry(cfg, {"left_port": "/dev/override"}, None).device
    assert overridden._left_port == "/dev/override"

    # The documented default applies when the option is omitted.
    assert glove.entry({}, {}, None).device._left_port == "/dev/ttyACM0"


def test_every_shipped_teleop_config_key_is_one_a_backend_reads():
    import pathlib
    import re

    from rlinf.envs.real.wrappers.teleop.backends import TeleopBackend

    source = pathlib.Path("rlinf/envs/real/wrappers/teleop/backends.py").read_text()
    read = set(re.findall(r'cfg\.get\(\s*"([a-z_]+)"', source))

    configured: set[str] = set()
    for path in pathlib.Path("examples/embodiment/config").glob("*.yaml"):
        text = path.read_text()
        if "teleop:" not in text:
            continue
        configured |= {
            key
            for key in re.findall(r"^\s{4}([a-z_]+):", text, re.MULTILINE)
            if key.split("_")[0] in TeleopBackend.names()
        }

    unread = sorted(configured - read)
    assert not unread, f"configs set {unread}, which no teleop backend reads"


def test_a_teleop_device_is_one_class_that_registers_itself():
    from rlinf.envs.real.wrappers.teleop.backends import TeleopBackend

    assert TeleopBackend.names() == [
        "gello",
        "gello_joint",
        "glove",
        "pico",
        "spacemouse",
    ]

    # Each registry entry contains both device and builder behavior.
    for name in TeleopBackend.names():
        backend = TeleopBackend.named(name)
        assert issubclass(backend, TeleopBackend)
        assert callable(backend.entry)
        assert callable(backend.streamer)

    # Only streaming backends override the default capability flag.
    quiet = [
        name
        for name in TeleopBackend.names()
        if "streamer" not in vars(TeleopBackend.named(name))
    ]
    assert "gello_joint" not in quiet, "the one device that streams must say so"
    assert set(quiet) == {"gello", "glove", "pico", "spacemouse"}

    with pytest.raises(ValueError, match="Unknown teleop device"):
        TeleopBackend.named("no_such_device")

    # Duplicate names fail deterministically instead of depending on import order.
    with pytest.raises(ValueError, match="already registered"):

        @TeleopBackend.register("pico")
        class Second(TeleopBackend):
            @classmethod
            def entry(cls, cfg, options, facts):
                raise AssertionError("never built")


def test_every_env_only_offers_teleop_devices_that_exist():
    from rlinf.envs.real.dosw1.base import DOSW1Env
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.franka.dual_base import DualFrankaEnv
    from rlinf.envs.real.wrappers.teleop.backends import TeleopBackend
    from rlinf.envs.real.xsquare.base import Turtle2Env

    known = set(TeleopBackend.names())
    for env_cls in (FrankaEnv, DualFrankaEnv, Turtle2Env, DOSW1Env):
        offered = set(getattr(env_cls, "TELEOP", ()))
        unknown = sorted(offered - known)
        assert not unknown, f"{env_cls.__name__} offers {unknown}, which do not exist"


def test_the_streamer_comes_from_the_registry_not_the_stack():
    from rlinf.envs.real.wrappers.teleop.builder import EnvFacts, TeleopBackend

    quiet = EnvFacts(layout={}, kinds={}, direct_stream=False)
    assert TeleopBackend.named("gello_joint").streamer({}, quiet, []) is None


# Environment action declarations


def _declared(cls, **attrs):
    """Return the action parts declared by an environment class."""
    return cls.action_parts(SimpleNamespace(**attrs))


def test_every_env_declares_parts_that_tile_its_action():
    from rlinf.envs.real.dosw1.base import DOSW1Env
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.gim_arm.base import GimArmEnv
    from rlinf.envs.real.xsquare.base import Turtle2Env

    cases = [
        (7, _declared(FrankaEnv, _is_hand=False)),
        (12, _declared(FrankaEnv, _is_hand=True)),
        (7, _declared(GimArmEnv)),
        (7, _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[1]))),
        (14, _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[0, 1]))),
        (14, _declared(DOSW1Env)),
    ]
    for width, parts in cases:
        assert sum(part.width for part in parts) == width


def test_a_two_armed_robot_names_both_arms():
    from rlinf.envs.real.xsquare.base import Turtle2Env

    parts = _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[0, 1]))

    assert [part.name for part in parts] == [
        "left.arm",
        "left.end_effector",
        "right.arm",
        "right.end_effector",
    ]


def test_two_arms_of_the_same_width_can_mean_different_things():
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.gim_arm.base import GimArmEnv
    from rlinf.robotics.teleop import ActionKind

    franka_arm = _declared(FrankaEnv, _is_hand=False)[0]
    gim_arm = _declared(GimArmEnv)[0]

    assert franka_arm.width == gim_arm.width == 6
    assert franka_arm.kind is ActionKind.CARTESIAN_DELTA
    assert gim_arm.kind is ActionKind.JOINT_POSITION


def test_an_env_that_declares_nothing_cannot_be_teleoperated():
    from rlinf.envs.real.wrappers.teleop.layout import action_spec

    class Bare:
        unwrapped = None

        def get_wrapper_attr(self, name):
            raise AttributeError(name)

    Bare.unwrapped = Bare()
    with pytest.raises(AttributeError, match="does not declare action_parts"):
        action_spec(Bare())


def test_a_declaration_that_does_not_tile_the_action_is_refused():
    import gymnasium as gym

    from rlinf.envs.real.wrappers.teleop.layout import action_spec
    from rlinf.robotics.teleop import ActionKind, ActionPart

    class Wrong:
        action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

        def action_parts(self):
            return (ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),)

        def get_wrapper_attr(self, name):
            return getattr(self, name)

    env = Wrong()
    env.unwrapped = env
    with pytest.raises(ValueError, match="declares parts covering 6"):
        action_spec(env)


def test_a_device_that_means_something_else_is_refused():
    from rlinf.robotics.teleop import (
        ActionKind,
        SpaceMouseBinding,
        TeleopEntry,
        TeleopGroup,
    )

    class Device:
        is_connected = True

        def get_observation(self):
            return {}

        def connect(self):
            pass

        def disconnect(self):
            pass

    joint_arm = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }
    with pytest.raises(ValueError, match="mean different things"):
        TeleopGroup([TeleopEntry(Device(), SpaceMouseBinding())], available=joint_arm)


# Teleoperation compatibility of shipped configurations


def _task_classes():
    """Return registered real-world task classes keyed by Gym ID."""
    from rlinf.envs.real import dosw1, franka, gim_arm, xsquare

    classes = {}
    for module in (franka, dosw1, xsquare, gim_arm):
        classes.update(getattr(module, "TASKS", {}))
    return classes


def _env_configs():
    """Return each shipped environment config and its Gym ID.

    A run config layers an ``env/<name>`` file into ``env.train`` or
    ``env.eval`` and may override the device there, so both halves are read.
    """
    import yaml

    roots = [_ROOT / "examples", _ROOT / "evaluations"]
    env_files = {}
    for root in roots:
        for path in root.rglob("env/*.yaml"):
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            env_id = (doc.get("init_params") or {}).get("id")
            # Simulated environments have no real-world teleoperation contract.
            if env_id and str(doc.get("env_type", "")) in ("real", "realworld"):
                env_files[path.stem] = (path, doc, str(env_id))

    for path, doc, env_id in env_files.values():
        yield path, doc, env_id

    pattern = re.compile(r"env/([\w-]+)@env\.(train|eval)")
    for root in roots:
        for path in root.rglob("*.yaml"):
            if path.parent.name == "env":
                continue
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            for entry in doc.get("defaults") or []:
                match = pattern.search(str(entry))
                if not match:
                    continue
                base = env_files.get(match.group(1))
                if base is None:
                    continue
                section = ((doc.get("env") or {}).get(match.group(2))) or {}
                if any(key in section for key in ("teleop", "teleop_device")):
                    yield path, section, base[2]


def test_shipped_configs_name_devices_their_env_can_drive():
    classes = _task_classes()
    offenders = []
    for path, section, env_id in _env_configs():
        env_cls = classes.get(env_id)
        if env_cls is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                resolve_teleop_devices(
                    section,
                    supported=getattr(env_cls, "TELEOP", ()),
                    default=getattr(env_cls, "TELEOP_DEFAULT", NO_DEVICE),
                )
        except ValueError as error:
            offenders.append(f"{path.relative_to(_ROOT)} ({env_id}): {error}")

    assert offenders == []


def test_every_shipped_config_names_a_registered_task():
    classes = _task_classes()
    unknown = sorted(
        {
            env_id
            for _, _, env_id in _env_configs()
            if env_id not in classes and env_id.endswith("-v1")
        }
    )

    assert unknown == []


def test_the_retired_env_type_still_resolves_and_warns():
    from rlinf.envs import SupportedEnvType

    assert SupportedEnvType("real") is SupportedEnvType.REAL
    with pytest.warns(DeprecationWarning, match="'realworld' is retired"):
        assert SupportedEnvType("realworld") is SupportedEnvType.REAL


def test_no_worker_compares_the_env_type_to_a_bare_string():
    offenders = []
    for path in (_ROOT / "rlinf" / "workers").rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'env_type\s*[!=]=\s*["\']', line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


def _resolved(doc, value):
    """Resolve a ``${a.b.c}`` interpolation within one document."""
    if not isinstance(value, str) or not value.startswith("${"):
        return value
    node = doc
    for key in value[2:-1].split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return _resolved(doc, node)


def _merged_sections(path, doc):
    """Yield each environment section after Hydra-style composition.

    A section layers its own keys over the ``env/<name>`` file its ``defaults``
    names, and ``override_cfg`` merges key by key rather than wholesale. What
    the section leaves out it inherits, which is how a task ends up on a
    different robot than the one it trained on.
    """
    import yaml

    pattern = re.compile(r"env/([\w-]+)@env\.(train|eval)")
    for entry in doc.get("defaults") or []:
        match = pattern.search(str(entry))
        if not match:
            continue
        base_path = path.parent / "env" / f"{match.group(1)}.yaml"
        if not base_path.exists():
            continue
        try:
            base = yaml.safe_load(base_path.read_text()) or {}
        except yaml.YAMLError:
            continue
        section = ((doc.get("env") or {}).get(match.group(2))) or {}
        merged = {**base, **section}
        merged["override_cfg"] = {
            **(base.get("override_cfg") or {}),
            **(section.get("override_cfg") or {}),
        }
        yield match.group(2), merged, str((base.get("init_params") or {}).get("id"))


def test_shipped_configs_give_the_policy_the_action_width_it_expects():
    import yaml

    from rlinf.envs.real.franka.base import FrankaEnv

    classes = _task_classes()
    offenders = []
    for path in (_ROOT / "examples").rglob("*.yaml"):
        if path.parent.name == "env":
            continue
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        action_dim = _resolved(
            doc, ((doc.get("rollout") or {}).get("model") or {}).get("action_dim")
        )
        if not isinstance(action_dim, int):
            continue
        for name, section, env_id in _merged_sections(path, doc):
            # This legacy action layout applies only to single-arm Franka.
            env_cls = classes.get(env_id)
            if env_cls is None or not issubclass(env_cls, FrankaEnv):
                continue
            end_effector = str(
                section["override_cfg"].get("end_effector_type", "franka_gripper")
            )
            parts = FrankaEnv.action_parts(
                SimpleNamespace(_is_hand=end_effector.endswith("hand"))
            )
            width = sum(part.width for part in parts)
            if width != action_dim:
                offenders.append(
                    f"{path.relative_to(_ROOT)} env.{name}: {end_effector} gives "
                    f"{width}, model wants {action_dim}"
                )

    assert offenders == []


def test_direct_stream_gello_opens_one_reader_per_port():
    import sys
    import types

    import numpy as np

    opened = []

    fake = types.ModuleType("rlinf.robotics.parts.teleop.readers.gello_joint")

    class GelloJointExpert:
        def __init__(self, port):
            opened.append(port)
            self.ready = True

        def get_action(self):
            return np.zeros(7), np.zeros(1)

        def close(self):
            pass

    fake.GelloJointExpert = GelloJointExpert
    saved = sys.modules.get("rlinf.robotics.parts.teleop.readers.gello_joint")
    sys.modules["rlinf.robotics.parts.teleop.readers.gello_joint"] = fake
    try:
        from rlinf.envs.real.wrappers.teleop.builder import EnvFacts, TeleopBackend
        from rlinf.robotics.parts.teleop.devices import TeleopLeaderArm
        from rlinf.robotics.teleop import TeleopEntry

        arms = {
            side: TeleopLeaderArm(port=f"/dev/{side}", joint_space=True)
            for side in ("left", "right")
        }
        entries = [TeleopEntry(arm, None, drives=side) for side, arm in arms.items()]
        for arm in arms.values():
            arm.connect()
        assert opened == ["/dev/left", "/dev/right"]

        facts = EnvFacts(layout={}, kinds={}, direct_stream=True)
        streamer = TeleopBackend.named("gello_joint").streamer({}, facts, entries)

        assert opened == ["/dev/left", "/dev/right"], (
            f"the streamer opened more readers: {opened}"
        )
        assert streamer.left_arm is arms["left"]
        assert streamer.right_arm is arms["right"]
    finally:
        if saved is None:
            sys.modules.pop("rlinf.robotics.parts.teleop.readers.gello_joint", None)
        else:
            sys.modules["rlinf.robotics.parts.teleop.readers.gello_joint"] = saved
