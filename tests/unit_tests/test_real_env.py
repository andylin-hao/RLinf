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

"""Real-world environments: task envs, teleop devices, config, and layout."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
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
from rlinf.envs.real.robot_task_env import RobotTask, RobotTaskEnv
from rlinf.envs.real.teleop.config import (  # noqa: E402
    NO_DEVICE,
    resolve_teleop_device,
)
from rlinf.envs.real.teleop.intervention import (  # noqa: E402
    TeleopDevice,
    TeleopIntervention,
    TeleopSample,
)
from rlinf.envs.real.xsquare.base import Turtle2Env, Turtle2RobotConfig
from rlinf.robotics import ControllablePart, Group, Robot

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --- from test_robot_task_env.py --------------------------------------


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
    robot = Robot(arm=Group(arm=driver))
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


# --- from test_realworld_robotics_compatibility.py --------------------


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
        hardware_info=None,
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
        hardware_info=None,
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
        hardware_info=None,
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
        hardware_info=None,
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
        hardware_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


# --- from test_teleop_intervention.py ---------------------------------


class FakeEnv:
    """Records the actions it is stepped with."""

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
    """Replays a list of samples, one per read."""

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
    """While the operator drives, their action is the one applied."""
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=EXPERT, active=True)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], EXPERT)
    assert np.array_equal(info["intervene_action"], EXPERT)


def test_inactive_device_leaves_the_policy_action_alone():
    """A device with nothing to say never touches the action or the info."""
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=None, active=False)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], POLICY)
    assert "intervene_action" not in info


def test_control_is_held_between_samples_then_released():
    """The hold window keeps the operator in control across quiet samples.

    A person moves far slower than the device samples, so without the window
    the applied action would flicker between operator and policy mid-motion.
    """
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)  # operator moves
    wrapper.step(POLICY)  # quiet, but still inside the window
    assert np.array_equal(env.stepped[1], EXPERT)

    device.timeout = 0.0  # window elapses
    wrapper.step(POLICY)
    assert np.array_equal(env.stepped[2], POLICY)


def test_release_uses_the_device_fallback():
    """A device holding state the policy does not command keeps it on release."""
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=EXPERT, active=False)])
    device.timeout = 0.0
    device.fallback_action = np.array([9.0, 9.0, 9.0])
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], device.fallback_action)


def test_mark_flag_is_opt_in():
    """Only formats that key on the flag pay for it."""
    sample = TeleopSample(action=EXPERT, active=True)

    plain = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]))
    flagged = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]), mark_flag=True)

    assert "intervene_flag" not in plain.step(POLICY)[4]
    assert flagged.step(POLICY)[4]["intervene_flag"] == np.ones(1)


def test_device_info_reaches_the_step_info():
    """Device state a collector records rides along with the transition."""
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env,
        ScriptedDevice([TeleopSample(action=EXPERT, active=True, info={"left": True})]),
    )

    assert wrapper.step(POLICY)[4]["left"] is True


def test_reset_resyncs_the_device_and_drops_the_hold():
    """A new episode must not inherit the previous one's intervention."""
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
    """Devices hold serial ports; they are released with the wrapper."""
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.close()

    assert device.closed
    assert env.closed


def test_before_step_runs_ahead_of_the_env():
    """Devices that stream need a hook before the env advances."""
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)

    assert device.before_steps == 1


def test_read_is_abstract():
    """A device is only required to say what the operator wants."""
    with pytest.raises(TypeError):
        TeleopDevice()


# --- from test_teleop_config.py ---------------------------------------

SINGLE_ARM = ("spacemouse", "gello", "pico")


def test_named_device_is_used():
    assert (
        resolve_teleop_device({"teleop_device": "gello"}, supported=SINGLE_ARM)
        == "gello"
    )


def test_missing_config_falls_back_to_the_default():
    assert resolve_teleop_device({}, supported=SINGLE_ARM) == NO_DEVICE
    assert (
        resolve_teleop_device({}, supported=SINGLE_ARM, default="spacemouse")
        == "spacemouse"
    )


def test_retired_boolean_still_selects_its_device_and_warns():
    """Configs written before the rename keep working for a release."""
    with pytest.warns(DeprecationWarning, match="use_pico"):
        device = resolve_teleop_device({"use_pico": True}, supported=SINGLE_ARM)

    assert device == "pico"


def test_all_retired_booleans_off_means_no_device():
    """Explicitly disabling every device is not the same as saying nothing."""
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
    """Env configs layer, so a half-migrated stack can hold both keys.

    Silently preferring either one hands somebody the wrong device with a robot
    already moving, so this is an error rather than a precedence rule.
    """
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
    """A dual-arm Franka has no single-arm Cartesian teleop path."""
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_device(
            {"teleop_device": "spacemouse"}, supported=("gello_joint", "pico")
        )


def test_none_is_always_allowed():
    assert (
        resolve_teleop_device({"teleop_device": "none"}, supported=("pico",))
        == NO_DEVICE
    )


def test_shipped_configs_use_the_new_key():
    """The retired booleans are gone from every config in the repo."""
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
                        "use_spacemouse",
                        "use_gello",
                        "use_gello_joint",
                        "use_pico",
                    )
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


# --- from test_real_env_layout.py -------------------------------------

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
    """Tasks sit at the top of their robot folder, not one level down."""
    leftovers = [name for name in _ROBOTS if (_REAL / name / "tasks").exists()]

    assert leftovers == []


def test_every_robot_folder_has_a_base():
    """Shared machinery is named so nobody mistakes it for a task."""
    missing = [name for name in _ROBOTS if not (_REAL / name / "base.py").exists()]

    assert missing == []


def test_all_task_ids_are_registered():
    """Reaching for any name registers every task.

    ``rlinf.envs.real`` loads lazily so that importing one teleop device does
    not drag in the env stack, which means registration happens on first use
    rather than on import. ``get_env_cls`` reaches it exactly this way.
    """
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    assert EXPECTED_IDS <= set(registry)


def test_every_entry_point_resolves():
    """A registered id whose entry point cannot be imported fails at rollout."""
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
    """The TASKS table is the one place a robot's tasks are declared."""
    from rlinf.envs.real import dosw1, franka, xsquare

    declared = set(franka.TASKS) | set(dosw1.TASKS) | set(xsquare.TASKS)

    # GimArm registers its env class directly, with no wrapper stack.
    assert declared == EXPECTED_IDS - {"GimArmPegInsertionEnv-v1"}


def test_pose_math_is_not_filed_under_a_robot():
    """construct_adjoint_matrix is SE(3) math the wrappers share.

    Leaving it in the Franka package made every wrapper importing it pull that
    package in, which is what turned the task registry into an import cycle.
    """
    from rlinf.envs.real import pose_utils

    assert hasattr(pose_utils, "construct_adjoint_matrix")
    assert not (_REAL / "franka" / "utils.py").exists()


def test_task_configs_state_only_their_compliance_deltas():
    """A task's config should show its tuning, not eighteen repeated gains.

    Every Franka task used to carry a full copy of the impedance gains, so the
    handful of numbers that actually differed were invisible.
    """
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

    # Every task keeps the full set of gains; only some are task-specific.
    for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig):
        assert set(cls().compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert {name: len(keys) for name, keys in deltas.items()} == {
        "PegInsertionConfig": 1,
        "BottleConfig": 8,
        "BinEnvConfig": 11,
        "DexpnpConfig": 6,
    }


def test_unknown_compliance_gain_is_refused():
    """A misspelled gain reaches the impedance controller and is ignored there."""
    import pytest

    from rlinf.envs.real.franka.base import compliance

    with pytest.raises(KeyError, match="Unknown compliance gains"):
        compliance(translational_stifness=1000)


# --- wrapper families ------------------------------------------------------


def test_wrappers_are_split_by_what_they_change():
    """Three families, each with an obvious home for a new wrapper.

    teleop replaces the action, transforms rewrite how it is expressed, and
    episode decides when a rollout starts, ends, and what it scored. The old
    flat wrappers/ package mixed all three.
    """
    real = _ROOT / "rlinf" / "envs" / "real"

    assert not (real / "wrappers").is_dir(), "wrappers/ became wrappers.py"
    assert (real / "wrappers.py").exists()
    for family in ("teleop", "transforms", "episode"):
        assert (real / family / "__init__.py").exists(), family


def test_no_teleop_wrapper_is_left_outside_teleop():
    """Every intervention lives with the devices it reads."""
    real = _ROOT / "rlinf" / "envs" / "real"
    strays = sorted(
        path.name
        for family in ("transforms", "episode")
        for path in (real / family).glob("*.py")
        if "intervention" in path.name and "leader_follower" not in path.name
    )

    assert strays == []


def test_a_held_button_device_does_not_keep_control_after_release():
    """PICO says exactly when it is driving, so it sets no hold window.

    The window exists for devices that sample faster than a person moves;
    applying it here would keep commanding the robot after the grip is released.
    """
    from rlinf.envs.real.teleop.pico import DualFrankaTcpPicoTeleop, PicoTeleop

    assert PicoTeleop.timeout == 0.0
    assert DualFrankaTcpPicoTeleop.timeout == 0.0


def test_streaming_device_lifecycle_without_hardware():
    """The command thread starts once aligned and is joined on close."""
    from rlinf.envs.real.teleop.intervention import TeleopSample
    from rlinf.envs.real.teleop.streaming import StreamingTeleopDevice

    ticks = []

    class Fake(StreamingTeleopDevice):
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
