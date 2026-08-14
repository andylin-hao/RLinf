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

from rlinf.envs.real.dosw1.dosw1_env import DOSW1Config, DOSW1Env
from rlinf.envs.real.franka.franka_env import FrankaEnv
from rlinf.envs.real.franka.tasks.dual_franka_joint_env import (
    DualFrankaJointEnv,
)
from rlinf.envs.real.gim_arm.gim_arm_env import GimArmEnv, GimArmRobotConfig
from rlinf.envs.real.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig


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
