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

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.robot_task_env import RobotTask, RobotTaskEnv
from rlinf.robotics import ControllablePart, Group, Robot


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
