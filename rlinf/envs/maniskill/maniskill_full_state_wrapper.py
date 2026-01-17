# Copyright 2025 The RLinf Authors.
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

import gymnasium as gym


class ManiskillFullStateWrapper(gym.Wrapper):
    """
    Wrapper that replaces partial states with full 42-dim states in rgb mode.

    In rgb mode, ManiSkill returns partial state (29 dim) in obs["states"].
    This wrapper replaces it with full state (42 dim) by directly querying
    robot qpos, qvel, and object poses from the env.

    Usage:
        In env_worker.py or config, enable this wrapper for rgb mode.
        Then use obs_dim: 42 in mlp_policy config.
    """

    def __init__(self, env: gym.Env, num_envs: int = 1):
        super().__init__(env)
        self.num_envs = num_envs
        self._unwrapped = self._get_unwrapped_env()

    def _get_unwrapped_env(self):
        """Get the innermost ManiSkill env."""
        unwrapped = self.env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "unwrapped"):
            unwrapped = unwrapped.unwrapped
        return unwrapped

    def _get_full_state(self):
        """
        Get full 42-dim state directly from ManiSkill env.

        State composition (PickCube-v1):
        - robot qpos: 9 dim (7 arm joints + 2 gripper)
        - robot qvel: 9 dim
        - tcp_pose: 7 dim (3 pos + 4 quat)
        - goal_pos: 3 dim
        - obj_pose: 7 dim (3 pos + 4 quat)
        - tcp_to_obj_pos: 3 dim
        - obj_to_goal_pos: 3 dim
        Total: ~42 dim (may vary slightly by task)
        """
        import torch
        from mani_skill.utils import common

        env = self._unwrapped

        # Temporarily get state observation
        original_mode = env._obs_mode
        env._obs_mode = "state"
        try:
            state_obs = env.get_obs()
        finally:
            env._obs_mode = original_mode

        # Flatten if dict
        if isinstance(state_obs, dict):
            state = common.flatten_state_dict(state_obs, use_torch=True, device=env.device)
        else:
            state = state_obs

        return state

    def _replace_states(self, obs):
        """Replace partial states with full states."""
        import torch

        if not isinstance(obs, dict):
            return obs

        # Only process if this is rgb mode (has main_images)
        if "main_images" not in obs:
            return obs

        # Get full state and replace
        full_state = self._get_full_state()
        obs["states"] = full_state

        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._replace_states(obs)
        return obs, info

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        obs = self._replace_states(obs)
        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        obs, rewards, terminations, truncations, infos = self.env.chunk_step(chunk_actions)
        obs = self._replace_states(obs)
        # Handle final_observation in infos (from auto_reset)
        if isinstance(infos, dict) and "final_observation" in infos:
            infos["final_observation"] = self._replace_states(infos["final_observation"])
        return obs, rewards, terminations, truncations, infos
