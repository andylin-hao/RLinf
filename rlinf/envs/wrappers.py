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

import os
import pickle
from typing import Any, Optional

import gymnasium as gym


class DataCollectorWrapper(gym.Wrapper):
    """
    A transparent wrapper that collects environment interaction data.

    This wrapper records observations, actions, rewards, termination flags,
    and info dicts for each step. Each env's episode is saved to a separate
    pkl file when that env's episode ends.

    Args:
        env: The environment to wrap.
        save_dir: Directory path where episode data will be saved.
        enabled: Whether data collection is enabled. If False, the wrapper
            acts as a pass-through.
        rank: The worker rank for file naming (default: 0).
        mode: Data collection mode - "train", "eval", or "all".
        num_envs: Number of parallel environments (default: 1).
        show_goal_site: Whether to show goal_site (green target marker) in renders.
        sample_rate_success: Sampling rate for successful episodes (default: 1.0).
        sample_rate_fail: Sampling rate for failed episodes (default: 0.1).
    """

    def __init__(
        self,
        env: gym.Env,
        save_dir: str,
        enabled: bool = True,
        rank: int = 0,
        mode: str = "all",
        num_envs: int = 1,
        show_goal_site: bool = True,
        sample_rate_success: float = 1.0,
        sample_rate_fail: float = 0.1,
    ):
        super().__init__(env)
        self.save_dir = save_dir
        self.enabled = enabled
        self.rank = rank
        self.mode = mode
        self.num_envs = num_envs
        self.show_goal_site = show_goal_site
        self.sample_rate_success = sample_rate_success
        self.sample_rate_fail = sample_rate_fail

        # Per-env episode counters
        self._episode_ids = [0] * num_envs
        
        # Per-env success status for current episode
        self._episode_success = [False] * num_envs
        
        # Per-env next episode initial obs (set when auto_reset occurs)
        self._next_episode_obs = [None] * num_envs

        # Per-env episode buffers
        self._buffers = [self._create_empty_buffer() for _ in range(num_envs)]

        # Create save directory if enabled
        if self.enabled:
            self._setup_save_dir()

    def _create_empty_buffer(self) -> dict[str, list]:
        """Create an empty episode buffer."""
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
        }

    def _setup_save_dir(self):
        """Create the save directory."""
        os.makedirs(self.save_dir, exist_ok=True)

    def _show_goal_site_visual(self):
        """
        Make goal_site visible in sensor renders.
        
        ManiSkill's PickCube task creates a green sphere (goal_site) but hides it
        by default. This method uses show_visual() which is the correct way to
        unhide objects in ManiSkill's batched rendering.
        """
        if not self.show_goal_site:
            return
            
        # Get the unwrapped ManiSkill env
        unwrapped = self.env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "unwrapped"):
            unwrapped = unwrapped.unwrapped
            
        if not hasattr(unwrapped, "goal_site"):
            return
            
        goal_site = unwrapped.goal_site
        
        # Remove from _hidden_objects list so ManiSkill doesn't re-hide it
        if hasattr(unwrapped, "_hidden_objects"):
            while goal_site in unwrapped._hidden_objects:
                unwrapped._hidden_objects.remove(goal_site)
        
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """
        Reset the environment and clear the episode buffers.
        """
        # Clear all buffers and success status for new episodes
        self._buffers = [self._create_empty_buffer() for _ in range(self.num_envs)]
        self._episode_success = [False] * self.num_envs

        # Call the underlying environment's reset
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Show goal_site after reset (ManiSkill hides it by default)
        self._show_goal_site_visual()

        if self.enabled:
            # Record the initial observation for each env
            self._record_initial_obs(obs)

        return obs, info

    def _record_initial_obs(self, obs):
        """Record initial observation for each env."""
        import torch

        if isinstance(obs, dict):
            # Extract per-env observations
            for env_idx in range(self.num_envs):
                env_obs = {}
                for key, value in obs.items():
                    if isinstance(value, (torch.Tensor,)) and value.shape[0] == self.num_envs:
                        env_obs[key] = self._copy_data(value[env_idx])
                    elif isinstance(value, list) and len(value) == self.num_envs:
                        env_obs[key] = self._copy_data(value[env_idx])
                    else:
                        env_obs[key] = self._copy_data(value)
                self._buffers[env_idx]["observations"].append(env_obs)
        else:
            # Tensor observation
            for env_idx in range(self.num_envs):
                self._buffers[env_idx]["observations"].append(
                    self._copy_data(obs[env_idx] if obs.shape[0] == self.num_envs else obs)
                )

    def step(self, action, **kwargs):
        """Execute a step in the environment and record the transition."""
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)

        if self.enabled:
            self._record_step_data(action, obs, reward, terminated, truncated, info)
            self._check_and_flush_episodes(terminated, truncated)

        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        """
        Execute a chunk of actions (for embodied environments like ManiSkill).

        Each env's episode is saved separately when that env's episode ends.
        """
        obs, rewards, terminations, truncations, infos = self.env.chunk_step(
            chunk_actions
        )

        if self.enabled:
            self._record_chunk_data(chunk_actions, obs, rewards, terminations, truncations, infos)
            self._check_and_flush_episodes(terminations, truncations)

        return obs, rewards, terminations, truncations, infos

    def _record_step_data(self, action, obs, reward, terminated, truncated, info):
        """Record step data for each env."""
        import torch

        for env_idx in range(self.num_envs):
            # Extract per-env data
            env_action = self._extract_env_data(action, env_idx)
            env_reward = self._extract_env_data(reward, env_idx)
            env_terminated = self._extract_env_data(terminated, env_idx)
            env_truncated = self._extract_env_data(truncated, env_idx)
            env_info = self._extract_env_data(info, env_idx)
            
            # Handle auto_reset: use final_observation for terminal obs
            # When auto_reset occurs, obs is the reset obs, not terminal obs
            # final_observation contains the true terminal observation
            has_final_obs = isinstance(info, dict) and "final_observation" in info
            if has_final_obs:
                final_obs = info["final_observation"]
                env_obs = self._extract_env_data(final_obs, env_idx)
                # Store reset obs for next episode
                self._next_episode_obs[env_idx] = self._extract_env_data(obs, env_idx)
            else:
                env_obs = self._extract_env_data(obs, env_idx)

            self._buffers[env_idx]["actions"].append(env_action)
            self._buffers[env_idx]["observations"].append(env_obs)
            self._buffers[env_idx]["rewards"].append(env_reward)
            self._buffers[env_idx]["terminated"].append(env_terminated)
            self._buffers[env_idx]["truncated"].append(env_truncated)
            self._buffers[env_idx]["infos"].append(env_info)
            
            # Track success status
            self._update_success_status(env_idx, info)

    def _record_chunk_data(self, chunk_actions, obs, rewards, terminations, truncations, infos):
        """Record chunk data for each env."""
        for env_idx in range(self.num_envs):
            # Extract per-env data
            env_actions = self._extract_env_data(chunk_actions, env_idx)
            env_rewards = self._extract_env_data(rewards, env_idx)
            env_terminated = self._extract_env_data(terminations, env_idx)
            env_truncated = self._extract_env_data(truncations, env_idx)
            env_info = self._extract_env_data(infos, env_idx)
            
            # Handle auto_reset: use final_observation for terminal obs
            has_final_obs = isinstance(infos, dict) and "final_observation" in infos
            if has_final_obs:
                final_obs = infos["final_observation"]
                env_obs = self._extract_env_data(final_obs, env_idx)
                # Store reset obs for next episode
                self._next_episode_obs[env_idx] = self._extract_env_data(obs, env_idx)
            else:
                env_obs = self._extract_env_data(obs, env_idx)

            self._buffers[env_idx]["actions"].append(env_actions)
            self._buffers[env_idx]["observations"].append(env_obs)
            self._buffers[env_idx]["rewards"].append(env_rewards)
            self._buffers[env_idx]["terminated"].append(env_terminated)
            self._buffers[env_idx]["truncated"].append(env_truncated)
            self._buffers[env_idx]["infos"].append(env_info)
            
            # Track success status
            self._update_success_status(env_idx, infos)

    def _update_success_status(self, env_idx, info):
        """Update success status for an env based on info dict."""
        import torch
        
        if isinstance(info, dict) and "success" in info:
            success = info["success"]
            if isinstance(success, torch.Tensor):
                if success.dim() == 0:
                    self._episode_success[env_idx] = success.item()
                elif success.shape[0] == self.num_envs:
                    self._episode_success[env_idx] = success[env_idx].item()
            else:
                self._episode_success[env_idx] = bool(success)

    def _extract_env_data(self, data, env_idx):
        """Extract data for a specific env from batched data."""
        import numpy as np
        import torch

        if isinstance(data, torch.Tensor):
            if data.shape[0] == self.num_envs:
                return self._copy_data(data[env_idx])
            return self._copy_data(data)
        elif isinstance(data, np.ndarray):
            if data.shape[0] == self.num_envs:
                return self._copy_data(data[env_idx])
            return self._copy_data(data)
        elif isinstance(data, dict):
            return {k: self._extract_env_data(v, env_idx) for k, v in data.items()}
        elif isinstance(data, list):
            if len(data) == self.num_envs:
                return self._copy_data(data[env_idx])
            return self._copy_data(data)
        else:
            return self._copy_data(data)

    def _check_and_flush_episodes(self, terminated, truncated):
        """Check each env's done status and flush finished episodes."""
        import torch

        for env_idx in range(self.num_envs):
            # Get done status for this env
            if isinstance(terminated, torch.Tensor):
                # For chunk_step, check the last step's done status
                if terminated.dim() > 1:
                    # Shape: [num_envs, chunk_size]
                    env_term = terminated[env_idx, -1].item()
                    env_trunc = truncated[env_idx, -1].item()
                else:
                    env_term = terminated[env_idx].item()
                    env_trunc = truncated[env_idx].item()
            else:
                env_term = bool(terminated)
                env_trunc = bool(truncated)

            if env_term or env_trunc:
                self._flush_env_episode(env_idx)

    def _flush_env_episode(self, env_idx: int):
        """Save one env's episode buffer to disk and reset its buffer."""
        import random
        
        buffer = self._buffers[env_idx]
        if not buffer["actions"]:
            return

        # Determine if we should save based on success status and sampling rate
        is_success = self._episode_success[env_idx]
        sample_rate = self.sample_rate_success if is_success else self.sample_rate_fail
        
        should_save = random.random() < sample_rate
        
        if should_save:
            # Construct the file path: rank_X_env_Y_episode_Z.pkl
            label = "success" if is_success else "fail"
            filename = f"rank_{self.rank}_env_{env_idx}_episode_{self._episode_ids[env_idx]}_{label}.pkl"
            save_path = os.path.join(self.save_dir, filename)

            # Add metadata to the episode data
            episode_data = {
                "mode": self.mode,
                "rank": self.rank,
                "env_idx": env_idx,
                "episode_id": self._episode_ids[env_idx],
                "success": is_success,
                "observations": buffer["observations"],
                "actions": buffer["actions"],
                "rewards": buffer["rewards"],
                "terminated": buffer["terminated"],
                "truncated": buffer["truncated"],
                "infos": buffer["infos"],
            }

            # Save the episode data
            with open(save_path, "wb") as f:
                pickle.dump(episode_data, f)

        # Increment episode counter for this env
        self._episode_ids[env_idx] += 1

        # Clear the buffer and reset success status
        self._buffers[env_idx] = self._create_empty_buffer()
        self._episode_success[env_idx] = False
        
        # Use next episode obs (from auto_reset) as initial obs for new episode
        if self._next_episode_obs[env_idx] is not None:
            self._buffers[env_idx]["observations"].append(self._next_episode_obs[env_idx])
            self._next_episode_obs[env_idx] = None

    def _copy_data(self, data):
        """Create a copy of the data to avoid reference issues."""
        import numpy as np
        import torch

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().clone()
        elif isinstance(data, np.ndarray):
            return data.copy()
        elif isinstance(data, dict):
            return {k: self._copy_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._copy_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._copy_data(item) for item in data)
        else:
            return data

    def set_mode(self, mode: str):
        """Change the collection mode (train/eval)."""
        self.mode = mode
        if self.enabled:
            self._setup_save_dir()

    def set_enabled(self, enabled: bool):
        """Enable or disable data collection."""
        self.enabled = enabled
        if enabled:
            self._setup_save_dir()


class FullStateWrapper(gym.Wrapper):
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
