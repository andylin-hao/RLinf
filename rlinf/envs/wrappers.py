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
        self.rank = rank
        self.collect_mode = mode  # "all", "train", or "eval"
        self.num_envs = num_envs
        self.show_goal_site = show_goal_site
        self.sample_rate_success = sample_rate_success
        self.sample_rate_fail = sample_rate_fail
        
        # Detect actual env mode by checking env's is_eval attribute
        self.env_mode = self._detect_env_mode()
        
        # Enable collection only if collect_mode matches
        should_collect = (self.collect_mode == "all") or (self.collect_mode == self.env_mode)
        self.enabled = enabled and should_collect
        
        # Debug info
        print(f"[DataCollector] env_mode={self.env_mode}, collect_mode={self.collect_mode}, enabled={self.enabled}")

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
    
    def _detect_env_mode(self) -> str:
        """Detect if this is a train or eval environment."""
        # Walk through wrapper chain, checking each level for cfg.is_eval
        current = self.env
        visited = set()
        
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            
            # Check cfg.is_eval at this level
            if hasattr(current, "cfg"):
                cfg = current.cfg
                is_eval = getattr(cfg, "is_eval", None)
                if is_eval is not None:
                    return "eval" if is_eval else "train"
            
            # Check direct is_eval attribute
            if hasattr(current, "is_eval") and not callable(getattr(current, "is_eval")):
                return "eval" if current.is_eval else "train"
            
            # Move to inner env
            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "unwrapped") and current.unwrapped is not current:
                current = current.unwrapped
            else:
                break
        
        # Fallback: assume train
        return "train"

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
        # Try with seed and options first, fall back if not supported
        try:
            obs, info = self.env.reset(seed=seed, options=options)
        except TypeError:
            # Fallback for environments that don't support seed/options
            obs, info = self.env.reset()
        
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
            self._update_success_status(env_idx, env_info)

    def _update_success_status(self, env_idx, info):
        """Update success status for an env based on info dict.
        
        Priority order:
        1. info["episode"]["success_at_end"] - ManiSkill's official end-of-episode success
        2. info["success_once"] - whether success was achieved at any point
        3. info["success"] - current step success (fallback)
        """
        import torch
        
        success = None
        
        # Priority 1: episode.success_once (cumulative - any step succeeded)
        # This is best for data collection: trajectories that ever succeeded are valuable
        if isinstance(info, dict) and "episode" in info:
            episode_info = info["episode"]
            if isinstance(episode_info, dict) and "success_once" in episode_info:
                success = episode_info["success_once"]
        
        # Priority 2: episode.success_at_end (final step success)
        if success is None and isinstance(info, dict) and "episode" in info:
            episode_info = info["episode"]
            if isinstance(episode_info, dict) and "success_at_end" in episode_info:
                success = episode_info["success_at_end"]
        
        # Priority 3: current step success (fallback)
        if success is None and isinstance(info, dict) and "success" in info:
            success = info["success"]
        
        if success is not None:
            if isinstance(success, torch.Tensor):
                if success.dim() == 0:
                    self._episode_success[env_idx] = bool(success.item())
                elif len(success.shape) > 0 and success.shape[0] == self.num_envs:
                    self._episode_success[env_idx] = bool(success[env_idx].item())
            else:
                self._episode_success[env_idx] = bool(success)

    def _get_episode_success(self, buffer, env_idx) -> bool:
        """Get episode success from buffer, checking final_info or second-to-last info."""
        import torch
        
        # First try: check last info's final_info field (set during auto_reset)
        if buffer["infos"]:
            last_info = buffer["infos"][-1]
            if isinstance(last_info, dict) and "final_info" in last_info:
                final_info = last_info["final_info"]
                if isinstance(final_info, dict) and "episode" in final_info:
                    episode = final_info["episode"]
                    if "success_once" in episode:
                        val = episode["success_once"]
                        return bool(val.item() if isinstance(val, torch.Tensor) else val)
        
        # Second try: check second-to-last info (before auto_reset overwrote last)
        if len(buffer["infos"]) >= 2:
            prev_info = buffer["infos"][-2]
            if isinstance(prev_info, dict) and "episode" in prev_info:
                episode = prev_info["episode"]
                if "success_once" in episode:
                    val = episode["success_once"]
                    return bool(val.item() if isinstance(val, torch.Tensor) else val)
        
        # Third try: check any info with episode.success_once (scan backwards)
        for info in reversed(buffer["infos"]):
            if isinstance(info, dict) and "episode" in info:
                episode = info["episode"]
                if "success_once" in episode:
                    val = episode["success_once"]
                    return bool(val.item() if isinstance(val, torch.Tensor) else val)
        
        # Fallback to stored success status
        return self._episode_success[env_idx]

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

        # Check final_info or second-to-last info for episode.success_once
        # (last info is from new episode after auto_reset)
        is_success = self._get_episode_success(buffer, env_idx)
        sample_rate = self.sample_rate_success if is_success else self.sample_rate_fail
        
        should_save = random.random() < sample_rate
        
        if should_save:
            # Construct the file path: rank_X_env_Y_episode_Z.pkl
            label = "success" if is_success else "fail"
            filename = f"rank_{self.rank}_env_{env_idx}_episode_{self._episode_ids[env_idx]}_{label}.pkl"
            save_path = os.path.join(self.save_dir, filename)

            # Add metadata to the episode data
            episode_data = {
                "mode": self.env_mode,
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
        self.collect_mode = mode
        # Re-evaluate if collection should be enabled
        should_collect = (self.collect_mode == "all") or (self.collect_mode == self.env_mode)
        self.enabled = should_collect

    def set_enabled(self, enabled: bool):
        """Enable or disable data collection."""
        self.enabled = enabled
        if enabled:
            self._setup_save_dir()


