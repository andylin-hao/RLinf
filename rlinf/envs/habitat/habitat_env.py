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

import copy
from typing import Optional, Union

import gym
import habitat
import numpy as np
import torch
from habitat_baselines.config.default import get_config
from hydra.core.global_hydra import GlobalHydra

from rlinf.envs.habitat.extensions.utils import observations_to_image
from rlinf.envs.habitat.venv import HabitatRLEnv, ReconfigureSubprocEnv
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    save_rollout_video,
    to_tensor,
)


class HabitatEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.start_idx = 0
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)

        self._init_env()
        self._init_metrics()

        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = {}
        self.current_raw_obs = None

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        for param in env_fn_params:

            def env_fn(p=param):
                config_path = p["config_path"]
                episode_ids = p["episode_ids"]
                seed = p["seed"]

                config = get_config(config_path)

                dataset = habitat.datasets.make_dataset(
                    config.habitat.dataset.type,
                    config=config.habitat.dataset,
                )

                dataset.episodes = [
                    ep for ep in dataset.episodes if ep.episode_id in episode_ids
                ]

                env = HabitatRLEnv(config=config, dataset=dataset)
                env.seed(seed)
                return env

            env_fns.append(env_fn)

        return env_fns

    def get_env_fn_params(self):
        env_fn_params = []

        # Habitat uses hydra to load the config,
        # but the hydra maybe initialized somewhere else,
        # so we need to clear it to avoid conflicts
        hydra_initialized = GlobalHydra.instance().is_initialized()
        if hydra_initialized:
            GlobalHydra.instance().clear()

        config_path = self.cfg.init_params.config_path
        habitat_config = get_config(config_path)

        habitat_dataset = habitat.datasets.make_dataset(
            habitat_config.habitat.dataset.type,
            config=habitat_config.habitat.dataset,
        )

        episode_ids = self._build_ordered_episodes(habitat_dataset)

        num_episodes = len(episode_ids)
        episodes_per_env = num_episodes // self.num_envs

        episode_ranges = []
        start = 0
        for i in range(self.num_envs - 1):
            episode_ranges.append((start, start + episodes_per_env))
            start += episodes_per_env
        episode_ranges.append((start, num_episodes))

        for env_id in range(self.num_envs):
            start, end = episode_ranges[env_id]
            assigned_ids = episode_ids[start:end]

            env_fn_params.append(
                {
                    "config_path": config_path,
                    "episode_ids": assigned_ids,
                    "seed": self.seed + env_id,
                }
            )

        return env_fn_params

    def _build_ordered_episodes(self, dataset):
        """
        rearrange the episode ids to be consecutive for each scene
        """
        scene_ids = []
        episode_ids = []
        scene_id_to_idx = {}  # scene_id(str) -> scene_idx(int)
        scene_to_episodes = {}  # scene_idx(int) -> episode_ids(list[int])

        for episode in dataset.episodes:
            sid = episode.scene_id
            eid = episode.episode_id
            if sid not in scene_id_to_idx:
                scene_idx = len(scene_ids)
                scene_id_to_idx[sid] = scene_idx
                scene_ids.append(sid)
                scene_to_episodes[scene_idx] = []
            else:
                scene_idx = scene_id_to_idx[sid]
            scene_to_episodes[scene_idx].append(eid)

        for scene_idx in range(len(scene_ids)):
            episode_ids.extend(scene_to_episodes[scene_idx])

        return episode_ids

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _wrap_obs(self, obs_list):
        image_list = []
        for obs in obs_list:
            image_list.append(observations_to_image(obs))

        image_tensor = to_tensor(list_of_dict_to_dict_of_list(image_list))

        obs = {}
        rgb_image_tensor = torch.stack(
            [value.clone().permute(2, 0, 1) for value in image_tensor["rgb"]]
        )
        obs["rgb"] = rgb_image_tensor

        if "depth" in image_tensor:
            depth_image_tensor = torch.stack(
                [value.clone().permute(2, 0, 1) for value in image_tensor["depth"]]
            )
            obs["depth"] = depth_image_tensor

        return obs

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            if self.task_ids[env_id] != task_ids[j]:
                reconfig_env_idx.append(env_id)
                self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        raw_obs = self.env.reset(id=env_idx)
        return raw_obs

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        raw_obs = self.env.reset(env_idx)

        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs

        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        obs = self._wrap_obs(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps

        # TODO: what if termination means failure? (e.g. robot falling down)
        step_reward = self._calc_step_reward(terminations)

        if self.video_cfg.save_video:
            episode_ids = self.env.get_current_episode_ids()
            for i in range(len(raw_obs)):
                frame = observations_to_image(raw_obs[i], info_lists[i])
                frame_concat = np.concatenate(
                    (frame["rgb"], frame["depth"], frame["top_down_map"]), axis=1
                )
                key = f"episode_{episode_ids[i]}"
                if key not in self.render_images:
                    self.render_images[key] = []
                self.render_images[key].append(frame_concat)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones.cpu().numpy(), extracted_obs, infos
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(env_idx=env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def flush_video(self, video_name, video_frames):
        save_rollout_video(
            video_frames,
            output_dir=self.video_cfg.video_base_dir,
            video_name=video_name,
            fps=self.video_cfg.fps,
        )
