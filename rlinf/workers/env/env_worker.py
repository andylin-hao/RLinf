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

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    """The EnvWorker is responsible for controlling the embodied environments like simulators or physical robots.

    It calls the corresponding gym env's step function to generate observations, rewards, and done signals based on the actions received from the RollerWorker, and sends them back to the RollerWorker.

    The EnvWorker supports running multiple environment instances in parallel to improve data collection efficiency.
    The main entry point is the `interact` method, which performs environment interactions for a specified number of steps (called chunk_step) and put the collected environment metrics into an output channel to the RolloutWorker.

    Specially, the EnvWorker supports pipeline rollout process, where the parallel environment instances are further divided into multiple stages. Each stage interacts with the environment sequentially, while different stages can run in parallel. This design helps to further improve the efficiency of data collection.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0

        self.train_env_list: list[EnvManager] = []
        self.eval_env_list: list[EnvManager] = []
        self.last_obs_list = []
        self.last_dones_list = []

        # Env configurations
        self.env_type = cfg.env.train.simulator_type
        self.eval_only = getattr(self.cfg.runner, "only_eval", False)
        self.train_batch_size_per_dp = self.cfg.env.train.num_envs
        self.eval_batch_size_per_dp = self.cfg.env.eval.num_envs
        assert (
            self.train_batch_size_per_dp
            == self.cfg.env.train.num_group * self.cfg.env.train.group_size
        ), (
            f"The train num_envs {self.train_batch_size_per_dp} does not match num_groups ({self.cfg.env.train.num_group}) * group_size ({self.cfg.env.train.group_size})."
        )
        assert (
            self.eval_batch_size_per_dp
            == self.cfg.env.eval.num_group * self.cfg.env.eval.group_size
        ), (
            f"The eval num_envs {self.eval_batch_size_per_dp} does not match num_groups ({self.cfg.env.eval.num_group}) * group_size ({self.cfg.env.eval.group_size})."
        )

        # For action and observation communication
        cluster = Cluster()
        placement = HybridComponentPlacement(cfg, cluster)
        self.gather_num = placement.get_world_size(
            "rollout"
        ) // placement.get_world_size("env")

        # Used for pipelined rollout interactions
        self.num_pipeline_stages = self.cfg.rollout.pipeline_stage_num

    def init_worker(self):
        enable_offload = self.cfg.env.enable_offload
        total_num_processes = self._world_size * self.num_pipeline_stages

        for stage_id in range(self.num_pipeline_stages):
            seed_offset = self._rank * self.num_pipeline_stages + stage_id
            if not self.eval_only:
                self.train_env_list.append(
                    EnvManager(
                        self.cfg,
                        rank=self._rank,
                        seed_offset=seed_offset,
                        total_num_processes=total_num_processes,
                        env_type=self.env_type,
                        is_eval=False,
                        enable_offload=enable_offload,
                    )
                )
            if self.cfg.runner.val_check_interval > 0 or self.eval_only:
                self.eval_env_list.append(
                    EnvManager(
                        self.cfg,
                        rank=self._rank,
                        seed_offset=seed_offset,
                        total_num_processes=total_num_processes,
                        env_type=self.env_type,
                        is_eval=True,
                        enable_offload=enable_offload,
                    )
                )

        if not self.eval_only:
            self._init_envs()

    def _init_envs(self):
        for i in range(self.num_pipeline_stages):
            self.train_env_list[i].start_env()
            extracted_obs, rewards, terminations, truncations, infos = (
                self.train_env_list[i].step()
            )
            self.last_obs_list.append(extracted_obs)
            dones = torch.logical_or(terminations, truncations)
            self.last_dones_list.append(
                dones.unsqueeze(1).repeat(1, self.cfg.actor.model.num_action_chunks)
            )
            self.train_env_list[i].stop_env()

    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.env_type,
            model_name=self.cfg.actor.model.model_name,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.train_env_list[stage_id].chunk_step(chunk_actions)
        )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            env_type=self.env_type,
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
        )
        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.env_type,
            model_name=self.cfg.actor.model.model_name,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            env_type=self.env_type,
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
        )
        return env_output, env_info

    def recv_chunk_actions(self, input_channel: Channel) -> np.ndarray:
        chunk_action = []
        for gather_id in range(self.gather_num):
            chunk_action.append(
                input_channel.get(
                    key=gather_id + self._rank * self.gather_num,
                )
            )
        chunk_action = np.concatenate(chunk_action, axis=0)
        return chunk_action

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            if self.cfg.env.train.video_cfg.save_video:
                for i in range(self.num_pipeline_stages):
                    self.train_env_list[i].flush_video()
            for i in range(self.num_pipeline_stages):
                self.train_env_list[i].update_reset_state_ids()
        elif mode == "eval":
            if self.cfg.env.eval.video_cfg.save_video:
                for i in range(self.num_pipeline_stages):
                    self.eval_env_list[i].flush_video()

    def split_env_batch(self, env_batch, gather_id, mode):
        env_batch_i = {}
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                env_batch_i[key] = value.chunk(self.gather_num, dim=0)[
                    gather_id
                ].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_batch_size_per_dp, (
                        f"key {key}, length: {length}, batch_size: {self.train_batch_size_per_dp}"
                    )
                elif mode == "eval":
                    assert length == self.eval_batch_size_per_dp, (
                        f"key {key}, length: {length}, batch_size: {self.eval_batch_size_per_dp}"
                    )
                env_batch_i[key] = value[
                    gather_id * length // self.gather_num : (gather_id + 1)
                    * length
                    // self.gather_num
                ]
            elif isinstance(value, dict):
                env_batch_i[key] = self.split_env_batch(value, gather_id, mode)
            else:
                env_batch_i[key] = value
        return env_batch_i

    def send_env_batch(self, output_channel: Channel, env_batch, mode="train"):
        # split env_batch into num_processes chunks, each chunk contains gather_num env_batch
        for gather_id in range(self.gather_num):
            env_batch_i = self.split_env_batch(env_batch, gather_id, mode)
            output_channel.put(
                item=env_batch_i,
                key=gather_id + self._rank * self.gather_num,
            )

    def interact(self, input_channel: Channel, output_channel: Channel):
        for env in self.train_env_list:
            env.start_env()

        env_metrics = defaultdict(list)
        for epoch in range(self.cfg.algorithm.rollout_epoch):
            env_output_list = []
            if not self.cfg.env.train.auto_reset:
                for i in range(self.num_pipeline_stages):
                    extracted_obs, infos = self.train_env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    dones = (
                        torch.zeros((self.cfg.env.train.num_envs,), dtype=bool)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    self.last_dones_list.append(dones)
                    env_output = EnvOutput(
                        env_type=self.env_type,
                        obs=extracted_obs,
                        dones=dones,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_output_list.append(env_output)
            else:
                self.num_done_envs = 0
                self.num_succ_envs = 0
                for i in range(self.num_pipeline_stages):
                    env_output = EnvOutput(
                        env_type=self.env_type,
                        obs=self.last_obs_list[i],
                        rewards=None,
                        dones=self.last_dones_list[i],
                    )
                    env_output_list.append(env_output)

            for stage_id in range(self.num_pipeline_stages):
                env_output: EnvOutput = env_output_list[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for _ in range(self.cfg.algorithm.n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    raw_chunk_actions = self.recv_chunk_actions(input_channel)
                    env_output, env_info = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    self.send_env_batch(output_channel, env_output.to_dict())
                    env_output_list[stage_id] = env_output
                    for key, value in env_info.items():
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if key in env_metrics and len(env_metrics[key]) > epoch:
                                env_metrics[key][epoch] = value
                            else:
                                env_metrics[key].append(value)
                        else:
                            env_metrics[key].append(value)

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_dones_list = [env_output.dones for env_output in env_output_list]
            self.finish_rollout()

        for env in self.train_env_list:
            env.stop_env()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        for i in range(self.num_pipeline_stages):
            self.eval_env_list[i].start_env()
            self.eval_env_list[i].is_start = True
            extracted_obs, _, _, _, infos = self.eval_env_list[i].step()
            env_output = EnvOutput(
                env_type=self.cfg.env.eval.simulator_type,
                obs=extracted_obs,
                final_obs=infos["final_observation"]
                if "final_observation" in infos
                else None,
            )
            self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

        eval_metrics = defaultdict(list)

        for eval_step in range(self.cfg.algorithm.n_eval_chunk_steps):
            for i in range(self.num_pipeline_stages):
                raw_chunk_actions = self.recv_chunk_actions(input_channel)
                env_output, env_info = self.env_evaluate_step(raw_chunk_actions, i)

                for key, value in env_info.items():
                    eval_metrics[key].append(value)
                if eval_step == self.cfg.algorithm.n_eval_chunk_steps - 1:
                    continue
                self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

        self.finish_rollout(mode="eval")
        for i in range(self.num_pipeline_stages):
            self.eval_env_list[i].stop_env()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
