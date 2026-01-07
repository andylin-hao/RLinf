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

import asyncio
from collections import defaultdict

import numpy as np
import torch

from rlinf.data.io_struct import EnvOutput
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class AsyncEnvWorker(EnvWorker):
    async def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        chunk_action = []
        for gather_id in range(self.gather_num):
            src_rank_in_rollout = gather_id + self._rank * self.gather_num
            work = self.recv(
                self.cfg.rollout.group_name, src_rank=src_rank_in_rollout, async_op=True
            )

            def _callback():
                rollout_mode, recv_action = work.wait()
                if rollout_mode == "train":
                    self.train_queue.put(recv_action)
                elif rollout_mode == "eval":
                    self.eval_queue.put(recv_action)

            work.then(_callback)

            if mode == "train":
                queue = self.train_queue
            elif mode == "eval":
                queue = self.eval_queue

            while queue.empty():
                await asyncio.sleep(0.001)
            action = queue.get_nowait()
            chunk_action.append(action)
        chunk_action = np.concatenate(chunk_action, axis=0)
        return chunk_action

    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        env_metric_channel: Channel,
    ):
        for env in self.env_list:
            env.start_env()

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        epoch = 0
        while not self.should_stop:
            env_metrics = defaultdict(list)
            env_output_list = []
            if not self.cfg.env.train.auto_reset:
                for i in range(self.stage_num):
                    extracted_obs, infos = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    dones = (
                        torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    terminations = dones.clone()
                    truncations = dones.clone()

                    self.last_dones_list.append(dones)
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        dones=dones,
                        terminations=terminations,
                        truncations=truncations,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                        intervene_actions=None,
                        intervene_flags=None,
                    )
                    env_output_list.append(env_output)
            else:
                self.num_done_envs = 0
                self.num_succ_envs = 0
                for i in range(self.stage_num):
                    env_output = EnvOutput(
                        obs=self.last_obs_list[i],
                        rewards=None,
                        dones=self.last_dones_list[i],
                        terminations=self.last_terminations_list[i],
                        truncations=self.last_truncations_list[i],
                        intervene_actions=self.last_intervened_info_list[i][0],
                        intervene_flags=self.last_intervened_info_list[i][1],
                    )
                    env_output_list.append(env_output)

            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_output_list[stage_id]
                self.send_env_batch(output_channel, env_output.to_dict())

            for _ in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = await self.recv_chunk_actions(input_channel)
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

            for key, value in env_metrics.items():
                env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()
            env_metric_channel.put(env_metrics)

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_dones_list = [env_output.dones for env_output in env_output_list]
            self.last_truncations_list = [
                env_output.truncations for env_output in env_output_list
            ]
            self.last_terminations_list = [
                env_output.terminations for env_output in env_output_list
            ]
            self.last_intervened_info_list = [
                (env_output.intervene_actions, env_output.intervene_flags)
                for env_output in env_output_list
            ]
            self.finish_rollout()
            epoch += 1

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        eval_metrics = defaultdict(list)

        for stage_id in range(self.stage_num):
            self.eval_env_list[stage_id].start_env()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            for stage_id in range(self.stage_num):
                self.eval_env_list[stage_id].is_start = True
                extracted_obs, infos = self.eval_env_list[stage_id].reset()
                env_output = EnvOutput(
                    obs=extracted_obs,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                )
                self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

            for eval_step in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = await self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)
                    if eval_step == n_chunk_steps - 1:
                        continue
                    self.send_env_batch(
                        output_channel, env_output.to_dict(), mode="eval"
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            self.eval_env_list[stage_id].stop_env()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    async def stop(self):
        self.should_stop = True
        for env in self.env_list:
            env.stop_env()
