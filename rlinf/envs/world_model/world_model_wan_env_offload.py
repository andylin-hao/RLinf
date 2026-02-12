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


"""Offload-enabled Wan world model environment."""

import io

import torch

from rlinf.envs.env_manager import EnvOffloadMixin
from rlinf.envs.maniskill.utils import recursive_to_device
from rlinf.envs.world_model.world_model_wan_env import WanEnv

__all__ = ["WanOffloadEnv"]


class WanOffloadEnv(WanEnv, EnvOffloadMixin):
    def get_state(self) -> bytes:
        """Serialize environment state to bytes buffer."""
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "elapsed_steps": self.elapsed_steps,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "_is_start": self._is_start,
            "video_cnt": self.video_cnt,
            "render_images": self.render_images,
            "condition_action": recursive_to_device(self.condition_action, "cpu"),
            "image_queue": recursive_to_device(self.image_queue, "cpu"),
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
        }

        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )

        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        """Load environment state from bytes buffer."""
        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        self.current_obs = (
            recursive_to_device(state["current_obs"], self.device)
            if state["current_obs"] is not None
            else None
        )
        self.task_descriptions = state["task_descriptions"]
        self.init_ee_poses = state["init_ee_poses"]
        self.elapsed_steps = state["elapsed_steps"]
        self.prev_step_reward = state["prev_step_reward"].to(self.device)
        self._is_start = state["_is_start"]
        self.video_cnt = state["video_cnt"]
        self.render_images = state["render_images"]

        if "condition_action" in state:
            self.condition_action = state["condition_action"].to(self.device)

        self.image_queue = recursive_to_device(state["image_queue"], self.device)
        self.reset_state_ids = state["reset_state_ids"].to(self.device)
        self._generator.set_state(state["generator_state"])

        if self.record_metrics and "success_once" in state:
            self.success_once = state["success_once"].to(self.device)
            self.returns = state["returns"].to(self.device)
