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


import hydra
import numpy as np

from rlinf.config import validate_cfg
from rlinf.envs.habitat.habitat_env import HabitatEnv


@hydra.main(
    version_base="1.1", config_path="config", config_name="habitat_r2r_grpo_cma"
)
def main(cfg):
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)

    num_envs = 3
    seed_offset = 0
    total_num_processes = 1

    env = HabitatEnv(
        cfg=cfg.env.eval,
        num_envs=num_envs,
        seed_offset=seed_offset,
        total_num_processes=total_num_processes,
    )
    env.reset()

    n_chunk_steps = cfg.env.eval.max_episode_steps // cfg.actor.model.num_action_chunks
    action_space = ["turn_left", "turn_right", "move_forward"]

    for i in range(n_chunk_steps):
        dummy_actions = np.random.choice(
            action_space, size=(num_envs, cfg.actor.model.num_action_chunks)
        )
        env.chunk_step(dummy_actions)
        print(f"step {i} done")

    for video_name, video_frames in env.render_images.items():
        env.flush_video(video_name, video_frames)


if __name__ == "__main__":
    main()
