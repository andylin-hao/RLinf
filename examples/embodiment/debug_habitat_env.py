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

import json
import os
import re
import signal
from collections import deque

import hydra
import numpy as np

from rlinf.config import validate_cfg
from rlinf.envs.habitat.habitat_env import HabitatEnv

action_map = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}


def get_gt_episode_ids(gt_dir_path: str):
    episode_ids = []
    if not os.path.exists(gt_dir_path):
        print(f"Error: No such file or directory: {gt_dir_path}")
        return []
    for filename in os.listdir(gt_dir_path):
        match = re.search(r"episode_(\d+)_actions\.json", filename)
        if match:
            episode_id = match.group(1)
            episode_ids.append(int(episode_id))
    episode_ids.sort(key=int)

    return episode_ids


def load_actions_from_file(episode_ids: list, gt_dir_path: str, gt_episode_list):
    """
    Load action sequences from a directory and return as deque.
    If an episode is not in the list, return an empty list.
    If all episodes are in the list, return a list of deque for the input episode_ids.
    """
    action_list = []

    for episode_id in episode_ids:
        if episode_id not in gt_episode_list:
            return []
        file_name = f"episode_{episode_id}_actions.json"
        file_path = os.path.join(gt_dir_path, file_name)
        with open(file_path, "r") as f:
            action_data = json.load(f)

        # Extract and map actions
        actions_per_file = [
            action_map[action_info["action"]] for action_info in action_data["actions"]
        ]
        action_list.append(deque(actions_per_file))

    return action_list


def test_habitat_env(cfg):
    action_dir_path = "VLN-CE/actions"
    num_envs = cfg.env.eval.total_num_envs
    chunk_size = cfg.actor.model.num_action_chunks
    max_steps_per_rollout_epoch = cfg.env.eval.max_steps_per_rollout_epoch
    n_chunk_steps = max_steps_per_rollout_epoch // chunk_size

    env = HabitatEnv(
        cfg=cfg.env.eval,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
    )
    env.reset()

    try:
        episode_ids = env.env.get_current_episode_ids()
        gt_episode_ids = get_gt_episode_ids(action_dir_path)
        action_queues = load_actions_from_file(
            episode_ids, action_dir_path, gt_episode_ids
        )
        current_episode_ids = episode_ids.copy()

        episode_num = len(gt_episode_ids)
        eval_rollout_epoch = episode_num // num_envs
        completed_episode_cnt = 0

        for rollout_epoch in range(eval_rollout_epoch):
            for chunk_step in range(n_chunk_steps):
                current_chunk_batch = []

                for env_idx in range(num_envs):
                    chunk = []
                    for step_in_chunk in range(chunk_size):
                        if action_queues[env_idx]:
                            chunk.append(action_queues[env_idx].popleft())
                        else:
                            chunk.append("no_op")
                    current_chunk_batch.append(chunk)

                actions_to_step = np.array(current_chunk_batch)
                env.chunk_step(actions_to_step)
                print(f"Step {chunk_step} of rollout {rollout_epoch} completed")

            # Finish rollout
            env.flush_video()
            env.update_reset_state_ids()

            # Load new episode actions and add to queue
            old_episode_ids = current_episode_ids.copy()
            episode_ids = env.env.get_current_episode_ids()
            for idx in range(num_envs):
                completed_episode_cnt += 1
                completed_episode_id = old_episode_ids[idx]
                new_episode_id = episode_ids[idx]
                print(
                    f"Progress: [{completed_episode_cnt} / {episode_num}] :"
                    f" Env {idx} completed episode {completed_episode_id},"
                    f" now processing episode {new_episode_id}."
                )
                # Load new episode actions and add to queue
                new_actions = load_actions_from_file(
                    [episode_ids[idx]], action_dir_path, gt_episode_ids
                )
                if new_actions:
                    action_queues[idx] = new_actions[0]
                    current_episode_ids[idx] = new_episode_id
    finally:
        # System-level kill: terminate all environment worker processes
        if hasattr(env, "env") and hasattr(env.env, "workers"):
            for worker in env.env.workers:
                if hasattr(worker, "process") and worker.process.is_alive():
                    try:
                        worker.process.terminate()
                        worker.process.join(timeout=2)
                        if worker.process.is_alive():
                            os.kill(worker.process.pid, signal.SIGKILL)
                            worker.process.join(timeout=1)
                    except (OSError, AttributeError):
                        # Process already terminated or doesn't exist
                        pass


@hydra.main(
    version_base="1.1", config_path="config", config_name="habitat_r2r_grpo_cma"
)
def main(cfg):
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)

    test_habitat_env(cfg=cfg)


if __name__ == "__main__":
    main()
