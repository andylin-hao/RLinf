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

import collections
import json
import logging
import os

import gymnasium as gym
import imageio
import numpy as np

os.environ["MUJOCO_GL"] = "egl"


def load_prompt_from_json(json_path, env_name):
    with open(json_path, "r") as f:
        prompt_data = json.load(f)
    return prompt_data.get(env_name, "")


PROMPT_JSON_PATH = "metaworld_config.json"
with open(PROMPT_JSON_PATH, "r") as f:
    config_data = json.load(f)
task_description_dict = config_data.get("TASK_DESCRIPTIONS", {})
difficulty_to_tasks = config_data.get("DIFFICULTY_TO_TASKS", {})
env_list = list(task_description_dict.keys())


# setup the policy
def setup_policy(pretrained_path, action_chunk):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    policy = _policy_config.create_trained_policy(
        _config.get_config("pi0_metaworld"),
        pretrained_path,
        sample_kwargs={"num_steps": action_chunk},
    )
    return policy


def main(args):
    # parameters
    pretrained_path = args.pretrained_path
    action_chunk = args.action_chunk
    exp_name = args.exp_name
    num_trials_per_task = args.num_trials_per_task
    max_steps = args.max_steps
    # Setup logging
    os.makedirs("log", exist_ok=True)
    log_file = os.path.join("log", f"{exp_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # policy setup
    logger.info("policy setup start")
    policy = setup_policy(pretrained_path, action_chunk)
    logger.info("policy setup done")

    total_episodes = 0
    total_successes = 0
    results_per_task = {}

    for env_name in env_list:
        logger.info(f"Start evaluating: {env_name}")
        logger.info(f"任务描述 (Prompt): {task_description_dict[env_name]}")
        env = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            render_mode="rgb_array",
            camera_id=2,
            disable_env_checker=True,
        )
        # Set camera position if necessary
        env.env.env.env.env.env.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        task_successes = 0
        for trial_id in range(num_trials_per_task):
            frames = []
            observation, info = env.reset()
            dummy_action = [0.0] * 4
            for _ in range(15):  # wait for objects to settle
                observation, _, _, _, _ = env.step(dummy_action)

            success = 0
            action_plan = collections.deque()

            for step in range(max_steps):
                image = env.render()
                image = image[::-1, ::-1]
                state = observation[:4]
                batch = {
                    "observation/image": image,
                    "observation/state": state,
                    "prompt": task_description_dict[env_name],
                }
                # Plan actions only if empty
                if not action_plan:
                    action_chunk = policy.infer(batch)["actions"]
                    action_plan.extend(action_chunk)
                action = action_plan.popleft()
                observation, reward, terminated, truncated, info = env.step(action)
                frames.append(image)
                if info.get("success", 0) or terminated or truncated:
                    success = int(info.get("success", 0))
                    # break  # end on success or termination

            # If episode succeeded, accumulate
            task_successes += success
            total_successes += success
            total_episodes += 1
            video_dir = f"video_{exp_name}"
            os.makedirs(video_dir, exist_ok=True)
            suffix = "success" if success else "failure"
            video_path = os.path.join(video_dir, f"{env_name}_{suffix}.mp4")
            imageio.mimsave(video_path, np.stack(frames), fps=25)

        env.close()
        task_success_rate = task_successes / num_trials_per_task
        results_per_task[env_name] = task_success_rate
        logger.info(
            f"Task: {env_name}, Successes: {task_successes}/{num_trials_per_task}, Success Rate: {task_success_rate:.2%}"
        )

    total_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    logger.info("\n===============")
    logger.info("Per-Task Success Rate:")
    for env_name, sr in results_per_task.items():
        logger.info(f"{env_name}: {sr:.2%}")

    # Calculate success rate by difficulty
    logger.info("\n===============")
    logger.info("Success Rate by Difficulty:")
    difficulty_rates = {}
    for difficulty, tasks in difficulty_to_tasks.items():
        task_rates = [
            results_per_task.get(task, 0.0)
            for task in tasks
            if task in results_per_task
        ]
        if task_rates:
            avg_rate = sum(task_rates) / len(task_rates)
            difficulty_rates[difficulty] = avg_rate
            logger.info(
                f"{difficulty}: {avg_rate:.2%} (averaged over {len(task_rates)} tasks)"
            )

    # Calculate overall average across all difficulties
    if difficulty_rates:
        overall_avg = sum(difficulty_rates.values()) / len(difficulty_rates)
        logger.info(
            f"\nOverall Average Success Rate (across all difficulties): {overall_avg:.2%}"
        )

    logger.info(
        f"\nTotal Success Rate: {total_successes}/{total_episodes} = {total_success_rate:.2%}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="metaworld_32")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/mnt/mnt/public/liuzhihao/openpi-main/checkpoints/sft/pi0_metaworld/metaworld_32",
    )
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=160)
    parser.add_argument("--action_chunk", type=int, default=5)
    args = parser.parse_args()
    main(args)
