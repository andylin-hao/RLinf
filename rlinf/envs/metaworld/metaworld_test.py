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

seed = 42

envs = gym.make_vec(
    "Meta-World/MT50", vector_strategy="async", seed=seed
)  # this returns a Synchronous Vector Environment with 50 environments

obs, info = envs.reset()  # reset all 50 environments

a = envs.action_space.sample()  # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a)  # step all 50 environments

print(obs.shape)
