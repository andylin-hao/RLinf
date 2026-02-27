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

import types

import numpy as np
import torch

from rlinf.models.embodiment.starvla.utils.action_space import (
    clip_actions_for_env,
    normalize_actions_for_model,
    resolve_action_norm_stats,
    unnormalize_actions_for_env,
)


def _dummy_model_with_norm_stats():
    norm_stats = {
        "franka": {
            "action": {
                "q99": [1.0] * 7,
                "q01": [-1.0] * 7,
                "mask": [True] * 7,
            }
        }
    }
    return types.SimpleNamespace(
        norm_stats=norm_stats, get_action_stats=lambda key: None
    )


def test_resolve_action_norm_stats_fallback_to_single_key():
    model = _dummy_model_with_norm_stats()
    stats, resolved_key = resolve_action_norm_stats(
        starvla_model=model,
        unnorm_key="missing_key",
        action_dim=7,
    )
    assert resolved_key == "franka"
    assert stats is not None
    assert stats["high"].shape[-1] == 7


def test_unnormalize_and_normalize_roundtrip_without_mapping(monkeypatch):
    monkeypatch.delenv("ROBOT_PLATFORM", raising=False)

    stats = {
        "high": np.ones((7,), dtype=np.float32),
        "low": -np.ones((7,), dtype=np.float32),
        "mask": np.ones((7,), dtype=bool),
    }

    normalized = np.array([[[0.0, 0.2, -0.5, 1.0, -1.0, 0.3, 0.8]]], dtype=np.float32)
    env_actions, warned = unnormalize_actions_for_env(
        normalized,
        action_norm_stats=stats,
        warned_missing_action_norm_stats=False,
    )
    assert warned is False
    env_tensor = torch.from_numpy(env_actions)
    back = normalize_actions_for_model(env_tensor, action_norm_stats=stats)

    expected = np.clip(normalized, -1.0, 1.0)
    expected[..., 6] = np.where(expected[..., 6] < 0.5, 0.0, 1.0)
    assert torch.allclose(back, torch.from_numpy(expected), atol=1e-5)


def test_libero_gripper_mapping(monkeypatch):
    monkeypatch.setenv("ROBOT_PLATFORM", "LIBERO")
    monkeypatch.setenv("STARVLA_LIBERO_GRIPPER_MODE", "open_is_one")

    normalized = np.zeros((1, 1, 7), dtype=np.float32)
    normalized[..., 6] = 1.0
    env_actions, _ = unnormalize_actions_for_env(
        normalized,
        action_norm_stats=None,
        warned_missing_action_norm_stats=True,
    )
    assert env_actions[..., 6].item() == -1.0

    back = normalize_actions_for_model(
        torch.from_numpy(env_actions), action_norm_stats=None
    )
    assert back[..., 6].item() == 1.0


def test_clip_actions_for_env_binarizes_gripper():
    actions = torch.tensor(
        [[[2.0, -2.0, 0.1, 0.2, 0.3, 0.4, 0.49]]], dtype=torch.float32
    )
    clipped = clip_actions_for_env(actions)

    assert clipped[..., :6].max().item() <= 1.0
    assert clipped[..., :6].min().item() >= -1.0
    assert clipped[..., 6].item() == 0.0
