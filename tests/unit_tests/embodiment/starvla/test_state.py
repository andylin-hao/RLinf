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
import warnings

import torch
import torch.nn as nn

from rlinf.models.embodiment.starvla.utils.state import (
    adapt_state_for_expected_dim,
    infer_expected_state_dim_from_head,
    prepare_state_tensor,
)


class _DummyHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(in_features, 16), nn.ReLU())


def _dummy_starvla_model_with_state_dim(state_dim: int):
    action_model_cfg = types.SimpleNamespace(state_dim=state_dim)
    framework = types.SimpleNamespace(action_model=action_model_cfg)
    config = types.SimpleNamespace(framework=framework)
    return types.SimpleNamespace(config=config)


def test_infer_expected_state_dim_from_head():
    head = _DummyHead(in_features=9)
    assert infer_expected_state_dim_from_head(head) == 9


def test_prepare_state_tensor_accepts_1d_2d_3d():
    model = _dummy_starvla_model_with_state_dim(7)
    warned = set()

    s1 = prepare_state_tensor(
        torch.randn(7),
        starvla_model=model,
        default_state_adapter_name="adapter",
        warned_keys=warned,
    )
    s2 = prepare_state_tensor(
        torch.randn(2, 7),
        starvla_model=model,
        default_state_adapter_name="adapter",
        warned_keys=warned,
    )
    s3 = prepare_state_tensor(
        torch.randn(2, 1, 7),
        starvla_model=model,
        default_state_adapter_name="adapter",
        warned_keys=warned,
    )

    assert s1.shape == (1, 1, 7)
    assert s2.shape == (2, 1, 7)
    assert s3.shape == (2, 1, 7)


def test_prepare_state_tensor_adapts_8_to_7_once_warning():
    model = _dummy_starvla_model_with_state_dim(7)
    warned = set()
    state = torch.randn(2, 1, 8)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out1 = prepare_state_tensor(
            state,
            starvla_model=model,
            default_state_adapter_name="adapter",
            warned_keys=warned,
            context="unit_test",
        )
        out2 = prepare_state_tensor(
            state,
            starvla_model=model,
            default_state_adapter_name="adapter",
            warned_keys=warned,
            context="unit_test",
        )

    assert out1.shape[-1] == 7
    assert out2.shape[-1] == 7
    assert len(rec) == 1


def test_adapt_state_truncate_and_pad():
    warned = set()
    trunc = adapt_state_for_expected_dim(torch.randn(1, 1, 10), 7, "t", warned)
    pad = adapt_state_for_expected_dim(torch.randn(1, 1, 5), 7, "p", warned)

    assert trunc.shape[-1] == 7
    assert pad.shape[-1] == 7
