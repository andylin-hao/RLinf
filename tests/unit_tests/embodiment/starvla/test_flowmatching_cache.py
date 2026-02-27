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

import pytest

from rlinf.models.embodiment.starvla.utils.action_heads import (
    resolve_flowmatching_prefix,
)
from rlinf.models.embodiment.starvla.utils.profile import RL_BATCH_TENSOR_KEYS_TO_IGNORE


def test_flowmatching_prefix_resolution():
    assert resolve_flowmatching_prefix("pi") == "pi"
    assert resolve_flowmatching_prefix("gr00t") == "gr00t"
    assert resolve_flowmatching_prefix("dual") == "dual"


def test_flowmatching_prefix_rejects_unsupported_head():
    with pytest.raises(NotImplementedError):
        resolve_flowmatching_prefix("fast")


def test_flowmatching_cache_keys_are_ignored_by_batch_tensor_collector():
    for prefix in ("pi", "gr00t", "dual"):
        assert f"{prefix}_chain_actions" in RL_BATCH_TENSOR_KEYS_TO_IGNORE
        assert f"{prefix}_t_bucket_indices" in RL_BATCH_TENSOR_KEYS_TO_IGNORE
        assert f"{prefix}_num_steps" in RL_BATCH_TENSOR_KEYS_TO_IGNORE
        assert f"{prefix}_sample_actions" in RL_BATCH_TENSOR_KEYS_TO_IGNORE
        assert f"{prefix}_step_std" in RL_BATCH_TENSOR_KEYS_TO_IGNORE
