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

from rlinf.models.embodiment.starvla.dispatch import (
    DEFAULT_FORWARD_HANDLERS,
    ROLLOUT_HANDLERS,
    get_default_forward_handler,
    get_rollout_handler,
)
from rlinf.models.embodiment.starvla.utils.profile import ACTION_HEAD_TYPES


def test_default_forward_dispatch_has_all_frameworks():
    assert set(DEFAULT_FORWARD_HANDLERS.keys()) == ACTION_HEAD_TYPES
    for name in ACTION_HEAD_TYPES:
        assert callable(get_default_forward_handler(name))


def test_rollout_dispatch_has_all_frameworks():
    assert set(ROLLOUT_HANDLERS.keys()) == ACTION_HEAD_TYPES
    for name in ACTION_HEAD_TYPES:
        assert callable(get_rollout_handler(name))


def test_unknown_action_head_returns_none():
    assert get_default_forward_handler("unknown_head") is None
    assert get_rollout_handler("unknown_head") is None
