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

import sys
import types

import pytest
import torch


class _DummyStarVLAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.tensor(1.0))
        self.qwen_vl_interface = types.SimpleNamespace(
            model=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=16))
        )
        self.framework_name = "QwenFast"
        self.config = types.SimpleNamespace(
            framework=types.SimpleNamespace(framework_name="QwenFast"),
        )
        self.norm_stats = None

    def parameters(self, recurse: bool = True):
        return iter([self._p])

    def to(self, *args, **kwargs):
        return self


def _install_fake_starvla_modules(monkeypatch, called):
    starvla_pkg = types.ModuleType("starVLA")
    starvla_pkg.__path__ = []
    starvla_model_pkg = types.ModuleType("starVLA.model")
    starvla_model_pkg.__path__ = []
    starvla_framework_pkg = types.ModuleType("starVLA.model.framework")
    starvla_framework_pkg.__path__ = []

    baseframework_mod = types.ModuleType("starVLA.model.framework.base_framework")

    class _FakeBaseFramework:
        @staticmethod
        def from_pretrained(path):
            called["ckpt_path"] = path
            return _DummyStarVLAModel()

    baseframework_mod.baseframework = _FakeBaseFramework

    monkeypatch.setitem(sys.modules, "starVLA", starvla_pkg)
    monkeypatch.setitem(sys.modules, "starVLA.model", starvla_model_pkg)
    monkeypatch.setitem(sys.modules, "starVLA.model.framework", starvla_framework_pkg)
    monkeypatch.setitem(
        sys.modules, "starVLA.model.framework.base_framework", baseframework_mod
    )


def test_get_model_requires_model_path():
    from rlinf.models.embodiment.starvla import get_model

    cfg = types.SimpleNamespace(
        model_path=None,
        action_dim=7,
        num_action_chunks=8,
    )
    with pytest.raises(ValueError):
        get_model(cfg, torch_dtype=None)


def test_starvla_loader_wraps_checkpoint(monkeypatch):
    called = {}
    _install_fake_starvla_modules(monkeypatch, called)

    from rlinf.models.embodiment.starvla import get_model
    from rlinf.models.embodiment.starvla.starvla_action_model import (
        StarVLAForRLActionPrediction,
    )

    cfg = types.SimpleNamespace(
        starvla=types.SimpleNamespace(framework_name="QwenFast"),
        model_path="dummy.pt",
        action_dim=7,
        num_action_chunks=8,
        add_value_head=True,
        unnorm_key=None,
    )

    model = get_model(cfg, torch_dtype=None)
    assert called["ckpt_path"] == "dummy.pt"
    assert isinstance(model, StarVLAForRLActionPrediction)
