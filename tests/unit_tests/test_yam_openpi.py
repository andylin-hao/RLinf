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

"""YAM OpenPI transforms and the recorded-data SFT path."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def local_tokenizer(monkeypatch):
    pytest.importorskip("openpi")
    from openpi.models import tokenizer

    tokenized_input = {}

    class LocalTokenizer:
        def __init__(self, max_len):
            self.max_len = max_len

        def tokenize(self, prompt, state=None):
            tokenized_input["prompt"] = prompt
            tokenized_input["state"] = state
            return np.ones(self.max_len, dtype=np.int32), np.ones(
                self.max_len, dtype=bool
            )

    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", LocalTokenizer)
    return tokenized_input


@pytest.mark.parametrize("stacked", [False, True])
def test_yam_policy_preserves_joint_and_camera_order(stacked):
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.yam_policy import YamInputs, YamOutputs

    state = np.linspace(-0.2, 0.2, 14, dtype=np.float32)
    state[[6, 13]] = [0.25, 0.75]
    images = [np.full((8, 10, 3), value, dtype=np.uint8) for value in (11, 22, 33)]
    data = {
        "observation/state": state,
        "observation/image": images[0],
        "prompt": b"pick block",
    }
    if stacked:
        data["observation/extra_view_image"] = np.stack(images[1:])
    else:
        data["observation/extra_view_image-0"] = images[1].transpose(2, 0, 1) / 255.0
        data["observation/extra_view_image-1"] = images[2].transpose(2, 0, 1) / 255.0
    inputs = YamInputs()(data)
    np.testing.assert_array_equal(inputs["state"], state)
    for name, expected in zip(
        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), images, strict=True
    ):
        np.testing.assert_array_equal(inputs["image"][name], expected)
        assert inputs["image_mask"][name]
    assert inputs["prompt"] == "pick block"
    actions = np.tile(state, (30, 1))
    np.testing.assert_array_equal(
        YamOutputs()({"actions": np.pad(actions, ((0, 0), (0, 18)))})["actions"],
        actions,
    )


@pytest.mark.parametrize(
    "field",
    ["state_shape", "state_nan", "action_shape", "action_nan", "views", "image"],
)
def test_yam_policy_rejects_invalid_input(field):
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.yam_policy import YamInputs

    data = {
        "observation/state": np.zeros(14, dtype=np.float32),
        "observation/image": np.zeros((8, 10, 3), dtype=np.uint8),
        "observation/extra_view_image": np.zeros((2, 8, 10, 3), dtype=np.uint8),
        "actions": np.zeros((30, 14), dtype=np.float32),
    }
    if field == "state_shape":
        data["observation/state"] = np.zeros(13)
    elif field == "state_nan":
        data["observation/state"][0] = np.nan
    elif field == "action_shape":
        data["actions"] = np.zeros((30, 13))
    elif field == "action_nan":
        data["actions"][0, 0] = np.nan
    elif field == "views":
        data["observation/extra_view_image"] = data["observation/extra_view_image"][:1]
    else:
        data["observation/image"] = np.full((8, 10, 3), np.nan)
    with pytest.raises(ValueError):
        YamInputs()(data)


@pytest.mark.parametrize("actions", [np.zeros((30, 13)), np.full((30, 32), np.nan)])
def test_yam_policy_rejects_invalid_model_output(actions):
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.yam_policy import YamOutputs

    with pytest.raises(ValueError):
        YamOutputs()({"actions": actions})


def test_yam_joint_deltas_round_trip_without_changing_grippers(
    local_tokenizer, tmp_path
):
    from openpi import transforms

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config("pi05_yam_joint", model_path=str(tmp_path))
    assert config.model.action_horizon == 30
    assert config.model.discrete_state_input
    data_config = config.data.create(config.assets_dirs, config.model)
    state = np.linspace(-0.2, 0.2, 14, dtype=np.float32)
    state[[6, 13]] = [0.25, 0.75]
    actions = np.tile(state, (30, 1)) + 0.05
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    encoded = transforms.compose(data_config.data_transforms.inputs)(
        {
            "observation/state": state,
            "observation/image": image,
            "observation/extra_view_image": np.stack([image, image]),
            "actions": actions.copy(),
        }
    )
    joints = [*range(6), *range(7, 13)]
    np.testing.assert_allclose(encoded["actions"][:, joints], 0.05, atol=1e-7)
    np.testing.assert_array_equal(encoded["actions"][:, [6, 13]], actions[:, [6, 13]])
    recovered = transforms.compose(data_config.data_transforms.outputs)(
        {
            "state": np.pad(state, (0, 18)),
            "actions": np.pad(encoded["actions"], ((0, 0), (0, 18))),
        }
    )
    np.testing.assert_allclose(recovered["actions"], actions, atol=1e-7)


@pytest.mark.parametrize(
    "config_directory,config_name,model_role,executed_length",
    [
        ("examples/sft/config", "realworld_dual_yam_sft_openpi_pi05", "actor", 30),
        ("evaluations/realworld", "realworld_dual_yam_openpi_pi05_eval", "rollout", 5),
        ("tests/e2e_tests/sft", "yam_sft_openpi_pi05", "actor", 30),
        (
            "tests/e2e_tests/evaluations",
            "realworld_dual_yam_openpi_pi05_mock_eval",
            "rollout",
            5,
        ),
    ],
)
def test_yam_openpi_recipes_use_the_same_model_contract(
    config_directory, config_name, model_role, executed_length, monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[2]
    for name, value in {
        "REPO_PATH": str(root),
        "EMBODIED_PATH": str(root / "examples/embodiment"),
        "YAM_TASK_DESCRIPTION": "pick block",
        "YAM_CI_LOG_PATH": str(tmp_path / "logs"),
        "YAM_CI_DATASET": str(tmp_path / "dataset"),
        "YAM_CI_CHECKPOINT": str(tmp_path / "checkpoint"),
    }.items():
        monkeypatch.setenv(name, value)
    with initialize_config_dir(
        config_dir=str(root / config_directory), version_base="1.1"
    ):
        config = compose(config_name=config_name)
    if model_role == "actor":
        from rlinf.config import validate_sft_cfg

        validate_sft_cfg(config)
    resolved = OmegaConf.to_container(config, resolve=True)
    model = resolved[model_role]["model"]
    assert model["openpi"]["config_name"] == "pi05_yam_joint"
    assert model["openpi"]["action_horizon"] == 30
    assert model["openpi"]["discrete_state_input"]
    assert model["action_dim"] == 14
    assert model["num_action_chunks"] == executed_length
    assert model["openpi"]["action_chunk"] == executed_length
    if config_name == "yam_sft_openpi_pi05":
        assert resolved["cluster"]["component_placement"]["actor"] == 0
        assert resolved["actor"]["micro_batch_size"] == 1
        assert resolved["actor"]["global_batch_size"] == 1
        assert model["precision"] == "bf16"
        assert resolved["runner"]["max_steps"] == 2
        assert resolved["runner"]["save_interval"] == 2


def test_yam_recorded_data_runs_through_statistics_and_official_sft_loader(
    local_tokenizer, tmp_path, monkeypatch
):
    import torch

    from rlinf.data.datasets.openpi import build_openpi_sft_dataloader
    from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter
    from toolkits.lerobot.calculate_norm_stats import main as calculate_norm_stats

    dataset_path = tmp_path / "dataset"
    writer = LeRobotDatasetWriter()
    writer.create(
        repo_id=str(dataset_path),
        robot_type="dual_yam",
        fps=30,
        state_dim=14,
        action_dim=14,
        image_shape=(8, 10, 3),
        extra_view_image_keys={
            "extra_view_image-0": (8, 10, 3),
            "extra_view_image-1": (8, 10, 3),
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )
    frames = []
    for index in range(40):
        state = np.full(14, index / 100, dtype=np.float32)
        state[[6, 13]] = [0.25, 0.75]
        frames.append(
            {
                "state": state,
                "actions": state + 0.02,
                "task": "pick block",
                "image": np.full((8, 10, 3), 11, dtype=np.uint8),
                "extra_view_image-0": np.full((8, 10, 3), 22, dtype=np.uint8),
                "extra_view_image-1": np.full((8, 10, 3), 33, dtype=np.uint8),
                "done": np.array([index == 39]),
                "is_success": np.array([True]),
                "intervene_flag": np.array([True]),
            }
        )
    writer.add_episode(frames)
    writer.finalize()
    monkeypatch.chdir(tmp_path)
    calculate_norm_stats(config_name="pi05_yam_joint", repo_id=str(dataset_path))
    assert (dataset_path / "norm_stats.json").is_file()
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/sft/config"), version_base="1.1"
    ):
        config = compose(
            config_name="realworld_dual_yam_sft_openpi_pi05",
            overrides=[
                f"data.train_data_paths={dataset_path}",
                "data.num_workers=0",
                "actor.micro_batch_size=1",
                f"actor.model.openpi_data.norm_stats_path={dataset_path / 'norm_stats.json'}",
            ],
        )
    loader, _ = build_openpi_sft_dataloader(
        config, world_size=1, rank=0, data_paths=str(dataset_path)
    )
    observation, actions = next(iter(loader))
    assert observation.state.shape == (1, 32)
    assert actions.shape == (1, 30, 32)
    assert torch.isfinite(observation.state).all() and torch.isfinite(actions).all()
    means = [
        observation.images[name].mean().item()
        for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    ]
    assert means[0] < means[1] < means[2]
    assert observation.tokenized_prompt.shape == (1, 200)
    assert observation.tokenized_prompt_mask.any()
    assert local_tokenizer["prompt"] == "pick block"
    assert local_tokenizer["state"].shape == (14,)
