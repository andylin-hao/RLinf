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

import os
import sys
import types
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Monkey-patch: inject a stub for rlinf.envs.libero.asset_paths so that
# env workers can import it even when the file is absent from the container.
# This avoids modifying the official rlinf/envs/libero/libero_env.py.
_OLD_ASSET_PATHS = sys.modules.get("rlinf.envs.libero.asset_paths")
if _OLD_ASSET_PATHS is None:
    _stub = types.ModuleType("rlinf.envs.libero.asset_paths")

    def _noop(*args, **kwargs):
        pass

    _stub.apply_standard_libero_env_vars = _noop
    sys.modules["rlinf.envs.libero.asset_paths"] = _stub


_LIBERO_EMBODIMENT_ALIASES = {"libero_panda", "libero_sim"}


def _is_libero_embodiment_tag(tag: object) -> bool:
    return str(tag).lower() in _LIBERO_EMBODIMENT_ALIASES


def _find_gr00t_checkpoint_file(path: str | Path) -> Path | None:
    candidate_root = Path(path)
    if candidate_root.is_file():
        return candidate_root

    candidate_paths = (
        candidate_root / "actor" / "model_state_dict" / "full_weights.pt",
        candidate_root / "actor" / "model_state_dict" / "model.pt",
        candidate_root / "model_state_dict" / "full_weights.pt",
    )
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def _classify_gr00t_model_input_path(path: str | Path) -> str:
    candidate = Path(path)
    if _find_gr00t_checkpoint_file(candidate) is not None:
        return "rlinf_checkpoint_directory"

    hf_required = {
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    }
    hf_weight_markers = {
        "model.safetensors",
        "pytorch_model.bin",
        "model-00001-of-00002.safetensors",
    }
    if candidate.is_dir():
        names = {child.name for child in candidate.iterdir()}
        if hf_required.issubset(names) and names.intersection(hf_weight_markers):
            return "official_hf_finetuned_directory"

    return "unknown"


def _resolve_gr00t_model_paths(cfg: DictConfig):
    model_path = Path(str(cfg.model_path))
    checkpoint_path = _find_gr00t_checkpoint_file(model_path)
    if checkpoint_path is None:
        processor_path = OmegaConf.select(cfg, "processor_path", default=None)
        backbone_model_path = OmegaConf.select(cfg, "backbone_model_path", default=None)
        return str(model_path), None, processor_path, backbone_model_path

    tensorboard_cfg_path = model_path / "tensorboard" / "config.yaml"
    if not tensorboard_cfg_path.exists():
        raise FileNotFoundError(
            f"Could not resolve GR00T base model for checkpoint directory: {model_path}"
        )

    saved_cfg = OmegaConf.load(tensorboard_cfg_path)
    saved_actor_model = OmegaConf.select(saved_cfg, "actor.model", default={})
    pretrained_model_path = str(saved_actor_model.get("model_path"))

    processor_path = OmegaConf.select(cfg, "processor_path", default=None)
    if processor_path is None:
        processor_path = saved_actor_model.get("processor_path")

    backbone_model_path = OmegaConf.select(cfg, "backbone_model_path", default=None)
    if backbone_model_path is None:
        backbone_model_path = saved_actor_model.get("backbone_model_path")

    return (
        pretrained_model_path,
        str(checkpoint_path),
        processor_path,
        backbone_model_path,
    )


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
    AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
    print(
        "Successfully registered custom architecture Gr00tN1d7, authentication passed!"
    )
    # FSDP wrap policy: N1.7 model module hierarchy differs slightly from N1.6,
    # causing exact class-name lookup (get_module_class_from_name) to miss some
    # backbone layers. Keyword-based auto-discovery is more robust.
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as fsdp_strategy

    if not hasattr(fsdp_strategy, "_is_gr00t_n1d7_patched"):
        _orig_policy = fsdp_strategy.get_fsdp_wrap_policy

        def _gr00t_n1d7_fsdp_wrap_policy(
            module, config=None, is_lora=False, model_type=None
        ):
            import functools

            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            found_classes = set()
            for _name, mod in module.named_modules():
                cname = mod.__class__.__name__
                if any(key in cname for key in ("DecoderLayer", "EncoderLayer")):
                    found_classes.add(mod.__class__)

            if found_classes:
                try:
                    result = _orig_policy(module, config, is_lora, model_type)
                    if result is not None:
                        return result
                except Exception:
                    pass
                return functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=found_classes,
                )
            return _orig_policy(module, config, is_lora, model_type)

        fsdp_strategy.get_fsdp_wrap_policy = _gr00t_n1d7_fsdp_wrap_policy
        fsdp_strategy._is_gr00t_n1d7_patched = True
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t_1_7.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t_1_7.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()

    from gr00t.data.embodiment_tags import EmbodimentTag

    from rlinf.models.embodiment.gr00t_1_7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t_1_7.utils import replace_dropout_with_identity

    if _is_libero_embodiment_tag(cfg.embodiment_tag):
        emb_tag = EmbodimentTag.LIBERO_PANDA
    elif cfg.embodiment_tag in [
        "libero_franka",
        "robocasa_panda_omron",
        "isaaclab_franka",
    ]:
        emb_tag = EmbodimentTag.LIBERO_PANDA
    elif cfg.embodiment_tag == "maniskill_widowx":
        emb_tag = getattr(EmbodimentTag, "MANISKILL_WIDOWX", EmbodimentTag.LIBERO_PANDA)
    elif cfg.embodiment_tag == "gr1":
        emb_tag = EmbodimentTag.GR1
    elif cfg.embodiment_tag == "behavior_r1_pro":
        emb_tag = EmbodimentTag.BEHAVIOR_R1_PRO
    else:
        raise ValueError(
            f"Invalid or unsupported embodiment tag: {cfg.embodiment_tag}. "
            f"Supported tags are: ['behavior_r1_pro', 'gr1', 'libero_panda', 'libero_sim']."
        )

    model_path = str(cfg.model_path)
    local_model_path = Path(model_path)

    processor_path = OmegaConf.select(cfg, "processor_path", default=None)
    backbone_model_path = OmegaConf.select(cfg, "backbone_model_path", default=None)
    libero_action_mode = OmegaConf.select(cfg, "libero_action_mode", default=None)
    libero_eval_action_mode = OmegaConf.select(
        cfg, "libero_eval_action_mode", default=None
    )

    model = GR00T_N1_7_ForRLActionPrediction.from_pretrained(
        local_model_path=str(local_model_path),
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        tune_visual=False,
        tune_llm=False,
        rl_head_config=cfg.rl_head_config,
        processor_path=processor_path,
        backbone_model_path=backbone_model_path,
        libero_action_mode=libero_action_mode,
        libero_eval_action_mode=libero_eval_action_mode,
    )

    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head and hasattr(model.action_head, "value_head"):
        # reinitialize the value head after model loading
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model


_RLINF_GLOBAL_PATCHED = False


def _patch_libero_calc_reward():
    """Patch LiberoEnv._calc_step_reward — disabled by default to match N1.5.

    The N1.5 reference uses the standard LIBERO reward.  This patch was adding
    a hard-coded reward shape (reward_coef * 5 success bonus, -0.01 non-terminal
    penalty) that makes experiments non-reproducible.

    Enable via env var RLINF_GR00T_PATCH_REWARD=1 if needed for legacy runs.
    """
    if os.environ.get("RLINF_GR00T_PATCH_REWARD", "0") != "1":
        return

    import numpy as np

    import rlinf.envs.libero.libero_env as le

    def _patched_calc(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        term_np = np.asarray(terminations, dtype=np.float32)
        success_bonus = self.cfg.reward_coef * 5.0
        termination_bonus = success_bonus * term_np
        reward = step_penalty + termination_bonus

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            if not self.use_step_penalty:
                non_term_mask = ~(term_np > 0)
                reward = np.asarray(reward, dtype=np.float32)
                reward = reward - 0.01 * non_term_mask
            return reward

    le.LiberoEnv._calc_step_reward = _patched_calc


def _patch_rollout_worker_predict():
    """Patch Rollout worker to add Gaussian noise to actions."""
    try:
        import torch

        from rlinf.workers.rollout.hf import huggingface_worker as hfw

        _orig_predict = hfw.MultiStepRolloutWorker.predict
        patch_noise_scale = 0.1

        def _patched_predict(self, env_obs, mode="train"):
            actions, result = _orig_predict(self, env_obs, mode)
            patch_noise_applied = False
            if mode == "train" and patch_noise_scale > 0:
                # Add Gaussian noise to actions for exploration
                noise = torch.randn_like(actions) * patch_noise_scale
                actions = actions + noise
                actions = torch.clamp(actions, -1.0, 1.0)
                patch_noise_applied = True

            result["_rlinf_patch_noise_applied"] = patch_noise_applied
            result["_rlinf_patch_noise_scale"] = (
                patch_noise_scale if patch_noise_applied else 0.0
            )
            return actions, result

        hfw.MultiStepRolloutWorker.predict = _patched_predict
        print(
            f"[GR00T patch] Rollout predict patched: Gaussian noise {patch_noise_scale} on actions"
        )
    except Exception as e:
        print(f"[GR00T patch] Failed to patch rollout predict: {e}")


def _patch_worker_init():
    """Patch EnvWorker.init_worker to apply reward patch before env creation."""
    try:
        from rlinf.scheduler import Worker

        for subcls in Worker.__subclasses__():
            if subcls.__name__ == "EnvWorker":
                orig_init = subcls.init_worker

                def _patched_init_worker(self):
                    try:
                        _patch_libero_calc_reward()
                    except Exception:
                        pass
                    return orig_init(self)

                subcls.init_worker = _patched_init_worker
                print("[GR00T patch] EnvWorker.init_worker patched to apply reward fix")
                return
    except Exception as e:
        print(f"[GR00T patch] Failed to patch EnvWorker.init_worker: {e}")


def _patch_get_env_cls():
    """Patch get_env_cls for good measure."""
    import rlinf.envs as env_module

    _orig = env_module.get_env_cls

    def _patched(env_type, env_cfg=None):
        result = _orig(env_type, env_cfg)
        if str(env_type) == "libero" or (
            hasattr(env_type, "value") and env_type.value == "libero"
        ):
            try:
                _patch_libero_calc_reward()
            except Exception:
                pass
        return result

    env_module.get_env_cls = _patched


def apply_global_rlinf_patches():
    global _RLINF_GLOBAL_PATCHED
    if _RLINF_GLOBAL_PATCHED:
        return
    _RLINF_GLOBAL_PATCHED = True

    # Only apply rollout predict patch at module level — it does not import
    # tensorflow and is needed by the RolloutGroup.  Env-related patches
    # (_patch_libero_calc_reward, _patch_get_env_cls) import tensorflow and
    # will crash Ray workers that don't need it (e.g. RolloutGroup, ActorGroup).
    # They are deferred and applied inside EnvWorker.init_worker via
    # _patch_worker_init below.

    # NOTE: _patch_rollout_worker_predict disabled — adding Gaussian noise to
    # rollout actions after logprobs are computed causes KL divergence to explode.
    # PPO's stochastic sampling already provides sufficient exploration.
    # try:
    #     _patch_rollout_worker_predict()
    # except Exception as e:
    #     print(f"[GR00T patch] Failed noise patch (rollout): {e}")

    try:
        _patch_worker_init()
    except Exception as e:
        print(f"[GR00T patch] Failed worker init patch: {e}")


apply_global_rlinf_patches()
