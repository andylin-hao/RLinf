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

"""Picking the HuggingFace attention implementation a venv can actually run."""

from __future__ import annotations

import transformers.utils as transformers_utils

from rlinf.utils.logging import get_logger

logger = get_logger()

# Newest first: a venv holds at most one flash-attention variant, because FA2
# and FA4 share the flash_attn package, so at most one of these can be True.
_FLASH_IMPLEMENTATIONS = {
    "flash_attention_4": "is_flash_attn_4_available",
    "flash_attention_3": "is_flash_attn_3_available",
    "flash_attention_2": "is_flash_attn_2_available",
}


def _is_available(implementation: str) -> bool:
    """Whether transformers can run `implementation` in this venv."""
    check = _FLASH_IMPLEMENTATIONS.get(implementation)
    if check is None:
        return True
    # Older transformers releases know fewer flash variants.
    probe = getattr(transformers_utils, check, None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # a probe may import the kernels and fail on the platform
        return False


def resolve_attn_implementation(preferred: str = "flash_attention_2") -> str:
    """Return `preferred`, or the closest implementation this venv can run.

    Images built for sm90+ ship FA4 instead of FA2, and platforms without
    flash-attn (Ascend, MUSA) ship neither, so a hard-coded ``flash_attention_2``
    fails there. Fall back to another installed flash variant when there is one,
    and to ``sdpa`` otherwise.
    """
    if _is_available(preferred):
        return preferred
    for implementation in _FLASH_IMPLEMENTATIONS:
        if implementation != preferred and _is_available(implementation):
            logger.warning(
                f"{preferred} is unavailable; using {implementation} instead."
            )
            return implementation
    logger.warning(
        f"{preferred} is unavailable and no flash-attn is installed; using sdpa."
    )
    return "sdpa"


def boolean_attention_mask(attn_mask):
    """The boolean form of a {0, -inf} float mask, or None if it carries biases."""
    import torch

    if attn_mask is None or not attn_mask.dtype.is_floating_point:
        return None
    floor = torch.finfo(attn_mask.dtype).min * 0.5
    keep = attn_mask >= 0
    if bool((keep | (attn_mask <= floor)).all()):
        return keep
    return None


def install_npu_sdpa_mask_cast() -> bool:
    """Cast {0, -inf} float SDPA masks to boolean so the NPU keeps its fused kernel.

    torch-npu decomposes SDPA into matmul-softmax for floating masks (4-13x
    slower); the additive causal masks HF models build are boolean in content.
    Masks carrying real biases are left untouched. Idempotent; returns whether
    the wrapper is installed.
    """
    import torch

    functional = torch.nn.functional
    if getattr(functional.scaled_dot_product_attention, "_rlinf_npu_mask_cast", False):
        return True
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False

    original = functional.scaled_dot_product_attention

    def scaled_dot_product_attention(
        query, key, value, attn_mask=None, *args, **kwargs
    ):
        if attn_mask is not None and attn_mask.dtype.is_floating_point:
            floor = torch.finfo(attn_mask.dtype).min * 0.5
            keep = attn_mask >= 0
            if bool((keep | (attn_mask <= floor)).all()):
                attn_mask = keep
        return original(query, key, value, attn_mask, *args, **kwargs)

    scaled_dot_product_attention._rlinf_npu_mask_cast = True
    functional.scaled_dot_product_attention = scaled_dot_product_attention
    logger.info("NPU SDPA float-mask cast installed (FlashAttentionScore path).")
    return True


def install_npu_fused_rotary() -> bool:
    """Serve HF Llama's rotary embedding from ``npu_rotary_mul`` (3x faster).

    Module-level, so every Llama-family model built afterwards picks it up.
    Idempotent; returns whether the patch is installed.
    """
    import torch

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False
    try:
        import torch_npu
        from transformers.models.llama import modeling_llama
    except ImportError:
        return False
    if getattr(modeling_llama.apply_rotary_pos_emb, "_rlinf_npu_fused", False):
        return True

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (
            torch_npu.npu_rotary_mul(q, cos, sin),
            torch_npu.npu_rotary_mul(k, cos, sin),
        )

    apply_rotary_pos_emb._rlinf_npu_fused = True
    modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb
    logger.info("NPU fused rotary embedding installed for Llama models.")
    return True
