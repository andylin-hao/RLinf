# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming Flow Policy knobs read from ``actor.model.openpi``."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class OpenPiPytorchSfpConfig:
    """Selects the SFP objective and its noise schedule.

    ``use_sfp`` picks the training objective only: the parameters, their shapes,
    and the checkpoint layout are the same as an ordinary Pi0.5, so an SFP run
    starts from a stock Pi0.5 checkpoint and its output loads back into one.
    """

    use_sfp: bool = False
    sigma: float = 0.16
    noise_decay: float = 4.0


def build_sfp_config(model_cfg: Any) -> OpenPiPytorchSfpConfig:
    """Build the optional SFP config from ``actor.model.openpi``."""
    from omegaconf import OmegaConf

    return OpenPiPytorchSfpConfig(
        use_sfp=bool(OmegaConf.select(model_cfg, "use_sfp", default=False)),
        sigma=float(OmegaConf.select(model_cfg, "sfp_sigma", default=0.16)),
        noise_decay=float(OmegaConf.select(model_cfg, "sfp_noise_decay", default=4.0)),
    )
