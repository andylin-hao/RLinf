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

import torch
from omegaconf import DictConfig, OmegaConf


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    '''
    Get the FlowStatePolicy for now.
    cfg: All configs under [actor.model] in yaml file.
    '''
    # print("--------------------------------------\n"*10)
    # print("In rlinf/models/embodiment/flow_policy/__init__.py, cfg is:",cfg)
    # print("--------------------------------------\n"*10)

    if cfg.input_type == "state":
        from rlinf.models.embodiment.flow_policy.flow_policy import FlowStateConfig, FlowStatePolicy
        model_config = FlowStateConfig()
        model_config.update_from_dict(OmegaConf.to_container(cfg, resolve=True))
        model = FlowStatePolicy(model_config)
    elif cfg.input_type == "mixed":
        from rlinf.models.embodiment.flow_policy.flow_policy import FlowConfig, FlowPolicy
        model_config = FlowConfig()
        model_config.update_from_dict(OmegaConf.to_container(cfg, resolve=True))
        model = FlowPolicy(model_config)
    else:
        raise NotImplementedError(f"{cfg.input_type=}")
        
    print("--------------------------------------\n"*10)
    print("In rlinf/models/embodiment/flow_policy/__init__.py, cfg is:",cfg)
    print("get_model() returns:",model)
    print("--------------------------------------\n"*10)

    

    return model
