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

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import warnings

from .modules.nature_cnn import NatureCNN, PlainConv, ResNetEncoder
from .modules.utils import layer_init, make_mlp, init_mlp_weights
from .modules.value_head import ValueHead
from .modules.q_head import MultiQHead
from .modules.flow_actor import FlowTActor, JaxFlowTActor
from .base_policy import BasePolicy

@dataclass
class FlowConfig:
    image_size: List[int] = field(default_factory=list)
    image_keys: List[str] = field(default_factory=str)
    action_dim: int = 4
    state_dim: int = 29
    num_action_chunks: int = 1
    backbone: str = "resnet"
    extra_config: Dict[str, Any] = field(default_factory=dict)
    add_value_head: bool = False
    add_q_head: bool = False

    state_latent_dim: int = 64
    action_scale = None
    final_tanh = True
    
    # Flow Matching specific parameters
    denoising_steps: int = 4
    d_model: int = 96
    n_head: int = 4
    n_layers: int = 2
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    flow_actor_type: str = "FlowTActor"  # "FlowTActor" or "JaxFlowTActor"

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                warnings.warn(f"FlowConfig does not contain the key {key=}")
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            self.action_scale = -1, 1
            self.final_tanh = True


class FlowPolicy(BasePolicy):
    def __init__(
            self, cfg: FlowConfig
        ):
        super().__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.image_size[0]

        # Image encoders (same as CNNPolicy)
        self.encoders = nn.ModuleDict()
        encoder_out_dim = 0
        if self.cfg.backbone == "plane_conv":
            for key in self.cfg.image_keys:
                self.encoders[key] = PlainConv(
                    in_channels=self.in_channels, out_dim=256, image_size=self.cfg.image_size
                )
                encoder_out_dim += self.encoders[key].out_dim
        elif self.cfg.backbone == "resnet":
            sample_x = torch.randn(1, *self.cfg.image_size)
            for key in self.cfg.image_keys:
                self.encoders[key] = ResNetEncoder(
                    sample_x, out_dim=256, model_cfg=self.cfg.extra_config
                )
                encoder_out_dim += self.encoders[key].out_dim
        else:
            raise NotImplementedError
        
        # State projection
        self.state_proj = nn.Sequential(*make_mlp(
            in_channels=self.cfg.state_dim,
            mlp_channels=[self.cfg.state_latent_dim, ], 
            act_builder=nn.Tanh, 
            use_layer_norm=True
        ))
        init_mlp_weights(self.state_proj, nonlinearity="tanh")
        
        # Mix projection to combine visual and state features
        self.mix_proj = nn.Sequential(*make_mlp(
            in_channels=encoder_out_dim+self.cfg.state_latent_dim, 
            mlp_channels=[256, 256],  
            act_builder=nn.Tanh, 
            use_layer_norm=True
        ))
        init_mlp_weights(self.mix_proj, nonlinearity="tanh")

        # Create flow actor
        # FlowTActor will receive mix_feature (256 dim) as obs input
        # So we set obs_dim to 256 (output of mix_proj)
        flow_obs_dim = 256
        
        # Action scaling for flow actor
        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            action_scale = torch.tensor((h - l) / 2.0, dtype=torch.float32)
            action_bias = torch.tensor((h + l) / 2.0, dtype=torch.float32)
        else:
            # Default to [-1, 1] range
            action_scale = torch.ones(self.cfg.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(self.cfg.action_dim, dtype=torch.float32)
        
        if self.cfg.flow_actor_type == "FlowTActor":
            self.flow_actor = FlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias
            )
        elif self.cfg.flow_actor_type == "JaxFlowTActor":
            self.flow_actor = JaxFlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias
            )
        else:
            raise ValueError(f"Unknown flow_actor_type: {self.cfg.flow_actor_type}")

        # Q-head for SAC
        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, 
                hidden_sizes=(256, 256, 256), 
                activation="relu"
            )
        if self.cfg.add_q_head:
            self.q_head = MultiQHead(
                hidden_size=encoder_out_dim+self.cfg.state_latent_dim,
                hidden_dims=[256, 256, 256], 
                num_q_heads=2, 
                action_feature_dim=self.cfg.action_dim
            )

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        else:
            self.action_scale = None

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed_env_obs = {}
        processed_env_obs["states"] = env_obs["states"].clone().to(device)
        for key, value in env_obs["images"].items():
            processed_env_obs[f"images/{key}"] = value.clone().to(device).float() / 255.0
        return processed_env_obs
    
    def get_feature(self, obs):
        """Extract features from observations (images + states)"""
        visual_features = []
        for key in self.cfg.image_keys:
            visual_features.append(self.encoders[key](obs[f"images/{key}"]))
        visual_feature = torch.cat(visual_features, dim=-1)
        
        state_feature = self.state_proj(obs["states"])
        full_feature = torch.cat([visual_feature, state_feature], dim=-1)
        
        return full_feature, visual_feature

    def sac_forward(self, obs, **kwargs):
        """SAC forward pass using Flow Matching actor"""
        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)
        
        # Use flow actor to generate actions
        # FlowTActor expects obs as input, we pass mix_feature as the observation
        action, log_prob = self.flow_actor(mix_feature, train=True, log_grad=False)
        
        return action, log_prob, full_feature

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Get Q-values for given observations and actions"""
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)

    def default_forward(self, obs, compute_entropy=False, compute_values=False, **kwargs):
        """Default forward pass"""
        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)
        
        # Use flow actor
        action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)
        
        output_dict = {
            "action": action,
            "log_prob": log_prob,
        }
        
        if compute_entropy:
            # For flow matching, entropy is computed from log_prob
            # Approximate entropy as negative log_prob (this is a simplification)
            entropy = -log_prob
            output_dict.update(entropy=entropy)
        if compute_values:
            if getattr(self, "value_head", None):
                values = self.value_head(mix_feature)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def predict_action_batch(
            self, env_obs, 
            calulate_logprobs=True,
            calulate_values=True,
            return_obs=True, 
            return_action_type="numpy_chunk", 
            return_shared_feature=False, 
            **kwargs
        ):
        """Predict actions in batch"""
        full_feature, visual_feature = self.get_feature(env_obs)
        mix_feature = self.mix_proj(full_feature)
        
        # Use flow actor
        action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)
        
        if return_action_type == "numpy_chunk":
            chunk_actions = action.reshape(-1, self.cfg.num_action_chunks, self.cfg.action_dim)
            chunk_actions = chunk_actions.cpu().numpy()
        elif return_action_type == "torch_flatten":
            chunk_actions = action.clone()
        else:
            raise NotImplementedError
        
        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(mix_feature)
        else:
            chunk_values = torch.zeros_like(log_prob[..., :1])
        
        forward_inputs = {
            "action": action
        }
        if return_obs:
            for key, value in env_obs.items():
                forward_inputs[f"obs/{key}"] = value
        
        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = visual_feature
        return chunk_actions, result
    
@dataclass
class FlowStateConfig:
    action_dim: int = 4
    obs_dim: int = 29
    num_action_chunks: int = 1
    extra_config: Dict[str, Any] = field(default_factory=dict)
    add_value_head: bool = False
    add_q_head: bool = False
    q_head_type: str = "default"

    action_scale = None
    final_tanh = True
    
    # Flow Matching specific parameters
    denoising_steps: int = 4
    d_model: int = 96
    n_head: int = 4
    n_layers: int = 2
    use_batch_norm: bool = False
    batch_norm_momentum: float = 0.99
    flow_actor_type: str = "FlowTActor"  # "FlowTActor" or "JaxFlowTActor"

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                warnings.warn(f"FlowConfig does not contain the key {key=}")
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            self.action_scale = -1, 1
            self.final_tanh = True

class FlowStatePolicy(BasePolicy):
    def __init__(
            self, cfg: FlowStateConfig
        ):
        super().__init__()
        self.cfg = cfg
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(self.cfg.obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),   
        )
        # Create flow actor
        # FlowTActor will receive mix_feature (256 dim) as obs input
        # So we set obs_dim to 256 (output of mix_proj)
        flow_obs_dim = 256
        
        # Action scaling for flow actor
        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            action_scale = torch.tensor((h - l) / 2.0, dtype=torch.float32)
            action_bias = torch.tensor((h + l) / 2.0, dtype=torch.float32)
        else:
            # Default to [-1, 1] range
            action_scale = torch.ones(self.cfg.action_dim, dtype=torch.float32)
            action_bias = torch.zeros(self.cfg.action_dim, dtype=torch.float32)
        
        if self.cfg.flow_actor_type == "FlowTActor":
            self.flow_actor = FlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias
            )
        elif self.cfg.flow_actor_type == "JaxFlowTActor":
            self.flow_actor = JaxFlowTActor(
                obs_dim=flow_obs_dim,
                action_dim=self.cfg.action_dim,
                d_model=self.cfg.d_model,
                n_head=self.cfg.n_head,
                n_layers=self.cfg.n_layers,
                denoising_steps=self.cfg.denoising_steps,
                use_batch_norm=self.cfg.use_batch_norm,
                batch_norm_momentum=self.cfg.batch_norm_momentum,
                action_scale=action_scale,
                action_bias=action_bias
            )
        else:
            raise ValueError(f"Unknown flow_actor_type: {self.cfg.flow_actor_type}")

        # Q-head for SAC
        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, 
                hidden_sizes=(256, 256, 256), 
                activation="relu"
            )
        if self.cfg.add_q_head:
            self.q_head = MultiQHead(
                hidden_size=self.cfg.obs_dim,
                hidden_dims=[256, 256, 256], 
                num_q_heads=2, 
                action_feature_dim=self.cfg.action_dim
            )

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        else:
            self.action_scale = None

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        return {"states": env_obs["states"].to(device)}

    def sac_forward(self, obs, **kwargs):
        """SAC forward pass using Flow Matching actor"""
        feat = self.backbone(obs["states"])
        
        # Use flow actor to generate actions
        # FlowTActor expects obs as input, we pass mix_feature as the observation
        action, log_prob = self.flow_actor(feat, train=True, log_grad=False)
        
        return action, log_prob, None

    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        """Get Q-values for given observations and actions"""
        return self.q_head(obs["states"], actions)

    def default_forward(self, obs, compute_entropy=False, compute_values=False, **kwargs):
        """Default forward pass"""
        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)
        
        # Use flow actor
        action, log_prob = self.flow_actor(mix_feature, train=False, log_grad=False)
        
        output_dict = {
            "action": action,
            "log_prob": log_prob,
        }
        
        if compute_entropy:
            # For flow matching, entropy is computed from log_prob
            # Approximate entropy as negative log_prob (this is a simplification)
            entropy = -log_prob
            output_dict.update(entropy=entropy)
        if compute_values:
            if getattr(self, "value_head", None):
                values = self.value_head(mix_feature)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def predict_action_batch(
            self, env_obs, 
            calulate_logprobs=True,
            calulate_values=True,
            return_obs=True, 
            return_action_type="numpy_chunk", 
            return_shared_feature=False, 
            **kwargs
        ):
        """Predict actions in batch"""
        feat = self.backbone(env_obs["states"])
        
        # Use flow actor
        action, log_prob = self.flow_actor(feat, train=False, log_grad=False)
        
        if return_action_type == "numpy_chunk":
            chunk_actions = action.reshape(-1, self.cfg.num_action_chunks, self.cfg.action_dim)
            chunk_actions = chunk_actions.cpu().numpy()
        elif return_action_type == "torch_flatten":
            chunk_actions = action.clone()
        else:
            raise NotImplementedError
        
        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(mix_feature)
        else:
            chunk_values = torch.zeros_like(log_prob[..., :1])
        
        forward_inputs = {
            "action": action
        }
        if return_obs:
            for key, value in env_obs.items():
                forward_inputs[f"obs/{key}"] = value
        
        result = {
            "prev_logprobs": log_prob,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = visual_feature
        return chunk_actions, result