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

from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml
from omegaconf import DictConfig

from ..hardware import NodeHardwareConfig
from .utils import dataclass_arg_check, parse_rank_config


@dataclass
class NodeGroupConfig:
    """Configuration for a group of nodes in the cluster with the same label.

    A node group is used to represent multiple nodes with identical hardware configurations.
    """

    label: str
    node_ranks: list[int]
    env_vars: Optional[list[dict[str, str]]] = field(default=None)
    hardware: Optional[list[NodeHardwareConfig]] = field(default=None)

    def __post_init__(self):
        """Post-initialization to convert hardware dicts to their respective dataclass instances."""
        if self.hardware is not None:
            # Arg check
            for hw in self.hardware:
                assert hasattr(hw, "keys"), (
                    f"Each hardware yaml config must be a dictionary. But got {type(hw)}: {hw}"
                )
                missing_args, unknown_args, valid_args = dataclass_arg_check(
                    NodeHardwareConfig, hw
                )
                assert not missing_args, (
                    f"Missing fields '{missing_args}' detected in cluster node hardware yaml config. Only got: {hw.keys()}."
                )
                assert not unknown_args, (
                    f"Unknown fields '{unknown_args}' detected in cluster node hardware yaml config. Valid fields are: {valid_args}."
                )
            self.hardware = [NodeHardwareConfig(**hw) for hw in self.hardware]

        if self.env_vars is not None:
            self.env_vars = [dict(env) for env in self.env_vars]


@dataclass
class ClusterConfig:
    """Configuration for the entire cluster."""

    num_nodes: int
    """Total number of nodes in the cluster."""

    component_placement: list[dict[str, str]]
    """Placement of each component."""

    nodes: Optional[list[NodeGroupConfig]] = None
    """List of node group configurations in the cluster."""

    @staticmethod
    def from_dict_cfg(cfg_dict: DictConfig) -> "ClusterConfig":
        """Create a ClusterConfig instance from a dictionary configuration.

        Args:
            cfg_dict (DictConfig): The dictionary configuration.

        Returns:
            ClusterConfig: The created ClusterConfig instance.
        """
        return ClusterConfig(**cfg_dict)

    def get_node_label_by_rank(self, node_rank: int) -> Optional[str]:
        """Get the node group label for a given node rank.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            Optional[str]: The label of the node group if found, else None.
        """
        if self.nodes is None:
            return None
        for node_group in self.nodes:
            if node_rank in node_group.node_ranks:
                return node_group.label
        return None

    def __post_init__(self):
        """Post-initialization to convert nodes dicts to their respective dataclass instances."""
        if self.nodes is not None:
            # Arg check
            for node in self.nodes:
                assert hasattr(node, "keys"), (
                    f"Each node yaml config must be a dictionary. But got {type(node)}: {node}"
                )
                missing_args, unknown_args, valid_args = dataclass_arg_check(
                    NodeGroupConfig, node
                )
                assert not missing_args, (
                    f"Missing fields '{missing_args}' detected in cluster node yaml config. Only got: {node.keys()}."
                )
                assert not unknown_args, (
                    f"Unknown fields '{unknown_args}' detected in cluster node yaml config. Valid fields are: {valid_args}."
                )
            self.nodes = [NodeGroupConfig(**node) for node in self.nodes]
            for node_group in self.nodes:
                if isinstance(node_group.node_ranks, str):
                    node_group.node_ranks = parse_rank_config(
                        node_group.node_ranks,
                        list(range(self.num_nodes)),
                    )

    def __str__(self) -> str:
        """String representation of the NodeInfo."""
        node_dict = asdict(self)
        node_dict.pop("component_placement", None)
        return yaml.dump(node_dict, sort_keys=False)
