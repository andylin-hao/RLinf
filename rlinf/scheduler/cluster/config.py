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

from dataclasses import asdict, dataclass
from typing import Any, Optional

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
    python_interpreter_path: Optional[str] = None
    env_vars: Optional[list[dict[str, str]]] = None
    hardware: Optional[list[NodeHardwareConfig]] = None

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

        self.label = str(self.label)

        # Convert env_vars list of dicts to ensure each dict has only one key-value pair
        if self.env_vars is not None:
            env_vars = self.env_vars
            self.env_vars = []
            for env_var in env_vars:
                assert hasattr(env_var, "keys"), (
                    f"Each node env_var must be a dict in config. But got {type(env_var)}: {env_var}"
                )
                assert len(env_var) == 1, (
                    f"Each node env_var dict must contain exactly one key-value pair. But got: {env_var}"
                )
                env_var_key = str(list(env_var.keys())[0])
                env_var_value = str(list(env_var.values())[0])
                self.env_vars.append({env_var_key: env_var_value})


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

    def get_node_labels_by_rank(self, node_rank: int) -> list[str]:
        """Get the node group labels for a given node rank.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            list[str]: The labels of the node group. Empty list if no matching node group is found.
        """
        if self.nodes is None:
            return []
        labels = []
        for node_group in self.nodes:
            if node_rank in node_group.node_ranks:
                labels.append(node_group.label)
        return labels

    def get_node_python_interpreter_path_by_rank(self, node_rank: int) -> Optional[str]:
        """Get the python interpreter path for a given node rank.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            Optional[str]: The python interpreter path of the node. None if no matching node group is found.
        """
        if self.nodes is None:
            return None
        paths = []
        for node_group in self.nodes:
            if (
                node_rank in node_group.node_ranks
                and node_group.python_interpreter_path is not None
            ):
                paths.append(node_group.python_interpreter_path)
        if len(paths) == 0:
            return None
        if len(paths) > 1:
            raise ValueError(
                f"Multiple python interpreter paths found for node rank {node_rank}: {paths}. Expected only one."
            )
        return paths[0]

    def get_node_hw_configs_by_rank(self, node_rank: int) -> list[Any]:
        """Get the hardware configurations for a given node rank.

        Args:
            node_rank (int): The rank of the node.

        Returns:
            list[Any]: The hardware configurations of the node. Empty list if no matching node group is found.
        """
        node_hw_configs: list[Any] = []
        if self.nodes is not None:
            for node_group in self.nodes:
                if node_rank in node_group.node_ranks:
                    if node_group.hardware is not None:
                        for hw_cfg in node_group.hardware:
                            node_hw_configs.extend(hw_cfg.configs)
        return node_hw_configs

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

            # Convert node_ranks from str to list[int] if needed
            for node_group in self.nodes:
                node_group.node_ranks = parse_rank_config(
                    node_group.node_ranks,
                    list(range(self.num_nodes)),
                )

            # Validate hardware node_ranks
            for node_group in self.nodes:
                if node_group.hardware is not None:
                    for hw_cfg in node_group.hardware:
                        for cfg in hw_cfg.configs:
                            assert cfg.node_rank in node_group.node_ranks, (
                                f"node_rank {cfg.node_rank} in hardware config must be within node_ranks {node_group.node_ranks} in node group '{node_group.label}'."
                            )

        assert type(self.num_nodes) is int and self.num_nodes > 0, (
            f"'num_nodes' must be a positive integer. But got {self.num_nodes} of type {type(self.num_nodes)}."
        )

    def __str__(self) -> str:
        """String representation of the NodeInfo."""
        node_dict = asdict(self)
        node_dict.pop("component_placement", None)
        return yaml.dump(node_dict, sort_keys=False)
