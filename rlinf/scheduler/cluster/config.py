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

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from ..hardware import NodeHardwareConfig


class DataclassProtocol(Protocol):
    """Protocol for dataclasses to enable type checking."""

    __dataclass_fields__: dict
    __dataclass_params__: dict
    __post_init__: Optional[Callable]


@dataclass
class NodeGroupConfig:
    """Configuration for a group of nodes in the cluster with the same label.

    A node group is used to represent multiple nodes with identical hardware configurations.
    """

    label: str
    node_ranks: str
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


@dataclass
class ClusterConfig:
    """Configuration for the entire cluster."""

    num_nodes: int
    """Total number of nodes in the cluster."""

    component_placement: list[dict]
    """Placement of each component."""

    nodes: list[NodeGroupConfig] = field(default_factory=lambda: [])
    """List of node group configurations in the cluster."""

    def __post_init__(self):
        """Post-initialization to convert nodes dicts to their respective dataclass instances."""
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


def dataclass_arg_check(dataclass: DataclassProtocol, kwargs: dict):
    """Check if the kwargs contain only valid fields for the given dataclass.

    Args:
        dataclass (DataclassProtocol): The dataclass to check against.
        kwargs (dict): The keyword arguments to check.
    """
    args = set(kwargs.keys())
    valid_args = set(dataclass.__dataclass_fields__.keys())

    missing_args = valid_args - args
    unknown_args = args - valid_args

    missing_required_args = []
    for missing_arg in missing_args:
        field_info = dataclass.__dataclass_fields__[missing_arg]
        if (
            field_info.default is field_info.default_factory
            and field_info.default_factory is field_info.default_factory
        ):
            missing_required_args.append(missing_arg)

    return missing_required_args, unknown_args, valid_args
