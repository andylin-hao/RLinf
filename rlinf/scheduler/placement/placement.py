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

import logging
from dataclasses import dataclass
from typing import overload

from omegaconf import DictConfig

from ..cluster import Cluster
from ..hardware import AcceleratorType


@dataclass
class Placement:
    """Class representing the placement of a worker on a specific GPU."""

    rank: int
    """Global rank of the worker in the cluster."""

    node_id: int
    """Node ID where the worker is placed."""

    node_rank: int
    """Rank of the node in the cluster."""

    local_accelerator_id: int
    """Local GPU ID on the node."""

    accelerator_type: AcceleratorType
    """Type of accelerators on the node."""

    local_rank: int
    """Local rank of the worker on the node."""

    local_world_size: int
    """Local world size (number of workers) on the node."""

    visible_accelerators: list[str]
    """List of CUDA visible devices for the worker."""

    isolate_accelerator: bool
    """Flag to indicate if the local rank should be set to zero. This is useful for workers that require multiple GPUs."""


class PlacementStrategy:
    """Base class for placement strategies."""

    def __init__(self):
        """Initialize the PlacementStrategy."""
        self._placement_strategy = None
        self._logger = logging.getLogger(name=self.__class__.__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(levelname)s %(asctime)s %(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    @overload
    def get_placement(
        self,
        cluster: Cluster,
        isolate_accelerator: bool = True,
    ) -> list[Placement]:
        return None


class ComponentPlacement:
    """Base component placement for parsing cluster.component_placement config.

    The component placement config is defined as:
    group_name1,group_name2,...: label:resource_ranks:process_ranks.

    - label is the node group label defined in cluster.nodes.label, which is optional. If not specified, all nodes in the cluster are used. A `node` label is reserved by the scheduler for allocating on node ranks only (no accelerators or other hardware).

    - resource_ranks are the ranks of the resources (e.g., GPUs, robots, or nodes) to use for the component(s). resource ranks are by default the local accelerator ranks within the label node group if no hardware is specified in the config. Alternatively, "all" can be used to specify all resources. If hardware is specified in the node group config, the resource ranks are the hardware ranks within the label node group, e.g., for nodes with robotic systems.

    - process_ranks are the ranks of the processes of the component(s), which will be evenly assigned to the specified resource ranks.

    An example config is:
    actor,inference: a800:0-4:0-8, which means both the actor and inference groups' process 0-8 evenly occupy accelerator 0 to 4 of node group with label 'a800'.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        """Parsing component placement configuration.

        Args:
            config (DictConfig): The configuration dictionary for the component placement.
            cluster (Cluster): The cluster to use for placement.
        """
        self._config = config
        assert hasattr(config, "cluster"), (
            f"Cluster config must be provided for component placement. But got: {config}"
        )
        assert hasattr(config.cluster, "component_placement"), (
            f"component_placement must be provided in cluster config for component placement. But got: {config.cluster}"
        )
        self._placement_config: DictConfig = config.cluster.component_placement

        self._components: list[str] = []
        self._component_resource_rank_map: dict[str, list[int]] = {}

        for components in self._placement_config.keys():
            components_gpus: str = self._placement_config[components]
            components = components.split(",")
            components = [c.strip() for c in components]
            components_gpus = self._parse_gpu_ids(components_gpus, components)

            for component in components:
                self._components.append(component)
                self._component_gpu_map[component] = components_gpus

            self._placements: dict[str, PlacementStrategy] = {}

    def _parse_gpu_ids(
        self, components_gpus: str, component_names: list[str]
    ) -> list[int]:
        """Parse a string of GPU IDs into a list of integers.

        Args:
            components_gpus (str): A string representing GPU IDs. The string can either be "all", representing all GPUs, or a comma-separated list of GPU IDs and ranges (e.g., "0,1,2-4").
            component_names (List[str]): The names of the components for error reporting.

        Returns:
            List[int]: A list of GPU IDs as integers.
        """
        gpu_ids: list[int] = []
        if components_gpus == "all":
            gpu_ids = list(range(0, self._cluster_num_gpus))
        else:
            # If the GPU placement is a single number
            # Omegaconf will parse it as an integer instead of a string
            components_gpus = str(components_gpus)
            # First split by comma
            gpu_id_ranges = components_gpus.split(",")
            for gpu_id_range in gpu_id_ranges:
                gpu_id_range = gpu_id_range.strip()
                if gpu_id_range == "":
                    continue
                # Then split by hyphen to get the start and end of the range
                gpu_id_range = gpu_id_range.split("-")
                try:
                    if len(gpu_id_range) == 1:
                        start_gpu = int(gpu_id_range[0])
                        end_gpu = start_gpu
                    elif len(gpu_id_range) == 2:
                        start_gpu = int(gpu_id_range[0])
                        end_gpu = int(gpu_id_range[1])
                    else:
                        raise ValueError
                except (ValueError, IndexError):
                    raise ValueError(
                        f'Invalid GPU placement format for components {component_names}: {components_gpus}, expected format: "a,b,c-d" or "all"'
                    )
                assert end_gpu >= start_gpu, (
                    f"Start GPU ID {start_gpu} must be less than or equal to end GPU ID {end_gpu}."
                )
                assert start_gpu < self._cluster_num_gpus, (
                    f"Start GPU ID {start_gpu} must be less than total number of GPUs {self._cluster_num_gpus}."
                )
                assert end_gpu < self._cluster_num_gpus, (
                    f"End GPU ID {end_gpu} must be less than total number of GPUs {self._cluster_num_gpus}."
                )
                gpu_ids.extend(list(range(start_gpu, end_gpu + 1)))
        return gpu_ids

    @property
    def placement_mode(self):
        """Get the placement mode for the component.

        Returns:
            PlacementMode: The placement mode for the component.
        """
        return self._placement_mode

    def get_world_size(self, component_name: str):
        """Get the world size for a specific component.

        Args:
            component_name (str): The name of the component.

        Returns:
            int: The world size for the specified component.
        """
        assert component_name in self._component_gpu_map, (
            f"Unknown component name: {component_name}"
        )
        return len(self._component_gpu_map[component_name])

    @overload
    def _generate_placements(self):
        raise NotImplementedError

    def get_strategy(self, component_name: str):
        """Get the placement strategy for a component based on the configuration.

        Args:
            component_name (str): The name of the component to retrieve the placement strategy for.

        Returns:
            PackedPlacementStrategy: The placement strategy for the specified component.
        """
        if len(self._placements.keys()) == 0:
            self._generate_placements()
        assert component_name in self._placements, (
            f"Component {component_name} does not exist in {type(self)} with placement mode {self._placement_mode}"
        )
        return self._placements[component_name]
