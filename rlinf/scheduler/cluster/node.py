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

import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

import ray
import ray.actor
import ray.util.scheduling_strategies
import yaml

from ..hardware import AcceleratorType, HardwareEnumerationPolicy, HardwareInfo
from .config import ClusterConfig


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""

    node_labels: list[str]
    """Labels of the node, corresponding to the node group label in the cluster configuration."""

    node_rank: int
    """Rank of the node in the cluster."""

    ray_id: str
    """Ray's unique identifier for the node."""

    node_ip: str
    """IP address of the node."""

    num_cpus: int
    """Number of CPUs available on the node."""

    python_interpreter_path: str
    """Path to the Python interpreter to be used on the node."""

    default_env_vars: dict[str, str]
    """Default environment variables on the node, which are the env vars set before ray start."""

    env_vars: dict[str, str]
    """Environment variables set on the node by the scheduler."""

    hardware_resources: list[HardwareInfo] = field(default_factory=list)
    """List of hardware resources available on the node."""

    @property
    def accelerator_type(self) -> AcceleratorType:
        """Type of accelerator available on the node."""
        for resource in self.hardware_resources:
            if resource.type in AcceleratorType._value2member_map_:
                return AcceleratorType(resource.type)
        return AcceleratorType.NO_ACCEL

    @property
    def num_accelerators(self) -> int:
        """Number of accelerators available on the node."""
        for resource in self.hardware_resources:
            if resource.type in AcceleratorType._value2member_map_:
                return resource.count
        return 0

    @property
    def accelerator_model(self) -> str:
        """Model of the accelerator available on the node."""
        for resource in self.hardware_resources:
            if resource.type in AcceleratorType._value2member_map_:
                return resource.model
        return "N/A"

    def __str__(self) -> str:
        """String representation of the NodeInfo."""
        node_dict = asdict(self)
        node_dict.pop("default_env_vars", None)
        node_dict.pop("env_vars", None)
        return yaml.dump(node_dict, sort_keys=False)


class NodeProbe:
    """Remote probe to get node hardware and environment information.

    This class launches one _RemoteNodeProbe actor on each node in the Ray cluster to collect hardware and environment information.
    """

    def __init__(self, cluster_cfg: Optional[ClusterConfig]):
        """Launch the HardwareEnumerator on the specified nodes."""
        from .cluster import Cluster

        assert ray.is_initialized(), (
            "Ray must be initialized before creating HardwareEnumerator."
        )

        self._probes: list[ray.actor.ActorHandle] = []
        self._nodes: list[NodeInfo] = []
        self._cluster_cfg = cluster_cfg

        node_infos = Cluster.get_alive_nodes()
        num_nodes = len(node_infos)
        for node_info in node_infos:
            node_ray_id = node_info["NodeID"]
            probe = _RemoteNodeProbe.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ray_id, soft=False
                ),
                name=f"NodeProbe_{node_ray_id}",
            ).remote(node_info, num_nodes, cluster_cfg, sys.executable)
            self._probes.append(probe)

        handles = []
        for probe in self._probes:
            handles.append(probe.get_node_info.remote())
        self._nodes = ray.get(handles)

        self._sort_nodes()
        self._configure_node_envs()

    @property
    def nodes(self):
        """Get the list of node information.

        Returns:
            list[NodeInfo]: List of node information.
        """
        return self._nodes

    @property
    def head_node(self):
        """Get the head node information, which is the node that initializes the cluster.

        Returns:
            NodeInfo: Head node information.
        """
        current_id = ray.get_runtime_context().get_node_id()
        head_node = next(
            (node for node in self._nodes if node.ray_id == current_id), None
        )
        assert head_node is not None, (
            f"Head node with Ray ID {current_id} not found in the cluster nodes: {[node.ray_id for node in self._nodes]}"
        )
        return head_node

    def _configure_node_envs(self):
        """Configure each node's environments based on the cluster configuration.

        The environment variables follow the following precedence, with the later ones overriding the previous ones if set:
        1. Default environment variables on the node (set before ray start).
        2. Environment variables set between ray start and RLinf initialization on the head node (usually via bash scripts). These env vars are likely set by users intended to configure all nodes in the cluster.
        3. The env_vars field in the ClusterConfig, which are set in yaml config files to configure each node in the cluster.
        """
        # Overwrite the the head node's python interpreter path as the current interpreter
        self.head_node.python_interpreter_path = sys.executable

        # First find env vars set between ray start and RLinf initialization on the head node
        head_node_default_env_vars = self.head_node.default_env_vars
        current_env_vars = os.environ
        modified_env_vars = {}
        for key, value in current_env_vars.items():
            if (
                key not in head_node_default_env_vars
                or head_node_default_env_vars[key] != value
            ):
                modified_env_vars[key] = value

        for node in self._nodes:
            # Start with default env vars on the node
            node.env_vars = node.default_env_vars.copy()

            # Update with modified env vars on the head node
            node.env_vars.update(modified_env_vars)

            # Finally, update with env vars from cluster config if available
            if self._cluster_cfg is not None and self._cluster_cfg.nodes is not None:
                for node_group in self._cluster_cfg.nodes:
                    if node.node_rank in node_group.node_ranks:
                        if node_group.env_vars is not None:
                            for env_var_dict in node_group.env_vars:
                                node.env_vars.update(env_var_dict)

    def _sort_nodes(self):
        """Sort the node info list by node rank if available, otherwise by accelerator type and IP."""
        from .cluster import Cluster, ClusterEnvVar

        # Sort the node info list by node rank if available
        if all(node_info.node_rank != -1 for node_info in self._nodes):
            # NODE_RANK should be larger than 0
            assert all(node_info.node_rank >= 0 for node_info in self._nodes), (
                f"{Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)} should not be smaller than 0, but got: {[node_info.node_rank for node_info in self._nodes if node_info.node_rank < 0]}"
            )

            # NODE_RANK should be smaller than the number of nodes
            assert all(
                node_info.node_rank < len(self._nodes) for node_info in self._nodes
            ), (
                f"{Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)} should be smaller than the number of nodes {len(self._nodes)}, but got: {[node_info.node_rank for node_info in self._nodes if node_info.node_rank >= len(self._nodes)]}"
            )

            self._nodes.sort(key=lambda x: x.node_rank)

        else:
            # Either all nodes set NODE_RANK, or none of them should have.
            assert all(node_info.node_rank == -1 for node_info in self._nodes), (
                f"Either all nodes set {Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)}, or none of them should have. But got: {[node_info.node_rank for node_info in self._nodes if node_info.node_rank != -1]}"
            )

            # NODE_RANK not set, sort first by accelerator type, then by IP
            nodes_group_by_accel: dict[str, list[NodeInfo]] = {}
            for node in self._nodes:
                accel_name = f"{node.accelerator_type.value}_{node.accelerator_model}"
                nodes_group_by_accel.setdefault(accel_name, [])
                nodes_group_by_accel[accel_name].append(node)
            for accel_name in nodes_group_by_accel.keys():
                nodes_group_by_accel[accel_name].sort(key=lambda x: x.node_ip)
            self._nodes = [
                node for nodes in nodes_group_by_accel.values() for node in nodes
            ]

            node_rank = 0
            for node in self._nodes:
                node.node_rank = node_rank
                node_rank += 1


@ray.remote
class _RemoteNodeProbe:
    """Remote Ray actor that collect information on a node."""

    def __init__(
        self,
        node_info: dict[str, str],
        num_nodes: int,
        cluster_cfg: Optional[ClusterConfig],
        head_python_interpreter: str,
    ):
        from .cluster import Cluster, ClusterEnvVar

        # Node rank
        try:
            node_rank = int(Cluster.get_sys_env_var(ClusterEnvVar.NODE_RANK, -1))
        except ValueError:
            raise ValueError(
                f"Invalid NODE_RANK value: {Cluster.get_sys_env_var(ClusterEnvVar.NODE_RANK)}. Must be an integer."
            )
        if num_nodes == 1:
            node_rank = 0

        # Node label
        node_labels = []
        if cluster_cfg is not None and cluster_cfg.nodes is not None:
            assert node_rank != -1, (
                f"{Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)} must be set when there are more than one nodes are connected in Ray and cluster's nodes configuration is provided."
            )
            node_labels = cluster_cfg.get_node_labels_by_rank(node_rank)

        # Node hardware resources
        node_hw_configs = []
        if cluster_cfg is not None:
            node_hw_configs = cluster_cfg.get_node_hw_configs_by_rank(node_rank)
        hardware_resources: list[HardwareInfo] = []
        for policy in HardwareEnumerationPolicy.policy_registry:
            hw_info = policy.enumerate(node_rank, node_hw_configs)
            if hw_info is not None:
                hardware_resources.append(hw_info)

        # Python interpreter path
        if sys.executable != head_python_interpreter:
            warnings.warn(
                f"Python interpreter used to launch Ray on node with IP {node_info['NodeManagerAddress']} is different from that on the head node {head_python_interpreter}. Keep using the current interpreter {sys.executable} on this node."
            )

        self._node_info = NodeInfo(
            node_labels=node_labels,
            node_rank=node_rank,
            ray_id=node_info["NodeID"],
            node_ip=node_info["NodeManagerAddress"],
            num_cpus=int(node_info["Resources"].get("CPU", 0)),
            python_interpreter_path=sys.executable,
            default_env_vars=os.environ.copy(),
            env_vars=os.environ.copy(),
            hardware_resources=hardware_resources,
        )

    def get_node_info(self):
        """Get the node information.

        Returns:
            NodeInfo: The node information.
        """
        return self._node_info
