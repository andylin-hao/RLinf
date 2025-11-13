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
from dataclasses import asdict, dataclass, field

import ray
import ray.actor
import ray.util.scheduling_strategies
import yaml

from ..hardware import AcceleratorType, HardwareEnumerationPolicy, HardwareInfo


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""

    node_rank: int
    """Rank of the node in the cluster."""

    ray_id: str
    """Ray's unique identifier for the node."""

    node_ip: str
    """IP address of the node."""

    num_cpus: int
    """Number of CPUs available on the node."""

    default_envs: dict[str, str]
    """Default environment variables on the node, which are the env vars set before ray start."""

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
        node_dict.pop("default_envs", None)
        return yaml.dump(node_dict, sort_keys=False)


class NodeProbe:
    """Remote probe to get node hardware and environment information.

    This class launches one _RemoteNodeProbe actor on each node in the Ray cluster to collect hardware and environment information.
    """

    def __init__(self):
        """Launch the HardwareEnumerator on the specified nodes."""
        from .cluster import Cluster

        assert ray.is_initialized(), (
            "Ray must be initialized before creating HardwareEnumerator."
        )

        self._probes: list[ray.actor.ActorHandle] = []
        self._nodes: list[NodeInfo] = []

        for node_info in Cluster.get_alive_nodes():
            node_ray_id = node_info["NodeID"]
            probe = _RemoteNodeProbe.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ray_id, soft=False
                ),
                name=f"NodeProbe_{node_ray_id}",
            ).remote(node_info)
            self._probes.append(probe)

        handles = []
        for probe in self._probes:
            handles.append(probe.get_node_info.remote())
        self._nodes = ray.get(handles)

        self._sort_nodes()

    @property
    def nodes(self):
        """Get the list of node information.

        Returns:
            list[NodeInfo]: List of node information.
        """
        return self._nodes

    def _sort_nodes(self):
        """Sort the node info list by node rank if available, otherwise by accelerator type and IP."""
        from .cluster import Cluster, ClusterEnvVar

        # Sort the node info list by node rank if available
        if all(node_info.node_rank != -1 for node_info in self._nodes):
            # NODE_RANK should be larger than 0
            assert all(node_info.node_rank > 0 for node_info in self._nodes), (
                f"{Cluster.get_full_env_var_name(ClusterEnvVar.NODE_RANK)} should not be smaller than 0, but got: {[node_info.node_rank for node_info in self._nodes if node_info.node_rank <= 0]}"
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

    def __init__(self, node_info: dict[str, str]):
        from .cluster import Cluster, ClusterEnvVar

        try:
            node_rank = int(Cluster.get_sys_env_var(ClusterEnvVar.NODE_RANK, -1))
        except ValueError:
            raise ValueError(
                f"Invalid NODE_RANK value: {Cluster.get_sys_env_var(ClusterEnvVar.NODE_RANK)}. Must be an integer."
            )

        hardware_resources: list[HardwareInfo] = []
        for policy in HardwareEnumerationPolicy.policy_registry:
            hardware_resources.append(policy.enumerate())

        self._node_info = NodeInfo(
            node_rank=node_rank,
            ray_id=node_info["NodeID"],
            node_ip=node_info["NodeManagerAddress"],
            num_cpus=int(node_info["Resources"].get("CPU", 0)),
            default_envs=os.environ.copy(),
            hardware_resources=hardware_resources,
        )

    def get_node_info(self):
        """Get the node information.

        Returns:
            NodeInfo: The node information.
        """
        return self._node_info
