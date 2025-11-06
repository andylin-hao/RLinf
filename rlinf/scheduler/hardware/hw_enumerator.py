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

from dataclasses import dataclass

import ray
import ray.actor
import ray.util.scheduling_strategies


@dataclass
class HardwareInfo:
    """Dataclass representing a hardware resource information."""

    type: str
    """Type of the hardware resource (e.g., Accelerator, Robot)."""

    model: str
    """Model of the hardware resource (e.g., 4090, A100, H100, Franka)."""

    count: int
    """Resource count of the hardware on a node."""


class HardwareEnumerationPolicy:
    """Enumeration policy for a type of hardware resource.

    This is the base class for different hardware to implement their enumeration policies.
    """

    policy_registry: list["HardwareEnumerationPolicy"] = []

    @classmethod
    def register_policy(cls, policy: "HardwareEnumerationPolicy"):
        """Register a new enumeration policy.

        This is to be used as a decorator for subclasses of EnumerationPolicy.

        Args:
            policy (HardwareEnumerationPolicy): The enumeration policy to register.
        """
        cls.policy_registry.append(policy)

    @classmethod
    def enumerate(cls) -> HardwareInfo:
        """Enumerate the hardware resources on a node.

        Returns:
            HardwareInfo: An object representing the hardware resources.
        """
        raise NotImplementedError


class HardwareEnumerator:
    """Hardware enumerator that enumerates hardware resources on a node.

    This class launches one _NodeHardwareEnumerator actor on each node in the Ray cluster to enumerate
    hardware resources available on that node.
    """

    def __init__(self):
        """Launch the HardwareEnumerator on the specified nodes."""
        assert ray.is_initialized(), (
            "Ray must be initialized before creating HardwareEnumerator."
        )

        self._enumerators: dict[str, ray.actor.ActorHandle] = {}
        self._hardware_resources: dict[str, dict] = {}

        for node_info in ray.nodes():
            node_ray_id = node_info["NodeID"]
            enumerator = _NodeHardwareEnumerator.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ray_id, soft=False
                ),
                name=f"HardwareEnumerator_{node_ray_id}",
            ).remote(HardwareEnumerationPolicy.policy_registry)
            self._enumerators[node_ray_id] = enumerator

    def enumerate(self) -> dict[str, list[HardwareInfo]]:
        """Enumerate hardware resources on all nodes in the Ray cluster.

        Returns:
            dict[str, list[HardwareInfo]]: A dictionary mapping node IDs to lists of HardwareInfo objects
            representing the hardware resources on each node.
        """
        handles = []
        for node_id, enumerator in self._enumerators.items():
            handles.append(enumerator.enumerate_hardware.remote())
        hardware_info = ray.get(handles)
        for node_id, info in zip(self._enumerators.keys(), hardware_info):
            self._hardware_resources[node_id] = info
        return self._hardware_resources


@ray.remote
class _NodeHardwareEnumerator:
    """Remote actor that enumerates hardware resources on a node."""

    def __init__(self, policies: list[HardwareEnumerationPolicy]):
        """Enumerate hardware resources on the node.

        Args:
            policies (list[EnumerationPolicy]): List of enumeration policies to use.

        Returns:
            dict: A dictionary containing hardware resource information.
        """
        self._policies = policies

    def enumerate_hardware(self) -> list[HardwareInfo]:
        """Enumerate hardware resources on the node.

        Returns:
            list[HardwareInfo]: A list of HardwareInfo objects representing the hardware resources.
        """
        hardware_info: list[HardwareInfo] = []
        for policy in self._policies:
            hardware_info.append(policy.enumerate())
        return hardware_info
