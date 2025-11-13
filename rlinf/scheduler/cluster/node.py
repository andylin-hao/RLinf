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

from ..hardware import AcceleratorType, HardwareInfo


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""

    node_rank: str
    """Rank of the node in the cluster."""

    ray_id: str
    """Ray's unique identifier for the node."""

    node_ip: str
    """IP address of the node."""

    num_cpus: int
    """Number of CPUs available on the node."""

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
