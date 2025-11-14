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
from typing import Any, ClassVar


@dataclass
class NodeHardwareConfig:
    """This represents hardware configs for a set of nodes."""

    type: str
    """Hardware type"""

    configs: list[dict]
    """List of hardware configurations."""

    _hardware_config_registry: ClassVar[dict[str, Any]] = {}

    @classmethod
    def register_hardware_config(cls, type: str):
        """Register a hardware config into the global registry.

        Args:
            type (str): The type of the hardware. This type is not case sensitive.
        """

        def hardware_config_decorator(hardware_config):
            cls._hardware_config_registry[type.lower()] = hardware_config
            return hardware_config

        return hardware_config_decorator

    def __post_init__(self):
        """Post-initialization to convert hardware_configs dicts to their respective dataclass instances."""
        self.type = self.type.lower()
        hardware_config_class = NodeHardwareConfig._hardware_config_registry.get(
            self.type
        )
        if hardware_config_class is None:
            raise ValueError(
                f"Unsupported hardware type: {self.type}. Currently supported types only include: {list(self._hardware_config_registry.keys())}"
            )

        from ..cluster import dataclass_arg_check

        # Arg check
        for config in self.configs:
            assert hasattr(config, "keys"), (
                f"Each hardware config must be a dictionary. But got {type(config)}: {config}"
            )
            missing_args, unknown_args, valid_args = dataclass_arg_check(
                hardware_config_class, config
            )
            assert not missing_args, (
                f"Missing fields '{missing_args}' detected in cluster node hardware configs yaml config. Only got: {config.keys()}."
            )
            assert not unknown_args, (
                f"Unknown fields '{unknown_args}' detected in cluster node hardware configs yaml config. Valid fields are: {valid_args}."
            )
        self.configs = [hardware_config_class(**config) for config in self.configs]


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

    policy_registry: list[type["HardwareEnumerationPolicy"]] = []

    @classmethod
    def register_policy(cls, policy: type["HardwareEnumerationPolicy"]):
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
