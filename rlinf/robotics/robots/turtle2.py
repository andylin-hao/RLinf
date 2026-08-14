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
from typing import Optional

from rlinf.scheduler.hardware import HardwareConfig, HardwareInfo, HardwareResource

from ..config import RobotAutoConfig
from ..discovery import RobotConfig, RobotDiscovery, RobotInfo, register_robot
from ..layout import ArmSpec, PartSpec, RobotSpec
from ..robot import Robot


class Turtle2Robot(Robot):
    """Composable Turtle2 robot."""

    ROBOT_TYPE = "Turtle2"


class Turtle2Discovery(RobotDiscovery):
    """Discover configured Turtle2 robots."""

    HW_TYPE = Turtle2Robot.ROBOT_TYPE

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware resources. None if no hardware is found.
        """
        assert configs is not None, "Robot hardware requires explicit configurations"
        robot_configs: list["Turtle2Config"] = []
        for config in configs:
            if isinstance(config, Turtle2Config) and config.node_rank == node_rank:
                robot_configs.append(config)

        if robot_configs:
            # Auto-detect any unset fields from environment variables.
            RobotAutoConfig.resolve(robot_configs)

            turtle2_infos: list[HardwareInfo] = []
            for config in robot_configs:
                turtle2_infos.append(
                    RobotInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

            return HardwareResource(type=cls.HW_TYPE, infos=turtle2_infos)
        return None


@dataclass
class Turtle2Config(RobotConfig):
    """Configuration for a robotic system."""

    # empty config

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in Turtle2 config must be an integer. But got {type(self.node_rank)}."
        )

    def to_spec(self) -> RobotSpec:
        """Return the Turtle2 dual-arm physical layout."""
        return RobotSpec(
            robot_type=Turtle2Robot.ROBOT_TYPE,
            node_rank=self.node_rank,
            arms=(
                ArmSpec(
                    name="left",
                    driver="turtle2",
                    node_rank=self.node_rank,
                    connection={"arm_id": 0},
                ),
                ArmSpec(
                    name="right",
                    driver="turtle2",
                    node_rank=self.node_rank,
                    connection={"arm_id": 1},
                ),
            ),
            parts=(
                PartSpec(
                    name="base",
                    kind="mobile_base",
                    driver="turtle2",
                    node_rank=self.node_rank,
                ),
                PartSpec(
                    name="head",
                    kind="head",
                    driver="turtle2",
                    node_rank=self.node_rank,
                ),
                PartSpec(
                    name="lift",
                    kind="lift",
                    driver="turtle2",
                    node_rank=self.node_rank,
                ),
            ),
        )


register_robot(Turtle2Config, Turtle2Robot)(Turtle2Discovery)
