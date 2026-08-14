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
from ..parts.base import Arm
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




def build_turtle2_robot(
    *,
    frequency: int,
    camera_ids: list[int],
    env_idx: int,
    node_rank: int,
    worker_rank: int,
) -> Turtle2Robot:
    """Place the coupled Turtle2 controller and compose both arms from it.

    One connection backs both arms, both grippers, and the wrist cameras; the
    driver decomposes itself into those parts.
    """
    from ..drivers.turtle2 import Turtle2Driver

    handle = Turtle2Driver.spawn(
        frequency,
        tuple(camera_ids),
        node_rank=node_rank,
        name=f"Turtle2Driver-{worker_rank}-{env_idx}",
    )
    cameras = {
        name: part for name, part in handle.parts.items() if name.startswith("wrist_")
    }
    return Turtle2Robot.dual_arm(
        Arm(handle.part("left"), handle.part("left_end_effector")),
        Arm(handle.part("right"), handle.part("right_end_effector")),
        cameras=cameras,
        drivers={"controller": handle},
    )


register_robot(Turtle2Config, Turtle2Robot, build=build_turtle2_robot)(
    Turtle2Discovery
)
