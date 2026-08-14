# Copyright 2026 The RLinf Authors.
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

from typing import Optional

from rlinf.robotics.drivers.franky import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    JOINT_VEL_LIMITS,
    FrankyDriver,
)
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker


class FrankyController(Worker):
    """Scheduler host for one pure :class:`FrankyDriver`."""

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        gripper_type: str = "robotiq",
        gripper_connection: Optional[str] = None,
    ):
        """Launch one Franky controller worker on the selected node."""
        return FrankyController.create_group(
            robot_ip, gripper_type, gripper_connection
        ).launch(
            cluster=Cluster(),
            placement_strategy=NodePlacementStrategy(node_ranks=[node_rank]),
            name=f"FrankyController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        gripper_type: str = "robotiq",
        gripper_connection: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._driver = FrankyDriver(robot_ip, gripper_type, gripper_connection)
        self._driver.connect()

    def is_robot_up(self) -> bool:
        """Return whether the hosted robot and gripper are responsive."""
        return self._driver.is_robot_up()

    def get_state(self):
        """Read the hosted driver's state."""
        return self._driver.get_state()

    def clear_errors(self) -> None:
        """Recover the hosted robot from errors."""
        self._driver.clear_errors()

    def move_joints(self, joint_positions) -> None:
        """Send an absolute joint target."""
        self._driver.move_joints(joint_positions)

    def move_tcp_pose(self, pose) -> None:
        """Send an absolute Cartesian pose target."""
        self._driver.move_tcp_pose(pose)

    def reset_joint(self, reset_pos) -> None:
        """Run a blocking joint-space reset motion."""
        self._driver.reset_joint(reset_pos)

    def open_gripper(self) -> None:
        """Open the hosted gripper."""
        self._driver.open_gripper()

    def close_gripper(self) -> None:
        """Close the hosted gripper."""
        self._driver.close_gripper()

    def cleanup(self) -> None:
        """Disconnect the pure driver during worker teardown."""
        self._driver.disconnect()


__all__ = [
    "JOINT_LIMITS_LOWER",
    "JOINT_LIMITS_UPPER",
    "JOINT_VEL_LIMITS",
    "FrankyController",
]
