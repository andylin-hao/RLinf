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

from rlinf.robotics.drivers.gim_arm import GimArmDriver
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker


class GimArmController(Worker):
    """Scheduler host for one pure :class:`GimArmDriver`."""

    @staticmethod
    def launch_controller(
        can_interface: str,
        arm_variant: str,
        enable_gripper: bool,
        gripper_type: str,
        control_mode: str = "momentum_observer",
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
    ):
        """Launch one GimArm controller worker on the selected node."""
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return GimArmController.create_group(
            can_interface, arm_variant, enable_gripper, gripper_type, control_mode
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"GimArmController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        can_interface: str,
        arm_variant: str,
        enable_gripper: bool,
        gripper_type: str,
        control_mode: str = "momentum_observer",
    ) -> None:
        super().__init__()
        self._driver = GimArmDriver(
            can_interface,
            arm_variant,
            enable_gripper,
            gripper_type,
            control_mode,
        )
        self._driver.connect()

    def is_robot_up(self) -> bool:
        """Return whether the hosted driver is responsive."""
        return self._driver.is_robot_up()

    def get_state(self):
        """Read the hosted driver's state."""
        return self._driver.get_state()

    def move_joints(self, q_target) -> None:
        """Send an absolute joint target."""
        self._driver.move_joints(q_target)

    def reset_joint(self, reset_qpos, duration: float = 3.0) -> None:
        """Run a smooth joint-space reset."""
        self._driver.reset_joint(reset_qpos, duration)

    def open_gripper(self) -> None:
        """Open the hosted gripper."""
        self._driver.open_gripper()

    def close_gripper(self) -> None:
        """Close the hosted gripper."""
        self._driver.close_gripper()

    def clear_errors(self) -> None:
        """Delegate fault recovery to the hosted driver."""
        self._driver.clear_errors()

    def stop(self) -> None:
        """Disconnect the hosted driver."""
        self._driver.disconnect()

    def cleanup(self) -> None:
        """Disconnect the hosted driver during worker teardown."""
        self._driver.disconnect()
