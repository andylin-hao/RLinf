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

from rlinf.robotics.drivers.turtle2 import Turtle2Driver
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker


class Turtle2SmoothController(Worker):
    """Scheduler host for one pure :class:`Turtle2Driver`."""

    @staticmethod
    def launch_controller(
        freq: int = 50,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
    ):
        """Launch one Turtle2 controller worker on the selected node."""
        return Turtle2SmoothController.create_group(freq).launch(
            cluster=Cluster(),
            placement_strategy=NodePlacementStrategy(node_ranks=[node_rank]),
            name=f"Turtle2SmoothController-{worker_rank}-{env_idx}",
        )

    def __init__(self, freq: int = 50) -> None:
        super().__init__()
        self._driver = Turtle2Driver(freq)
        self._driver.connect()

    def get_state(self):
        return self._driver.get_state()

    def move_arm(self, left_arm_target, right_arm_target):
        return self._driver.move_arm(left_arm_target, right_arm_target)

    def move_left_arm(self, target):
        return self._driver.move_left_arm(target)

    def move_right_arm(self, target):
        return self._driver.move_right_arm(target)

    def move_left_gripper(self, target):
        return self._driver.move_left_gripper(target)

    def move_right_gripper(self, target):
        return self._driver.move_right_gripper(target)

    def reset_arms(self):
        return self._driver.reset_arms()

    def check_cams(self, timeout: float = 0.5):
        return self._driver.check_cams(timeout)

    def get_cams(self, ids):
        return self._driver.get_cams(ids)

    def get_camera(self, camera_id):
        return self._driver.get_camera(camera_id)

    def cleanup(self) -> None:
        self._driver.disconnect()
