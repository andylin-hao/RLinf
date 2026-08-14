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

from rlinf.robotics.drivers.franka_ros import FrankaROSDriver
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker


class FrankaController(Worker):
    """Scheduler host for one pure :class:`FrankaROSDriver`."""

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        ros_pkg: str = "serl_franka_controllers",
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ):
        """Launch one Franka ROS worker on the selected node."""
        return FrankaController.create_group(
            robot_ip,
            ros_pkg,
            end_effector_type,
            end_effector_config or {},
            gripper_type,
            gripper_connection,
        ).launch(
            cluster=Cluster(),
            placement_strategy=NodePlacementStrategy(node_ranks=[node_rank]),
            name=f"FrankaController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        ros_pkg: str = "serl_franka_controllers",
        end_effector_type: str = "franka_gripper",
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ) -> None:
        super().__init__()
        resolved_robot_ip = robot_ip or self._resolve_robot_ip_from_node()
        if not resolved_robot_ip:
            raise ValueError(
                "Franka 'robot_ip' is not set and could not be resolved from "
                f"node rank {self._cluster_node_rank}'s hardware infos."
            )
        self._driver = FrankaROSDriver(
            resolved_robot_ip,
            ros_pkg,
            end_effector_type,
            end_effector_config,
            gripper_type,
            gripper_connection,
        )
        self._driver.connect()

    def _resolve_robot_ip_from_node(self) -> Optional[str]:
        try:
            node_info = Cluster().get_node_info(self._cluster_node_rank)
        except Exception as exc:
            self.log_warning(f"Could not resolve Franka robot_ip: {exc}")
            return None
        for resource in node_info.hardware_resources:
            for info in resource.infos:
                robot_ip = getattr(getattr(info, "config", None), "robot_ip", None)
                if robot_ip:
                    return robot_ip
        return None

    def reconfigure_compliance_params(self, params):
        return self._driver.reconfigure_compliance_params(params)

    def is_robot_up(self):
        return self._driver.is_robot_up()

    def get_state(self):
        return self._driver.get_state()

    def clear_errors(self):
        return self._driver.clear_errors()

    def reset_joint(self, reset_pos):
        return self._driver.reset_joint(reset_pos)

    def move_arm(self, position):
        return self._driver.move_arm(position)

    def command_end_effector(self, action):
        return self._driver.command_end_effector(action)

    def reset_end_effector(self, target_state=None):
        return self._driver.reset_end_effector(target_state)

    def open_gripper(self):
        return self._driver.open_gripper()

    def close_gripper(self):
        return self._driver.close_gripper()

    def move_gripper(self, position, speed: float = 0.3):
        return self._driver.move_gripper(position, speed)

    def get_hand_type(self):
        return self._driver.get_hand_type()

    def get_hand_state(self):
        return self._driver.get_hand_state()

    def get_hand_detailed_state(self):
        return self._driver.get_hand_detailed_state()

    def get_hand_finger_names(self):
        return self._driver.get_hand_finger_names()

    def cleanup(self) -> None:
        self._driver.disconnect()
