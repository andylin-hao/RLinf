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

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..part import Camera, ControllablePart
from ..robot import Robot


class RobotRuntime:
    """Coordinate local or remote parts through one composed robot API."""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot

    @property
    def is_connected(self) -> bool:
        """Whether every robot component is connected."""
        return self.robot.is_connected

    @property
    def observation_features(self) -> dict[str, Any]:
        """Return canonical observation features."""
        return self.robot.observation_features

    @property
    def action_features(self) -> dict[str, Any]:
        """Return canonical action features."""
        return self.robot.action_features

    def connect(self) -> None:
        """Connect all robot components."""
        self.robot.connect()

    def attach_camera(
        self,
        name: str,
        camera: Camera,
        *,
        arm: str | None = None,
    ) -> None:
        """Attach an already constructed camera to the robot or one arm."""
        cameras = self.robot.cameras if arm is None else self.robot.arms[arm].cameras
        if not name:
            raise ValueError("Camera name must be a non-empty string.")
        if name in cameras:
            raise ValueError(f"Camera {name!r} is already attached.")
        cameras[name] = camera

    def reset(self) -> None:
        """Reset all arms in parallel and then reset extra parts."""
        with ThreadPoolExecutor(max_workers=max(1, len(self.robot.arms))) as executor:
            futures = [executor.submit(arm.reset) for arm in self.robot.arms.values()]
            for future in futures:
                future.result()
        for part in self.robot.parts.values():
            part.reset()

    def get_observation(self) -> dict[str, Any]:
        """Read independent arms and cameras concurrently."""
        observation: dict[str, Any] = {"arms": {}}
        jobs: dict[tuple[str, str], Any] = {}
        component_count = (
            len(self.robot.arms) + len(self.robot.cameras) + len(self.robot.parts)
        )
        with ThreadPoolExecutor(max_workers=max(1, component_count)) as executor:
            for name, arm in self.robot.arms.items():
                jobs[("arms", name)] = executor.submit(arm.get_observation)
            for name, camera in self.robot.cameras.items():
                jobs[("cameras", name)] = executor.submit(camera.get_observation)
            for name, part in self.robot.parts.items():
                jobs[("parts", name)] = executor.submit(part.get_observation)
            for (section, name), future in jobs.items():
                observation.setdefault(section, {})[name] = future.result()
        return observation

    def send_action(
        self, action: Mapping[str, Mapping[str, dict[str, Any]]]
    ) -> dict[str, Any]:
        """Dispatch independent arm actions concurrently."""
        unknown_sections = set(action) - {"arms", "parts"}
        if unknown_sections:
            raise KeyError(f"Unknown robot action sections: {sorted(unknown_sections)}")

        applied: dict[str, Any] = {}
        arm_actions = action.get("arms", {})
        unknown_arms = set(arm_actions) - set(self.robot.arms)
        if unknown_arms:
            raise KeyError(f"Unknown robot arms: {sorted(unknown_arms)}")
        if arm_actions:
            applied["arms"] = {}
            with ThreadPoolExecutor(max_workers=len(arm_actions)) as executor:
                futures = {
                    name: executor.submit(
                        self.robot.arms[name].send_action, dict(arm_action)
                    )
                    for name, arm_action in arm_actions.items()
                }
                for name, future in futures.items():
                    applied["arms"][name] = future.result()

        part_actions = action.get("parts", {})
        unknown_parts = set(part_actions) - set(self.robot.parts)
        if unknown_parts:
            raise KeyError(f"Unknown robot parts: {sorted(unknown_parts)}")
        if part_actions:
            applied["parts"] = {}
            for name, part_action in part_actions.items():
                part = self.robot.parts[name]
                if not isinstance(part, ControllablePart):
                    raise TypeError(f"Robot part {name!r} is not controllable.")
                applied["parts"][name] = part.send_action(part_action)
        return applied

    def disconnect(self) -> None:
        """Disconnect all robot components."""
        self.robot.disconnect()
