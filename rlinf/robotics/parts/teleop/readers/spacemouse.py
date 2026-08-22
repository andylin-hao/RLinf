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

import threading

import numpy as np


class SpaceMouseExpert:
    """Read SpaceMouse motion and button state in a background thread."""

    def __init__(self, device_index: int = 0) -> None:
        import pyspacemouse

        self._device = pyspacemouse.open(device_index=device_index)

        self.state_lock = threading.Lock()
        self.latest_data: dict = {"action": np.zeros(6), "buttons": [0, 0]}
        self.thread = threading.Thread(target=self._read_spacemouse, daemon=True)
        self.thread.start()

    def _read_spacemouse(self) -> None:
        while True:
            state = self._device.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # Express SpaceMouse axes in the robot base frame.
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> tuple[np.ndarray, list]:
        """Return the latest motion and button state."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]


if __name__ == "__main__":
    import time

    def test_spacemouse() -> None:
        """Print SpaceMouse readings at 10 Hz until interrupted."""
        spacemouse = SpaceMouseExpert()
        with np.printoptions(precision=3, suppress=True):
            while True:
                action, buttons = spacemouse.get_action()
                print(f"Spacemouse action: {action}, buttons: {buttons}")
                time.sleep(0.1)

    test_spacemouse()
