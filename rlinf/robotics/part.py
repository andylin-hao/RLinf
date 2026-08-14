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

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RobotPart(ABC):
    """Observable part of a physical robot, such as an arm or camera."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the part is ready for observations."""

    @property
    @abstractmethod
    def observation_features(self) -> dict[str, Any]:
        """Describe the values returned by :meth:`get_observation`."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the physical part."""

    @abstractmethod
    def get_observation(self) -> dict[str, np.ndarray]:
        """Read the current part observation."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources owned by the part."""


class ControllablePart(RobotPart):
    """Robot part that accepts commands in addition to observations."""

    @property
    @abstractmethod
    def action_features(self) -> dict[str, Any]:
        """Describe the values accepted by :meth:`send_action`."""

    @abstractmethod
    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply an action and return the action actually sent."""
