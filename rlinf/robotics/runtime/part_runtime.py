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

from typing import Any, Optional

import numpy as np

from rlinf.scheduler import Worker

from ..part import ControllablePart, RobotPart


class PartRuntime(Worker):
    """Host one robot part in an RLinf scheduler worker."""

    def __init__(
        self,
        part_cls: type[RobotPart],
        part_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._part_cls = part_cls
        self._part_kwargs = dict(part_kwargs or {})
        self._part: Optional[RobotPart] = None

    def initialize(self) -> None:
        """Construct and connect the part on the worker's node."""
        if self._part is not None:
            raise RuntimeError("PartRuntime is already initialized.")
        part = self._part_cls(**self._part_kwargs)
        part.connect()
        self._part = part

    def is_connected(self) -> bool:
        """Return whether the hosted part is connected."""
        return self._require_part().is_connected

    def get_observation(self) -> dict[str, np.ndarray]:
        """Read an observation from the hosted part."""
        return self._require_part().get_observation()

    def send_action(self, action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Send an action to a controllable hosted part."""
        part = self._require_part()
        if not isinstance(part, ControllablePart):
            raise TypeError(f"{type(part).__name__} is not controllable.")
        return part.send_action(action)

    def shutdown(self) -> None:
        """Disconnect the hosted part."""
        part = self._part
        self._part = None
        if part is not None and part.is_connected:
            part.disconnect()

    def _require_part(self) -> RobotPart:
        if self._part is None:
            raise RuntimeError("PartRuntime.initialize() must be called first.")
        return self._part
