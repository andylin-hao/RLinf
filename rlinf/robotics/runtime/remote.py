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

from typing import Any

from ..part import Camera, ControllablePart, EndEffector, RobotPart


def _wait_first(result: Any) -> Any:
    values = result.wait()
    if len(values) != 1:
        raise RuntimeError(
            f"A remote robot part requires one worker, but got {len(values)} results."
        )
    return values[0]


class RemotePart(RobotPart):
    """Present a one-worker ``PartRuntime`` group as a local robot part."""

    def __init__(self, worker_group: Any) -> None:
        self._worker_group = worker_group
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the remote runtime has been initialized."""
        return self._connected

    @property
    def observation_features(self) -> dict[str, Any]:
        """Fetch observation features from the remote part."""
        return _wait_first(self._worker_group.get_observation_features())

    def connect(self) -> None:
        """Initialize the remote runtime."""
        self._worker_group.initialize().wait()
        self._connected = True

    def reset(self) -> None:
        """Reset the remote part."""
        self._require_connected()
        self._worker_group.reset().wait()

    def get_observation(self) -> dict[str, Any]:
        """Fetch an observation from the remote part."""
        self._require_connected()
        return _wait_first(self._worker_group.get_observation())

    def disconnect(self) -> None:
        """Shut down the remote runtime."""
        if self._connected:
            try:
                try:
                    self._worker_group.shutdown().wait()
                finally:
                    close = getattr(self._worker_group, "_close", None)
                    if callable(close):
                        close()
            finally:
                self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Remote robot part is not connected.")


class RemoteControllablePart(RemotePart, ControllablePart):
    """Remote part that accepts namespaced actions."""

    @property
    def action_features(self) -> dict[str, Any]:
        """Fetch action features from the remote part."""
        return _wait_first(self._worker_group.get_action_features())

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action through the remote runtime."""
        self._require_connected()
        return _wait_first(self._worker_group.send_action(action))


class RemoteCamera(RemotePart, Camera):
    """Camera hosted by a remote ``PartRuntime``."""


class RemoteEndEffector(RemoteControllablePart, EndEffector):
    """End effector hosted by a remote ``PartRuntime``."""
