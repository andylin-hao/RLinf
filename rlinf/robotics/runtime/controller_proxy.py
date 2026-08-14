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

from dataclasses import asdict, is_dataclass
from typing import Any, Optional, cast

import numpy as np

from ..part import Camera, ControllablePart, EndEffector


def _wait_first(result: Any) -> Any:
    values = result.wait()
    if len(values) != 1:
        raise RuntimeError(
            f"A robot controller requires one worker, but got {len(values)} results."
        )
    return values[0]


def _state_dict(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        return state
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        return cast(dict[str, Any], to_dict())
    if is_dataclass(state) and not isinstance(state, type):
        return cast(dict[str, Any], asdict(state))
    raise TypeError(f"Controller state {type(state).__name__} is not dictionary-like.")


class RemoteControllerArm(ControllablePart):
    """Expose an existing one-worker controller through the arm driver API."""

    def __init__(
        self,
        controller: Any,
        command_methods: dict[str, str],
        state_fields: Optional[tuple[str, ...] | dict[str, str]] = None,
        owns_controller: bool = False,
    ) -> None:
        self.controller = controller
        self.command_methods = dict(command_methods)
        self.state_fields = (
            dict(state_fields)
            if isinstance(state_fields, dict)
            else {name: name for name in state_fields or ()}
        )
        self.owns_controller = owns_controller
        self._connected = True

    @property
    def is_connected(self) -> bool:
        """Whether the already-launched controller is available."""
        return self._connected

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the state fields exposed by this controller."""
        return {name: {} for name in self.state_fields}

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe the controller's canonical command names."""
        return {name: {} for name in self.command_methods}

    def connect(self) -> None:
        """Mark the controller proxy connected; launching owns setup."""
        self._connected = True

    def reset(self) -> None:
        """Leave task-specific reset motion to the task environment."""

    def get_observation(self) -> dict[str, Any]:
        """Read and normalize the remote controller state."""
        self._require_connected()
        state = _state_dict(_wait_first(self.controller.get_state()))
        if not self.state_fields:
            return state
        return {
            name: state[source_name] for name, source_name in self.state_fields.items()
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Dispatch canonical command fields to controller RPC methods."""
        self._require_connected()
        unknown = set(action) - set(self.command_methods)
        if unknown:
            raise KeyError(f"Unknown controller arm actions: {sorted(unknown)}")
        applied: dict[str, Any] = {}
        for name, value in action.items():
            method = getattr(self.controller, self.command_methods[name])
            method(value).wait()
            applied[name] = value
        return applied

    def disconnect(self) -> None:
        """Detach the proxy and optionally terminate its controller worker."""
        try:
            if self._connected and self.owns_controller:
                try:
                    self.controller.cleanup().wait()
                finally:
                    self.controller._close()
        finally:
            self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Remote controller arm is not connected.")


class RemoteControllerEndEffector(EndEffector):
    """Expose an end effector implemented inside a controller worker."""

    def __init__(
        self,
        controller: Any,
        state_field: str,
        action_dim: int = 1,
        command_method: Optional[str] = None,
        open_method: str = "open_gripper",
        close_method: str = "close_gripper",
        state_index: Optional[int | slice] = None,
    ) -> None:
        self.controller = controller
        self.state_field = state_field
        self.action_dim = action_dim
        self.command_method = command_method
        self.open_method = open_method
        self.close_method = close_method
        self.state_index = state_index
        self._connected = True

    @property
    def is_connected(self) -> bool:
        """Whether the controller proxy is available."""
        return self._connected

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the end-effector state vector."""
        return {"state": {"shape": (self.action_dim,), "dtype": "float32"}}

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe the end-effector target vector."""
        return {"target": {"shape": (self.action_dim,), "dtype": "float32"}}

    def connect(self) -> None:
        """Mark the controller-backed end effector connected."""
        self._connected = True

    def reset(self) -> None:
        """Leave task-specific end-effector reset to the task environment."""

    def get_observation(self) -> dict[str, np.ndarray]:
        """Read the end-effector field from the shared controller state."""
        self._require_connected()
        state = _state_dict(_wait_first(self.controller.get_state()))
        value = np.asarray(state[self.state_field])
        if self.state_index is not None:
            value = value[self.state_index]
        return {"state": np.asarray(value).reshape(-1)}

    def send_action(self, action: dict[str, Any]) -> dict[str, np.ndarray]:
        """Apply a continuous target or a binary open/close command."""
        self._require_connected()
        if set(action) != {"target"}:
            raise KeyError("End-effector action must contain only 'target'.")
        target = np.asarray(action["target"]).reshape(-1)
        if target.shape != (self.action_dim,):
            raise ValueError(
                f"Expected end-effector target shape {(self.action_dim,)}, "
                f"got {target.shape}."
            )
        if self.command_method is not None:
            getattr(self.controller, self.command_method)(target).wait()
        else:
            method = self.open_method if target[0] >= 0 else self.close_method
            getattr(self.controller, method)().wait()
        return {"target": target}

    def disconnect(self) -> None:
        """Detach the proxy without terminating its shared controller worker."""
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Remote controller end effector is not connected.")


class RemoteMethodCamera(Camera):
    """Expose a camera frame returned by one controller RPC method."""

    def __init__(
        self,
        controller: Any,
        method: str,
        *method_args: Any,
    ) -> None:
        self.controller = controller
        self.method = method
        self.method_args = method_args
        self._connected = True

    @property
    def is_connected(self) -> bool:
        """Whether the controller proxy is available."""
        return self._connected

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the raw frame returned by the controller."""
        return {"frame": {}}

    def connect(self) -> None:
        """Mark the controller-backed camera connected."""
        self._connected = True

    def reset(self) -> None:
        """Camera proxies have no resettable state."""

    def get_observation(self) -> dict[str, Any]:
        """Fetch one frame through the configured controller RPC."""
        if not self._connected:
            raise RuntimeError("Remote method camera is not connected.")
        method = getattr(self.controller, self.method)
        return {"frame": _wait_first(method(*self.method_args))}

    def disconnect(self) -> None:
        """Detach the proxy without terminating its shared controller worker."""
        self._connected = False
