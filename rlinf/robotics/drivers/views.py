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

"""Part views over a driver that speaks in methods rather than parts.

Some hardware exposes one connection that drives several logical components:
a dual-arm controller with ``move_left_arm`` / ``move_right_arm``, an arm whose
gripper lives behind ``open_gripper`` / ``close_gripper``, a controller that
returns frames from ``get_camera(id)``. These adapters turn such method
surfaces into proper parts, so composition sees a uniform interface.

A driver declares its views in :meth:`~.base.Driver.parts`, in Python, next to
the methods they wrap -- not as command/state dictionaries assembled in a
separate factory module.
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Union, cast

import numpy as np

from ..part import Camera, ControllablePart, EndEffector


def state_to_dict(state: Any) -> dict[str, Any]:
    """Normalize a driver state object into a plain dictionary."""
    if isinstance(state, dict):
        return state
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        return cast(dict[str, Any], to_dict())
    if is_dataclass(state) and not isinstance(state, type):
        return cast(dict[str, Any], asdict(state))
    raise TypeError(f"Driver state {type(state).__name__} is not dictionary-like.")


class DriverArm(ControllablePart):
    """One arm of a driver that commands arms through named methods.

    Args:
        driver: The driver owning the connection.
        commands: Map from canonical action field to driver method name, e.g.
            ``{"tcp_pose": "move_left_arm"}``.
        state_fields: Canonical observation names, either a tuple selecting
            driver state fields verbatim or a map from canonical name to the
            driver's own field name.
    """

    def __init__(
        self,
        driver: Any,
        commands: dict[str, str],
        state_fields: Optional[Union[tuple[str, ...], dict[str, str]]] = None,
    ) -> None:
        self._driver = driver
        self.commands = dict(commands)
        self.state_fields = (
            dict(state_fields)
            if isinstance(state_fields, dict)
            else {name: name for name in state_fields or ()}
        )

    @property
    def is_connected(self) -> bool:
        """Follow the owning driver's connection state."""
        return self._driver.is_connected

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the state fields this view exposes."""
        return {name: {} for name in self.state_fields}

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe the canonical command names this view accepts."""
        return {name: {} for name in self.commands}

    def connect(self) -> None:
        """No-op: the owning driver holds the connection."""

    def reset(self) -> None:
        """Leave task-specific reset motion to the task environment."""

    def get_observation(self) -> dict[str, Any]:
        """Select this view's fields out of the shared driver state."""
        state = state_to_dict(self._driver.get_state())
        if not self.state_fields:
            return state
        return {name: state[source] for name, source in self.state_fields.items()}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Dispatch canonical command fields to the driver's methods."""
        unknown = set(action) - set(self.commands)
        if unknown:
            raise KeyError(f"Unknown arm actions: {sorted(unknown)}")
        applied: dict[str, Any] = {}
        for name, value in action.items():
            getattr(self._driver, self.commands[name])(value)
            applied[name] = value
        return applied

    def disconnect(self) -> None:
        """No-op: the owning driver holds the connection."""


class DriverGripper(EndEffector):
    """End effector implemented inside an arm driver's own connection.

    Args:
        driver: The driver owning the connection.
        state_field: Driver state field holding the end-effector value.
        action_dim: Width of the target vector.
        command: Driver method taking a continuous target. When ``None`` the
            view falls back to binary open/close on the sign of ``target[0]``.
        state_index: Optional index or slice selecting the end-effector value
            out of a wider state field.
    """

    def __init__(
        self,
        driver: Any,
        state_field: str,
        action_dim: int = 1,
        command: Optional[str] = None,
        open_method: str = "open_gripper",
        close_method: str = "close_gripper",
        state_index: Optional[Union[int, slice]] = None,
    ) -> None:
        self._driver = driver
        self.state_field = state_field
        self.action_dim = action_dim
        self.command = command
        self.open_method = open_method
        self.close_method = close_method
        self.state_index = state_index

    @property
    def is_connected(self) -> bool:
        """Follow the owning driver's connection state."""
        return self._driver.is_connected

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the end-effector state vector."""
        return {"state": {"shape": (self.action_dim,), "dtype": "float32"}}

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe the end-effector target vector."""
        return {"target": {"shape": (self.action_dim,), "dtype": "float32"}}

    def connect(self) -> None:
        """No-op: the owning driver holds the connection."""

    def reset(self) -> None:
        """Leave task-specific end-effector reset to the task environment."""

    def get_observation(self) -> dict[str, np.ndarray]:
        """Read the end-effector field out of the shared driver state."""
        state = state_to_dict(self._driver.get_state())
        value = np.asarray(state[self.state_field])
        if self.state_index is not None:
            value = value[self.state_index]
        return {"state": np.asarray(value).reshape(-1)}

    def send_action(self, action: dict[str, Any]) -> dict[str, np.ndarray]:
        """Apply a continuous target, or a binary open/close command."""
        if set(action) != {"target"}:
            raise KeyError("End-effector action must contain only 'target'.")
        target = np.asarray(action["target"]).reshape(-1)
        if target.shape != (self.action_dim,):
            raise ValueError(
                f"Expected end-effector target shape {(self.action_dim,)}, "
                f"got {target.shape}."
            )
        if self.command is not None:
            getattr(self._driver, self.command)(target)
        else:
            method = self.open_method if target[0] >= 0 else self.close_method
            getattr(self._driver, method)()
        return {"target": target}

    def disconnect(self) -> None:
        """No-op: the owning driver holds the connection."""


class DriverCamera(Camera):
    """A camera frame returned by one driver method.

    Args:
        driver: The driver owning the connection.
        method: Driver method returning a frame.
        method_args: Fixed arguments identifying the camera, e.g. its id.
    """

    def __init__(self, driver: Any, method: str, *method_args: Any) -> None:
        self._driver = driver
        self.method = method
        self.method_args = method_args

    @property
    def is_connected(self) -> bool:
        """Follow the owning driver's connection state."""
        return self._driver.is_connected

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the raw frame returned by the driver."""
        return {"frame": {}}

    def connect(self) -> None:
        """No-op: the owning driver holds the connection."""

    def reset(self) -> None:
        """Camera views have no resettable state."""

    def get_observation(self) -> dict[str, Any]:
        """Fetch one frame through the configured driver method."""
        return {"frame": getattr(self._driver, self.method)(*self.method_args)}

    def disconnect(self) -> None:
        """No-op: the owning driver holds the connection."""
