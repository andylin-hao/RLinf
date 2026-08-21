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

"""Part views over hardware that speaks in methods rather than parts.

Some hardware exposes one connection that drives several logical components:
a dual-arm controller with ``move_left_arm`` / ``move_right_arm``, an arm whose
gripper lives behind ``open_gripper`` / ``close_gripper``, a controller that
returns frames from ``get_camera(id)``. These adapters turn such method
surfaces into proper parts, so composition sees a uniform interface.

A part declares these in :attr:`~.parts.base.Connection.parts`, in Python,
next to the methods they wrap -- not as command/state dictionaries assembled in
a separate factory module.

Each is named for the part category it presents -- an arm, an end effector, a
camera -- with ``Method`` saying how it gets there: by calling methods on the
part that carries it, whose names you pass to the constructor. What it presents
is what a policy sees; how it is wired is what a driver author writes. Nothing
here names a *sub*category, because one view serves the whole of one: the same
:class:`MethodEndEffector` presents a two-fingered gripper and a six-fingered
hand.

Each view holds its host twice under two names, because it uses it for two
things: ``_host`` is what it calls methods on, and ``_owner`` is what the base
class opens, closes and reports connection state for. Setting the second is the
whole of a view's lifecycle -- there is no ``connect`` here, and no ``_open``,
because a view has no device of its own.
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Union, cast

import numpy as np

from .base import ControllablePart
from .cameras.base import Camera
from .end_effectors.base import EndEffector


def state_to_dict(state: Any) -> dict[str, Any]:
    """Normalize a host part's state object into a plain dictionary."""
    if isinstance(state, dict):
        return state
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        return cast(dict[str, Any], to_dict())
    if is_dataclass(state) and not isinstance(state, type):
        return cast(dict[str, Any], asdict(state))
    raise TypeError(f"Host state {type(state).__name__} is not dictionary-like.")


class MethodArm(ControllablePart):
    """One arm of a part that commands arms through named methods.

    Args:
        host: The part owning the connection.
        commands: Map from canonical action field to host method name, e.g.
            ``{"tcp_pose": "move_left_arm"}``.
        state_fields: Canonical observation names, either a tuple selecting
            host state fields verbatim or a map from canonical name to the
            host's own field name.
    """

    def __init__(
        self,
        host: Any,
        commands: dict[str, str],
        state_fields: Optional[Union[tuple[str, ...], dict[str, str]]] = None,
    ) -> None:
        self._host = self._owner = host
        self.commands = dict(commands)
        self.state_fields = (
            dict(state_fields)
            if isinstance(state_fields, dict)
            else {name: name for name in state_fields or ()}
        )

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the state fields this view exposes."""
        return {name: {} for name in self.state_fields}

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe the canonical command names this view accepts."""
        return {name: {} for name in self.commands}

    def reset(self) -> None:
        """Leave task-specific reset motion to the task environment."""

    def get_observation(self) -> dict[str, Any]:
        """Select this view's fields out of the shared host state."""
        state = state_to_dict(self._host.get_state())
        if not self.state_fields:
            return state
        return {name: state[source] for name, source in self.state_fields.items()}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Dispatch canonical command fields to the host's methods."""
        unknown = set(action) - set(self.commands)
        if unknown:
            raise KeyError(f"Unknown arm actions: {sorted(unknown)}")
        applied: dict[str, Any] = {}
        for name, value in action.items():
            getattr(self._host, self.commands[name])(value)
            applied[name] = value
        return applied


class MethodEndEffector(EndEffector):
    """An end effector reached through the methods of the part carrying it.

    Named for what it presents, not for one shape of it: with ``dims=6`` and a
    continuous command this is a dexterous hand, and the shipped Franka path
    composes one. It was called ``MethodGripper``, which was wrong there twice
    over -- a six-fingered hand is not a gripper, and it never had a gripper's
    open/close/width vocabulary either.

    Args:
        host: The part owning the connection.
        state_field: Host state field holding the end-effector value.
        dims: Width of the state and target vectors.
        command: Host method taking a continuous target. When ``None`` the
            view falls back to binary open/close on the sign of ``target[0]``.
        open_method: Host method that opens, in binary mode.
        close_method: Host method that closes, in binary mode.
        state_index: Optional index or slice selecting the end-effector value
            out of a wider state field.
    """

    def __init__(
        self,
        host: Any,
        state_field: str,
        dims: int = 1,
        command: Optional[str] = None,
        open_method: str = "open_gripper",
        close_method: str = "close_gripper",
        state_index: Optional[Union[int, slice]] = None,
    ) -> None:
        self._host = self._owner = host
        self.state_field = state_field
        self.dims = dims
        self.method = command
        self.open_method = open_method
        self.close_method = close_method
        self.state_index = state_index

    @property
    def action_dim(self) -> int:
        """As wide as the host field this drives."""
        return self.dims

    @property
    def state_dim(self) -> int:
        """As wide as the host field this reads."""
        return self.dims

    @property
    def control_mode(self) -> str:
        """Continuous when the host takes a target, binary when it opens and closes."""
        return "continuous" if self.method is not None else "binary"

    def reset(self) -> None:
        """Leave task-specific end-effector reset to the task environment."""

    def get_state(self) -> np.ndarray:
        """Read the end-effector field out of the shared host state."""
        state = state_to_dict(self._host.get_state())
        value = np.asarray(state[self.state_field])
        if self.state_index is not None:
            value = value[self.state_index]
        return np.asarray(value).reshape(-1)

    def command(self, action: np.ndarray) -> bool:
        """Apply a continuous target, or open and close on its sign."""
        target = np.asarray(action).reshape(-1)
        if target.shape != (self.dims,):
            raise ValueError(
                f"Expected end-effector target shape {(self.dims,)}, "
                f"got {target.shape}."
            )
        if self.method is not None:
            getattr(self._host, self.method)(target)
            return True
        opening = bool(target[0] >= 0)
        getattr(self._host, self.open_method if opening else self.close_method)()
        return True


class MethodCamera(Camera):
    """A camera frame returned by one host method.

    Args:
        host: The part owning the connection.
        method: Host method returning a frame.
        method_args: Fixed arguments identifying the camera, e.g. its id.
    """

    def __init__(self, host: Any, method: str, *method_args: Any) -> None:
        self._host = self._owner = host
        self.method = method
        self.method_args = method_args

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the raw frame returned by the host."""
        return {"frame": {}}

    def reset(self) -> None:
        """Camera views have no resettable state."""

    def get_observation(self) -> dict[str, Any]:
        """Fetch one frame through the configured host method."""
        return {"frame": getattr(self._host, self.method)(*self.method_args)}
