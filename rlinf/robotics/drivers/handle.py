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

"""Uniform access to a driver, whether it runs here or in a worker.

A handle answers two questions the same way in both cases: *what parts does
this connection expose* and *how do I call a driver method that is not on the
part interface* (``is_robot_up``, ``clear_errors``, ``reset_joint``).

Off-interface calls always return a result object with ``wait() -> list``, so
call sites read identically local or remote::

    handle.is_robot_up().wait()[0]
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..part import Camera, ControllablePart, EndEffector, RobotPart


class LocalResult:
    """A value computed in this process, shaped like a worker-group result."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def wait(self) -> list[Any]:
        """Return the value in the one-element list callers expect."""
        return [self._value]


class DriverHandle(ABC):
    """Common surface over a local driver or a hosted one."""

    @property
    @abstractmethod
    def parts(self) -> dict[str, RobotPart]:
        """Return the parts this connection exposes, keyed by driver-local name."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the connection and, when hosted, its worker."""

    def part(self, name: str) -> RobotPart:
        """Return one named part, or raise a clear configuration error."""
        if name not in self.parts:
            raise KeyError(
                f"Driver exposes no part {name!r}. Available: {sorted(self.parts)}."
            )
        return self.parts[name]


class LocalDriverHandle(DriverHandle):
    """Handle for a driver constructed in this process."""

    def __init__(self, driver: Any) -> None:
        self._driver = driver
        self._parts = dict(driver.parts())

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Return the driver's own part objects."""
        return self._parts

    @property
    def driver(self) -> Any:
        """Return the underlying driver, for code that legitimately needs it."""
        return self._driver

    def disconnect(self) -> None:
        """Disconnect the driver if it is still connected."""
        if self._driver.is_connected:
            self._driver.disconnect()

    def __getattr__(self, name: str) -> Any:
        """Forward off-interface driver methods, wrapping results for symmetry."""
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._driver, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> LocalResult:
            return LocalResult(attr(*args, **kwargs))

        return call


class RemoteDriverHandle(DriverHandle):
    """Handle for a driver hosted in a one-worker scheduler group."""

    def __init__(self, worker_group: Any, described: dict[str, dict[str, Any]]) -> None:
        self._worker_group = worker_group
        self._parts: dict[str, RobotPart] = {
            name: _make_remote_part(worker_group, name, entry)
            for name, entry in described.items()
        }
        self._connected = True

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Return proxies for the hosted driver's parts."""
        return self._parts

    @property
    def worker_group(self) -> Any:
        """Return the underlying worker group."""
        return self._worker_group

    def disconnect(self) -> None:
        """Shut the hosted driver down and terminate its worker."""
        if not self._connected:
            return
        self._connected = False
        try:
            self._worker_group.shutdown().wait()
        finally:
            close = getattr(self._worker_group, "_close", None)
            if callable(close):
                close()

    def __getattr__(self, name: str) -> Any:
        """Forward off-interface driver methods to the worker group.

        ``WorkerGroup`` binds every public method of the hosted class, so this
        reaches ``is_robot_up``, ``reset_joint``, ``clear_errors`` and friends
        with no per-driver declaration.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._worker_group, name)


def _first(result: Any) -> Any:
    """Unwrap a one-worker result, rejecting groups sized for more."""
    values = result.wait()
    if len(values) != 1:
        raise RuntimeError(
            f"A driver must be hosted by exactly one worker, got {len(values)} results."
        )
    return values[0]


class RemotePart(RobotPart):
    """One part of a hosted driver, addressed by name through its worker."""

    def __init__(
        self,
        worker_group: Any,
        part_name: str,
        observation_features: dict[str, Any],
    ) -> None:
        self._worker_group = worker_group
        self._part_name = part_name
        self._observation_features = observation_features

    @property
    def is_connected(self) -> bool:
        """Parts of a hosted driver are live for as long as the driver is."""
        return True

    @property
    def observation_features(self) -> dict[str, Any]:
        """Return the features captured when the driver was described."""
        return self._observation_features

    def connect(self) -> None:
        """No-op: the hosted driver connects when its worker starts."""

    def reset(self) -> None:
        """Reset this part through the hosted driver."""
        self._worker_group.part_reset(self._part_name).wait()

    def get_observation(self) -> dict[str, Any]:
        """Read this part's observation through the hosted driver."""
        return _first(self._worker_group.part_observation(self._part_name))

    def disconnect(self) -> None:
        """No-op: the handle owns the hosted driver's lifetime."""


class RemoteControllablePart(RemotePart, ControllablePart):
    """Hosted part that also accepts actions."""

    def __init__(
        self,
        worker_group: Any,
        part_name: str,
        observation_features: dict[str, Any],
        action_features: dict[str, Any],
    ) -> None:
        super().__init__(worker_group, part_name, observation_features)
        self._action_features = action_features

    @property
    def action_features(self) -> dict[str, Any]:
        """Return the action features captured at describe time."""
        return self._action_features

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action to this part through the hosted driver."""
        return _first(self._worker_group.part_action(self._part_name, action))


class RemoteEndEffector(RemoteControllablePart, EndEffector):
    """Hosted end effector."""


class RemoteCamera(RemotePart, Camera):
    """Hosted camera."""


_REMOTE_PART_BY_KIND: dict[str, Callable[..., RemotePart]] = {
    "end_effector": RemoteEndEffector,
    "camera": RemoteCamera,
    "controllable": RemoteControllablePart,
    "part": RemotePart,
}


def _make_remote_part(
    worker_group: Any,
    name: str,
    described: dict[str, Any],
) -> RemotePart:
    """Build a proxy mirroring the interface of the hosted part."""
    kind = described["kind"]
    part_cls = _REMOTE_PART_BY_KIND[kind]
    if kind in ("end_effector", "controllable"):
        return part_cls(
            worker_group,
            name,
            described["observation"],
            described["action"],
        )
    return part_cls(worker_group, name, described["observation"])
