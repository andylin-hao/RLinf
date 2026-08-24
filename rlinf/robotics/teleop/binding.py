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

"""Interfaces for mapping device readings to named robot actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .kinds import ActionKind

#: Context fields that bindings may request through ``TeleopBinding.NEEDS``.
CONTEXT_KEYS = (
    "tcp_pose",
    "action_scale",
    "joint_positions",
    "gripper_open",
    "hand_reset_pose",
)


@dataclass
class TeleopAction:
    """Action contribution produced from one device reading.

    Attributes:
        parts: The action parts this device fills, by name.
        driving: Whether the operator is actually driving. Devices report small
            residual motion constantly, so each binding decides its own
            threshold.
        info: Device state worth recording alongside the step it produced.
    """

    parts: dict[str, np.ndarray] = field(default_factory=dict)
    driving: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class TeleopBinding(ABC):
    """Map one device's readings to named robot action parts."""

    #: Action parts this binding can fill and their semantic kinds.
    PRODUCES: Mapping[str, "ActionKind"] = {}

    #: Motion below this threshold is treated as device noise.
    MOVEMENT_EPSILON: float = 0.001

    #: Hold duration after the last active reading; ``None`` uses the default.
    HOLD_WINDOW: float | None = None

    #: Whether produced actions must be clipped to the environment space.
    CLIPS_TO_ACTION_SPACE: bool = False

    #: Whether this binding's parts still apply while the operator is idle.
    APPLIES_WHILE_IDLE: bool = False

    #: Context keys this binding cannot work without, from :data:`CONTEXT_KEYS`.
    NEEDS: tuple[str, ...] = ()

    @abstractmethod
    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Convert one device reading into an action contribution."""

    def publish(self, reading: Mapping[str, Any]) -> dict[str, Any]:
        """Return context made available to subsequent bindings."""
        return {}

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return actions that hold this binding's controlled parts in place."""
        return {}

    def on_action_chunk_begin(self) -> None:
        """Let go of anything held only until the next chunk of actions."""

    def reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Reset internal state using the robot's post-reset context."""
