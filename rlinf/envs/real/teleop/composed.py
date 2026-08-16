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

"""Presenting a composed teleop group as one device to the env.

:class:`~rlinf.robotics.teleop.group.TeleopGroup` produces action *parts*, named
after what they drive. An env takes one flat vector. This is the only place that
knows how to get from one to the other, and it is the only piece of the teleop
story that has to be on the env side at all.

The layout comes from the env, which built the action space in the first place.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

from rlinf.robotics.teleop import TeleopGroup

from .intervention import TeleopDevice, TeleopSample


def context_from(env: gym.Env) -> dict[str, Any]:
    """Gather what bindings may ask about the robot they are driving.

    Each value is fetched only if the env offers it, so an env without joints
    costs nothing and a binding that needs them fails where it is used rather
    than here.
    """
    context: dict[str, Any] = {}
    for key, getter in (
        ("tcp_pose", "get_tcp_pose"),
        ("action_scale", "get_action_scale"),
        ("joint_positions", "get_joint_positions"),
    ):
        try:
            value = env.get_wrapper_attr(getter)
        except AttributeError:
            continue
        if callable(value):
            context[key] = value()

    state = getattr(env.unwrapped, "_franka_state", None)
    if state is not None:
        context["gripper_open"] = bool(getattr(state, "gripper_open", True))
    return context


class ComposedTeleop(TeleopDevice):
    """A group of devices, flattened into this env's action.

    Args:
        group: The devices and bindings in play.
        layout: Where each named part sits in the action vector. Parts the
            group does not fill keep whatever the policy asked for.
        timeout: How long the operator keeps control after their last active
            reading. Zero for rigs whose devices say exactly when they are
            driving.
    """

    def __init__(
        self,
        group: TeleopGroup,
        layout: Mapping[str, slice],
        timeout: Optional[float] = None,
    ) -> None:
        unknown = set(group.parts) - set(layout)
        if unknown:
            raise ValueError(
                f"The teleop group drives {sorted(unknown)}, which this env's "
                f"action layout does not have. It has {sorted(layout)}."
            )
        self.group = group
        self.layout = dict(layout)
        if timeout is not None:
            self.timeout = timeout

    def reset(self, env: gym.Env) -> None:
        """Let every binding drop what it held from the previous episode."""
        self.group.reset()

    def read(self, env: gym.Env, policy_action: np.ndarray) -> TeleopSample:
        """Read every device, then write each part into the action."""
        parts, driving, info = self.group.action(context_from(env))
        if not parts:
            return TeleopSample(action=None, active=False, info=info)

        action = np.array(policy_action, dtype=np.float64, copy=True)
        for name, value in parts.items():
            action[self.layout[name]] = np.asarray(value, dtype=np.float64)
        return TeleopSample(action=action, active=driving, info=info)

    def close(self) -> None:
        """Release every device in the group."""
        self.group.disconnect()
