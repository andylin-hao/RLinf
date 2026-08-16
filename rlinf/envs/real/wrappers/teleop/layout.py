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

"""Where each named part sits in an env's action vector.

An env already knows this -- ``FrankaEnv.step`` documents
``[x, y, z, rx, ry, rz, gripper]`` and splits ``action[:6]`` from the rest, and
the dual-arm envs reshape by ``PER_ARM_ACTION_DIM``. Naming it is what lets a
teleop group say "the arm" instead of computing a slice.

An env may declare :attr:`ACTION_PARTS` itself. This module derives it for the
envs that have not, from the same values their ``step`` uses.
"""

from __future__ import annotations

import gymnasium as gym

#: Cartesian arm commands, before any end effector.
ARM_DIM = 6


def action_layout(env: gym.Env) -> dict[str, slice]:
    """Return the action layout for ``env``, by part name.

    An env that declares ``ACTION_PARTS`` gets that. Otherwise the layout comes
    from its action space and, for dual-arm envs, ``PER_ARM_ACTION_DIM``.
    """
    inner = env.unwrapped
    declared = getattr(inner, "ACTION_PARTS", None)
    if declared:
        return dict(declared)

    total = int(env.action_space.shape[0])
    per_arm = getattr(inner, "PER_ARM_ACTION_DIM", 0) or 0

    if per_arm and total == 2 * per_arm:
        # Dual arm: each side is an arm followed by its end effector.
        layout: dict[str, slice] = {}
        for index, side in enumerate(("left", "right")):
            start = index * per_arm
            arm_end = start + per_arm - 1
            layout[f"{side}.arm"] = slice(start, arm_end)
            layout[f"{side}.end_effector"] = slice(arm_end, start + per_arm)
        return layout

    if total == ARM_DIM:
        return {"arm": slice(0, ARM_DIM)}

    config = getattr(inner, "config", None)
    end_effector = str(getattr(config, "end_effector_type", "franka_gripper"))
    tail = "hand" if end_effector.endswith("hand") else "end_effector"
    return {"arm": slice(0, ARM_DIM), tail: slice(ARM_DIM, total)}
