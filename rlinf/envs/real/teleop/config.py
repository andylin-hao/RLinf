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

"""Choosing which teleop device an environment listens to.

At most one device drives an env, so the choice is one name --
``teleop_device: spacemouse`` -- rather than a boolean per device that has to be
checked for conflicts at runtime.

The retired booleans (``use_spacemouse``, ``use_gello``, ``use_pico``,
``use_gello_joint``) still work and warn. They are read here rather than in the
wrapper builders so that every env stack retires them on the same schedule.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Sequence

#: Retired flag -> the device it selected.
LEGACY_FLAGS: dict[str, str] = {
    "use_spacemouse": "spacemouse",
    "use_gello": "gello",
    "use_pico": "pico",
    "use_gello_joint": "gello_joint",
}

#: The name meaning "nobody takes over from the policy".
NO_DEVICE = "none"


def _legacy_selection(cfg: Mapping[str, Any]) -> tuple[Optional[str], list[str]]:
    """Return the device the retired booleans select, and which were present."""
    present = [flag for flag in LEGACY_FLAGS if flag in cfg]
    enabled = [flag for flag in present if bool(cfg[flag])]

    if len(enabled) > 1:
        raise ValueError(
            "Only one teleop device can be active at a time, but "
            f"{', '.join(sorted(enabled))} are all enabled. Replace them with a "
            f"single 'teleop_device: {LEGACY_FLAGS[enabled[0]]}'."
        )
    if not present:
        return None, []
    return (LEGACY_FLAGS[enabled[0]] if enabled else NO_DEVICE), present


def resolve_teleop_device(
    cfg: Mapping[str, Any],
    *,
    supported: Sequence[str],
    default: str = NO_DEVICE,
) -> str:
    """Return the teleop device this env config selects.

    Args:
        cfg: The env config section, e.g. ``env.eval``.
        supported: Device names this env stack can drive. A name outside this
            set is an error rather than a silently ignored setting, because a
            missing takeover is only discovered with a robot already moving.
        default: The device to use when the config says nothing.

    Raises:
        ValueError: If the config names an unsupported device, enables more than
            one of the retired booleans, or sets ``teleop_device`` and a retired
            boolean that disagree with each other.
    """
    declared = cfg.get("teleop_device")
    legacy, legacy_present = _legacy_selection(cfg)

    if declared is not None and legacy_present:
        # Env configs layer, so a half-migrated stack can hold both keys. If they
        # agree that is merely redundant; if they disagree, choosing either one
        # silently gives somebody the wrong device on real hardware.
        if legacy != str(declared):
            raise ValueError(
                f"This env config sets 'teleop_device: {declared}' and also "
                f"{', '.join(sorted(legacy_present))}, which select "
                f"'{legacy}'. Remove the retired flags; they are ignored in a "
                "future release and cannot be reconciled here."
            )
        warnings.warn(
            f"'{', '.join(sorted(legacy_present))}' is retired and redundant "
            f"alongside 'teleop_device: {declared}'. Remove the old flags.",
            DeprecationWarning,
            stacklevel=2,
        )

    if declared is not None:
        device = str(declared)
    elif legacy is not None:
        warnings.warn(
            f"'{', '.join(sorted(legacy_present))}' is retired. Use "
            f"'teleop_device: {legacy}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        device = legacy
    else:
        device = default

    allowed = {*supported, NO_DEVICE}
    if device not in allowed:
        raise ValueError(
            f"Unsupported teleop device {device!r} for this environment. "
            f"Supported: {', '.join(sorted(allowed))}."
        )
    return device
