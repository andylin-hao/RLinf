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

"""Building the teleop a config asks for.

Each entry in :data:`DEVICES` knows how to construct one device and the binding
that says what it means. Adding a device is an entry here plus the two classes,
rather than another branch in the wrapper stack.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import gymnasium as gym

from rlinf.robotics.parts.teleop import Glove, SpaceMouse, TeleopLeaderArm
from rlinf.robotics.teleop import (
    GloveBinding,
    LeaderArmBinding,
    LeaderJointBinding,
    SpaceMouseBinding,
    TeleopEntry,
    TeleopGroup,
)

from .composed import ComposedTeleop
from .layout import action_layout

#: Builder for one entry, given the env config and this entry's own options.
EntryBuilder = Callable[[Mapping[str, Any], Mapping[str, Any]], TeleopEntry]


def _spacemouse(cfg: Mapping[str, Any], options: Mapping[str, Any]) -> TeleopEntry:
    return TeleopEntry(
        SpaceMouse(device_index=int(options.get("device_index", 0))),
        SpaceMouseBinding(),
        drives=options.get("drives"),
    )


def _gello(cfg: Mapping[str, Any], options: Mapping[str, Any]) -> TeleopEntry:
    port = options.get("port", cfg.get("gello_port"))
    if port is None:
        raise ValueError(
            "teleop device 'gello' requires 'gello_port' in the env config "
            "(e.g. env.eval.gello_port)."
        )
    return TeleopEntry(
        TeleopLeaderArm(port=port), LeaderArmBinding(), drives=options.get("drives")
    )


def _gello_joint(cfg: Mapping[str, Any], options: Mapping[str, Any]) -> TeleopEntry:
    drives = options.get("drives")
    if drives is None:
        raise ValueError(
            "teleop device 'gello_joint' drives one arm, so it says which. List "
            "one entry per arm, e.g. teleop: [{gello_joint: {drives: left}}, "
            "{gello_joint: {drives: right}}]."
        )
    port = options.get("port") or cfg.get(f"{drives}_gello_port")
    if port is None:
        raise ValueError(
            "teleop device 'gello_joint' requires a 'port', or "
            f"'{drives}_gello_port' in the env config."
        )
    side = {"left": 0, "right": 1}.get(str(drives), 0)
    return TeleopEntry(
        TeleopLeaderArm(port=port, joint_space=True),
        LeaderJointBinding(
            side=side,
            use_delta=bool(options.get("use_delta", False)),
            action_scale=float(options.get("action_scale", 0.1)),
        ),
        drives=drives,
    )


def _glove(cfg: Mapping[str, Any], options: Mapping[str, Any]) -> TeleopEntry:
    glove_cfg = dict(cfg.get("glove_config", {}))
    glove_cfg.update(options)
    return TeleopEntry(
        Glove(
            left_port=glove_cfg.get("left_port", "/dev/ttyACM0"),
            right_port=glove_cfg.get("right_port"),
            frequency=int(glove_cfg.get("frequency", 60)),
            config_file=glove_cfg.get("config_file"),
        ),
        GloveBinding(),
        drives=options.get("drives"),
    )


#: Device name -> how to build its entry.
DEVICES: dict[str, EntryBuilder] = {
    "spacemouse": _spacemouse,
    "gello": _gello,
    "gello_joint": _gello_joint,
    "glove": _glove,
}


def build_teleop(
    env: gym.Env,
    cfg: Mapping[str, Any],
    devices: Sequence[Any],
    timeout: Optional[float] = None,
) -> ComposedTeleop:
    """Build the teleop this env config asks for.

    Args:
        env: The env being wrapped, which supplies the action layout.
        cfg: The env config section, for options devices share.
        devices: Device names, or single-key mappings carrying options, e.g.
            ``{"gello_joint": {"port": "/dev/left", "drives": "left"}}``.
        timeout: Hold window, when the rig wants one other than the default.
    """
    layout = action_layout(env)
    entries = []
    for item in devices:
        if isinstance(item, str):
            name, options = item, {}
        else:
            ((name, options),) = dict(item).items()
        if name not in DEVICES:
            raise ValueError(
                f"Unknown teleop device {name!r}. Known: {sorted(DEVICES)}."
            )
        entries.append(DEVICES[name](cfg, dict(options or {})))

    group = TeleopGroup(entries, available=layout)
    group.connect()
    return ComposedTeleop(group, layout, timeout=timeout)
