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

What each device *is* lives in :mod:`.backends`, one class per name. This is
only the order they are built in: every entry first, because a streamer drives
the devices the group composed rather than opening its own.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import gymnasium as gym

from rlinf.robotics.teleop import TeleopGroup

from .backends import EnvFacts, TeleopBackend
from .composed import ComposedTeleop
from .layout import action_spec

__all__ = ["EnvFacts", "TeleopBackend", "build_teleop"]


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
        timeout: Hold window. Left out, the bindings decide: a device held
            down to take over asks for none.
    """
    spec = action_spec(env)
    facts = EnvFacts.about(env, spec.layout, spec.kinds)

    entries: list[Any] = []
    asked: list[type[TeleopBackend]] = []
    for item in devices:
        if isinstance(item, str):
            name, options = item, {}
        else:
            ((name, options),) = dict(item).items()
        backend = TeleopBackend.named(name)
        entries.append(backend.entry(cfg, options or {}, facts))
        if backend not in asked:
            asked.append(backend)

    # After the loop: a streamer drives the devices the group composed, so it
    # cannot be built until they all exist. One env has one rate, so the first
    # backend that wants a thread is the one that gets it.
    streamer = None
    for backend in asked:
        streamer = backend.streamer(cfg, facts, entries)
        if streamer is not None:
            break

    group = TeleopGroup(entries, available=facts.kinds)
    group.connect()
    if timeout is None:
        timeout = group.hold_window
    return ComposedTeleop(group, facts.layout, timeout=timeout, streamer=streamer)
