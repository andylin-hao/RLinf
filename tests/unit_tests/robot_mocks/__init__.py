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

"""Vendor SDKs, faked, so a real part can be run without its hardware.

Every part imports its SDK inside ``_open`` or ``connect`` rather than at module
scope. Putting a fake in :data:`sys.modules` first is therefore enough for the
part's own code -- its lifecycle, its state conversion, its action dispatch --
to run unchanged against something that answers. Nothing about the part is
stubbed; only what is on the other end of the cable.

That is the difference from a test double: a double proves the code around the
part is right, and this proves the part is.

Use it as a context manager::

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        robot = build_robot("DualFranka", left_robot_ip="1.2.3.4", ...)
        robot.connect()

The bench check takes ``--mock`` and does exactly this, so the same command
that verifies a real robot also runs in CI.
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any, Iterator

from . import arms, cameras, grippers
from ._fakes import Recorder, module

__all__ = ["Recorder", "mocked_sdks", "module", "sdk_modules"]

#: Attributes to swap on modules that import their dependency at module scope,
#: where a :data:`sys.modules` entry arrives too late. Keyed by dotted module.
_PATCHES: dict[str, dict[str, Any]] = {}


def _no_processes() -> Any:
    """A ``psutil`` whose Popen starts nothing.

    The Franka ROS arm launches its impedance controller with
    ``psutil.Popen(["roslaunch", ...])``. Faking the SDK is not enough when the
    part also starts a process, so this stands in for one.
    """

    class Popen:
        def __init__(self, args, **_kwargs):
            self.args = args
            self.returncode = None

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def is_running(self):
            return self.returncode is None

        def status(self):
            return "running" if self.returncode is None else "terminated"

        def name(self):
            return self.args[0] if self.args else "process"

        def children(self, recursive=False):
            return []

    return types.SimpleNamespace(
        Popen=Popen,
        NoSuchProcess=Exception,
        # Nothing is running, so nothing has to be found or waited for.
        process_iter=lambda *_a, **_k: iter(()),
        pid_exists=lambda _pid: False,
    )


def sdk_modules() -> dict[str, types.ModuleType]:
    """Every faked SDK, by the name a part imports it as."""
    made: dict[str, types.ModuleType] = {}
    # Parts that launch a process rather than only talking to an SDK: the ROS
    # transport starts roscore, and the Franka arm its impedance controller.
    made["psutil"] = _no_processes()
    made.update(cameras.modules())
    made.update(arms.modules())
    made.update(grippers.modules())
    return made


@contextlib.contextmanager
def mocked_sdks(*, extra: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Install the fake SDKs for the duration of the block.

    Args:
        extra: More modules to install, by dotted name.

    Yields:
        The installed modules, so a test can assert on what a part did to them.
    """
    made = sdk_modules()
    if extra:
        made.update(extra)

    saved = {name: sys.modules.get(name) for name in made}
    sys.modules.update(made)

    patches: list[tuple[Any, str, Any]] = []

    # A faked SDK lives in this process, and a part placed on a node is built
    # in another one that never saw it. Mocked runs therefore build every part
    # here: what they check is the code, not the cluster.
    from rlinf.robotics.placement import specs as _specs

    def _place_here(self):
        return self.part_cls.spawn(
            *self.args, node_rank=None, name=self.name, **self.kwargs
        )

    patches.append((_specs.PartSpec, "place", _specs.PartSpec.place))
    _specs.PartSpec.place = _place_here
    # Modules that start processes rather than only talking to an SDK. A fake
    # module in sys.modules arrives too late for these, because they import
    # psutil at module scope.
    processes = _no_processes()
    for dotted, attributes in {
        "rlinf.robotics.parts.arms.franka_ros": {"psutil": processes},
        "rlinf.robotics.parts.transports.ros.ros_controller": {"psutil": processes},
        **_PATCHES,
    }.items():
        target = sys.modules.get(dotted)
        if target is None:
            continue
        for name, value in attributes.items():
            patches.append((target, name, getattr(target, name, None)))
            setattr(target, name, value)

    try:
        yield made
    finally:
        for target, name, original in reversed(patches):
            if original is None:
                delattr(target, name)
            else:
                setattr(target, name, original)
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
