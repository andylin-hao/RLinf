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

r"""Robot parts: what a component *is* to the policy.

:mod:`.base` holds the taxonomy. Each device category is defined beside the
drivers that implement it, so a subpackage owns its category and its hardware
together::

    parts/
      base.py                     Connection, RobotPart, ControllablePart,
                                  PartGroup
      arms/                       Franky, Franka ROS, GimArm, Turtle2, DOSW1
      cameras/        base.py:    Camera
                                  RealSense, ZED, Lumos
      end_effectors/  base.py:    EndEffector
        grippers/                 Franka, Robotiq
        hands/                    Ruiyan
      mobility/       base.py:    MobileBase
      transports/                 ROS

A part says what a component means to the policy: its observation contract and,
when controllable, its action contract. A link that presents several components
at once -- a dual-arm controller, a two-armed SDK session -- is a plain
:class:`~.base.Connection` and not a part: reading it would mean nothing, so it
lists what rides on it in :attr:`~.base.Connection.parts` and the robot composes
those. Either way it is a connection, so either way it takes a ``node_rank`` and
opens and closes the same way.

Subpackages are not imported here, and that is load-bearing rather than tidy: a
node needs only the vendor SDKs for the hardware it actually has, and importing
one subpackage's category would drag in every driver beside it. Reach a
category through its own subpackage -- ``from rlinf.robotics.parts.cameras
import Camera`` -- or through :mod:`rlinf.robotics`, which resolves names
lazily.
"""

from .base import (
    Connection,
    ControllablePart,
    PartGroup,
    RobotPart,
    run_parallel,
)

__all__ = [
    "Connection",
    "ControllablePart",
    "PartGroup",
    "RobotPart",
    "run_parallel",
]
