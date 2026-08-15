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

"""Robot parts: what a component *is* to the policy.

The taxonomy lives in :mod:`.base`; each category's implementations live in the
subpackage named after it::

    parts/
      base.py                     RobotPart, ControllablePart, Group, Camera,
                                  EndEffector, MobileBase, LeggedBase
      arms/                       Franky, Franka ROS, GimArm, Turtle2, DOSW1
      cameras/                    RealSense, ZED, Lumos
      end_effectors/
        grippers/                 Franka, Robotiq
        hands/                    Ruiyan
      transports/                 ROS

A part says what a component means to the policy: its observation and action
contract. Hardware that presents several components over one connection -- a
dual-arm controller, a two-armed SDK session -- exposes them through
:meth:`~.base.RobotPart.subparts`, so "owns a connection" is a property some
parts have rather than a separate kind of thing.

Subpackages are not imported here: a node needs only the vendor SDKs for the
hardware it actually has. Import ``rlinf.robotics.parts.cameras`` directly.
"""

from .base import (
    Camera,
    ControllablePart,
    EndEffector,
    Group,
    LeggedBase,
    MobileBase,
    RobotPart,
    run_parallel,
)

__all__ = [
    "Group",
    "Camera",
    "ControllablePart",
    "EndEffector",
    "LeggedBase",
    "MobileBase",
    "RobotPart",
    "run_parallel",
]
