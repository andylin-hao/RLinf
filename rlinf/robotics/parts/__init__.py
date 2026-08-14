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
      base.py                     RobotPart, ControllablePart, Camera,
                                  EndEffector, Arm, MobileBase, LeggedBase
      cameras/                    RealSense, ZED, Lumos
      end_effectors/
        grippers/                 Franka, Robotiq
        hands/                    Ruiyan

A part says what a component means to the policy -- its observation and action
contract. How that contract is fulfilled over a wire is a
:class:`~rlinf.robotics.drivers.base.Driver`'s concern, and one driver may back
several parts.

Subpackages are not imported here: a node needs only the vendor SDKs for the
hardware it actually has. Import ``rlinf.robotics.parts.cameras`` directly.
"""

from .base import (
    Arm,
    Camera,
    ControllablePart,
    EndEffector,
    LeggedBase,
    MobileBase,
    RobotPart,
    run_parallel,
)

__all__ = [
    "Arm",
    "Camera",
    "ControllablePart",
    "EndEffector",
    "LeggedBase",
    "MobileBase",
    "RobotPart",
    "run_parallel",
]
