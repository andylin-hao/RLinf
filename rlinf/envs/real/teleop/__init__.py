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

"""Teleoperation devices an operator drives.

A leader arm, glove, motion controller, or spacemouse reads the operator rather
than the robot, and no policy ever sees one. That is why they are not
:class:`~rlinf.robotics.parts.base.RobotPart` implementations and are not
composed into a ``Robot``: a part answers what a component means to the policy,
and a teleop device has no such answer. Its output is an action for the
environment to apply, so it belongs on the environment side of the boundary.

The wrappers in :mod:`rlinf.envs.real.wrappers` turn a device reading into an
intervention. A device module itself only reads hardware, and stays free of
Gymnasium so the bench scripts in ``toolkits/realworld_check`` can drive one
directly.

Imports are left to callers: each device needs its own vendor package.
"""
