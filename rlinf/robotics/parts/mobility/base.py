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

"""What carries the robot around: the base a policy drives."""

from rlinf.robotics.parts.base import ControllablePart


class MobileBase(ControllablePart):
    """A base the policy drives, whatever it rolls or walks on.

    Wheels, tracks and legs differ in what a command means -- a velocity, a
    gait -- and a driver says which in its :attr:`action_features`. They do not
    differ in what the robot tree needs to know, which is that this part moves
    the whole machine rather than something attached to it.

    There is deliberately no separate legged category. One existed, nothing
    used it, and it split a distinction that the action contract already makes
    more precisely.
    """
