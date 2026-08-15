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

"""Raw readers for the hardware an operator drives.

These modules talk to a serial port, an HID device, or a headset, and nothing
else. They import no Gymnasium, so the bench scripts in
``toolkits/realworld_check`` can drive a device on a machine with no environment
and no cluster -- which is how you check a leader arm is wired correctly before
involving a robot at all.

Turning a reading into an action for an environment is the job of
:mod:`rlinf.envs.real.teleop.adapters`. Keeping that split is what lets the same
device serve a single-arm Cartesian env and a dual-arm joint-space one.

Imports are left to callers: each device needs its own vendor package.
"""
