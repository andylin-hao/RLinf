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

"""Teleoperation devices.

A leader arm, glove, motion controller, or spacemouse is hardware the operator
drives, so it belongs with the other device drivers rather than with the
environments that happen to read it. These modules stay free of Gymnasium and
the scheduler like every other driver; the wrappers that turn their output into
interventions live in ``rlinf.envs.real.wrappers``.

Imports are left to callers: each device needs its own vendor package.
"""
