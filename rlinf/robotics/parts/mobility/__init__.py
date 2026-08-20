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

"""Bases that move the whole robot, and the drivers for them.

Only the category lives here so far. A wheeled or legged base implementation
goes beside it, the way a camera goes in :mod:`rlinf.robotics.parts.cameras`.
"""

from .base import MobileBase

__all__ = ["MobileBase"]
