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

"""Running a connection on the node it belongs to.

A part whose vendor SDK only exists on one machine cannot be built here and
moved. So a connection bound for another node is rebuilt there from the recipe
its constructor left behind, and the object already in the robot's tree becomes
a view of it. Nothing is swapped, and a driver writes no remote counterpart:
both the worker and the view are derived from the driver class.

:mod:`.handles` is the whole of it, and is the one module in ``rlinf.robotics``
allowed to import the scheduler.
"""

from .handles import PartWorkerHost, host, remote_view_of, shutdown

__all__ = ["PartWorkerHost", "host", "remote_view_of", "shutdown"]
