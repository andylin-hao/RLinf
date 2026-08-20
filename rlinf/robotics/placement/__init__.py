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

"""Opening a robot's connections where they belong, and reaching them there.

A part whose vendor SDK only exists on one machine cannot be built here and
moved. So a connection bound for another node is rebuilt there from the recipe
its constructor left behind, and what comes back is a handle that behaves the
same whether the part ended up local or remote.

:mod:`.plan` is the bookkeeping half -- what a connect opened, in what order,
and what to release if it fails -- and imports nothing from the scheduler.
:mod:`.handles` is the running half, and is the one module in
``rlinf.robotics`` allowed to import it.
"""

from .handles import (
    LocalPartHandle,
    PartHandle,
    PartWorkerHost,
    RemoteCamera,
    RemoteControllablePart,
    RemoteEndEffector,
    RemotePart,
    RemotePartHandle,
)
from .plan import Placement

__all__ = [
    "LocalPartHandle",
    "PartHandle",
    "Placement",
    "RemoteCamera",
    "RemoteControllablePart",
    "RemoteEndEffector",
    "RemotePart",
    "RemotePartHandle",
    "PartWorkerHost",
]
