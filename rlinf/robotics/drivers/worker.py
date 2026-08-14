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

"""Hosting a driver in a scheduler worker, generically.

There is no per-robot worker class. For any driver, :func:`driver_worker_cls`
synthesizes ``type(name, (Worker, DriverCls), ...)`` once and caches it.
``WorkerGroup`` then binds every public method of that class as an RPC, so a
hosted driver exposes its whole surface -- the part-addressed calls *and*
off-interface methods like ``is_robot_up`` or ``reset_joint`` -- with no
delegation written by hand.

This module is the only place in ``rlinf.robotics`` that imports the scheduler
at module level. Driver implementations never do, which keeps them usable from
plain scripts and keeps the scheduler free of hardware dependencies.
"""

import inspect
from abc import ABCMeta
from typing import Any, Optional

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.scheduler.worker.worker import WorkerMeta

from .handle import RemoteDriverHandle, _first


class WorkerDriverMeta(WorkerMeta, ABCMeta):
    """Reconcile ``Worker``'s metaclass with the ``ABCMeta`` drivers carry.

    ``Worker`` uses ``WorkerMeta(type)`` and every driver is an ABC, so a class
    deriving from both needs a metaclass deriving from both.
    """


_WORKER_CLS_CACHE: dict[type, type] = {}


def driver_worker_cls(driver_cls: type) -> type:
    """Return (and cache) the ``Worker`` subclass that hosts ``driver_cls``."""
    cached = _WORKER_CLS_CACHE.get(driver_cls)
    if cached is not None:
        return cached

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        Worker.__init__(self)
        driver_cls.__init__(self, *args, **kwargs)
        self.connect()

    namespace: dict[str, Any] = {
        "__init__": __init__,
        "__module__": driver_cls.__module__,
        "__qualname__": f"{driver_cls.__name__}Worker",
        "__doc__": f"Scheduler host for :class:`{driver_cls.__name__}`.",
    }

    # Re-declare the driver's public methods in the new class body so
    # ``WorkerMeta`` wraps them for failure capture; inherited attributes are
    # invisible to it. This is a loop, not hand-written delegation, and the
    # bodies stay in the driver.
    for name, func in inspect.getmembers(driver_cls, inspect.isfunction):
        if not name.startswith("_") and name not in namespace:
            namespace[name] = func

    worker_cls = WorkerDriverMeta(
        f"{driver_cls.__name__}Worker",
        (Worker, driver_cls),
        namespace,
    )
    _WORKER_CLS_CACHE[driver_cls] = worker_cls
    return worker_cls


def spawn_driver_worker(
    driver_cls: type,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    node_rank: int,
    name: Optional[str] = None,
) -> RemoteDriverHandle:
    """Host one driver on ``node_rank`` and return a handle to it.

    Args:
        driver_cls: The driver class to construct on the target node.
        args: Positional arguments for the driver constructor.
        kwargs: Keyword arguments for the driver constructor.
        node_rank: Cluster node rank that is physically wired to the device.
        name: Worker-group name. Must be unique across concurrently running
            drivers; callers that spawn one driver per environment should
            include the environment index.

    Returns:
        RemoteDriverHandle: Handle whose parts proxy to the hosted driver.
    """
    worker_cls = driver_worker_cls(driver_cls)
    group = worker_cls.create_group(*args, **kwargs).launch(
        cluster=Cluster(),
        placement_strategy=NodePlacementStrategy(node_ranks=[node_rank]),
        name=name or f"{driver_cls.__name__}-node{node_rank}",
    )
    return RemoteDriverHandle(group, _first(group.describe_parts()))
