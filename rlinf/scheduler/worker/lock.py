# Copyright 2025 The RLinf Authors.
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


import errno
import fcntl
import os
import tempfile
from contextlib import AbstractContextManager
from typing import Optional

from ..manager import DeviceLockManager, PortLockManager
from .worker import Worker


class DeviceLock(AbstractContextManager):
    """The lock (can be used as a context manager like conventional locks) to manage accelerator device resources.

    When multiple workers run on the same accelerators, this lock ensures that only one worker can access the accelerator resources at a time.
    This is useful for preventing contention on device memory and computation resources, especially when multiple workers colocate on the same device.

    This class is the worker-side handle for the device lock, which interacts with a global lock manager to acquire and release locks on behalf of the worker.
    """

    def __init__(self, worker: Worker):
        """Initialize the device lock."""
        self._worker = worker
        self._lock_manager = DeviceLockManager.get_proxy()

    def acquire(self):
        """Lock accelerator devices for the current worker.

        This is useful for resource isolation, e.g., accelerator memory and computation resources, when multiple workers run on the same accelerators.

        Raises:
            RuntimeError: If the worker is not running in a worker context.
        """
        if self._worker is not None:
            self._lock_manager.acquire_devices(
                self._worker.worker_address, self._worker.global_accelerator_ids
            )
        else:
            raise RuntimeError("Cannot lock accelerators when not running in a worker.")

    def release(self):
        """Unlock accelerators for the current worker.

        Raises:
            RuntimeError: If the worker is not running in a worker context.
        """
        if self._worker is not None:
            self._lock_manager.release_devices(
                self._worker.worker_address, self._worker.global_accelerator_ids
            )
        else:
            raise RuntimeError(
                "Cannot unlock accelerators when not running in a worker."
            )

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.release()


class PortLock:
    """A global lock to manage network port resources.

    A reservation is taken in two places:

    1. The cluster's :class:`PortLockManager`, which serializes workers inside one Ray cluster.
    2. An ``flock`` on a file under ``RLINF_PORT_LOCK_DIR``, which extends the reservation to
       every RLinf process on the *host*. Two Ray clusters on one machine -- concurrent CI
       jobs being the common case -- each get their own manager but share a single kernel
       port space, so the manager alone cannot keep them apart.

    The ``flock`` is held through an open file descriptor for the lifetime of the worker
    process, so the kernel drops it when the process dies. No stale lock files need
    reaping, and a crashed worker cannot strand a port.
    """

    def __init__(self, worker: Worker):
        """Initialize the port lock."""
        self._worker = worker
        self._lock_manager = PortLockManager.get_proxy()
        # Keep the flock fds alive; closing them would release the locks.
        self._held_fds: dict[int, int] = {}

    @staticmethod
    def _lock_dir() -> str:
        from ..cluster import Cluster, ClusterEnvVar

        return os.getenv(
            f"{Cluster.SYS_NAME.upper()}_{ClusterEnvVar.PORT_LOCK_DIR.value}",
            os.path.join(tempfile.gettempdir(), "rlinf-port-locks"),
        )

    def _acquire_host_lock(self, port: int) -> tuple[bool, Optional[int]]:
        """Take a non-blocking ``flock`` on ``port``'s lock file.

        Args:
            port (int): The port to lock host-wide.

        Returns:
            tuple[bool, Optional[int]]: ``(False, None)`` if another process on this host
            holds the port. ``(True, fd)`` if the lock was taken; the caller owns ``fd``
            and must keep it open. ``(True, None)`` if the lock directory is unusable, in
            which case the reservation degrades to the cluster-scoped manager alone.
        """
        try:
            lock_dir = self._lock_dir()
            os.makedirs(lock_dir, exist_ok=True)
            fd = os.open(
                os.path.join(lock_dir, f"{port}.lock"), os.O_CREAT | os.O_RDWR, 0o666
            )
        except OSError:
            # An unwritable or unshared lock dir must not take down the run: degrade to
            # the cluster-scoped manager, which is what previous releases relied on.
            return True, None

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            os.close(fd)
            if err.errno in (errno.EACCES, errno.EAGAIN):
                return False, None
            # Filesystems without flock support (some network mounts) raise other errnos.
            return True, None
        return True, fd

    def acquire(self, port: int) -> bool:
        """Lock a network port for the current worker.

        This is useful for preventing port conflicts when multiple workers run on the same node.

        Args:
            port (int): The network port to lock.

        Returns:
            bool: True if the port is successfully locked, False otherwise.

        Raises:
            RuntimeError: If the worker is not running in a worker context.
        """
        if self._worker is None:
            raise RuntimeError("Cannot lock ports when not running in a worker.")

        if port in self._held_fds:
            # Already ours. Re-locking would open a second file description and the
            # non-blocking flock would fail against our own lock, reporting a conflict
            # with ourselves.
            return True

        locked, fd = self._acquire_host_lock(port)
        if not locked:
            return False

        if not self._lock_manager.acquire(
            self._worker._cluster_node_rank, self._worker._worker_name, port
        ):
            if fd is not None:
                os.close(fd)
            return False

        if fd is not None:
            self._held_fds[port] = fd
        return True

    def release(self, port: int) -> None:
        """Unlock a network port held by the current worker.

        Both halves of the reservation are dropped: the ``flock`` (by closing its file
        descriptor) and the cluster manager's entry.

        Args:
            port (int): The network port to unlock.

        Raises:
            RuntimeError: If the worker is not running in a worker context.
        """
        if self._worker is None:
            raise RuntimeError("Cannot unlock ports when not running in a worker.")

        fd = self._held_fds.pop(port, None)
        if fd is not None:
            os.close(fd)
        self._lock_manager.release(
            self._worker._cluster_node_rank, self._worker._worker_name, port
        )
