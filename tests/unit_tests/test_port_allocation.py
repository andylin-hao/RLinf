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

"""Tests for port allocation and the host-wide port lock.

These cover the invariant that makes the allocation safe: a port handed out by
``find_free_port`` must sit outside the kernel's ephemeral range, so the kernel can
never assign it to an unrelated outbound connection between reservation and bind.
"""

import contextlib
import socket
from types import SimpleNamespace

import pytest

from rlinf.scheduler.cluster.cluster import Cluster, ClusterEnvVar
from rlinf.scheduler.worker.lock import PortLock

PORT_RANGE_ENV = f"{Cluster.SYS_NAME.upper()}_{ClusterEnvVar.PORT_RANGE.value}"
PORT_LOCK_DIR_ENV = f"{Cluster.SYS_NAME.upper()}_{ClusterEnvVar.PORT_LOCK_DIR.value}"


def _ephemeral_range() -> tuple[int, int] | None:
    """Return the kernel's ephemeral port range, or None if it is not readable."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as handle:
            low, high = handle.read().split()
        return int(low), int(high)
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def _bound(port: int):
    """Hold a real listening socket on ``port`` for the duration of the block."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
        sock.listen(1)
        yield
    finally:
        sock.close()


class TestPortRangeConfig:
    """Parsing and validation of ``RLINF_PORT_RANGE``."""

    def test_defaults_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(PORT_RANGE_ENV, raising=False)
        assert Cluster.get_port_range() == Cluster.default_port_range()

    def test_parses_configured_band(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PORT_RANGE_ENV, "21000-21500")
        assert Cluster.get_port_range() == (21000, 21500)

    @pytest.mark.parametrize(
        "raw", ["", "not-a-range", "21000", "21500-21000", "0-100", "1000-70000"]
    )
    def test_rejects_invalid_band(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv(PORT_RANGE_ENV, raw)
        if raw == "":
            # An empty value is treated as unset rather than as an error.
            assert Cluster.get_port_range() == Cluster.default_port_range()
        else:
            with pytest.raises(ValueError):
                Cluster.get_port_range()

    def test_default_band_avoids_ephemeral_range(self) -> None:
        """The default band must not overlap where the kernel draws source ports."""
        ephemeral = _ephemeral_range()
        if ephemeral is None:
            pytest.skip("Kernel ephemeral port range is not readable on this platform.")
        eph_low, eph_high = ephemeral
        if eph_low <= 1024 + Cluster.DEFAULT_PORT_RANGE_WIDTH:
            pytest.skip("Kernel leaves no usable window below its ephemeral range.")
        low, high = Cluster.default_port_range()
        assert high < eph_low or low > eph_high, (
            f"Default band {low}-{high} overlaps the ephemeral range {eph_low}-{eph_high}; "
            "allocated ports could be reassigned to outbound connections."
        )

    def test_default_band_tracks_a_retuned_kernel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A host with a lowered ephemeral range still gets a non-overlapping default."""
        monkeypatch.setattr(
            Cluster, "get_ephemeral_port_range", classmethod(lambda cls: (25000, 60999))
        )
        low, high = Cluster.default_port_range()
        assert high < 25000
        assert low >= 1024


class TestFindFreePort:
    """Allocation behavior of ``Cluster.find_free_port``."""

    def test_allocates_within_configured_band(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PORT_RANGE_ENV, "24000-24999")
        for _ in range(20):
            assert 24000 <= Cluster.find_free_port() <= 24999

    def test_allocates_outside_ephemeral_range_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ephemeral = _ephemeral_range()
        if ephemeral is None:
            pytest.skip("Kernel ephemeral port range is not readable on this platform.")
        monkeypatch.delenv(PORT_RANGE_ENV, raising=False)
        eph_low, eph_high = ephemeral
        port = Cluster.find_free_port()
        assert not (eph_low <= port <= eph_high), (
            f"find_free_port returned {port}, inside the ephemeral range "
            f"{eph_low}-{eph_high}; the kernel may reassign it before the caller binds."
        )

    def test_respects_max_port_num(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PORT_RANGE_ENV, "24000-24999")
        assert Cluster.find_free_port(max_port_num=24100) <= 24100

    def test_skips_a_port_that_is_actually_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PORT_RANGE_ENV, "24000-24999")
        taken = Cluster.find_free_port()
        with _bound(taken):
            # Narrow the band to the single taken port: it cannot be handed out, so
            # allocation must fall back to an ephemeral port rather than return it.
            monkeypatch.setenv(PORT_RANGE_ENV, f"{taken}-{taken}")
            for _ in range(5):
                assert Cluster.find_free_port() != taken

    def test_falls_back_when_band_is_unusable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A band above ``max_port_num`` leaves nothing to scan; allocation still succeeds."""
        monkeypatch.setenv(PORT_RANGE_ENV, "30000-30100")
        port = Cluster.find_free_port(max_port_num=25000)
        assert 0 < port <= 25000


class TestHostWidePortLock:
    """``PortLock``'s ``flock``, which spans Ray clusters sharing one host."""

    @staticmethod
    def _make_lock(tmp_path, monkeypatch: pytest.MonkeyPatch) -> PortLock:
        """Build a PortLock without Ray, since only the host-lock half is under test."""
        monkeypatch.setenv(PORT_LOCK_DIR_ENV, str(tmp_path))
        lock = object.__new__(PortLock)
        lock._worker = object()
        lock._lock_manager = None
        lock._held_fds = {}
        return lock

    def test_second_holder_is_refused(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = self._make_lock(tmp_path, monkeypatch)
        second = self._make_lock(tmp_path, monkeypatch)

        locked, fd = first._acquire_host_lock(24321)
        assert locked and fd is not None
        try:
            # A separate open file description on the same file conflicts, which is
            # exactly what a second Ray cluster on this host would hit.
            assert second._acquire_host_lock(24321) == (False, None)
        finally:
            import os

            os.close(fd)

    def test_lock_is_released_when_fd_closes(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        first = self._make_lock(tmp_path, monkeypatch)
        locked, fd = first._acquire_host_lock(24322)
        assert locked and fd is not None
        os.close(fd)

        second = self._make_lock(tmp_path, monkeypatch)
        locked_again, fd_again = second._acquire_host_lock(24322)
        assert locked_again and fd_again is not None
        os.close(fd_again)

    def test_degrades_when_lock_dir_is_unusable(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unwritable lock dir must not fail the reservation outright."""
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        monkeypatch.setenv(PORT_LOCK_DIR_ENV, str(blocker / "locks"))
        lock = object.__new__(PortLock)
        lock._worker = object()
        lock._lock_manager = None
        lock._held_fds = {}

        assert lock._acquire_host_lock(24323) == (True, None)

    def test_reacquiring_an_owned_port_succeeds(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-locking a port we already hold must not read as a conflict with ourselves."""
        import os

        lock = self._make_lock(tmp_path, monkeypatch)
        lock._worker = SimpleNamespace(_cluster_node_rank=0, _worker_name="w")
        lock._lock_manager = _AlwaysGrantingManager()
        try:
            assert lock.acquire(24324) is True
            assert lock.acquire(24324) is True
        finally:
            for fd in lock._held_fds.values():
                os.close(fd)

    def test_release_frees_the_port_for_another_holder(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After release, a different process-level holder can take the same port."""
        import os

        lock = self._make_lock(tmp_path, monkeypatch)
        lock._worker = SimpleNamespace(_cluster_node_rank=0, _worker_name="w")
        lock._lock_manager = _AlwaysGrantingManager()
        assert lock.acquire(24325) is True

        other = self._make_lock(tmp_path, monkeypatch)
        assert other._acquire_host_lock(24325) == (False, None)

        lock.release(24325)
        assert lock._held_fds == {}
        assert lock._lock_manager.released == [24325]

        locked, fd = other._acquire_host_lock(24325)
        assert locked and fd is not None
        os.close(fd)


class _AlwaysGrantingManager:
    """Stand-in for the Ray-side PortLockManager proxy."""

    def __init__(self) -> None:
        self.released: list[int] = []

    def acquire(self, node_rank: int, worker_name: str, port: int) -> bool:
        del node_rank, worker_name, port
        return True

    def release(self, node_rank: int, worker_name: str, port: int) -> None:
        del node_rank, worker_name
        self.released.append(port)


class _RecordingPortLock:
    """Port lock stub that records reservations and can refuse specific ports."""

    def __init__(self, refuse: tuple[int, ...] = ()) -> None:
        self.acquired: list[int] = []
        self.released: list[int] = []
        self._refuse = set(refuse)

    def acquire(self, port: int) -> bool:
        if port in self._refuse:
            return False
        self.acquired.append(port)
        return True

    def release(self, port: int) -> None:
        self.released.append(port)


class TestDerivedPortReservation:
    """``Worker.acquire_free_port`` must reserve the ports its caller derives."""

    @staticmethod
    def _worker(lock: _RecordingPortLock):
        from rlinf.scheduler.worker.worker import Worker

        worker = object.__new__(Worker)
        worker._port_lock = lock
        return worker

    def test_reserves_derived_offsets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rlinf.scheduler.worker.worker import Worker

        monkeypatch.setenv(PORT_RANGE_ENV, "24000-24999")
        lock = _RecordingPortLock()
        port = Worker.acquire_free_port(self._worker(lock), derived_offsets=(10000,))

        assert port in lock.acquired
        assert port + 10000 in lock.acquired, (
            "The derived port was not reserved, so another worker could be handed it."
        )

    def test_retries_when_a_derived_port_is_taken(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rlinf.scheduler.worker.worker import Worker

        monkeypatch.setenv(PORT_RANGE_ENV, "24000-24999")
        # Refuse one candidate's derived port; allocation must move to another base port.
        blocked_base = 24500
        lock = _RecordingPortLock(refuse=(blocked_base + 10000,))
        port = Worker.acquire_free_port(self._worker(lock), derived_offsets=(10000,))

        assert port != blocked_base
        assert port + 10000 in lock.acquired
        # Anything reserved for the rejected candidate must be handed back, otherwise a
        # repeatedly unlucky loop would drain the band.
        assert blocked_base not in lock.acquired or blocked_base in lock.released

    def test_no_offsets_reserves_only_the_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rlinf.scheduler.worker.worker import Worker

        monkeypatch.setenv(PORT_RANGE_ENV, "24000-24999")
        lock = _RecordingPortLock()
        port = Worker.acquire_free_port(self._worker(lock))

        assert lock.acquired == [port]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
