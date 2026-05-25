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

"""Tests for NvidiaGPUManager profiling methods in nvidia_gpu.py."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import rlinf.scheduler.hardware.accelerators.nvidia_gpu as nv_module
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager


@pytest.fixture(autouse=True)
def _reset_profiling_flag():
    """Reset module-level flag between tests for isolation."""
    nv_module._nv_profiling_active = False
    yield
    nv_module._nv_profiling_active = False


class TestStartStop:
    """NvidiaGPUManager.start_profiling / stop_profiling drive torch.cuda.profiler."""

    def test_start_sets_active_and_calls_cuda_profiler_start(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            NvidiaGPUManager.start_profiling(step_idx=5)
        assert NvidiaGPUManager.is_profiling_active() is True
        mock_start.assert_called_once_with()

    def test_stop_clears_active_and_calls_cuda_profiler_stop(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()
        with patch("torch.cuda.profiler.stop") as mock_stop:
            NvidiaGPUManager.stop_profiling()
        assert NvidiaGPUManager.is_profiling_active() is False
        mock_stop.assert_called_once_with()

    def test_double_start_is_idempotent(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            NvidiaGPUManager.start_profiling()
            NvidiaGPUManager.start_profiling()
        assert mock_start.call_count == 1

    def test_stop_without_start_is_noop(self):
        with patch("torch.cuda.profiler.stop") as mock_stop:
            NvidiaGPUManager.stop_profiling()
        mock_stop.assert_not_called()


class TestProfilingRangeSync:
    """NvidiaGPUManager.profiling_range on sync code is transparent off / wraps on."""

    def test_passes_through_when_inactive(self):
        results = []
        with NvidiaGPUManager.profiling_range("test/op"):
            results.append(42)
        assert results == [42]
        assert NvidiaGPUManager.is_profiling_active() is False

    def test_emits_range_when_active(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        with (
            patch.object(nv_module, "_nvtx") as mock_nvtx,
            patch.object(nv_module, "_NVTX_AVAILABLE", True),
        ):
            mock_nvtx.start_range.return_value = 1
            with NvidiaGPUManager.profiling_range("test/op", color="green"):
                pass

        mock_nvtx.start_range.assert_called_once_with(
            message="test/op", color="green", domain=None
        )
        mock_nvtx.end_range.assert_called_once_with(1)

    def test_ends_range_on_exception(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        with (
            patch.object(nv_module, "_nvtx") as mock_nvtx,
            patch.object(nv_module, "_NVTX_AVAILABLE", True),
        ):
            mock_nvtx.start_range.return_value = 42
            with pytest.raises(ValueError, match="oops"):
                with NvidiaGPUManager.profiling_range("test/op"):
                    raise ValueError("oops")

        mock_nvtx.end_range.assert_called_once_with(42)

    def test_no_nvtx_package_is_passthrough(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()
        with patch.object(nv_module, "_NVTX_AVAILABLE", False):
            with NvidiaGPUManager.profiling_range("test/op"):
                pass  # must not raise


class TestProfilingRangeAsync:
    """profiling_range works correctly inside async coroutines."""

    def test_passes_through_when_inactive(self):
        async def coro():
            with NvidiaGPUManager.profiling_range("test/async_op"):
                return 99

        assert asyncio.get_event_loop().run_until_complete(coro()) == 99

    def test_emits_range_when_active(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        async def coro():
            with NvidiaGPUManager.profiling_range("test/async_op"):
                return 7

        with (
            patch.object(nv_module, "_nvtx") as mock_nvtx,
            patch.object(nv_module, "_NVTX_AVAILABLE", True),
        ):
            mock_nvtx.start_range.return_value = 99
            result = asyncio.get_event_loop().run_until_complete(coro())

        assert result == 7
        mock_nvtx.start_range.assert_called_once_with(
            message="test/async_op", color=None, domain=None
        )
        mock_nvtx.end_range.assert_called_once_with(99)
