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

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap

import pytest

import rlinf.scheduler.hardware.accelerators.nvidia_gpu as nvidia_gpu
import rlinf.utils.mujoco as mujoco_utils
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

_MANAGED_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "MUJOCO_GL",
    mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV,
    mujoco_utils.SCHEDULER_EGL_DEVICE_ID_ENV,
)


@pytest.fixture(autouse=True)
def _clean_mujoco_env(monkeypatch):
    for name in _MANAGED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    mujoco_utils.select_mujoco_egl_device.cache_clear()
    yield
    mujoco_utils.select_mujoco_egl_device.cache_clear()


def _become_accelerator_worker(monkeypatch, cuda_ordinal: str = "3") -> None:
    """Reproduce the environment the scheduler injects into a GPU worker."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_ordinal)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv(mujoco_utils.SCHEDULER_EGL_DEVICE_ID_ENV, cuda_ordinal)


def _probe(egl_index: int):
    return lambda: egl_index


def _forbidden_probe() -> int:
    raise AssertionError("the EGL device query must not run in this case")


def test_scheduler_does_not_hand_a_cuda_ordinal_to_mujoco(monkeypatch):
    monkeypatch.setattr(nvidia_gpu, "_torch_needs_avoid_record_streams", lambda: False)

    env_vars = NvidiaGPUManager.get_accelerator_env_var(["3"])

    assert env_vars["CUDA_VISIBLE_DEVICES"] == "3"
    assert env_vars[mujoco_utils.SCHEDULER_EGL_DEVICE_ID_ENV] == "3"
    # The CUDA ordinal is not a valid EGL index, so it must not reach MuJoCo
    # under the name MuJoCo actually reads.
    assert mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV not in env_vars


def test_worker_resolves_the_egl_index_of_its_own_gpu(monkeypatch):
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setattr(mujoco_utils, "_egl_index_of_local_cuda_device_zero", _probe(1))

    mujoco_utils.select_mujoco_egl_device()

    assert os.environ[mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV] == "1"


def test_explicit_device_wins_and_skips_the_query(monkeypatch):
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setenv(mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV, "7")
    monkeypatch.setattr(
        mujoco_utils, "_egl_index_of_local_cuda_device_zero", _forbidden_probe
    )

    mujoco_utils.select_mujoco_egl_device()

    assert os.environ[mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV] == "7"


def test_a_process_without_an_assigned_gpu_resolves_nothing(monkeypatch):
    # The driver sees every GPU. Resolving here would cache one node's EGL
    # ordering and leak it to workers on every other node.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(
        mujoco_utils, "_egl_index_of_local_cuda_device_zero", _forbidden_probe
    )

    mujoco_utils.select_mujoco_egl_device()

    assert mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV not in os.environ


@pytest.mark.parametrize("backend", ["osmesa", "glx", "OSMesa"])
def test_cpu_rendering_resolves_nothing(monkeypatch, backend):
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setenv("MUJOCO_GL", backend)
    monkeypatch.setattr(
        mujoco_utils, "_egl_index_of_local_cuda_device_zero", _forbidden_probe
    )

    mujoco_utils.select_mujoco_egl_device()

    assert mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV not in os.environ


@pytest.mark.parametrize("backend", ["", "glfw"])
def test_backends_robosuite_rewrites_to_egl_are_resolved(monkeypatch, backend):
    # robosuite 1.4.1 forces GPU rendering to EGL unless a CPU backend is named
    # explicitly, so anything else has to be treated as EGL.
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setenv("MUJOCO_GL", backend)
    monkeypatch.setattr(mujoco_utils, "_egl_index_of_local_cuda_device_zero", _probe(1))

    mujoco_utils.select_mujoco_egl_device()

    assert os.environ[mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV] == "1"


def test_query_failure_falls_back_to_the_cuda_ordinal(monkeypatch, caplog):
    _become_accelerator_worker(monkeypatch, "3")

    def failing_probe() -> int:
        raise RuntimeError("libEGL.so.1: cannot open shared object file")

    monkeypatch.setattr(
        mujoco_utils, "_egl_index_of_local_cuda_device_zero", failing_probe
    )

    with caplog.at_level(logging.WARNING, logger=mujoco_utils.logger.name):
        mujoco_utils.select_mujoco_egl_device()

    assert os.environ[mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV] == "3"
    assert "cannot open shared object file" in caplog.text


def test_resolved_index_is_logged(monkeypatch, caplog):
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setattr(mujoco_utils, "_egl_index_of_local_cuda_device_zero", _probe(1))

    with caplog.at_level(logging.INFO, logger=mujoco_utils.logger.name):
        mujoco_utils.select_mujoco_egl_device()

    assert "EGL index 1" in caplog.text


def test_the_query_runs_at_most_once_per_process(monkeypatch):
    _become_accelerator_worker(monkeypatch, "3")
    calls = []

    def counting_probe() -> int:
        calls.append(None)
        return 1

    monkeypatch.setattr(
        mujoco_utils, "_egl_index_of_local_cuda_device_zero", counting_probe
    )

    mujoco_utils.select_mujoco_egl_device()
    mujoco_utils.select_mujoco_egl_device()

    assert len(calls) == 1


def test_the_device_is_hidden_from_the_robosuite_import_check(monkeypatch):
    # robosuite 1.4.1 asserts MUJOCO_EGL_DEVICE_ID occurs in
    # CUDA_VISIBLE_DEVICES, which a correct EGL index generally does not.
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setattr(mujoco_utils, "_egl_index_of_local_cuda_device_zero", _probe(1))

    with mujoco_utils.mujoco_egl_device_selected():
        assert mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV not in os.environ

    assert os.environ[mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV] == "1"


def test_a_failed_import_still_restores_the_device(monkeypatch):
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setattr(mujoco_utils, "_egl_index_of_local_cuda_device_zero", _probe(1))

    with pytest.raises(RuntimeError, match="import failed"):
        with mujoco_utils.mujoco_egl_device_selected():
            raise RuntimeError("import failed")

    assert os.environ[mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV] == "1"


def test_nested_guards_restore_exactly_once(monkeypatch):
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setenv(mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV, "7")

    with mujoco_utils.mujoco_egl_device_selected():
        assert mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV not in os.environ
        with mujoco_utils.mujoco_egl_device_selected():
            assert mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV not in os.environ
        assert mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV not in os.environ

    assert os.environ[mujoco_utils.MUJOCO_EGL_DEVICE_ID_ENV] == "7"


def test_a_spawned_environment_subprocess_reuses_the_resolved_device(monkeypatch):
    # SubprocVectorEnv spawns its workers, so each child re-imports this module
    # with the parent's resolved value already in its environment.
    _become_accelerator_worker(monkeypatch, "3")
    monkeypatch.setattr(mujoco_utils, "_egl_index_of_local_cuda_device_zero", _probe(1))

    with mujoco_utils.mujoco_egl_device_selected():
        pass

    child_code = textwrap.dedent(
        """
        import os

        import rlinf.utils.mujoco as mujoco_utils

        def forbidden_probe():
            raise AssertionError("an inherited device must skip the query")

        mujoco_utils._egl_index_of_local_cuda_device_zero = forbidden_probe

        with mujoco_utils.mujoco_egl_device_selected():
            assert "MUJOCO_EGL_DEVICE_ID" not in os.environ
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
        """
    )
    subprocess.run(
        [sys.executable, "-c", child_code],
        check=True,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
