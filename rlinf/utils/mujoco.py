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

"""MuJoCo EGL device selection.

CUDA ordinals and EGL device indices are different namespaces. A CUDA ordinal
covers only the GPUs allocated to this container, renumbered from zero in PCI
order; EGL enumeration is not namespaced by the container at all and lists every
device the driver can see, in driver order. On one Kubernetes node holding eight
GPUs, a container given four of them saw CUDA devices 0-3 as EGL indices 2, 3, 0
and 1, among nine enumerated EGL devices.

Passing a CUDA ordinal as ``MUJOCO_EGL_DEVICE_ID`` therefore renders on a GPU
the worker holds no CUDA context for. Usually this is silent and merely wrong;
under contention MuJoCo aborts inside ``mjr_readPixels`` and the parent process
only sees the environment subprocess disappear.

Leaving ``MUJOCO_EGL_DEVICE_ID`` unset does not avoid this. robosuite 1.4.1
falls back to ``CUDA_VISIBLE_DEVICES`` and uses that as an EGL index instead::

    selected_device = (
        os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if os.environ.get("MUJOCO_EGL_DEVICE_ID", None) is None
        else os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    )

The variable must therefore be set, and set to a real EGL index.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

logger = logging.getLogger(__name__)

#: Set by the scheduler on accelerator workers. Its presence marks a process
#: that owns a GPU; its value is that GPU's CUDA ordinal, kept only as a
#: fallback for nodes where the EGL device query is unavailable.
SCHEDULER_EGL_DEVICE_ID_ENV = "RLINF_DEFAULT_MUJOCO_EGL_DEVICE_ID"

MUJOCO_EGL_DEVICE_ID_ENV = "MUJOCO_EGL_DEVICE_ID"

_EGL_CUDA_DEVICE_NV = 0x323A
_MAX_EGL_DEVICES = 64

# robosuite 1.4.1 rewrites MUJOCO_GL to "egl" for anything that is not one of
# these explicit CPU backends, so an unset or unrelated value still means EGL.
_CPU_RENDERING_BACKENDS = frozenset({"glx", "osmesa"})


def _egl_index_of_local_cuda_device_zero() -> int:
    """Return the EGL enumeration index of this process's first CUDA device.

    ``eglQueryDevicesEXT`` lists every device on the node and is not filtered
    by ``CUDA_VISIBLE_DEVICES``, whereas ``EGL_CUDA_DEVICE_NV`` is readable
    only for CUDA-visible devices and reports the CUDA *local* ordinal. Local
    ordinal zero is therefore the device this worker was assigned.

    The query goes through ``ctypes`` rather than PyOpenGL because PyOpenGL
    resolves extension entry points by calling ``eglQueryString`` on a display,
    and obtaining the right display is what the device index is needed for.

    Returns:
        The EGL enumeration index to use as ``MUJOCO_EGL_DEVICE_ID``.

    Raises:
        RuntimeError: If the EGL device extensions are unavailable, the query
            fails, or no EGL device reports CUDA local ordinal zero.
        OSError: If ``libEGL.so.1`` cannot be loaded.
    """
    libegl = ctypes.CDLL("libEGL.so.1")
    libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    libegl.eglGetProcAddress.restype = ctypes.c_void_p

    device_t = ctypes.c_void_p
    boolean_t = ctypes.c_uint
    int_t = ctypes.c_int
    attrib_t = ctypes.c_ssize_t

    query_devices_ptr = libegl.eglGetProcAddress(b"eglQueryDevicesEXT")
    query_attrib_ptr = libegl.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not query_devices_ptr or not query_attrib_ptr:
        raise RuntimeError(
            "EGL_EXT_device_enumeration and EGL_EXT_device_query are required"
        )
    query_devices = ctypes.CFUNCTYPE(
        boolean_t, int_t, ctypes.POINTER(device_t), ctypes.POINTER(int_t)
    )(query_devices_ptr)
    query_attrib = ctypes.CFUNCTYPE(
        boolean_t, device_t, int_t, ctypes.POINTER(attrib_t)
    )(query_attrib_ptr)

    devices = (device_t * _MAX_EGL_DEVICES)()
    device_count = int_t()
    if not query_devices(_MAX_EGL_DEVICES, devices, ctypes.byref(device_count)):
        raise RuntimeError("eglQueryDevicesEXT failed")

    for egl_index in range(device_count.value):
        cuda_ordinal = attrib_t(-1)
        queried = query_attrib(
            devices[egl_index], _EGL_CUDA_DEVICE_NV, ctypes.byref(cuda_ordinal)
        )
        if queried and cuda_ordinal.value == 0:
            return egl_index
    raise RuntimeError(
        f"none of the {device_count.value} EGL devices reports CUDA local ordinal 0"
    )


def _renders_with_egl() -> bool:
    """Report whether robosuite will end up rendering through EGL."""
    backend = os.environ.get("MUJOCO_GL", "").strip().lower()
    return backend not in _CPU_RENDERING_BACKENDS


@cache
def select_mujoco_egl_device() -> None:
    """Point ``MUJOCO_EGL_DEVICE_ID`` at the EGL index of the assigned GPU.

    Idempotent, and a no-op unless this process is an accelerator worker that
    renders through EGL and has no device pinned already.
    """
    if MUJOCO_EGL_DEVICE_ID_ENV in os.environ:
        # Either explicit user configuration, or a value this function already
        # resolved in the parent and a spawned subprocess inherited.
        return
    fallback = os.environ.get(SCHEDULER_EGL_DEVICE_ID_ENV)
    if fallback is None:
        # Not an accelerator worker. Resolving on, say, the driver would cache
        # one node's EGL ordering and leak it to workers on every other node.
        return
    if not _renders_with_egl():
        return

    try:
        egl_device_id = str(_egl_index_of_local_cuda_device_zero())
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "Could not map CUDA local ordinal 0 to an EGL device (%s). Falling "
            "back to the scheduler-assigned CUDA ordinal %s, which renders on "
            "the wrong GPU when the two namespaces disagree.",
            exc,
            fallback,
        )
        egl_device_id = fallback
    else:
        logger.info(
            "MuJoCo EGL device: CUDA_VISIBLE_DEVICES=%s local ordinal 0 -> EGL "
            "index %s",
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            egl_device_id,
        )
    os.environ[MUJOCO_EGL_DEVICE_ID_ENV] = egl_device_id


@contextmanager
def mujoco_egl_device_selected() -> Iterator[None]:
    """Select the EGL device around an import that pulls in robosuite.

    robosuite 1.4.1 asserts at import time that ``MUJOCO_EGL_DEVICE_ID`` occurs
    as a substring of ``CUDA_VISIBLE_DEVICES``, which is wrong whenever the EGL
    and CUDA namespaces disagree. The variable is hidden for the duration of
    the import and restored before robosuite creates a GL context.

    Nesting is safe: an inner scope finds nothing to save and leaves the
    restore to the outer one.
    """
    select_mujoco_egl_device()
    saved = os.environ.pop(MUJOCO_EGL_DEVICE_ID_ENV, None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ[MUJOCO_EGL_DEVICE_ID_ENV] = saved
