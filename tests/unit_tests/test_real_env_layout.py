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

"""Each robot folder holds its tasks, and every task still builds."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_REAL = _ROOT / "rlinf" / "envs" / "real"
_ROBOTS = ("franka", "dosw1", "gim_arm", "xsquare")

#: Every gym id the real-world envs register. These are written into user
#: configs and dataset metadata, so renaming one silently breaks both.
EXPECTED_IDS = {
    "FrankaEnv-v1",
    "PegInsertionEnv-v1",
    "FrankaBinRelocationEnv-v1",
    "BottleEnv-v1",
    "DexpnpEnv-v1",
    "DualFrankaJointEnv-v1",
    "DualFrankaTCPEnv-v1",
    "DOSW1PickEnv-v1",
    "ButtonEnv-v1",
    "GimArmPegInsertionEnv-v1",
}


def test_no_robot_keeps_a_tasks_subpackage():
    """Tasks sit at the top of their robot folder, not one level down."""
    leftovers = [name for name in _ROBOTS if (_REAL / name / "tasks").exists()]

    assert leftovers == []


def test_every_robot_folder_has_a_base():
    """Shared machinery is named so nobody mistakes it for a task."""
    missing = [name for name in _ROBOTS if not (_REAL / name / "base.py").exists()]

    assert missing == []


def test_all_task_ids_are_registered():
    from gymnasium.envs.registration import registry

    import rlinf.envs.real  # noqa: F401  (registration happens on import)

    assert EXPECTED_IDS <= set(registry)


def test_every_entry_point_resolves():
    """A registered id whose entry point cannot be imported fails at rollout."""
    from gymnasium.envs.registration import registry

    import rlinf.envs.real  # noqa: F401

    unresolved = []
    for env_id in sorted(EXPECTED_IDS):
        entry_point = registry[env_id].entry_point
        module_name, _, attribute = str(entry_point).partition(":")
        module = importlib.import_module(module_name)
        if getattr(module, attribute, None) is None:
            unresolved.append(f"{env_id} -> {entry_point}")

    assert unresolved == []


def test_task_tables_cover_the_wrapped_robots():
    """The TASKS table is the one place a robot's tasks are declared."""
    from rlinf.envs.real import dosw1, franka, xsquare

    declared = set(franka.TASKS) | set(dosw1.TASKS) | set(xsquare.TASKS)

    # GimArm registers its env class directly, with no wrapper stack.
    assert declared == EXPECTED_IDS - {"GimArmPegInsertionEnv-v1"}


def test_pose_math_is_not_filed_under_a_robot():
    """construct_adjoint_matrix is SE(3) math the wrappers share.

    Leaving it in the Franka package made every wrapper importing it pull that
    package in, which is what turned the task registry into an import cycle.
    """
    from rlinf.envs.real import pose_utils

    assert hasattr(pose_utils, "construct_adjoint_matrix")
    assert not (_REAL / "franka" / "utils.py").exists()
