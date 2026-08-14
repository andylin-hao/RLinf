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

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from enum import Enum
from typing import Optional

from rlinf.utils.robosuite_compat import install_robosuite_egl_device_shim

# Must run before any simulator is imported, in worker processes and in the
# simulator subprocesses they spawn alike. See ``rlinf.utils.robosuite_compat``.
install_robosuite_egl_device_shim()


#: Environments used to sit directly under ``rlinf.envs``; simulated ones now
#: live under ``sim`` and real-world ones under ``real``. Environments are
#: selected by the ``env_type`` string, so configs are unaffected -- this only
#: covers code that imported a module path directly.
_MOVED_ENV_PACKAGES: dict[str, str] = {
    name: f"rlinf.envs.sim.{name}"
    for name in (
        "behavior",
        "calvin",
        "d4rl",
        "embodichain",
        "frankasim",
        "genesis",
        "habitat",
        "isaaclab",
        "libero",
        "maniskill",
        "metaworld",
        "polaris",
        "robocasa",
        "robocasa365",
        "robotwin",
        "roboverse",
        "world_model",
    )
}
_MOVED_ENV_PACKAGES["realworld"] = "rlinf.envs.real"


class _MovedModuleLoader(importlib.abc.Loader):
    """Bind an old module path to the module now living at the new path."""

    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec):
        """Return the module from its new location rather than a fresh one."""
        return importlib.import_module(self._target)

    def exec_module(self, module) -> None:
        """No-op: the target module executed when it was imported."""


class _MovedEnvFinder(importlib.abc.MetaPathFinder):
    """Resolve pre-split ``rlinf.envs.<name>`` paths to their new homes.

    Installed at the front of ``sys.meta_path``. It has to precede the path
    finder: once an old package name is aliased, its ``__path__`` points into
    the new directory, so the path finder would happily load a *second* copy of
    each submodule under the old name, and identity checks against the new name
    would fail. Names outside the moved set fall through untouched.

    Deprecated: it will be removed once downstream code has moved.
    """

    _PREFIX = "rlinf.envs."

    def find_spec(self, fullname: str, path=None, target=None) -> Optional[object]:
        """Map a moved module name onto a loader for its new location."""
        if not fullname.startswith(self._PREFIX):
            return None
        head, _, tail = fullname[len(self._PREFIX) :].partition(".")
        new_head = _MOVED_ENV_PACKAGES.get(head)
        if new_head is None:
            return None
        new_name = f"{new_head}.{tail}" if tail else new_head
        warnings.warn(
            f"{fullname!r} moved to {new_name!r}; update the import. "
            "The compatibility alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.util.spec_from_loader(fullname, _MovedModuleLoader(new_name))


if not any(isinstance(finder, _MovedEnvFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _MovedEnvFinder())


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    MANISKILL_RLT = "maniskill_rlt"
    LIBERO = "libero"
    ROBOTWIN = "robotwin"
    ISAACLAB = "isaaclab"
    METAWORLD = "metaworld"
    BEHAVIOR = "behavior"
    CALVIN = "calvin"
    ROBOCASA = "robocasa"
    ROBOCASA365 = "robocasa365"
    REALWORLD = "realworld"
    FRANKASIM = "frankasim"
    HABITAT = "habitat"
    OPENSORAWM = "opensora_wm"
    WANWM = "wan_wm"
    GENESIS = "genesis"
    EMBODICHAIN = "embodichain"
    ROBOVERSE = "roboverse"
    D4RL = "d4rl"
    POLARIS = "polaris"


def get_env_cls(env_type: str, env_cfg=None):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "maniskill", "libero", "isaaclab", etc.)
        env_cfg: Optional environment configuration. Required for "isaaclab" environment type.

    Returns:
        Environment class corresponding to the environment type.
    """

    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.MANISKILL:
        if env_cfg.get("enable_offload", False):
            from rlinf.envs.sim.maniskill.maniskill_offload_env import (
                ManiskillOffloadEnv,
            )

            return ManiskillOffloadEnv
        else:
            from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

            return ManiskillEnv
    elif env_type == SupportedEnvType.MANISKILL_RLT:
        from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv

        return ManiskillRLTEnv
    elif env_type == SupportedEnvType.LIBERO:
        from rlinf.envs.sim.libero.libero_env import LiberoEnv

        return LiberoEnv
    elif env_type == SupportedEnvType.ROBOTWIN:
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

        return RoboTwinEnv
    elif env_type == SupportedEnvType.ISAACLAB:
        from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS

        if env_cfg is None:
            raise ValueError(
                "env_cfg is required for isaaclab environment type. "
                "Please provide env_cfg.init_params.id to select the task."
            )

        task_id = env_cfg.init_params.id
        assert task_id in REGISTER_ISAACLAB_ENVS, (
            f"Task type {task_id} has not been registered! "
            f"Available tasks: {list(REGISTER_ISAACLAB_ENVS.keys())}"
        )
        return REGISTER_ISAACLAB_ENVS[task_id]
    elif env_type == SupportedEnvType.METAWORLD:
        from rlinf.envs.sim.metaworld.metaworld_env import MetaWorldEnv

        return MetaWorldEnv
    elif env_type == SupportedEnvType.BEHAVIOR:
        from rlinf.envs.sim.behavior.behavior_env import BehaviorEnv

        return BehaviorEnv
    elif env_type == SupportedEnvType.CALVIN:
        from rlinf.envs.sim.calvin.calvin_gym_env import CalvinEnv

        return CalvinEnv
    elif env_type == SupportedEnvType.ROBOCASA:
        from rlinf.envs.sim.robocasa.robocasa_env import RobocasaEnv

        return RobocasaEnv
    elif env_type == SupportedEnvType.ROBOCASA365:
        from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env

        return Robocasa365Env
    elif env_type == SupportedEnvType.REALWORLD:
        from rlinf.envs.real import RealWorldEnv

        return RealWorldEnv
    elif env_type == SupportedEnvType.HABITAT:
        from rlinf.envs.sim.habitat.habitat_env import HabitatEnv

        return HabitatEnv
    elif env_type == SupportedEnvType.FRANKASIM:
        from rlinf.envs.sim.frankasim.frankasim_env import FrankaSimEnv

        return FrankaSimEnv
    elif env_type == SupportedEnvType.GENESIS:
        from rlinf.envs.sim.genesis.genesis_env import GenesisEnv

        return GenesisEnv
    elif env_type == SupportedEnvType.OPENSORAWM:
        from rlinf.envs.sim.world_model.world_model_opensora_env import OpenSoraEnv

        return OpenSoraEnv
    elif env_type == SupportedEnvType.WANWM:
        from rlinf.envs.sim.world_model.world_model_wan_env import WanEnv

        return WanEnv
    elif env_type == SupportedEnvType.EMBODICHAIN:
        from rlinf.envs.sim.embodichain.embodichain_env import EmbodiChainEnv

        return EmbodiChainEnv
    elif env_type == SupportedEnvType.ROBOVERSE:
        from rlinf.envs.sim.roboverse.roboverse_env import RoboVerseEnv

        return RoboVerseEnv
    elif env_type == SupportedEnvType.D4RL:
        from rlinf.envs.sim.d4rl.d4rl_env import D4RLEnv

        return D4RLEnv
    elif env_type == SupportedEnvType.POLARIS:
        from rlinf.envs.sim.polaris.polaris_env import PolarisEnv

        return PolarisEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
