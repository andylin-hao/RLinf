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

"""Registry for environment-specific teleoperation device backends.

Each backend constructs a device together with the binding that maps its
readings to the environment's action semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence

import gymnasium as gym

from rlinf.robotics.parts.teleop import (
    Glove,
    PicoController,
    SpaceMouse,
    TeleopLeaderArm,
)
from rlinf.robotics.teleop import (
    ActionKind,
    GloveBinding,
    LeaderArmBinding,
    LeaderJointBinding,
    PicoBinding,
    PicoTcpBinding,
    SpaceMouseBinding,
    TeleopEntry,
)

from .adapters import DualGelloJointStream
from .pico_config import split_dual_config
from .streaming import TeleopStreamer

#: The env config section a device reads shared options from, e.g. ``env.eval``.
EnvConfig = Mapping[str, Any]

#: One entry's own options, from the mapping form of a ``teleop`` list item.
DeviceOptions = Mapping[str, Any]


@dataclass(frozen=True)
class EnvFacts:
    """Action metadata required to construct teleoperation bindings.

    Attributes:
        layout: Slice occupied by each named action part.
        kinds: Semantic action type for each part.
        joint_action_scale: Divisor used to normalize joint deltas.
        direct_stream: Whether joint targets bypass ``step`` through a stream.
    """

    layout: Mapping[str, slice]
    kinds: Mapping[str, ActionKind]
    joint_action_scale: float = 0.1
    direct_stream: bool = False

    @classmethod
    def about(
        cls,
        env: gym.Env,
        layout: Mapping[str, slice],
        kinds: Mapping[str, ActionKind],
    ) -> "EnvFacts":
        """Build action metadata from an environment."""
        config = getattr(env.unwrapped, "config", None)
        return cls(
            layout=layout,
            kinds=kinds,
            joint_action_scale=float(getattr(config, "joint_action_scale", 0.1)),
            direct_stream=bool(getattr(config, "teleop_direct_stream", False)),
        )


class TeleopBackend(ABC):
    """Base class for registered teleoperation device builders.

    Example::

        @TeleopBackend.register("spacemouse")
        class SpaceMouseBackend(TeleopBackend): ...
    """

    #: Name to backend, filled by :meth:`register`.
    _REGISTRY: ClassVar[dict[str, type["TeleopBackend"]]] = {}

    @classmethod
    def register(cls, *names: str) -> Callable[[type], type]:
        """Register a backend under the names a config spells it with."""

        def add(backend: type) -> type:
            for name in names:
                key = name.lower()
                taken = TeleopBackend._REGISTRY.get(key)
                if taken is not None and taken is not backend:
                    raise ValueError(
                        f"Teleop device {name!r} is already registered to "
                        f"{taken.__name__}; {backend.__name__} cannot take it."
                    )
                TeleopBackend._REGISTRY[key] = backend
            return backend

        return add

    @classmethod
    def named(cls, name: str) -> type["TeleopBackend"]:
        """Return the backend registered under ``name``."""
        backend = cls._REGISTRY.get(str(name).lower())
        if backend is None:
            raise ValueError(f"Unknown teleop device {name!r}. Known: {cls.names()}.")
        return backend

    @classmethod
    def names(cls) -> list[str]:
        """Return registered backend names in sorted order."""
        return sorted(cls._REGISTRY)

    @classmethod
    @abstractmethod
    def entry(
        cls,
        cfg: EnvConfig,
        options: DeviceOptions,
        facts: EnvFacts,
    ) -> TeleopEntry:
        """Build a device and its environment-specific binding.

        Args:
            cfg: Environment configuration containing shared device options.
            options: Options for this device entry.
            facts: Action metadata declared by the environment.
        """

    @classmethod
    def streamer(
        cls,
        cfg: EnvConfig,
        facts: EnvFacts,
        entries: Sequence[TeleopEntry],
    ) -> Optional[TeleopStreamer]:
        """Build an optional direct-action streamer after all entries exist."""
        return None


@TeleopBackend.register("spacemouse")
class SpaceMouseBackend(TeleopBackend):
    """Build a SpaceMouse with Cartesian-delta and gripper bindings."""

    @classmethod
    def entry(
        cls,
        cfg: EnvConfig,
        options: DeviceOptions,
        facts: EnvFacts,
    ) -> TeleopEntry:
        return TeleopEntry(
            SpaceMouse(device_index=int(options.get("device_index", 0))),
            SpaceMouseBinding(),
            drives=options.get("drives"),
        )


@TeleopBackend.register("gello")
class GelloBackend(TeleopBackend):
    """Build a GELLO leader arm with Cartesian-pose input."""

    @classmethod
    def entry(
        cls,
        cfg: EnvConfig,
        options: DeviceOptions,
        facts: EnvFacts,
    ) -> TeleopEntry:
        port = options.get("port", cfg.get("gello_port"))
        if port is None:
            raise ValueError(
                "teleop device 'gello' requires 'gello_port' in the env config "
                "(e.g. env.eval.gello_port)."
            )
        return TeleopEntry(
            TeleopLeaderArm(port=port),
            LeaderArmBinding(),
            drives=options.get("drives"),
        )


@TeleopBackend.register("gello_joint")
class GelloJointBackend(TeleopBackend):
    """Build a GELLO leader arm with joint-space input."""

    @classmethod
    def entry(
        cls,
        cfg: EnvConfig,
        options: DeviceOptions,
        facts: EnvFacts,
    ) -> TeleopEntry:
        drives = options.get("drives")
        if drives is None:
            raise ValueError(
                "teleop device 'gello_joint' drives one arm, so it says which. "
                "List one entry per arm, e.g. teleop: [{gello_joint: {drives: "
                "left}}, {gello_joint: {drives: right}}]."
            )
        port = options.get("port") or cfg.get(f"{drives}_gello_port")
        if port is None:
            raise ValueError(
                "teleop device 'gello_joint' requires a 'port', or "
                f"'{drives}_gello_port' in the env config."
            )
        side = {"left": 0, "right": 1}.get(str(drives), 0)
        # Match the binding to absolute or delta joint semantics.
        arm = facts.kinds.get(f"{drives}.arm", facts.kinds.get("arm"))
        return TeleopEntry(
            TeleopLeaderArm(port=port, joint_space=True),
            LeaderJointBinding(
                side=side,
                use_delta=bool(options.get("use_delta", arm is ActionKind.JOINT_DELTA)),
                action_scale=float(
                    options.get("action_scale", facts.joint_action_scale)
                ),
            ),
            drives=drives,
        )

    @classmethod
    def streamer(
        cls,
        cfg: EnvConfig,
        facts: EnvFacts,
        entries: Sequence[TeleopEntry],
    ) -> Optional[TeleopStreamer]:
        """Build the optional 1 kHz dual-leader-arm streamer.

        The streamer reuses devices from ``entries`` to avoid opening each
        serial port twice.
        """
        if not facts.direct_stream:
            return None
        arms = {entry.drives: entry.device for entry in entries if entry.drives}
        missing = {"left", "right"} - set(arms)
        if missing:
            raise ValueError(
                "Direct-stream GELLO drives both arms from their leader arms, "
                f"so it needs an entry for each. Missing: {sorted(missing)}."
            )
        return DualGelloJointStream(
            left_arm=arms["left"],
            right_arm=arms["right"],
            gripper_enabled=True,
            use_delta=facts.kinds.get("left.arm") is ActionKind.JOINT_DELTA,
            action_scale=facts.joint_action_scale,
            direct_stream=True,
            stream_period=cfg.get("gello_joint_stream_period", 0.001),
        )


@TeleopBackend.register("glove")
class GloveBackend(TeleopBackend):
    """Build a glove device with a dexterous-hand binding."""

    @classmethod
    def entry(
        cls,
        cfg: EnvConfig,
        options: DeviceOptions,
        facts: EnvFacts,
    ) -> TeleopEntry:
        # Per-entry options override the shared glove configuration.
        glove_cfg = dict(cfg.get("glove_config", {}))
        glove_cfg.update(options)
        return TeleopEntry(
            Glove(
                left_port=glove_cfg.get("left_port", "/dev/ttyACM0"),
                right_port=glove_cfg.get("right_port"),
                frequency=int(glove_cfg.get("frequency", 60)),
                config_file=glove_cfg.get("config_file"),
            ),
            GloveBinding(),
            drives=options.get("drives"),
        )


@TeleopBackend.register("pico")
class PicoBackend(TeleopBackend):
    """Build a PICO controller with pose or delta semantics."""

    @classmethod
    def entry(
        cls,
        cfg: EnvConfig,
        options: DeviceOptions,
        facts: EnvFacts,
    ) -> TeleopEntry:
        drives = options.get("drives")
        pico_cfg = dict(cfg.get("pico", {}))
        hold = bool(
            options.get(
                "hold_current_when_inactive",
                pico_cfg.pop("hold_current_when_inactive", True),
            )
        )
        gripper = bool(options.get("gripper", not bool(cfg.get("no_gripper", True))))

        # Match the binding to the environment's Cartesian action semantics.
        arm = facts.kinds.get("arm" if drives is None else f"{drives}.arm")
        absolute = arm is ActionKind.CARTESIAN_POSE

        if drives in ("left", "right"):
            left_cfg, right_cfg = split_dual_config(pico_cfg)
            device_cfg = left_cfg if drives == "left" else right_cfg
            side = 0 if drives == "left" else 1
        else:
            device_cfg = {
                key: value
                for key, value in pico_cfg.items()
                if key not in ("left", "right")
            }
            device_cfg.setdefault("hand", "right")
            side = 0 if str(device_cfg["hand"]).lower() == "left" else 1

        binding = (
            PicoTcpBinding(gripper=gripper, side=side, hold_current_when_inactive=hold)
            if absolute
            else PicoBinding(gripper=gripper, side=side)
        )
        return TeleopEntry(PicoController(**device_cfg), binding, drives=drives)
