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

"""The teleop devices a config can name, and how each is built for an env.

One class per name, registering itself the way a robotics driver does. A device
is a pairing rather than a single object -- the hardware, and the binding that
says what its numbers mean for this robot -- so what registers here is the
knowledge of how to make that pair, not either half of it.

That knowledge is env-layer: it reads the env config, and it asks the env what
its arm command means. Which is why this lives here and not beside the device
classes in :mod:`rlinf.robotics.parts.teleop`, where an env config has no
business being.
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


@dataclass(frozen=True)
class EnvFacts:
    """What the env tells a device backend about the robot it will drive.

    Attributes:
        layout: Where each named part sits in the action vector.
        kinds: What each part's numbers mean, which is what decides whether a
            device can drive it at all.
        joint_action_scale: The divisor turning a joint delta into a normalized
            action. Belongs to the robot rather than to the operator.
        direct_stream: Whether this env wants joint targets pushed on their own
            thread rather than dispatched by ``step``.
    """

    layout: Mapping[str, slice]
    kinds: Mapping[str, ActionKind]
    joint_action_scale: float = 0.1
    direct_stream: bool = False

    @classmethod
    def about(cls, env: gym.Env, layout, kinds) -> "EnvFacts":
        """Read the facts a device backend may ask for off an env."""
        config = getattr(env.unwrapped, "config", None)
        return cls(
            layout=layout,
            kinds=kinds,
            joint_action_scale=float(getattr(config, "joint_action_scale", 0.1)),
            direct_stream=bool(getattr(config, "teleop_direct_stream", False)),
        )


class TeleopBackend(ABC):
    """One device a config can name, and how to build it for this env.

    Registering is a decorator in the file that implements the backend, the way
    a camera or an arm driver registers::

        @TeleopBackend.register("spacemouse")
        class SpaceMouseBackend(TeleopBackend): ...


        TeleopBackend.named("spacemouse")  # the class
        TeleopBackend.names()  # every name a config may use

    Adding a device used to mean an entry in one table, sometimes a second
    entry in another table for the thread it streams on, and a module-level
    function far from either -- three edits in a file that knows about every
    device, for something that belongs to one.
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
        """The backend a config name selects, or a list of what exists."""
        backend = cls._REGISTRY.get(str(name).lower())
        if backend is None:
            raise ValueError(f"Unknown teleop device {name!r}. Known: {cls.names()}.")
        return backend

    @classmethod
    def names(cls) -> list[str]:
        """Every name a config may use, sorted."""
        return sorted(cls._REGISTRY)

    @classmethod
    @abstractmethod
    def entry(
        cls,
        cfg: Mapping[str, Any],
        options: Mapping[str, Any],
        facts: EnvFacts,
    ) -> TeleopEntry:
        """Build this device and the binding that says what it means.

        Args:
            cfg: The env config section, for options devices share.
            options: This entry's own options from the config.
            facts: What the env says about the robot being driven.
        """

    @classmethod
    def streamer(
        cls,
        cfg: Mapping[str, Any],
        facts: EnvFacts,
        entries: Sequence[TeleopEntry],
    ) -> Optional[Any]:
        """The thread this device also commands the robot through, if any.

        Most devices command through the group, so the default is none. One
        that overrides this is saying that composition alone does not describe
        it: the action is one thing, the rate it is delivered at another.

        It is asked after every entry exists, because a streamer drives the
        devices the group composed rather than opening its own.
        """
        return None


@TeleopBackend.register("spacemouse")
class SpaceMouseBackend(TeleopBackend):
    """A six-axis puck: the twist drives the arm, the buttons latch the grip."""

    @classmethod
    def entry(cls, cfg, options, facts) -> TeleopEntry:
        return TeleopEntry(
            SpaceMouse(device_index=int(options.get("device_index", 0))),
            SpaceMouseBinding(),
            drives=options.get("drives"),
        )


@TeleopBackend.register("gello")
class GelloBackend(TeleopBackend):
    """A leader arm the operator poses, read as a Cartesian target."""

    @classmethod
    def entry(cls, cfg, options, facts) -> TeleopEntry:
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
    """The same leader arm, read as joint targets for one arm of the robot."""

    @classmethod
    def entry(cls, cfg, options, facts) -> TeleopEntry:
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
        # Whether the env reads a target or a change is what it declared its
        # arm command to be; there is no second place to ask.
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
    def streamer(cls, cfg, facts, entries) -> Optional[Any]:
        """The 1 kHz thread a pair of leader arms uses, when this env asks.

        Follower tracking is unstable at the policy's step rate, so the joint
        targets go straight to the controllers on their own thread.

        It streams the arms the group already composed rather than opening its
        own. Two readers on one serial port is at best two pollers competing
        for it, and the second pair would be built from the config's global
        ports -- so a per-entry ``port`` override would drive the action and be
        ignored by the stream.
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
    """A pair of gloves, read as finger angles for a dexterous hand."""

    @classmethod
    def entry(cls, cfg, options, facts) -> TeleopEntry:
        glove_cfg = dict(cfg.get("glove", {}))
        return TeleopEntry(
            Glove(
                left_port=glove_cfg.get("left_port"),
                right_port=glove_cfg.get("right_port"),
                frequency=int(glove_cfg.get("frequency", 60)),
                config_file=glove_cfg.get("config_file"),
            ),
            GloveBinding(),
            drives=options.get("drives"),
        )


@TeleopBackend.register("pico")
class PicoBackend(TeleopBackend):
    """A VR controller, bound to a pose or to a delta as the env asks."""

    @classmethod
    def entry(cls, cfg, options, facts) -> TeleopEntry:
        drives = options.get("drives")
        pico_cfg = dict(cfg.get("pico", {}))
        hold = bool(
            options.get(
                "hold_current_when_inactive",
                pico_cfg.pop("hold_current_when_inactive", True),
            )
        )
        gripper = bool(options.get("gripper", not bool(cfg.get("no_gripper", True))))

        # The env says whether its arm takes a pose to reach or a delta to
        # apply.
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
