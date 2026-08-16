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

"""Building the wrapper stack an env asks for.

There used to be one builder per arm count, each with an if-chain over device
names inside it. What varies between robots is not the procedure -- narrow the
action, hand it to an operator, let someone mark the episode, change the
representation -- but which pieces take part. So the env says which, and there
is one builder.
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym

from rlinf.envs.real.episode import (
    KeyboardEvalControlWrapper,
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
    KeyboardRLTPolicySwitchWrapper,
    KeyboardStartEndWrapper,
)
from rlinf.envs.real.teleop.adapters import DualGelloJointStream
from rlinf.envs.real.teleop.builder import build_teleop
from rlinf.envs.real.teleop.config import NO_DEVICE, resolve_teleop_device
from rlinf.envs.real.teleop.intervention import TeleopIntervention
from rlinf.envs.real.teleop.pico import (
    DualFrankaTcpPicoIntervention,
    PicoTeleop,
)
from rlinf.envs.real.transforms import (
    GripperCloseEnv,
    Quat2EulerWrapper,
    RelativeFrame,
)

#: Wrappers an env may name in ``ACTION_WRAPPERS`` or ``TRANSFORMS``. Named
#: rather than imported there so an env module does not depend on the wrappers
#: it happens to use.
WRAPPERS: dict[str, type] = {
    "GripperCloseEnv": GripperCloseEnv,
    "Quat2EulerWrapper": Quat2EulerWrapper,
    "RelativeFrame": RelativeFrame,
}

#: Keyboard modes, by the ``keyboard_reward_wrapper`` value that selects them.
KEYBOARD_MODES: dict[str, type] = {
    "multi_stage": KeyboardRewardDoneMultiStageWrapper,
    "single_stage": KeyboardRewardDoneWrapper,
    "start_end": KeyboardStartEndWrapper,
    "eval_control": KeyboardEvalControlWrapper,
    "rlt_policy_switch": KeyboardRLTPolicySwitchWrapper,
}


def _wanted(wrapper: type, cfg: Mapping[str, Any]) -> bool:
    """Whether ``cfg`` switches this wrapper on.

    A wrapper with no flag is always applied; one with a flag says its own
    name and default, so this does not grow a branch per wrapper.
    """
    flag = getattr(wrapper, "CONFIG_FLAG", None)
    if flag is None:
        return True
    return bool(cfg.get(flag, getattr(wrapper, "CONFIG_DEFAULT", True)))


def _teleop_entries(device: str, cfg: Mapping[str, Any], inner: Any) -> list[Any]:
    """The devices making up a rig, for the one named in the config."""
    if device == "gello_joint":
        use_delta = getattr(inner, "config", None) and (
            getattr(inner.config, "joint_action_mode", None) == "delta"
        )
        scale = getattr(getattr(inner, "config", None), "joint_action_scale", 0.1)
        shared = {"use_delta": bool(use_delta), "action_scale": scale}
        return [
            {"gello_joint": {"drives": "left", **shared}},
            {"gello_joint": {"drives": "right", **shared}},
        ]

    end_effector = str(
        getattr(getattr(inner, "config", None), "end_effector_type", "franka_gripper")
    )
    if device == "spacemouse" and end_effector.endswith("hand"):
        # A hand needs the glove alongside the mouse; a gripper does not.
        return ["spacemouse", "glove"]
    return [device]


def _apply_teleop(env: gym.Env, cfg: Mapping[str, Any], inner: Any) -> gym.Env:
    """Hand the action to an operator, if this env config asks for one."""
    device = resolve_teleop_device(
        cfg,
        supported=getattr(inner, "TELEOP", ()),
        default=getattr(inner, "TELEOP_DEFAULT", NO_DEVICE),
    )
    if device == NO_DEVICE or getattr(inner.config, "is_dummy", False):
        return env

    mark_flag = bool(getattr(inner, "TELEOP_MARK_FLAG", False))

    # PICO is still a device rather than a binding: its dual-arm TCP variant
    # carries rot6d hold logic and an API realworld_env finds by name.
    if device == "pico":
        pico_cfg = dict(cfg.get("pico", {}))
        if getattr(inner, "PER_ARM_ACTION_DIM", 0):
            if getattr(inner, "PER_ARM_ACTION_DIM", None) != 10:
                raise ValueError(
                    "teleop_device: pico for dual-arm Franka is implemented for "
                    "DualFrankaTcpEnv-v1 only. Use env/realworld_dual_franka_tcp_rot6d."
                )
            return DualFrankaTcpPicoIntervention(env, gripper_enabled=True, **pico_cfg)
        gripper_enabled = not bool(cfg.get("no_gripper", True))
        return TeleopIntervention(
            env, PicoTeleop(gripper_enabled=gripper_enabled, **pico_cfg)
        )

    teleop = build_teleop(env, cfg, _teleop_entries(device, cfg, inner))

    if device == "gello_joint" and getattr(inner.config, "teleop_direct_stream", False):
        teleop.streamer = DualGelloJointStream(
            left_port=cfg.get("left_gello_port"),
            right_port=cfg.get("right_gello_port"),
            gripper_enabled=True,
            use_delta=getattr(inner.config, "joint_action_mode", None) == "delta",
            action_scale=getattr(inner.config, "joint_action_scale", 0.1),
            direct_stream=True,
            stream_period=cfg.get("gello_joint_stream_period", 0.001),
        )
    return TeleopIntervention(env, teleop, mark_flag=mark_flag)


def build_stack(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrap ``env`` in what it declares and this config asks for.

    Order matters and is the same for every robot: narrow the action first so
    the operator drives what the policy drives, then teleop, then whoever marks
    the episode, then the representation the policy expects.
    """
    inner = env.unwrapped

    for flag in getattr(inner, "REFUSE_FLAGS", ()):
        if cfg.get(flag, False):
            raise NotImplementedError(
                f"{type(inner).__name__} does not support {flag!r}."
            )

    for name in getattr(inner, "ACTION_WRAPPERS", ()):
        wrapper = WRAPPERS[name]
        if _wanted(wrapper, cfg):
            env = wrapper(env)

    env = _apply_teleop(env, cfg, inner)

    mode = cfg.get("keyboard_reward_wrapper", None)
    if mode and not getattr(inner.config, "is_dummy", False):
        env = KEYBOARD_MODES[mode](env)

    for extra in getattr(inner, "episode_wrappers", lambda cfg: ())(cfg):
        env = extra(env)

    for name in getattr(inner, "TRANSFORMS", ()):
        wrapper = WRAPPERS[name]
        if _wanted(wrapper, cfg):
            env = wrapper(env)

    return env


#: The three per-robot builders were the same procedure with different pieces.
apply_single_arm_wrappers = build_stack
apply_dual_franka_joint_wrappers = build_stack
