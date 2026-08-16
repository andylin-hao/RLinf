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

"""Wrapper-stack builders shared by realworld task factories."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym

from rlinf.envs.real.episode import (
    KeyboardEvalControlWrapper,
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
    KeyboardRLTPolicySwitchWrapper,
    KeyboardStartEndWrapper,
)
from rlinf.envs.real.teleop.adapters import DualGelloJointStream
from rlinf.envs.real.teleop.config import resolve_teleop_device
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


def _teleop(env, cfg, devices):
    """Build a composed teleop, reporting a missing vendor package plainly."""
    from rlinf.envs.real.teleop.builder import build_teleop

    try:
        return build_teleop(env, cfg, devices)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "rlinf_dexhand":
            raise ModuleNotFoundError(
                "Dex-hand teleoperation requires optional dependency "
                "'rlinf_dexhand'. Install it before enabling it."
            ) from exc
        raise


def _apply_keyboard_wrapper(env: gym.Env, mode: Optional[str]) -> gym.Env:
    config = env.get_wrapper_attr("config")
    if config.is_dummy or not mode:
        return env
    if mode == "multi_stage":
        return KeyboardRewardDoneMultiStageWrapper(env)
    if mode == "single_stage":
        return KeyboardRewardDoneWrapper(env)
    if mode == "start_end":
        return KeyboardStartEndWrapper(env)
    if mode == "eval_control":
        return KeyboardEvalControlWrapper(env)
    if mode == "rlt_policy_switch":
        return KeyboardRLTPolicySwitchWrapper(env)
    return env


def apply_single_arm_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for single-arm realworld envs (franka single, xsquare)."""
    end_effector_type = str(
        getattr(getattr(env, "config", None), "end_effector_type", "franka_gripper")
    )
    is_dex_hand = end_effector_type.endswith("hand")

    no_gripper = cfg.get("no_gripper", True)
    if no_gripper and not is_dex_hand:
        env = GripperCloseEnv(env)

    device = resolve_teleop_device(
        cfg,
        supported=("spacemouse", "gello", "pico"),
        default="spacemouse",
    )
    use_spacemouse = device == "spacemouse"
    use_gello = device == "gello"
    use_pico = device == "pico"

    gripper_enabled = not no_gripper

    if not env.config.is_dummy and use_spacemouse:
        # A hand needs the glove alongside the mouse; a gripper does not.
        devices = ["spacemouse", "glove"] if is_dex_hand else ["spacemouse"]
        env = TeleopIntervention(env, _teleop(env, cfg, devices))

    if not env.config.is_dummy and use_gello:
        if is_dex_hand:
            raise ValueError("teleop_device: gello is not supported for ruiyan_hand.")
        env = TeleopIntervention(env, _teleop(env, cfg, ["gello"]))

    if not env.config.is_dummy and use_pico:
        if is_dex_hand:
            raise ValueError(
                "teleop_device: pico is not supported for dexterous hands."
            )
        pico_cfg = dict(cfg.get("pico", {}))
        env = TeleopIntervention(
            env, PicoTeleop(gripper_enabled=gripper_enabled, **pico_cfg)
        )

    env = _apply_keyboard_wrapper(env, cfg.get("keyboard_reward_wrapper", None))

    if cfg.get("use_relative_frame", True):
        env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    return env


def apply_dual_franka_joint_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    config = env.get_wrapper_attr("config")
    if cfg.get("no_gripper", True):
        # No DualGripperCloseEnv yet, so a 12D action would blow up as reshape(2,7).
        raise NotImplementedError(
            "no_gripper=True not supported for dual-arm envs (no DualGripperCloseEnv)."
        )

    # A dual-arm Franka has no single-arm Cartesian teleop path, so naming one
    # here is a mistake worth reporting rather than a setting to ignore.
    device = resolve_teleop_device(cfg, supported=("gello_joint", "pico"))
    use_gello_joint = device == "gello_joint"
    use_pico = device == "pico"

    if not config.is_dummy and use_gello_joint:
        left_port = cfg.get("left_gello_port", None)
        right_port = cfg.get("right_gello_port", None)
        if left_port is None or right_port is None:
            raise ValueError(
                "teleop_device: gello_joint requires both "
                "'left_gello_port' and 'right_gello_port' in the env config."
            )
        from rlinf.envs.real.teleop.builder import build_teleop

        use_delta = getattr(config, "joint_action_mode", None) == "delta"
        action_scale = getattr(config, "joint_action_scale", 0.1)
        direct_stream = bool(getattr(config, "teleop_direct_stream", False))
        streamer = (
            DualGelloJointStream(
                left_port=left_port,
                right_port=right_port,
                gripper_enabled=True,
                use_delta=use_delta,
                action_scale=action_scale,
                direct_stream=True,
                stream_period=cfg.get("gello_joint_stream_period", 0.001),
            )
            if direct_stream
            else None
        )
        teleop = build_teleop(
            env,
            cfg,
            [
                {
                    "gello_joint": {
                        "port": left_port,
                        "drives": "left",
                        "use_delta": use_delta,
                        "action_scale": action_scale,
                    }
                },
                {
                    "gello_joint": {
                        "port": right_port,
                        "drives": "right",
                        "use_delta": use_delta,
                        "action_scale": action_scale,
                    }
                },
            ],
        )
        teleop.streamer = streamer
        env = TeleopIntervention(env, teleop, mark_flag=True)

    if not config.is_dummy and use_pico:
        if getattr(env.unwrapped, "PER_ARM_ACTION_DIM", None) != 10:
            raise ValueError(
                "teleop_device: pico for dual-arm Franka is implemented for "
                "DualFrankaTcpEnv-v1 only. Use env/realworld_dual_franka_tcp_rot6d."
            )
        pico_cfg = dict(cfg.get("pico", {}))
        env = DualFrankaTcpPicoIntervention(
            env,
            gripper_enabled=True,
            **pico_cfg,
        )

    env = _apply_keyboard_wrapper(env, cfg.get("keyboard_reward_wrapper", None))
    return env
