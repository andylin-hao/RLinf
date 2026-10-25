#!/usr/bin/env python3
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

"""Drive or verify a Robbyant R2 robot through ``R2ReachEnv``.

Examples::

    # Run against fake RFrame SDKs (no hardware required)
    python -m toolkits.realworld_check.test_r2_env --mock --steps 5

    # Connect to a real R2 robot and inspect observations without moving
    python -m toolkits.realworld_check.test_r2_env --no-move

    # Interactive CLI on a real or mock R2 robot
    python -m toolkits.realworld_check.test_r2_env --mock --interactive
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rlinf.robotics.parts.arms.r2 import (
    R2_LEFT_ARM_LIMITS_HIGH,
    R2_LEFT_ARM_LIMITS_LOW,
    R2_LEFT_READY_QPOS,
    R2_NECK_LIMITS_HIGH,
    R2_NECK_LIMITS_LOW,
    R2_RIGHT_ARM_LIMITS_HIGH,
    R2_RIGHT_ARM_LIMITS_LOW,
    R2_RIGHT_READY_QPOS,
    R2_WAIST_LIMITS_HIGH,
    R2_WAIST_LIMITS_LOW,
    compute_r2_arm_cartesian_step,
    compute_r2_arm_fk_and_jacobian,
)

HELP = """
  === Arm 3D Cartesian Control (moves hand in base frame, default 3 cm) ===
  l|r <up|down|fwd|back|left|right> [cm]    e.g. "r fwd 4", "r up 4", "l left 3"

  === Arm Intuitive Joint Control (symmetric left/right, default --step deg) ===
  l|r <raise|lower> [mult]                  raise/lower shoulder forward/back (J1)
  l|r <out|in> [mult]                       open/close shoulder outward/inward (J2)
  l|r <twist_out|twist_in> [mult]           rotate upper arm outward/inward (J3)
  l|r <bend|extend> [mult]                  bend/straighten elbow (J4)
  l|r <wrist_up|wrist_down> [mult]          pitch wrist up/down (J6)
  l|r <roll_cw|roll_ccw> [mult]             roll wrist clockwise/counter-clockwise (J7)
  left|right <1..7> [mult]                  raw joint step (J1..J7)

  === Gripper Control ===
  open / close                              open or close both grippers
  l|r <open|close>                          open or close left or right gripper only

  === Waist Control (5-DoF lower-body & torso) ===
  waist <down|up> [mult]                    squat down / stand up (keeps torso upright)
  waist <fwd|back> [mult]                   bow torso forward / lean backward (HipPitch)
  waist <left|right> [mult]                 lean torso left / right (HipRoll)
  waist <turn_left|turn_right> [mult]       twist torso yaw left / right (HipYaw)
  waist <1..5> [mult]                       step waist joint 1..5 directly
  waist home                                return waist to upright zero pose

  === Head / Neck Control (2-DoF) ===
  head <left|right|up|down|home> [mult]     turn or pitch head, or return to zero

  === Mobile Base Control (omnidirectional wheeled chassis) ===
  base <fwd|back|left|right> [scale]        drive/strafe chassis for 0.5s and stop
  base <turn_left|turn_right> [scale]       rotate chassis for 0.5s and stop
  base stop                                 stop chassis immediately

  === Postures & Status ===
  ready / pose ready                        move active arms to bent-elbow ready posture
  home  / pose home                         return arms, waist, and head to zero
  where                                     print joints, TCP xyz, grippers, waist, head, base
  help / quit
"""


@contextlib.contextmanager
def _deferred_interrupts() -> Any:
    """Hold Ctrl-C until teardown completes."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _describe_obs(obs: dict[str, Any]) -> str:
    parts: list[str] = []
    for group in ("state", "frames"):
        sub = obs.get(group)
        if not isinstance(sub, dict):
            continue
        items = ", ".join(f"{k}:{tuple(v.shape)}/{v.dtype}" for k, v in sub.items())
        parts.append(f"{group}={{{items}}}")
    return " ".join(parts)


def _run_batch_steps(
    env: Any,
    *,
    steps: int,
    delta_rad: float,
    reset: bool,
    no_move: bool,
) -> None:
    if no_move:
        obs = env._get_observation()
        assert env.observation_space.contains(obs), (
            "observation does not match env.observation_space"
        )
        print(f"Observation: {_describe_obs(obs)}")
        joints = obs["state"]["arm_joint_position"]
        print(f"Joints (deg): {np.round(np.rad2deg(joints), 2).tolist()}")
        return

    if reset:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), (
            "reset() observation does not match env.observation_space"
        )
        print(f"Reset observation: {_describe_obs(obs)}")
    else:
        obs = env._get_observation()

    base_joints = np.asarray(obs["state"]["arm_joint_position"], dtype=np.float32)
    for step_idx in range(steps):
        sign = 1.0 if (step_idx % 2 == 0) else -1.0
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        offset = 0
        for idx, _side in enumerate(env.config.active_arms):
            q_slice = slice(idx * 7, (idx + 1) * 7)
            action[offset : offset + 7] = base_joints[q_slice] + sign * delta_rad
            offset += 7
            if env._ee_dim > 0:
                action[offset : offset + env._ee_dim] = 1.0
                offset += env._ee_dim
        obs, reward, terminated, truncated, _info = env.step(action)
        assert env.observation_space.contains(obs), (
            f"step {step_idx} observation does not match env.observation_space"
        )
        q0 = float(obs["state"]["arm_joint_position"][0])
        print(
            f"step={step_idx + 1}/{steps} reward={reward:.4f} "
            f"terminated={terminated} truncated={truncated} "
            f"left_j1={np.rad2deg(q0):+.2f} deg"
        )


def _get_connection_host(env: Any) -> Any:
    if getattr(env, "robot", None) is None:
        return None
    arm = env.robot.child(env.config.active_arms[0]).child("arm")
    return getattr(arm, "_host", getattr(arm, "owner", None))


def _fmt_vec(arr: Any, decimals: int = 1) -> list[float]:
    return np.round(np.asarray(arr, dtype=np.float64), decimals).tolist()


def _print_status(env: Any, obs: dict[str, Any], host: Any) -> None:
    measured = np.asarray(obs["state"]["arm_joint_position"], dtype=np.float64)
    for idx, side in enumerate(env.config.active_arms):
        q_arm = measured[idx * 7 : (idx + 1) * 7]
        tcp = obs["state"].get("tcp_pose")
        if tcp is not None and len(tcp) >= (idx + 1) * 6:
            xyz = np.asarray(tcp[idx * 6 : idx * 6 + 3], dtype=np.float64)
        else:
            xyz, _, _, _ = compute_r2_arm_fk_and_jacobian(side, q_arm)
        print(
            f"  {side:5s} arm (deg): {_fmt_vec(np.rad2deg(q_arm), 1)} "
            f"| tcp xyz (m): {_fmt_vec(xyz, 3)}"
        )
    if "gripper_position" in obs["state"]:
        print("  end_effector:   ", _fmt_vec(obs["state"]["gripper_position"], 2))
    if host is not None:
        try:
            rstate = host.get_state()
            print(
                "  waist (deg):    ",
                _fmt_vec(np.rad2deg(rstate.waist_positions), 1),
                "| head (deg):",
                _fmt_vec(np.rad2deg(rstate.neck_positions), 1),
            )
            print(
                "  base pose:      ",
                _fmt_vec(rstate.base_pose, 3),
                "| base vel:",
                _fmt_vec(rstate.base_velocity, 3),
            )
        except Exception:
            pass


def _step_arm_smooth(
    env: Any,
    current_joints: np.ndarray,
    target_joints: np.ndarray,
    grips: np.ndarray,
    max_delta_rad: float,
    *,
    gripper_changed: bool = False,
) -> dict[str, Any]:
    diff = target_joints - current_joints
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    is_real_rate = not env.config.is_dummy and float(env.config.step_frequency) < 1000.0
    step_cap = (
        min(max(max_delta_rad, 0.01), 0.018)
        if is_real_rate
        else max(max_delta_rad, 0.03)
    )
    n_interp = max(1, int(np.ceil(max_abs / step_cap)))
    hold_steps = (12 if gripper_changed else 5) if is_real_rate else 0
    total_steps = n_interp + hold_steps
    for step_i in range(1, total_steps + 1):
        alpha = min(1.0, float(step_i) / float(n_interp))
        q_interp = current_joints + alpha * diff
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        offset = 0
        for idx, _side in enumerate(env.config.active_arms):
            q_slice = slice(idx * 7, (idx + 1) * 7)
            action[offset : offset + 7] = q_interp[q_slice]
            offset += 7
            if env._ee_dim > 0:
                g_val = float(grips[idx]) if idx < grips.size else float(grips[0])
                action[offset : offset + env._ee_dim] = g_val
                offset += env._ee_dim
        env.step(action)
    return env._get_observation()


def _apply_arm_command(
    side: str,
    sub_words: Sequence[str],
    q_arm: np.ndarray,
    q_waist: np.ndarray,
    step_rad: float,
) -> Optional[np.ndarray]:
    if not sub_words:
        return None
    op = sub_words[0].lower()
    arg = float(sub_words[1]) if len(sub_words) >= 2 else None
    lo = R2_LEFT_ARM_LIMITS_LOW if side == "left" else R2_RIGHT_ARM_LIMITS_LOW
    hi = R2_LEFT_ARM_LIMITS_HIGH if side == "left" else R2_RIGHT_ARM_LIMITS_HIGH

    cartesian_dirs: dict[str, np.ndarray] = {
        "fwd": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "back": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "backward": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "left": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "right": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "down": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    if op in cartesian_dirs:
        cm = 3.0 if arg is None else float(arg)
        delta_xyz = cartesian_dirs[op] * float(cm * 0.01)
        return compute_r2_arm_cartesian_step(
            side, q_arm, delta_xyz, q_waist=q_waist, max_step_rad=0.35
        )

    mult = 1.0 if arg is None else float(arg)
    delta = float(mult * step_rad)
    q_new = q_arm.copy()
    out_sign = 1.0 if side == "left" else -1.0
    bend_sign = -1.0 if side == "left" else 1.0

    semantic_map: dict[str, tuple[int, float]] = {
        "raise": (0, -1.0),
        "lower": (0, 1.0),
        "out": (1, out_sign),
        "in": (1, -out_sign),
        "twist_out": (2, 1.0),
        "twist_in": (2, -1.0),
        "bend": (3, bend_sign),
        "extend": (3, -bend_sign),
        "wrist_up": (5, -1.0),
        "wrist_down": (5, 1.0),
        "roll_cw": (6, 1.0),
        "roll_ccw": (6, -1.0),
    }
    if op in semantic_map:
        j_idx, sign = semantic_map[op]
        q_new[j_idx] = float(np.clip(q_new[j_idx] + sign * delta, lo[j_idx], hi[j_idx]))
        return q_new

    try:
        joint_idx = int(op) - 1
    except ValueError:
        return None
    if not (0 <= joint_idx < 7):
        print("Joint index must be 1..7")
        return None
    q_new[joint_idx] = float(
        np.clip(q_new[joint_idx] + delta, lo[joint_idx], hi[joint_idx])
    )
    return q_new


def _run_interactive(
    env: Any,
    *,
    step_rad: float,
    reset: bool,
    commands: Optional[Sequence[str]] = None,
) -> None:
    if reset:
        env.reset()
    host = _get_connection_host(env)
    obs = env._get_observation()
    joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
    num_arms = len(env.config.active_arms)
    if (
        "gripper_position" in obs["state"]
        and len(obs["state"]["gripper_position"]) >= num_arms
    ):
        grips = np.array(obs["state"]["gripper_position"][:num_arms], dtype=np.float32)
    else:
        grips = np.ones(num_arms, dtype=np.float32)

    if host is not None:
        rstate = host.get_state()
        waist_q = np.array(rstate.waist_positions, dtype=np.float32)
        neck_q = np.array(rstate.neck_positions, dtype=np.float32)
    else:
        waist_q = np.zeros(5, dtype=np.float32)
        neck_q = np.zeros(2, dtype=np.float32)

    if not commands:
        print(HELP)
    cmd_queue = list(commands) if commands else None

    while True:
        if cmd_queue is not None:
            if not cmd_queue:
                return
            raw_line = cmd_queue.pop(0).strip()
            if not raw_line:
                continue
            print(f"r2> {raw_line}")
            words = raw_line.split()
        else:
            try:
                words = input("r2> ").split()
            except (EOFError, KeyboardInterrupt):
                print()
                return
        if not words:
            continue
        cmd = words[0].lower()
        if cmd in ("quit", "exit", "q"):
            return
        if cmd in ("help", "?"):
            print(HELP)
            continue
        if cmd in ("where", "status"):
            obs = env._get_observation()
            _print_status(env, obs, host)
            continue

        target_joints = joints.copy()
        needs_arm_step = False
        gripper_changed = False

        if cmd == "pose" and len(words) >= 2:
            cmd = words[1].lower()
            words = [cmd] + list(words[2:])

        if cmd == "home":
            if host is not None:
                is_real_rate = (
                    not env.config.is_dummy
                    and float(env.config.step_frequency) < 1000.0
                )
                dt = 1.0 / max(float(env.config.step_frequency), 1.0)
                rstate = host.get_state()
                waist_start = np.array(rstate.waist_positions, dtype=np.float32)
                neck_start = np.array(rstate.neck_positions, dtype=np.float32)
                waist_q = np.zeros_like(waist_start)
                if waist_q.size >= 5:
                    waist_q[0] = max(0.01, min(float(waist_start[0]), 0.015))
                    waist_q[1] = max(0.02, min(float(waist_start[1]), 0.03))
                neck_q = np.zeros_like(neck_start)
                n_w = max(
                    1,
                    int(
                        np.ceil(
                            max(
                                float(np.max(np.abs(waist_start - waist_q))),
                                float(np.max(np.abs(neck_start - neck_q))),
                            )
                            / (0.004 if is_real_rate else 0.05)
                        )
                    ),
                )
                for s_i in range(1, n_w + 1):
                    alpha = float(s_i) / float(n_w)
                    host.move_waist_joint(
                        (waist_start + alpha * (waist_q - waist_start)).astype(
                            np.float32
                        )
                    )
                    host.move_neck_joint(
                        (neck_start + alpha * (neck_q - neck_start)).astype(np.float32)
                    )
                    if is_real_rate:
                        time.sleep(dt)
            obs, _ = env.reset()
            joints = np.array(obs["state"]["arm_joint_position"], dtype=np.float32)
            _print_status(env, obs, host)
            continue

        if cmd == "ready":
            for idx, side in enumerate(env.config.active_arms):
                target_joints[idx * 7 : (idx + 1) * 7] = (
                    R2_LEFT_READY_QPOS if side == "left" else R2_RIGHT_READY_QPOS
                )
            needs_arm_step = True
        elif cmd == "open":
            grips[:] = 1.0
            needs_arm_step = True
            gripper_changed = True
        elif cmd == "close":
            grips[:] = 0.0
            needs_arm_step = True
            gripper_changed = True
        elif cmd in ("l", "left", "r", "right") and len(words) >= 2:
            side = "left" if cmd in ("l", "left") else "right"
            if side not in env.config.active_arms:
                print(f"Arm {side!r} is not active in this environment")
                continue
            arm_idx = env.config.active_arms.index(side)
            sub_op = words[1].lower()
            if sub_op in ("open", "close"):
                grips[arm_idx] = 1.0 if sub_op == "open" else 0.0
                needs_arm_step = True
                gripper_changed = True
            else:
                q_slice = slice(arm_idx * 7, (arm_idx + 1) * 7)
                try:
                    q_updated = _apply_arm_command(
                        side, words[1:], target_joints[q_slice], waist_q, step_rad
                    )
                except ValueError:
                    print("Invalid numeric argument; type help")
                    continue
                if q_updated is None:
                    print(f"Unknown arm command {' '.join(words)!r}; type help")
                    continue
                target_joints[q_slice] = q_updated
                needs_arm_step = True
        elif cmd in ("waist", "w") and len(words) >= 2:
            if host is None:
                print("Waist control requires a connected R2 host")
                continue
            sub_op = words[1].lower()
            mult = float(words[2]) if len(words) >= 3 else 1.0
            delta = float(mult * step_rad)
            is_real_rate = (
                not env.config.is_dummy and float(env.config.step_frequency) < 1000.0
            )
            if is_real_rate:
                waist_q = np.array(host.get_state().waist_positions, dtype=np.float32)
            w_target = waist_q.copy()
            if sub_op == "home":
                if w_target.size >= 5:
                    w_target[0] = max(0.01, min(float(waist_q[0]), 0.015))
                    w_target[1] = max(0.02, min(float(waist_q[1]), 0.03))
                    w_target[2:] = 0.0
                else:
                    w_target[:] = 0.0
            elif sub_op in ("down", "squat") and w_target.size >= 3:
                if is_real_rate:
                    safe_joints = joints.copy()
                    need_clearance = False
                    for arm_i, arm_s in enumerate(env.config.active_arms):
                        j2_idx = arm_i * 7 + 1
                        if arm_s == "left" and safe_joints[j2_idx] < 0.12:
                            safe_joints[j2_idx] = 0.15
                            need_clearance = True
                        elif arm_s == "right" and safe_joints[j2_idx] > -0.12:
                            safe_joints[j2_idx] = -0.15
                            need_clearance = True
                    if need_clearance:
                        obs = _step_arm_smooth(
                            env, joints, safe_joints, grips, step_rad
                        )
                        joints = safe_joints.copy()
                sq_delta = min(delta, 0.03)
                w_target[0] = max(0.01, float(w_target[0])) + sq_delta
                w_target[1] = max(0.02, float(w_target[1])) + 2.0 * sq_delta
                w_target[2] += sq_delta
            elif sub_op in ("up", "stand") and w_target.size >= 3:
                sq_delta = min(delta, 0.03)
                w_target[0] = max(0.01, float(w_target[0] - sq_delta))
                w_target[1] = max(0.02, float(w_target[1] - 2.0 * sq_delta))
                w_target[2] = max(0.0, float(w_target[2] - sq_delta))
            elif sub_op in ("fwd", "forward", "bow"):
                idx_p = 2 if w_target.size >= 5 else 1
                w_target[idx_p] -= delta
            elif sub_op in ("back", "backward"):
                idx_p = 2 if w_target.size >= 5 else 1
                w_target[idx_p] += delta
            elif sub_op == "left":
                idx_r = 3 if w_target.size >= 5 else 2
                w_target[idx_r] -= delta
            elif sub_op == "right":
                idx_r = 3 if w_target.size >= 5 else 2
                w_target[idx_r] += delta
            elif sub_op == "turn_left":
                w_target[-1] += delta
            elif sub_op == "turn_right":
                w_target[-1] -= delta
            else:
                try:
                    w_idx = int(sub_op) - 1
                except ValueError:
                    print(f"Unknown waist command {sub_op!r}; type help")
                    continue
                if not (0 <= w_idx < w_target.size):
                    print(f"Waist joint index must be 1..{w_target.size}")
                    continue
                w_target[w_idx] += delta

            if w_target.size == 5:
                w_target = np.clip(w_target, R2_WAIST_LIMITS_LOW, R2_WAIST_LIMITS_HIGH)
            w_step_cap = 0.004 if is_real_rate else max(step_rad, 0.03)
            n_interp = max(
                1,
                int(np.ceil(float(np.max(np.abs(w_target - waist_q))) / w_step_cap)),
            )
            total_steps = n_interp + (5 if is_real_rate else 0)
            dt = 1.0 / max(float(env.config.step_frequency), 1.0)
            for s_i in range(1, total_steps + 1):
                alpha = min(1.0, float(s_i) / float(n_interp))
                q_step = waist_q + alpha * (w_target - waist_q)
                host.move_waist_joint(q_step.astype(np.float32))
                if is_real_rate:
                    time.sleep(dt)
            waist_q = w_target.astype(np.float32)
            obs = env._get_observation()
            _print_status(env, obs, host)
            continue
        elif cmd in ("head", "neck", "h") and len(words) >= 2:
            if host is None:
                print("Head/neck control requires a connected R2 host")
                continue
            sub_op = words[1].lower()
            mult = float(words[2]) if len(words) >= 3 else 1.0
            delta = float(mult * step_rad)
            is_real_rate = (
                not env.config.is_dummy and float(env.config.step_frequency) < 1000.0
            )
            if is_real_rate:
                neck_q = np.array(host.get_state().neck_positions, dtype=np.float32)
            n_target = neck_q.copy()
            if sub_op == "home":
                n_target[:] = 0.0
            elif sub_op == "left":
                n_target[0] += delta
            elif sub_op == "right":
                n_target[0] -= delta
            elif sub_op == "up":
                n_target[1] += delta
            elif sub_op == "down":
                n_target[1] -= delta
            else:
                print(f"Unknown head command {sub_op!r}; type help")
                continue
            n_target = np.clip(n_target, R2_NECK_LIMITS_LOW, R2_NECK_LIMITS_HIGH)
            n_step_cap = 0.008 if is_real_rate else max(step_rad, 0.03)
            n_interp = max(
                1,
                int(np.ceil(float(np.max(np.abs(n_target - neck_q))) / n_step_cap)),
            )
            total_steps = n_interp + (6 if is_real_rate else 0)
            dt = 1.0 / max(float(env.config.step_frequency), 1.0)
            for s_i in range(1, total_steps + 1):
                alpha = min(1.0, float(s_i) / float(n_interp))
                q_step = neck_q + alpha * (n_target - neck_q)
                host.move_neck_joint(q_step.astype(np.float32))
                if is_real_rate:
                    time.sleep(dt)
            neck_q = n_target.astype(np.float32)
            obs = env._get_observation()
            _print_status(env, obs, host)
            continue
        elif cmd in ("base", "b") and len(words) >= 2:
            if host is None:
                print("Base control requires a connected R2 host")
                continue
            sub_op = words[1].lower()
            scale = float(words[2]) if len(words) >= 3 else 1.0
            base_dirs: dict[str, np.ndarray] = {
                "fwd": np.array([0.15, 0.0, 0.0], dtype=np.float32),
                "forward": np.array([0.15, 0.0, 0.0], dtype=np.float32),
                "back": np.array([-0.15, 0.0, 0.0], dtype=np.float32),
                "backward": np.array([-0.15, 0.0, 0.0], dtype=np.float32),
                "left": np.array([0.0, 0.15, 0.0], dtype=np.float32),
                "right": np.array([0.0, -0.15, 0.0], dtype=np.float32),
                "turn_left": np.array([0.0, 0.0, 0.25], dtype=np.float32),
                "turn_right": np.array([0.0, 0.0, -0.25], dtype=np.float32),
            }
            if sub_op == "stop":
                host.stop_base()
                obs = env._get_observation()
                _print_status(env, obs, host)
                continue
            if sub_op not in base_dirs:
                print(f"Unknown base command {sub_op!r}; type help")
                continue
            vel = base_dirs[sub_op] * float(scale)
            pulse_info = host.move_base_pulse(
                vel,
                duration=0.45,
                rate_hz=float(env.config.step_frequency),
            )
            obs = env._get_observation()
            print(
                "  base pulse vel: ",
                _fmt_vec(pulse_info["base_velocity"], 3),
                "| base pose:",
                _fmt_vec(pulse_info["base_pose"], 3),
            )
            continue
        else:
            print(f"Unknown command {cmd!r}; type help")
            continue

        if needs_arm_step:
            obs = _step_arm_smooth(
                env,
                joints,
                target_joints,
                grips,
                step_rad,
                gripper_changed=gripper_changed,
            )
            joints = target_joints.copy()
            _print_status(env, obs, host)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run against fake RFrame SDKs, no hardware required",
    )
    parser.add_argument(
        "--body-dof",
        type=str,
        default="auto",
        choices=("auto", "20", "21"),
        help="R2 whole-body DoF (auto, 20, or 21; default auto)",
    )
    parser.add_argument(
        "--end-effector",
        type=str,
        default="auto",
        choices=("auto", "gripper", "hand", "none"),
        help="end-effector type on both arms (auto, gripper, hand, or none; default auto)",
    )
    parser.add_argument(
        "--enable-base",
        action="store_true",
        help="include the omnidirectional wheeled chassis",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=["head", "wrist_left", "wrist_right"],
        help="R2 camera names to subscribe to",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=2.0,
        help="degrees per move (default 2.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="number of batch steps when not running interactively (default 5)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="control updates per second (default 20)",
    )
    parser.add_argument(
        "--reset-speed",
        type=float,
        default=30.0,
        help="maximum reset joint speed in degrees per second (default 30)",
    )
    parser.add_argument(
        "--no-reset",
        dest="reset",
        action="store_false",
        help="start from current arm pose instead of resetting first",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="only connect and print observation without sending joint commands",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="enter interactive command prompt",
    )
    parser.add_argument(
        "--commands",
        type=str,
        default=None,
        help="semicolon-separated interactive commands to execute in sequence",
    )
    parser.add_argument(
        "--enable-camera-player",
        action="store_true",
        help="enable the camera preview window (disabled by default)",
    )
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not np.isfinite(args.reset_speed) or args.reset_speed <= 0:
        parser.error("--reset-speed must be finite and positive")

    if args.mock:
        import sys

        sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
        from robot_mocks import mocked_sdks

        context: Any = mocked_sdks()
    else:
        context = contextlib.nullcontext()

    with context:
        from rlinf.envs.real.r2 import R2ReachEnv
        from rlinf.robotics import R2Config
        from rlinf.robotics.discovery import RobotDiscovery

        enable_base = bool(
            args.enable_base or args.interactive or args.commands is not None
        )
        cfg = R2Config(
            node_rank=0,
            body_dof=None if args.body_dof == "auto" else int(args.body_dof),
            end_effector_type=(
                None if args.end_effector == "auto" else args.end_effector
            ),
            enable_base=enable_base,
            cameras=list(args.cameras),
        )
        discovery = RobotDiscovery.registry["R2"].discovery_cls
        resources = discovery.enumerate(node_rank=0, configs=[cfg])
        resolved_cfg = resources.infos[0].config
        print(
            f"R2 hardware config: body_dof={resolved_cfg.body_dof}, "
            f"end_effector_type={resolved_cfg.end_effector_type!r}, "
            f"enable_base={resolved_cfg.enable_base}"
        )

        env = R2ReachEnv(
            {
                "enable_camera_player": args.enable_camera_player,
                "step_frequency": args.fps,
                "reset_joint_speed": float(np.deg2rad(args.reset_speed)),
                "reset_on_init": False,
            },
            robot_info=resources.infos[0],
        )
        try:
            if args.interactive or args.commands is not None:
                parsed_cmds = (
                    [c.strip() for c in args.commands.split(";") if c.strip()]
                    if args.commands is not None
                    else None
                )
                _run_interactive(
                    env,
                    step_rad=float(np.deg2rad(args.step)),
                    reset=args.reset,
                    commands=parsed_cmds,
                )
            else:
                _run_batch_steps(
                    env,
                    steps=args.steps,
                    delta_rad=float(np.deg2rad(args.step)),
                    reset=args.reset,
                    no_move=args.no_move,
                )
        finally:
            with _deferred_interrupts():
                env.close()


if __name__ == "__main__":
    main()
