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
"""Drive an SO-101 by hand, through the SO-101 environment.

Run it, then type commands::

    python so101_manual.py --port /dev/ttyACM0 --id my-arm
    python so101_manual.py --mock          # no hardware, to try the commands

Add --leader to hand the arm over to an SO-101 leader, then type "teleop"::

    python so101_manual.py --port /dev/ttyACM0 --leader /dev/ttyACM1

The arm has joint encoders and no kinematic model, so "up" and "forward"
are single joint moves rather than Cartesian ones: up bends the shoulder
and forward bends the elbow.
"""

import argparse

import numpy as np

# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
COMMANDS = {
    "up": (1, +1),
    "down": (1, -1),
    "forward": (2, +1),
    "back": (2, -1),
    "left": (0, +1),
    "right": (0, -1),
    "tilt": (3, +1),
    "untilt": (3, -1),
    "roll": (4, +1),
    "unroll": (4, -1),
}

HELP = """
  up / down        bend the shoulder
  forward / back   bend the elbow
  left / right     turn the base
  tilt / untilt    bend the wrist
  roll / unroll    turn the wrist
  open / close     the gripper
  home             back to the rest pose
  where            print the joint angles
  teleop           follow the leader arm until Ctrl-C (needs --leader)
  quit

Add a number to repeat a move: "up 3". One move is --step degrees.
If a direction goes the wrong way on your arm, pass --invert.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0", help="serial port")
    parser.add_argument("--id", default=None, help="lerobot calibration id")
    parser.add_argument(
        "--leader", default=None, help="serial port of an SO-101 leader arm"
    )
    parser.add_argument(
        "--leader-id", default=None, help="lerobot calibration id of the leader"
    )
    parser.add_argument(
        "--step", type=float, default=5.0, help="degrees per move (default 5)"
    )
    parser.add_argument(
        "--invert", action="store_true", help="flip every direction's sign"
    )
    parser.add_argument(
        "--mock", action="store_true", help="run against a fake arm, no hardware"
    )
    args = parser.parse_args()

    if args.mock:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent / "tests"))
        from robot_mocks import mocked_sdks

        context = mocked_sdks()
    else:
        import contextlib

        context = contextlib.nullcontext()

    with context:
        from rlinf.envs.real.so101 import SO101ReachEnv

        env = SO101ReachEnv(
            {
                "port": args.port,
                "calibration_id": args.id,
                "enable_camera_player": False,
            }
        )
        try:
            drive(
                env,
                step=np.deg2rad(args.step),
                invert=args.invert,
                leader_port=args.leader,
                leader_id=args.leader_id,
            )
        finally:
            env.close()


def teleop(env, port: str, calibration_id) -> None:
    """Let an SO-101 leader arm drive the follower until Ctrl-C.

    The leader is the same five joints and gripper, so its reading is the
    follower's target with no conversion. It reports a pose whether or not
    anyone is holding it, and only takes over once it has actually moved.
    """
    from rlinf.robotics.parts.teleop import SO101Leader

    leader = SO101Leader(port=port, calibration_id=calibration_id)
    try:
        leader.connect()
    except RuntimeError as error:
        # An uncalibrated leader explains itself; do not bury that in a stack
        # trace when the operator is sitting right here.
        print(f"\n{error}\n")
        return
    print("Following the leader arm. Ctrl-C to stop.")
    try:
        while True:
            action = leader.drive({"joint_positions": env.get_joint_positions()})
            if not action.driving:
                continue
            command = np.append(action.parts["arm"], action.parts["end_effector"])
            env.step(command.astype(np.float32))
    except KeyboardInterrupt:
        print("\nStopped following.")
    finally:
        leader.disconnect()


def drive(env, step: float, invert: bool, leader_port=None, leader_id=None) -> None:
    """Read commands until the operator quits."""
    observation, _ = env.reset()
    # Absolute joint targets: start where the arm already is, so the first
    # command moves it by one step rather than snapping somewhere.
    target = np.asarray(observation["state"]["arm_joint_position"], dtype=float)
    grip = float(observation["state"]["gripper_position"][0])
    sign = -1.0 if invert else 1.0
    print(HELP)

    while True:
        try:
            words = input("so101> ").split()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not words:
            continue

        name, count = words[0].lower(), 1
        if len(words) > 1:
            try:
                count = int(words[1])
            except ValueError:
                print(f"'{words[1]}' is not a number")
                continue

        if name in ("quit", "exit", "q"):
            return
        if name in ("help", "?"):
            print(HELP)
            continue
        if name == "where":
            print("  joints:", np.round(np.rad2deg(target), 1), "deg")
            print("  gripper:", round(grip, 2))
            continue
        if name == "teleop":
            if leader_port is None:
                print("teleop needs a leader arm: pass --leader /dev/ttyACM1")
                continue
            teleop(env, leader_port, leader_id)
            # Pick up where the leader left the arm. Commanding the pre-teleop
            # target here would snap it back across the workspace.
            target = env.get_joint_positions()[0].astype(float)
            print("  joints:", np.round(np.rad2deg(target), 1), "deg")
            continue
        if name == "home":
            target = np.zeros_like(target)
        elif name == "open":
            grip = 1.0
        elif name == "close":
            grip = 0.0
        elif name in COMMANDS:
            joint, direction = COMMANDS[name]
            target[joint] += sign * direction * step * count
        else:
            print(f"'{name}' is not a command; type help")
            continue

        observation, *_ = env.step(np.append(target, grip).astype(np.float32))
        # The arm clips what it cannot reach, so follow what it actually did.
        target = np.asarray(observation["state"]["arm_joint_position"], dtype=float)
        print("  joints:", np.round(np.rad2deg(target), 1), "deg")


if __name__ == "__main__":
    main()
