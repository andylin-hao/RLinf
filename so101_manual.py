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
  quit

Add a number to repeat a move: "up 3". One move is --step degrees.
If a direction goes the wrong way on your arm, pass --invert.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0", help="serial port")
    parser.add_argument("--id", default=None, help="lerobot calibration id")
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
            drive(env, step=np.deg2rad(args.step), invert=args.invert)
        finally:
            env.close()


def drive(env, step: float, invert: bool) -> None:
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
