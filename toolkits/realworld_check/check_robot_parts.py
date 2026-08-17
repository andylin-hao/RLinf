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

"""Check that a robot composes, connects, observes, and lets go.

This is the bench counterpart to the unit tests: they cover composition with
fake hardware, and this runs the same path against the real thing. It reports
what a robot is made of, which connection backs each part, where each was
placed, and what every part observes -- then disconnects and checks that
nothing still claims to be connected.

Run it after changing a part, a connection, or how a robot is composed::

    python -m toolkits.realworld_check.check_robot_parts Franka \\
        --arg robot_ip=10.0.0.1 --arg node_rank=1

    python -m toolkits.realworld_check.check_robot_parts DualFranka \\
        --arg left_robot_ip=10.0.0.1 --arg right_robot_ip=10.0.0.2

Values are parsed as Python literals when they look like one, so
``--arg node_rank=1`` passes an int and ``--arg robot_ip=10.0.0.1`` a string.

Exit code is 0 when every part connects, observes and releases; 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
import traceback
from typing import Any

from rlinf.robotics.parts.base import Connection, Group


def _mocked_sdks():
    """The fake vendor SDKs, which live with the tests rather than the package."""
    tests = pathlib.Path(__file__).resolve().parents[2] / "tests" / "unit_tests"
    if str(tests) not in sys.path:
        sys.path.insert(0, str(tests))
    from robot_mocks import mocked_sdks

    return mocked_sdks()


def literal(text: str) -> Any:
    """Parse a value the way a config would spell it."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def walk(part: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every leaf part of a robot, by dotted path."""
    if isinstance(part, Group):
        found: list[tuple[str, Any]] = []
        for name, child in part.parts.items():
            found += walk(child, f"{prefix}{name}.")
        return found
    return [(prefix.rstrip("."), part)]


def describe(robot: Any) -> list[str]:
    """What the robot is made of, and what backs each part."""
    lines = []
    for path, part in walk(robot):
        backing = type(part).__name__
        host = getattr(part, "_host", None)
        if host is not None:
            backing = f"{type(part).__name__} on {type(host).__name__}"
        lines.append(f"    {path:24} {backing}")
    return lines


def check(robot_type: str, kwargs: dict[str, Any]) -> int:
    print(f"[1/5] composing {robot_type} with {kwargs}")
    # Importing the robots is what registers them.
    import rlinf.robotics.robots  # noqa: F401
    from rlinf.robotics.discovery import build_robot

    robot = build_robot(robot_type, **kwargs)

    print("[2/5] parts, before connecting")
    for line in describe(robot):
        print(line)

    print("[3/5] connecting")
    robot.connect()
    if not robot.is_connected:
        print("    FAIL: the robot does not report itself connected")
        return 1
    for name, handle in robot.handles.items():
        print(f"    {name:24} placed as {type(handle).__name__}")

    print("[4/5] observing every part")
    failures = []
    for path, part in walk(robot):
        if isinstance(part, Connection):
            failures.append(f"{path} is a Connection and should not be in the tree")
            continue
        try:
            observation = part.get_observation()
        except Exception as error:  # noqa: BLE001 - a bench check reports anything
            failures.append(f"{path}: {type(error).__name__}: {error}")
            continue
        shapes = {
            key: getattr(value, "shape", type(value).__name__)
            for key, value in observation.items()
        }
        declared = set(part.observation_features)
        extra = set(observation) - declared
        missing = declared - set(observation)
        note = ""
        if extra or missing:
            note = f"  MISMATCH extra={sorted(extra)} missing={sorted(missing)}"
            failures.append(
                f"{path} observes {sorted(observation)}, declares {sorted(declared)}"
            )
        print(f"    {path:24} {shapes}{note}")

    print("[5/5] disconnecting")
    robot.disconnect()
    still = [path for path, part in walk(robot) if getattr(part, "is_connected", False)]
    if still:
        failures.append(f"still connected after disconnect: {still}")
    if robot.is_connected:
        failures.append("the robot still reports itself connected")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK: every part connected, observed what it declares, and let go.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("robot_type", help="a registered robot type, e.g. Franka")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run against faked vendor SDKs instead of hardware, so the same "
        "command works on a laptop and on the bench",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="an argument for the robot's build(); repeatable",
    )
    args = parser.parse_args()

    kwargs = {}
    for item in args.arg:
        if "=" not in item:
            parser.error(f"--arg wants NAME=VALUE, got {item!r}")
        name, _, value = item.partition("=")
        kwargs[name] = literal(value)

    try:
        if args.mock:
            print("[mock] vendor SDKs are faked; this checks the code, not a robot")
            with _mocked_sdks():
                return check(args.robot_type, kwargs)
        return check(args.robot_type, kwargs)
    except Exception:  # noqa: BLE001 - a bench check reports anything
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
