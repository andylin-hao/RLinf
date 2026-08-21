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

"""The conformance suites, pointed at the robots that ship and at broken ones.

Half of this file is negative: a check that cannot fail is worse than no check,
because it reports a promise as kept. Every contract here is shown catching an
implementation that breaks it, and the broken implementations are the real bugs
this package has had -- a camera that cannot be opened twice, a part observing
a value wider than it declared, a robot that keeps what it built when a later
part refuses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "tests", _ROOT / "tests" / "robot_mocks"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from robot_contracts import (  # noqa: E402
    ConformanceError,
    ConnectionContract,
    PartContract,
    RobotContract,
)

from rlinf.robotics.parts.base import (  # noqa: E402
    Connection,
    ControllablePart,
    RobotPart,
)

#: Every shipped robot, and the settings its builder needs.
SHIPPED = {
    "Franka": {
        "robot_ip": "0.0.0.0",
        "node_rank": 1,
        "env_idx": 0,
        "worker_rank": 0,
    },
    "DualFranka": {
        "left_robot_ip": "0.0.0.0",
        "right_robot_ip": "0.0.0.1",
        "left_gripper_connection": "/dev/mock-left",
        "right_gripper_connection": "/dev/mock-right",
        "node_rank": 0,
        "env_idx": 0,
        "worker_rank": 0,
    },
    "GimArm": {
        "can_interface": "can0",
        "arm_variant": "gim_arm_xl",
        "enable_gripper": True,
        "gripper_type": "parallel",
        "control_mode": "momentum_observer",
        "node_rank": 0,
        "env_idx": 0,
        "worker_rank": 0,
    },
    "Turtle2": {
        "frequency": 50,
        "camera_ids": [1],
        "node_rank": 0,
        "env_idx": 0,
        "worker_rank": 0,
    },
    "DOSW1": {"node_rank": 0},
}


@pytest.mark.parametrize("robot_type", sorted(SHIPPED))
def test_every_shipped_robot_keeps_the_robot_contract(robot_type):
    """Connect, read, disconnect, do it again, and roll back a failure.

    Against faked SDKs, so this is the code being checked and not a robot.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        import rlinf.robotics.robots  # noqa: F401
        from rlinf.robotics.discovery import build_robot

        settings = SHIPPED[robot_type]
        failures = RobotContract(
            lambda: build_robot(robot_type, **settings)  # noqa: B023
        ).failures()

    assert failures == [], "\n".join(failures)


#: Every camera backend, with an identifier its fake SDK answers to and the
#: list that fake appends to when a device is opened.
SHIPPED_CAMERAS = {
    "realsense": ("MOCK0001", "pyrealsense2", "opened"),
    "zed": ("12345678", "pyzed.sl", "opened"),
    "lumos": ("video0", "cv2", "opens"),
}


@pytest.mark.parametrize("camera_type", sorted(SHIPPED_CAMERAS))
def test_every_shipped_camera_keeps_the_part_contract(camera_type):
    """Connect, read, disconnect, and open again -- for every backend.

    Only RealSense and ZED had fake SDKs, so LUMOS was the one camera nobody
    ever connected: it opened its device in ``__init__`` and implemented no
    ``_open``, so ``connect()`` raised ``NotImplementedError`` every time. The
    contract that would have caught it existed; it was never pointed here.
    """
    from robot_mocks import mocked_sdks

    with mocked_sdks():
        from rlinf.robotics.parts.cameras import Camera, CameraInfo

        info = CameraInfo(
            name="wrist",
            serial_number=SHIPPED_CAMERAS[camera_type][0],
            camera_type=camera_type,
            resolution=(64, 48),
        )
        failures = PartContract(lambda: Camera.of(info)).failures()  # noqa: B023

    assert failures == [], "\n".join(failures)


@pytest.mark.parametrize("camera_type", sorted(SHIPPED_CAMERAS))
def test_building_a_camera_opens_no_device(camera_type):
    """Composing a robot must not touch hardware, cameras included.

    A camera is declared on the machine composing the robot and opened on the
    machine it is plugged into, so a constructor that opens the device breaks
    placement -- and breaks describing a robot from a laptop.
    """
    from robot_mocks import mocked_sdks

    serial, sdk_name, opened_attr = SHIPPED_CAMERAS[camera_type]

    with mocked_sdks() as sdks:
        from rlinf.robotics.parts.cameras import Camera, CameraInfo

        info = CameraInfo(
            name="wrist",
            serial_number=serial,
            camera_type=camera_type,
            resolution=(64, 48),
        )
        camera = Camera.of(info, node_rank=2)

        assert not camera.is_connected
        assert camera.node_rank == 2
        if sdk_name is not None:
            opened = getattr(sdks[sdk_name], opened_attr)
            assert opened == [], (
                f"{camera_type} opened {opened} while being constructed"
            )

        # And it does open once connected, so the check above is not passing
        # because the fake records nothing.
        camera.connect()
        assert camera.is_connected
        if sdk_name is not None:
            assert getattr(sdks[sdk_name], opened_attr), (
                f"{camera_type} recorded no device after connect()"
            )
        camera.disconnect()


def test_a_gripper_opens_when_connected_and_not_before():
    """A gripper is a part, and parts do not open in their constructor.

    ``RobotiqGripper.__init__`` used to open the serial port and run the
    activation sequence -- which moves the fingers -- while ``connect()``
    merely checked that it had worked. A robot composed with one reported
    itself connected before anything connected it, and stayed connected after
    disconnecting.
    """
    from robot_mocks import mocked_sdks

    from rlinf.robotics.robot import Robot

    class Bench(Robot):
        ROBOT_TYPE = "BenchGripper"

    with mocked_sdks():
        from rlinf.robotics.parts.end_effectors.grippers.robotiq import RobotiqGripper

        gripper = RobotiqGripper(port="/dev/mock-gripper")
        assert not gripper.is_connected, "the constructor opened the port"
        assert not gripper.is_ready(), "the constructor activated the gripper"

        robot = Bench(hand=gripper)
        assert not robot.is_connected

        robot.connect()
        assert robot.is_connected and gripper.is_ready()

        robot.disconnect()
        assert not robot.is_connected, "the gripper stayed connected after teardown"
        assert not gripper.is_ready()

        # Stall recovery opens it again, like any other part.
        robot.connect()
        assert robot.is_connected
        robot.disconnect()


def test_a_gripper_keeps_the_part_contract():
    """The full part lifecycle for a gripper that can run on fakes.

    The Franka gripper is left out: it rides the arm's ROS session, so there
    is no gripper to build without an arm around it, and the DualFranka robot
    contract already covers that path.
    """
    import numpy as np
    from robot_mocks import mocked_sdks

    from rlinf.robotics.parts.end_effectors import EndEffector

    with mocked_sdks():
        failures = PartContract(
            lambda: EndEffector.of("robotiq", port="/dev/mock-gripper"),
            action={"target": np.zeros(1, dtype=np.float32)},
        ).failures()

    assert failures == [], "\n".join(failures)


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("rlinf.robotics.parts.arms.turtle2", "Turtle2Connection"),
        ("rlinf.robotics.parts.arms.dosw1", "DOSW1Connection"),
    ],
)
def test_every_shipped_connection_keeps_the_connection_contract(
    module_name, class_name
):
    """A connection owns a session and offers what rides on it."""
    from importlib import import_module

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        connection_cls = getattr(import_module(module_name), class_name)
        failures = ConnectionContract(connection_cls).failures()

    assert failures == [], "\n".join(failures)


# --- the checks, shown catching what they exist to catch ---------------------


class _Well(RobotPart):
    """A part that keeps every promise, for the broken ones to differ from."""

    def __init__(self):
        self.opens = 0

    def _open(self):
        self.opens += 1
        return object()

    @property
    def observation_features(self):
        return {"state": {"shape": (1,), "dtype": "float64"}}

    def get_observation(self):
        return {"state": np.zeros(1)}


def test_the_part_contract_passes_a_part_that_keeps_it():
    assert PartContract(_Well).failures() == []


def test_the_part_contract_catches_a_part_that_cannot_be_opened_twice():
    """The camera bug: a thread built once, and stall recovery reopening."""

    class OnceOnly(_Well):
        def connect(self):
            if self.opens:
                raise RuntimeError("threads can only be started once")
            super().connect()

    failures = PartContract(OnceOnly).failures()

    assert any("reconnect" in failure for failure in failures), failures


def test_the_part_contract_catches_an_observation_wider_than_declared():
    """The tcp_pose bug: right key, wrong width, silently into a policy."""

    class TooWide(_Well):
        def get_observation(self):
            return {"state": np.zeros(2)}

    failures = PartContract(TooWide).failures()

    assert any("shape (2,)" in failure and "(1,)" in failure for failure in failures), (
        failures
    )


def test_the_part_contract_catches_an_observation_it_never_declared():
    class Extra(_Well):
        def get_observation(self):
            return {"state": np.zeros(1), "surprise": np.zeros(1)}

    failures = PartContract(Extra).failures()

    assert any("never declared" in failure for failure in failures), failures


def test_the_part_contract_catches_cleanup_that_is_not_idempotent():
    class Fragile(_Well):
        def disconnect(self):
            if not self.is_connected:
                raise RuntimeError("already disconnected")
            super().disconnect()

    failures = PartContract(Fragile).failures()

    assert any("idempotent" in failure for failure in failures), failures


def test_the_part_contract_catches_an_action_naming_a_part_it_does_not_have():
    class Loose(_Well, ControllablePart):
        @property
        def action_features(self):
            return {"target": {"shape": (1,), "dtype": "float64"}}

        def send_action(self, action):
            return dict(action)

    failures = PartContract(Loose, action={"target": np.zeros(1)}).failures()

    assert any("does not have" in failure for failure in failures), failures


def test_the_connection_contract_catches_one_that_is_also_a_part():
    """The shape this refactor removed: a connection composable into a tree."""

    class Hybrid(RobotPart):
        def _open(self):
            return object()

        @property
        def parts(self):
            return {"arm": _Well()}

        @property
        def observation_features(self):
            return {}

        def get_observation(self):
            return {}

    failures = ConnectionContract(Hybrid).failures()

    assert any("is a RobotPart" in failure for failure in failures), failures


def test_the_connection_contract_catches_one_that_offers_nothing():
    class Empty(Connection):
        def _open(self):
            return object()

    failures = ConnectionContract(Empty).failures()

    assert any("backs no parts" in failure for failure in failures), failures


def test_the_robot_contract_catches_a_robot_that_keeps_what_it_built():
    """A half-built robot should never be handed back.

    The first part opens, the second refuses, and a robot that does not roll
    back leaves the first one connected with nobody holding it.
    """
    from rlinf.robotics.robot import Robot

    class Stubborn(Robot):
        ROBOT_TYPE = "Stubborn"

        def connect(self):
            try:
                super().connect()
            except Exception:
                # Swallowing the failure is the bug: the caller is told
                # nothing and the parts that did open stay open.
                pass

    def build():
        return Stubborn(first=_Well(), second=_Well())

    failures = RobotContract(build).failures()

    assert any("refused to open" in failure for failure in failures), failures


def test_assert_kept_raises_with_every_failure_listed():
    """A contributor should see all of it at once, not one thing per run."""

    class Broken(_Well):
        def get_observation(self):
            return {"surprise": np.zeros(3)}

    with pytest.raises(ConformanceError) as raised:
        PartContract(Broken).assert_kept()

    assert len(raised.value.failures) >= 2
    assert "never declared" in str(raised.value)
