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

"""Fake arm SDKs: franky, ROS, the GimArm CAN stack, and the DOSW1 SDK."""

from __future__ import annotations

import threading
import time
import types
from typing import Any

import numpy as np

from ._fakes import module, package

#: Reachable, nonzero arm state used by the fake SDKs.
HOME_JOINTS = (0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8)
HOME_TCP = (0.4, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0)


def franky() -> types.ModuleType:
    """Return a ``franky`` module that reports a fixed robot pose."""

    class Affine:
        def __init__(self, matrix=None):
            self.matrix = matrix
            self.translation = np.asarray(HOME_TCP[:3], dtype=np.float64)
            self.quaternion = np.asarray(HOME_TCP[3:], dtype=np.float64)

    class Gripper:
        """Franka Hand as libfranka exposes it: a width, in metres."""

        #: Stroke of the Franka Hand.
        MAX_WIDTH = 0.08

        def __init__(self, *_args, **_kwargs):
            self.width = 0.04
            self.max_width = self.MAX_WIDTH
            #: Commands received, for assertions.
            self.commands: list[Any] = []

        def open(self, speed):
            self.commands.append(("open", speed))
            self.width = self.max_width

        def move(self, width, speed):
            self.commands.append(("move", width, speed))
            self.width = width

        def grasp(self, width, speed, force, epsilon_inner=0.0, epsilon_outer=0.0):
            self.commands.append(("grasp", width, speed, force))
            self.width = width

        def stop(self):
            self.commands.append(("stop",))

    class Model:
        def zero_jacobian(self, _frame, _state):
            return np.zeros((6, 7))

    class Robot:
        def __init__(self, ip):
            self.ip = ip
            self.relative_dynamics_factor = 1.0
            self.model = Model()
            self.moved = []

        @property
        def state(self):
            return types.SimpleNamespace(
                O_T_EE=Affine(),
                q=np.asarray(HOME_JOINTS, dtype=np.float64),
                dq=np.zeros(7),
                # Wrench in the stiffness frame, which is what the arm reads.
                K_F_ext_hat_K=np.zeros(6),
                O_F_ext_hat_K=np.zeros(6),
                elbow=np.zeros(2),
            )

        def recover_from_errors(self):
            return True

        def set_collision_behavior(self, *_args):
            return True

        def move(self, motion):
            self.moved.append(motion)

        def join_motion(self):
            return True

    class _Tracker:
        """Record impedance targets for assertions."""

        def __init__(self, *_args, **_kwargs):
            self.targets: list[Any] = []

        def set_target(self, target, **kwargs):
            self.targets.append((target, kwargs))

        def stop(self):
            return None

    return module(
        "franky",
        Robot=Robot,
        Gripper=Gripper,
        Affine=Affine,
        Frame=types.SimpleNamespace(EndEffector="EndEffector"),
        JointImpedanceTracker=_Tracker,
        CartesianImpedanceTracker=_Tracker,
        JointMotion=lambda *a, **k: ("joint", a, k),
        CartesianMotion=lambda *a, **k: ("cartesian", a, k),
        # A joint waypoint, which a reset motion is built from.
        JointState=lambda position=None, **_k: types.SimpleNamespace(
            position=np.asarray(
                HOME_JOINTS if position is None else position, dtype=np.float64
            )
        ),
        ReferenceType=types.SimpleNamespace(Absolute="Absolute"),
    )


def ros() -> dict[str, types.ModuleType]:
    """Return fake ``rospy`` and Franka ROS message modules."""
    published: list[tuple[str, Any]] = []

    class Publisher:
        """Record the declared message type and published messages."""

        def __init__(self, name, data_class, **_kwargs):
            self.name = name
            self.data_class = data_class

        def publish(self, message):
            published.append((self.name, message))

    class Subscriber:
        """Deliver an initial message when the subscriber is created."""

        def __init__(self, name, data_class, callback, **_kwargs):
            self.name = name
            self.callback = callback
            self.data_class = data_class
            self.publish()

        def publish(self, message=None):
            """Publish a supplied or default message to the topic."""
            self.callback(message if message is not None else self.data_class())

    class _Timer:
        """Run a ROS timer callback on a daemon thread."""

        def __init__(self, period, callback, oneshot=False, **_kwargs):
            self.period = float(period)
            self.callback = callback
            self._running = True
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

        def _tick(self):
            while self._running:
                time.sleep(max(self.period, 0.001))
                if not self._running:
                    return
                try:
                    self.callback(types.SimpleNamespace(current_real=0.0))
                except Exception:  # pragma: no cover - a timer must not die
                    return

        def shutdown(self):
            self._running = False

    def _message(**fields):
        """Create a message class with default fields."""

        names = list(fields)

        def __init__(self, *positional, **overrides):
            # ROS message types take their fields positionally too, which is
            # how geometry_msgs.Point(x, y, z) is spelled at the call site.
            values = dict(fields)
            values.update(dict(zip(names, positional)))
            values.update(overrides)
            for name, value in values.items():
                setattr(self, name, value)

        return type("Message", (), {"__init__": __init__, "_fields": fields})

    rospy = module(
        "rospy",
        init_node=lambda *a, **k: None,
        Publisher=Publisher,
        Subscriber=Subscriber,
        Message=object,
        set_param=lambda *a, **k: None,
        get_param=lambda *a, **k: None,
        Time=types.SimpleNamespace(now=lambda: 0.0),
        Duration=lambda seconds: seconds,
        Timer=_Timer,
        Rate=lambda _hz: types.SimpleNamespace(sleep=lambda: None),
        sleep=lambda _seconds: None,
        loginfo=lambda *_a, **_k: None,
        logwarn=lambda *_a, **_k: None,
        is_shutdown=lambda: False,
        signal_shutdown=lambda *_a: None,
    )
    rospy.published = published

    pose = _message(
        header=types.SimpleNamespace(stamp=0.0, frame_id=""),
        pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    geometry = module(
        "geometry_msgs.msg",
        PoseStamped=pose,
        Pose=_message(),
        Point=_message(x=0.0, y=0.0, z=0.0),
        Quaternion=_message(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    franka_msgs = module(
        "franka_msgs.msg",
        FrankaState=_message(
            # Column-major, as libfranka publishes it.
            O_T_EE=list(np.eye(4).flatten(order="F")),
            q=list(HOME_JOINTS),
            dq=[0.0] * 7,
            # Wrench in the stiffness frame; the arm reads force and torque
            # out of this one message.
            K_F_ext_hat_K=[0.0] * 6,
            O_F_ext_hat_K=[0.0] * 6,
            robot_mode=2,
        ),
        ErrorRecoveryActionGoal=_message(),
    )
    serl = module(
        "serl_franka_controllers.msg", ZeroJacobian=_message(zero_jacobian=[0.0] * 42)
    )
    reconfigure = module(
        "dynamic_reconfigure.client",
        Client=lambda *a, **k: types.SimpleNamespace(
            update_configuration=lambda *_: None
        ),
    )

    def _gripper_goal():
        """Create a gripper goal with writable command fields."""
        return types.SimpleNamespace(
            width=0.0,
            speed=0.0,
            force=0.0,
            epsilon=types.SimpleNamespace(inner=0.0, outer=0.0),
        )

    gripper_msgs = module(
        "franka_gripper.msg",
        GraspActionGoal=_message(goal=_gripper_goal()),
        MoveActionGoal=_message(goal=_gripper_goal()),
    )
    sensor_msgs = module("sensor_msgs.msg", JointState=_message(position=[0.0, 0.0]))
    bridge = module(
        "cv_bridge",
        CvBridge=lambda: types.SimpleNamespace(
            imgmsg_to_cv2=lambda *_a, **_k: np.zeros((48, 64, 3), dtype=np.uint8)
        ),
    )

    made = {
        "rospy": rospy,
        "geometry_msgs": module("geometry_msgs"),
        "geometry_msgs.msg": geometry,
        "franka_msgs": module("franka_msgs"),
        "franka_msgs.msg": franka_msgs,
        "serl_franka_controllers": module("serl_franka_controllers"),
        "serl_franka_controllers.msg": serl,
        "dynamic_reconfigure": module("dynamic_reconfigure"),
        "dynamic_reconfigure.client": reconfigure,
        "franka_gripper": module("franka_gripper"),
        "franka_gripper.msg": gripper_msgs,
        "sensor_msgs": module("sensor_msgs"),
        "sensor_msgs.msg": sensor_msgs,
        "cv_bridge": bridge,
    }
    for dotted, leaf in (
        ("geometry_msgs", geometry),
        ("franka_msgs", franka_msgs),
        ("serl_franka_controllers", serl),
        ("dynamic_reconfigure", reconfigure),
        ("franka_gripper", gripper_msgs),
        ("sensor_msgs", sensor_msgs),
    ):
        setattr(made[dotted], dotted.split(".")[-1] if False else "msg", leaf)
    made["dynamic_reconfigure"].client = reconfigure
    return made


def lerobot() -> dict[str, types.ModuleType]:
    """Fake lerobot package exposing an SO-101 follower.

    The follower holds one position per motor in lerobot's own units --
    degrees for the arm joints, ``0..100`` for the gripper -- so a test can
    assert that the driver converts rather than passes values through.
    """

    class FakeSO101FollowerConfig:
        def __init__(
            self,
            port: str,
            id: Any = None,
            cameras: Any = None,
            max_relative_target: Any = None,
            use_degrees: bool = False,
            **extra: Any,
        ) -> None:
            self.port = port
            self.id = id
            self.cameras = dict(cameras or {})
            self.max_relative_target = max_relative_target
            self.use_degrees = use_degrees
            self.extra = extra

    class FakeSO101Follower:
        #: Set false to model an arm whose calibration file is missing.
        calibrated = True

        def __init__(self, config: Any) -> None:
            self.config = config
            self.is_connected = False
            self.is_calibrated = type(self).calibrated
            self.calibrate_calls = 0
            self.sent: list[dict[str, float]] = []
            self.positions = {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
            }

        def connect(self, calibrate: bool = True) -> None:
            self.is_connected = True
            if calibrate:
                self.calibrate()

        def calibrate(self) -> None:
            # The real one blocks on input(); count calls so a test can show
            # the driver never reaches it.
            self.calibrate_calls += 1

        def disconnect(self) -> None:
            self.is_connected = False

        def get_observation(self) -> dict[str, float]:
            return dict(self.positions)

        def send_action(self, action: dict[str, float]) -> dict[str, float]:
            self.sent.append(dict(action))
            self.positions.update(action)
            return dict(action)

    made = {parent.__name__: parent for parent in package("lerobot.robots.so_follower")}
    # lerobot 0.4 merged the SO-family followers into so_follower; the driver
    # prefers that path and falls back to the older one, so fake both.
    for leaf in ("so_follower", "so101_follower"):
        follower = module(
            f"lerobot.robots.{leaf}",
            SO101Follower=FakeSO101Follower,
            SO101FollowerConfig=FakeSO101FollowerConfig,
        )
        made[f"lerobot.robots.{leaf}"] = follower
        setattr(made["lerobot.robots"], leaf, follower)
    made["lerobot"].robots = made["lerobot.robots"]
    # The driver gates on the Feetech SDK, which lerobot's feetech extra brings.
    made["scservo_sdk"] = module("scservo_sdk")
    return made


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Return fake arm SDKs keyed by import name."""
    made = {"franky": franky()}
    made.update(ros())
    made.update(lerobot())
    return made
