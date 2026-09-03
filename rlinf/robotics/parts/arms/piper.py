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

"""AgileX Piper arm, driven through ``pyAgxArm`` over a CAN bus.

The backend registers as ``pyagxarm``, after the SDK rather than the arm, so a
driver for the same hardware over a different SDK registers beside it.

The Piper is a 6-DOF arm whose AgxGripper hangs off the same CAN bus, so the
gripper is exported as a part beneath the arm rather than wired separately.

``pyAgxArm`` reports and accepts radians, metres, and newtons, so this module
converts only what RLinf states differently: the gripper is carried as an
opening fraction rather than a width, and the end pose as a quaternion rather
than Euler angles.

Three SDK behaviours shape the lifecycle here. Reads return ``None`` until the
first CAN frame of their kind arrives, so opening waits for feedback before
handing the arm over. An effector can be initialised once per instance and
wants to exist before ``connect()``. And both ``disable()`` and the SDK's own
``reset()`` cut motor power, which drops a raised arm, so neither is used for
releasing a connection or for clearing a fault.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.base import Action, Features, Observation, RobotPart
from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.robotics.parts.views import MethodEndEffector
from rlinf.utils.logging import get_logger

from .base import Arm, BaseArm


@dataclass
class PiperRobotState:
    """State snapshot for the AgileX Piper arm."""

    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    """Tool pose ``[x, y, z, qx, qy, qz, qw]`` in the arm base frame, metres
    and a quaternion. Equal to the flange pose until a TCP offset is set."""

    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint positions ``[q1, ..., q6]`` in radians."""

    gripper_position: np.ndarray = field(default_factory=lambda: np.zeros(1))
    """Gripper opening as a fraction, ``0.0`` shut to ``1.0`` at full stroke."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


@Arm.register("pyagxarm")
class PiperArm(BaseArm):
    """AgileX Piper arm on a CAN bus, via ``pyAgxArm``.

    Args:
        channel: CAN channel. A netdev name such as ``"can0"`` for socketcan,
            or a serial path for slcan. Bring the interface up at the
            configured bitrate before connecting.
        interface: python-can interface backend: ``"socketcan"`` on Linux,
            ``"slcan"`` on macOS, ``"agx_cando"`` on Windows.
        bitrate: CAN bitrate. The Piper runs at 1 Mbit/s.
        model: Arm variant, one of ``"piper"``, ``"piper_h"``, ``"piper_l"``,
            or ``"piper_x"``. The variants differ in reach and joint travel.
        firmware: Firmware profile the SDK should speak, one of ``"default"``
            (S-V1.8-2 and older), ``"v183"``, ``"v188"``, or ``"v189"``. The
            wrong profile talks the wrong protocol to the same arm.
        speed_percent: Percentage of maximum speed for commanded motion.
        gripper_force: Gripping force in newtons, up to 3.0.
        gripper_max_width: Stroke at full opening, in metres. The AgxGripper is
            configured at the factory for either 0.07 or 0.1 m.
        with_gripper: Whether an AgxGripper is fitted. When false the arm
            exports no end effector.
    """

    SDK = ("pyAgxArm", "pyAgxArm")

    #: Number of arm joints reported as ``arm_joint_position``.
    DOF: ClassVar[int] = 6

    #: Joint travel of the standard ``piper``, in radians, as the SDK's own
    #: configuration reports it. The other variants differ, so a connected arm
    #: clips against the limits its own configuration carries.
    JOINT_LIMITS_LOWER: ClassVar[np.ndarray] = np.array(
        [-2.617994, 0.0, -2.967060, -1.745330, -1.221730, -2.094396]
    )
    JOINT_LIMITS_UPPER: ClassVar[np.ndarray] = np.array(
        [2.617994, 3.141593, 0.0, 1.745330, 1.221730, 2.094396]
    )

    #: Longest wait for the motors to enable and for the first feedback frame,
    #: in seconds.
    ENABLE_TIMEOUT_S: ClassVar[float] = 5.0
    FEEDBACK_TIMEOUT_S: ClassVar[float] = 2.0

    #: Poll interval while waiting on the arm, in seconds.
    POLL_S: ClassVar[float] = 0.01

    #: The Piper reports joints and an end pose. It exposes no Cartesian
    #: velocity, external force, or Jacobian.
    STATE_FIELDS = ("tcp_pose", "arm_joint_position")

    def __init__(
        self,
        channel: str = "can0",
        *,
        interface: str = "socketcan",
        bitrate: int = 1000000,
        model: str = "piper",
        firmware: str = "default",
        speed_percent: int = 30,
        gripper_force: float = 1.0,
        gripper_max_width: float = 0.07,
        with_gripper: bool = True,
    ) -> None:
        self._logger = get_logger()
        self._channel = channel
        self._interface = interface
        self._bitrate = int(bitrate)
        self._model = model
        self._firmware = firmware
        self._speed_percent = int(np.clip(speed_percent, 1, 100))
        self._gripper_force = float(np.clip(gripper_force, 0.0, 3.0))
        self._gripper_max_width = float(gripper_max_width)
        self._with_gripper = with_gripper
        self._robot: Any = None
        self._gripper: Any = None
        self._limits_lower = self.JOINT_LIMITS_LOWER
        self._limits_upper = self.JOINT_LIMITS_UPPER
        # One session per CAN channel: a second arm here would fight the first
        # for the bus rather than fail, and the error would name a socket.
        self._claim = DeviceClaim(f"piper-arm:{channel}", type(self).__name__)

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        **placement: Any,
    ) -> "PiperArm":
        """Declare a Piper on the CAN channel named by ``address``.

        The AgxGripper shares that bus and is initialised through the arm's own
        SDK handle, so it is neither chosen nor wired separately. Whether one
        is fitted is the ``with_gripper`` setting.
        """
        cls.refuse_unused(
            gripper_type=gripper_type,
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        settings = {
            key: placement.pop(key)
            for key in (
                "interface",
                "bitrate",
                "model",
                "firmware",
                "speed_percent",
                "gripper_force",
                "gripper_max_width",
                "with_gripper",
            )
            if key in placement
        }
        return cls(address, **settings, **placement)

    @property
    def action_features(self) -> Features:
        """Describe the absolute joint target, in radians."""
        return {"joint_position": {}}

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Return the gripper this arm carries on its own bus.

        The gripper takes any opening in range, so the view drives it through
        :meth:`move_gripper` rather than falling back to open and close.
        """
        if not self._with_gripper:
            return {}
        return {
            "end_effector": MethodEndEffector(
                self, state_field="gripper_position", command="move_gripper"
            )
        }

    def _open(self) -> Any:
        """Build the SDK driver, open the bus, and enable the motors."""
        self._claim.acquire()
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config

            # The vendor spells the keyword "firmeware_version".
            config = create_agx_arm_config(
                robot=self._model,
                firmeware_version=self._firmware,
                interface=self._interface,
                channel=self._channel,
                bitrate=self._bitrate,
            )
            robot = AgxArmFactory.create_arm(config)
            # An effector can only be initialised once per driver, and the SDK
            # asks for it before the read thread starts.
            if self._with_gripper:
                self._gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            robot.connect()
            self._robot = robot
            self._adopt_joint_limits(config)
            self._enable(robot)
            robot.set_speed_percent(self._speed_percent)
            self._await_feedback(robot)
        except BaseException:
            self._close(self._robot)
            self._claim.release()
            raise
        self._logger.info(
            "Piper %s connected on %s (%s)",
            self._model,
            self._channel,
            self._interface,
        )
        return robot

    def _adopt_joint_limits(self, config: dict) -> None:
        """Clip against the limits this variant's configuration carries."""
        limits = config.get("joint_limits") or {}
        names = config.get("joint_names") or []
        if len(names) != self.DOF or not all(name in limits for name in names):
            return
        bounds = np.asarray([limits[name] for name in names], dtype=float)
        self._limits_lower, self._limits_upper = bounds[:, 0], bounds[:, 1]

    def _enable(self, robot: Any) -> None:
        """Wait for every joint to report itself enabled.

        Raises:
            RuntimeError: If the motors do not enable in time.
        """
        deadline = time.time() + self.ENABLE_TIMEOUT_S
        while not robot.enable():
            if time.time() >= deadline:
                raise RuntimeError(
                    f"The Piper on {self._channel!r} did not enable within "
                    f"{self.ENABLE_TIMEOUT_S:g}s. Check that the arm is "
                    f"powered, that {self._channel!r} is up at "
                    f"{self._bitrate} bit/s, and that its emergency stop is "
                    "released."
                )
            time.sleep(self.POLL_S)

    def _await_feedback(self, robot: Any) -> None:
        """Wait for the first joint frame, so a read never starts empty.

        Raises:
            RuntimeError: If no joint feedback arrives in time.
        """
        deadline = time.time() + self.FEEDBACK_TIMEOUT_S
        while robot.get_joint_angles() is None:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"The Piper on {self._channel!r} enabled but sent no joint "
                    f"feedback within {self.FEEDBACK_TIMEOUT_S:g}s. Check that "
                    "the arm is in follower mode: a leader arm does not report."
                )
            time.sleep(self.POLL_S)

    def _close(self, robot: Any) -> None:
        """Drop the SDK handles, leaving the motors holding their position."""
        try:
            if robot is not None:
                robot.disconnect()
        finally:
            self._robot = None
            self._gripper = None

    def _release(self, device: Any) -> None:
        """Close the CAN session, leaving the motors holding their position.

        The arm is neither disabled nor reset here: both cut motor power, which
        drops a raised arm, and closing a connection should not move anything.
        """
        try:
            self._close(device)
        finally:
            self._claim.release()

    def get_state(self) -> PiperRobotState:
        """Read joints, tool pose, and gripper into canonical units."""
        return PiperRobotState(
            tcp_pose=self._tcp_pose(),
            arm_joint_position=np.asarray(
                self._require("joint angles", self._robot.get_joint_angles()),
                dtype=float,
            ),
            gripper_position=np.asarray([self._gripper_opening()], dtype=float),
        )

    def _require(self, what: str, reading: Any) -> Any:
        """Return a reading's payload, or say that the feed has stopped.

        Raises:
            RuntimeError: If the SDK has no message of this kind. Feedback was
                present when the arm opened, so this means the feed stopped.
        """
        if reading is None:
            raise RuntimeError(
                f"The Piper on {self._channel!r} stopped reporting {what}. "
                "The CAN feed was live when it connected, so check the bus and "
                "that the arm is still powered."
            )
        return reading.msg

    def _tcp_pose(self) -> np.ndarray:
        """Read the tool pose and convert its Euler angles to a quaternion."""
        pose = self._require("a tool pose", self._robot.get_tcp_pose())
        translation = np.asarray(pose[:3], dtype=float)
        # The SDK composes orientation as R = Rz @ Ry @ Rx, which is an
        # extrinsic xyz rotation, and returns [qx, qy, qz, qw] like RLinf.
        quaternion = R.from_euler("xyz", np.asarray(pose[3:6], dtype=float)).as_quat()
        return np.concatenate([translation, quaternion])

    def _gripper_opening(self) -> float:
        """Return the gripper opening as a fraction of its stroke."""
        if self._gripper is None:
            return 0.0
        status = self._require("gripper status", self._gripper.get_gripper_status())
        if status.mode != "width":
            raise RuntimeError(
                f"The Piper gripper on {self._channel!r} is reporting "
                f"{status.mode!r}, but this driver reads and commands a width. "
                "Configure the gripper in width mode."
            )
        return float(np.clip(status.value / self._gripper_max_width, 0.0, 1.0))

    def send_action(self, action: Action) -> Observation:
        """Move the arm joints to an absolute target in radians."""
        if set(action) != {"joint_position"}:
            raise KeyError(
                "A Piper arm action holds only 'joint_position'; the gripper "
                "is commanded through its own end-effector part."
            )
        sent = self.move_joints(action["joint_position"])
        return {"joint_position": sent}

    def move_joints(self, q_target: "Sequence[float]") -> np.ndarray:
        """Command absolute joint positions in radians.

        Targets are held to the arm's travel, because a request beyond it
        faults the arm rather than being ignored.

        Returns:
            The target actually sent, in radians.
        """
        target = np.asarray(q_target, dtype=float).reshape(-1)
        if target.shape != (self.DOF,):
            raise ValueError(
                f"Expected {self.DOF} joint targets for a Piper, got shape "
                f"{target.shape}."
            )
        target = np.clip(target, self._limits_lower, self._limits_upper)
        # move_j selects joint mode itself, so the mode is not restated here.
        self._robot.move_j([float(value) for value in target])
        return target

    def move_gripper(self, target: "Sequence[float]") -> None:
        """Command the gripper to an opening fraction in ``0..1``.

        Raises:
            RuntimeError: If no gripper is fitted.
        """
        if self._gripper is None:
            raise RuntimeError(
                f"The Piper on {self._channel!r} was declared without a "
                "gripper, so it cannot be commanded. Set with_gripper=True."
            )
        value = float(np.asarray(target, dtype=float).reshape(-1)[0])
        width = float(np.clip(value, 0.0, 1.0)) * self._gripper_max_width
        self._gripper.move_gripper_m(value=width, force=self._gripper_force)

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        self.move_gripper([1.0])

    def close_gripper(self) -> None:
        """Close the gripper fully."""
        self.move_gripper([0.0])

    def reset_joint(self, positions: "Sequence[float]", duration: float = 3.0) -> None:
        """Move to a rest pose in radians.

        The arm paces itself from ``speed_percent``, so ``duration`` is
        accepted for the arm contract and not used. This is a move, not the
        SDK's ``reset()``, which powers the motors off.
        """
        del duration
        self.move_joints(positions)

    def clear_errors(self) -> None:
        """Clear latched joint faults, without cutting motor power."""
        if self._robot is not None:
            self._robot.clear_joint_error()

    def is_robot_up(self) -> bool:
        """Report whether the CAN session is open and delivering frames."""
        return bool(
            self._robot is not None
            and self._robot.is_connected()
            and self._robot.is_ok()
        )
