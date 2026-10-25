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

"""Robbyant R2 robot connection and part views over the RFrame middleware."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.base import (
    Connection,
    FaultLevel,
    Features,
    HealthStatus,
    RobotPart,
)
from rlinf.robotics.parts.end_effectors import BaseHand
from rlinf.robotics.parts.mobility.base import MobileBase
from rlinf.robotics.parts.views import (
    MethodArm,
    MethodCamera,
    MethodEndEffector,
    host_state,
)
from rlinf.utils.logging import get_logger

R2_MODE_HOLD = 0
R2_MODE_MOVEJ = 1
R2_MODE_SERVOJ = 2
R2_MODE_GRAVITY_COMP = 3

R2_SLAVE_ADDR_TO_JOINT: dict[int, str] = {
    1002: "J_LeftArm1",
    1003: "J_LeftArm2",
    1004: "J_LeftArm3",
    1005: "J_LeftArm4",
    1006: "J_LeftArm5",
    1007: "J_LeftArm6",
    1008: "J_LeftArm7",
    1012: "J_RightArm1",
    1013: "J_RightArm2",
    1014: "J_RightArm3",
    1015: "J_RightArm4",
    1016: "J_RightArm5",
    1017: "J_RightArm6",
    1018: "J_RightArm7",
    1022: "J_NeckYaw",
    1023: "J_NeckPitch",
    1024: "J_WaistRoll",
    1025: "J_WaistYaw",
    1026: "J_Hip",
    1027: "J_Knee",
    1028: "J_Ankle",
}

R2_BODY_SLAVE_ADDRS: tuple[int, ...] = tuple(R2_SLAVE_ADDR_TO_JOINT.keys())

R2_ERROR_CODES: dict[int, str] = {
    0: "SUCCESS",
    301: "JOINT_LIMIT_EXCEEDED",
    302: "VELOCITY_LIMIT_EXCEEDED",
    303: "ACCELERATION_LIMIT_EXCEEDED",
    304: "COLLISION_DETECTED",
    401: "IK_NO_SOLUTION",
    402: "SINGULARITY_DETECTED",
    403: "TRAJECTORY_PLANNING_FAILED",
    501: "AXIS_NOT_ENABLED",
    502: "AXIS_FAULT",
    503: "EMERGENCY_STOP_ACTIVE",
    504: "COMMUNICATION_TIMEOUT",
    505: "PROTECTIVE_STOP_ACTIVE",
    601: "INVALID_PARAMETER",
    602: "MODE_NOT_SUPPORTED",
    603: "ROBOT_BUSY",
    604: "TASK_ID_CONFLICT",
}

R2_LEFT_ARM_JOINTS: tuple[str, ...] = (
    "J_ArmL_1",
    "J_ArmL_2",
    "J_ArmL_3",
    "J_ArmL_4",
    "J_ArmL_5",
    "J_ArmL_6",
    "J_ArmL_7",
)
R2_RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "J_ArmR_1",
    "J_ArmR_2",
    "J_ArmR_3",
    "J_ArmR_4",
    "J_ArmR_5",
    "J_ArmR_6",
    "J_ArmR_7",
)
R2_REAL_LEFT_ARM_JOINTS: tuple[str, ...] = (
    "J_LeftArm1",
    "J_LeftArm2",
    "J_LeftArm3",
    "J_LeftArm4",
    "J_LeftArm5",
    "J_LeftArm6",
    "J_LeftArm7",
)
R2_REAL_RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "J_RightArm1",
    "J_RightArm2",
    "J_RightArm3",
    "J_RightArm4",
    "J_RightArm5",
    "J_RightArm6",
    "J_RightArm7",
)
R2_NECK_JOINTS: tuple[str, ...] = ("J_NeckYaw", "J_NeckPitch")
R2_REAL_NECK_JOINTS_21DOF: tuple[str, ...] = ("J_NeckYaw", "J_NeckPitch")
R2_REAL_NECK_JOINTS_20DOF: tuple[str, ...] = ("J_NeckPitch",)
R2_WAIST_JOINTS_21DOF: tuple[str, ...] = (
    "J_WaistPitch_1",
    "J_WaistPitch_2",
    "J_WaistPitch_3",
    "J_WaistRoll",
    "J_WaistYaw",
)
R2_WAIST_JOINTS_20DOF: tuple[str, ...] = (
    "J_WaistLift",
    "J_WaistPitch",
    "J_WaistRoll",
    "J_WaistYaw",
)
R2_REAL_WAIST_JOINTS: tuple[str, ...] = (
    "J_AnklePitch",
    "J_Knee",
    "J_HipPitch",
    "J_HipRoll",
    "J_HipYaw",
)
R2_GRIPPER_JOINTS: tuple[str, ...] = ("GripperL", "GripperR")
R2_HAND_FINGER_ORDER: tuple[str, ...] = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
)
R2_HAND_JOINTS_LEFT: tuple[str, ...] = (
    "HandL_Little",
    "HandL_Ring",
    "HandL_Middle",
    "HandL_Index",
    "HandL_ThumbBend",
    "HandL_ThumbRot",
)
R2_HAND_JOINTS_RIGHT: tuple[str, ...] = (
    "HandR_Little",
    "HandR_Ring",
    "HandR_Middle",
    "HandR_Index",
    "HandR_ThumbBend",
    "HandR_ThumbRot",
)

R2_LEFT_ARM_LIMITS_LOW = np.array(
    [-2.74, 0.0, -2.82, -2.33, -2.82, -1.45, -1.45], dtype=np.float32
)
R2_LEFT_ARM_LIMITS_HIGH = np.array(
    [1.34, 3.14, 2.82, 0.02, 2.82, 1.45, 1.45], dtype=np.float32
)
R2_RIGHT_ARM_LIMITS_LOW = np.array(
    [-2.74, -3.14, -2.82, -0.02, -2.82, -1.45, -1.45], dtype=np.float32
)
R2_RIGHT_ARM_LIMITS_HIGH = np.array(
    [1.34, 0.0, 2.82, 2.33, 2.82, 1.45, 1.45], dtype=np.float32
)
R2_WAIST_LIMITS_LOW = np.array([0.0, 0.0, -0.38, -0.47, -1.95], dtype=np.float32)
R2_WAIST_LIMITS_HIGH = np.array([1.48, 2.39, 1.62, 0.47, 1.95], dtype=np.float32)
R2_NECK_LIMITS_LOW = np.array([-1.57, -0.698], dtype=np.float32)
R2_NECK_LIMITS_HIGH = np.array([1.57, 0.349], dtype=np.float32)

R2_LEFT_READY_QPOS = np.array(
    [-0.35, 0.20, 1.00, -0.70, 0.0, -0.30, 0.0], dtype=np.float32
)
R2_RIGHT_READY_QPOS = np.array(
    [-0.35, -0.20, 1.00, 0.70, 0.0, -0.30, 0.0], dtype=np.float32
)

_R2_WAIST_URDF_CHAIN: tuple[tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([0.0, 0.073, 0.345]), np.array([0.0, -1.0, 0.0])),
    (np.array([0.0, 0.0, 0.34]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, 0.3]), np.array([0.0, -1.0, 0.0])),
    (np.array([-0.056, -0.0732, 0.14]), np.array([1.0, 0.0, 0.0])),
    (np.array([0.056, 0.0, 0.065]), np.array([0.0, 0.0, 1.0])),
)

_R2_LEFT_ARM_URDF_CHAIN: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...] = (
    ("fixed", np.array([0.0, 0.0925, 0.242]), np.zeros(3), np.zeros(3)),
    ("rev", np.array([0.0, 0.0229, 0.0]), np.zeros(3), np.array([0.0, 1.0, 0.0])),
    ("rev", np.array([-0.0594, 0.058, 0.0]), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("rev", np.array([0.06, 0.0, -0.09]), np.zeros(3), np.array([0.0, 0.0, 1.0])),
    ("rev", np.array([-0.059, -0.01, -0.21]), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("rev", np.array([0.059, 0.01, -0.0956]), np.zeros(3), np.array([0.0, 0.0, 1.0])),
    ("rev", np.array([0.0, 0.034, -0.1374]), np.zeros(3), np.array([0.0, -1.0, 0.0])),
    (
        "rev",
        np.array([-0.0364, -0.034, -0.058]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
    ),
    ("fixed", np.array([0.0364, 0.0, -0.058]), np.zeros(3), np.zeros(3)),
    (
        "fixed",
        np.array([0.0, 0.0, -0.013]),
        np.array([3.1415926, 0.0, -1.5707963]),
        np.zeros(3),
    ),
    (
        "fixed",
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.571428]),
        np.zeros(3),
    ),
    ("fixed", np.array([0.0, 0.0, 0.122]), np.array([0.0, 0.0, -1.57]), np.zeros(3)),
)

_R2_RIGHT_ARM_URDF_CHAIN: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...] = (
    ("fixed", np.array([0.0, -0.0925, 0.242]), np.zeros(3), np.zeros(3)),
    ("rev", np.array([0.0, -0.0229, 0.0]), np.zeros(3), np.array([0.0, 1.0, 0.0])),
    ("rev", np.array([-0.0594, -0.058, 0.0]), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("rev", np.array([0.06, 0.0, -0.09]), np.zeros(3), np.array([0.0, 0.0, -1.0])),
    ("rev", np.array([-0.059, 0.01, -0.21]), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("rev", np.array([0.059, -0.01, -0.0956]), np.zeros(3), np.array([0.0, 0.0, -1.0])),
    ("rev", np.array([0.0, 0.034, -0.1374]), np.zeros(3), np.array([0.0, -1.0, 0.0])),
    (
        "rev",
        np.array([-0.0364, -0.034, -0.058]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
    ),
    ("fixed", np.array([0.0364, 0.0, -0.058]), np.zeros(3), np.zeros(3)),
    (
        "fixed",
        np.array([0.0, 0.0, -0.013]),
        np.array([3.1415926, 0.0, -1.5707963]),
        np.zeros(3),
    ),
    ("fixed", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.57]), np.zeros(3)),
    ("fixed", np.array([0.0, 0.0, 0.122]), np.array([0.0, 0.0, -1.57]), np.zeros(3)),
)


def compute_r2_arm_fk_and_jacobian(
    side: str,
    q_arm: Sequence[float] | np.ndarray,
    q_waist: Optional[Sequence[float] | np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TCP position, Euler RPY, and 3x7 position/rotation Jacobians in ``base_link``."""
    if side not in ("left", "right"):
        raise ValueError(f"Expected side 'left' or 'right', got {side!r}.")
    q_a = np.asarray(q_arm, dtype=np.float64).reshape(7)
    w = (
        np.zeros(5, dtype=np.float64)
        if q_waist is None
        else np.asarray(q_waist, dtype=np.float64).reshape(-1)
    )
    T = np.eye(4, dtype=np.float64)
    for idx, (xyz, axis) in enumerate(_R2_WAIST_URDF_CHAIN):
        T_o = np.eye(4, dtype=np.float64)
        T_o[:3, 3] = xyz
        T = T @ T_o
        if idx < w.size and abs(w[idx]) > 0.0:
            T_q = np.eye(4, dtype=np.float64)
            T_q[:3, :3] = R.from_rotvec(axis * float(w[idx])).as_matrix()
            T = T @ T_q

    chain = _R2_LEFT_ARM_URDF_CHAIN if side == "left" else _R2_RIGHT_ARM_URDF_CHAIN
    origins: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    a_idx = 0
    for jtype, xyz, rpy, axis in chain:
        T_o = np.eye(4, dtype=np.float64)
        if np.any(rpy):
            T_o[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
        T_o[:3, 3] = xyz
        T = T @ T_o
        if jtype == "rev":
            origins.append(T[:3, 3].copy())
            axes.append(T[:3, :3] @ axis)
            T_q = np.eye(4, dtype=np.float64)
            T_q[:3, :3] = R.from_rotvec(axis * float(q_a[a_idx])).as_matrix()
            T = T @ T_q
            a_idx += 1

    p_tip = T[:3, 3].copy()
    rpy_tip = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
    J_pos = np.zeros((3, 7), dtype=np.float64)
    J_rot = np.zeros((3, 7), dtype=np.float64)
    for i in range(7):
        J_pos[:, i] = np.cross(axes[i], p_tip - origins[i])
        J_rot[:, i] = axes[i]
    return (
        p_tip.astype(np.float32),
        rpy_tip.astype(np.float32),
        J_pos.astype(np.float32),
        J_rot.astype(np.float32),
    )


def compute_r2_arm_cartesian_step(
    side: str,
    q_init: Sequence[float] | np.ndarray,
    delta_xyz: Sequence[float] | np.ndarray,
    *,
    q_waist: Optional[Sequence[float] | np.ndarray] = None,
    max_step_rad: float = 0.35,
) -> np.ndarray:
    """Solve a joint-limit-projected damped-least-squares IK step for a 3D TCP delta ``[dx, dy, dz]``."""
    lo = (R2_LEFT_ARM_LIMITS_LOW if side == "left" else R2_RIGHT_ARM_LIMITS_LOW).astype(
        np.float64
    )
    hi = (
        R2_LEFT_ARM_LIMITS_HIGH if side == "left" else R2_RIGHT_ARM_LIMITS_HIGH
    ).astype(np.float64)
    q0 = np.clip(np.asarray(q_init, dtype=np.float64).reshape(7), lo, hi)
    q = q0.copy()
    d_xyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    p0, _, Jp0, _ = compute_r2_arm_fk_and_jacobian(side, q, q_waist)
    target_p = p0.astype(np.float64) + d_xyz

    if d_xyz[2] > 0.0 and float(np.linalg.norm(Jp0[2, :])) < 0.12:
        out_sign = 1.0 if side == "left" else -1.0
        elbow_sign = -1.0 if side == "left" else 1.0
        q[0] = np.clip(q[0] - 0.15, lo[0], hi[0])
        q[1] = np.clip(q[1] + out_sign * 0.18, lo[1], hi[1])
        q[3] = np.clip(q[3] + elbow_sign * 0.30, lo[3], hi[3])

    weights = np.diag([1.0, 1.0, 0.8, 1.0, 0.5, 0.4, 0.4])
    for _ in range(30):
        p, _, Jp_f32, _ = compute_r2_arm_fk_and_jacobian(side, q, q_waist)
        err = target_p - p.astype(np.float64)
        if float(np.linalg.norm(err)) < 1e-4:
            break
        J_eff = Jp_f32.astype(np.float64).copy()
        for j in range(7):
            if (q[j] <= lo[j] + 1e-3 and float(np.dot(J_eff[:, j], err)) < 0.0) or (
                q[j] >= hi[j] - 1e-3 and float(np.dot(J_eff[:, j], err)) > 0.0
            ):
                J_eff[:, j] = 0.0
        JJT = J_eff @ weights @ J_eff.T + (0.03**2) * np.eye(3, dtype=np.float64)
        dq = weights @ J_eff.T @ np.linalg.solve(JJT, err)
        q = np.clip(q + 0.6 * dq, lo, hi)

    dq_total = q - q0
    max_abs = float(np.max(np.abs(dq_total)))
    if max_abs > max_step_rad:
        dq_total = dq_total * (max_step_rad / max_abs)
    return np.clip(q0 + dq_total, lo, hi).astype(np.float32)


R2_CAMERA_TOPICS: dict[str, dict[str, Any]] = {
    "head": {
        "color": "/drivers/camera/zed/image_raw",
        "compressed": "/drivers/camera/zed/image_raw/compressed",
        "depth": "/drivers/camera/zed/depth",
        "real_color": "/robby/sensors/camera/head/color/raw",
        "real_compressed": "/robby/sensors/camera/head/color/compressed",
        "real_depth": "/robby/sensors/camera/head/depth/raw",
        "resolution": (1280, 720),
    },
    "wrist_left": {
        "color": "/drivers/camera/wrist_left/color/image_raw",
        "compressed": "/drivers/camera/wrist_left/color/image_raw/compressed",
        "depth": "/drivers/camera/wrist_left/depth/image_rect_raw",
        "real_color": "/robby/sensors/camera/wrist_left/color/raw",
        "real_compressed": "/robby/sensors/camera/wrist_left/color/compressed",
        "real_depth": "/robby/sensors/camera/wrist_left/depth/raw",
        "resolution": (640, 480),
    },
    "wrist_right": {
        "color": "/drivers/camera/wrist_right/color/image_raw",
        "compressed": "/drivers/camera/wrist_right/color/image_raw/compressed",
        "depth": "/drivers/camera/wrist_right/depth/image_rect_raw",
        "real_color": "/robby/sensors/camera/wrist_right/color/raw",
        "real_compressed": "/robby/sensors/camera/wrist_right/color/compressed",
        "real_depth": "/robby/sensors/camera/wrist_right/depth/raw",
        "resolution": (640, 480),
    },
    "navigation_front": {
        "color": "/drivers/camera/navigation_front/color/image_raw",
        "compressed": "/drivers/camera/navigation_front/color/image_raw/compressed",
        "depth": "/drivers/camera/navigation_front/depth/image_rect_raw",
        "real_color": "/robby/sensors/camera/chassis_front/color/raw",
        "real_compressed": "/robby/sensors/camera/chassis_front/color/compressed",
        "real_depth": "/robby/sensors/camera/chassis_front/depth/raw",
        "resolution": (640, 480),
    },
    "navigation_rear": {
        "color": "/drivers/camera/navigation_rear/color/image_raw",
        "compressed": "/drivers/camera/navigation_rear/color/image_raw/compressed",
        "depth": "/drivers/camera/navigation_rear/depth/image_rect_raw",
        "real_color": "/robby/sensors/camera/chassis_rear/color/raw",
        "real_compressed": "/robby/sensors/camera/chassis_rear/color/compressed",
        "real_depth": "/robby/sensors/camera/chassis_rear/depth/raw",
        "resolution": (640, 480),
    },
}

_SHARED_RFRAME_APP: Any = None
_SHARED_RFRAME_LOCK = threading.Lock()


def _get_shared_rframe_app(rframe_mod: Any) -> Any:
    """Return a process-wide initialized ``rframe.App`` singleton."""
    global _SHARED_RFRAME_APP
    with _SHARED_RFRAME_LOCK:
        if "FASTRTPS_DEFAULT_PROFILES_FILE" not in os.environ and os.path.isfile(
            "/opt/robbyant/dynamic_participant.xml"
        ):
            os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = (
                "/opt/robbyant/dynamic_participant.xml"
            )
        if _SHARED_RFRAME_APP is None:
            app = rframe_mod.App()
            if hasattr(app, "init"):
                app.init()
            _SHARED_RFRAME_APP = app
        return _SHARED_RFRAME_APP


def _ros2_type_name(msg_cls: Any) -> str:
    """Derive a ``pkg/msg/Type`` string from a ROS2 Python message class."""
    mod = getattr(msg_cls, "__module__", "")
    parts = mod.split(".")
    if len(parts) >= 2 and parts[1] in ("msg", "srv"):
        return f"{parts[0]}/{parts[1]}/{msg_cls.__name__}"
    return msg_cls.__name__


def _pose_to_xyz_rpy(pose: Any) -> np.ndarray:
    """Convert a ROS geometry_msgs/Pose into a 6-element [x, y, z, r, p, y] array."""
    pos = pose.position
    ori = pose.orientation
    quat = np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        rpy = np.zeros(3, dtype=np.float64)
    else:
        rpy = R.from_quat(quat / norm).as_euler("xyz", degrees=False)
    return np.array([pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32)


def _xyz_rpy_to_pose(pose_vec: np.ndarray, pose_cls: Any) -> Any:
    """Populate a ROS geometry_msgs/Pose from a 6-element [x, y, z, r, p, y] vector."""
    arr = np.asarray(pose_vec, dtype=np.float64).reshape(6)
    quat = R.from_euler("xyz", arr[3:6], degrees=False).as_quat()
    msg = pose_cls()
    msg.position.x = float(arr[0])
    msg.position.y = float(arr[1])
    msg.position.z = float(arr[2])
    msg.orientation.x = float(quat[0])
    msg.orientation.y = float(quat[1])
    msg.orientation.z = float(quat[2])
    msg.orientation.w = float(quat[3])
    return msg


def _decode_image_message(
    msg: Any,
    *,
    compressed: bool = False,
    depth: bool = False,
    fallback_resolution: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Decode a sensor_msgs/Image or CompressedImage message into a NumPy array."""
    width_default, height_default = fallback_resolution
    if compressed:
        raw = bytes(getattr(msg, "data", b""))
        if not raw:
            return np.zeros((height_default, width_default, 3), dtype=np.uint8)
        try:
            import cv2

            buf = np.frombuffer(raw, dtype=np.uint8)
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if decoded is not None:
                return np.ascontiguousarray(decoded[:, :, ::-1], dtype=np.uint8)
        except ImportError:
            pass
        expected = height_default * width_default * 3
        if len(raw) >= expected:
            return (
                np.frombuffer(raw[:expected], dtype=np.uint8)
                .reshape((height_default, width_default, 3))
                .copy()
            )
        return np.zeros((height_default, width_default, 3), dtype=np.uint8)

    height = int(getattr(msg, "height", 0) or height_default)
    width = int(getattr(msg, "width", 0) or width_default)
    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    raw = bytes(getattr(msg, "data", b""))

    if depth:
        if "16uc1" in encoding or "mono16" in encoding:
            count = height * width
            if len(raw) >= count * 2:
                arr = np.frombuffer(raw[: count * 2], dtype=np.uint16).reshape(
                    (height, width)
                )
                return arr.astype(np.float32) * 0.001
        count = height * width
        if len(raw) >= count * 4:
            return (
                np.frombuffer(raw[: count * 4], dtype=np.float32)
                .reshape((height, width))
                .copy()
            )
        return np.zeros((height, width), dtype=np.float32)

    if (
        encoding in ("16uc1", "yuyv", "yuv422", "yuv422_yuy2")
        or len(raw) == height * width * 2
    ):
        count = height * width * 2
        if len(raw) >= count:
            yuyv = np.frombuffer(raw[:count], dtype=np.uint8).reshape(
                (height, width, 2)
            )
            try:
                import cv2

                return np.ascontiguousarray(
                    cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2), dtype=np.uint8
                )
            except ImportError:
                y = yuyv[:, :, 0].astype(np.float32)
                u = np.repeat(yuyv[:, 0::2, 1].astype(np.float32) - 128.0, 2, axis=1)
                v = np.repeat(yuyv[:, 1::2, 1].astype(np.float32) - 128.0, 2, axis=1)
                r = np.clip(y + 1.402 * v, 0, 255).astype(np.uint8)
                g = np.clip(y - 0.344136 * u - 0.714136 * v, 0, 255).astype(np.uint8)
                b = np.clip(y + 1.772 * u, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(np.stack([r, g, b], axis=-1))

    channels = 4 if ("rgba" in encoding or "bgra" in encoding) else 3
    count = height * width * channels
    if len(raw) < count:
        return np.zeros((height, width, 3), dtype=np.uint8)
    frame = np.frombuffer(raw[:count], dtype=np.uint8).reshape(
        (height, width, channels)
    )
    if "bgr" in encoding:
        frame = frame[:, :, :3][:, :, ::-1]
    elif channels == 4:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame, dtype=np.uint8)


def infer_r2_hardware(
    names: Sequence[str] = (),
    positions: Sequence[float] = (),
) -> tuple[Optional[int], Optional[str]]:
    """Infer ``(body_dof, end_effector_type)`` from a ``JointState`` sample.

    Returns ``None`` for either field when the message does not carry enough
    information to distinguish the hardware variant.
    """
    name_set = set(names)
    body_dof: Optional[int] = None
    end_effector_type: Optional[str] = None

    if name_set:
        has_21 = any(
            j in name_set
            for j in ("J_WaistPitch_1", "J_WaistPitch_2", "J_WaistPitch_3")
        )
        has_20 = any(j in name_set for j in ("J_WaistLift", "J_WaistPitch"))
        has_real_joints = any(
            j in name_set
            for j in R2_REAL_WAIST_JOINTS
            + R2_REAL_LEFT_ARM_JOINTS
            + R2_REAL_RIGHT_ARM_JOINTS
        )
        if has_21 and not has_20:
            body_dof = 21
        elif has_20 and not has_21:
            body_dof = 20
        elif has_21:
            body_dof = 21
        elif has_real_joints:
            body_dof = (
                21 if ("J_NeckYaw" in name_set and "J_NeckPitch" in name_set) else 20
            )

        has_hand = any(j.startswith(("HandL_", "HandR_")) for j in name_set)
        has_gripper = "GripperL" in name_set or "GripperR" in name_set
        if has_hand and not has_gripper:
            end_effector_type = "hand"
        elif has_gripper and not has_hand:
            end_effector_type = "gripper"
        elif has_gripper:
            end_effector_type = "gripper"
        elif any(
            j in name_set
            for j in R2_LEFT_ARM_JOINTS
            + R2_RIGHT_ARM_JOINTS
            + R2_REAL_LEFT_ARM_JOINTS
            + R2_REAL_RIGHT_ARM_JOINTS
        ):
            end_effector_type = "none"
    elif positions:
        n = len(positions)
        if n in (20, 22, 32):
            body_dof = 20
        elif n in (21, 23, 25, 33):
            body_dof = 21
        if n in (32, 33):
            end_effector_type = "hand"
        elif n in (22, 23, 25):
            end_effector_type = "gripper"
        elif n in (20, 21):
            end_effector_type = "none"

    return body_dof, end_effector_type


@dataclass
class R2RobotState:
    """Snapshot of the R2 dual-arm, torso, end-effector, and chassis state."""

    left_tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    left_tcp_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    left_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    left_joint_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    left_joint_torques: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    left_wrench: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    left_gripper_position: np.ndarray = field(
        default_factory=lambda: np.ones(1, dtype=np.float32)
    )
    left_hand_position: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )

    right_tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    right_tcp_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    right_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    right_joint_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    right_joint_torques: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    right_wrench: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    right_gripper_position: np.ndarray = field(
        default_factory=lambda: np.ones(1, dtype=np.float32)
    )
    right_hand_position: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )

    waist_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32)
    )
    waist_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32)
    )
    neck_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    neck_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )

    base_pose: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    imu_orientation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    imu_angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    imu_linear_acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    battery_percentage: float = 100.0
    is_healthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state snapshot into a plain dictionary."""
        return asdict(self)


class R2Arm(MethodArm):
    """Single 7-DoF R2 arm view backed by :class:`R2Connection`."""

    def __init__(self, host: Any, side: str) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"Expected R2 arm side 'left' or 'right', got {side!r}.")
        self.side = side
        super().__init__(
            host,
            commands={
                "joint_position": f"move_{side}_arm_joint",
                "joint_positions": f"move_{side}_arm_joint",
                "move_joint": f"move_{side}_arm_joint",
                "tcp_pose": f"move_{side}_arm_cartesian",
                "move_cartesian": f"move_{side}_arm_cartesian",
                "move_Cartesian": f"move_{side}_arm_cartesian",
                "tcp_velocity": f"move_{side}_arm_velocity",
                "move_cartesian_velocity": f"move_{side}_arm_velocity",
                "joint_velocity": f"move_{side}_arm_joint_velocity",
                "move_joint_velocity": f"move_{side}_arm_joint_velocity",
                "joint_torque": f"apply_{side}_arm_torque",
                "apply_torque": f"apply_{side}_arm_torque",
            },
            state_fields={
                "tcp_pose": f"{side}_tcp_pose",
                "tcp_vel": f"{side}_tcp_vel",
                "joint_position": f"{side}_joint_positions",
                "joint_positions": f"{side}_joint_positions",
                "joint_velocity": f"{side}_joint_velocities",
                "joint_velocities": f"{side}_joint_velocities",
                "joint_torque": f"{side}_joint_torques",
                "joint_torques": f"{side}_joint_torques",
                "wrench": f"{side}_wrench",
            },
        )
        self.num_joints = 7

    @property
    def observation_features(self) -> Features:
        """Declare shapes and dtypes of the 7-DoF arm state fields."""
        return {
            "tcp_pose": {"shape": (6,), "dtype": "float32"},
            "tcp_vel": {"shape": (6,), "dtype": "float32"},
            "joint_position": {"shape": (7,), "dtype": "float32"},
            "joint_positions": {"shape": (7,), "dtype": "float32"},
            "joint_velocity": {"shape": (7,), "dtype": "float32"},
            "joint_velocities": {"shape": (7,), "dtype": "float32"},
            "joint_torque": {"shape": (7,), "dtype": "float32"},
            "joint_torques": {"shape": (7,), "dtype": "float32"},
            "wrench": {"shape": (6,), "dtype": "float32"},
        }

    def reset_joint(
        self,
        positions: Sequence[float] | np.ndarray,
        duration: float = 3.0,
        *,
        max_velocity: Optional[float | Sequence[float] | np.ndarray] = None,
    ) -> bool:
        """Move the arm smoothly to ``positions`` using the MoveJ service."""
        fn = getattr(self._host, f"move_{self.side}_arm_movej")
        return bool(fn(positions, duration=duration, max_velocity=max_velocity))

    def clear_errors(self) -> bool:
        """Clear motion controller faults on the shared R2 connection."""
        return bool(self._host.clear_errors())


class R2HandView(MethodEndEffector, BaseHand):
    """Six-DoF R2 dexterous hand view backed by :class:`R2Connection`."""

    FINGER_NAMES = R2_HAND_FINGER_ORDER

    def __init__(self, host: Any, side: str) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"Expected R2 hand side 'left' or 'right', got {side!r}.")
        self.side = side
        super().__init__(
            host,
            f"{side}_hand_position",
            dims=len(R2_HAND_FINGER_ORDER),
            command=f"move_{side}_hand",
            is_gripper=False,
        )


class R2MobileBase(MobileBase):
    """Omnidirectional wheeled chassis view backed by :class:`R2Connection`."""

    def __init__(self, host: Any) -> None:
        self._host = self._owner = host

    @property
    def observation_features(self) -> dict[str, dict[str, Any]]:
        """planar pose ``[x, y, yaw]`` and velocity ``[vx, vy, wz]``."""
        return {
            "base_pose": {"shape": (3,), "dtype": "float32"},
            "base_velocity": {"shape": (3,), "dtype": "float32"},
        }

    @property
    def action_features(self) -> dict[str, dict[str, Any]]:
        """Planar chassis velocity command ``[vx, vy, wz]``."""
        return {
            "base_velocity": {"shape": (3,), "dtype": "float32"},
        }

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return the latest planar pose and velocity from the host state."""
        state = host_state(self._host)
        return {
            "base_pose": np.asarray(state["base_pose"], dtype=np.float32).reshape(3),
            "base_velocity": np.asarray(
                state["base_velocity"], dtype=np.float32
            ).reshape(3),
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Publish a planar chassis velocity command ``[vx, vy, wz]``."""
        if set(action) != {"base_velocity"}:
            raise ValueError(
                f"R2MobileBase expects action key 'base_velocity', got {sorted(action)}."
            )
        vel = np.asarray(action["base_velocity"], dtype=np.float32).reshape(3)
        self._host.move_base(vel)
        return {"base_velocity": vel}

    def get_pose(self) -> np.ndarray:
        """Return the current planar pose ``[x, y, yaw]``."""
        return self.get_observation()["base_pose"]

    def get_velocity(self) -> np.ndarray:
        """Return the current planar velocity ``[vx, vy, wz]``."""
        return self.get_observation()["base_velocity"]

    def move_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Command planar chassis velocity ``(vx, vy, wz)``."""
        self._host.move_base(np.array([vx, vy, wz], dtype=np.float32))

    def navigate_to(
        self, x: float, y: float, yaw: float = 0.0, timeout: float = 30.0
    ) -> bool:
        """Dispatch a waypoint navigation goal to the R2 navigation stack."""
        return bool(self._host.navigate_to(x, y, yaw=yaw, timeout=timeout))

    def stop(self) -> None:
        """Zero chassis velocity and stop active navigation."""
        self._host.stop_base()


class R2Camera(MethodCamera):
    """Camera view over an RFrame image topic on :class:`R2Connection`."""

    def __init__(
        self,
        host: Any,
        camera_name: str,
        resolution: tuple[int, int] = (640, 480),
        *,
        enable_depth: bool = False,
    ) -> None:
        super().__init__(host, "get_camera", camera_name, check_method="check_camera")
        self.name = camera_name
        self.camera_name = camera_name
        self.resolution = resolution
        self.enable_depth = enable_depth

    @property
    def observation_features(self) -> dict[str, dict[str, Any]]:
        """Declare RGB frame and optional depth image shapes."""
        width, height = self.resolution
        features: dict[str, dict[str, Any]] = {
            "frame": {"shape": (height, width, 3), "dtype": "uint8"},
        }
        if self.enable_depth:
            features["depth"] = {"shape": (height, width), "dtype": "float32"}
        return features

    def get_observation(self) -> dict[str, Any]:
        """Return the latest RGB frame and optional depth image."""
        if not self.enable_depth:
            return {"frame": self._host.get_camera(self.camera_name)}
        return self._host.get_camera_observation(self.camera_name)

    def get_frame(self, **_kwargs: Any) -> np.ndarray:
        """Return the latest RGB image array."""
        return self._host.get_camera(self.camera_name)

    def get_depth(self) -> Optional[np.ndarray]:
        """Return the latest depth image in meters when depth is enabled."""
        return self._host.get_camera_depth(self.camera_name)

    def is_ready(self, timeout: float = 0.5) -> bool:
        """Return True once the underlying camera topic has delivered a frame."""
        return bool(self._host.check_camera(self.camera_name, timeout))

    def reopen(self) -> None:
        """No-op because camera subscriptions share the RFrame session."""
        return None


class R2Connection(Connection):
    """Shared RFrame session managing R2 arms, end-effectors, chassis, and cameras."""

    def __init__(
        self,
        *,
        body_dof: Optional[int | str] = None,
        end_effector_type: Optional[str] = "gripper",
        enable_base: bool = False,
        cameras: Sequence[str] = ("head", "wrist_left", "wrist_right"),
        enable_depth: bool = False,
        use_compressed_camera: bool = False,
        enable_shm: bool = False,
        controller_mode: int = R2_MODE_SERVOJ,
        enable_fault_recovery: bool = True,
        enable_stall_detection: bool = True,
        stall_position_threshold: float = 0.05,
        stall_velocity_threshold: float = 0.002,
        stall_steps_threshold: int = 15,
        max_recovery_attempts: int = 3,
        recovery_cooldown_window: float = 10.0,
        node_name: str = "rlinf_r2_controller",
    ) -> None:
        if body_dof in (None, "auto"):
            self._auto_body_dof = True
            resolved_body_dof = 21
        elif body_dof in (20, 21, "20", "21"):
            self._auto_body_dof = False
            resolved_body_dof = int(body_dof)
        else:
            raise ValueError(
                f"R2 body_dof must be 20, 21, or 'auto'/None, got {body_dof!r}."
            )

        if end_effector_type in (None, "auto"):
            self._auto_end_effector_type = True
            resolved_ee_type = "gripper"
        elif end_effector_type in ("gripper", "hand", "none"):
            self._auto_end_effector_type = False
            resolved_ee_type = str(end_effector_type)
        else:
            raise ValueError(
                "R2 end_effector_type must be 'gripper', 'hand', 'none', or 'auto'/None, "
                f"got {end_effector_type!r}."
            )

        if controller_mode not in (
            R2_MODE_HOLD,
            R2_MODE_MOVEJ,
            R2_MODE_SERVOJ,
            R2_MODE_GRAVITY_COMP,
        ):
            raise ValueError(
                f"R2 controller_mode must be one of 0..3, got {controller_mode!r}."
            )
        unknown_cameras = [c for c in cameras if c not in R2_CAMERA_TOPICS]
        if unknown_cameras:
            raise ValueError(
                f"Unknown R2 camera names {unknown_cameras}; "
                f"valid names are {sorted(R2_CAMERA_TOPICS)}."
            )
        self.body_dof = resolved_body_dof
        self.end_effector_type = resolved_ee_type
        self.detected_body_dof: Optional[int] = None
        self.detected_end_effector_type: Optional[str] = None
        self.enable_base = enable_base
        self.camera_names = tuple(cameras)
        self.enable_depth = enable_depth
        self.use_compressed_camera = use_compressed_camera
        self.enable_shm = enable_shm
        self.controller_mode = controller_mode
        self.configure_fault_recovery(
            enable_fault_recovery=enable_fault_recovery,
            enable_stall_detection=enable_stall_detection,
            stall_position_threshold=stall_position_threshold,
            stall_velocity_threshold=stall_velocity_threshold,
            stall_steps_threshold=stall_steps_threshold,
            max_recovery_attempts=max_recovery_attempts,
            recovery_cooldown_window=recovery_cooldown_window,
        )
        self.node_name = node_name
        self._logger = get_logger()

        self.waist_joint_names = (
            R2_WAIST_JOINTS_21DOF if resolved_body_dof == 21 else R2_WAIST_JOINTS_20DOF
        )
        self.body_joint_names: tuple[str, ...] = (
            self.waist_joint_names
            + R2_NECK_JOINTS
            + R2_LEFT_ARM_JOINTS
            + R2_RIGHT_ARM_JOINTS
        )

        self._lock = threading.Lock()
        self._state = R2RobotState(
            waist_positions=np.zeros(len(self.waist_joint_names), dtype=np.float32),
            waist_velocities=np.zeros(len(self.waist_joint_names), dtype=np.float32),
        )
        self._joint_map_positions: dict[str, float] = {}
        self._joint_map_velocities: dict[str, float] = {}
        self._joint_map_efforts: dict[str, float] = {}
        self._state_ready = threading.Event()

        self._camera_frames: dict[str, np.ndarray] = {}
        self._camera_depths: dict[str, np.ndarray] = {}
        self._camera_events: dict[str, threading.Event] = {
            name: threading.Event() for name in self.camera_names
        }

        self._session: Any = None
        self._is_real_rframe: bool = False
        self._active_real_mode: Optional[int] = None
        self._rclpy_node: Any = None
        self._rclpy_clients: dict[str, Any] = {}
        self._readers: list[Any] = []
        self._writers: dict[str, Any] = {}
        self._services: dict[str, Any] = {}
        self._msg_types: dict[str, Any] = {}
        self._cmd_left_arm: Optional[np.ndarray] = None
        self._cmd_right_arm: Optional[np.ndarray] = None
        self._cmd_waist: Optional[np.ndarray] = None
        self._cmd_neck: Optional[np.ndarray] = None
        self._cmd_left_gripper: Optional[float] = None
        self._cmd_right_gripper: Optional[float] = None
        self._cmd_left_hand: Optional[np.ndarray] = None
        self._cmd_right_hand: Optional[np.ndarray] = None
        self._chassis_cmd_started: bool = False
        self._system_state_code: int = 2
        self._last_error_code: int = 0
        self._last_error_msg: str = ""
        self._disabled_slaves: tuple[int, ...] = ()
        self._stall_counter: int = 0
        self._prev_actual_joints: Optional[np.ndarray] = None
        self._recovery_timestamps: list[float] = []

    def _set_body_dof(self, body_dof: int) -> None:
        """Update the waist and body joint tables to match ``body_dof`` (20 or 21)."""
        self.body_dof = body_dof
        self.waist_joint_names = (
            R2_WAIST_JOINTS_21DOF if body_dof == 21 else R2_WAIST_JOINTS_20DOF
        )
        self.body_joint_names = (
            self.waist_joint_names
            + R2_NECK_JOINTS
            + R2_LEFT_ARM_JOINTS
            + R2_RIGHT_ARM_JOINTS
        )
        target_len = len(self.waist_joint_names)
        if self._state.waist_positions.size != target_len:
            self._state.waist_positions = np.zeros(target_len, dtype=np.float32)
            self._state.waist_velocities = np.zeros(target_len, dtype=np.float32)

    @classmethod
    def probe_hardware(
        cls,
        *,
        timeout: float = 1.0,
        enable_shm: bool = False,
        default_body_dof: int = 21,
        default_end_effector_type: str = "gripper",
        node_name: str = "rlinf_r2_probe",
    ) -> tuple[int, str]:
        """Probe ``JointState`` over RFrame to detect ``(body_dof, end_effector_type)``."""
        try:
            import rframe
            from sensor_msgs.msg import JointState
        except ImportError:
            return default_body_dof, default_end_effector_type

        detected_dof: Optional[int] = None
        detected_ee: Optional[str] = None
        ready = threading.Event()

        def _on_msg(msg: Any) -> None:
            nonlocal detected_dof, detected_ee
            names = list(getattr(msg, "name", []))
            positions = list(getattr(msg, "position", []))
            dof, ee = infer_r2_hardware(names, positions)
            if dof is not None:
                detected_dof = dof
            if ee is not None:
                detected_ee = ee
            ready.set()

        is_mock = hasattr(rframe, "Session")
        session = (
            rframe.Session(node_name) if is_mock else _get_shared_rframe_app(rframe)
        )
        channel: Any = None
        try:
            if is_mock:
                cfg = rframe.ChannelConfig()
                cfg.topic_name = "/Motion/JointState"
                cfg.channel_type = rframe.ChannelType.READER
                if hasattr(cfg, "enable_shm"):
                    cfg.enable_shm = enable_shm
                channel = session.create_hybrid_channel(cfg)
                channel.register_typed_callback(JointState, _on_msg)
                ready.wait(timeout=timeout)
            else:
                cfg = rframe.ChannelConfig(
                    topic="/robotic/joint_states",
                    handler_type=rframe.HandlerType.READER,
                    channel_type=rframe.ChannelType.HYBRID,
                    enable_shm=True,
                    enable_ros2=True,
                    type_name="sensor_msgs/msg/JointState",
                    ros2_qos={"reliability": "best_effort"},
                )
                channel = rframe.HybridChannel(cfg)
                channel.register_typed_callback(JointState, _on_msg)
                ready.wait(timeout=max(timeout, 2.5))
        finally:
            if channel is not None and hasattr(channel, "close"):
                channel.close()
            if is_mock and hasattr(session, "close"):
                session.close()

        return (
            detected_dof if detected_dof in (20, 21) else default_body_dof,
            detected_ee
            if detected_ee in ("gripper", "hand", "none")
            else default_end_effector_type,
        )

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Decompose the shared R2 connection into arms, end effectors, base, and cameras."""
        parts: dict[str, RobotPart] = {
            "left": R2Arm(self, "left"),
            "right": R2Arm(self, "right"),
        }
        if self.end_effector_type == "gripper":
            parts["left_end_effector"] = MethodEndEffector(
                self,
                "left_gripper_position",
                dims=1,
                command="move_left_gripper",
                is_gripper=True,
            )
            parts["right_end_effector"] = MethodEndEffector(
                self,
                "right_gripper_position",
                dims=1,
                command="move_right_gripper",
                is_gripper=True,
            )
        elif self.end_effector_type == "hand":
            parts["left_end_effector"] = R2HandView(self, "left")
            parts["right_end_effector"] = R2HandView(self, "right")
        if self.enable_base:
            parts["base"] = R2MobileBase(self)
        for name in self.camera_names:
            parts[name] = R2Camera(
                self,
                name,
                resolution=R2_CAMERA_TOPICS[name]["resolution"],
                enable_depth=self.enable_depth,
            )
        return parts

    def _create_channel(self, config: Any) -> Any:
        """Create a HybridChannel from ``self._session`` or ``rframe``."""
        rframe = self._msg_types["rframe"]
        if hasattr(self._session, "create_hybrid_channel"):
            return self._session.create_hybrid_channel(config)
        if self._is_real_rframe:
            return rframe.HybridChannel(config)
        return rframe.HybridChannel(self._session, config)

    def _subscribe(
        self,
        topic: str,
        msg_cls: Any,
        callback: Any,
        *,
        reliability: str = "reliable",
        memory_size: int = 65536,
    ) -> Any:
        """Register a typed reader channel on ``topic``."""
        rframe = self._msg_types["rframe"]
        if self._is_real_rframe:
            cfg = rframe.ChannelConfig(
                topic=topic,
                handler_type=rframe.HandlerType.READER,
                channel_type=rframe.ChannelType.HYBRID,
                memory_size=memory_size,
                enable_shm=True,
                enable_ros2=True,
                type_name=_ros2_type_name(msg_cls),
                ros2_qos={"reliability": reliability},
            )
        else:
            cfg = rframe.ChannelConfig()
            cfg.topic_name = topic
            cfg.channel_type = rframe.ChannelType.READER
            if hasattr(cfg, "enable_shm"):
                cfg.enable_shm = self.enable_shm
        channel = self._create_channel(cfg)
        channel.register_typed_callback(msg_cls, callback)
        self._readers.append(channel)
        return channel

    def _advertise(self, topic: str, msg_cls: Any) -> Any:
        """Create a writer channel on ``topic``."""
        rframe = self._msg_types["rframe"]
        if self._is_real_rframe:
            cfg = rframe.ChannelConfig(
                topic=topic,
                handler_type=rframe.HandlerType.WRITER,
                channel_type=rframe.ChannelType.HYBRID,
                enable_shm=True,
                enable_ros2=True,
                type_name=_ros2_type_name(msg_cls),
            )
        else:
            cfg = rframe.ChannelConfig()
            cfg.topic_name = topic
            cfg.channel_type = rframe.ChannelType.WRITER
            if hasattr(cfg, "enable_shm"):
                cfg.enable_shm = self.enable_shm
        channel = self._create_channel(cfg)
        self._writers[topic] = (channel, msg_cls)
        return channel

    def _client(self, service_name: str, srv_cls: Any) -> Any:
        """Create an RPC client channel on ``service_name``."""
        if self._is_real_rframe:
            if self._rclpy_node is not None:
                client = self._rclpy_node.create_client(srv_cls, service_name)
                self._rclpy_clients[service_name] = (client, srv_cls)
                self._services[service_name] = (client, srv_cls)
                return client
            self._services[service_name] = (None, srv_cls)
            return None
        rframe = self._msg_types["rframe"]
        cfg = rframe.ChannelConfig()
        cfg.topic_name = service_name
        cfg.channel_type = getattr(
            rframe.ChannelType, "RPC_CLIENT", getattr(rframe.ChannelType, "CLIENT", 2)
        )
        channel = self._create_channel(cfg)
        self._services[service_name] = (channel, srv_cls)
        return channel

    def _open(self) -> Any:
        """Initialize the RFrame session, topic subscriptions, writers, and services."""
        import rframe
        from geometry_msgs.msg import Pose, Twist, WrenchStamped
        from navigation_msgs.msg import NaviState
        from r2_motion_msgs.msg import TCP
        from r2_motion_msgs.srv import MoveJ
        from rdiagnostic_msgs.msg import HealthState
        from sensor_msgs.msg import (
            BatteryState,
            CompressedImage,
            Image,
            Imu,
            JointState,
        )
        from std_srvs.srv import Trigger

        self._msg_types = {
            "rframe": rframe,
            "Pose": Pose,
            "Twist": Twist,
            "WrenchStamped": WrenchStamped,
            "NaviState": NaviState,
            "TCP": TCP,
            "MoveJ": MoveJ,
            "HealthState": HealthState,
            "BatteryState": BatteryState,
            "CompressedImage": CompressedImage,
            "Image": Image,
            "Imu": Imu,
            "JointState": JointState,
            "Trigger": Trigger,
        }

        self._is_real_rframe = not hasattr(rframe, "Session")
        self._readers.clear()
        self._writers.clear()
        self._services.clear()
        self._rclpy_clients.clear()

        if self._is_real_rframe:
            from r2_motion_msgs.msg import ServoJ, ServoRYHandDual
            from r2_motion_msgs.srv import (
                GetSystemState,
                GetTcp,
                MoveGripper,
                MoveL,
                MoveRYHandDual,
            )
            from r2_motion_msgs.srv import SwitchMode as MotionSwitchMode
            from state_machine_msgs.srv import ChangeState

            self._msg_types.update(
                {
                    "ServoJ": ServoJ,
                    "ServoRYHandDual": ServoRYHandDual,
                    "GetSystemState": GetSystemState,
                    "GetTcp": GetTcp,
                    "MoveGripper": MoveGripper,
                    "MoveL": MoveL,
                    "MoveRYHandDual": MoveRYHandDual,
                    "MotionSwitchMode": MotionSwitchMode,
                    "ChangeState": ChangeState,
                }
            )
            self._session = _get_shared_rframe_app(rframe)

            self._subscribe(
                "/robotic/joint_states",
                JointState,
                self._on_joint_state,
                reliability="best_effort",
            )
            self._subscribe(
                "/robotic/tcp_data",
                TCP,
                self._on_tcp,
                reliability="reliable",
            )
            self._subscribe(
                "/robotic/ft_sensor_left",
                WrenchStamped,
                self._on_left_wrench,
                reliability="reliable",
            )
            self._subscribe(
                "/robotic/ft_sensor_right",
                WrenchStamped,
                self._on_right_wrench,
                reliability="reliable",
            )
            self._subscribe(
                "/robotic/Imu",
                Imu,
                self._on_imu,
                reliability="best_effort",
            )
            self._subscribe(
                "/battery_state",
                BatteryState,
                self._on_battery,
                reliability="best_effort",
            )
            self._subscribe(
                "/diagnostic",
                HealthState,
                self._on_health_state,
                reliability="best_effort",
            )
            try:
                from r2_motion_msgs.msg import ControllerState

                self._subscribe(
                    "/robby_motion_node/state",
                    ControllerState,
                    self._on_controller_state,
                    reliability="best_effort",
                )
            except ImportError:
                pass

            if self.enable_base:
                self._subscribe(
                    "/navi/state",
                    NaviState,
                    self._on_navi_state,
                    reliability="best_effort",
                )
                try:
                    from nav_msgs.msg import Odometry

                    self._subscribe(
                        "/chassis/est_odom",
                        Odometry,
                        self._on_navi_state,
                        reliability="best_effort",
                    )
                except ImportError:
                    pass

            for cam_name in self.camera_names:
                spec = R2_CAMERA_TOPICS[cam_name]
                resolution = spec["resolution"]
                if self.use_compressed_camera:
                    self._subscribe(
                        spec["real_compressed"],
                        CompressedImage,
                        self._make_camera_callback(
                            cam_name,
                            compressed=True,
                            depth=False,
                            resolution=resolution,
                        ),
                        reliability="best_effort",
                        memory_size=4 * 1024 * 1024,
                    )
                else:
                    self._subscribe(
                        spec["real_color"],
                        Image,
                        self._make_camera_callback(
                            cam_name,
                            compressed=False,
                            depth=False,
                            resolution=resolution,
                        ),
                        reliability="best_effort",
                        memory_size=8 * 1024 * 1024,
                    )
                if self.enable_depth:
                    self._subscribe(
                        spec["real_depth"],
                        Image,
                        self._make_camera_callback(
                            cam_name,
                            compressed=False,
                            depth=True,
                            resolution=resolution,
                        ),
                        reliability="best_effort",
                        memory_size=8 * 1024 * 1024,
                    )

            self._advertise("/robotic/servoj_data", ServoJ)
            self._advertise("/robotic/servo_ry_hand_data", ServoRYHandDual)
            if self.enable_base:
                self._advertise("/chassis/cmd_vel", Twist)

            try:
                import rclpy

                if not rclpy.ok():
                    rclpy.init()
                self._rclpy_node = rclpy.create_node(self.node_name)
                self._msg_types["rclpy"] = rclpy
            except Exception:
                self._rclpy_node = None

            self._client("/robotic/switch_mode_cmd", MotionSwitchMode)
            self._client("/robotic/movej_cmd", MoveJ)
            self._client("/robotic/movel_cmd", MoveL)
            self._client("/robotic/move_gripper_cmd", MoveGripper)
            self._client("/robotic/move_ryhand_cmd", MoveRYHandDual)
            self._client("/robotic/release_stop", Trigger)
            self._client("/robotic/get_system_state_cmd", GetSystemState)
            self._client("/robotic/get_tcp_cmd", GetTcp)
            self._client("/change_state", ChangeState)
            try:
                from r2_motion_msgs.srv import SetAxisEnable

                self._client("/robotic/set_axis_enable_cmd", SetAxisEnable)
            except ImportError:
                pass
            try:
                from rdiagnostic_msgs.srv import ClearFaults

                self._client("/clear_faults", ClearFaults)
            except ImportError:
                pass
            if self.enable_base:
                try:
                    from std_srvs.srv import SetBool

                    self._client("/chassis/cmd_start", SetBool)
                except ImportError:
                    pass

            self._state_ready.wait(timeout=5.0)
            for cam_name in self.camera_names:
                self._camera_events[cam_name].wait(timeout=2.0)
            st_resp = self._call_service(
                "/robotic/get_system_state_cmd", lambda _req: None, timeout=2.0
            )
            if (
                (st_resp is not None and getattr(st_resp, "state", 2) == 4)
                or self._last_error_code != 0
                or len(self._disabled_slaves) > 0
            ):
                self.clear_errors()
            else:
                self.switch_controller(self.controller_mode)
            return self._session

        from navigation_msgs.srv import NavigationControl
        from r2_motion_msgs.srv import (
            ForwardKine,
            InverseKine,
            MoveC,
            SetLoad,
            SwitchController,
        )
        from state_machine_msgs.srv import SwitchMode
        from task_msgs.srv import AssignTask

        self._msg_types.update(
            {
                "NavigationControl": NavigationControl,
                "ForwardKine": ForwardKine,
                "InverseKine": InverseKine,
                "MoveC": MoveC,
                "SetLoad": SetLoad,
                "SwitchController": SwitchController,
                "SwitchMode": SwitchMode,
                "AssignTask": AssignTask,
            }
        )

        if hasattr(rframe, "init"):
            rframe.init()
        self._session = rframe.Session(self.node_name)

        self._subscribe("/Motion/JointState", JointState, self._on_joint_state)
        self._subscribe("/Motion/TCP", TCP, self._on_tcp)
        self._subscribe("/drivers/ftsensor/fts_l", WrenchStamped, self._on_left_wrench)
        self._subscribe("/drivers/ftsensor/fts_r", WrenchStamped, self._on_right_wrench)
        self._subscribe("/drivers/imu/body_imu", Imu, self._on_imu)
        self._subscribe("/drivers/bms/battery_state", BatteryState, self._on_battery)
        self._subscribe(
            "/safety/health_monitor/state", HealthState, self._on_health_state
        )

        if self.enable_base:
            self._subscribe("/navigation/state", NaviState, self._on_navi_state)

        for cam_name in self.camera_names:
            spec = R2_CAMERA_TOPICS[cam_name]
            resolution = spec["resolution"]
            if self.use_compressed_camera:
                self._subscribe(
                    spec["compressed"],
                    CompressedImage,
                    self._make_camera_callback(
                        cam_name, compressed=True, depth=False, resolution=resolution
                    ),
                )
            else:
                self._subscribe(
                    spec["color"],
                    Image,
                    self._make_camera_callback(
                        cam_name, compressed=False, depth=False, resolution=resolution
                    ),
                )
            if self.enable_depth:
                self._subscribe(
                    spec["depth"],
                    Image,
                    self._make_camera_callback(
                        cam_name, compressed=False, depth=True, resolution=resolution
                    ),
                )

        self._advertise("/Motion/ServoJ", JointState)
        if self.enable_base:
            self._advertise("/teleop/chassis_Twist", Twist)

        self._client("/Motion/SwitchController", SwitchController)
        self._client("/Motion/MoveJ", MoveJ)
        self._client("/Motion/MoveC", MoveC)
        self._client("/Motion/ClearError", Trigger)
        self._client("/Motion/SetLoad", SetLoad)
        self._client("/Motion/ForwardKine", ForwardKine)
        self._client("/Motion/InverseKine", InverseKine)
        self._client("/state_machine/switch_mode", SwitchMode)
        if self.enable_base:
            self._client("/task_manager/assign_task", AssignTask)
            self._client("/navigation/control", NavigationControl)

        self._state_ready.wait(timeout=1.0)
        self.switch_controller(self.controller_mode)
        return self._session

    def _release(self, device: Any) -> None:
        """Close all channels and shut down the RFrame session."""
        if self.enable_base and (
            "/teleop/chassis_Twist" in self._writers
            or "/chassis/cmd_vel" in self._writers
        ):
            try:
                self.stop_base()
            except Exception:
                pass
            if self._chassis_cmd_started and "/chassis/cmd_start" in self._services:
                try:
                    self._call_service(
                        "/chassis/cmd_start",
                        lambda req: setattr(req, "data", False),
                        timeout=1.0,
                    )
                except Exception:
                    pass
                self._chassis_cmd_started = False
        for channel in self._readers:
            if hasattr(channel, "close"):
                channel.close()
        for channel, _ in self._writers.values():
            if hasattr(channel, "close"):
                channel.close()
        for channel, _ in self._services.values():
            if hasattr(channel, "close"):
                channel.close()
        if self._rclpy_node is not None:
            try:
                self._rclpy_node.destroy_node()
            except Exception:
                pass
            self._rclpy_node = None
        self._rclpy_clients.clear()
        self._readers.clear()
        self._writers.clear()
        self._services.clear()
        self._active_real_mode = None
        self._cmd_left_arm = None
        self._cmd_right_arm = None
        self._cmd_waist = None
        self._cmd_neck = None
        self._cmd_left_gripper = None
        self._cmd_right_gripper = None
        self._cmd_left_hand = None
        self._cmd_right_hand = None
        self._system_state_code = 2
        self._last_error_code = 0
        self._last_error_msg = ""
        self._disabled_slaves = ()
        self._stall_counter = 0
        self._prev_actual_joints = None
        self._recovery_timestamps.clear()
        self._last_recovery_event = None
        if not self._is_real_rframe and hasattr(device, "close"):
            device.close()
        self._session = None

    def _make_camera_callback(
        self,
        camera_name: str,
        *,
        compressed: bool,
        depth: bool,
        resolution: tuple[int, int],
    ) -> Any:
        def _callback(msg: Any) -> None:
            decoded = _decode_image_message(
                msg,
                compressed=compressed,
                depth=depth,
                fallback_resolution=resolution,
            )
            with self._lock:
                if depth:
                    self._camera_depths[camera_name] = decoded
                else:
                    self._camera_frames[camera_name] = decoded
                    self._camera_events[camera_name].set()

        return _callback

    @staticmethod
    def _lookup_joint_values(
        source: dict[str, float],
        primary_names: Sequence[str],
        fallback_names: Sequence[str] = (),
    ) -> np.ndarray:
        out = np.zeros(len(primary_names), dtype=np.float32)
        for idx, name in enumerate(primary_names):
            if name in source:
                out[idx] = source[name]
            elif idx < len(fallback_names) and fallback_names[idx] in source:
                out[idx] = source[fallback_names[idx]]
        return out

    def _on_joint_state(self, msg: Any) -> None:
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        efforts = list(getattr(msg, "effort", []))

        detected_dof, detected_ee = infer_r2_hardware(names, positions)

        with self._lock:
            if detected_dof is not None:
                self.detected_body_dof = detected_dof
                if self._auto_body_dof and detected_dof != self.body_dof:
                    self._set_body_dof(detected_dof)
            if detected_ee is not None:
                self.detected_end_effector_type = detected_ee
                if self._auto_end_effector_type:
                    self.end_effector_type = detected_ee

            for idx, name in enumerate(names):
                if idx < len(positions):
                    self._joint_map_positions[name] = float(positions[idx])
                if idx < len(velocities):
                    self._joint_map_velocities[name] = float(velocities[idx])
                if idx < len(efforts):
                    self._joint_map_efforts[name] = float(efforts[idx])

            if names:
                self._state.left_joint_positions = self._lookup_joint_values(
                    self._joint_map_positions,
                    R2_LEFT_ARM_JOINTS,
                    R2_REAL_LEFT_ARM_JOINTS,
                )
                self._state.left_joint_velocities = self._lookup_joint_values(
                    self._joint_map_velocities,
                    R2_LEFT_ARM_JOINTS,
                    R2_REAL_LEFT_ARM_JOINTS,
                )
                self._state.left_joint_torques = self._lookup_joint_values(
                    self._joint_map_efforts,
                    R2_LEFT_ARM_JOINTS,
                    R2_REAL_LEFT_ARM_JOINTS,
                )
                self._state.right_joint_positions = self._lookup_joint_values(
                    self._joint_map_positions,
                    R2_RIGHT_ARM_JOINTS,
                    R2_REAL_RIGHT_ARM_JOINTS,
                )
                self._state.right_joint_velocities = self._lookup_joint_values(
                    self._joint_map_velocities,
                    R2_RIGHT_ARM_JOINTS,
                    R2_REAL_RIGHT_ARM_JOINTS,
                )
                self._state.right_joint_torques = self._lookup_joint_values(
                    self._joint_map_efforts,
                    R2_RIGHT_ARM_JOINTS,
                    R2_REAL_RIGHT_ARM_JOINTS,
                )
                self._state.waist_positions = self._lookup_joint_values(
                    self._joint_map_positions,
                    self.waist_joint_names,
                    R2_REAL_WAIST_JOINTS,
                )
                self._state.waist_velocities = self._lookup_joint_values(
                    self._joint_map_velocities,
                    self.waist_joint_names,
                    R2_REAL_WAIST_JOINTS,
                )
                self._state.neck_positions = np.array(
                    [self._joint_map_positions.get(j, 0.0) for j in R2_NECK_JOINTS],
                    dtype=np.float32,
                )
                self._state.neck_velocities = np.array(
                    [self._joint_map_velocities.get(j, 0.0) for j in R2_NECK_JOINTS],
                    dtype=np.float32,
                )
                if "GripperL" in self._joint_map_positions:
                    raw_l = float(self._joint_map_positions["GripperL"])
                    val_l = 1.0 - raw_l if self._is_real_rframe else raw_l
                    self._state.left_gripper_position = np.array(
                        [np.clip(val_l, 0.0, 1.0)], dtype=np.float32
                    )
                if "GripperR" in self._joint_map_positions:
                    raw_r = float(self._joint_map_positions["GripperR"])
                    val_r = 1.0 - raw_r if self._is_real_rframe else raw_r
                    self._state.right_gripper_position = np.array(
                        [np.clip(val_r, 0.0, 1.0)], dtype=np.float32
                    )
                if any(j in self._joint_map_positions for j in R2_HAND_JOINTS_LEFT):
                    self._state.left_hand_position = np.array(
                        [
                            self._joint_map_positions.get(j, 0.0)
                            for j in R2_HAND_JOINTS_LEFT
                        ],
                        dtype=np.float32,
                    )
                if any(j in self._joint_map_positions for j in R2_HAND_JOINTS_RIGHT):
                    self._state.right_hand_position = np.array(
                        [
                            self._joint_map_positions.get(j, 0.0)
                            for j in R2_HAND_JOINTS_RIGHT
                        ],
                        dtype=np.float32,
                    )
            elif len(positions) in (20, 21):
                arr = np.asarray(positions, dtype=np.float32)
                w = len(positions) - 16
                self._state.waist_positions = arr[:w].copy()
                self._state.neck_positions = arr[w : w + 2].copy()
                self._state.left_joint_positions = arr[w + 2 : w + 9].copy()
                self._state.right_joint_positions = arr[w + 9 : w + 16].copy()
            elif len(positions) >= 14:
                arr = np.asarray(positions, dtype=np.float32)
                self._state.left_joint_positions = arr[:7].copy()
                self._state.right_joint_positions = arr[7:14].copy()

            self._state_ready.set()

    def _on_tcp(self, msg: Any) -> None:
        poses = list(getattr(msg, "tcp_pose", []))
        twists = list(getattr(msg, "tcp_twist", []))
        with self._lock:
            if hasattr(msg, "pose_l"):
                self._state.left_tcp_pose = _pose_to_xyz_rpy(msg.pose_l)
            elif len(poses) >= 1:
                self._state.left_tcp_pose = _pose_to_xyz_rpy(poses[0])
            if hasattr(msg, "pose_r"):
                self._state.right_tcp_pose = _pose_to_xyz_rpy(msg.pose_r)
            elif len(poses) >= 2:
                self._state.right_tcp_pose = _pose_to_xyz_rpy(poses[1])
            if len(twists) >= 1:
                t0 = twists[0]
                self._state.left_tcp_vel = np.array(
                    [
                        t0.linear.x,
                        t0.linear.y,
                        t0.linear.z,
                        t0.angular.x,
                        t0.angular.y,
                        t0.angular.z,
                    ],
                    dtype=np.float32,
                )
            if len(twists) >= 2:
                t1 = twists[1]
                self._state.right_tcp_vel = np.array(
                    [
                        t1.linear.x,
                        t1.linear.y,
                        t1.linear.z,
                        t1.angular.x,
                        t1.angular.y,
                        t1.angular.z,
                    ],
                    dtype=np.float32,
                )

    def _on_left_wrench(self, msg: Any) -> None:
        w = getattr(msg, "wrench", msg)
        with self._lock:
            self._state.left_wrench = np.array(
                [
                    w.force.x,
                    w.force.y,
                    w.force.z,
                    w.torque.x,
                    w.torque.y,
                    w.torque.z,
                ],
                dtype=np.float32,
            )

    def _on_right_wrench(self, msg: Any) -> None:
        w = getattr(msg, "wrench", msg)
        with self._lock:
            self._state.right_wrench = np.array(
                [
                    w.force.x,
                    w.force.y,
                    w.force.z,
                    w.torque.x,
                    w.torque.y,
                    w.torque.z,
                ],
                dtype=np.float32,
            )

    def _on_imu(self, msg: Any) -> None:
        with self._lock:
            ori = msg.orientation
            ang = msg.angular_velocity
            acc = msg.linear_acceleration
            self._state.imu_orientation = np.array(
                [ori.x, ori.y, ori.z, ori.w], dtype=np.float32
            )
            self._state.imu_angular_velocity = np.array(
                [ang.x, ang.y, ang.z], dtype=np.float32
            )
            self._state.imu_linear_acceleration = np.array(
                [acc.x, acc.y, acc.z], dtype=np.float32
            )

    def _on_battery(self, msg: Any) -> None:
        with self._lock:
            self._state.battery_percentage = float(getattr(msg, "percentage", 100.0))

    def _on_health_state(self, msg: Any) -> None:
        with self._lock:
            if hasattr(msg, "is_healthy"):
                self._state.is_healthy = bool(msg.is_healthy)
            elif hasattr(msg, "fault_level"):
                self._state.is_healthy = int(msg.fault_level) == 0
            elif hasattr(msg, "severity_level"):
                self._state.is_healthy = int(msg.severity_level) == 0

    def _on_controller_state(self, msg: Any) -> None:
        with self._lock:
            if hasattr(msg, "run_mode"):
                self._active_real_mode = int(msg.run_mode)
            if hasattr(msg, "state"):
                self._system_state_code = int(msg.state)
            err_code = int(getattr(msg, "error_code", 0))
            if err_code != 0:
                self._last_error_code = err_code
                self._last_error_msg = str(getattr(msg, "error_msg", ""))
            slaves = list(getattr(msg, "slave_addrs", []))
            enabled = list(getattr(msg, "enabled", []))
            if slaves and len(slaves) == len(enabled):
                disabled = [
                    int(addr)
                    for addr, is_en in zip(slaves, enabled, strict=False)
                    if int(addr) in R2_SLAVE_ADDR_TO_JOINT and not bool(is_en)
                ]
                self._disabled_slaves = tuple(disabled)

    def _on_navi_state(self, msg: Any) -> None:
        with self._lock:
            if hasattr(msg, "current_pose"):
                cp = msg.current_pose
                self._state.base_pose = np.array(
                    [
                        float(getattr(cp, "x", 0.0)),
                        float(getattr(cp, "y", 0.0)),
                        float(getattr(cp, "theta", 0.0)),
                    ],
                    dtype=np.float32,
                )
            elif hasattr(msg, "pose"):
                inner_pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
                pose_vec = _pose_to_xyz_rpy(inner_pose)
                self._state.base_pose = np.array(
                    [pose_vec[0], pose_vec[1], pose_vec[5]], dtype=np.float32
                )
            elif hasattr(msg, "x"):
                self._state.base_pose = np.array(
                    [
                        float(getattr(msg, "x", 0.0)),
                        float(getattr(msg, "y", 0.0)),
                        float(getattr(msg, "yaw", 0.0)),
                    ],
                    dtype=np.float32,
                )
            if hasattr(msg, "velocity"):
                vel = msg.velocity
                self._state.base_velocity = np.array(
                    [vel.linear.x, vel.linear.y, vel.angular.z], dtype=np.float32
                )
            elif hasattr(msg, "twist"):
                tw = msg.twist.twist if hasattr(msg.twist, "twist") else msg.twist
                self._state.base_velocity = np.array(
                    [tw.linear.x, tw.linear.y, tw.angular.z], dtype=np.float32
                )

    def _require_connected(self) -> Any:
        """Ensure the RFrame session is active before reading or commanding."""
        if self._session is None:
            raise RuntimeError("R2Connection is not connected. Call connect() first.")
        return self._session

    def get_state(self) -> R2RobotState:
        """Return a thread-safe copy of the latest :class:`R2RobotState`."""
        self._require_connected()
        with self._lock:
            return R2RobotState(
                left_tcp_pose=self._state.left_tcp_pose.copy(),
                left_tcp_vel=self._state.left_tcp_vel.copy(),
                left_joint_positions=self._state.left_joint_positions.copy(),
                left_joint_velocities=self._state.left_joint_velocities.copy(),
                left_joint_torques=self._state.left_joint_torques.copy(),
                left_wrench=self._state.left_wrench.copy(),
                left_gripper_position=self._state.left_gripper_position.copy(),
                left_hand_position=self._state.left_hand_position.copy(),
                right_tcp_pose=self._state.right_tcp_pose.copy(),
                right_tcp_vel=self._state.right_tcp_vel.copy(),
                right_joint_positions=self._state.right_joint_positions.copy(),
                right_joint_velocities=self._state.right_joint_velocities.copy(),
                right_joint_torques=self._state.right_joint_torques.copy(),
                right_wrench=self._state.right_wrench.copy(),
                right_gripper_position=self._state.right_gripper_position.copy(),
                right_hand_position=self._state.right_hand_position.copy(),
                waist_positions=self._state.waist_positions.copy(),
                waist_velocities=self._state.waist_velocities.copy(),
                neck_positions=self._state.neck_positions.copy(),
                neck_velocities=self._state.neck_velocities.copy(),
                base_pose=self._state.base_pose.copy(),
                base_velocity=self._state.base_velocity.copy(),
                imu_orientation=self._state.imu_orientation.copy(),
                imu_angular_velocity=self._state.imu_angular_velocity.copy(),
                imu_linear_acceleration=self._state.imu_linear_acceleration.copy(),
                battery_percentage=self._state.battery_percentage,
                is_healthy=self._state.is_healthy,
            )

    def read_state(self) -> dict[str, Any]:
        """Return the latest state snapshot as a plain dictionary for part views."""
        return self.get_state().to_dict()

    def _ensure_cmd_state(self, state: R2RobotState) -> None:
        """Lazily initialize commanded target state from the first measured state snapshot."""
        if self._cmd_left_arm is None:
            self._cmd_left_arm = state.left_joint_positions.copy()
        if self._cmd_right_arm is None:
            self._cmd_right_arm = state.right_joint_positions.copy()
        if (
            self._cmd_waist is None
            or self._cmd_waist.size != state.waist_positions.size
        ):
            self._cmd_waist = state.waist_positions.copy()
        if self._cmd_neck is None:
            self._cmd_neck = state.neck_positions.copy()
        if self._cmd_left_gripper is None:
            self._cmd_left_gripper = float(state.left_gripper_position[0])
        if self._cmd_right_gripper is None:
            self._cmd_right_gripper = float(state.right_gripper_position[0])
        if self._cmd_left_hand is None:
            self._cmd_left_hand = state.left_hand_position.copy()
        if self._cmd_right_hand is None:
            self._cmd_right_hand = state.right_hand_position.copy()

    def _sync_cmd_to_actual(self, state: Optional[R2RobotState] = None) -> None:
        """Synchronize internal commanded joint targets to the latest measured robot state."""
        snap = state if state is not None else self.get_state()
        self._cmd_left_arm = snap.left_joint_positions.copy()
        self._cmd_right_arm = snap.right_joint_positions.copy()
        self._cmd_waist = snap.waist_positions.copy()
        self._cmd_neck = snap.neck_positions.copy()
        self._cmd_left_gripper = float(snap.left_gripper_position[0])
        self._cmd_right_gripper = float(snap.right_gripper_position[0])
        self._cmd_left_hand = snap.left_hand_position.copy()
        self._cmd_right_hand = snap.right_hand_position.copy()
        self._stall_counter = 0
        self._prev_actual_joints = None

    def _apply_cmd_targets(self, name_to_pos: dict[str, float]) -> None:
        """Update cached whole-body command targets from a ``joint_name -> position`` mapping."""
        assert (
            self._cmd_left_arm is not None
            and self._cmd_right_arm is not None
            and self._cmd_waist is not None
            and self._cmd_neck is not None
            and self._cmd_left_gripper is not None
            and self._cmd_right_gripper is not None
        )
        for idx, (jn, rjn) in enumerate(
            zip(R2_LEFT_ARM_JOINTS, R2_REAL_LEFT_ARM_JOINTS, strict=True)
        ):
            if jn in name_to_pos:
                self._cmd_left_arm[idx] = name_to_pos[jn]
            elif rjn in name_to_pos:
                self._cmd_left_arm[idx] = name_to_pos[rjn]
        for idx, (jn, rjn) in enumerate(
            zip(R2_RIGHT_ARM_JOINTS, R2_REAL_RIGHT_ARM_JOINTS, strict=True)
        ):
            if jn in name_to_pos:
                self._cmd_right_arm[idx] = name_to_pos[jn]
            elif rjn in name_to_pos:
                self._cmd_right_arm[idx] = name_to_pos[rjn]
        for idx, jn in enumerate(self.waist_joint_names):
            if jn in name_to_pos and idx < self._cmd_waist.size:
                self._cmd_waist[idx] = name_to_pos[jn]
        for idx, rjn in enumerate(R2_REAL_WAIST_JOINTS):
            if rjn in name_to_pos and idx < self._cmd_waist.size:
                self._cmd_waist[idx] = name_to_pos[rjn]
        for idx, jn in enumerate(R2_NECK_JOINTS):
            if jn in name_to_pos:
                self._cmd_neck[idx] = name_to_pos[jn]
        if "GripperL" in name_to_pos:
            self._cmd_left_gripper = float(np.clip(name_to_pos["GripperL"], 0.0, 1.0))
        if "GripperR" in name_to_pos:
            self._cmd_right_gripper = float(np.clip(name_to_pos["GripperR"], 0.0, 1.0))

    def _sync_mock_diagnostics(self) -> None:
        """Pull simulated fault and mode fields from a mock RFrame session when present."""
        if self._is_real_rframe or self._session is None:
            return
        mock_st = getattr(self._session, "state", None)
        if not isinstance(mock_st, dict):
            return
        with self._lock:
            fc = int(mock_st.get("fault_code", 0))
            if fc != 0:
                self._last_error_code = fc
                self._last_error_msg = str(mock_st.get("error_msg", ""))
            slaves = mock_st.get("disabled_slaves")
            if slaves:
                self._disabled_slaves = tuple(int(x) for x in slaves)
            sys_st = int(mock_st.get("system_state", 2))
            if sys_st != 2:
                self._system_state_code = sys_st
            if "mode" in mock_st:
                self._active_real_mode = int(mock_st["mode"])

    def _has_active_fault_or_mode_drop(self, *, check_mode: bool = True) -> bool:
        """Fast non-RPC check for latched faults, disabled axes, or mode drop."""
        self._sync_mock_diagnostics()
        with self._lock:
            if self._last_error_code != 0:
                return True
            if len(self._disabled_slaves) > 0:
                return True
            if self._system_state_code == 4:
                return True
            if not self._state.is_healthy:
                return True
            if (
                check_mode
                and self.controller_mode == R2_MODE_SERVOJ
                and self._active_real_mode is not None
                and self._active_real_mode != R2_MODE_SERVOJ
            ):
                return True
        return False

    def diagnose(self) -> HealthStatus:
        """Return a structured hardware diagnostic snapshot for the R2 connection."""
        if self._owner is not None:
            return self._owner.diagnose()
        if self._session is None:
            return HealthStatus(
                healthy=False,
                level=FaultLevel.UNRECOVERABLE_FAULT,
                error_code=-1,
                error_message="R2Connection is not connected.",
            )
        self._sync_mock_diagnostics()
        if "/robotic/get_system_state_cmd" in self._services:
            resp = self._call_service(
                "/robotic/get_system_state_cmd", lambda _req: None, timeout=0.5
            )
            if resp is not None and hasattr(resp, "state"):
                with self._lock:
                    self._system_state_code = int(resp.state)

        state = self.get_state()
        with self._lock:
            err_code = self._last_error_code
            err_msg = self._last_error_msg
            disabled_slaves = self._disabled_slaves
            sys_state = self._system_state_code
            active_mode = (
                self._active_real_mode
                if self._active_real_mode is not None
                else self.controller_mode
            )
            stall_steps = self._stall_counter

        fault_parts = tuple(
            R2_SLAVE_ADDR_TO_JOINT.get(addr, str(addr)) for addr in disabled_slaves
        )
        if err_code in (401, 402):
            level = FaultLevel.UNRECOVERABLE_FAULT
        elif (
            err_code != 0
            or len(disabled_slaves) > 0
            or sys_state == 4
            or not state.is_healthy
        ):
            level = FaultLevel.RECOVERABLE_FAULT
        elif (
            self.controller_mode == R2_MODE_SERVOJ and active_mode != R2_MODE_SERVOJ
        ) or (
            self.enable_stall_detection and stall_steps >= self.stall_steps_threshold
        ):
            level = FaultLevel.MODE_DROP
        else:
            level = FaultLevel.OK

        reasons: list[str] = []
        if err_code != 0:
            desc = R2_ERROR_CODES.get(err_code, "Motion controller error")
            detail_suffix = f" ({err_msg})" if err_msg and err_msg != desc else ""
            reasons.append(f"error_code={err_code}: {desc}{detail_suffix}")
        if fault_parts:
            reasons.append(f"disabled_axes={list(fault_parts)}")
        if sys_state == 4:
            reasons.append("system_state=4 (Fault)")
        if not state.is_healthy:
            reasons.append("health_monitor=unhealthy")
        if self.controller_mode == R2_MODE_SERVOJ and active_mode != R2_MODE_SERVOJ:
            reasons.append(
                f"controller_mode_drop (expected={R2_MODE_SERVOJ}, active={active_mode})"
            )
        if self.enable_stall_detection and stall_steps >= self.stall_steps_threshold:
            reasons.append(f"joint_tracking_stall (steps={stall_steps})")

        return HealthStatus(
            healthy=(level == FaultLevel.OK),
            level=level,
            error_code=err_code,
            error_message="; ".join(reasons),
            fault_parts=fault_parts,
            details={
                "system_state": sys_state,
                "controller_mode": active_mode,
                "disabled_slaves": list(disabled_slaves),
                "stall_counter": stall_steps,
                "battery_percentage": state.battery_percentage,
            },
        )

    def consume_recovery_event(self) -> Optional[dict[str, Any]]:
        """Pop and return the most recent automatic recovery event, if any."""
        if self._owner is not None:
            return self._owner.consume_recovery_event()
        with self._lock:
            event = self._last_recovery_event
            self._last_recovery_event = None
            return event

    def _recover_mode(self) -> str:
        """Synchronize command state and re-assert ServoJ mode after a mode drop or stall."""
        self._sync_cmd_to_actual()
        self.switch_controller(R2_MODE_SERVOJ)
        return "switch_controller(SERVOJ)"

    def _check_and_recover(self, trigger_reason: str) -> HealthStatus:
        """Diagnose and execute tiered recovery via :meth:`Connection.recover`."""
        return self.recover(trigger=trigger_reason)

    def _update_stall_watchdog(self) -> None:
        """Update the ServoJ joint-tracking stall counter via :meth:`Connection.update_stall_watchdog`."""
        if not self.enable_fault_recovery or not self.enable_stall_detection:
            return
        if (
            self._cmd_left_arm is None
            or self._cmd_right_arm is None
            or self._cmd_waist is None
        ):
            return
        post_state = self.get_state()
        cmd_vec = np.concatenate(
            [
                self._cmd_left_arm,
                self._cmd_right_arm,
                self._cmd_waist,
            ]
        )
        act_vec = np.concatenate(
            [
                post_state.left_joint_positions,
                post_state.right_joint_positions,
                post_state.waist_positions,
            ]
        )
        vel_vec = np.concatenate(
            [
                post_state.left_joint_velocities,
                post_state.right_joint_velocities,
                post_state.waist_velocities,
            ]
        )
        self.update_stall_watchdog(cmd_vec, act_vec, vel_vec, trigger="servoj_stall")

    def _build_whole_body_joint_vector(
        self,
        *,
        left_arm: Optional[Sequence[float]] = None,
        right_arm: Optional[Sequence[float]] = None,
        waist: Optional[Sequence[float]] = None,
        neck: Optional[Sequence[float]] = None,
    ) -> list[float]:
        """Pack whole-body joints in R2 order: ``waist (5) + left_arm (7) + right_arm (7) + neck (1 or 2)``."""
        state = self.get_state()
        self._ensure_cmd_state(state)
        w_vec = np.zeros(5, dtype=np.float32)
        src_w = (
            np.asarray(waist, dtype=np.float32).reshape(-1)
            if waist is not None
            else (
                self._cmd_waist
                if self._cmd_waist is not None
                else state.waist_positions
            )
        )
        w_vec[: min(5, src_w.size)] = src_w[: min(5, src_w.size)]
        l_vec = (
            np.asarray(left_arm, dtype=np.float32).reshape(7)
            if left_arm is not None
            else (
                self._cmd_left_arm
                if self._cmd_left_arm is not None
                else state.left_joint_positions
            )
        )
        r_vec = (
            np.asarray(right_arm, dtype=np.float32).reshape(7)
            if right_arm is not None
            else (
                self._cmd_right_arm
                if self._cmd_right_arm is not None
                else state.right_joint_positions
            )
        )
        n_vec = (
            np.asarray(neck, dtype=np.float32).reshape(2)
            if neck is not None
            else (
                self._cmd_neck if self._cmd_neck is not None else state.neck_positions
            )
        )
        neck_slice = n_vec if self.body_dof == 21 else n_vec[1:2]
        return [float(x) for x in np.concatenate([w_vec, l_vec, r_vec, neck_slice])]

    def _publish_servoj(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        velocities: Optional[Sequence[float]] = None,
        efforts: Optional[Sequence[float]] = None,
    ) -> None:
        self._require_connected()
        if self.enable_fault_recovery and self._has_active_fault_or_mode_drop():
            self._check_and_recover("servoj_fault")

        name_to_pos = {
            str(n): float(p) for n, p in zip(joint_names, positions, strict=False)
        }
        state = self.get_state()
        self._ensure_cmd_state(state)
        is_hand_cmd = any(n.startswith(("HandL_", "HandR_")) for n in name_to_pos)
        is_body_cmd = any(
            not n.startswith(("HandL_", "HandR_", "Gripper")) for n in name_to_pos
        )
        if not is_hand_cmd:
            self._apply_cmd_targets(name_to_pos)

        if "/Motion/ServoJ" in self._writers:
            channel, msg_cls = self._writers["/Motion/ServoJ"]
            msg = msg_cls()
            msg.name = list(joint_names)
            msg.position = [float(x) for x in positions]
            msg.velocity = (
                [float(x) for x in velocities] if velocities is not None else []
            )
            msg.effort = [float(x) for x in efforts] if efforts is not None else []
            channel.send(msg)
            if is_body_cmd:
                self._update_stall_watchdog()
            return

        if "/robotic/servoj_data" not in self._writers:
            return

        if is_hand_cmd and "/robotic/servo_ry_hand_data" in self._writers:
            h_ch, h_cls = self._writers["/robotic/servo_ry_hand_data"]
            h_msg = h_cls()
            assert self._cmd_left_hand is not None and self._cmd_right_hand is not None
            for idx, jn in enumerate(R2_HAND_JOINTS_LEFT):
                if jn in name_to_pos:
                    self._cmd_left_hand[idx] = name_to_pos[jn]
            for idx, jn in enumerate(R2_HAND_JOINTS_RIGHT):
                if jn in name_to_pos:
                    self._cmd_right_hand[idx] = name_to_pos[jn]
            h_msg.left_finger_open_ratio = [float(x) for x in self._cmd_left_hand]
            h_msg.right_finger_open_ratio = [float(x) for x in self._cmd_right_hand]
            h_ch.send(h_msg)
            return

        assert (
            self._cmd_left_arm is not None
            and self._cmd_right_arm is not None
            and self._cmd_waist is not None
            and self._cmd_neck is not None
            and self._cmd_left_gripper is not None
            and self._cmd_right_gripper is not None
        )

        if self._is_real_rframe and self._active_real_mode != R2_MODE_SERVOJ:
            self.switch_controller(R2_MODE_SERVOJ)

        channel, msg_cls = self._writers["/robotic/servoj_data"]
        msg = msg_cls()
        msg.joint = self._build_whole_body_joint_vector(
            left_arm=self._cmd_left_arm,
            right_arm=self._cmd_right_arm,
            waist=self._cmd_waist,
            neck=self._cmd_neck,
        )
        msg.gripper_l = float(np.clip(1.0 - self._cmd_left_gripper, 0.0, 1.0))
        msg.gripper_r = float(np.clip(1.0 - self._cmd_right_gripper, 0.0, 1.0))
        channel.send(msg)
        if is_body_cmd:
            self._update_stall_watchdog()

    def _call_service(
        self, service_name: str, configure_request: Any, timeout: float = 5.0
    ) -> Any:
        self._require_connected()
        if service_name not in self._services:
            raise RuntimeError(f"R2 service {service_name!r} is not initialized.")
        channel, srv_cls = self._services[service_name]
        req_cls = getattr(srv_cls, "Request", srv_cls)
        req = req_cls()
        configure_request(req)
        resp = None
        if hasattr(channel, "call_async") and self._rclpy_node is not None:
            rclpy = self._msg_types.get("rclpy")
            if hasattr(channel, "wait_for_service"):
                if not channel.wait_for_service(timeout_sec=min(timeout, 2.0)):
                    return None
            future = channel.call_async(req)
            if rclpy is not None:
                rclpy.spin_until_future_complete(
                    self._rclpy_node, future, timeout_sec=timeout
                )
            resp = future.result() if future.done() else None
        elif hasattr(channel, "call"):
            resp = channel.call(req, timeout=timeout)

        err_obj = getattr(resp, "error_code", None) if resp is not None else None
        if err_obj is not None:
            code = int(
                getattr(err_obj, "code", err_obj if isinstance(err_obj, int) else 0)
            )
            if code != 0:
                with self._lock:
                    self._last_error_code = code
                    self._last_error_msg = str(
                        getattr(err_obj, "msg", getattr(resp, "error_msg", ""))
                    )
        return resp

    def switch_controller(self, mode: int) -> bool:
        """Switch the R2 motion controller mode (0: Hold, 1: MoveJ, 2: ServoJ, 3: GravityComp)."""
        if mode not in (
            R2_MODE_HOLD,
            R2_MODE_MOVEJ,
            R2_MODE_SERVOJ,
            R2_MODE_GRAVITY_COMP,
        ):
            raise ValueError(f"Unsupported R2 controller mode {mode!r}.")

        def _fill(req: Any) -> None:
            req.mode = int(mode)

        srv_name = (
            "/Motion/SwitchController"
            if "/Motion/SwitchController" in self._services
            else "/robotic/switch_mode_cmd"
        )
        resp = self._call_service(srv_name, _fill)
        self.controller_mode = int(mode)
        if resp is not None:
            self._active_real_mode = int(mode)
        return bool(getattr(resp, "success", resp is not None))

    def switch_system_mode(self, mode: int) -> bool:
        """Switch the R2 system state machine mode via ``/state_machine/switch_mode``."""

        def _fill(req: Any) -> None:
            if hasattr(req, "mode"):
                req.mode = int(mode)
            elif hasattr(req, "transition") and hasattr(req.transition, "id"):
                req.transition.id = int(mode)

        srv_name = (
            "/state_machine/switch_mode"
            if "/state_machine/switch_mode" in self._services
            else "/change_state"
        )
        resp = self._call_service(srv_name, _fill)
        return bool(getattr(resp, "success", True))

    def clear_errors(self) -> bool:
        """Clear motion controller faults, re-enable body axes, and restore controller mode."""
        self._sync_cmd_to_actual()
        if self._is_real_rframe:
            if "/clear_faults" in self._services:
                self._call_service("/clear_faults", lambda _req: None, timeout=1.5)
            if "/robotic/set_axis_enable_cmd" in self._services:
                body_slaves = list(R2_BODY_SLAVE_ADDRS)

                def _fill_enable(req: Any) -> None:
                    req.slave_addrs = body_slaves
                    req.enable = True

                self._call_service(
                    "/robotic/set_axis_enable_cmd", _fill_enable, timeout=10.0
                )
        srv_name = (
            "/Motion/ClearError"
            if "/Motion/ClearError" in self._services
            else "/robotic/release_stop"
        )
        resp = self._call_service(srv_name, lambda _req: None, timeout=1.5)
        with self._lock:
            self._last_error_code = 0
            self._last_error_msg = ""
            self._disabled_slaves = ()
            self._system_state_code = 2
            self._state.is_healthy = True
            self._stall_counter = 0
        self.switch_controller(self.controller_mode)
        return bool(getattr(resp, "success", True))

    def set_payload(
        self,
        side: str,
        mass: float,
        center_of_mass: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> bool:
        """Configure the end-effector payload mass and center of mass on ``/Motion/SetLoad``."""
        if side not in ("left", "right"):
            raise ValueError(f"Expected side 'left' or 'right', got {side!r}.")
        com = np.asarray(center_of_mass, dtype=np.float32).reshape(3)

        def _fill(req: Any) -> None:
            req.arm = 0 if side == "left" else 1
            req.mass = float(mass)
            req.com = [float(com[0]), float(com[1]), float(com[2])]

        resp = self._call_service("/Motion/SetLoad", _fill)
        return bool(getattr(resp, "success", True))

    def forward_kinematics(
        self, joint_positions: Sequence[float]
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics for whole-body or dual-arm joint positions."""
        q = [float(x) for x in joint_positions]
        if "/Motion/ForwardKine" in self._services:

            def _fill(req: Any) -> None:
                req.joint_positions = q

            resp = self._call_service("/Motion/ForwardKine", _fill)
            poses = list(getattr(resp, "tcp_pose", []))
            if len(poses) >= 2:
                return {
                    "left_tcp_pose": _pose_to_xyz_rpy(poses[0]),
                    "right_tcp_pose": _pose_to_xyz_rpy(poses[1]),
                }
        state = self.get_state()
        if len(q) >= 14:
            q_arr = np.asarray(q, dtype=np.float32)
            if len(q) >= 19:
                w_q = q_arr[:5]
                l_q = q_arr[5:12]
                r_q = q_arr[12:19]
            else:
                w_q = state.waist_positions
                l_q = q_arr[:7]
                r_q = q_arr[7:14]
            l_p, l_rpy, _, _ = compute_r2_arm_fk_and_jacobian("left", l_q, w_q)
            r_p, r_rpy, _, _ = compute_r2_arm_fk_and_jacobian("right", r_q, w_q)
            return {
                "left_tcp_pose": np.concatenate([l_p, l_rpy]).astype(np.float32),
                "right_tcp_pose": np.concatenate([r_p, r_rpy]).astype(np.float32),
            }
        return {
            "left_tcp_pose": state.left_tcp_pose,
            "right_tcp_pose": state.right_tcp_pose,
        }

    def inverse_kinematics(
        self,
        *,
        left_tcp_pose: Optional[np.ndarray] = None,
        right_tcp_pose: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Compute inverse kinematics via ``/Motion/InverseKine`` or URDF Jacobian IK."""
        state = self.get_state()
        left_target = (
            np.asarray(left_tcp_pose, dtype=np.float32).reshape(6)
            if left_tcp_pose is not None
            else state.left_tcp_pose
        )
        right_target = (
            np.asarray(right_tcp_pose, dtype=np.float32).reshape(6)
            if right_tcp_pose is not None
            else state.right_tcp_pose
        )
        if "/Motion/InverseKine" in self._services:
            pose_cls = self._msg_types["Pose"]

            def _fill(req: Any) -> None:
                req.tcp_pose = [
                    _xyz_rpy_to_pose(left_target, pose_cls),
                    _xyz_rpy_to_pose(right_target, pose_cls),
                ]
                req.reference_joint_positions = list(
                    np.concatenate(
                        [state.left_joint_positions, state.right_joint_positions]
                    ).astype(float)
                )

            resp = self._call_service("/Motion/InverseKine", _fill)
            sol = np.asarray(getattr(resp, "joint_positions", []), dtype=np.float32)
            if sol.size >= 14:
                return {
                    "left_joint_positions": sol[:7].copy(),
                    "right_joint_positions": sol[7:14].copy(),
                }
        self._ensure_cmd_state(state)
        q_l = (
            self._cmd_left_arm
            if self._cmd_left_arm is not None
            else state.left_joint_positions
        )
        q_r = (
            self._cmd_right_arm
            if self._cmd_right_arm is not None
            else state.right_joint_positions
        )
        q_w = self._cmd_waist if self._cmd_waist is not None else state.waist_positions
        if left_tcp_pose is not None:
            p_l, _, _, _ = compute_r2_arm_fk_and_jacobian("left", q_l, q_w)
            q_l = compute_r2_arm_cartesian_step(
                "left", q_l, left_target[:3] - p_l, q_waist=q_w
            )
        if right_tcp_pose is not None:
            p_r, _, _, _ = compute_r2_arm_fk_and_jacobian("right", q_r, q_w)
            q_r = compute_r2_arm_cartesian_step(
                "right", q_r, right_target[:3] - p_r, q_waist=q_w
            )
        return {
            "left_joint_positions": q_l.copy(),
            "right_joint_positions": q_r.copy(),
        }

    def move_left_arm_joint(self, positions: np.ndarray) -> None:
        """Stream 7-DoF left arm joint targets over ``/Motion/ServoJ``."""
        q = np.asarray(positions, dtype=np.float32).reshape(7)
        self._publish_servoj(R2_LEFT_ARM_JOINTS, q)

    def move_right_arm_joint(self, positions: np.ndarray) -> None:
        """Stream 7-DoF right arm joint targets over ``/Motion/ServoJ``."""
        q = np.asarray(positions, dtype=np.float32).reshape(7)
        self._publish_servoj(R2_RIGHT_ARM_JOINTS, q)

    def move_dual_arm_joint(
        self, left_positions: np.ndarray, right_positions: np.ndarray
    ) -> None:
        """Stream synchronized 14-DoF dual-arm joint targets over ``/Motion/ServoJ``."""
        ql = np.asarray(left_positions, dtype=np.float32).reshape(7)
        qr = np.asarray(right_positions, dtype=np.float32).reshape(7)
        self._publish_servoj(
            R2_LEFT_ARM_JOINTS + R2_RIGHT_ARM_JOINTS,
            np.concatenate([ql, qr]),
        )

    def move_waist_joint(self, positions: np.ndarray) -> None:
        """Stream waist joint targets over ``/Motion/ServoJ``."""
        expected = len(self.waist_joint_names)
        q = np.asarray(positions, dtype=np.float32).reshape(expected)
        self._publish_servoj(self.waist_joint_names, q)

    def move_neck_joint(self, positions: np.ndarray) -> None:
        """Stream 2-DoF neck joint targets over ``/Motion/ServoJ``."""
        q = np.asarray(positions, dtype=np.float32).reshape(2)
        self._publish_servoj(R2_NECK_JOINTS, q)

    def move_left_arm_joint_velocity(self, velocities: np.ndarray) -> None:
        """Command 7-DoF left arm joint velocities via ``/Motion/ServoJ``."""
        dq = np.asarray(velocities, dtype=np.float32).reshape(7)
        state = self.get_state()
        self._publish_servoj(
            R2_LEFT_ARM_JOINTS, state.left_joint_positions, velocities=dq
        )

    def move_right_arm_joint_velocity(self, velocities: np.ndarray) -> None:
        """Command 7-DoF right arm joint velocities via ``/Motion/ServoJ``."""
        dq = np.asarray(velocities, dtype=np.float32).reshape(7)
        state = self.get_state()
        self._publish_servoj(
            R2_RIGHT_ARM_JOINTS, state.right_joint_positions, velocities=dq
        )

    def apply_left_arm_torque(self, torques: np.ndarray) -> None:
        """Command 7-DoF left arm joint torques via ``/Motion/ServoJ``."""
        tau = np.asarray(torques, dtype=np.float32).reshape(7)
        state = self.get_state()
        self._publish_servoj(
            R2_LEFT_ARM_JOINTS, state.left_joint_positions, efforts=tau
        )

    def apply_right_arm_torque(self, torques: np.ndarray) -> None:
        """Command 7-DoF right arm joint torques via ``/Motion/ServoJ``."""
        tau = np.asarray(torques, dtype=np.float32).reshape(7)
        state = self.get_state()
        self._publish_servoj(
            R2_RIGHT_ARM_JOINTS, state.right_joint_positions, efforts=tau
        )

    def move_left_arm_movej(
        self,
        positions: np.ndarray,
        duration: float = 3.0,
        *,
        max_velocity: Optional[np.ndarray] = None,
    ) -> bool:
        """Execute a point-to-point joint trajectory on the left arm via ``MoveJ``."""
        if self.enable_fault_recovery and self._has_active_fault_or_mode_drop(
            check_mode=False
        ):
            self._check_and_recover("movej_precheck")
        q = np.asarray(positions, dtype=np.float32).reshape(7)
        self._ensure_cmd_state(self.get_state())
        self._cmd_left_arm = q.copy()
        if "/Motion/MoveJ" not in self._services:
            if self._is_real_rframe and self._active_real_mode != R2_MODE_HOLD:
                self.switch_controller(R2_MODE_HOLD)
            whole_q = self._build_whole_body_joint_vector(left_arm=q)

            def _fill_real(req: Any) -> None:
                req.task_id = 1
                req.joint = whole_q
                req.scale = 0.5
                req.start = True

            resp = self._call_service(
                "/robotic/movej_cmd", _fill_real, timeout=max(duration + 5.0, 5.0)
            )
            err = getattr(resp, "error_code", None)
            ok = bool(getattr(err, "code", 0) == 0) if err is not None else True
            if not ok and self.enable_fault_recovery:
                self._check_and_recover("movej_retry")
                self._cmd_left_arm = q.copy()
                if self._is_real_rframe and self._active_real_mode != R2_MODE_HOLD:
                    self.switch_controller(R2_MODE_HOLD)
                resp = self._call_service(
                    "/robotic/movej_cmd", _fill_real, timeout=max(duration + 5.0, 5.0)
                )
                err = getattr(resp, "error_code", None)
                ok = bool(getattr(err, "code", 0) == 0) if err is not None else True
            return ok

        def _fill(req: Any) -> None:
            req.joint_names = list(R2_LEFT_ARM_JOINTS)
            req.target_positions = [float(x) for x in q]
            req.duration = float(duration)
            if max_velocity is not None:
                req.max_velocity = [
                    float(x)
                    for x in np.asarray(max_velocity, dtype=np.float32).reshape(-1)
                ]

        resp = self._call_service(
            "/Motion/MoveJ", _fill, timeout=max(duration + 5.0, 5.0)
        )
        ok = bool(getattr(resp, "success", True))
        if not ok and self.enable_fault_recovery:
            self._check_and_recover("movej_retry")
            self._cmd_left_arm = q.copy()
            resp = self._call_service(
                "/Motion/MoveJ", _fill, timeout=max(duration + 5.0, 5.0)
            )
            ok = bool(getattr(resp, "success", True))
        return ok

    def move_right_arm_movej(
        self,
        positions: np.ndarray,
        duration: float = 3.0,
        *,
        max_velocity: Optional[np.ndarray] = None,
    ) -> bool:
        """Execute a point-to-point joint trajectory on the right arm via ``MoveJ``."""
        if self.enable_fault_recovery and self._has_active_fault_or_mode_drop(
            check_mode=False
        ):
            self._check_and_recover("movej_precheck")
        q = np.asarray(positions, dtype=np.float32).reshape(7)
        self._ensure_cmd_state(self.get_state())
        self._cmd_right_arm = q.copy()
        if "/Motion/MoveJ" not in self._services:
            if self._is_real_rframe and self._active_real_mode != R2_MODE_HOLD:
                self.switch_controller(R2_MODE_HOLD)
            whole_q = self._build_whole_body_joint_vector(right_arm=q)

            def _fill_real(req: Any) -> None:
                req.task_id = 1
                req.joint = whole_q
                req.scale = 0.5
                req.start = True

            resp = self._call_service(
                "/robotic/movej_cmd", _fill_real, timeout=max(duration + 5.0, 5.0)
            )
            err = getattr(resp, "error_code", None)
            ok = bool(getattr(err, "code", 0) == 0) if err is not None else True
            if not ok and self.enable_fault_recovery:
                self._check_and_recover("movej_retry")
                self._cmd_right_arm = q.copy()
                if self._is_real_rframe and self._active_real_mode != R2_MODE_HOLD:
                    self.switch_controller(R2_MODE_HOLD)
                resp = self._call_service(
                    "/robotic/movej_cmd", _fill_real, timeout=max(duration + 5.0, 5.0)
                )
                err = getattr(resp, "error_code", None)
                ok = bool(getattr(err, "code", 0) == 0) if err is not None else True
            return ok

        def _fill(req: Any) -> None:
            req.joint_names = list(R2_RIGHT_ARM_JOINTS)
            req.target_positions = [float(x) for x in q]
            req.duration = float(duration)
            if max_velocity is not None:
                req.max_velocity = [
                    float(x)
                    for x in np.asarray(max_velocity, dtype=np.float32).reshape(-1)
                ]

        resp = self._call_service(
            "/Motion/MoveJ", _fill, timeout=max(duration + 5.0, 5.0)
        )
        ok = bool(getattr(resp, "success", True))
        if not ok and self.enable_fault_recovery:
            self._check_and_recover("movej_retry")
            self._cmd_right_arm = q.copy()
            resp = self._call_service(
                "/Motion/MoveJ", _fill, timeout=max(duration + 5.0, 5.0)
            )
            ok = bool(getattr(resp, "success", True))
        return ok

    def move_left_arm_cartesian(self, pose: np.ndarray, duration: float = 1.0) -> bool:
        """Move the left arm TCP to ``pose`` ``[x, y, z, r, p, y]``."""
        target = np.asarray(pose, dtype=np.float32).reshape(6)
        pose_cls = self._msg_types["Pose"]
        if "/Motion/MoveC" not in self._services:
            if self._is_real_rframe and self._active_real_mode != R2_MODE_HOLD:
                self.switch_controller(R2_MODE_HOLD)
            state = self.get_state()

            def _fill_real(req: Any) -> None:
                req.task_id = 1
                req.pose_l = _xyz_rpy_to_pose(target, pose_cls)
                req.pose_r = _xyz_rpy_to_pose(state.right_tcp_pose, pose_cls)
                req.scale = 0.5
                req.start = True

            resp = self._call_service(
                "/robotic/movel_cmd", _fill_real, timeout=max(duration + 5.0, 5.0)
            )
            err = getattr(resp, "error_code", None)
            return bool(getattr(err, "code", 0) == 0) if err is not None else True

        def _fill(req: Any) -> None:
            req.arm = 0
            req.target_pose = _xyz_rpy_to_pose(target, pose_cls)
            req.duration = float(duration)

        resp = self._call_service(
            "/Motion/MoveC", _fill, timeout=max(duration + 5.0, 5.0)
        )
        return bool(getattr(resp, "success", True))

    def move_right_arm_cartesian(self, pose: np.ndarray, duration: float = 1.0) -> bool:
        """Move the right arm TCP to ``pose`` ``[x, y, z, r, p, y]``."""
        target = np.asarray(pose, dtype=np.float32).reshape(6)
        pose_cls = self._msg_types["Pose"]
        if "/Motion/MoveC" not in self._services:
            if self._is_real_rframe and self._active_real_mode != R2_MODE_HOLD:
                self.switch_controller(R2_MODE_HOLD)
            state = self.get_state()

            def _fill_real(req: Any) -> None:
                req.task_id = 1
                req.pose_l = _xyz_rpy_to_pose(state.left_tcp_pose, pose_cls)
                req.pose_r = _xyz_rpy_to_pose(target, pose_cls)
                req.scale = 0.5
                req.start = True

            resp = self._call_service(
                "/robotic/movel_cmd", _fill_real, timeout=max(duration + 5.0, 5.0)
            )
            err = getattr(resp, "error_code", None)
            return bool(getattr(err, "code", 0) == 0) if err is not None else True

        def _fill(req: Any) -> None:
            req.arm = 1
            req.target_pose = _xyz_rpy_to_pose(target, pose_cls)
            req.duration = float(duration)

        resp = self._call_service(
            "/Motion/MoveC", _fill, timeout=max(duration + 5.0, 5.0)
        )
        return bool(getattr(resp, "success", True))

    def move_left_arm_velocity(self, velocity: np.ndarray, dt: float = 0.02) -> None:
        """Step the left arm TCP by integrating a 6D Cartesian velocity twist."""
        vel = np.asarray(velocity, dtype=np.float32).reshape(6)
        state = self.get_state()
        target_pose = state.left_tcp_pose + vel * float(dt)
        self.move_left_arm_cartesian(target_pose, duration=dt)

    def move_right_arm_velocity(self, velocity: np.ndarray, dt: float = 0.02) -> None:
        """Step the right arm TCP by integrating a 6D Cartesian velocity twist."""
        vel = np.asarray(velocity, dtype=np.float32).reshape(6)
        state = self.get_state()
        target_pose = state.right_tcp_pose + vel * float(dt)
        self.move_right_arm_cartesian(target_pose, duration=dt)

    def move_left_gripper(self, position: Any) -> None:
        """Command the left parallel gripper position in ``[0.0, 1.0]``."""
        val = float(
            np.clip(np.asarray(position, dtype=np.float32).reshape(-1)[0], 0.0, 1.0)
        )
        with self._lock:
            self._state.left_gripper_position = np.array([val], dtype=np.float32)
        self._publish_servoj(("GripperL",), (val,))

    def move_right_gripper(self, position: Any) -> None:
        """Command the right parallel gripper position in ``[0.0, 1.0]``."""
        val = float(
            np.clip(np.asarray(position, dtype=np.float32).reshape(-1)[0], 0.0, 1.0)
        )
        with self._lock:
            self._state.right_gripper_position = np.array([val], dtype=np.float32)
        self._publish_servoj(("GripperR",), (val,))

    def move_left_hand(self, positions: Any) -> None:
        """Command the 6-DoF left dexterous hand ``(little, ring, middle, index, thumb_bend, thumb_rotation)``."""
        arr = np.asarray(positions, dtype=np.float32).reshape(6)
        with self._lock:
            self._state.left_hand_position = arr.copy()
        self._publish_servoj(R2_HAND_JOINTS_LEFT, arr)

    def move_right_hand(self, positions: Any) -> None:
        """Command the 6-DoF right dexterous hand ``(little, ring, middle, index, thumb_bend, thumb_rotation)``."""
        arr = np.asarray(positions, dtype=np.float32).reshape(6)
        with self._lock:
            self._state.right_hand_position = arr.copy()
        self._publish_servoj(R2_HAND_JOINTS_RIGHT, arr)

    def move_base(self, velocity: np.ndarray) -> None:
        """Publish planar chassis velocity ``[vx, vy, wz]``."""
        self._require_connected()
        topic = (
            "/teleop/chassis_Twist"
            if "/teleop/chassis_Twist" in self._writers
            else "/chassis/cmd_vel"
        )
        if topic not in self._writers:
            raise RuntimeError(
                "R2 mobile base is not enabled on this connection (enable_base=False)."
            )
        vel = np.asarray(velocity, dtype=np.float32).reshape(3)
        if (
            self._is_real_rframe
            and not self._chassis_cmd_started
            and bool(np.any(np.abs(vel) > 1e-6))
            and "/chassis/cmd_start" in self._services
        ):
            self._call_service(
                "/chassis/cmd_start",
                lambda req: setattr(req, "data", True),
                timeout=1.0,
            )
            self._chassis_cmd_started = True
        channel, msg_cls = self._writers[topic]
        msg = msg_cls()
        msg.linear.x = float(vel[0])
        msg.linear.y = float(vel[1])
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(vel[2])
        channel.send(msg)
        with self._lock:
            self._state.base_velocity = vel.copy()

    def move_base_pulse(
        self,
        velocity: Sequence[float] | np.ndarray,
        duration: float = 0.5,
        rate_hz: float = 20.0,
    ) -> dict[str, np.ndarray]:
        """Stream planar chassis velocity ``[vx, vy, wz]`` for ``duration`` seconds and stop."""
        vel = np.asarray(velocity, dtype=np.float32).reshape(3)
        dt = 1.0 / max(float(rate_hz), 1.0)
        steps = max(1, int(round(max(float(duration), 0.0) / dt)))
        for _ in range(steps):
            self.move_base(vel)
            if self._is_real_rframe:
                time.sleep(dt)
        state_during = self.get_state()
        self.stop_base()
        state_after = self.get_state()
        return {
            "base_pose": state_after.base_pose,
            "base_velocity": state_during.base_velocity,
        }

    def stop_base(self) -> None:
        """Zero the chassis velocity command and cancel active navigation if supported."""
        if (
            "/teleop/chassis_Twist" in self._writers
            or "/chassis/cmd_vel" in self._writers
        ):
            self.move_base(np.zeros(3, dtype=np.float32))
        if "/navigation/control" in self._services:
            self._call_service(
                "/navigation/control",
                lambda req: setattr(req, "command", 0),
            )

    def navigate_to(
        self, x: float, y: float, yaw: float = 0.0, timeout: float = 30.0
    ) -> bool:
        """Assign a navigation goal ``(x, y, yaw)`` via ``/task_manager/assign_task``."""
        if "/task_manager/assign_task" not in self._services:
            raise RuntimeError(
                "R2 mobile base is not enabled on this connection (enable_base=False)."
            )

        def _fill(req: Any) -> None:
            req.task_type = "navigation"
            req.target_x = float(x)
            req.target_y = float(y)
            req.target_yaw = float(yaw)

        resp = self._call_service("/task_manager/assign_task", _fill, timeout=timeout)
        return bool(getattr(resp, "success", True))

    def check_camera(self, camera_name: str, timeout: float = 0.5) -> bool:
        """Return True if ``camera_name`` has delivered at least one RGB frame."""
        self._require_connected()
        event = self._camera_events.get(camera_name)
        if event is None:
            raise KeyError(f"Unknown R2 camera {camera_name!r}.")
        return bool(event.wait(timeout=max(timeout, 0.0)))

    def get_camera(self, camera_name: str) -> np.ndarray:
        """Return the latest RGB frame for ``camera_name``."""
        self._require_connected()
        if camera_name not in R2_CAMERA_TOPICS:
            raise KeyError(f"Unknown R2 camera {camera_name!r}.")
        event = self._camera_events.get(camera_name)
        if event is not None and not event.is_set():
            event.wait(timeout=0.5)
        with self._lock:
            frame = self._camera_frames.get(camera_name)
            if frame is not None:
                return frame.copy()
        width, height = R2_CAMERA_TOPICS[camera_name]["resolution"]
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_camera_depth(self, camera_name: str) -> Optional[np.ndarray]:
        """Return the latest depth image for ``camera_name`` when ``enable_depth=True``."""
        self._require_connected()
        if not self.enable_depth:
            return None
        with self._lock:
            depth = self._camera_depths.get(camera_name)
            if depth is not None:
                return depth.copy()
        width, height = R2_CAMERA_TOPICS[camera_name]["resolution"]
        return np.zeros((height, width), dtype=np.float32)

    def get_camera_observation(self, camera_name: str) -> dict[str, Any]:
        """Return ``{'frame': rgb}`` plus ``'depth'`` when depth is enabled."""
        obs: dict[str, Any] = {"frame": self.get_camera(camera_name)}
        if self.enable_depth:
            obs["depth"] = self.get_camera_depth(camera_name)
        return obs
