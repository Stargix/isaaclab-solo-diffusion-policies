"""Geometry helpers for hindsight goal construction."""

from __future__ import annotations

import math

import numpy as np


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert an Isaac Lab WXYZ quaternion to a 3x3 rotation matrix."""

    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def yaw_from_rotmat(rot: np.ndarray) -> float:
    """Extract yaw from a world-from-body rotation matrix."""

    return math.atan2(float(rot[1, 0]), float(rot[0, 0]))


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def transform_point_to_local(
    point_w: np.ndarray,
    origin_w: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Express a world point in the robot body frame."""

    rot_w_b = quat_wxyz_to_rotmat(quat_wxyz)
    return rot_w_b.T @ (point_w - origin_w)


def transform_point_to_yaw_frame(
    point_w: np.ndarray,
    origin_w: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Express a world point in the robot's gravity-aligned yaw frame (roll=0, pitch=0)."""
    yaw = yaw_from_rotmat(quat_wxyz_to_rotmat(quat_wxyz))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    dx = float(point_w[0] - origin_w[0])
    dy = float(point_w[1] - origin_w[1])
    dz = float(point_w[2] - origin_w[2])
    return np.array([
        cos_y * dx + sin_y * dy,
        -sin_y * dx + cos_y * dy,
        dz,
    ], dtype=np.float32)



def relative_yaw(from_quat_wxyz: np.ndarray, to_quat_wxyz: np.ndarray) -> float:
    """Yaw of ``to`` expressed relative to ``from``."""

    yaw_from = yaw_from_rotmat(quat_wxyz_to_rotmat(from_quat_wxyz))
    yaw_to = yaw_from_rotmat(quat_wxyz_to_rotmat(to_quat_wxyz))
    return wrap_to_pi(yaw_to - yaw_from)


def cumulative_xy_lengths(pos_w: np.ndarray) -> np.ndarray:
    """Return cumulative XY path length with shape (T,)."""

    if len(pos_w) == 0:
        return np.zeros(0, dtype=np.float32)
    deltas = np.diff(pos_w[:, :2], axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.zeros(len(pos_w), dtype=np.float32)
    cumulative[1:] = np.cumsum(step_lengths, dtype=np.float32)
    return cumulative


def local_waypoints_by_path_distance(
    pos_w: np.ndarray,
    start_idx: int,
    cumulative_lengths: np.ndarray,
    origin_w: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    distances: tuple[float, ...],
    end_idx: int,
) -> np.ndarray:
    """Pick future waypoints by accumulated path distance and express them locally."""

    start_s = cumulative_lengths[start_idx]
    waypoints: list[float] = []
    max_idx = min(end_idx, len(pos_w) - 1)
    for distance in distances:
        target_s = min(start_s + distance, cumulative_lengths[max_idx])
        idx = int(np.searchsorted(cumulative_lengths, target_s, side="left"))
        idx = min(max(idx, start_idx), max_idx)
        local = transform_point_to_yaw_frame(pos_w[idx], origin_w, quat_wxyz)
        waypoints.extend([float(local[0]), float(local[1])])
    return np.asarray(waypoints, dtype=np.float32)

