"""A deterministic full-path tracker producing ``[vx, vy, wz, height]``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def cumulative_xy(path_w: np.ndarray) -> np.ndarray:
    lengths = np.zeros(len(path_w), dtype=np.float32)
    if len(path_w) > 1:
        lengths[1:] = np.cumsum(np.linalg.norm(np.diff(path_w[:, :2], axis=0), axis=-1))
    return lengths


def path_yaws(path: np.ndarray) -> np.ndarray:
    if path.shape[1] >= 4:
        return path[:, 3].astype(np.float32)
    delta = np.diff(path[:, :2], axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    if yaw.size == 0:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate([yaw, yaw[-1:]]).astype(np.float32)


def align_path_to_pose(path: np.ndarray, robot_pos_w: np.ndarray, robot_quat_w: np.ndarray) -> np.ndarray:
    """Treat path XY/yaw as robot-relative while preserving absolute height."""

    aligned = np.asarray(path, dtype=np.float32).copy()
    relative_xy = aligned[:, :2] - aligned[0, :2]
    yaw = yaw_from_quat_wxyz(np.asarray(robot_quat_w))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float32)
    aligned[:, :2] = np.asarray(robot_pos_w)[:2] + relative_xy @ rotation.T
    if aligned.shape[1] >= 4:
        aligned[:, 3] = np.asarray([wrap_angle(float(value) + yaw) for value in aligned[:, 3]])
    return aligned


@dataclass(frozen=True)
class PathTrackerConfig:
    desired_speed: float = 0.4
    position_lookahead_s: float = 0.75
    height_lookahead_s: float = 2.0
    yaw_gain: float = 1.5
    stop_distance_m: float = 0.10
    max_vx: float = 0.75
    max_vy: float = 0.50
    max_wz: float = 0.50
    search_back: int = 5
    search_forward: int = 50


class PathCommandTracker:
    """Track a complete route while exposing only local commands to locomotion."""

    def __init__(self, path: np.ndarray, config: PathTrackerConfig, num_envs: int = 1):
        path = np.asarray(path, dtype=np.float32)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] not in (3, 4):
            raise ValueError("Path must have shape [N,3] or [N,4], with N >= 2.")
        if not np.isfinite(path).all():
            raise ValueError("Path contains NaN or Inf.")
        self.path_w = path[:, :3].copy()
        self.yaws_w = path_yaws(path)
        self.cumulative = cumulative_xy(self.path_w)
        self.config = config
        self.progress = np.zeros(num_envs, dtype=np.int32)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        config: PathTrackerConfig,
        num_envs: int = 1,
        *,
        origin_pos_w: np.ndarray | None = None,
        origin_quat_w: np.ndarray | None = None,
    ):
        array = np.load(path)
        if (origin_pos_w is None) != (origin_quat_w is None):
            raise ValueError("origin_pos_w and origin_quat_w must be supplied together.")
        if origin_pos_w is not None:
            array = align_path_to_pose(array, origin_pos_w, origin_quat_w)
        return cls(array, config, num_envs=num_envs)

    def reset(self, env_idx: int | None = None) -> None:
        if env_idx is None:
            self.progress[:] = 0
        else:
            self.progress[int(env_idx)] = 0

    def _advance_progress(self, robot_xy: np.ndarray, env_idx: int) -> int:
        current = int(self.progress[env_idx])
        low = max(0, current - self.config.search_back)
        high = min(len(self.path_w), current + self.config.search_forward + 1)
        distance = np.linalg.norm(self.path_w[low:high, :2] - robot_xy[None, :2], axis=-1)
        closest = low + int(np.argmin(distance))
        self.progress[env_idx] = max(current, closest)
        return int(self.progress[env_idx])

    def _index_at_distance(self, progress_idx: int, distance_m: float) -> int:
        target_s = min(float(self.cumulative[progress_idx]) + max(0.0, distance_m), float(self.cumulative[-1]))
        return int(np.clip(np.searchsorted(self.cumulative, target_s, side="left"), progress_idx, len(self.path_w) - 1))

    def command(self, robot_pos_w: np.ndarray, robot_quat_w: np.ndarray, env_idx: int = 0) -> np.ndarray:
        progress_idx = self._advance_progress(np.asarray(robot_pos_w)[:2], env_idx)
        remaining = float(self.cumulative[-1] - self.cumulative[progress_idx])
        speed = min(self.config.desired_speed, self.config.desired_speed * remaining / self.config.stop_distance_m)
        speed = max(0.0, speed)

        motion_idx = self._index_at_distance(progress_idx, speed * self.config.position_lookahead_s)
        height_idx = self._index_at_distance(progress_idx, speed * self.config.height_lookahead_s)
        robot_yaw = yaw_from_quat_wxyz(np.asarray(robot_quat_w))
        delta_w = self.path_w[motion_idx, :2] - np.asarray(robot_pos_w)[:2]
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        delta_b = np.asarray((c * delta_w[0] + s * delta_w[1], -s * delta_w[0] + c * delta_w[1]))
        norm = float(np.linalg.norm(delta_b))
        direction = delta_b / max(norm, 1.0e-6)
        vx = float(np.clip(speed * direction[0], -self.config.max_vx, self.config.max_vx))
        vy = float(np.clip(speed * direction[1], -self.config.max_vy, self.config.max_vy))
        yaw_error = wrap_angle(float(self.yaws_w[motion_idx]) - robot_yaw)
        wz = float(np.clip(self.config.yaw_gain * yaw_error, -self.config.max_wz, self.config.max_wz))
        height = float(self.path_w[height_idx, 2])
        return np.asarray((vx, vy, wz, height), dtype=np.float32)

    def command_batch(self, robot_pos_w: np.ndarray, robot_quat_w: np.ndarray) -> np.ndarray:
        if len(robot_pos_w) != len(self.progress):
            raise ValueError("Robot batch size does not match tracker state.")
        return np.stack(
            [self.command(robot_pos_w[index], robot_quat_w[index], index) for index in range(len(self.progress))]
        )
