"""Shared 11D goal construction for training hindsight and closed-loop inference."""

from __future__ import annotations

import math

import numpy as np

from .geometry import local_waypoints_by_path_distance, relative_yaw, transform_point_to_yaw_frame


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """Build a yaw-only WXYZ quaternion."""

    half = 0.5 * float(yaw)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def closest_path_index(path_w: np.ndarray, robot_xy: np.ndarray) -> int:
    """Return the index of the closest point on a 3D path (uses XY only)."""

    path_xy = np.asarray(path_w, dtype=np.float32)[:, :2]
    robot_xy = np.asarray(robot_xy, dtype=np.float32).reshape(-1)[:2]
    dists = np.linalg.norm(path_xy - robot_xy[None, :], axis=-1)
    return int(np.argmin(dists))


def advance_path_progress(path_w: np.ndarray, robot_xy: np.ndarray, progress_idx: int) -> int:
    """Monotonic path index: never move backward when the robot overshoots."""

    closest = closest_path_index(path_w, robot_xy)
    return max(int(progress_idx), closest)


def resolve_end_idx(
    cumulative_xy: np.ndarray,
    start_idx: int,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
) -> int:
    """Pick an end index using arc-length lookahead (matches training scale).

    Training hindsight uses ``segment_steps in [50, 150]`` ticks at 50 Hz, i.e.
    about 1–3 s ahead. We map that to ``speed * goal_horizon_steps * dt`` meters
    along the reference path instead of jumping to the global terminus.
    """

    if len(cumulative_xy) == 0:
        return 0
    start_idx = int(np.clip(start_idx, 0, len(cumulative_xy) - 1))
    lookahead_m = max(float(speed) * float(goal_horizon_steps) * float(dt), float(dt) * float(speed))
    target_s = min(cumulative_xy[start_idx] + lookahead_m, cumulative_xy[-1])
    end_idx = int(np.searchsorted(cumulative_xy, target_s, side="left"))
    return int(np.clip(end_idx, start_idx, len(cumulative_xy) - 1))


def build_goal_vector(
    pos_w: np.ndarray,
    cumulative_xy: np.ndarray,
    step_idx: int,
    end_idx: int,
    origin_w: np.ndarray,
    quat_origin: np.ndarray,
    *,
    quat_w: np.ndarray | None = None,
    yaws_w: np.ndarray | None = None,
    dt: float = 0.02,
    waypoint_distances: tuple[float, float, float] = (0.4, 0.8, 1.2),
    v_req_clip: float = 2.0,
) -> np.ndarray:
    """Build the 11D goal vector shared by the DataLoader and inference."""

    if quat_w is None and yaws_w is None:
        raise ValueError("Either quat_w or yaws_w must be provided.")

    waypoints = local_waypoints_by_path_distance(
        pos_w,
        step_idx,
        cumulative_xy,
        origin_w,
        quat_origin,
        distances=waypoint_distances,
        end_idx=end_idx,
    )
    target_w = pos_w[end_idx]
    target_rel = transform_point_to_yaw_frame(target_w, origin_w, quat_origin)

    if quat_w is not None:
        dyaw = relative_yaw(quat_origin, quat_w[end_idx])
    else:
        dyaw = relative_yaw(quat_origin, yaw_to_quat_wxyz(float(yaws_w[end_idx])))

    path_length_remaining = float(cumulative_xy[end_idx] - cumulative_xy[step_idx])
    time_remaining = max((end_idx - step_idx) * dt, dt)
    v_req = float(np.clip(path_length_remaining / (time_remaining + 1.0e-3), 0.0, v_req_clip))

    return np.asarray(
        [
            *waypoints.tolist(),
            float(target_rel[0]),
            float(target_rel[1]),
            float(target_w[2]),  # Absolute target base height (project-specific spatial goal)
            float(dyaw),
            v_req,
        ],
        dtype=np.float32,
    )


def build_goal_from_path(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    yaws_w: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    start_idx: int | None = None,
    waypoint_distances: tuple[float, float, float] = (0.4, 0.8, 1.2),
    v_req_clip: float = 2.0,
) -> np.ndarray:
    """Build a single-env goal from a planned world-frame path."""

    if start_idx is None:
        start_idx = closest_path_index(path_w, robot_pos_w)
    start_idx = int(np.clip(start_idx, 0, len(path_w) - 1))
    end_idx = resolve_end_idx(
        cumulative_xy,
        start_idx,
        goal_horizon_steps=goal_horizon_steps,
        dt=dt,
        speed=speed,
    )
    path_length_remaining = float(cumulative_xy[end_idx] - cumulative_xy[start_idx])
    time_remaining = max(float(goal_horizon_steps) * dt, dt)
    v_req = float(np.clip(path_length_remaining / (time_remaining + 1.0e-3), 0.0, v_req_clip))

    waypoints = local_waypoints_by_path_distance(
        path_w,
        start_idx,
        cumulative_xy,
        robot_pos_w,
        robot_quat_w,
        distances=waypoint_distances,
        end_idx=end_idx,
    )
    target_rel = transform_point_to_yaw_frame(path_w[end_idx], robot_pos_w, robot_quat_w)
    dyaw = relative_yaw(robot_quat_w, yaw_to_quat_wxyz(float(yaws_w[end_idx])))

    return np.asarray(
        [
            *waypoints.tolist(),
            float(target_rel[0]),
            float(target_rel[1]),
            float(path_w[end_idx, 2]),  # Absolute target base height
            float(dyaw),
            v_req,
        ],
        dtype=np.float32,
    )


def build_goal_batch_from_path(
    path_w: np.ndarray,
    cumulative_xy: np.ndarray,
    yaws_w: np.ndarray,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    path_progress: np.ndarray | None = None,
    waypoint_distances: tuple[float, float, float] = (0.4, 0.8, 1.2),
    v_req_clip: float = 2.0,
) -> np.ndarray:
    """Build goals for a batch of robots following the same planned path."""

    num_envs = robot_pos_w.shape[0]
    if path_progress is None:
        path_progress = np.zeros(num_envs, dtype=np.int32)

    goals = []
    for i in range(num_envs):
        start_idx = advance_path_progress(path_w, robot_pos_w[i], int(path_progress[i]))
        path_progress[i] = start_idx
        goals.append(
            build_goal_from_path(
                path_w,
                cumulative_xy,
                yaws_w,
                robot_pos_w[i],
                robot_quat_w[i],
                goal_horizon_steps=goal_horizon_steps,
                dt=dt,
                speed=speed,
                start_idx=start_idx,
                waypoint_distances=waypoint_distances,
                v_req_clip=v_req_clip,
            )
        )
    return np.stack(goals, axis=0).astype(np.float32)
