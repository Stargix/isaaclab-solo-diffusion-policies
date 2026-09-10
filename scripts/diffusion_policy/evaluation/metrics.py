"""Pure NumPy route metrics used by the Phase A causal audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RouteMetrics:
    """Longitudinal and transverse metrics for one finite-horizon rollout."""

    target_progress_m: float
    final_progress_m: float
    furthest_progress_m: float
    completion_ratio: float
    horizon_mean_speed_m_s: float
    horizon_speed_ratio: float
    arrived: bool
    arrival_time_s: float
    arrival_mean_speed_m_s: float
    arrival_speed_ratio: float
    active_mean_speed_m_s: float
    active_speed_ratio: float
    schedule_mae_m: float
    final_schedule_error_m: float
    cross_track_rmse_m: float
    cross_track_p95_m: float
    terminal_position_error_m: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSuccessMetrics:
    """Task result evaluated at the first valid entry into the goal region."""

    arrived: bool
    success: bool
    time_s: float
    position_error_m: float
    yaw_error_rad: float
    height_error_m: float
    mean_speed_error_m_s: float
    profile_height_mae_m: float
    profile_height_within_tolerance_fraction: float
    yaw_ok: bool
    terminal_height_ok: bool
    mean_speed_ok: bool
    profile_height_ok: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def physical_state_valid(
    root_positions_w: np.ndarray,
    planar_speeds_m_s: np.ndarray,
    tilt_deg: np.ndarray,
    actions: np.ndarray,
    *,
    min_base_height_m: float = 0.08,
    max_base_height_m: float = 0.50,
    max_planar_speed_m_s: float = 5.0,
    max_tilt_deg: float = 60.0,
) -> np.ndarray:
    """Return a per-environment simulator sanity mask.

    IsaacLab's base task only terminates on configured contacts. A PhysX escape
    can therefore remain finite while the robot is metres below the plane and
    be reported as a survivor. These deliberately loose bounds reject only
    states outside the locomotion experiment's physical envelope; they are not
    tracking objectives and do not alter training.
    """

    positions = np.asarray(root_positions_w)
    speeds = np.asarray(planar_speeds_m_s).reshape(-1)
    tilts = np.asarray(tilt_deg).reshape(-1)
    action_values = np.asarray(actions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("root_positions_w must have shape (N, 3).")
    count = len(positions)
    if speeds.shape != (count,) or tilts.shape != (count,):
        raise ValueError("speed and tilt arrays must have shape (N,).")
    if action_values.ndim != 2 or action_values.shape[0] != count:
        raise ValueError("actions must have shape (N, action_dim).")
    if not 0.0 < min_base_height_m < max_base_height_m:
        raise ValueError("base-height sanity bounds are invalid.")
    if max_planar_speed_m_s <= 0.0 or not 0.0 < max_tilt_deg <= 180.0:
        raise ValueError("speed and tilt sanity bounds must be positive.")
    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(speeds)
        & np.isfinite(tilts)
        & np.isfinite(action_values).all(axis=1)
    )
    return (
        finite
        & (positions[:, 2] >= min_base_height_m)
        & (positions[:, 2] <= max_base_height_m)
        & (speeds <= max_planar_speed_m_s)
        & (tilts <= max_tilt_deg)
    )


def valid_post_step_mask(horizon_steps: int, failure_action_step: int | None) -> np.ndarray:
    """Mask metrics returned after IsaacLab's in-step auto-reset.

    ``failure_action_step`` is one-based. On that action IsaacLab has already
    reset before callers can read robot tensors, so the corresponding sample
    and everything after it are invalid for the terminated rollout.
    """

    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive.")
    mask = np.ones(horizon_steps, dtype=bool)
    if failure_action_step is None:
        return mask
    if not 1 <= failure_action_step <= horizon_steps:
        raise ValueError("failure_action_step must be within the evaluation horizon.")
    mask[failure_action_step - 1 :] = False
    return mask


def _validate_polyline(path_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path_xy must have shape (N, 2) with N >= 2.")
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1.0e-9):
        raise ValueError("path_xy contains a zero-length segment.")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return path, segments, cumulative


def truncate_path_to_arc_length(
    path_w: np.ndarray,
    yaws_w: np.ndarray,
    length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a 3-D path ending exactly at the requested XY arc length."""

    path = np.asarray(path_w, dtype=np.float32)
    yaws = np.asarray(yaws_w, dtype=np.float32).reshape(-1)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError("path_w must have shape (N, 3) with N >= 2.")
    if len(path) != len(yaws):
        raise ValueError("path_w and yaws_w must have matching lengths.")
    if length_m <= 0.0:
        raise ValueError("length_m must be positive.")
    _, _, arc = _validate_polyline(path[:, :2])
    if length_m > float(arc[-1]) + 1.0e-5:
        raise ValueError(
            f"Requested route length {length_m:.3f} m exceeds generated path "
            f"length {float(arc[-1]):.3f} m."
        )
    if length_m >= float(arc[-1]) - 1.0e-6:
        return path.copy(), yaws.copy()
    upper = int(np.clip(np.searchsorted(arc, length_m, side="left"), 1, len(path) - 1))
    lower = upper - 1
    segment_length = max(float(arc[upper] - arc[lower]), 1.0e-8)
    alpha = float((length_m - arc[lower]) / segment_length)
    terminal = path[lower] + alpha * (path[upper] - path[lower])
    segment = path[upper, :2] - path[lower, :2]
    terminal_yaw = float(np.arctan2(segment[1], segment[0]))
    return (
        np.concatenate((path[:upper], terminal[None, :]), axis=0),
        np.concatenate((yaws[:upper], np.asarray([terminal_yaw], dtype=np.float32))),
    )


def point_at_progress(path_xy: np.ndarray, progress_m: float) -> np.ndarray:
    """Interpolate a point at arc length ``progress_m`` on a polyline."""

    path, segments, cumulative = _validate_polyline(path_xy)
    progress = float(np.clip(progress_m, 0.0, cumulative[-1]))
    segment_index = min(int(np.searchsorted(cumulative, progress, side="right") - 1), len(segments) - 1)
    segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
    fraction = (progress - cumulative[segment_index]) / segment_length
    return path[segment_index] + fraction * segments[segment_index]


def project_trajectory_to_polyline(
    positions_xy: np.ndarray,
    path_xy: np.ndarray,
    *,
    search_back: int = 5,
    search_forward: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a trajectory onto a locally tracked polyline.

    The search window advances monotonically by segment index.  This prevents a
    circle or self-crossing route from jumping to a geometrically coincident but
    temporally distant branch, while still allowing small backwards motion to
    remain visible in the returned instantaneous progress.
    """

    positions = np.asarray(positions_xy, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_xy must have shape (T, 2).")
    if len(positions) == 0:
        raise ValueError("positions_xy must contain at least one point.")
    if search_back < 0 or search_forward < 1:
        raise ValueError("Require search_back >= 0 and search_forward >= 1.")

    path, segments, cumulative = _validate_polyline(path_xy)
    lengths_sq = np.sum(np.square(segments), axis=1)
    segment_cursor = 0
    progress = np.empty(len(positions), dtype=np.float64)
    cross_track = np.empty(len(positions), dtype=np.float64)

    for step, position in enumerate(positions):
        low = max(0, segment_cursor - search_back)
        high = min(len(segments), segment_cursor + search_forward + 1)
        starts = path[low:high]
        local_segments = segments[low:high]
        fractions = np.sum((position - starts) * local_segments, axis=1) / lengths_sq[low:high]
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts + fractions[:, None] * local_segments
        distances_sq = np.sum(np.square(position - projections), axis=1)
        local_best = int(np.argmin(distances_sq))
        segment_index = low + local_best
        segment_cursor = max(segment_cursor, segment_index)
        segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
        progress[step] = cumulative[segment_index] + fractions[local_best] * segment_length
        cross_track[step] = np.sqrt(distances_sq[local_best])

    return progress.astype(np.float32), cross_track.astype(np.float32)


def compute_route_metrics(
    positions_xy: np.ndarray,
    path_xy: np.ndarray,
    *,
    requested_speed_m_s: float,
    dt: float,
    horizon_steps: int,
    start_position_xy: np.ndarray | None = None,
) -> RouteMetrics:
    """Compute route tracking without conflating survival and progress.

    ``positions_xy`` contains only samples up to failure.  Horizon-normalized
    speed and completion still use the full requested duration, so falling or
    stopping early cannot look artificially successful.
    """

    if requested_speed_m_s <= 0.0:
        raise ValueError("requested_speed_m_s must be positive.")
    if dt <= 0.0 or horizon_steps < 1:
        raise ValueError("Require dt > 0 and horizon_steps >= 1.")
    positions = np.asarray(positions_xy, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions_xy must have shape (T, 2) with T >= 1.")
    if len(positions) > horizon_steps:
        raise ValueError("positions_xy cannot be longer than horizon_steps.")

    path, _, cumulative = _validate_polyline(path_xy)
    if start_position_xy is None:
        start_progress = 0.0
        progress_absolute, cross_track = project_trajectory_to_polyline(positions, path)
    else:
        start = np.asarray(start_position_xy, dtype=np.float64).reshape(-1)
        if start.size != 2:
            raise ValueError("start_position_xy must contain exactly two values.")
        combined_progress, combined_cross_track = project_trajectory_to_polyline(
            np.vstack((start, positions)), path
        )
        start_progress = float(combined_progress[0])
        progress_absolute = combined_progress[1:]
        cross_track = combined_cross_track[1:]
    progress = progress_absolute - start_progress
    path_length = float(cumulative[-1])
    horizon_duration = float(horizon_steps) * dt
    active_duration = float(len(positions)) * dt
    remaining_path_length = max(0.0, path_length - start_progress)
    target_progress = min(requested_speed_m_s * horizon_duration, remaining_path_length)
    scheduled = np.minimum(
        requested_speed_m_s * (np.arange(len(positions), dtype=np.float64) + 1.0) * dt,
        remaining_path_length,
    )
    final_progress = float(progress[-1])
    furthest_progress = float(np.max(progress))
    terminal_target = point_at_progress(path, start_progress + target_progress)
    terminal_error = float(np.linalg.norm(positions[-1] - terminal_target))
    completion_ratio = final_progress / max(target_progress, 1.0e-9)
    horizon_mean_speed = final_progress / horizon_duration
    active_mean_speed = final_progress / active_duration
    monotonic_progress = np.maximum.accumulate(progress)
    arrival_candidates = np.flatnonzero(monotonic_progress >= target_progress)
    arrived = bool(arrival_candidates.size)
    if arrived:
        arrival_index = int(arrival_candidates[0])
        previous_progress = 0.0 if arrival_index == 0 else float(monotonic_progress[arrival_index - 1])
        current_progress = float(monotonic_progress[arrival_index])
        progress_delta = current_progress - previous_progress
        interpolation = (
            (target_progress - previous_progress) / progress_delta
            if progress_delta > 1.0e-9
            else 1.0
        )
        arrival_time = (float(arrival_index) + float(np.clip(interpolation, 0.0, 1.0))) * dt
        arrival_mean_speed = target_progress / max(arrival_time, 1.0e-9)
        arrival_speed_ratio = arrival_mean_speed / requested_speed_m_s
    else:
        arrival_time = float("nan")
        arrival_mean_speed = float("nan")
        arrival_speed_ratio = float("nan")

    return RouteMetrics(
        target_progress_m=target_progress,
        final_progress_m=final_progress,
        furthest_progress_m=furthest_progress,
        completion_ratio=completion_ratio,
        horizon_mean_speed_m_s=horizon_mean_speed,
        # Retain a horizon-normalized displacement diagnostic, but do not use
        # it as the primary speed metric on routes shorter than speed * time.
        horizon_speed_ratio=horizon_mean_speed / requested_speed_m_s,
        arrived=arrived,
        arrival_time_s=arrival_time,
        arrival_mean_speed_m_s=arrival_mean_speed,
        arrival_speed_ratio=arrival_speed_ratio,
        active_mean_speed_m_s=active_mean_speed,
        active_speed_ratio=active_mean_speed / requested_speed_m_s,
        schedule_mae_m=float(np.mean(np.abs(progress - scheduled))),
        final_schedule_error_m=final_progress - target_progress,
        cross_track_rmse_m=float(np.sqrt(np.mean(np.square(cross_track)))),
        cross_track_p95_m=float(np.percentile(cross_track, 95)),
        terminal_position_error_m=terminal_error,
    )


def compute_first_task_success(
    positions_xy: np.ndarray,
    path_xy: np.ndarray,
    *,
    yaws_rad: np.ndarray,
    heights_m: np.ndarray,
    requested_speed_m_s: float,
    target_progress_m: float,
    target_yaw_rad: float,
    target_height_m: float,
    dt: float,
    start_position_xy: np.ndarray,
    target_heights_m: np.ndarray | None = None,
    profile_height_mae_tolerance_m: float | None = None,
    position_tolerance_m: float = 0.15,
    yaw_tolerance_rad: float = 0.40,
    height_tolerance_m: float = 0.05,
    mean_speed_tolerance_m_s: float = 0.08,
    corridor_half_width_m: float = 0.60,
) -> TaskSuccessMetrics:
    """Reproduce the DPPO terminal predicate on an offline trajectory.

    A prior corridor or overshoot violation is absorbing, as it is during
    training. Position defines the first arrival event. Pose, height and mean
    route speed are evaluated exactly once at that event, so arriving early
    and waiting at the endpoint cannot turn a failed attempt into success. If
    ``target_heights_m`` is supplied, the distance-weighted height-profile MAE
    is accumulated up to first arrival exactly as in DPPO task contract v4.
    """

    positions = np.asarray(positions_xy, dtype=np.float64)
    yaws = np.asarray(yaws_rad, dtype=np.float64).reshape(-1)
    heights = np.asarray(heights_m, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions_xy must have shape (T, 2) with T >= 1.")
    if len(yaws) != len(positions) or len(heights) != len(positions):
        raise ValueError("positions, yaws and heights must have the same length.")
    target_heights = None
    if target_heights_m is not None:
        target_heights = np.asarray(target_heights_m, dtype=np.float64).reshape(-1)
        if len(target_heights) != len(positions):
            raise ValueError("target_heights_m must have the same length as positions.")
        if profile_height_mae_tolerance_m is None:
            raise ValueError(
                "profile_height_mae_tolerance_m is required with target_heights_m."
            )
    if (
        profile_height_mae_tolerance_m is not None
        and profile_height_mae_tolerance_m <= 0.0
    ):
        raise ValueError("profile_height_mae_tolerance_m must be positive.")
    if requested_speed_m_s <= 0.0 or target_progress_m <= 0.0 or dt <= 0.0:
        raise ValueError("speed, target progress and dt must be positive.")
    tolerances = (
        position_tolerance_m,
        yaw_tolerance_rad,
        height_tolerance_m,
        mean_speed_tolerance_m_s,
        corridor_half_width_m,
    )
    if any(value <= 0.0 for value in tolerances):
        raise ValueError("task tolerances must be positive.")

    start = np.asarray(start_position_xy, dtype=np.float64).reshape(-1)
    if start.size != 2:
        raise ValueError("start_position_xy must contain exactly two values.")
    combined_progress, combined_cross_track = project_trajectory_to_polyline(
        np.vstack((start, positions)), path_xy
    )
    progress = np.maximum.accumulate(combined_progress[1:] - float(combined_progress[0]))
    cross_track = combined_cross_track[1:]
    target_point = point_at_progress(path_xy, float(combined_progress[0]) + target_progress_m)
    elapsed = (np.arange(len(positions), dtype=np.float64) + 1.0) * dt
    position_error = np.linalg.norm(positions - target_point[None, :], axis=1)
    yaw_error = np.abs(np.arctan2(
        np.sin(target_yaw_rad - yaws), np.cos(target_yaw_rad - yaws)
    ))
    height_error = np.abs(heights - target_height_m)
    arrival_speed_error = target_progress_m / elapsed - requested_speed_m_s
    # RouteBank uses min(0.12 m, 5% of route length) as its near-terminal gate.
    near_terminal_distance = min(0.12, 0.05 * target_progress_m)
    near_terminal = target_progress_m - progress <= near_terminal_distance
    corridor_violation = np.abs(cross_track) > corridor_half_width_m
    corridor_absorbed = np.maximum.accumulate(corridor_violation)
    terminal_along_error = (
        np.cos(target_yaw_rad) * (positions[:, 0] - target_point[0])
        + np.sin(target_yaw_rad) * (positions[:, 1] - target_point[1])
    )
    overshoot_absorbed = np.maximum.accumulate(near_terminal & (terminal_along_error > 0.50))
    valid_arrival = (
        near_terminal
        & (position_error <= position_tolerance_m)
        & ~corridor_absorbed
        & ~overshoot_absorbed
    )
    candidates = np.flatnonzero(valid_arrival)
    if not candidates.size:
        return TaskSuccessMetrics(
            False,
            False,
            *(float("nan"),) * 7,
            False,
            False,
            False,
            False,
        )
    index = int(candidates[0])
    yaw_ok = bool(yaw_error[index] <= yaw_tolerance_rad)
    terminal_height_ok = bool(height_error[index] <= height_tolerance_m)
    mean_speed_ok = bool(abs(arrival_speed_error[index]) <= mean_speed_tolerance_m_s)
    profile_height_mae = float("nan")
    profile_height_within_tolerance = float("nan")
    profile_height_ok = True
    if target_heights is not None:
        progress_delta = np.diff(np.concatenate(([0.0], progress))).clip(min=0.0)
        distance = progress_delta[: index + 1]
        distance_sum = float(np.sum(distance))
        if distance_sum <= 1.0e-6:
            profile_height_ok = False
        else:
            profile_error = np.abs(
                heights[: index + 1] - target_heights[: index + 1]
            )
            profile_height_mae = float(np.sum(profile_error * distance) / distance_sum)
            profile_height_within_tolerance = float(
                np.sum(
                    (profile_error <= profile_height_mae_tolerance_m).astype(np.float64)
                    * distance
                )
                / distance_sum
            )
            profile_height_ok = bool(
                profile_height_mae <= profile_height_mae_tolerance_m
            )
    success = bool(
        yaw_ok
        and terminal_height_ok
        and mean_speed_ok
        and profile_height_ok
    )
    return TaskSuccessMetrics(
        True,
        success,
        float(elapsed[index]),
        float(position_error[index]),
        float(yaw_error[index]),
        float(height_error[index]),
        float(arrival_speed_error[index]),
        profile_height_mae,
        profile_height_within_tolerance,
        yaw_ok,
        terminal_height_ok,
        mean_speed_ok,
        profile_height_ok,
    )
