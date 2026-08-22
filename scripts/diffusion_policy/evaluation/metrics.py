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
    active_mean_speed_m_s: float
    active_speed_ratio: float
    schedule_mae_m: float
    final_schedule_error_m: float
    cross_track_rmse_m: float
    cross_track_p95_m: float
    terminal_position_error_m: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


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

    return RouteMetrics(
        target_progress_m=target_progress,
        final_progress_m=final_progress,
        furthest_progress_m=furthest_progress,
        completion_ratio=completion_ratio,
        horizon_mean_speed_m_s=horizon_mean_speed,
        horizon_speed_ratio=horizon_mean_speed / requested_speed_m_s,
        active_mean_speed_m_s=active_mean_speed,
        active_speed_ratio=active_mean_speed / requested_speed_m_s,
        schedule_mae_m=float(np.mean(np.abs(progress - scheduled))),
        final_schedule_error_m=final_progress - target_progress,
        cross_track_rmse_m=float(np.sqrt(np.mean(np.square(cross_track)))),
        cross_track_p95_m=float(np.percentile(cross_track, 95)),
        terminal_position_error_m=terminal_error,
    )
