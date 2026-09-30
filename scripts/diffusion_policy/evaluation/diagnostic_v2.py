"""Post-process path--posture--time evaluation traces without changing a policy.

The v1 evaluator writes raw, vectorized rollouts.  This module turns that
artifact into paper-facing diagnostics while retaining one materialized route
as the unit of analysis.  In particular, height transitions are aligned in
*route distance*, never by an arbitrary percentage of a straight trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def route_descriptors(path_xy: np.ndarray) -> dict[str, float]:
    """Geometry descriptors used to report an empirical capability boundary."""

    points = np.asarray(path_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("path_xy must be (N, 2), N >= 3")
    segments = np.diff(points, axis=0)
    length = np.linalg.norm(segments, axis=1)
    if np.any(length < 1.0e-8):
        raise ValueError("route contains a zero-length segment")
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    turn = np.diff(headings)
    local_distance = 0.5 * (length[:-1] + length[1:])
    curvature = turn / np.maximum(local_distance, 1.0e-8)
    significant = np.abs(turn) > np.deg2rad(8.0)
    signs = np.sign(curvature[np.abs(curvature) > 0.08])
    sign_changes = int(np.count_nonzero(np.diff(signs) != 0)) if signs.size > 1 else 0
    return {
        "route_length_m": float(np.sum(length)),
        "max_abs_curvature_rad_m": float(np.max(np.abs(curvature))) if curvature.size else 0.0,
        "mean_abs_curvature_rad_m": float(np.mean(np.abs(curvature))) if curvature.size else 0.0,
        "total_abs_turn_rad": float(np.sum(np.abs(turn))),
        "max_local_turn_deg": float(np.rad2deg(np.max(np.abs(turn)))) if turn.size else 0.0,
        "turn_sign_changes": sign_changes,
        "endpoint_displacement_m": float(np.linalg.norm(points[-1] - points[0])),
        "significant_turn_count": int(np.count_nonzero(significant)),
    }


def _route_curvature_at_progress(path_xy: np.ndarray, cumulative: np.ndarray, progress: np.ndarray) -> np.ndarray:
    points = np.asarray(path_xy, dtype=np.float64)
    arc = np.asarray(cumulative, dtype=np.float64)
    segments = np.diff(points, axis=0)
    heading = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    if len(heading) < 2:
        return np.zeros_like(progress, dtype=np.float64)
    turn = np.diff(heading)
    ds = 0.5 * (np.diff(arc)[:-1] + np.diff(arc)[1:])
    curvature = turn / np.maximum(ds, 1.0e-8)
    centers = 0.5 * (arc[1:-1] + arc[2:])
    return np.interp(progress, centers, curvature, left=curvature[0], right=curvature[-1])


def _signed_cross_track(positions: np.ndarray, path_xy: np.ndarray, cumulative: np.ndarray) -> np.ndarray:
    """Signed local lateral distance, using nearest path segment per sample."""

    path = np.asarray(path_xy, dtype=np.float64)
    xy = np.asarray(positions, dtype=np.float64)
    segments = np.diff(path, axis=0)
    lengths_sq = np.sum(segments * segments, axis=1)
    output = np.full(len(xy), np.nan, dtype=np.float64)
    for index, point in enumerate(xy):
        fractions = np.sum((point - path[:-1]) * segments, axis=1) / lengths_sq
        fractions = np.clip(fractions, 0.0, 1.0)
        projected = path[:-1] + fractions[:, None] * segments
        distances_sq = np.sum((point - projected) ** 2, axis=1)
        best = int(np.argmin(distances_sq))
        relative = point - projected[best]
        tangent = segments[best] / np.sqrt(lengths_sq[best])
        output[index] = tangent[0] * relative[1] - tangent[1] * relative[0]
    return output


def _first_height_events(required: np.ndarray, progress: np.ndarray, valid: np.ndarray) -> list[tuple[int, float]]:
    """Return all requirement discontinuities as (step, route-progress)."""

    active = np.asarray(valid, dtype=bool)
    heights = np.asarray(required, dtype=np.float64)
    values = np.flatnonzero(active)
    if values.size < 2:
        return []
    changes = values[1:][np.abs(np.diff(heights[values])) > 1.0e-5]
    return [(int(step), float(progress[step])) for step in changes]


def _interpolate_first_passage(progress: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a spatial signal on first passage through route progress.

    Sorting raw progress would mix samples recorded before and after backward
    motion.  The running maximum instead represents the first time the robot
    reaches each route location, which is the relevant quantity for event
    alignment.
    """

    x = np.maximum.accumulate(np.asarray(progress, dtype=np.float64))
    y = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    unique, indices = np.unique(x, return_index=True)
    if unique.size < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.interp(grid, unique, y[indices], left=np.nan, right=np.nan)


def _settling_distance(
    progress: np.ndarray,
    error: np.ndarray,
    *,
    event_progress: float,
    upper_progress: float,
    tolerance_m: float = 0.02,
    hold_distance_m: float = 0.20,
) -> float:
    """Distance after an event until error stays within tolerance for a span."""

    grid = np.arange(event_progress, upper_progress + 1.0e-9, 0.01)
    if grid.size < 2:
        return math.nan
    interpolated = _interpolate_first_passage(progress, np.abs(error), grid)
    hold_samples = max(1, int(np.ceil(hold_distance_m / 0.01)))
    for index in range(max(0, len(grid) - hold_samples + 1)):
        window = interpolated[index:index + hold_samples]
        if len(window) == hold_samples and np.all(np.isfinite(window)) and np.all(window <= tolerance_m):
            return float(grid[index] - event_progress)
    return math.nan


def _anticipation_distance(
    progress: np.ndarray,
    height: np.ndarray,
    *,
    event_progress: float,
    previous_progress: float,
    before_height: float,
    after_height: float,
    lookback_m: float = 0.8,
    response_threshold: float = 0.20,
    hold_distance_m: float = 0.10,
) -> float:
    """Distance of a sustained pre-boundary posture response.

    The response is measured relative to the achieved pre-event plateau, which
    avoids calling ordinary height bias "anticipation".
    """

    plateau = (progress >= previous_progress + 0.10) & (progress <= event_progress - 0.25)
    baseline = float(np.median(height[plateau])) if np.any(plateau) else float(before_height)
    denominator = float(after_height - baseline)
    if abs(denominator) < 1.0e-6:
        return math.nan
    start = max(previous_progress, event_progress - lookback_m)
    grid = np.arange(start, event_progress + 1.0e-9, 0.01)
    response = (height - baseline) / denominator
    interpolated = _interpolate_first_passage(progress, response, grid)
    hold_samples = max(1, int(np.ceil(hold_distance_m / 0.01)))
    for index in range(max(0, len(grid) - hold_samples + 1)):
        window = interpolated[index:index + hold_samples]
        if len(window) == hold_samples and np.all(np.isfinite(window)) and np.all(window >= response_threshold):
            return float(event_progress - grid[index])
    return 0.0


@dataclass(frozen=True)
class TraceBundle:
    trace: dict[str, np.ndarray]
    metadata: dict[str, object]
    outcomes: dict[int, dict[str, str]] = field(default_factory=dict)


def load_trace(trace_dir: Path) -> TraceBundle:
    trace_dir = Path(trace_dir)
    with np.load(trace_dir / "timeseries.npz") as data:
        trace = {key: np.asarray(data[key]) for key in data.files}
    metadata = json.loads((trace_dir / "timeseries_metadata.json").read_text(encoding="utf-8"))
    required = {"time_s", "positions_xy", "progress_m", "tangent_speed_m_s", "achieved_height_m", "required_height_m", "valid", "requested_speed_m_s"}
    missing = sorted(required.difference(trace))
    if missing:
        raise ValueError(f"Trace is missing required arrays: {', '.join(missing)}")
    if trace["positions_xy"].shape[:2] != trace["progress_m"].shape:
        raise ValueError("Trace position/progress shapes are inconsistent")
    if trace["progress_m"].shape[1] != len(metadata.get("scenarios", [])):
        raise ValueError("Trace scenario metadata count does not match trace width")
    outcomes: dict[int, dict[str, str]] = {}
    summary_path = trace_dir / "evaluation_summary.csv"
    if summary_path.is_file():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("scenario_id", "").strip():
                    outcomes[int(row["scenario_id"])] = row
    return TraceBundle(trace, metadata, outcomes)


def analyze_trace(bundle: TraceBundle, *, alignment_window_m: tuple[float, float], alignment_step_m: float) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    trace, metadata = bundle.trace, bundle.metadata
    time_s = trace["time_s"].astype(np.float64)
    positions = trace["positions_xy"].astype(np.float64)
    progress = trace["progress_m"].astype(np.float64)
    tangent_speed = trace["tangent_speed_m_s"].astype(np.float64)
    height = trace["achieved_height_m"].astype(np.float64)
    required = trace["required_height_m"].astype(np.float64)
    valid = trace["valid"].astype(bool)
    requested = trace["requested_speed_m_s"].astype(np.float64)
    conditioning_speed = trace.get("conditioning_speed_m_s", np.broadcast_to(requested, progress.shape)).astype(np.float64)
    grid = np.arange(alignment_window_m[0], alignment_window_m[1] + 0.5 * alignment_step_m, alignment_step_m)
    metrics: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    aligned: dict[str, list[np.ndarray]] = {
        "height_error_m": [],
        "height_response_fraction": [],
        "signed_cte_m": [],
        "tangent_speed_m_s": [],
        "schedule_debt_m": [],
        "required_height_m": [],
    }
    event_groups: list[str] = []
    event_speeds: list[float] = []
    event_families: list[str] = []

    for index, scenario in enumerate(metadata["scenarios"]):
        active = valid[:, index]
        if not np.any(active):
            continue
        path = np.asarray(scenario["path_xy"], dtype=np.float64)
        arc = np.asarray(scenario["path_cumulative_m"], dtype=np.float64)
        descriptors = route_descriptors(path)
        p = progress[:, index]
        signed_cte = _signed_cross_track(positions[:, index], path, arc)
        curvature = _route_curvature_at_progress(path, arc, p)
        scheduled_progress = np.minimum(requested[index] * time_s, descriptors["route_length_m"])
        # Positive debt means that the robot is behind the requested schedule.
        schedule_debt = scheduled_progress - p
        active_height_error = np.abs(height[:, index] - required[:, index])
        active_values = active & np.isfinite(signed_cte)
        outcome = bundle.outcomes.get(index)
        height_profile = scenario.get("height_profile")
        if height_profile is None and outcome is not None:
            height_profile = outcome.get("height_profile")
        events = _first_height_events(required[:, index], p, active)
        record: dict[str, object] = {
            "scenario": index,
            "path_shape": scenario["path_shape"],
            "repeat": int(scenario["repeat"]),
            "requested_speed_m_s": float(requested[index]),
            "transition_direction": scenario.get("transition_direction"),
            "transition_fraction": scenario.get("transition_fraction"),
            "height_profile": height_profile or "unknown",
            "height_change_count": len(events),
            "valid_steps": int(np.count_nonzero(active)),
            "survived": bool(np.all(active)),
            "cte_rmse_m": float(np.sqrt(np.mean(signed_cte[active_values] ** 2))),
            "cte_p95_m": float(np.percentile(np.abs(signed_cte[active_values]), 95)),
            "height_mae_m": float(np.mean(active_height_error[active])),
            "required_height_mean_m": float(np.mean(required[active, index])),
            "achieved_height_mean_m": float(np.mean(height[active, index])),
            "height_bias_m": float(np.mean(height[active, index] - required[active, index])),
            "tangent_speed_mean_m_s": float(np.mean(tangent_speed[active, index])),
            "speed_abs_error_mean_m_s": float(np.mean(np.abs(tangent_speed[active, index] - requested[index]))),
            "conditioning_speed_mean_m_s": float(np.mean(conditioning_speed[active, index])),
            "final_schedule_debt_m": float(schedule_debt[np.flatnonzero(active)[-1]]),
            "max_abs_curvature_active_rad_m": float(np.max(np.abs(curvature[active]))),
            **descriptors,
        }
        if outcome is not None:
            for key in ("task_success", "route_arrived", "base_failure", "physical_sanity_failure"):
                value = outcome.get(key, "").strip().lower()
                record[key] = value == "true" if value in {"true", "false"} else ""
        metrics.append(record)
        for event_number, (event_index, event_progress) in enumerate(events):
            before, after = required[event_index - 1, index], required[event_index, index]
            offsets = p[active] - event_progress
            delta = float(after - before)
            values = {
                "height_error_m": height[:, index][active] - required[:, index][active],
                "height_response_fraction": (height[:, index][active] - before) / delta,
                "signed_cte_m": signed_cte[active],
                "tangent_speed_m_s": tangent_speed[:, index][active],
                "schedule_debt_m": schedule_debt[active],
                "required_height_m": required[:, index][active],
            }
            for key, value in values.items():
                aligned[key].append(_interpolate_first_passage(offsets, value, grid))
            direction = "crouch_to_walk" if after > before else "walk_to_crouch"
            event_groups.append(direction)
            event_speeds.append(float(requested[index]))
            event_families.append(str(scenario["path_shape"]))
            previous_progress = events[event_number - 1][1] if event_number > 0 else 0.0
            next_progress = events[event_number + 1][1] if event_number + 1 < len(events) else descriptors["route_length_m"]
            pre = active & (p >= previous_progress + 0.25) & (p <= event_progress - 0.25)
            post = active & (p >= event_progress + 0.25) & (p <= next_progress - 0.25)
            local_post = active & (p >= event_progress) & (p <= next_progress)
            direction_sign = float(np.sign(delta))
            overshoot = (
                float(max(0.0, np.max((height[local_post, index] - after) * direction_sign)))
                if np.any(local_post) else math.nan
            )
            event_rows.append({
                "scenario": index,
                "path_shape": scenario["path_shape"],
                "repeat": int(scenario["repeat"]),
                "requested_speed_m_s": float(requested[index]),
                "direction": direction,
                "transition_progress_m": event_progress,
                "height_before_m": float(before),
                "height_after_m": float(after),
                "pre_height_mae_m": float(np.mean(active_height_error[pre])) if np.any(pre) else np.nan,
                "post_height_mae_m": float(np.mean(active_height_error[post])) if np.any(post) else np.nan,
                "post_cte_rmse_m": float(np.sqrt(np.mean(signed_cte[post] ** 2))) if np.any(post) else np.nan,
                "anticipation_distance_m": _anticipation_distance(
                    p[active],
                    height[active, index],
                    event_progress=event_progress,
                    previous_progress=previous_progress,
                    before_height=float(before),
                    after_height=float(after),
                ),
                "overshoot_m": overshoot,
                "settling_distance_m": _settling_distance(
                    p[active],
                    height[active, index] - after,
                    event_progress=event_progress,
                    upper_progress=next_progress,
                ),
            })
    aligned_output = {key: np.stack(value, axis=0) if value else np.empty((0, len(grid))) for key, value in aligned.items()}
    aligned_output["grid_m"] = grid
    aligned_output["event_direction"] = np.asarray(event_groups, dtype="U24")
    aligned_output["event_requested_speed_m_s"] = np.asarray(event_speeds, dtype=np.float64)
    aligned_output["event_path_shape"] = np.asarray(event_families, dtype="U32")
    return metrics, event_rows, aligned_output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nan_band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise median and descriptive 10--90% route band."""

    with np.errstate(all="ignore"):
        return (
            np.nanmedian(values, axis=0),
            np.nanquantile(values, 0.1, axis=0),
            np.nanquantile(values, 0.9, axis=0),
        )


def _family_difficulty(row: dict[str, object]) -> tuple[float, str]:
    """Return a geometry-appropriate difficulty coordinate."""

    if str(row["path_shape"]) in {"sweep_hard_turn", "hard_waypoint", "ood_corner"}:
        return float(row["max_local_turn_deg"]), "Corner angle [deg]"
    return float(row["max_abs_curvature_rad_m"]), "Peak curvature [rad/m]"


def plot_diagnostics(bundle: TraceBundle, metrics: list[dict[str, object]], aligned: dict[str, np.ndarray], output_dir: Path, *, examples: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = aligned["grid_m"]
    directions = aligned["event_direction"]
    if len(directions):
        event_speeds = aligned["event_requested_speed_m_s"]
        speed_levels = np.unique(np.round(event_speeds, 6))
        speed_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, max(1, len(speed_levels))))
        direction_order = ("walk_to_crouch", "crouch_to_walk")

        figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True, facecolor="white")
        for column, direction in enumerate(direction_order):
            axes[0, column].set_title(direction.replace("_to_", " $\\rightarrow$ ").replace("_", " "))
            for speed_value, color in zip(speed_levels, speed_colors):
                mask = (directions == direction) & np.isclose(event_speeds, speed_value)
                if not np.any(mask):
                    continue
                response_median, response_low, response_high = _nan_band(aligned["height_response_fraction"][mask])
                cte_median, cte_low, cte_high = _nan_band(np.abs(aligned["signed_cte_m"][mask]) * 100.0)
                label = f"{speed_value:.2f} m/s"
                axes[0, column].plot(grid, response_median, color=color, linewidth=1.8, label=label)
                axes[0, column].fill_between(grid, response_low, response_high, color=color, alpha=0.13)
                axes[1, column].plot(grid, cte_median, color=color, linewidth=1.8, label=label)
                axes[1, column].fill_between(grid, cte_low, cte_high, color=color, alpha=0.13)
            axes[0, column].plot(grid, (grid >= 0.0).astype(float), "--", color="#202020", linewidth=1.2, label="required")
            for row in range(2):
                axes[row, column].axvline(0.0, color="#555555", linestyle=":", linewidth=1)
                axes[row, column].grid(alpha=0.22)
            axes[1, column].set_xlabel("Route distance from posture change [m]")
        axes[0, 0].set_ylabel("Normalized posture response")
        axes[1, 0].set_ylabel("Absolute CTE [cm]")
        axes[0, 0].legend(fontsize=8, ncol=2)
        figure.suptitle("Posture transitions aligned by route distance\nmedian with descriptive 10--90% route band", fontweight="bold")
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(output_dir / "height_transition_event_aligned.png", dpi=190)
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True, facecolor="white")
        for column, direction in enumerate(direction_order):
            axes[0, column].set_title(direction.replace("_to_", " $\\rightarrow$ ").replace("_", " "))
            for speed_value, color in zip(speed_levels, speed_colors):
                mask = (directions == direction) & np.isclose(event_speeds, speed_value)
                if not np.any(mask):
                    continue
                local_speed, speed_low, speed_high = _nan_band(aligned["tangent_speed_m_s"][mask])
                debt, debt_low, debt_high = _nan_band(aligned["schedule_debt_m"][mask])
                label = f"{speed_value:.2f} m/s"
                axes[0, column].plot(grid, local_speed, color=color, linewidth=1.8, label=label)
                axes[0, column].fill_between(grid, speed_low, speed_high, color=color, alpha=0.13)
                axes[0, column].axhline(speed_value, color=color, linestyle="--", linewidth=0.8, alpha=0.75)
                axes[1, column].plot(grid, debt, color=color, linewidth=1.8, label=label)
                axes[1, column].fill_between(grid, debt_low, debt_high, color=color, alpha=0.13)
            for row in range(2):
                axes[row, column].axvline(0.0, color="#555555", linestyle=":", linewidth=1)
                axes[row, column].grid(alpha=0.22)
            axes[1, column].axhline(0.0, color="#202020", linestyle="--", linewidth=1)
            axes[1, column].set_xlabel("Route distance from posture change [m]")
        axes[0, 0].set_ylabel("Tangent speed [m/s]")
        axes[1, 0].set_ylabel("Uniform-reference progress error [m] (+ behind)")
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(
            "Local temporal allocation around posture transitions\n"
            "dashed speed = episode-average target, not a local setpoint",
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(output_dir / "temporal_allocation_event_aligned.png", dpi=190)
        plt.close(figure)

    if metrics:
        constant = [
            row
            for row in metrics
            if str(row.get("height_profile")) == "constant" and int(row["height_change_count"]) == 0
        ]
        requested_heights = sorted({round(float(row["required_height_mean_m"]), 4) for row in constant})
        if len(requested_heights) >= 3:
            figure, axis = plt.subplots(figsize=(6.0, 5.0), facecolor="white")
            for row in constant:
                axis.scatter(float(row["required_height_mean_m"]), float(row["achieved_height_mean_m"]), color="#4c78a8", s=9, alpha=0.10)
            medians, lows, highs = [], [], []
            for target in requested_heights:
                achieved = np.asarray([float(row["achieved_height_mean_m"]) for row in constant if np.isclose(float(row["required_height_mean_m"]), target, atol=5.0e-4)])
                medians.append(float(np.median(achieved)))
                lows.append(float(np.quantile(achieved, 0.1)))
                highs.append(float(np.quantile(achieved, 0.9)))
            axis.errorbar(requested_heights, medians, yerr=[np.asarray(medians) - lows, np.asarray(highs) - medians], fmt="o-", color="#1f4e79", capsize=3, label="median, 10--90% route band")
            limits = (min(requested_heights) - 0.01, max(requested_heights) + 0.01)
            axis.plot(limits, limits, "--", color="#333333", linewidth=1.2, label="ideal")
            axis.set(xlim=limits, ylim=limits, xlabel="Required base height [m]", ylabel="Achieved mean base height [m]")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8)
            axis.set_title("Intermediate-height calibration")
            figure.tight_layout()
            figure.savefig(output_dir / "intermediate_height_calibration.png", dpi=190)
            plt.close(figure)

        families = list(dict.fromkeys(str(row["path_shape"]) for row in metrics))
        figure, axes = plt.subplots(
            2,
            len(families),
            figsize=(max(5.5, 3.6 * len(families)), 7.0),
            facecolor="white",
            squeeze=False,
            constrained_layout=True,
        )
        all_speeds = np.asarray([float(row["requested_speed_m_s"]) for row in metrics])
        normalization = matplotlib.colors.Normalize(vmin=float(np.min(all_speeds)), vmax=float(np.max(all_speeds)))
        image = None
        for column, family in enumerate(families):
            rows_family = [row for row in metrics if str(row["path_shape"]) == family]
            difficulty_and_label = [_family_difficulty(row) for row in rows_family]
            difficulty = np.asarray([item[0] for item in difficulty_and_label])
            x_label = difficulty_and_label[0][1]
            speed_values = np.asarray([float(row["requested_speed_m_s"]) for row in rows_family])
            plot_values = (
                100.0 * np.asarray([float(row["cte_p95_m"]) for row in rows_family]),
                np.asarray([float(row["speed_abs_error_mean_m_s"]) for row in rows_family]),
            )
            for row_index, (value, label) in enumerate(zip(plot_values, ("CTE p95 [cm]", "Mean |speed error| [m/s]"))):
                image = axes[row_index, column].scatter(difficulty, value, c=speed_values, cmap="viridis", norm=normalization, s=12, alpha=0.32)
                axes[row_index, column].set_xlabel(x_label)
                axes[row_index, column].set_ylabel(label if column == 0 else "")
                axes[row_index, column].grid(alpha=0.18)
            axes[0, column].set_title(family.replace("sweep_", "").replace("_", " "))
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), label="Requested average speed [m/s]", fraction=0.018, pad=0.02)
        figure.suptitle("Capability by route family", fontweight="bold")
        figure.savefig(output_dir / "geometry_speed_capability.png", dpi=190)
        plt.close(figure)

        # Representative closed-loop traces make the local allocation visible:
        # colour is achieved tangent speed, while the dashed line remains the
        # immutable geometric reference.  The route closest to the median
        # family difficulty is selected by rule, not visual attractiveness.
        by_family: dict[str, dict[str, object]] = {}
        target_speed = 0.85
        for family in families:
            family_rows = [row for row in metrics if str(row["path_shape"]) == family]
            speed_distance = min(abs(float(row["requested_speed_m_s"]) - target_speed) for row in family_rows)
            candidates = [row for row in family_rows if np.isclose(abs(float(row["requested_speed_m_s"]) - target_speed), speed_distance)]
            route_difficulty = {int(row["repeat"]): _family_difficulty(row)[0] for row in candidates}
            median_difficulty = float(np.median(list(route_difficulty.values())))
            chosen_repeat = min(route_difficulty, key=lambda repeat: abs(route_difficulty[repeat] - median_difficulty))
            by_family[family] = next(row for row in candidates if int(row["repeat"]) == chosen_repeat)
        selected = list(by_family.values())[:max(1, examples)]
        cols = min(4, len(selected))
        rows = int(np.ceil(len(selected) / cols))
        figure, axes = plt.subplots(
            rows,
            cols,
            figsize=(max(5.5, 4.0 * cols), 3.8 * rows),
            facecolor="white",
            squeeze=False,
            constrained_layout=True,
        )
        positions = bundle.trace["positions_xy"].astype(np.float64)
        speed = bundle.trace["tangent_speed_m_s"].astype(np.float64)
        valid = bundle.trace["valid"].astype(bool)
        scenarios = bundle.metadata["scenarios"]
        selected_indices = [int(row["scenario"]) for row in selected]
        selected_speed_max = max(float(np.nanmax(speed[valid[:, index], index])) for index in selected_indices)
        speed_normalization = matplotlib.colors.Normalize(vmin=0.0, vmax=max(0.1, selected_speed_max))
        image = None
        for axis, row in zip(axes.flat, selected):
            index = int(row["scenario"])
            scenario = scenarios[index]
            path = np.asarray(scenario["path_xy"], dtype=np.float64)
            mask = valid[:, index]
            axis.plot(path[:, 0], path[:, 1], "--", color="#444444", linewidth=1.5, label="reference")
            image = axis.scatter(positions[mask, index, 0], positions[mask, index, 1], c=speed[mask, index], cmap="plasma", norm=speed_normalization, s=7, label="actual")
            axis.set_title(f"{row['path_shape']}\nmedian difficulty, target {row['requested_speed_m_s']:.2f} m/s")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_xlabel("world x [m]")
            axis.set_ylabel("world y [m]")
        for axis in axes.flat[len(selected):]:
            axis.set_visible(False)
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), label="Achieved tangent speed [m/s]", fraction=0.025, pad=0.02)
        figure.suptitle("Route tracking and local speed (predeclared examples)", fontweight="bold")
        figure.savefig(output_dir / "trajectory_speed_colored.png", dpi=190)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--alignment_before_m", type=float, default=0.8)
    parser.add_argument("--alignment_after_m", type=float, default=1.2)
    parser.add_argument("--alignment_step_m", type=float, default=0.02)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    if args.alignment_before_m <= 0 or args.alignment_after_m <= 0 or args.alignment_step_m <= 0:
        raise ValueError("alignment distances must be positive")
    output_dir = (args.output_dir or args.trace_dir / "diagnostic_v2").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_trace(args.trace_dir)
    metrics, events, aligned = analyze_trace(
        bundle,
        alignment_window_m=(-args.alignment_before_m, args.alignment_after_m),
        alignment_step_m=args.alignment_step_m,
    )
    _write_csv(output_dir / "route_diagnostics.csv", metrics)
    _write_csv(output_dir / "height_transition_events.csv", events)
    np.savez_compressed(output_dir / "height_transition_aligned.npz", **aligned)
    plot_diagnostics(bundle, metrics, aligned, output_dir, examples=args.examples)
    (output_dir / "diagnostic_manifest.json").write_text(json.dumps({
        "source_trace_dir": str(args.trace_dir.resolve()),
        "scenarios": len(metrics),
        "height_events": len(events),
        "alignment_window_m": [-args.alignment_before_m, args.alignment_after_m],
        "statistical_unit": "one materialized route; repeated speed/height conditions are not independent geometries",
        "route_band": "descriptive 10--90% band across route-condition events; not a confidence interval",
        "schedule_debt_definition": "min(requested_speed*time, route_length) - route_progress; positive means behind a uniform-progress diagnostic reference, not a required local schedule",
        "transition_metric_definitions": {
            "anticipation_distance_m": "distance before boundary at first sustained 20% response over 0.10 m, relative to the achieved pre-event plateau and within a 0.8 m lookback",
            "overshoot_m": "maximum height beyond the new target in the direction of the transition",
            "settling_distance_m": "first post-boundary distance within 0.02 m for a sustained 0.20 m route span",
        },
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "scenarios": len(metrics), "height_events": len(events)}, indent=2))


if __name__ == "__main__":
    main()
