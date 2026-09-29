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
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


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


def _interpolate_monotonic(progress: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(progress)
    x = np.asarray(progress, dtype=np.float64)[order]
    y = np.asarray(values, dtype=np.float64)[order]
    unique, indices = np.unique(x, return_index=True)
    if unique.size < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.interp(grid, unique, y[indices], left=np.nan, right=np.nan)


@dataclass(frozen=True)
class TraceBundle:
    trace: dict[str, np.ndarray]
    metadata: dict[str, object]


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
    return TraceBundle(trace, metadata)


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
    aligned: dict[str, list[np.ndarray]] = {"height_error_m": [], "signed_cte_m": [], "tangent_speed_m_s": [], "schedule_debt_m": [], "required_height_m": []}
    event_groups: list[str] = []

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
        schedule_debt = p - requested[index] * time_s
        active_height_error = np.abs(height[:, index] - required[:, index])
        active_values = active & np.isfinite(signed_cte)
        record: dict[str, object] = {
            "scenario": index,
            "path_shape": scenario["path_shape"],
            "repeat": int(scenario["repeat"]),
            "requested_speed_m_s": float(requested[index]),
            "transition_direction": scenario.get("transition_direction"),
            "transition_fraction": scenario.get("transition_fraction"),
            "valid_steps": int(np.count_nonzero(active)),
            "survived": bool(np.all(active)),
            "cte_rmse_m": float(np.sqrt(np.mean(signed_cte[active_values] ** 2))),
            "cte_p95_m": float(np.percentile(np.abs(signed_cte[active_values]), 95)),
            "height_mae_m": float(np.mean(active_height_error[active])),
            "tangent_speed_mean_m_s": float(np.mean(tangent_speed[active, index])),
            "speed_abs_error_mean_m_s": float(np.mean(np.abs(tangent_speed[active, index] - requested[index]))),
            "conditioning_speed_mean_m_s": float(np.mean(conditioning_speed[active, index])),
            "final_schedule_debt_m": float(schedule_debt[np.flatnonzero(active)[-1]]),
            "max_abs_curvature_active_rad_m": float(np.max(np.abs(curvature[active]))),
            **descriptors,
        }
        metrics.append(record)
        for event_index, event_progress in _first_height_events(required[:, index], p, active):
            before, after = required[event_index - 1, index], required[event_index, index]
            offsets = p[active] - event_progress
            values = {
                "height_error_m": height[:, index][active] - required[:, index][active],
                "signed_cte_m": signed_cte[active],
                "tangent_speed_m_s": tangent_speed[:, index][active],
                "schedule_debt_m": schedule_debt[active],
                "required_height_m": required[:, index][active],
            }
            for key, value in values.items():
                aligned[key].append(_interpolate_monotonic(offsets, value, grid))
            direction = "crouch_to_walk" if after > before else "walk_to_crouch"
            event_groups.append(direction)
            post = active & (p >= event_progress + 0.25)
            event_rows.append({
                "scenario": index,
                "path_shape": scenario["path_shape"],
                "repeat": int(scenario["repeat"]),
                "requested_speed_m_s": float(requested[index]),
                "direction": direction,
                "transition_progress_m": event_progress,
                "height_before_m": float(before),
                "height_after_m": float(after),
                "pre_height_mae_m": float(np.mean(active_height_error[active & (p < event_progress - 0.25)])) if np.any(active & (p < event_progress - 0.25)) else np.nan,
                "post_height_mae_m": float(np.mean(active_height_error[post])) if np.any(post) else np.nan,
                "post_cte_rmse_m": float(np.sqrt(np.mean(signed_cte[post] ** 2))) if np.any(post) else np.nan,
            })
    aligned_output = {key: np.stack(value, axis=0) if value else np.empty((0, len(grid))) for key, value in aligned.items()}
    aligned_output["grid_m"] = grid
    aligned_output["event_direction"] = np.asarray(event_groups, dtype="U24")
    return metrics, event_rows, aligned_output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(bundle: TraceBundle, metrics: list[dict[str, object]], aligned: dict[str, np.ndarray], output_dir: Path, *, examples: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = aligned["grid_m"]
    directions = aligned["event_direction"]
    if len(directions):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, facecolor="white")
        specs = (("height_error_m", "Height error [m]"), ("signed_cte_m", "Signed CTE [m]"), ("tangent_speed_m_s", "Tangent speed [m/s]"), ("schedule_debt_m", "Schedule debt [m]"))
        for axis, (key, title) in zip(axes.flat, specs):
            for direction, color in (("walk_to_crouch", "#d95f02"), ("crouch_to_walk", "#1b9e77")):
                values = aligned[key][directions == direction]
                if not len(values):
                    continue
                median = np.nanmedian(values, axis=0)
                low, high = np.nanquantile(values, [0.1, 0.9], axis=0)
                axis.plot(grid, median, color=color, linewidth=2, label=direction.replace("_", " -> "))
                axis.fill_between(grid, low, high, color=color, alpha=0.16)
            axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1)
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)
        axes[1, 0].set_xlabel("Route distance from required height change [m]")
        axes[1, 1].set_xlabel("Route distance from required height change [m]")
        axes[0, 0].legend(fontsize=8)
        figure.suptitle("Height transitions aligned by route progress (median, 10--90%)", fontweight="bold")
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(output_dir / "height_transition_event_aligned.png", dpi=190)
        plt.close(figure)

    if metrics:
        requested = np.asarray([float(row["requested_speed_m_s"]) for row in metrics])
        curvature = np.asarray([float(row["max_abs_curvature_active_rad_m"]) for row in metrics])
        cte = 100.0 * np.asarray([float(row["cte_p95_m"]) for row in metrics])
        speed_error = np.asarray([float(row["speed_abs_error_mean_m_s"]) for row in metrics])
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="white")
        for axis, value, title in zip(axes, (cte, speed_error), ("CTE p95 [cm]", "Mean |tangent speed error| [m/s]")):
            image = axis.scatter(requested, curvature, c=value, cmap="viridis", s=32, alpha=0.85)
            axis.set_xlabel("Requested mean speed [m/s]")
            axis.set_ylabel("Max active curvature [rad/m]")
            axis.set_title(title)
            axis.grid(alpha=0.2)
            figure.colorbar(image, ax=axis)
        figure.suptitle("Capability boundary: speed × realized route curvature", fontweight="bold")
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        figure.savefig(output_dir / "geometry_speed_capability.png", dpi=190)
        plt.close(figure)

        # Representative closed-loop traces make the local allocation visible:
        # colour is achieved tangent speed, while the dashed line remains the
        # immutable geometric reference.  Select diverse route families first
        # rather than silently choosing visually attractive successes.
        by_family: dict[str, dict[str, object]] = {}
        for row in metrics:
            family = str(row["path_shape"])
            previous = by_family.get(family)
            if previous is None or float(row["max_abs_curvature_active_rad_m"]) > float(previous["max_abs_curvature_active_rad_m"]):
                by_family[family] = row
        selected = list(by_family.values())[:max(1, examples)]
        cols = min(4, len(selected))
        rows = int(np.ceil(len(selected) / cols))
        figure, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.7 * rows), facecolor="white", squeeze=False)
        positions = bundle.trace["positions_xy"].astype(np.float64)
        speed = bundle.trace["tangent_speed_m_s"].astype(np.float64)
        valid = bundle.trace["valid"].astype(bool)
        scenarios = bundle.metadata["scenarios"]
        image = None
        for axis, row in zip(axes.flat, selected):
            index = int(row["scenario"])
            scenario = scenarios[index]
            path = np.asarray(scenario["path_xy"], dtype=np.float64)
            mask = valid[:, index]
            axis.plot(path[:, 0], path[:, 1], "--", color="#444444", linewidth=1.5, label="reference")
            image = axis.scatter(positions[mask, index, 0], positions[mask, index, 1], c=speed[mask, index], cmap="plasma", s=7, label="actual")
            axis.set_title(f"{row['path_shape']} | {row['requested_speed_m_s']:.2f} m/s")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_xlabel("world x [m]")
            axis.set_ylabel("world y [m]")
        for axis in axes.flat[len(selected):]:
            axis.set_visible(False)
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), label="Achieved tangent speed [m/s]", fraction=0.025, pad=0.02)
        figure.suptitle("Representative route tracking coloured by local speed", fontweight="bold")
        figure.subplots_adjust(top=0.84, wspace=0.28, hspace=0.34)
        figure.savefig(output_dir / "trajectory_speed_colored.png", dpi=190)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--alignment_before_m", type=float, default=0.8)
    parser.add_argument("--alignment_after_m", type=float, default=1.2)
    parser.add_argument("--alignment_step_m", type=float, default=0.02)
    parser.add_argument("--examples", type=int, default=12)
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
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "scenarios": len(metrics), "height_events": len(events)}, indent=2))


if __name__ == "__main__":
    main()
