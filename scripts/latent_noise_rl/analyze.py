"""Summarize paired latent-pilot traces without importing Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.diffusion_policy.evaluation.analyze_temporal_allocation import (
    _contact_metrics,
    _leg_indices,
    _valid_until_first_arrival,
)


def _unloaded_runs(contact: np.ndarray, dt: float) -> list[float]:
    unloaded = ~np.any(contact, axis=1)
    lengths: list[float] = []
    run = 0
    for value in unloaded:
        if value:
            run += 1
        elif run:
            lengths.append(run * dt)
            run = 0
    if run:
        lengths.append(run * dt)
    return lengths


def _height_m(scenario: dict) -> float:
    return float(scenario.get("path_height_m", 0.2932) or 0.2932)


def _find_scenario(trace, meta, *, shape, speed, height=0.2932, repeat=0):
    for index, scenario in enumerate(meta["scenarios"]):
        if (scenario.get("path_shape") == shape
                and int(scenario.get("repeat", -1)) == repeat
                and abs(float(trace["requested_speed_m_s"][index]) - speed) < 1e-5
                and abs(_height_m(scenario) - height) < 0.01):
            return index
    return None


def analyze_runs(runs: dict[str, Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    traces: dict[str, tuple] = {}
    rows: list[dict] = []
    for label, directory in runs.items():
        trace_path = directory / "timeseries.npz"
        meta_path = directory / "timeseries_metadata.json"
        if not trace_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"{label}: expected timeseries.npz and timeseries_metadata.json in {directory}")
        trace = np.load(trace_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scenarios = meta["scenarios"]
        valid = np.asarray(trace["valid"], dtype=bool)
        positions = np.asarray(trace["positions_xy"], dtype=np.float64)
        progress = np.asarray(trace["progress_m"], dtype=np.float64)
        valid = _valid_until_first_arrival(positions, progress, valid, scenarios)
        times = np.asarray(trace["time_s"], dtype=np.float64)
        speed = np.asarray(trace["tangent_speed_m_s"], dtype=np.float64)
        height = np.asarray(trace["achieved_height_m"], dtype=np.float64)
        required = np.asarray(trace["required_height_m"], dtype=np.float64)
        if "foot_contact" not in trace.files:
            raise ValueError(f"{label}: timeseries lacks foot_contact.")
        contact = np.asarray(trace["foot_contact"], dtype=bool)
        if contact.shape[:2] != valid.shape:
            raise ValueError(f"{label}: contact data dimensions do not match validity trace.")
        names = [str(name) for name in meta.get("feet_body_names", [])]
        legs = _leg_indices(names)
        task_rows: dict[int, dict] = {}
        summary_path = directory / "evaluation_summary.csv"
        if summary_path.is_file():
            with summary_path.open(newline="", encoding="utf-8-sig") as stream:
                task_rows = {int(row["scenario_id"]): row for row in csv.DictReader(stream)
                             if "scenario_id" in row}
        traces[label] = (trace, meta, valid, names, legs)

        dt = float(meta["dt_s"])
        for index, scenario in enumerate(scenarios):
            active = valid[:, index] & (times >= 1.0)
            sample_count = int(active.sum())
            usable_s = sample_count * dt
            selected_speed = speed[active, index]
            selected_height = height[active, index]
            selected_required = required[active, index]
            row = {
                "run": label,
                "scenario_id": scenario.get("index", index),
                "path_shape": scenario.get("path_shape", "unknown"),
                "repeat": scenario.get("repeat", ""),
                "requested_speed_m_s": float(trace["requested_speed_m_s"][index]),
                "required_height_m": float(np.nanmedian(selected_required)) if sample_count else "",
                "valid_gait_samples": sample_count,
                "valid_gait_seconds": usable_s,
                "gait_data_status": "sufficient" if usable_s >= 3.0 else "insufficient",
                "achieved_speed_mean_m_s": float(np.nanmean(selected_speed)) if sample_count else "",
                "height_mae_m": (float(np.nanmean(np.abs(selected_height - selected_required)))
                                 if sample_count else ""),
                "height_p95_p05_span_m": float(np.nanpercentile(selected_height, 95)
                                                - np.nanpercentile(selected_height, 5)
                                                ) if sample_count else "",
                "contact_sensor_order": ",".join(names),
            }
            task = task_rows.get(int(scenario.get("index", index)), {})
            for key in (
                "survived", "base_failure", "task_arrived", "task_success",
                "active_cross_track_rmse_m", "active_cross_track_p95_m",
                "active_height_mae_m", "task_first_arrival_raw_height_error_m",
                "task_success_mean_speed_error_m_s", "route_arrival_mean_speed_m_s",
            ):
                row[key] = task.get(key, "")
            if usable_s >= 3.0:
                contact_metrics = _contact_metrics(contact[active, index], legs)
                contact_metrics["unloaded_fraction"] = contact_metrics.pop("flight_fraction")
                row.update(contact_metrics)
                intervals = _unloaded_runs(contact[active, index], dt)
                row["unloaded_interval_count"] = len(intervals)
                row["unloaded_interval_max_s"] = max(intervals, default=0.0)
            else:
                row.update({
                    "duty_factor_mean": "", "unloaded_fraction": "",
                    "diagonal_contact_correlation": "", "ipsilateral_contact_correlation": "",
                    "unloaded_interval_count": "", "unloaded_interval_max_s": "",
                })
            rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with (output / "latent_gait_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [name for name in ("checkpoint", "initial", "gaussian", "dppo") if name in traces]
    colors = {"checkpoint": "#2878B5", "initial": "#E07A24", "gaussian": "#4A8F62", "dppo": "#6A5AA6"}
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.0), sharex="col", constrained_layout=True)
    for col, requested_speed in enumerate((0.2, 0.5)):
        lane_labels = []
        for mode_index, label in enumerate(modes):
            trace, meta, valid, names, _ = traces[label]
            index = _find_scenario(trace, meta, shape="procedural", speed=requested_speed)
            if index is None:
                continue
            times = np.asarray(trace["time_s"], dtype=np.float64)
            keep = valid[:, index] & (times >= 1.0)
            if not np.any(keep):
                continue
            t = times[keep]
            feet = np.asarray(trace["foot_contact"], dtype=bool)[keep, index]
            base_lane = len(lane_labels)
            for sensor_index, sensor_name in enumerate(names):
                lane = base_lane + sensor_index
                axes[0, col].step(t, feet[:, sensor_index].astype(float) + lane,
                                  where="post", color=colors[label], lw=1.0)
                lane_labels.append(f"{label}: {sensor_name}")
            actual_height = np.asarray(trace["achieved_height_m"])[keep, index]
            required_height = np.asarray(trace["required_height_m"])[keep, index]
            actual_speed = np.asarray(trace["tangent_speed_m_s"])[keep, index]
            axes[1, col].plot(t, actual_height, color=colors[label], lw=1.2, label=label)
            axes[2, col].plot(t, actual_speed, color=colors[label], lw=1.2, label=label)
            if mode_index == 0:
                axes[1, col].plot(t, required_height, color="#222222", ls="--", lw=1.0,
                                  label="required height")
        axes[0, col].set_title(f"Procedural repeat 0 at {requested_speed:.1f} m/s")
        axes[0, col].set_yticks(np.arange(len(lane_labels)) + 0.5)
        axes[0, col].set_yticklabels(lane_labels, fontsize=7)
        axes[0, col].set_ylabel("Foot contacts (1 = loaded)")
        axes[0, col].grid(alpha=0.2)
        axes[1, col].set_ylabel("Base height [m]")
        axes[1, col].grid(alpha=0.2)
        axes[2, col].set_ylabel("Forward speed [m/s]")
        axes[2, col].axhline(requested_speed, color="#222222", ls="--", lw=1.0,
                             label="requested mean")
        axes[2, col].set_xlabel("Active time after first second [s]")
        axes[2, col].grid(alpha=0.2)
    handles, labels_for_legend = axes[1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_for_legend, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=4, frameon=False)
    fig.savefig(output / "contact_height_speed_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    families = ("coherent_smooth", "rounded_waypoint", "hard_waypoint")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    speed_colors = {0.35: "#2878B5", 0.65: "#E07A24"}
    line_styles = {"checkpoint": "-", "initial": "--", "gaussian": ":", "dppo": "-."}
    for ax, shape in zip(axes, families):
        for label in modes:
            trace, meta, valid, _, _ = traces[label]
            for speed_value, color in speed_colors.items():
                index = _find_scenario(trace, meta, shape=shape, speed=speed_value)
                if index is None:
                    continue
                if label == modes[0] and speed_value == 0.35:
                    reference = np.asarray(meta["scenarios"][index]["path_xy"], dtype=np.float64)
                    ax.plot(reference[:, 0], reference[:, 1], color="#333333", ls="--", lw=1.0,
                            label="reference")
                keep = valid[:, index]
                xy = np.asarray(trace["positions_xy"])[keep, index]
                if xy.size:
                    ax.plot(xy[:, 0], xy[:, 1], color=color, ls=line_styles[label], lw=1.25,
                            alpha=0.9, label=f"{label}, {speed_value:.2f} m/s")
        ax.set_title(shape.replace("_", " "))
        ax.set_xlabel("World x [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("World y [m]")
    handles, labels_for_legend = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_for_legend, loc="upper center", bbox_to_anchor=(0.5, 1.04),
                   ncol=4, frameon=False, fontsize=8)
    fig.savefig(output / "paired_geometry_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="Repeat as LABEL=EVALUATION_DIR; use checkpoint, initial, gaussian or dppo.")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for entry in args.run:
        if "=" not in entry:
            raise ValueError("--run must be LABEL=PATH")
        label, path = entry.split("=", 1)
        if label in runs:
            raise ValueError(f"Duplicate run label {label!r}.")
        runs[label] = Path(path).resolve()
    analyze_runs(runs, args.output_dir.resolve())


if __name__ == "__main__":
    main()
