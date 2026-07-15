"""Write reproducible tables and figures for command-height evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


STATIC_COLUMNS = (
    "scenario_id",
    "repeat",
    "requested_vx",
    "requested_vy",
    "requested_wz",
    "requested_height",
    "samples",
    "survived",
    "time_to_failure_s",
    "achieved_vx_mean",
    "achieved_vy_mean",
    "achieved_wz_mean",
    "achieved_height_mean",
    "achieved_height_std",
    "vx_rmse",
    "vy_rmse",
    "wz_rmse",
    "height_rmse",
    "tilt_rms_deg",
    "tilt_max_deg",
    "action_delta_rms",
)


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = tuple(rows[0]) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_static_height(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Plot the requested/achieved height relation, aggregated over repeats."""

    plt = _mpl()
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    commands = sorted(
        {(row["requested_vx"], row["requested_vy"], row["requested_wz"]) for row in rows}
    )
    all_heights = [float(row["requested_height"]) for row in rows]
    if all_heights:
        low, high = min(all_heights), max(all_heights)
        margin = max(0.005, 0.05 * (high - low))
        axis.plot([low - margin, high + margin], [low - margin, high + margin], "k--", label="ideal")

    for command in commands:
        subset = [
            row
            for row in rows
            if (row["requested_vx"], row["requested_vy"], row["requested_wz"]) == command
        ]
        heights = sorted({float(row["requested_height"]) for row in subset})
        means = []
        deviations = []
        for height in heights:
            values = np.asarray(
                [row["achieved_height_mean"] for row in subset if row["requested_height"] == height],
                dtype=np.float64,
            )
            means.append(float(np.nanmean(values)))
            deviations.append(float(np.nanstd(values)))
        axis.errorbar(heights, means, yerr=deviations, marker="o", capsize=3, label=str(command))

    axis.set_xlabel("Requested base height [m]")
    axis.set_ylabel("Achieved base height [m]")
    axis.set_title("Emergent interpolation of desired base height")
    axis.grid(True, alpha=0.3)
    axis.legend(title="(vx, vy, wz)", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_static_velocity(rows: list[dict[str, Any]], path: str | Path) -> None:
    plt = _mpl()
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    fields = (("vx", "Linear x"), ("vy", "Linear y"), ("wz", "Yaw rate"))
    heights = sorted({float(row["requested_height"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(heights))))
    for axis, (field, title) in zip(axes, fields):
        requested_key = f"requested_{field}"
        achieved_key = f"achieved_{field}_mean"
        all_values = []
        for color, height in zip(colors, heights):
            subset = [row for row in rows if float(row["requested_height"]) == height]
            requested = np.asarray([row[requested_key] for row in subset], dtype=np.float64)
            achieved = np.asarray([row[achieved_key] for row in subset], dtype=np.float64)
            axis.scatter(requested, achieved, color=color, label=f"h={height:.4f}", alpha=0.8)
            all_values.extend(requested.tolist())
            all_values.extend(achieved[np.isfinite(achieved)].tolist())
        if all_values:
            low, high = min(all_values), max(all_values)
            margin = max(0.02, 0.08 * (high - low + 1.0e-6))
            axis.plot([low - margin, high + margin], [low - margin, high + margin], "k--")
        axis.set_title(title)
        axis.set_xlabel("Requested")
        axis.set_ylabel("Achieved")
        axis.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_dynamic_height(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    plt = _mpl()
    time_s = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    requested = np.asarray([row["requested_height"] for row in rows], dtype=np.float64)
    achieved = np.asarray([row["achieved_height"] for row in rows], dtype=np.float64)
    requested_vx = np.asarray([row["requested_vx"] for row in rows], dtype=np.float64)
    vx = np.asarray([row["achieved_vx"] for row in rows], dtype=np.float64)
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    axes[0].plot(time_s, requested, label="requested", linewidth=2)
    axes[0].plot(time_s, achieved, label="achieved", linewidth=1.5)
    axes[0].set_ylabel("Base height [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(time_s, requested_vx, color="tab:blue", linewidth=2, label="requested vx")
    axes[1].plot(time_s, vx, color="tab:green", linewidth=1.5, label="achieved vx")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Forward velocity vx [m/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.suptitle("Closed-loop response to unseen height commands")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_latency(latencies_ms: list[float], deadline_ms: float, path: str | Path) -> None:
    if not latencies_ms:
        return
    plt = _mpl()
    values = np.asarray(latencies_ms, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.hist(values, bins=min(40, max(10, int(np.sqrt(len(values))))), alpha=0.8)
    axis.axvline(deadline_ms, color="tab:red", linestyle="--", label=f"deadline {deadline_ms:.1f} ms")
    for percentile, color in ((50, "tab:green"), (95, "tab:orange"), (99, "tab:purple")):
        value = float(np.percentile(values, percentile))
        axis.axvline(value, color=color, alpha=0.8, label=f"p{percentile}={value:.2f} ms")
    axis.set_xlabel("One-environment diffusion inference [ms]")
    axis.set_ylabel("Count")
    axis.set_title("Deployment latency (batch size 1)")
    axis.grid(True, alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
