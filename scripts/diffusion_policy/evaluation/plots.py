"""Audit-specific plots kept separate from the Isaac rollout loop."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _finite(values: list[Any]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    return output[np.isfinite(output)]


def plot_route_outcomes(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot the four metrics that determine finite-horizon route success."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_specs = (
        ("route_completion_ratio", "Completion ratio", 1.0),
        ("route_horizon_speed_ratio", "Horizon mean-speed ratio", 1.0),
        ("route_terminal_position_error_m", "Terminal position error [m]", None),
        ("route_cross_track_rmse_m", "Cross-track RMSE [m]", None),
    )
    shapes = sorted({str(row["path_shape"]) for row in summary_rows})
    speeds = sorted({float(row["requested_speed"]) for row in summary_rows})
    palette = ("#0052CC", "#FF5A5F", "#00A86B", "#FFB300", "#7B1FA2")
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")

    for axis, (key, label, ideal) in zip(axes.flat, metric_specs):
        for shape_index, shape in enumerate(shapes):
            means = []
            standard_deviations = []
            for speed in speeds:
                values = _finite([
                    row.get(key, math.nan)
                    for row in summary_rows
                    if row["path_shape"] == shape and float(row["requested_speed"]) == speed
                ])
                means.append(float(np.mean(values)) if values.size else np.nan)
                standard_deviations.append(float(np.std(values)) if values.size else np.nan)
            means_array = np.asarray(means)
            std_array = np.asarray(standard_deviations)
            color = palette[shape_index % len(palette)]
            axis.plot(speeds, means_array, "o-", color=color, label=shape.replace("_", " ").title())
            valid = np.isfinite(means_array) & np.isfinite(std_array)
            if np.any(valid):
                speed_array = np.asarray(speeds)
                axis.fill_between(
                    speed_array[valid],
                    (means_array - std_array)[valid],
                    (means_array + std_array)[valid],
                    color=color,
                    alpha=0.15,
                )
        if ideal is not None:
            axis.axhline(ideal, color="#555555", linestyle="--", linewidth=1.2, label="Ideal")
        axis.set_xlabel("Requested speed [m/s]")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)
    figure.suptitle("Finite-horizon route outcomes", fontweight="bold", y=0.985)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
