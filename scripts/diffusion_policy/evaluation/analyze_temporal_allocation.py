#!/usr/bin/env python3
"""Analyze whether a policy allocates speed locally while preserving route time.

This is deliberately an evaluation-side tool.  It never changes a checkpoint or
the training reward.  It consumes the opt-in ``timeseries.npz`` produced by
``evaluate_policy.py --save_timeseries`` and reports local speed, schedule
slack, and height-conditioned behavior by route-progress segment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _finite_mean(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--segment_m", type=float, default=0.8)
    args = parser.parse_args()
    if args.segment_m <= 0.0:
        raise ValueError("--segment_m must be positive")

    trace_path = args.trace_dir / "timeseries.npz"
    metadata_path = args.trace_dir / "timeseries_metadata.json"
    if not trace_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expected {trace_path.name} and {metadata_path.name} in {args.trace_dir}"
        )
    trace = np.load(trace_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    progress = np.asarray(trace["progress_m"], dtype=np.float64)
    speed = np.asarray(trace["tangent_speed_m_s"], dtype=np.float64)
    required_height = np.asarray(trace["required_height_m"], dtype=np.float64)
    valid = np.asarray(trace["valid"], dtype=bool)
    requested = np.asarray(trace["requested_speed_m_s"], dtype=np.float64)
    if progress.shape != speed.shape or progress.shape != valid.shape:
        raise ValueError("Trace arrays have incompatible shapes")

    schedule_error = progress - time_s[:, None] * requested[None, :]
    rows: list[dict[str, object]] = []
    scenario_summaries: list[dict[str, object]] = []
    for i, scenario in enumerate(metadata["scenarios"]):
        max_progress = float(np.max(progress[:, i][valid[:, i]])) if np.any(valid[:, i]) else 0.0
        max_segment = int(np.floor(max_progress / args.segment_m))
        for segment in range(max_segment + 1):
            mask = valid[:, i] & (progress[:, i] >= segment * args.segment_m) & (
                progress[:, i] < (segment + 1) * args.segment_m
            )
            if not np.any(mask):
                continue
            rows.append(
                {
                    "scenario": i,
                    "path_shape": scenario["path_shape"],
                    "repeat": scenario["repeat"],
                    "requested_speed_m_s": float(requested[i]),
                    "progress_start_m": segment * args.segment_m,
                    "progress_end_m": (segment + 1) * args.segment_m,
                    "required_height_m": _finite_mean(required_height[mask, i]),
                    "tangent_speed_mean_m_s": _finite_mean(speed[mask, i]),
                    "speed_error_mean_m_s": _finite_mean(speed[mask, i] - requested[i]),
                    "schedule_error_mean_m": _finite_mean(schedule_error[mask, i]),
                    "samples": int(np.count_nonzero(mask)),
                }
            )

        scenario_valid = valid[:, i]
        scenario_summaries.append(
            {
                "scenario": i,
                "path_shape": scenario["path_shape"],
                "repeat": scenario["repeat"],
                "requested_speed_m_s": float(requested[i]),
                "max_progress_m": max_progress,
                "mean_tangent_speed_m_s": _finite_mean(speed[scenario_valid, i]),
                "final_schedule_error_m": (
                    float(schedule_error[scenario_valid, i][-1]) if np.any(scenario_valid) else None
                ),
                "height_speed": {
                    str(round(float(h), 4)): _finite_mean(
                        speed[scenario_valid & np.isclose(required_height[:, i], h), i]
                    )
                    for h in np.unique(required_height[scenario_valid, i])
                },
            }
        )

    with (args.trace_dir / "temporal_allocation_segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["scenario"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "trace_dir": str(args.trace_dir),
        "segment_m": args.segment_m,
        "scenarios": scenario_summaries,
        "segments": rows,
        "interpretation_note": (
            "Positive speed_error in one segment followed by negative speed_error in a later segment "
            "is evidence of local speed allocation; schedule_error alone is not interpreted after route completion."
        ),
    }
    (args.trace_dir / "temporal_allocation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Keep plotting optional at import time but always produce one compact plot
    # for the targeted experiment; no plotting is needed by the evaluator.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for i, scenario in enumerate(metadata["scenarios"]):
        mask = valid[:, i]
        label = f"{scenario['path_shape']} r{scenario['repeat']} ({requested[i]:.1f} m/s)"
        axes[0].plot(progress[:, i][mask], speed[:, i][mask], alpha=0.75, linewidth=1.0, label=label)
        axes[1].plot(progress[:, i][mask], schedule_error[mask, i], alpha=0.75, linewidth=1.0, label=label)
    axes[0].axhline(float(np.mean(requested)), color="black", linestyle="--", linewidth=1.0, label="requested")
    axes[0].set_ylabel("Tangent speed [m/s]")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Schedule error [m]")
    axes[1].set_xlabel("Route progress [m]")
    axes[0].set_title("Local speed allocation and temporal slack")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.trace_dir / "temporal_allocation.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"scenarios": len(scenario_summaries), "segments": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
