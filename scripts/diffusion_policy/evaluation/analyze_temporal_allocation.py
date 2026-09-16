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


def _binary_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 3 or np.std(first) < 1.0e-8 or np.std(second) < 1.0e-8:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _leg_indices(names: list[str]) -> dict[str, int]:
    aliases = {
        "fl": ("fl_", "front_left", "left_front"),
        "fr": ("fr_", "front_right", "right_front"),
        "hl": ("hl_", "rl_", "hind_left", "left_hind", "rear_left", "left_rear"),
        "hr": ("hr_", "rr_", "hind_right", "right_hind", "rear_right", "right_rear"),
    }
    lowered = [name.lower() for name in names]
    result: dict[str, int] = {}
    for leg, candidates in aliases.items():
        matches = [index for index, name in enumerate(lowered) if any(token in name for token in candidates)]
        if len(matches) == 1:
            result[leg] = matches[0]
    return result


def _contact_metrics(contact: np.ndarray, leg_indices: dict[str, int]) -> dict[str, object]:
    if contact.size == 0:
        return {}
    values: dict[str, object] = {
        "duty_factor_mean": float(np.mean(contact)),
        "flight_fraction": float(np.mean(~np.any(contact, axis=1))),
    }
    for leg, index in leg_indices.items():
        values[f"duty_factor_{leg}"] = float(np.mean(contact[:, index]))
    if all(leg in leg_indices for leg in ("fl", "fr", "hl", "hr")):
        fl, fr, hl, hr = (leg_indices[leg] for leg in ("fl", "fr", "hl", "hr"))
        diagonal = [
            _binary_correlation(contact[:, fl], contact[:, hr]),
            _binary_correlation(contact[:, fr], contact[:, hl]),
        ]
        ipsilateral = [
            _binary_correlation(contact[:, fl], contact[:, hl]),
            _binary_correlation(contact[:, fr], contact[:, hr]),
        ]
        diagonal_finite = [value for value in diagonal if value is not None]
        ipsilateral_finite = [value for value in ipsilateral if value is not None]
        values["diagonal_contact_correlation"] = (
            float(np.mean(diagonal_finite)) if diagonal_finite else None
        )
        values["ipsilateral_contact_correlation"] = (
            float(np.mean(ipsilateral_finite)) if ipsilateral_finite else None
        )
    return values


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
    foot_contact = (
        np.asarray(trace["foot_contact"], dtype=bool)
        if "foot_contact" in trace.files
        else None
    )
    feet_body_names = [str(name) for name in metadata.get("feet_body_names", [])]
    leg_indices = _leg_indices(feet_body_names)
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
            row = {
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
            if foot_contact is not None:
                row.update(_contact_metrics(foot_contact[mask, i], leg_indices))
            rows.append(row)

        scenario_valid = valid[:, i]
        scenario_summary = {
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
        if foot_contact is not None:
            scenario_summary.update(
                _contact_metrics(foot_contact[scenario_valid, i], leg_indices)
            )
        scenario_summaries.append(scenario_summary)

    with (args.trace_dir / "temporal_allocation_segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["scenario"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "trace_dir": str(args.trace_dir),
        "segment_m": args.segment_m,
        "feet_body_names": feet_body_names,
        "leg_indices": leg_indices,
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
    unique_speeds = np.unique(requested)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, max(len(unique_speeds), 1)))
    for requested_speed, color in zip(unique_speeds, colors):
        speed_rows = [
            row for row in rows
            if np.isclose(float(row["requested_speed_m_s"]), requested_speed)
        ]
        segment_starts = sorted({float(row["progress_start_m"]) for row in speed_rows})
        centers: list[float] = []
        speed_median: list[float] = []
        speed_low: list[float] = []
        speed_high: list[float] = []
        slack_median: list[float] = []
        slack_low: list[float] = []
        slack_high: list[float] = []
        for start in segment_starts:
            segment_rows = [
                row for row in speed_rows
                if np.isclose(float(row["progress_start_m"]), start)
            ]
            segment_speed = np.asarray(
                [row["tangent_speed_mean_m_s"] for row in segment_rows], dtype=np.float64
            )
            segment_slack = np.asarray(
                [row["schedule_error_mean_m"] for row in segment_rows], dtype=np.float64
            )
            segment_speed = segment_speed[np.isfinite(segment_speed)]
            segment_slack = segment_slack[np.isfinite(segment_slack)]
            if not segment_speed.size or not segment_slack.size:
                continue
            centers.append(start + 0.5 * args.segment_m)
            speed_low.append(float(np.quantile(segment_speed, 0.1)))
            speed_median.append(float(np.median(segment_speed)))
            speed_high.append(float(np.quantile(segment_speed, 0.9)))
            slack_low.append(float(np.quantile(segment_slack, 0.1)))
            slack_median.append(float(np.median(segment_slack)))
            slack_high.append(float(np.quantile(segment_slack, 0.9)))
        if not centers:
            continue
        label = f"{requested_speed:.2f} m/s"
        axes[0].plot(centers, speed_median, color=color, linewidth=2.0, label=label)
        axes[0].fill_between(centers, speed_low, speed_high, color=color, alpha=0.16)
        axes[0].axhline(requested_speed, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        axes[1].plot(centers, slack_median, color=color, linewidth=2.0, label=label)
        axes[1].fill_between(centers, slack_low, slack_high, color=color, alpha=0.16)
    axes[0].set_ylabel("Tangent speed [m/s]")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Schedule error [m]")
    axes[1].set_xlabel("Route progress [m]")
    axes[0].set_title("Route-progress profiles (median and 10–90% across routes)")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(title="Requested mean speed", fontsize=8, ncol=min(3, len(unique_speeds)))
    fig.tight_layout()
    fig.savefig(args.trace_dir / "temporal_allocation.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"scenarios": len(scenario_summaries), "segments": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
