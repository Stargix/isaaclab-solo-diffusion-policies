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


def _contact_metrics_by_height(
    contact: np.ndarray,
    required_height: np.ndarray,
    valid: np.ndarray,
    leg_indices: dict[str, int],
) -> dict[str, dict[str, object]]:
    """Summarize contacts in low- and high-height route sections.

    The names describe the commanded body-height section, not a latent skill.
    This deliberately avoids inferring a gait label from height alone.
    """

    heights = np.asarray(required_height, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(heights)
    if not np.any(valid):
        return {}
    low = float(np.min(heights[valid]))
    high = float(np.max(heights[valid]))
    if high - low < 1.0e-3:
        sections = {"constant_height": valid}
    else:
        midpoint = 0.5 * (low + high)
        sections = {
            "crouch_height": valid & (heights <= midpoint),
            "walk_height": valid & (heights > midpoint),
        }
    output: dict[str, dict[str, object]] = {}
    for name, mask in sections.items():
        if not np.any(mask):
            continue
        output[name] = {
            "required_height_m": float(np.mean(heights[mask])),
            **_contact_metrics(contact[mask], leg_indices),
        }
    return output


def _low_height_spans(
    rows: list[dict[str, object]], segment_m: float
) -> list[tuple[float, float]]:
    """Return contiguous low-height spans for plot shading."""

    starts = sorted({float(row["progress_start_m"]) for row in rows})
    if not starts:
        return []
    median_height: dict[float, float] = {}
    for start in starts:
        values = np.asarray(
            [
                row["required_height_m"]
                for row in rows
                if np.isclose(float(row["progress_start_m"]), start)
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size:
            median_height[start] = float(np.median(values))
    if not median_height:
        return []
    low = min(median_height.values())
    high = max(median_height.values())
    if high - low < 1.0e-3:
        return []
    midpoint = 0.5 * (low + high)
    spans: list[tuple[float, float]] = []
    span_start: float | None = None
    previous = 0.0
    for start in starts:
        is_low = median_height.get(start, high) <= midpoint
        if is_low and span_start is None:
            span_start = start
        if not is_low and span_start is not None:
            spans.append((span_start, previous + segment_m))
            span_start = None
        previous = start
    if span_start is not None:
        spans.append((span_start, previous + segment_m))
    return spans


def _valid_until_first_arrival(
    positions_xy: np.ndarray,
    progress_m: np.ndarray,
    valid: np.ndarray,
    scenarios: list[dict[str, object]],
    *,
    position_tolerance_m: float = 0.15,
    near_terminal_distance_m: float = 0.12,
) -> np.ndarray:
    """Stop diagnostic traces at the first geometric endpoint arrival.

    The task contract defines arrival by endpoint position while evaluating
    pose, speed and height exactly once at that event.  The trace artifact is
    written before summary metrics are assembled, so this function reproduces
    only the geometric arrival gate needed to exclude post-task standing or
    reset behavior from temporal and contact diagnostics.
    """

    positions = np.asarray(positions_xy, dtype=np.float64)
    progress = np.asarray(progress_m, dtype=np.float64)
    active = np.asarray(valid, dtype=bool).copy()
    if positions.shape[:2] != progress.shape or active.shape != progress.shape:
        raise ValueError("positions, progress and valid must share (time, scenario) dimensions")
    if positions.shape[1] != len(scenarios):
        raise ValueError("scenario metadata count does not match trace width")
    for index, scenario in enumerate(scenarios):
        path_xy = np.asarray(scenario["path_xy"], dtype=np.float64)
        path_cumulative = np.asarray(scenario["path_cumulative_m"], dtype=np.float64)
        if path_xy.ndim != 2 or path_xy.shape[1] != 2 or not path_cumulative.size:
            raise ValueError("scenario path metadata is malformed")
        target_progress = float(path_cumulative[-1] - path_cumulative[0])
        position_error = np.linalg.norm(positions[:, index] - path_xy[-1], axis=1)
        near_terminal = target_progress - progress[:, index] <= near_terminal_distance_m
        candidates = np.flatnonzero(
            active[:, index] & near_terminal & (position_error <= position_tolerance_m)
        )
        if candidates.size:
            active[int(candidates[0]) + 1 :, index] = False
    return active


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
    positions_xy = np.asarray(trace["positions_xy"], dtype=np.float64)
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
    valid = _valid_until_first_arrival(
        positions_xy,
        progress,
        valid,
        metadata["scenarios"],
    )

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
            scenario_summary["contact_by_height_section"] = _contact_metrics_by_height(
                foot_contact[:, i],
                required_height[:, i],
                scenario_valid,
                leg_indices,
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
    low_height_spans = _low_height_spans(rows, args.segment_m)
    for axis in axes:
        for span_index, (start, end) in enumerate(low_height_spans):
            axis.axvspan(
                start,
                end,
                color="#777777",
                alpha=0.10,
                label="Crouch-height section" if axis is axes[0] and span_index == 0 else None,
                zorder=0,
            )
    axes[0].set_ylabel("Tangent speed [m/s]")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Schedule error [m]")
    axes[1].set_xlabel("Route progress [m]")
    axes[0].set_title("Route-progress profiles (median and 10-90% across routes)")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(title="Requested mean speed", fontsize=8, ncol=min(3, len(unique_speeds)))
    fig.tight_layout()
    fig.savefig(args.trace_dir / "temporal_allocation.png", dpi=180)
    plt.close(fig)

    if foot_contact is not None and leg_indices:
        gait_metrics = (
            ("duty_factor_mean", "Mean duty factor", (0.0, 1.0)),
            ("flight_fraction", "Full-flight fraction", (0.0, 0.35)),
            ("diagonal_contact_correlation", "Diagonal contact correlation", (-1.0, 1.0)),
            ("ipsilateral_contact_correlation", "Ipsilateral contact correlation", (-1.0, 1.0)),
        )
        section_style = {
            "walk_height": ("#0052CC", "Walk-height section"),
            "crouch_height": ("#F59E0B", "Crouch-height section"),
            "constant_height": ("#555555", "Constant-height section"),
        }
        observed_section_heights = np.asarray(
            [
                float(section_metrics["required_height_m"])
                for scenario in scenario_summaries
                for section_metrics in scenario.get("contact_by_height_section", {}).values()
            ],
            dtype=np.float64,
        )
        height_range = (
            float(np.min(observed_section_heights)),
            float(np.max(observed_section_heights)),
        )
        height_midpoint = 0.5 * (height_range[0] + height_range[1])

        def canonical_height_section(section_metrics: dict[str, object]) -> str:
            if height_range[1] - height_range[0] < 1.0e-3:
                return "constant_height"
            return (
                "crouch_height"
                if float(section_metrics["required_height_m"]) <= height_midpoint
                else "walk_height"
            )

        gait_figure, gait_axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
        for axis, (metric, title, limits) in zip(gait_axes.flat, gait_metrics):
            for section, (color, label) in section_style.items():
                centers: list[float] = []
                medians: list[float] = []
                low_values: list[float] = []
                high_values: list[float] = []
                for requested_speed in unique_speeds:
                    values = []
                    for scenario in scenario_summaries:
                        if not np.isclose(float(scenario["requested_speed_m_s"]), requested_speed):
                            continue
                        for section_metrics in scenario.get("contact_by_height_section", {}).values():
                            if canonical_height_section(section_metrics) != section:
                                continue
                            value = section_metrics.get(metric)
                            if value is not None and np.isfinite(float(value)):
                                values.append(float(value))
                    if not values:
                        continue
                    array = np.asarray(values, dtype=np.float64)
                    centers.append(float(requested_speed))
                    medians.append(float(np.median(array)))
                    low_values.append(float(np.quantile(array, 0.1)))
                    high_values.append(float(np.quantile(array, 0.9)))
                if centers:
                    axis.plot(centers, medians, "o-", color=color, linewidth=2.0, label=label)
                    axis.fill_between(centers, low_values, high_values, color=color, alpha=0.16)
            axis.set_title(title)
            axis.set_xlabel("Requested mean speed [m/s]")
            axis.set_ylim(*limits)
            axis.grid(alpha=0.25)
        handles, labels = gait_axes.flat[0].get_legend_handles_labels()
        if handles:
            gait_figure.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
        gait_figure.suptitle(
            "Post-hoc contact signature (median and 10-90% across routes; no skill labels)",
            fontweight="bold",
            y=0.995,
        )
        if handles:
            gait_figure.legends[0].set_bbox_to_anchor((0.5, 0.965))
        gait_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
        gait_figure.savefig(args.trace_dir / "gait_signature.png", dpi=180, bbox_inches="tight")
        plt.close(gait_figure)

        # Fixed qualitative selection rule: lexicographically first route at
        # the evaluated speed nearest 0.85 m/s.  This prevents cherry-picking
        # a visually convenient footfall trace after seeing the result.
        raster_speed = float(unique_speeds[np.argmin(np.abs(unique_speeds - 0.85))])
        candidates = [
            index
            for index, scenario in enumerate(metadata["scenarios"])
            if np.isclose(requested[index], raster_speed)
        ]
        raster_index = min(
            candidates,
            key=lambda index: (
                str(metadata["scenarios"][index]["path_shape"]),
                int(metadata["scenarios"][index]["repeat"]),
                index,
            ),
        )
        raster_valid = valid[:, raster_index]
        raster_time = time_s[raster_valid]
        raster_contact = foot_contact[raster_valid, raster_index]
        raster_height = required_height[raster_valid, raster_index]
        raster_speed_trace = speed[raster_valid, raster_index]
        raster_figure, raster_axes = plt.subplots(
            2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": (1.0, 1.3)}
        )
        raster_axes[0].plot(raster_time, raster_speed_trace, color="#0052CC", linewidth=1.5)
        raster_axes[0].set_ylabel("Tangent speed [m/s]", color="#0052CC")
        raster_axes[0].tick_params(axis="y", labelcolor="#0052CC")
        height_axis = raster_axes[0].twinx()
        height_axis.plot(raster_time, raster_height, color="#F59E0B", linewidth=1.5, linestyle="--")
        height_axis.set_ylabel("Required height [m]", color="#C47A00")
        height_axis.tick_params(axis="y", labelcolor="#C47A00")
        raster_axes[0].grid(alpha=0.2)
        if raster_time.size:
            raster_axes[1].imshow(
                raster_contact.T,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                cmap="Greys",
                vmin=0,
                vmax=1,
                extent=(float(raster_time[0]), float(raster_time[-1]), -0.5, raster_contact.shape[1] - 0.5),
            )
        ordered_names = feet_body_names if feet_body_names else [f"foot_{i}" for i in range(raster_contact.shape[1])]
        raster_axes[1].set_yticks(np.arange(len(ordered_names)), ordered_names)
        raster_axes[1].set_xlabel("Time [s]")
        raster_axes[1].set_ylabel("Contact (black = stance)")
        selected = metadata["scenarios"][raster_index]
        raster_figure.suptitle(
            f"Pre-specified contact raster: {selected['path_shape']} r{selected['repeat']}, "
            f"{raster_speed:.2f} m/s",
            fontweight="bold",
        )
        raster_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        raster_figure.savefig(
            args.trace_dir / "representative_contact_raster.png", dpi=180, bbox_inches="tight"
        )
        plt.close(raster_figure)
    print(json.dumps({"scenarios": len(scenario_summaries), "segments": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
