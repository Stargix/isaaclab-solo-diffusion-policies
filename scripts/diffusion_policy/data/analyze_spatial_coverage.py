#!/usr/bin/env python3
"""Measure temporal-preview goal coverage before the first spatial train."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT))

from train.conditioning.goal_builder import (
    GOAL_SCHEMA_NAME, REFERENCE_GOAL_SCHEMA_NAME, HOLONOMIC_REFERENCE_GOAL_SCHEMA_NAME,
    PATH_GUIDANCE_GOAL_SCHEMA_NAME, PATH_GUIDANCE_FRACTIONS,
    GEOMETRIC_HINDSIGHT_GOAL_SCHEMA_NAME, GEOMETRIC_WAYPOINT_FRACTIONS,
    GEOMETRIC_HEIGHT_PROFILE_GOAL_SCHEMA_NAME,
    HOLONOMIC_TOKEN_TIMES_S, REFERENCE_GOAL_REPRESENTATIONS, WAYPOINT_TIME_OFFSETS_S,
)
from train.data.dataset import SpatialHindsightDataset


FEATURE_NAMES = (
    "preview_0p5_x",
    "preview_0p5_y",
    "preview_1p0_x",
    "preview_1p0_y",
    "preview_1p5_x",
    "preview_1p5_y",
    "target_2p0_x",
    "target_2p0_y",
    "target_height_abs",
    "target_yaw_rel",
    "preview_speed_2p0",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def feature_stats(values: np.ndarray, feature_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    output = {}
    for index, name in enumerate(feature_names):
        column = values[:, index]
        output[name] = {
            "min": float(np.min(column)),
            "p01": float(np.percentile(column, 1)),
            "p05": float(np.percentile(column, 5)),
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "p50": float(np.percentile(column, 50)),
            "p95": float(np.percentile(column, 95)),
            "p99": float(np.percentile(column, 99)),
            "max": float(np.max(column)),
        }
    return output


def derived_metrics(values: np.ndarray) -> dict[str, float]:
    p1, p2, p3, target = (values[:, start : start + 2] for start in (0, 2, 4, 6))
    distances = np.stack(
        [np.linalg.norm(point, axis=-1) for point in (p1, p2, p3, target)], axis=-1
    )
    increments = np.stack(
        [
            np.linalg.norm(p2 - p1, axis=-1),
            np.linalg.norm(p3 - p2, axis=-1),
            np.linalg.norm(target - p3, axis=-1),
        ],
        axis=-1,
    )
    first = p2 - p1
    second = p3 - p2
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = np.sum(first * second, axis=-1)
    turn_angle = np.arctan2(cross, dot)
    duplicate = increments < 1.0e-4
    non_monotonic = np.any(np.diff(distances, axis=-1) < -1.0e-3, axis=-1)
    return {
        "preview_duplicate_fraction": float(np.mean(np.any(duplicate, axis=-1))),
        "individual_duplicate_fraction": float(np.mean(duplicate)),
        "radial_non_monotonic_fraction": float(np.mean(non_monotonic)),
        "target_distance_p05_m": float(np.percentile(distances[:, -1], 5)),
        "target_distance_p50_m": float(np.percentile(distances[:, -1], 50)),
        "target_distance_p95_m": float(np.percentile(distances[:, -1], 95)),
        "abs_turn_angle_p50_rad": float(np.percentile(np.abs(turn_angle), 50)),
        "abs_turn_angle_p95_rad": float(np.percentile(np.abs(turn_angle), 95)),
    }


def write_samples(path: Path, values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FEATURE_NAMES)
        writer.writerows(values.tolist())


def make_plots(values: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(values[:, 8], bins=60)
    axes[0, 0].axvline(0.1705, color="tab:red", linestyle="--", label="crouch reference")
    axes[0, 0].axvline(0.2932, color="tab:green", linestyle="--", label="walk reference")
    axes[0, 0].set(title="Hindsight target height", xlabel="height [m]", ylabel="count")
    axes[0, 0].legend(fontsize=8)

    target_distance = np.linalg.norm(values[:, 6:8], axis=-1)
    axes[0, 1].scatter(values[:, 10], target_distance, c=values[:, 8], s=4, alpha=0.25)
    axes[0, 1].set(title="Speed and two-second reach", xlabel="preview speed [m/s]", ylabel="target distance [m]")

    axes[1, 0].scatter(values[:, 6], values[:, 7], c=values[:, 8], s=4, alpha=0.25)
    axes[1, 0].set(title="Terminal XY support", xlabel="target x [m]", ylabel="target y [m]")
    axes[1, 0].axis("equal")

    preview_distance = np.stack(
        [np.linalg.norm(values[:, start : start + 2], axis=-1) for start in (0, 2, 4, 6)], axis=-1
    )
    axes[1, 1].boxplot(preview_distance, tick_labels=("0.5s", "1.0s", "1.5s", "2.0s"), showfliers=False)
    axes[1, 1].set(title="Temporal preview distances", ylabel="distance [m]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "spatial_goal_coverage.png", dpi=180)
    plt.close(figure)


def holonomic_feature_names() -> tuple[str, ...]:
    names = []
    for time_s in HOLONOMIC_TOKEN_TIMES_S:
        prefix = f"t{time_s:.1f}".replace(".", "p")
        names.extend((
            f"{prefix}_x", f"{prefix}_y", f"{prefix}_sin_dyaw", f"{prefix}_cos_dyaw",
            f"{prefix}_vx_ref", f"{prefix}_vy_ref", f"{prefix}_wz_ref", f"{prefix}_tau",
        ))
    return tuple(names)


def holonomic_derived(values: np.ndarray) -> dict[str, float]:
    tokens = values.reshape(-1, 4, 8)
    distances = np.linalg.norm(tokens[:, :, :2], axis=-1)
    speed = np.linalg.norm(tokens[:, :, 4:6], axis=-1)
    heading = np.arctan2(tokens[:, :, 2], tokens[:, :, 3])
    return {
        "terminal_distance_p05_m": float(np.percentile(distances[:, -1], 5)),
        "terminal_distance_p50_m": float(np.percentile(distances[:, -1], 50)),
        "terminal_distance_p95_m": float(np.percentile(distances[:, -1], 95)),
        "lateral_velocity_nonzero_fraction": float(np.mean(np.abs(tokens[:, :, 5]) > 0.02)),
        "reverse_velocity_fraction": float(np.mean(tokens[:, :, 4] < -0.02)),
        "diagonal_velocity_fraction": float(np.mean((np.abs(tokens[:, :, 4]) > 0.02) & (np.abs(tokens[:, :, 5]) > 0.02))),
        "abs_heading_p95_rad": float(np.percentile(np.abs(heading), 95)),
        "abs_wz_p95_rad_s": float(np.percentile(np.abs(tokens[:, :, 6]), 95)),
        "speed_p95_m_s": float(np.percentile(speed, 95)),
    }


def make_holonomic_plots(values: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = values.reshape(-1, 4, 8)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    terminal = tokens[:, -1]
    axes[0, 0].scatter(terminal[:, 0], terminal[:, 1], s=4, alpha=0.2)
    axes[0, 0].set(title="Terminal XY support", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].axis("equal")
    axes[0, 1].scatter(terminal[:, 4], terminal[:, 5], c=terminal[:, 6], s=4, alpha=0.2)
    axes[0, 1].set(title="Terminal reference velocity", xlabel="vx [m/s]", ylabel="vy [m/s]")
    axes[0, 1].axis("equal")
    heading = np.arctan2(tokens[:, :, 2], tokens[:, :, 3]).reshape(-1)
    axes[1, 0].hist(heading, bins=60)
    axes[1, 0].set(title="Preview heading support", xlabel="delta yaw [rad]", ylabel="count")
    time_labels = tuple(f"{time_s:.1f}s" for time_s in HOLONOMIC_TOKEN_TIMES_S)
    axes[1, 1].boxplot(
        np.linalg.norm(tokens[:, :, 4:6], axis=-1),
        tick_labels=time_labels,
        showfliers=False,
    )
    axes[1, 1].set(title="Reference speed by preview", ylabel="speed [m/s]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "holonomic_goal_coverage.png", dpi=180)
    plt.close(figure)


def path_guidance_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for index, fraction in enumerate(PATH_GUIDANCE_FRACTIONS):
        names.extend((
            f"guide_{index}_dir_x",
            f"guide_{index}_dir_y",
            f"guide_{index}_log_distance",
            f"guide_{index}_arc_offset_fraction_{fraction:.3f}",
        ))
    names.extend((
        "terminal_x", "terminal_y", "terminal_sin_dyaw", "terminal_cos_dyaw",
        "terminal_height", "terminal_time_to_go", "terminal_phase", "guide_terminal_error",
    ))
    return tuple(names)


def path_guidance_derived(values: np.ndarray) -> dict[str, float]:
    tokens = values[:, :28].reshape(-1, 7, 4)
    guide_distance = np.expm1(tokens[:, :, 2])
    guide_xy = tokens[:, :, :2] * guide_distance[:, :, None]
    terminal_xy = values[:, 28:30]
    guide_terminal_error = np.linalg.norm(guide_xy[:, -1] - terminal_xy, axis=-1)
    return {
        "terminal_distance_p05_m": float(np.percentile(np.linalg.norm(terminal_xy, axis=-1), 5)),
        "terminal_distance_p50_m": float(np.percentile(np.linalg.norm(terminal_xy, axis=-1), 50)),
        "terminal_distance_p95_m": float(np.percentile(np.linalg.norm(terminal_xy, axis=-1), 95)),
        "guide_endpoint_terminal_error_p50_m": float(np.percentile(guide_terminal_error, 50)),
        "guide_endpoint_terminal_error_p95_m": float(np.percentile(guide_terminal_error, 95)),
        "guide_endpoint_terminal_error_nonzero_fraction": float(np.mean(guide_terminal_error > 0.01)),
        "time_to_go_p05_s": float(np.percentile(values[:, 33], 5)),
        "time_to_go_p50_s": float(np.percentile(values[:, 33], 50)),
        "time_to_go_p95_s": float(np.percentile(values[:, 33], 95)),
        "terminal_hold_fraction": float(np.mean(values[:, 33] <= 1.0e-6)),
        "terminal_phase_fraction": float(np.mean(values[:, 34] > 0.5)),
        "declared_guide_terminal_error_p95_m": float(np.percentile(values[:, 35], 95)),
    }


def make_path_guidance_plots(values: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = values[:, :28].reshape(-1, 7, 4)
    distance = np.expm1(tokens[:, :, 2])
    guide_xy = tokens[:, :, :2] * distance[:, :, None]
    terminal_xy = values[:, 28:30]
    mismatch = np.linalg.norm(guide_xy[:, -1] - terminal_xy, axis=-1)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(terminal_xy[:, 0], terminal_xy[:, 1], s=4, alpha=0.2)
    axes[0, 0].set(title="Clean terminal-pose support", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].axis("equal")
    axes[0, 1].hist(mismatch, bins=60)
    axes[0, 1].set(title="Noisy guide endpoint vs clean terminal", xlabel="XY mismatch [m]", ylabel="count")
    axes[1, 0].boxplot(distance, tick_labels=tuple(f"{v:.2f}" for v in PATH_GUIDANCE_FRACTIONS), showfliers=False)
    axes[1, 0].set(title="Guide distance by remaining-path fraction", xlabel="path fraction", ylabel="distance [m]")
    axes[1, 1].hist(values[:, 33], bins=60)
    axes[1, 1].set(title="Terminal timing and hold coverage", xlabel="time-to-go [s]", ylabel="count")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "path_guidance_goal_coverage.png", dpi=180)
    plt.close(figure)


GEOMETRIC_FEATURE_NAMES = (
    "waypoint_25_x", "waypoint_25_y",
    "waypoint_50_x", "waypoint_50_y",
    "waypoint_75_x", "waypoint_75_y",
    "terminal_x", "terminal_y",
    "terminal_sin_dyaw", "terminal_cos_dyaw",
    "terminal_height_abs", "average_path_speed",
)

HEIGHT_PROFILE_FEATURE_NAMES = (
    "waypoint_25_x", "waypoint_25_y", "waypoint_25_height_abs",
    "waypoint_50_x", "waypoint_50_y", "waypoint_50_height_abs",
    "waypoint_75_x", "waypoint_75_y", "waypoint_75_height_abs",
    "terminal_x", "terminal_y", "terminal_height_abs",
    "current_height_abs", "terminal_sin_dyaw", "terminal_cos_dyaw", "average_path_speed",
)


def geometric_derived(values: np.ndarray) -> dict[str, float]:
    points = values[:, :8].reshape(-1, 4, 2)
    increments = np.linalg.norm(np.diff(points, axis=1), axis=-1)
    return {
        "terminal_distance_p05_m": float(np.percentile(np.linalg.norm(points[:, -1], axis=-1), 5)),
        "terminal_distance_p50_m": float(np.percentile(np.linalg.norm(points[:, -1], axis=-1), 50)),
        "terminal_distance_p95_m": float(np.percentile(np.linalg.norm(points[:, -1], axis=-1), 95)),
        "average_speed_p05_m_s": float(np.percentile(values[:, 11], 5)),
        "average_speed_p50_m_s": float(np.percentile(values[:, 11], 50)),
        "average_speed_p95_m_s": float(np.percentile(values[:, 11], 95)),
        "stop_fraction_v_avg_lt_0p02": float(np.mean(values[:, 11] < 0.02)),
        "duplicate_waypoint_fraction": float(np.mean(np.any(increments < 1.0e-4, axis=-1))),
    }


def make_geometric_plots(values: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = values[:, :8].reshape(-1, 4, 2)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(points[:, -1, 0], points[:, -1, 1], c=values[:, 11], s=4, alpha=0.2)
    axes[0, 0].set(title="Terminal XY support", xlabel="terminal x [m]", ylabel="terminal y [m]")
    axes[0, 0].axis("equal")
    axes[0, 1].scatter(values[:, 11], np.linalg.norm(points[:, -1], axis=-1), s=4, alpha=0.2)
    axes[0, 1].set(title="Average speed and terminal reach", xlabel="v_avg [m/s]", ylabel="terminal distance [m]")
    axes[1, 0].boxplot(
        np.linalg.norm(points, axis=-1),
        tick_labels=("25%", "50%", "75%", "terminal"),
        showfliers=False,
    )
    axes[1, 0].set(title="Geometric waypoint distance", ylabel="distance [m]")
    axes[1, 1].hist(values[:, 11], bins=60)
    axes[1, 1].set(title="Average path speed", xlabel="v_avg [m/s]", ylabel="count")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "hindsight_geom_avg12_coverage.png", dpi=180)
    plt.close(figure)


def height_profile_derived(values: np.ndarray) -> dict[str, float]:
    tokens = values[:, :12].reshape(-1, 4, 3)
    points = tokens[:, :, :2]
    increments = np.linalg.norm(np.diff(points, axis=1), axis=-1)
    profile_changes = np.any(np.abs(np.diff(tokens[:, :, 2], axis=1)) > 1.0e-4, axis=-1)
    current_to_future = np.any(np.abs(tokens[:, :, 2] - values[:, 12:13]) > 1.0e-4, axis=-1)
    return {
        "terminal_distance_p05_m": float(np.percentile(np.linalg.norm(points[:, -1], axis=-1), 5)),
        "terminal_distance_p50_m": float(np.percentile(np.linalg.norm(points[:, -1], axis=-1), 50)),
        "terminal_distance_p95_m": float(np.percentile(np.linalg.norm(points[:, -1], axis=-1), 95)),
        "average_speed_p05_m_s": float(np.percentile(values[:, 15], 5)),
        "average_speed_p50_m_s": float(np.percentile(values[:, 15], 50)),
        "average_speed_p95_m_s": float(np.percentile(values[:, 15], 95)),
        "stop_fraction_v_avg_lt_0p02": float(np.mean(values[:, 15] < 0.02)),
        "duplicate_waypoint_fraction": float(np.mean(np.any(increments < 1.0e-4, axis=-1))),
        "within_preview_height_change_fraction": float(np.mean(profile_changes)),
        "future_height_differs_from_current_fraction": float(np.mean(current_to_future)),
    }


def make_height_profile_plots(values: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = values[:, :12].reshape(-1, 4, 3)
    points = tokens[:, :, :2]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(points[:, -1, 0], points[:, -1, 1], c=values[:, 15], s=4, alpha=0.2)
    axes[0, 0].set(title="Terminal XY support", xlabel="terminal x [m]", ylabel="terminal y [m]")
    axes[0, 0].axis("equal")
    axes[0, 1].hist(values[:, 12], bins=60, alpha=0.7, label="required now")
    axes[0, 1].hist(tokens[:, -1, 2], bins=60, alpha=0.5, label="terminal preview")
    axes[0, 1].set(title="Task-height support", xlabel="height [m]", ylabel="count")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].boxplot(tokens[:, :, 2], tick_labels=("25%", "50%", "75%", "terminal"), showfliers=False)
    axes[1, 0].set(title="Required height by spatial preview", ylabel="height [m]")
    axes[1, 1].scatter(values[:, 15], np.linalg.norm(points[:, -1], axis=-1), s=4, alpha=0.2)
    axes[1, 1].set(title="Average speed and terminal reach", xlabel="v_avg [m/s]", ylabel="terminal distance [m]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "hindsight_geom_profile16_coverage.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal_horizon_steps", type=int, default=100)
    parser.add_argument("--goal_source", choices=["achieved", "reference"], default="achieved")
    parser.add_argument("--goal_representation", choices=REFERENCE_GOAL_REPRESENTATIONS, default="path11")
    parser.add_argument("--include_padded_starts", action="store_true")
    parser.add_argument("--startup_sample_multiplier", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = SpatialHindsightDataset(
        args.datasets,
        goal_horizon_steps=args.goal_horizon_steps,
        waypoint_time_offsets_s=WAYPOINT_TIME_OFFSETS_S,
        goal_source=args.goal_source,
        goal_representation=args.goal_representation,
        include_padded_starts=args.include_padded_starts,
        startup_sample_multiplier=args.startup_sample_multiplier,
        symmetry_mode="none",
    )
    rng = np.random.default_rng(args.seed)
    sample_count = min(max(1, args.max_samples), len(dataset.samples))
    indices = np.sort(rng.choice(len(dataset.samples), sample_count, replace=False))
    values = np.stack([dataset.goal_for_sample(int(index)) for index in indices]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Spatial goal coverage contains NaN or Inf.")

    skill_demo_counts = {}
    for demo in dataset.demos:
        if demo.skill_idx is None or len(demo.skill_idx) == 0:
            name = "unknown"
        else:
            skill_id = int(np.bincount(demo.skill_idx.astype(np.int64)).argmax())
            name = dataset.skill_names[skill_id]
        skill_demo_counts[name] = skill_demo_counts.get(name, 0) + 1

    holonomic = args.goal_representation == "holonomic_se2_32"
    path_guidance = args.goal_representation == "path_guidance_se2_36"
    geometric = args.goal_representation == "hindsight_geom_avg12"
    height_profile = args.goal_representation == "hindsight_geom_profile16"
    names = (
        path_guidance_feature_names() if path_guidance else holonomic_feature_names() if holonomic
        else HEIGHT_PROFILE_FEATURE_NAMES if height_profile
        else GEOMETRIC_FEATURE_NAMES if geometric else FEATURE_NAMES
    )
    summary = {
        "goal_schema": (GEOMETRIC_HEIGHT_PROFILE_GOAL_SCHEMA_NAME if height_profile
                        else GEOMETRIC_HINDSIGHT_GOAL_SCHEMA_NAME if geometric
                        else PATH_GUIDANCE_GOAL_SCHEMA_NAME if path_guidance
                        else HOLONOMIC_REFERENCE_GOAL_SCHEMA_NAME if holonomic
                        else REFERENCE_GOAL_SCHEMA_NAME if args.goal_source == "reference" else GOAL_SCHEMA_NAME),
        "waypoint_time_offsets_s": WAYPOINT_TIME_OFFSETS_S,
        "holonomic_token_times_s": HOLONOMIC_TOKEN_TIMES_S if holonomic else None,
        "path_guidance_fractions": PATH_GUIDANCE_FRACTIONS if path_guidance else None,
        "geometric_waypoint_fractions": GEOMETRIC_WAYPOINT_FRACTIONS if (geometric or height_profile) else None,
        "goal_horizon_steps": args.goal_horizon_steps,
        "goal_source": args.goal_source,
        "goal_representation": args.goal_representation,
        "control_rate_hz": 50.0,
        "datasets": {str(Path(path).resolve()): sha256(path) for path in args.datasets},
        "demos": len(dataset.demos),
        "indexed_windows": len(dataset.samples),
        "sampled_goals": sample_count,
        "skill_demo_counts": skill_demo_counts,
        "feature_stats": feature_stats(values, names),
        "derived": (height_profile_derived(values) if height_profile
                    else geometric_derived(values) if geometric
                    else path_guidance_derived(values) if path_guidance
                    else holonomic_derived(values) if holonomic else derived_metrics(values)),
    }
    (output / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "goal_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(values.tolist())
    if path_guidance:
        make_path_guidance_plots(values, output)
    elif holonomic:
        make_holonomic_plots(values, output)
    elif geometric:
        make_geometric_plots(values, output)
    elif height_profile:
        make_height_profile_plots(values, output)
    else:
        make_plots(values, output)
    print(f"[DONE] Spatial coverage report: {output.resolve()}")
    print(json.dumps(summary["derived"], indent=2))


if __name__ == "__main__":
    main()
