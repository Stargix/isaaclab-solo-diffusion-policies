#!/usr/bin/env python3
"""Compare Phase-A training goals with the analytic paths used in evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_POLICY_ROOT = _PROJECT_ROOT / "scripts" / "diffusion_policy"
sys.path.insert(0, str(_POLICY_ROOT))

from train.conditioning.geometry import cumulative_xy_lengths
from train.conditioning.goal_builder import (
    WAYPOINT_TIME_OFFSETS_S,
    build_goal_from_path,
    yaw_to_quat_wxyz,
)
from train.data.dataset import SpatialHindsightDataset


FEATURE_NAMES = (
    "preview_0p5_x", "preview_0p5_y", "preview_1p0_x", "preview_1p0_y",
    "preview_1p5_x", "preview_1p5_y", "target_2p0_x", "target_2p0_y",
    "target_height_abs", "target_yaw_rel", "preview_speed_2p0",
    "preview_distance_0p5", "preview_distance_1p0", "preview_distance_1p5",
    "preview_curvature_signed", "target_lateral_rate_2p0", "target_yaw_rate_2p0",
)


def expand_goal_features(goals: np.ndarray) -> np.ndarray:
    p0 = np.zeros((len(goals), 2), dtype=np.float32)
    p1, p2, p3 = goals[:, 0:2], goals[:, 2:4], goals[:, 4:6]
    side_a = np.linalg.norm(p1 - p0, axis=1)
    side_b = np.linalg.norm(p2 - p1, axis=1)
    side_c = np.linalg.norm(p2 - p0, axis=1)
    cross = p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]
    curvature = 2.0 * cross / np.maximum(side_a * side_b * side_c, 1.0e-6)
    derived = np.column_stack(
        (
            np.linalg.norm(p1, axis=1),
            np.linalg.norm(p2, axis=1),
            np.linalg.norm(p3, axis=1),
            curvature,
            goals[:, 7] / 2.0,
            goals[:, 9] / 2.0,
        )
    ).astype(np.float32)
    return np.concatenate((goals, derived), axis=1)


def build_path(shape: str, *, length_m: float = 10.0, points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    if shape == "straight":
        s = np.linspace(0.0, length_m, points, dtype=np.float32)
        xy = np.stack((s, np.zeros_like(s)), axis=-1)
    elif shape == "s_curve":
        s = np.linspace(0.0, length_m, points, dtype=np.float32)
        xy = np.stack((s, 0.3 * np.sin(2.0 * np.pi * s / 6.0)), axis=-1)
    elif shape == "circle":
        radius = 1.5
        theta = np.linspace(0.0, length_m / radius, points, dtype=np.float32)
        xy = np.stack((radius * np.sin(theta), radius * (1.0 - np.cos(theta))), axis=-1)
    else:
        raise ValueError(f"Unknown path shape {shape!r}.")
    path = np.column_stack((xy, np.full(len(xy), 0.2932, dtype=np.float32))).astype(np.float32)
    delta = np.gradient(xy, axis=0)
    yaws = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0])).astype(np.float32)
    return path, yaws


def sample_training_goals(
    dataset_path: str,
    *,
    max_samples: int,
    seed: int,
    goal_horizon_steps: int,
) -> np.ndarray:
    dataset = SpatialHindsightDataset(
        [dataset_path],
        goal_horizon_steps=goal_horizon_steps,
        waypoint_time_offsets_s=WAYPOINT_TIME_OFFSETS_S,
        goal_source="reference",
        include_padded_starts=True,
        startup_sample_multiplier=1,
        symmetry_mode="none",
    )
    rng = np.random.default_rng(seed)
    count = min(max_samples, len(dataset.samples))
    indices = np.sort(rng.choice(len(dataset.samples), count, replace=False))
    return np.stack([dataset.goal_for_sample(int(index)) for index in indices]).astype(np.float32)


def sample_evaluation_goals(
    shapes: tuple[str, ...],
    speeds: tuple[float, ...],
    *,
    goal_horizon_steps: int,
    dt: float,
    samples_per_scenario: int,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    goals: list[np.ndarray] = []
    labels: list[dict[str, float | str]] = []
    for shape in shapes:
        path, yaws = build_path(shape)
        cumulative = cumulative_xy_lengths(path)
        for speed in speeds:
            terminal_margin = max(speed * goal_horizon_steps * dt, 0.05)
            valid = np.flatnonzero(cumulative <= cumulative[-1] - terminal_margin)
            if not len(valid):
                continue
            chosen = np.linspace(valid[0], valid[-1], min(samples_per_scenario, len(valid)), dtype=int)
            for index in np.unique(chosen):
                goal = build_goal_from_path(
                    path,
                    cumulative,
                    yaws,
                    path[index],
                    yaw_to_quat_wxyz(float(yaws[index])),
                    goal_horizon_steps=goal_horizon_steps,
                    dt=dt,
                    speed=speed,
                    start_idx=int(index),
                    waypoint_time_offsets_s=WAYPOINT_TIME_OFFSETS_S,
                )
                goals.append(goal)
                labels.append({"path_shape": shape, "requested_speed": float(speed)})
    if not goals:
        raise ValueError("No evaluation goals were generated.")
    return np.stack(goals).astype(np.float32), labels


def summarize_support(train: np.ndarray, evaluation: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0), 1.0e-6)
    minimum = train.min(axis=0)
    maximum = train.max(axis=0)
    p01 = np.percentile(train, 1, axis=0)
    p99 = np.percentile(train, 99, axis=0)
    z = (evaluation - mean) / std
    outside_range = (evaluation < minimum) | (evaluation > maximum)
    outside_central = (evaluation < p01) | (evaluation > p99)
    per_feature = {}
    for index, name in enumerate(FEATURE_NAMES):
        per_feature[name] = {
            "train_min": float(minimum[index]),
            "train_p01": float(p01[index]),
            "train_mean": float(mean[index]),
            "train_std": float(std[index]),
            "train_p99": float(p99[index]),
            "train_max": float(maximum[index]),
            "eval_min": float(evaluation[:, index].min()),
            "eval_max": float(evaluation[:, index].max()),
            "outside_train_range_fraction": float(outside_range[:, index].mean()),
            "outside_train_p01_p99_fraction": float(outside_central[:, index].mean()),
        }
    row_metrics = np.column_stack(
        (
            np.max(np.abs(z), axis=1),
            np.sqrt(np.mean(np.square(z), axis=1)),
            np.mean(outside_range, axis=1),
            np.mean(outside_central, axis=1),
        )
    )
    summary = {
        "train_samples": int(len(train)),
        "evaluation_samples": int(len(evaluation)),
        "any_feature_outside_train_range_fraction": float(np.any(outside_range, axis=1).mean()),
        "any_feature_outside_train_p01_p99_fraction": float(np.any(outside_central, axis=1).mean()),
        "max_abs_z_p50_p95_p99": [float(np.percentile(row_metrics[:, 0], q)) for q in (50, 95, 99)],
        "features": per_feature,
    }
    return summary, row_metrics


def write_outputs(
    output: Path,
    train: np.ndarray,
    evaluation: np.ndarray,
    labels: list[dict[str, float | str]],
    summary: dict[str, object],
    row_metrics: np.ndarray,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, goal, metrics in zip(labels, evaluation, row_metrics):
        rows.append({
            **label,
            **{name: float(goal[index]) for index, name in enumerate(FEATURE_NAMES)},
            "max_abs_z": float(metrics[0]),
            "rms_z": float(metrics[1]),
            "outside_train_range_fraction": float(metrics[2]),
            "outside_train_p01_p99_fraction": float(metrics[3]),
        })
    with (output / "evaluation_goal_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scenario_summary = {}
    for shape in sorted({str(item["path_shape"]) for item in labels}):
        for speed in sorted({float(item["requested_speed"]) for item in labels}):
            mask = np.asarray(
                [item["path_shape"] == shape and float(item["requested_speed"]) == speed for item in labels]
            )
            if np.any(mask):
                scenario_summary[f"{shape}@{speed:g}"] = {
                    "samples": int(mask.sum()),
                    "max_abs_z_p95": float(np.percentile(row_metrics[mask, 0], 95)),
                    "outside_train_range_fraction": float(np.mean(row_metrics[mask, 2] > 0.0)),
                    "outside_train_p01_p99_fraction": float(np.mean(row_metrics[mask, 3] > 0.0)),
                }
    summary["scenarios"] = scenario_summary
    (output / "goal_support_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(train[:, 6], train[:, 7], s=2, alpha=0.08, label="train")
    for shape in sorted({str(item["path_shape"]) for item in labels}):
        mask = np.asarray([item["path_shape"] == shape for item in labels])
        axes[0].scatter(evaluation[mask, 6], evaluation[mask, 7], s=9, alpha=0.45, label=shape)
    axes[0].set(title="Terminal goal support", xlabel="target x [m]", ylabel="target y [m]")
    axes[0].axis("equal")
    axes[0].legend(fontsize=8)

    axes[1].hist(row_metrics[:, 0], bins=50)
    axes[1].axvline(3.0, color="tab:red", linestyle="--", label="3 std")
    axes[1].set(title="Evaluation goal distance", xlabel="max absolute z-score", ylabel="count")
    axes[1].legend(fontsize=8)

    positions = np.arange(len(FEATURE_NAMES))
    outside = [summary["features"][name]["outside_train_p01_p99_fraction"] for name in FEATURE_NAMES]
    axes[2].bar(positions, outside)
    axes[2].set_xticks(positions, FEATURE_NAMES, rotation=75, ha="right", fontsize=7)
    axes[2].set(title="Outside central train support", ylabel="fraction")
    for axis in axes:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "goal_support_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0))
    parser.add_argument("--shapes", nargs="+", default=("straight", "circle", "s_curve"))
    parser.add_argument("--max_train_samples", type=int, default=50_000)
    parser.add_argument("--samples_per_scenario", type=int, default=200)
    parser.add_argument("--goal_horizon_steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = sample_training_goals(
        args.dataset,
        max_samples=args.max_train_samples,
        seed=args.seed,
        goal_horizon_steps=args.goal_horizon_steps,
    )
    evaluation, labels = sample_evaluation_goals(
        tuple(args.shapes),
        tuple(args.speeds),
        goal_horizon_steps=args.goal_horizon_steps,
        dt=args.dt,
        samples_per_scenario=args.samples_per_scenario,
    )
    train = expand_goal_features(train)
    evaluation = expand_goal_features(evaluation)
    summary, row_metrics = summarize_support(train, evaluation)
    summary.update({
        "dataset": str(Path(args.dataset).resolve()),
        "speeds": list(args.speeds),
        "shapes": list(args.shapes),
        "goal_horizon_steps": args.goal_horizon_steps,
        "dt": args.dt,
    })
    write_outputs(Path(args.output_dir), train, evaluation, labels, summary, row_metrics)
    print(json.dumps({key: summary[key] for key in (
        "train_samples", "evaluation_samples", "any_feature_outside_train_range_fraction",
        "any_feature_outside_train_p01_p99_fraction", "max_abs_z_p50_p95_p99",
    )}, indent=2))


if __name__ == "__main__":
    main()
