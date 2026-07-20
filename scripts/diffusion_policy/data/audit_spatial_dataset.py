#!/usr/bin/env python3
"""Audit whether spatial deployment goals and reset states are supported by train data.

Hindsight relabeling is only valid for goals and state histories covered by the
offline demonstrations.  This tool quantifies two frequent deployment gaps:
planned-route goals outside the hindsight support, and zero-action reset
histories absent from expert locomotion data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT))

from train.data.dataset import SpatialHindsightDataset
from train.data.normalization import NormalizerStats


TEMPORAL_FEATURE_NAMES = (
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
    "requested_speed",
)

GEOMETRIC_FEATURE_NAMES = (
    "waypoint_25_x", "waypoint_25_y", "waypoint_50_x", "waypoint_50_y",
    "waypoint_75_x", "waypoint_75_y", "terminal_x", "terminal_y",
    "terminal_sin_dyaw", "terminal_cos_dyaw", "terminal_height_abs", "average_path_speed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speeds", type=float, nargs="+", default=(0.2, 0.4, 0.6))
    parser.add_argument("--heights", type=float, nargs="+", default=(0.1705, 0.2932))
    return parser.parse_args()


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def planned_straight_goal(speed: float, height: float, goal_representation: str) -> np.ndarray:
    if goal_representation == "hindsight_geom_avg12":
        return np.asarray(
            [
                0.5 * speed, 0.0, 1.0 * speed, 0.0, 1.5 * speed, 0.0, 2.0 * speed, 0.0,
                0.0, 1.0, height, speed,
            ], dtype=np.float32,
        )
    return np.asarray(
        [
            0.5 * speed,
            0.0,
            1.0 * speed,
            0.0,
            1.5 * speed,
            0.0,
            2.0 * speed,
            0.0,
            height,
            0.0,
            speed,
        ],
        dtype=np.float32,
    )


def goal_support(
    train_goals: np.ndarray,
    normalizer: NormalizerStats,
    speeds: list[float],
    heights: list[float],
    goal_representation: str,
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    mean = normalizer.goal.mean
    std = normalizer.goal.std
    normalized_train = (train_goals - mean) / std
    results = []
    for speed in speeds:
        for height in heights:
            goal = planned_straight_goal(speed, height, goal_representation)
            normalized = (goal - mean) / std
            distances = np.linalg.norm(normalized_train - normalized[None, :], axis=1)
            nearest = int(np.argmin(distances))
            results.append(
                {
                    "speed_mps": float(speed),
                    "height_m": float(height),
                    "nearest_goal_z_l2": float(distances[nearest]),
                    "p01_goal_z_l2": float(np.percentile(distances, 1)),
                    "max_abs_feature_z": float(np.max(np.abs(normalized))),
                    "nearest_training_goal": {
                        name: float(train_goals[nearest, index])
                        for index, name in enumerate(feature_names)
                    },
                }
            )
    return results


def make_plot(
    output: Path,
    action_rms: np.ndarray,
    height_delta: np.ndarray,
    support: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].hist(action_rms, bins=60)
    axes[0].axvline(0.0, color="tab:red", linestyle="--", label="deployment reset")
    axes[0].set(title="Train action-history RMS", xlabel="RMS", ylabel="windows")
    axes[0].legend(fontsize=8)

    axes[1].hist(height_delta, bins=60)
    axes[1].axvline(0.03, color="tab:red", linestyle="--", label="3 cm")
    axes[1].axvline(-0.03, color="tab:red", linestyle="--")
    axes[1].set(title="2 s target-height minus current height", xlabel="delta height [m]")
    axes[1].legend(fontsize=8)

    labels = [f"v={row['speed_mps']:.1f}, h={row['height_m']:.3f}" for row in support]
    distances = [row["nearest_goal_z_l2"] for row in support]
    axes[2].bar(np.arange(len(labels)), distances)
    axes[2].set(title="Straight planned-goal nearest support", ylabel="nearest z-score L2")
    axes[2].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output / "spatial_dataset_audit.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max_samples must be positive.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_kind") not in {
        "spatial_time_preview_ddpm", "spatial_reference_path_ddpm", "spatial_hindsight_geometry_ddpm",
    }:
        raise ValueError("Checkpoint is not a supported spatial diffusion policy.")
    config = checkpoint["config"]
    dataset_cfg = config["dataset"]
    dataset = SpatialHindsightDataset(
        args.datasets,
        history=int(dataset_cfg["history"]),
        prediction_horizon=int(dataset_cfg["prediction_horizon"]),
        execution_offset=int(dataset_cfg["execution_offset"]),
        goal_horizon_steps=int(dataset_cfg["goal_horizon_steps"]),
        waypoint_time_offsets_s=tuple(dataset_cfg["waypoint_time_offsets_s"]),
        goal_source=str(dataset_cfg.get("goal_source", "achieved")),
        goal_representation=str(dataset_cfg.get("goal_representation", "path11")),
        include_padded_starts=bool(dataset_cfg.get("include_padded_starts", False)),
        startup_sample_multiplier=int(dataset_cfg.get("startup_sample_multiplier", 1)),
        symmetry_mode="none",
    )
    normalizer = NormalizerStats.from_dict(checkpoint["normalizer_stats"])
    rng = np.random.default_rng(args.seed)
    sample_count = min(int(args.max_samples), len(dataset.samples))
    sample_indices = rng.choice(len(dataset.samples), sample_count, replace=False)

    goal_representation = str(dataset_cfg.get("goal_representation", "path11"))
    feature_names = GEOMETRIC_FEATURE_NAMES if goal_representation == "hindsight_geom_avg12" else TEMPORAL_FEATURE_NAMES
    goals = np.empty((sample_count, len(feature_names)), dtype=np.float32)
    action_rms = np.empty(sample_count, dtype=np.float32)
    height_delta = np.empty(sample_count, dtype=np.float32)
    skill_boundary = np.zeros(sample_count, dtype=bool)
    for output_index, sample_index in enumerate(sample_indices):
        sample = dataset.samples[int(sample_index)]
        demo = dataset.demos[sample.demo_idx]
        goals[output_index] = dataset.goal_for_sample(int(sample_index))
        _, action_hist, _, _ = dataset._raw_sample(sample)
        action_rms[output_index] = float(torch.sqrt(torch.mean(torch.square(action_hist))).item())
        state_step = sample.anchor_step - 1
        target_step = state_step + dataset.goal_horizon_steps
        height_delta[output_index] = float(demo.root_pos_w[target_step, 2] - demo.root_pos_w[state_step, 2])
        if demo.skill_idx is not None:
            skill_boundary[output_index] = demo.skill_idx[state_step] != demo.skill_idx[target_step]

    support = goal_support(goals, normalizer, list(args.speeds), list(args.heights), goal_representation, feature_names)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "datasets": [str(Path(path).resolve()) for path in args.datasets],
        "policy_kind": checkpoint.get("policy_kind"),
        "schema": config.get("goal_schema"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "goal_source": dataset.goal_source,
        "sample_count": sample_count,
        "demos": len(dataset.demos),
        "action_history_rms": summary(action_rms),
        "zero_action_history_fraction": float(np.mean(action_rms < 1.0e-4)),
        "low_action_history_fraction_rms_lt_0p05": float(np.mean(action_rms < 0.05)),
        "target_minus_current_height_m": summary(height_delta),
        "height_transition_fraction_abs_gt_3cm": float(np.mean(np.abs(height_delta) > 0.03)),
        "skill_boundary_within_2s_fraction": float(np.mean(skill_boundary)),
        "straight_goal_support": support,
    }
    (output / "spatial_dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(output / "sampled_action_history_rms.csv", action_rms, delimiter=",", header="action_history_rms", comments="")
    make_plot(output, action_rms, height_delta, support)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
