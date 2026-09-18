#!/usr/bin/env python3
"""Generate the preregistered v5 route/speed/profile audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dppo_diffusion_rl.supported_hybrid_v5 import SupportedHybridV5RouteBank


COVERAGE_LABELS = (
    "v3 replay",
    "fast C->W",
    "fast W->C",
    "repeated W start",
    "repeated C start",
)
FAMILY_LABELS = ("smooth-v1", "coherent", "rounded", "hard")


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(values.float(), torch.tensor([0.05, 0.5, 0.95]))
    return {
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=Path(__file__).parent / "route_generation"
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.samples < 128 or args.samples % 128 != 0:
        raise ValueError("Use a multiple of 128 with at least 128 samples.")

    torch.manual_seed(args.seed)
    bank = SupportedHybridV5RouteBank(
        args.samples,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(args.samples), stratified=True, speed_max=1.0)
    family = torch.where(
        bank.geometry_class == bank.SMOOTH,
        bank.smooth_subclass,
        bank.geometry_class + 1,
    )
    coverage_counts = torch.bincount(bank.coverage_class, minlength=5)
    expected = torch.tensor(
        [args.samples // 2] + [args.samples // 8] * 4, dtype=torch.long
    )
    if not torch.equal(coverage_counts, expected):
        raise RuntimeError("V5 coverage mixture is not the preregistered 50/25/25 split.")
    cross = torch.zeros(5, 4, dtype=torch.long)
    for coverage_class in range(5):
        cross[coverage_class] = torch.bincount(
            family[bank.coverage_class == coverage_class], minlength=4
        )
    if not torch.all(cross == expected[:, None] // 4):
        raise RuntimeError("V5 coverage is correlated with geometry family.")

    fast = (bank.coverage_class == bank.FAST_CROUCH_TO_WALK) | (
        bank.coverage_class == bank.FAST_WALK_TO_CROUCH
    )
    repeated = (bank.coverage_class == bank.REPEATED_WALK_START) | (
        bank.coverage_class == bank.REPEATED_CROUCH_START
    )
    if torch.any(bank.speed[fast] > bank.maximum_feasible_mean_speed[fast] + 1.0e-6):
        raise RuntimeError("A fast target exceeds its private feasible ceiling.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.7))
    bottom = np.zeros(5)
    for route_family, label in enumerate(FAMILY_LABELS):
        values = cross[:, route_family].numpy()
        axes[0].bar(COVERAGE_LABELS, values, bottom=bottom, label=label)
        bottom += values
    axes[0].set(title="Exact task coverage", ylabel="routes")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, fontsize=8)

    bins = np.linspace(0.2, 1.0, 33)
    for coverage_class, label in enumerate(COVERAGE_LABELS):
        values = bank.speed[bank.coverage_class == coverage_class].numpy()
        axes[1].hist(values, bins=bins, alpha=0.5, label=label)
    axes[1].set(title="Global mean-speed targets", xlabel="target [m/s]", ylabel="count")
    axes[1].legend(frameon=False, fontsize=8)

    example_classes = (
        bank.FAST_CROUCH_TO_WALK,
        bank.FAST_WALK_TO_CROUCH,
        bank.REPEATED_WALK_START,
        bank.REPEATED_CROUCH_START,
    )
    for coverage_class in example_classes:
        env_id = int(torch.nonzero(bank.coverage_class == coverage_class)[0])
        axes[2].step(
            bank.arc[env_id].numpy(),
            bank.height[env_id].numpy(),
            where="post",
            label=COVERAGE_LABELS[coverage_class],
        )
    axes[2].set(
        title="Targeted height profiles",
        xlabel="route progress [m]",
        ylabel="required height [m]",
        ylim=(0.15, 0.31),
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output_dir / "supported_hybrid_v5.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "seed": args.seed,
        "samples": args.samples,
        "coverage_counts": dict(zip(COVERAGE_LABELS, coverage_counts.tolist())),
        "coverage_by_geometry": {
            COVERAGE_LABELS[row]: dict(zip(FAMILY_LABELS, cross[row].tolist()))
            for row in range(5)
        },
        "speed_mps": {
            COVERAGE_LABELS[coverage_class]: _quantiles(
                bank.speed[bank.coverage_class == coverage_class]
            )
            for coverage_class in range(5)
        },
        "fast_frontier_fraction": bank.FAST_FRONTIER_FRACTION,
        "fast_target_fraction_at_or_above": {
            "0.75_mps": float((bank.speed[fast] >= 0.75).float().mean()),
            "0.80_mps": float((bank.speed[fast] >= 0.80).float().mean()),
            "0.84_mps": float((bank.speed[fast] >= 0.84).float().mean()),
        },
        "fast_target_exceeds_private_ceiling": int(
            (bank.speed[fast] > bank.maximum_feasible_mean_speed[fast] + 1.0e-6).sum()
        ),
        "repeated_profile_boundaries": _quantiles(
            torch.isfinite(bank.profile_boundaries[repeated]).sum(dim=1).float()
        ),
        "pace_consistent_preview": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
