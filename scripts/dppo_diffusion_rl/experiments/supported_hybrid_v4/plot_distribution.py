#!/usr/bin/env python3
"""Audit v4's private feasibility and fast-task sampling before training."""

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

from scripts.dppo_diffusion_rl.supported_hybrid_v4 import SupportedHybridV4RouteBank


PROFILE_LABELS = ("walk", "crouch", "walk->crouch", "crouch->walk")
FAMILY_LABELS = ("smooth-v1", "coherent", "rounded", "hard")


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(value.float(), torch.tensor([0.05, 0.5, 0.95]))
    return {"p05": float(q[0]), "median": float(q[1]), "p95": float(q[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default=Path(__file__).parent / "route_generation"
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.samples < 64 or args.samples % 32 != 0:
        raise ValueError("Use a multiple of 32 with at least 64 samples.")

    torch.manual_seed(args.seed)
    bank = SupportedHybridV4RouteBank(
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
    cell = family * 8 + bank.profile_class * 2 + bank.pace_tier
    cell_counts = torch.bincount(cell, minlength=32)
    ceiling = bank.maximum_feasible_mean_speed.clamp(max=1.0).clamp_min(0.2)
    if not torch.all(cell_counts == args.samples // 32):
        raise RuntimeError("The 32-cell family/profile/pace cross is not balanced.")
    if torch.any(bank.speed > ceiling + 1.0e-6):
        raise RuntimeError("A sampled speed exceeds its private feasible ceiling.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6))
    bins = np.linspace(0.2, 1.0, 25)
    for profile, label in enumerate(PROFILE_LABELS):
        mask = bank.profile_class == profile
        axes[0].hist(bank.speed[mask].numpy(), bins=bins, alpha=0.45, label=label)
    axes[0].set(title="Global mean-speed targets", xlabel="target [m/s]", ylabel="count")
    axes[0].legend(frameon=False, fontsize=8)

    for tier, label in enumerate(("full feasible", "upper feasible quartile")):
        mask = bank.pace_tier == tier
        axes[1].scatter(
            ceiling[mask][::8].numpy(), bank.speed[mask][::8].numpy(),
            s=7, alpha=0.3, label=label,
        )
    axes[1].plot([0.2, 1.0], [0.2, 1.0], "--", color="#333333")
    axes[1].set(
        title="Ceiling is private, target is global",
        xlabel="private ceiling [m/s]", ylabel="sampled target [m/s]",
    )
    axes[1].legend(frameon=False, fontsize=8)

    fast_fraction = []
    for route_family in range(4):
        mask = family == route_family
        fast_fraction.append(float((bank.speed[mask] > 0.65).float().mean()))
    axes[2].bar(FAMILY_LABELS, fast_fraction, color="#2878B5")
    axes[2].set(title="Fast coverage by family", ylabel="fraction target > 0.65 m/s", ylim=(0, 1))
    axes[2].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "speed_support_v4.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "seed": args.seed,
        "samples": args.samples,
        "cell_counts_min_max": [int(cell_counts.min()), int(cell_counts.max())],
        "pace_tier_counts": torch.bincount(bank.pace_tier, minlength=2).tolist(),
        "sampled_speed_mps": _quantiles(bank.speed),
        "feasible_ceiling_mps": _quantiles(ceiling),
        "fraction_speed_above_0_65": float((bank.speed > 0.65).float().mean()),
        "fraction_speed_above_0_80": float((bank.speed > 0.80).float().mean()),
        "fast_fraction_by_family": dict(zip(FAMILY_LABELS, fast_fraction)),
        "speed_exceeds_private_ceiling": int((bank.speed > ceiling + 1.0e-6).sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
