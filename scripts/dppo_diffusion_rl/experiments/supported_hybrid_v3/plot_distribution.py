#!/usr/bin/env python3
"""Plot the v3 transition/deadline distribution before the simulator train."""

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

from scripts.dppo_diffusion_rl.supported_hybrid_v3 import SupportedHybridV3RouteBank


PROFILE_LABELS = ("walk", "crouch", "walk->crouch", "crouch->walk")
CONTEXT_LABELS = ("uniform", "low curvature", "high curvature")
COLORS = ("#2878B5", "#D64550", "#46A378", "#F39C34")


def _transition_gallery(bank: SupportedHybridV3RouteBank, output: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 9.6), squeeze=False)
    for context in range(3):
        ids = torch.nonzero(
            bank.transition_context == context, as_tuple=False
        ).squeeze(1)[:4]
        for column, env_id in enumerate(ids.tolist()):
            ax = axes[context, column]
            xy = bank.xy[env_id].numpy()
            arc = bank.arc[env_id]
            boundary = bank.transition_boundary[env_id]
            boundary_idx = int(torch.argmin((arc - boundary).abs()))
            height = bank.height[env_id].numpy()
            segments = ax.scatter(
                xy[:, 0], xy[:, 1], c=height, cmap="viridis", s=11,
                vmin=bank.CROUCH_HEIGHT_M, vmax=bank.WALK_HEIGHT_M,
            )
            ax.plot(xy[:, 0], xy[:, 1], color="#333333", alpha=0.28, linewidth=0.8)
            ax.scatter(
                xy[boundary_idx, 0], xy[boundary_idx, 1], marker="x",
                color="#F2B134", s=60, linewidths=2.0,
            )
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
            ax.set_title(
                f"{PROFILE_LABELS[int(bank.profile_class[env_id])]}\n"
                f"mean target={float(bank.speed[env_id]):.2f} m/s",
                fontsize=9,
            )
            if column == 0:
                ax.set_ylabel(f"{CONTEXT_LABELS[context]}\ny [m]")
            ax.set_xlabel("x [m]")
    fig.colorbar(segments, ax=axes, label="required height [m]", shrink=0.78)
    fig.suptitle("supported_hybrid_v3: transition-context samples", fontsize=14)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _speed_support(bank: SupportedHybridV3RouteBank, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5))
    for profile, (label, color) in enumerate(zip(PROFILE_LABELS, COLORS)):
        mask = bank.profile_class == profile
        axes[0].hist(
            bank.speed[mask].numpy(), bins=np.linspace(0.2, 1.0, 25),
            density=True, alpha=0.48, color=color, label=label,
        )
        axes[1].scatter(
            bank.maximum_feasible_mean_speed[mask][::8].numpy(),
            bank.speed[mask][::8].numpy(), s=8, alpha=0.32, color=color, label=label,
        )
    axes[0].set_title("Sampled global mean-speed target")
    axes[0].set_xlabel("desired mean speed [m/s]")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot([0.2, 1.2], [0.2, 1.2], "--", color="#333333", linewidth=1.0)
    axes[1].set_title("Feasibility is a ceiling, not a target")
    axes[1].set_xlabel("private feasible upper bound [m/s]")
    axes[1].set_ylabel("sampled global target [m/s]")
    axes[1].grid(alpha=0.2)

    transition = bank.transition_context >= 0
    context_values = [
        bank.transition_boundary[bank.transition_context == context].numpy()
        for context in range(3)
    ]
    axes[2].hist(
        context_values, bins=np.linspace(0.8, 3.2, 25), stacked=False,
        alpha=0.55, label=CONTEXT_LABELS,
    )
    axes[2].set_title(f"Transition boundary ({int(transition.sum())} routes)")
    axes[2].set_xlabel("route arc [m]")
    axes[2].set_ylabel("count")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(value.float(), torch.tensor([0.05, 0.5, 0.95]))
    return {"p05": float(q[0]), "median": float(q[1]), "p95": float(q[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path(__file__).parent / "route_generation",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.samples < 64 or args.samples % 16 != 0:
        raise ValueError("Use a multiple of 16 with at least 64 samples.")

    torch.manual_seed(args.seed)
    bank = SupportedHybridV3RouteBank(
        args.samples, "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(args.samples), stratified=True, speed_max=1.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _transition_gallery(bank, args.output_dir / "transition_gallery.png")
    _speed_support(bank, args.output_dir / "speed_support.png")

    summary = {
        "seed": args.seed,
        "samples": args.samples,
        "geometry_counts": torch.bincount(bank.geometry_class, minlength=3).tolist(),
        "profile_counts": torch.bincount(bank.profile_class, minlength=4).tolist(),
        "transition_context_counts": torch.bincount(
            bank.transition_context[bank.transition_context >= 0], minlength=3
        ).tolist(),
        "sampled_speed_mps": _quantiles(bank.speed),
        "feasible_mean_speed_mps": _quantiles(bank.maximum_feasible_mean_speed),
        "speed_exceeds_private_ceiling": int(
            (bank.speed > bank.maximum_feasible_mean_speed + 1.0e-6).sum()
        ),
        "transition_boundary_m": _quantiles(
            bank.transition_boundary[bank.transition_context >= 0]
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
