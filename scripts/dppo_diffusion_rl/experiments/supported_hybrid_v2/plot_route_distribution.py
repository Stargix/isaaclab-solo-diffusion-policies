#!/usr/bin/env python3
"""Render the candidate v2 geometry before enabling it for DPPO training."""

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

# Running this file directly places only its experiment directory on sys.path.
# Add the repository root so the documented command is shell-independent.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dppo_diffusion_rl.hybrid_routes import (
    SupportedHybridRouteBank,
    sample_waypoint_geometry,
)


TOP_LABELS = ("smooth total", "rounded waypoints", "hard waypoints")
DISPLAY_LABELS = (
    "smooth v1",
    "coherent smooth",
    "rounded waypoints",
    "hard waypoints",
)
DISPLAY_COLORS = ("#2878B5", "#46A378", "#F39C34", "#D64550")


def _display_mask(bank: SupportedHybridRouteBank, display_class: int) -> torch.Tensor:
    if display_class < 2:
        return (bank.geometry_class == bank.SMOOTH) & (
            bank.smooth_subclass == display_class
        )
    return bank.geometry_class == (display_class - 1)


def _active_turns(bank: SupportedHybridRouteBank, mask: torch.Tensor) -> np.ndarray:
    turns = bank.waypoint_turn_angles[mask].cpu()
    counts = bank.waypoint_segment_count[mask].cpu()
    active = torch.arange(turns.shape[1])[None, :] < (counts - 1)[:, None]
    return torch.rad2deg(turns[active].abs()).numpy()


def _gallery(bank: SupportedHybridRouteBank, output: Path, examples: int) -> None:
    fig, axes = plt.subplots(4, examples, figsize=(2.65 * examples, 10.0), squeeze=False)
    for display_class, (label, color) in enumerate(zip(DISPLAY_LABELS, DISPLAY_COLORS)):
        mask = _display_mask(bank, display_class)
        ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
        for column, env_id in enumerate(ids[:examples].tolist()):
            ax = axes[display_class, column]
            xy = bank.xy[env_id].cpu().numpy()
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.3)
            ax.scatter(xy[0, 0], xy[0, 1], s=28, color="#35A854", zorder=3)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=34, color="#222222", marker="x", zorder=3)
            if display_class >= 2:
                lengths = bank.waypoint_segment_lengths[env_id]
                count = int(bank.waypoint_segment_count[env_id])
                boundaries = torch.cumsum(lengths[:count], dim=0)[:-1]
                indices = torch.searchsorted(bank.arc[env_id], boundaries).cpu().numpy()
                ax.scatter(
                    xy[indices, 0], xy[indices, 1], s=18, color=color,
                    edgecolors="white", linewidths=0.5, zorder=4,
                )
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.22)
            ax.set_xlabel("x [m]")
            if column == 0:
                ax.set_ylabel(f"{label}\ny [m]")
            else:
                ax.set_yticklabels([])
    fig.suptitle("Candidate supported_hybrid_v2 — independent 4 m route draws", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _coverage(bank: SupportedHybridRouteBank, output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.6))
    for display_class, (label, color) in enumerate(zip(DISPLAY_LABELS, DISPLAY_COLORS)):
        mask = _display_mask(bank, display_class)
        for xy in bank.xy[mask][:: max(1, int(mask.sum()) // 90)].cpu().numpy():
            axes[0].plot(xy[:, 0], xy[:, 1], color=color, alpha=0.10, linewidth=0.8)
        endpoint = bank.xy[mask, -1].cpu().numpy()
        axes[1].scatter(endpoint[:, 0], endpoint[:, 1], s=8, alpha=0.45, color=color, label=label)
    axes[0].set_title("Route support (subsample)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(alpha=0.2)
    axes[1].set_title("Terminal-position support")
    axes[1].set_xlabel("terminal x [m]")
    axes[1].set_ylabel("terminal y [m]")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    bins = np.linspace(15.0, 90.0, 16)
    for route_class, display_class in ((bank.ROUNDED, 2), (bank.HARD, 3)):
        mask = bank.geometry_class == route_class
        axes[2].hist(
            _active_turns(bank, mask), bins=bins, alpha=0.58,
            color=DISPLAY_COLORS[display_class], label=DISPLAY_LABELS[display_class], density=True,
        )
    axes[2].axvline(90.0, color="#222222", linestyle="--", linewidth=1.0)
    axes[2].set_title("Sampled waypoint turn magnitudes")
    axes[2].set_xlabel("absolute turn [deg]")
    axes[2].set_ylabel("density")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.2)

    curvature_bins = np.linspace(0.0, 0.8, 17)
    for subclass in (bank.SMOOTH_V1, bank.SMOOTH_COHERENT):
        mask = (bank.geometry_class == bank.SMOOTH) & (bank.smooth_subclass == subclass)
        axes[3].hist(
            bank.maximum_abs_curvature[mask].cpu().numpy(),
            bins=curvature_bins,
            alpha=0.60,
            color=DISPLAY_COLORS[subclass],
            label=DISPLAY_LABELS[subclass],
            density=True,
        )
    axes[3].axvline(0.8, color="#222222", linestyle="--", linewidth=1.0)
    axes[3].set_title("Realized smooth peak curvature")
    axes[3].set_xlabel("max |curvature| [rad/m]")
    axes[3].set_ylabel("density")
    axes[3].legend(frameon=False)
    axes[3].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _paired_corner_realizations(output: Path, *, seed: int, examples: int = 4) -> None:
    """Compare hard and rounded realizations of identical waypoint parameters."""

    torch.manual_seed(seed)
    rounded = sample_waypoint_geometry(examples, "cpu", rounded=True)
    torch.manual_seed(seed)
    hard = sample_waypoint_geometry(examples, "cpu", rounded=False)
    fig, axes = plt.subplots(2, examples, figsize=(3.25 * examples, 6.1), squeeze=False)
    for index in range(examples):
        rounded_xy = rounded.xy[index].numpy()
        hard_xy = hard.xy[index].numpy()
        axes[0, index].plot(
            hard_xy[:, 0], hard_xy[:, 1], color=DISPLAY_COLORS[3], linewidth=1.8,
            linestyle="--", label="hard",
        )
        axes[0, index].plot(
            rounded_xy[:, 0], rounded_xy[:, 1], color=DISPLAY_COLORS[2], linewidth=2.3,
            label="rounded",
        )
        axes[0, index].scatter(0.0, 0.0, s=25, color="#35A854", zorder=3)
        axes[0, index].set_aspect("equal", adjustable="datalim")
        axes[0, index].grid(alpha=0.2)
        axes[0, index].set_title(f"same sampled route {index + 1}")
        axes[0, index].set_xlabel("x [m]")
        if index == 0:
            axes[0, index].set_ylabel("y [m]")
            axes[0, index].legend(frameon=False)

        arc = rounded.arc[index].numpy()
        axes[1, index].plot(
            arc, np.rad2deg(hard.yaw[index].numpy()), color=DISPLAY_COLORS[3],
            linewidth=1.8, linestyle="--", label="hard",
        )
        axes[1, index].plot(
            arc, np.rad2deg(rounded.yaw[index].numpy()), color=DISPLAY_COLORS[2],
            linewidth=2.3, label="rounded",
        )
        axes[1, index].set_xlabel("arc length [m]")
        axes[1, index].grid(alpha=0.2)
        if index == 0:
            axes[1, index].set_ylabel("path heading [deg]")
    fig.suptitle("Identical waypoint samples: corner realization only", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "route_generation")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.samples < 12 or args.examples < 1:
        raise ValueError("Use at least 12 samples and one gallery example.")

    torch.manual_seed(args.seed)
    bank = SupportedHybridRouteBank(args.samples, "cpu")
    bank.reset(torch.arange(args.samples), stratified=True, speed_max=0.8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _gallery(bank, args.output_dir / "route_gallery.png", args.examples)
    _coverage(bank, args.output_dir / "route_coverage.png")
    _paired_corner_realizations(
        args.output_dir / "paired_corner_realizations.png", seed=args.seed + 1
    )

    counts = torch.bincount(bank.geometry_class, minlength=3)
    profile_geometry_counts = torch.stack(
        [
            torch.bincount(bank.geometry_class[bank.profile_class == profile], minlength=3)
            for profile in range(4)
        ]
    )
    waypoint_mask = bank.geometry_class != bank.SMOOTH
    turn_values = _active_turns(bank, waypoint_mask)
    summary = {
        "seed": args.seed,
        "samples": args.samples,
        "geometry_counts": {TOP_LABELS[i]: int(counts[i]) for i in range(3)},
        "smooth_subclass_counts": {
            DISPLAY_LABELS[i]: int((bank.smooth_subclass == i).sum()) for i in range(2)
        },
        "profile_x_geometry_counts": {
            str(profile): {TOP_LABELS[i]: int(profile_geometry_counts[profile, i]) for i in range(3)}
            for profile in range(4)
        },
        "route_length_m": float(bank.length_m),
        "endpoint_distance_m": {
            "min": float(torch.linalg.vector_norm(bank.xy[:, -1], dim=1).min()),
            "median": float(torch.linalg.vector_norm(bank.xy[:, -1], dim=1).median()),
            "max": float(torch.linalg.vector_norm(bank.xy[:, -1], dim=1).max()),
        },
        "waypoint_turn_abs_deg": {
            "min": float(turn_values.min()),
            "median": float(np.median(turn_values)),
            "max": float(turn_values.max()),
        },
        "waypoint_max_abs_heading_deg": float(
            torch.rad2deg(bank.yaw[waypoint_mask].abs()).max()
        ),
        "waypoint_segment_length_m": {
            "min": float(
                bank.waypoint_segment_lengths[bank.waypoint_segment_lengths > 0].min()
            ),
            "max": float(bank.waypoint_segment_lengths.max()),
        },
        "waypoint_segment_count": {
            str(value): int((bank.waypoint_segment_count[waypoint_mask] == value).sum())
            for value in range(2, 6)
        },
        "coherent_target_peak_curvature_rad_m": {
            "min": float(
                bank.coherent_target_peak_curvature[bank.smooth_subclass == bank.SMOOTH_COHERENT].min()
            ),
            "median": float(
                bank.coherent_target_peak_curvature[bank.smooth_subclass == bank.SMOOTH_COHERENT].median()
            ),
            "max": float(
                bank.coherent_target_peak_curvature[bank.smooth_subclass == bank.SMOOTH_COHERENT].max()
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
