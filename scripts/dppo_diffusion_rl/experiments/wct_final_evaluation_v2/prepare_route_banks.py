#!/usr/bin/env python3
"""Materialize the independent geometry-difficulty bank for evaluation v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diffusion_policy.evaluation.route_bank import (  # noqa: E402
    DIAGNOSTIC_FAMILIES,
    generate_bank,
    load_route_bank,
    save_route_bank,
    sha256_file,
    write_manifest,
)


def _plot_gallery(bank_path: Path, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    routes = load_route_bank(bank_path)
    families = list(DIAGNOSTIC_FAMILIES)
    figure, axes = plt.subplots(1, len(families), figsize=(4.2 * len(families), 4), facecolor="white")
    for axis, family in zip(axes, families):
        for (name, _), route in routes.items():
            if name == family:
                axis.plot(route.xy[:, 0], route.xy[:, 1], color="#31688e", alpha=0.25, linewidth=0.8)
        axis.scatter(0.0, 0.0, color="#222222", s=14, zorder=3)
        axis.set_title(family.removeprefix("sweep_").replace("_", " ").title())
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_xlabel("local x [m]")
    axes[0].set_ylabel("local y [m]")
    figure.suptitle("Evaluation-v2 geometry-difficulty bank (independent routes)", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "route_banks")
    parser.add_argument("--routes_per_family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=390_771)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.routes_per_family < 10:
        raise ValueError("Use at least 10 independent routes per family.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "wct_geometry_sweep_v2.npz"
    manifest_path = output_dir / "manifest.json"
    if (bank_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite the frozen v2 route bank.")
    routes = generate_bank(
        DIAGNOSTIC_FAMILIES,
        routes_per_family=args.routes_per_family,
        base_seed=args.seed,
        split="diagnostic_geometry_v2",
    )
    entry = save_route_bank(bank_path, routes)
    entry["path"] = bank_path.name
    manifest = {
        "protocol": "wct_final_evaluation_v2",
        "purpose": "Evaluation-only geometry/speed capability boundary; never used for training.",
        "statistical_unit": "one materialized (family, repeat) route",
        "routes_per_family": args.routes_per_family,
        "seed": args.seed,
        "route_length_m": 4.0,
        "families": {
            "sweep_s_curve": "Smooth S turns: peak curvature 0.4--1.2 rad/m, 1--3 reversals.",
            "sweep_hard_turn": "One 45--135 degree hard corner.",
            "sweep_rounded_turn": "One smooth 45--135 degree turn.",
            "sweep_compound_turn": "Two independently signed smooth turns.",
        },
        "bank": entry,
    }
    write_manifest(manifest_path, manifest)
    _plot_gallery(bank_path, output_dir / "route_gallery.png")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
