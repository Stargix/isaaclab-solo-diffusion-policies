#!/usr/bin/env python3
"""Materialize the frozen ID and OOD geometry banks for the WCT benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diffusion_policy.evaluation.route_bank import (  # noqa: E402
    ID_FAMILIES,
    OOD_FAMILIES,
    generate_bank,
    save_route_bank,
    write_manifest,
)


def _plot_gallery(bank_paths: list[Path], output_path: Path) -> None:
    from scripts.diffusion_policy.evaluation.route_bank import load_route_bank

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    routes = {}
    for bank_path in bank_paths:
        routes.update(load_route_bank(bank_path))
    families = sorted({family for family, _ in routes})
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="white")
    for axis, family in zip(axes.flat, families):
        family_routes = [route for (name, _), route in routes.items() if name == family]
        for route in family_routes:
            axis.plot(route.xy[:, 0], route.xy[:, 1], linewidth=0.7, alpha=0.35)
        axis.scatter([0.0], [0.0], color="black", s=18, zorder=3)
        axis.set_title(family.replace("_", " ").title())
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_xlabel("local x [m]")
        axis.set_ylabel("local y [m]")
    for axis in axes.flat[len(families):]:
        axis.set_visible(False)
    figure.suptitle("Frozen WCT evaluation routes — all independent draws", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "route_banks",
    )
    parser.add_argument("--routes_per_family", type=int, default=50)
    parser.add_argument("--id_seed", type=int, default=160_142)
    parser.add_argument("--ood_seed", type=int, default=260_542)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    id_path = output_dir / "wct_id_routes_v1.npz"
    ood_path = output_dir / "wct_ood_routes_v1.npz"
    manifest_path = output_dir / "manifest.json"
    targets = (id_path, ood_path, manifest_path)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to replace a frozen benchmark: " + ", ".join(str(path) for path in existing)
        )

    id_routes = generate_bank(
        ID_FAMILIES,
        routes_per_family=args.routes_per_family,
        base_seed=args.id_seed,
        split="test_id",
    )
    ood_routes = generate_bank(
        OOD_FAMILIES,
        routes_per_family=args.routes_per_family,
        base_seed=args.ood_seed,
        split="test_ood",
    )
    id_entry = save_route_bank(id_path, id_routes)
    ood_entry = save_route_bank(ood_path, ood_routes)
    id_entry["path"] = id_path.name
    ood_entry["path"] = ood_path.name
    manifest = {
        "protocol": "wct_final_evaluation_v1",
        "purpose": (
            "Frozen paired test geometries. The statistical unit is (family, repeat); "
            "speeds and height profiles are repeated conditions on that unit."
        ),
        "generator": "scripts/diffusion_policy/evaluation/route_bank.py",
        "routes_per_family": args.routes_per_family,
        "route_length_m": 4.0,
        "points_per_route": 101,
        "id_seed": args.id_seed,
        "ood_seed": args.ood_seed,
        "id": id_entry,
        "ood": ood_entry,
        "interpretation": {
            "id": "Held-out draws from the four supported_hybrid_v2 train families.",
            "ood": (
                "Randomized constant arcs, higher-frequency S curves and 90--120 degree "
                "hard corners beyond the declared train envelope. Report separately from ID."
            ),
        },
    }
    write_manifest(manifest_path, manifest)
    _plot_gallery([id_path, ood_path], output_dir / "route_gallery.png")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
