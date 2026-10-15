"""Freeze four existing geometries and a small paired transfer development bank."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.diffusion_policy.evaluation.route_bank import MaterializedRoute, load_route_bank, save_route_bank
from .convert import sha


def prepare(output, *, paper=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[3]
    banks = root / "scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/route_banks"
    id_path = banks / "wct_id_routes_v1.npz"; ood_path = banks / "wct_ood_routes_v1.npz"
    ordinary = load_route_bank(id_path); stress = load_route_bank(ood_path)
    xy = np.column_stack((np.linspace(0, 4, 101), np.zeros(101))).astype(np.float32)
    routes = [MaterializedRoute("straight", 0, 42, "transfer_development", xy, np.zeros(101, np.float32)),
              ordinary[("coherent_smooth", 0)], ordinary[("rounded_waypoint", 0)], stress[("ood_arc", 0)]]
    speeds = (.2, .35)
    if paper:
        routes = [routes[0]] + [ordinary[(f, 0)] for f in
                  ("procedural", "coherent_smooth", "rounded_waypoint", "hard_waypoint")] + [
                  stress[(f, 0)] for f in ("ood_arc", "ood_s_curve", "ood_corner")]
        speeds = (.25, .5, .85)
    output.mkdir(parents=True, exist_ok=True)
    bank = save_route_bank(output / "routes.npz", routes)
    conditions = []
    # Order within each suite matches the existing evaluator's Cartesian product.
    for route in routes:
        for speed in speeds:
            for profile in ("walk", "crouch", "walk_to_crouch", "crouch_to_walk"):
                conditions.append({"id": f"{route.family}_{speed:g}_{profile}", "family": route.family,
                                   "repeat": 0, "speed_mps": speed, "profile": profile})
    result = {"schema": 2 if paper else 1, "route_bank": "routes.npz", "route_bank_sha256": bank["sha256"],
              "source_banks": {str(id_path.relative_to(root)): sha(id_path),
                               str(ood_path.relative_to(root)): sha(ood_path)},
              "selection": "repeat 0 per declared family, fixed before results; straight is explicit 4 m",
              "conditions": conditions, "seed": 42, "dt_s": .02, "duration_s": 24.,
              "warmup_steps": 25, "posture_settle_steps": 50,
              "walk_height_m": .2932, "crouch_height_m": .1705, "transition_fraction": .5,
              "profile_transition_margin_m": .25, "scope": f"{len(conditions)}-condition paired development transfer bank, not sealed holdout",
              "smoke_ids": ["straight_0.2_walk", "coherent_smooth_0.35_walk",
                            "rounded_waypoint_0.2_crouch", "ood_arc_0.35_crouch_to_walk"]}
    if paper:
        result["evaluation_seeds"] = [42, 43, 44]
        result["smoke_ids"] = ["coherent_smooth_0.5_walk", "rounded_waypoint_0.5_walk_to_crouch",
                               "ood_s_curve_0.5_walk", "ood_corner_0.85_crouch_to_walk"]
        result["reporting"] = "Report every attempt; high-speed crouch/corners are stress, not guaranteed feasible."
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    bank = path.parent / manifest["route_bank"]
    if sha(bank) != manifest["route_bank_sha256"]:
        raise ValueError("Transfer geometry bank was changed")
    conditions = manifest["conditions"]
    count = {1: 32, 2: 96, 3: 32}.get(manifest["schema"])
    if count is None or len(conditions) != count or len({c["id"] for c in conditions}) != count:
        raise ValueError("Unexpected number of unique predeclared conditions")
    if manifest["dt_s"] != .02 or manifest["duration_s"] != 24.:
        raise ValueError("Transfer timing contract differs")
    return manifest, load_route_bank(bank)


def path_for(condition, route, manifest):
    xy = route.xy.copy(); arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    walk, crouch = manifest["walk_height_m"], manifest["crouch_height_m"]
    profile = condition["profile"]
    if profile in ("walk", "crouch"):
        height = np.full(len(xy), walk if profile == "walk" else crouch)
    else:
        first, last = (walk, crouch) if profile == "walk_to_crouch" else (crouch, walk)
        height = np.where(arc < arc[-1] * manifest["transition_fraction"], first, last)
    return np.column_stack((xy, height)).astype(np.float32), arc.astype(np.float32), route.yaw.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, paper=args.paper), indent=2))


if __name__ == "__main__":
    main()
