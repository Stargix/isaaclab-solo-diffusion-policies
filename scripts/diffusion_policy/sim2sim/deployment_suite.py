"""Freeze a bounded paired robustness bank before executing any outcomes."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from .scenarios import load_manifest
from .convert import sha
from .perturbations import PROTOCOLS
from scripts.diffusion_policy.evaluation.route_bank import save_route_bank


def prepare(source, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    m, banks = load_manifest(source)
    families = ("straight", "coherent_smooth", "rounded_waypoint", "ood_corner")
    output.mkdir(parents=True, exist_ok=True)
    bank = save_route_bank(output / "routes.npz", [banks[(f, 0)] for f in families])
    m = dict(m, schema=3, route_bank="routes.npz", route_bank_sha256=bank["sha256"], seed=42,
             evaluation_seeds=[42], selection="fixed archived repeat0; no outcome-dependent selection",
             scope="32-condition nominal/parameter/impulse development bank; one actor and inference seed",
             conditions=[dict(id=f"{f}_{v:g}_{p}", family=f, repeat=0, speed_mps=v, profile=p)
                         for f in families for v in (.35, .65)
                         for p in ("walk", "crouch", "walk_to_crouch", "crouch_to_walk")],
             smoke_ids=["coherent_smooth_0.35_walk", "rounded_waypoint_0.35_walk_to_crouch",
                        "straight_0.65_walk", "ood_corner_0.65_crouch_to_walk"])
    m["perturbations"] = [asdict(p) for p in PROTOCOLS.values()]
    m["parent_manifest_sha256"] = sha(source)
    (output / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); print(json.dumps(prepare(a.source, a.output_dir), indent=2))


if __name__ == "__main__":
    main()
