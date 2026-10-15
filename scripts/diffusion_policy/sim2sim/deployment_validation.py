"""Run the fixed development bank; preserve every case and fail on missing output."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .scenarios import load_manifest
from .perturbations import PROTOCOLS


def execute(args):
    manifest, _ = load_manifest(args.manifest)
    if manifest["schema"] != 3:
        raise ValueError("Requires the predeclared deployment bank")
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    for variant in PROTOCOLS:
        parts = ("constants", "transitions") if args.engine == "isaac" else (None,)
        for part in parts:
            name = f"deployment_{args.engine}_{part+'_' if part else ''}{variant}_20261014"
            output = root/name; expected = output/("evaluation_summary.json" if part else "summary.json")
            if expected.is_file():
                print(f"Preserving completed {name}", flush=True); continue
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Incomplete existing run; inspect rather than overwrite: {output}")
            if part:
                command = [sys.executable, "-u", "scripts/diffusion_policy/evaluate_policy.py", "--checkpoint", args.checkpoint,
                    "--output_dir", str(output), "--route_bank", str(Path(args.manifest).parent/"routes.npz"),
                    "--path_shapes", "straight", "coherent_smooth", "rounded_waypoint", "ood_corner", "--speeds", "0.35", "0.65",
                    "--route_length_m", "4", "--duration_s", "24", "--repeats", "1", "--seed", "42",
                    "--warmup_steps", "25", "--posture_settle_steps", "50", "--profile_transition_margin_m", "0.25",
                    "--record_staging_failures", "--deterministic_resets", "--save_timeseries", "--require_empty_output_dir",
                    "--deployment_sampler", "graph", "--deployment_perturbation", variant, "--headless", "--device", "cuda:0"]
                command += ["--path_heights", "0.2932", "0.1705"] if part == "constants" else ["--transition_fractions", "0.5", "--transition_directions", "both"]
            else:
                command = [sys.executable, "-u", "-m", "scripts.diffusion_policy.sim2sim.run", "--checkpoint", args.checkpoint,
                    "--model", args.model, "--validation", args.validation, "--manifest", args.manifest,
                    "--dependency-dir", args.dependency_dir, "--output-dir", str(output), "--sampler", "graph", "--perturbation", variant]
            (root/f"{name}_command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            print(f"Starting {name}", flush=True)
            with (root/f"{name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            if completed.returncode != 0 or not expected.is_file():
                raise RuntimeError(f"Run did not produce verified completion: {name}, exit={completed.returncode}")
            print(f"Completed {name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("isaac", "mujoco"), required=True)
    for key in ("checkpoint", "manifest", "model", "validation", "dependency-dir"):
        p.add_argument(f"--{key}", required=True)
    p.add_argument("--root", default="scratch/wc_deployment")
    execute(p.parse_args())


if __name__ == "__main__":
    main()
