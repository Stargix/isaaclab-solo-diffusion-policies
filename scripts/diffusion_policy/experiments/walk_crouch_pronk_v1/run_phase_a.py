"""Verified from-scratch Phase-A launch; no simulator, resume or DPPO options."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_data import ROOT, OUT, PROTOCOL, verify_artifacts

DEFAULT_OUTPUT = ROOT / "scripts/diffusion_policy/runs/walk_crouch_pronk_phase_a_v1_k10"


def training_command(output, device, dataset=None):
    return [sys.executable, "-u", str(ROOT / "scripts/diffusion_policy/train/train.py"),
            "--datasets", str(dataset if dataset is not None else OUT / "walk_crouch_pronk_linked.hdf5"),
            "--config", str(ROOT / PROTOCOL["train_config"]),
            "--output_dir", str(output), "--run_name", output.name,
            "--policy_backend", "diffusion", "--device", device,
            "--wandb_project", "solo12-diffusion-policy"]


def launch_verified(manifest, dataset, output, device, *, preflight_only=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory; refusing to mix runs in {output}")
    command = training_command(output, device, dataset)
    print("Verified: walk + crouch + pronk, balanced, 2 s hindsight, Phase A from scratch.", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if preflight_only:
        return 0
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("Refusing to train from dirty Git; preserve changes and use a clean checkout.\n" + status)
    import torch
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Phase A requires an allocated CUDA GPU; do not run on the login node.")
    provenance = {
        "experiment": manifest.get("experiment", PROTOCOL["name"]), "start_mode": "from_scratch",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": False, "python": sys.version, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "device": device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": command,
        "data_manifest": manifest,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    completed = subprocess.run(command, cwd=ROOT, env=environment)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="Read-only checks; never starts training.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    raise SystemExit(launch_verified(verify_artifacts(), OUT / "walk_crouch_pronk_linked.hdf5",
                                    args.output_dir, args.device, preflight_only=args.preflight_only))


if __name__ == "__main__":
    main()
