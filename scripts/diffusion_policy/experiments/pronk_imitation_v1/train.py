"""Train pronk-only Phase A from the existing frozen nominal demonstrations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def budget(config, num_demos):
    from scripts.diffusion_policy.train.data.episode_split import split_episode_indices
    split = split_episode_indices(num_demos, config["val_fraction"], config["seed"])
    anchors = P["frames_per_episode"] - config["goal_horizon_steps"] + 1
    augmented_per_demo = anchors * 4  # pinned quadruped augmentation
    train_samples = len(split.train_demo_indices) * augmented_per_demo
    steps = math.ceil(train_samples / config["batch_size"])
    return dict(train_episodes=len(split.train_demo_indices), validation_episodes=len(split.val_demo_indices),
                unique_anchors=num_demos * anchors, samples_after_symmetry=num_demos * augmented_per_demo,
                train_samples=train_samples, steps_per_epoch=steps,
                epochs=config["epochs"], total_updates=steps * config["epochs"],
                lr_warmup_updates=config["lr_warmup_steps"])


def audit():
    data_path, config_path = ROOT / P["dataset"], ROOT / P["base_config"]
    for path, expected in ((data_path, P["dataset_sha256"]), (config_path, P["base_config_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Frozen artifact mismatch: {path}. No fallback or automatic repair.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from scripts.diffusion_policy.train.data.dataset import _validate_hdf5
    _validate_hdf5(str(data_path), require_reference_path=False, require_desired_height_profile=True)
    amplitudes = []
    with h5py.File(data_path, "r") as file:
        data = file["data"]
        if list(data.attrs["skill_names"]) != [b"pronk"]:
            raise ValueError("This experiment accepts only pronk, not a mixed index.")
        if len(data) != P["episodes"] or data.attrs.get("policy_input_timing") != "synchronous_pre_action_v2":
            raise ValueError("Wrong demonstration count or observation/action timing.")
        for demo in data.values():
            if len(demo["actions"]) != P["frames_per_episode"] or "training_anchor_steps" in demo:
                raise ValueError("Use full nominal episodes, without speed or impulse-window selection.")
            if np.any(demo["skill_idx"][:] != 0):
                raise ValueError("Unexpected second skill.")
            if not np.all(demo["obs/desired_base_height"][:] == np.float32(P["nominal_height_m"])):
                raise ValueError("Height labels differ from the frozen nominal upright annotation.")
            height = demo["obs/root_pos_w"][25:, 2]
            amplitudes.append(float(np.percentile(height, 95) - np.percentile(height, 5)))
    result = dict(experiment=P["name"], start_mode="from_scratch", source=P,
                  budget=budget(config, P["episodes"]),
                  median_episode_height_p95_minus_p05_m=float(np.median(amplitudes)),
                  warning="Height oscillation is descriptive, not verified flight. Data retain only surviving episodes.")
    if result["budget"]["total_updates"] <= config["lr_warmup_steps"]:
        raise ValueError("Training would not complete its LR warmup.")
    return result


def training_command(output, device, wandb=False):
    command = [sys.executable, "-u", str(ROOT / "scripts/diffusion_policy/train/train.py"),
               "--datasets", str(ROOT / P["dataset"]), "--config", str(ROOT / P["base_config"]),
               "--output_dir", str(output), "--run_name", output.name, "--device", device,
               "--policy_backend", "diffusion", "--seed", str(P["seed"])]
    if wandb:
        command += ["--wandb_project", "solo12-diffusion-policy"]
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    report = audit()
    output = ROOT / P["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to mix or overwrite runs: {output}")
    command = training_command(output, args.device, args.wandb)
    print(json.dumps(report, indent=2), flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if args.preflight_only:
        return
    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU, not the login node.")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("Use a clean research checkout; preserve unrelated changes, do not reset them.\n" + dirty)
    report.update(command=command, git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), git_dirty=False,
        torch_version=torch.__version__, python=sys.version, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    # Describe the actual trainer/model source, not just an approximate branch.
    files = [*sorted((ROOT / "scripts/diffusion_policy/train").rglob("*.py")),
             *sorted((ROOT / "scripts/diffusion_policy/model").rglob("*.py")), HERE / "train.py", HERE / "protocol.json"]
    report["source_hashes"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
