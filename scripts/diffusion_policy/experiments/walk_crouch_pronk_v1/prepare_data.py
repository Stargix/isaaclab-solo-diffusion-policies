"""Reproduce the frozen temporal protocol without launching a training run.

Run from the repository root with env_isaaclab Python. Each simulator runs in a
fresh process. Existing outputs are never overwritten. A linked merged index
avoids duplicating the ~1 GB of demonstrations; its sources must travel with it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import h5py

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "scripts/diffusion_policy/data/datasets"
OUT = DATA / "walk_crouch_pronk_v1_aligned"
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_command(skill, output, frames, envs, *, diagnostics=False, self_collisions=False):
    p, s = PROTOCOL, PROTOCOL["sources"][skill]
    command = [sys.executable, "-u", "scripts/diffusion_policy/data/collect_data.py",
               "--task", s["task"], "--mode", "single", "--skill_name", skill,
               "--checkpoint", s["checkpoint"], "--desired_base_height", str(s["height_m"]),
               "--route_profile", "phase_a", "--include_warmup_frames", "--warmup_steps", "0",
               "--startup_hold_steps", str(p["startup_hold_steps"]),
               "--command_resample_time_s", str(p["command_resample_s"]),
               "--phase_a_stop_probability", str(p["stop_probability"]),
               "--phase_a_stop_hold_s", str(p["stop_hold_s"]),
               "--physics_dr_mode", "light", "--fall_gravity_z", "-0.8660254037844386",
               "--num_envs", str(envs), "--num_steps", str(frames), "--seed", str(s["seed"]),
               "--output_name", str(output), "--max_sim_steps", "30000", "--headless", "--device", "cuda:0"]
    for name, value in {
        "forward_speed_min": s["forward_min"], "forward_speed_max": s["forward_max"],
        "arc_forward_speed_max": s["arc_max"], "reverse_speed_abs_max": s["reverse_max"],
        "lateral_speed_abs_max": s["lateral_max"], "lateral_forward_speed_min": s["lateral_forward_min"],
        "lateral_forward_speed_max": s["lateral_forward_max"], "yaw_rate_abs_max": s["yaw_max"]
    }.items():
        command.extend(["--phase_a_" + name, str(value)])
    if diagnostics:
        command.append("--record_expert_diagnostics")
    if self_collisions:
        command.append("--check_self_collisions")
    return command


def validate_temporal(path, skill, frames):
    with h5py.File(path, "r") as file:
        data = file["data"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in data.attrs["skill_names"]]
        if names != [skill]:
            raise ValueError(f"{path}: wrong expert names {names}")
        if skill != "crouch" and data.attrs.get("policy_input_timing") != "synchronous_pre_action_v2":
            raise ValueError(f"{path}: stale policy-input timing; do not train on this collection")
        lengths = [len(demo["actions"]) for demo in data.values()]
        if not lengths or set(lengths) != {299} or sum(lengths) < frames:
            raise ValueError(f"{path}: incomplete/wrong-duration collection ({sum(lengths)} frames).")
        cfg = json.loads(data.attrs["route_generation_config_json"])
        params = cfg["phase_a_command_distribution"]
        if (float(data.attrs["control_rate_hz"]) != 50 or
            params["command_resample_time_s"] != 3 or params["stop_probability"] != 0.1 or
            params["stop_hold_s"] != 2):
            raise ValueError(f"{path}: mismatched temporal protocol: {params}")
        if data.attrs["collection_saved_demos"] != len(lengths):
            raise ValueError(f"{path}: no completed collection statistics.")
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                "demos": len(lengths), "frames": sum(lengths),
                "valid_2s_windows": sum(n - 100 for n in lengths),
                "padded_training_anchors": sum(n - 100 + 1 for n in lengths),
                "survival_rate": float(data.attrs["collection_survival_rate"]),
                "attempted_demos": int(data.attrs["collection_attempted_demos"])}


def run_collection(skill, suffix, frames, envs, *, self_collisions=False):
    output = OUT / f"{skill}_{suffix}.hdf5"
    if output.exists():
        raise FileExistsError(f"Refusing to rerun over {output}; inspect the existing result.")
    required_bytes = frames * (550 if suffix != "full" else 400) + 256 * 1024 * 1024
    if shutil.disk_usage(ROOT).free < required_bytes:
        raise RuntimeError(f"Not enough free disk for {skill}/{suffix} plus a 256 MiB safety reserve.")
    source = PROTOCOL["sources"][skill]
    checkpoint = ROOT / source["checkpoint"]
    digest = sha256(checkpoint)
    if source.get("checkpoint_sha256", digest) != digest:
        raise ValueError(f"{skill}: expert checkpoint hash changed")
    command = collection_command(skill, output, frames, envs, diagnostics=suffix != "full",
                                 self_collisions=self_collisions)
    collector_digest = sha256(ROOT / "scripts/diffusion_policy/data/collect_data.py")
    logs = ROOT / "scratch/walk_crouch_pronk_v1/aligned"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(ROOT / "source" / name) for name in
                                              ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "isaaclab_rl")) + os.pathsep + environment.get("PYTHONPATH", "")
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"Collecting {skill}/{suffix}; log: {logs / (skill + '_' + suffix + '.log')}", flush=True)
    with (logs / f"{skill}_{suffix}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Collection exited {result.returncode}; inspect log, do not reuse partial data.")
    report = validate_temporal(output, skill, frames)
    report.update(expert_checkpoint=source["checkpoint"], expert_sha256=digest,
                  collector_sha256=collector_digest,
                  self_collisions=self_collisions, command=command[2:])
    (logs / f"{skill}_{suffix}_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def merge_and_verify():
    sys.path.insert(0, str(ROOT / "scripts/diffusion_policy/data"))
    from merge_datasets import merge_hdf5_files
    repair_new_files("full")
    sources = [OUT / "walk_full.hdf5", ROOT / PROTOCOL["sources"]["crouch"]["reuse"], OUT / "pronk_full.hdf5"]
    report = {skill: validate_temporal(path, skill, PROTOCOL["target_frames_per_source"])
              for skill, path in zip(("walk", "crouch", "pronk"), sources)}
    for skill in ("walk", "pronk"):
        provenance = ROOT / f"scratch/walk_crouch_pronk_v1/aligned/{skill}_full_provenance.json"
        recorded = json.loads(provenance.read_text(encoding="utf-8"))
        recovery = json.loads((OUT / f"{skill}_full.reset_repair.json").read_text(encoding="utf-8"))
        report[skill].update(
            expert_checkpoint=recorded["expert_checkpoint"],
            expert_sha256=recorded["expert_sha256"],
            collection_command=recorded["command"],
            collector_sha256_during_collection=recorded["collector_sha256"],
            clean_reset_row_recovery=recovery,
        )
    if len({item["valid_2s_windows"] for item in report.values()}) != 1:
        raise ValueError("Accepted hindsight windows are not balanced.")
    merged = OUT / "walk_crouch_pronk_linked.hdf5"
    merge_hdf5_files([str(path) for path in sources], str(merged), link_data=True)
    manifest = {"protocol_sha256": sha256(HERE / "protocol.json"), "sources": report,
                "merged": {"path": merged.relative_to(ROOT).as_posix(), "sha256": sha256(merged)},
                "collector_code_sha256_at_finalization": sha256(ROOT / "scripts/diffusion_policy/data/collect_data.py"),
                "train_config_sha256": sha256(ROOT / PROTOCOL["train_config"])}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def repair_new_files(suffix):
    from reset_row_repair import repair_dataset
    for skill in (("pronk",) if suffix == "collision_pilot" else ("walk", "pronk")):
        path = OUT / f"{skill}_{suffix}.hdf5"
        provenance = ROOT / f"scratch/walk_crouch_pronk_v1/aligned/{skill}_{suffix}_provenance.json"
        recorded = json.loads(provenance.read_text(encoding="utf-8"))
        report = repair_dataset(path, expected_original_sha256=recorded["sha256"])
        print(f"{skill}/{suffix}: {len(report['modified_demo_names'])} exact reset-row recoveries", flush=True)


def verify_artifacts():
    """Read-only preflight, usable on a cluster without expert checkpoints."""
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["sources"]) != {"walk", "crouch", "pronk"}:
        raise ValueError("Expected exactly walk/crouch/pronk sources; no sprint fallback.")
    for item in [*manifest["sources"].values(), manifest["merged"]]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {item['path']}")
    if sha256(HERE / "protocol.json") != manifest["protocol_sha256"] or sha256(ROOT / PROTOCOL["train_config"]) != manifest["train_config_sha256"]:
        raise ValueError("Protocol or training config changed after data preparation.")
    counts = {(item["demos"], item["frames"], item["valid_2s_windows"])
              for item in manifest["sources"].values()}
    if counts != {(3345, 1000155, 665655)}:
        raise ValueError("Expected the frozen balanced full collection.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "collision_pilot", "full", "merge", "verify"), required=True)
    parser.add_argument("--skill", choices=("walk", "pronk"), default=None)
    args = parser.parse_args()
    if args.stage == "merge":
        merge_and_verify()
    elif args.stage == "verify":
        verify_artifacts()
        print("All prepared artifact hashes match.")
    else:
        skills = [args.skill] if args.skill else ["walk", "pronk"]
        if args.stage == "collision_pilot":
            skills = ["pronk"]
        for skill in skills:
            run_collection(skill, args.stage, 1000000 if args.stage == "full" else
                           4784 if args.stage == "collision_pilot" else 29900,
                           256 if args.stage == "full" else 32 if args.stage == "collision_pilot" else 128,
                           self_collisions=args.stage == "collision_pilot")


if __name__ == "__main__":
    main()
