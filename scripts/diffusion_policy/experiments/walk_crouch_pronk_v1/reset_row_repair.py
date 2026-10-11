"""Exact recovery of a known recording-only reset-indexing error.

Only this experiment's newly collected files are eligible. Clean resets write
the SAME known safe pose with zero velocity/noise. The teacher input was already
refreshed independently of the recorded cache. No action or simulated motion is
modified, and no unknown state is interpolated. Unexpected poses abort before
writing. The original file hash and affected row names remain in provenance.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned"
SAFE_POSE = np.asarray([0, .4, -.8, 0, .4, -.8, 0, -.4, .8, 0, -.4, .8], dtype=np.float32)
ALLOWED = {"walk_full.hdf5", "pronk_full.hdf5", "walk_pilot.hdf5", "pronk_pilot.hdf5",
           "pronk_collision_pilot.hdf5"}


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def repair_dataset(path, *, expected_original_sha256):
    path = Path(path).resolve()
    if path.parent != OUT.resolve() or path.name not in ALLOWED:
        raise ValueError("Repair is restricted to the known newly collected experiment files.")
    with h5py.File(path, "r") as file:
        if "clean_reset_joint_row_recovery_json" in file["data"].attrs:
            previous = json.loads(file["data"].attrs["clean_reset_joint_row_recovery_json"])
        else:
            previous = None
    if previous is not None:
        sidecar = json.loads(path.with_suffix(".reset_repair.json").read_text(encoding="utf-8"))
        if sha256(path) != sidecar["repaired_sha256"]:
            raise ValueError("Dataset changed after recovery; do not silently certify it again.")
        return previous
    before = sha256(path)
    if before != expected_original_sha256:
        raise ValueError("Dataset changed since collection; original provenance mismatch.")
    affected = []
    with h5py.File(path, "r") as file:
        data = file["data"]
        if data.attrs.get("policy_input_timing") != "synchronous_pre_action_v2":
            raise ValueError("Teacher-input refresh must have been active; cannot recover unknown actions.")
        if int(data.attrs["collection_saved_demos"]) != len(data):
            raise ValueError("Incomplete collection; do not repair an open/partial dataset.")
        # Two exact signatures were observed: the clean safe pose and its first
        # hip value (0) broadcast into all 12 joints by paired advanced indexing.
        valid_count = 0
        for name, demo in data.items():
            obs = demo["obs"]
            q, velocity = obs["joint_pos"][0], obs["joint_vel"][0]
            if np.array_equal(q, SAFE_POSE):
                valid_count += 1
            elif np.all(q == 0) and np.all(velocity == 0):
                if not np.all(obs["command_speed"][:25] == 0) or not np.isclose(obs["root_pos_w"][0, 2], .35, atol=1e-6):
                    raise ValueError(f"{name}: not the certified clean-reset signature.")
                affected.append(name)
            else:
                raise ValueError(f"{name}: unexpected initial pose; no guessed repair permitted.")
        if not valid_count:
            raise ValueError("No intact safe resets to confirm the expected pose.")
    report = {"method": "exact_known_noise_free_safe_reset_v1", "original_sha256": before,
              "modified_demo_names": affected, "modified_rows_per_demo": [0],
              "original_joint_pos": [0.0] * 12, "restored_joint_pos": SAFE_POSE.tolist(),
              "actions_and_other_rows_modified": False, "intact_reset_rows": valid_count}
    with h5py.File(path, "r+") as file:
        for name in affected:
            file["data"][name]["obs/joint_pos"][0] = SAFE_POSE
        file["data"].attrs["clean_reset_joint_row_recovery_json"] = json.dumps(report, sort_keys=True)
    sidecar = {**report, "repaired_sha256": sha256(path)}
    path.with_suffix(".reset_repair.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return report
