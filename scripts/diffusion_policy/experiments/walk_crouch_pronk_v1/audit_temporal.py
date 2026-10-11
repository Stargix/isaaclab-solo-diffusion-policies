"""Verify recorded command holds, pre-action alignment and nominal integration.

Unlike the old route preflight, this checks nonzero turns rather than requiring
wz > 0.2: pronk deliberately has a 0.2 rad/s proposal cap. No gate is disabled.
"""
import json
import sys

import h5py
import numpy as np

from prepare_data import ROOT, OUT, PROTOCOL

sys.path.insert(0, str(ROOT / "scripts/diffusion_policy/data"))
from validate_phase_a_dataset import expected_reference_delta, yaw_wrap


def audit(path, *, recorded_startup_ticks=25):
    max_xy, max_yaw, max_last_action = 0.0, 0.0, 0.0
    stop_frames, turn_frames, frames, changes = 0, 0, 0, []
    with h5py.File(path, "r") as file:
        for name, demo in file["data"].items():
            obs = demo["obs"]
            cmd = obs["command_speed"][:]
            reference_cmd = obs["reference_command"][:]
            if len(cmd) != 299 or not np.all(cmd[:recorded_startup_ticks] == 0):
                raise ValueError(f"{name}: wrong length/startup hold")
            if not np.allclose(reference_cmd, cmd, atol=1e-6, rtol=0):
                raise ValueError(f"{name}: reference differs from issued command")
            ticks = np.flatnonzero(np.any(np.diff(cmd, axis=0) != 0, axis=1)) + 1
            boundaries = np.r_[0, ticks, len(cmd)]
            for start, end in zip(boundaries[:-2], boundaries[1:-1]):
                zero = np.all(cmd[start] == 0)
                base = recorded_startup_ticks if start == 0 else 0
                period = 100 if zero else 150
                if end - start < base or (end - start - base) % period != 0:
                    raise ValueError(f"{name}: invalid hold from {start} to {end}")
            changes.extend(ticks.tolist())
            reference = obs["reference_pos_w"][:]
            yaw = obs["reference_yaw_w"][:].reshape(-1)
            max_xy = max(max_xy, float(np.linalg.norm(
                np.diff(reference[:, :2], axis=0) - expected_reference_delta(cmd[:-1], yaw[:-1], .02), axis=-1).max()))
            max_yaw = max(max_yaw, float(np.abs(yaw_wrap(np.diff(yaw) - .02 * cmd[:-1, 2])).max()))
            last, action = obs["last_action"][:], demo["actions"][:]
            if not np.all(last[0] == 0):
                raise ValueError(f"{name}: nonzero reset last_action")
            max_last_action = max(max_last_action, float(np.abs(last[1:] - action[:-1]).max()))
            frames += len(cmd)
            stop_frames += int((np.linalg.norm(cmd[:, :2], axis=-1) < .02).sum())
            turn_frames += int((np.abs(cmd[:, 2]) > 1e-6).sum())
    if max_xy > 2e-5 or max_yaw > 2e-5 or max_last_action > 1e-6:
        raise ValueError(f"Alignment/integration mismatch: {max_xy}, {max_yaw}, {max_last_action}")
    if not stop_frames or not turn_frames:
        raise ValueError("No stop/turn command support")
    return {"valid": True, "frames": frames, "configured_startup_hold_ticks": 25,
            "recorded_initial_command_hold_ticks": recorded_startup_ticks,
            "moving_hold_ticks": 150, "stop_hold_ticks": 100,
            "observed_command_change_ticks": sorted(set(changes)),
            "stop_command_frame_fraction": stop_frames / frames,
            "nonzero_turn_command_frame_fraction": turn_frames / frames,
            "max_reference_xy_integration_error_m": max_xy,
            "max_reference_yaw_integration_error_rad": max_yaw,
            "max_last_action_alignment_error": max_last_action}


if __name__ == "__main__":
    paths = {"walk": OUT / "walk_full.hdf5",
             "crouch": ROOT / PROTOCOL["sources"]["crouch"]["reuse"],
             "pronk": OUT / "pronk_full.hdf5"}
    # Frozen crouch has a one-tick command-recording lag in the historical
    # collector: its recorded changes are 26/176 instead of 25/175. Never shift
    # physical states/actions or modify this existing source to hide that fact.
    reports = {skill: audit(path, recorded_startup_ticks=26 if skill == "crouch" else 25)
               for skill, path in paths.items()}
    output = ROOT / "scripts/diffusion_policy/data/audits/walk_crouch_pronk_v1/full"
    output.mkdir(parents=True, exist_ok=True)
    (output / "temporal_alignment.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
