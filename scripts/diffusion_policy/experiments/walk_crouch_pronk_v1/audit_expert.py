"""Summarize accepted demonstration support and optional physical diagnostics.

This is a data gate, not an expert survival benchmark: all read episodes have
already passed whole-episode filtering. Rejection counts are reported separately.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def summary(values):
    values = np.asarray(values).reshape(-1)
    return {key: float(value) for key, value in zip(("min", "p05", "median", "p95", "max"),
            np.percentile(values, [0, 5, 50, 95, 100]))} if len(values) else None


def audit(path):
    commands, velocities, heights, spacings, torques, margins, flights, durations = ([] for _ in range(8))
    settled_spacings, settled_flights = [], []
    with h5py.File(path, "r") as file:
        data = file["data"]
        report = {"dataset": str(path), "accepted_demos": len(data),
                  "attempted_demos": int(data.attrs["collection_attempted_demos"]),
                  "collection_survival_rate": float(data.attrs["collection_survival_rate"]),
                  "scope": "accepted whole episodes; startup included; no inference about rejected command bins"}
        for demo in data.values():
            obs = demo["obs"]
            pos = obs["root_pos_w"][:]
            quat = obs["root_quat_w"][:]
            command = obs["command_speed"][:]
            delta = np.diff(pos[:, :2], axis=0) * 50
            yaw = np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                             1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
            vx = delta[:, 0] * np.cos(yaw[:-1]) + delta[:, 1] * np.sin(yaw[:-1])
            # Exclude first 0.5 s after EACH command change for this diagnostic
            # only. No frame is removed from the imitation dataset.
            changed = np.r_[True, np.any(np.abs(np.diff(command, axis=0)) > 1e-6, axis=1)]
            age = np.arange(len(command)) - np.maximum.accumulate(np.where(changed, np.arange(len(command)), 0))
            steady = age[:-1] >= 25
            commands.extend(command[:-1, 0][steady])
            velocities.extend(vx[steady])
            heights.extend(pos[:, 2])
            if "diagnostics" in demo:
                diag = demo["diagnostics"]
                torques.extend(np.abs(diag["applied_torque"][:]).reshape(-1))
                margins.extend(diag["joint_limit_margin"][:].reshape(-1))
                if "foot_centers_b" in diag:
                    feet = diag["foot_centers_b"][:]
                    # Left-right separation within front and rear pairs. Full
                    # 3D distance is not a collision test for leg meshes.
                    spacings.extend(np.linalg.norm(feet[:, [0, 2]] - feet[:, [1, 3]], axis=-1).reshape(-1))
                    settled_spacings.extend(np.linalg.norm(feet[25:, [0, 2]] - feet[25:, [1, 3]], axis=-1).reshape(-1))
                    flights.extend(diag["physical_flight"][:].reshape(-1))
                    settled_flights.extend(diag["physical_flight"][25:].reshape(-1))
                    completed = diag["completed"][:].reshape(-1).astype(bool)
                    durations.extend(diag["flight_duration"][:].reshape(-1)[completed])
        report.update(measured_height_m=summary(heights),
                      accepted_steady_command_vx_mps=summary(commands),
                      accepted_steady_achieved_vx_mps=summary(velocities),
                      left_right_foot_center_separation_m=summary(spacings),
                      left_right_foot_center_separation_after_startup_m=summary(settled_spacings),
                      absolute_applied_torque_nm=summary(torques),
                      soft_joint_limit_margin_rad=summary(margins),
                      soft_limit_violation_fraction=float(np.mean(np.asarray(margins) < -1e-6)) if margins else None,
                      physical_flight_fraction=float(np.mean(flights)) if flights else None,
                      physical_flight_after_startup_fraction=float(np.mean(settled_flights)) if settled_flights else None,
                      confirmed_flight_duration_s=summary(durations))
    command, velocity = np.asarray(commands), np.asarray(velocities)
    report["accepted_steady_forward_bins"] = [
        {"command_bin": [low, high], "frames": int(mask.sum()),
         "achieved_vx_mps": summary(velocity[mask])}
        for low, high in zip([0.0, 0.2, 0.5, 0.8, 1.0, 1.25], [0.2, 0.5, 0.8, 1.0, 1.25, 1.51])
        for mask in [(command >= low) & (command < high)]]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = {Path(path).stem: audit(path) for path in args.datasets}
    (output / "expert_diagnostics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
