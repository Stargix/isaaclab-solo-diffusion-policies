#!/usr/bin/env python3
"""Preflight validation for a Phase-A fixed-reference route dataset.

The tool verifies the collection contract before spending a GPU train: reference
poses must integrate the saved nominal body-frame command, starts must contain
a held command, and a closed-loop dataset must keep the nominal route separate
from the route-tracker command consumed by the expert.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def yaw_wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def expected_reference_delta(command: np.ndarray, yaw: np.ndarray, dt: float) -> np.ndarray:
    vx, vy = command[:, 0], command[:, 1]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return dt * np.stack((cos_yaw * vx - sin_yaw * vy, sin_yaw * vx + cos_yaw * vy), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--startup_hold_steps", type=int, default=25)
    args = parser.parse_args()

    if args.startup_hold_steps < 0:
        raise ValueError("--startup_hold_steps must be non-negative.")
    path = Path(args.dataset)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_commands: list[np.ndarray] = []
    all_teacher_commands: list[np.ndarray] = []
    all_tracking_errors: list[np.ndarray] = []
    integration_errors: list[np.ndarray] = []
    yaw_errors: list[np.ndarray] = []
    start_command_norms: list[float] = []
    startup_hold_norms: list[np.ndarray] = []
    demo_lengths: list[int] = []
    with h5py.File(path, "r") as file:
        data = file["data"]
        route_raw = data.attrs.get("route_profile", "")
        schema_raw = data.attrs.get("reference_schema", "")
        route_profile = route_raw.decode() if isinstance(route_raw, bytes) else str(route_raw)
        reference_schema = schema_raw.decode() if isinstance(schema_raw, bytes) else str(schema_raw)
        skill_names = [item.decode() if isinstance(item, bytes) else str(item) for item in data.attrs["skill_names"]]
        if len(skill_names) != 1:
            raise ValueError(f"Phase A currently requires one expert, got skill_names={skill_names}.")
        for demo_name in sorted(data.keys()):
            demo = data[demo_name]
            obs = demo["obs"]
            required = ("reference_pos_w", "reference_yaw_w", "reference_command", "command_speed")
            missing = [key for key in required if key not in obs]
            if missing:
                raise KeyError(f"{demo_name}: missing {missing}; recollect with --route_profile phase_a.")
            reference_pos = obs["reference_pos_w"][:].astype(np.float32)
            reference_yaw = obs["reference_yaw_w"][:].astype(np.float32).reshape(-1)
            command = obs["reference_command"][:].astype(np.float32)
            recorded_command = obs["command_speed"][:].astype(np.float32)
            is_closed_loop = route_profile == "phase_a_closed_loop"
            if not is_closed_loop and not np.allclose(command, recorded_command, atol=1.0e-6, rtol=0.0):
                raise ValueError(f"{demo_name}: legacy reference_command and command_speed differ.")
            if len(reference_pos) < 2:
                continue
            dt = 0.02
            delta = reference_pos[1:, :2] - reference_pos[:-1, :2]
            expected = expected_reference_delta(command[:-1], reference_yaw[:-1], dt)
            integration_errors.append(np.linalg.norm(delta - expected, axis=-1))
            yaw_errors.append(np.abs(yaw_wrap(reference_yaw[1:] - reference_yaw[:-1] - dt * command[:-1, 2])))
            all_commands.append(command)
            all_teacher_commands.append(recorded_command)
            start_command_norms.append(float(np.linalg.norm(command[0])))
            startup_hold_norms.append(np.linalg.norm(command[: args.startup_hold_steps], axis=-1))
            demo_lengths.append(len(command))

            if is_closed_loop:
                extra = ("reference_progress", "tracking_error_frenet", "route_family_idx")
                missing_extra = [key for key in extra if key not in obs]
                if missing_extra:
                    raise KeyError(f"{demo_name}: closed-loop Phase A is missing {missing_extra}.")
                progress = obs["reference_progress"][:].reshape(-1)
                if not np.array_equal(progress, np.arange(len(progress), dtype=progress.dtype)):
                    raise ValueError(f"{demo_name}: reference_progress must be the fixed route time index.")
                errors = obs["tracking_error_frenet"][:]
                if not np.isfinite(errors).all():
                    raise ValueError(f"{demo_name}: non-finite tracking errors.")
                all_tracking_errors.append(errors)

    if not all_commands:
        raise ValueError("Dataset has no usable demonstrations.")
    commands = np.concatenate(all_commands)
    teacher_commands = np.concatenate(all_teacher_commands)
    integration = np.concatenate(integration_errors)
    yaw_error = np.concatenate(yaw_errors)
    speed_xy = np.linalg.norm(commands[:, :2], axis=-1)
    hold_norms = np.concatenate(startup_hold_norms) if startup_hold_norms else np.zeros(1, dtype=np.float32)
    summary = {
        "dataset": str(path.resolve()),
        "route_profile": route_profile,
        "reference_schema": reference_schema,
        "demos": len(demo_lengths),
        "steps": int(sum(demo_lengths)),
        "length_p05_p50_p95": [float(np.percentile(demo_lengths, q)) for q in (5, 50, 95)],
        "start_command_norm_max": float(np.max(start_command_norms)),
        "start_command_zero_fraction": float(np.mean(np.asarray(start_command_norms) < 1.0e-6)),
        "startup_hold_zero_fraction": float(np.mean(hold_norms < 1.0e-6)),
        "stop_fraction_speed_lt_0p02": float(np.mean(speed_xy < 0.02)),
        "turn_fraction_abs_wz_gt_0p2": float(np.mean(np.abs(commands[:, 2]) > 0.2)),
        "lateral_fraction_abs_vy_gt_0p12": float(np.mean(np.abs(commands[:, 1]) > 0.12)),
        "teacher_lateral_fraction_abs_vy_gt_0p02": float(np.mean(np.abs(teacher_commands[:, 1]) > 0.02)),
        "reference_integration_error_xy_max_m": float(np.max(integration)),
        "reference_integration_error_xy_p99_m": float(np.percentile(integration, 99)),
        "reference_yaw_integration_error_max_rad": float(np.max(yaw_error)),
        "reference_yaw_integration_error_p99_rad": float(np.percentile(yaw_error, 99)),
    }
    if route_profile not in {"phase_a", "phase_a_closed_loop"}:
        raise ValueError(f"Expected a Phase-A route profile, got {route_profile!r}.")
    if summary["start_command_zero_fraction"] < 0.999:
        raise ValueError("Not every demo begins with a zero command; reset/start coverage is invalid.")
    if summary["startup_hold_zero_fraction"] < 0.999:
        raise ValueError("The recorded startup hold is not fully zero-command.")
    if summary["reference_integration_error_xy_p99_m"] > 2.0e-5:
        raise ValueError("Reference XY integration does not match the saved command.")
    if summary["reference_yaw_integration_error_p99_rad"] > 2.0e-5:
        raise ValueError("Reference yaw integration does not match the saved command.")
    if min(summary["stop_fraction_speed_lt_0p02"], summary["turn_fraction_abs_wz_gt_0p2"]) <= 0.0:
        raise ValueError("Phase A lacks stop or turn support in its nominal reference.")
    if route_profile == "phase_a_closed_loop":
        if summary["teacher_lateral_fraction_abs_vy_gt_0p02"] <= 0.0:
            raise ValueError("Closed-loop teacher never issued a lateral correction.")
        tracking_errors = np.concatenate(all_tracking_errors)
        summary["tracking_error_lateral_abs_p95_m"] = float(np.percentile(np.abs(tracking_errors[:, 1]), 95))
        summary["tracking_error_heading_abs_p95_rad"] = float(np.percentile(np.abs(tracking_errors[:, 2]), 95))

    (output / "phase_a_preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
