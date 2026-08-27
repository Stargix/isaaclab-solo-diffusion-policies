#!/usr/bin/env python3
"""Audit raw and hindsight support of a merged multi-skill diffusion dataset.

This tool is deliberately independent of policy training. It reads the aligned
HDF5 contract, samples frames and exact two-second geometric-hindsight goals,
and writes a JSON report plus two plots. It never modifies its input dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT))

from train.conditioning.geometry import cumulative_xy_lengths  # noqa: E402
from train.conditioning.goal_builder import build_goal_vector  # noqa: E402


EXPECTED_CONVENTION_PREFIX = (
    "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]"
)
REQUIRED_OBS_KEYS = (
    "joint_pos",
    "joint_vel",
    "base_ang_vel",
    "projected_gravity",
    "last_action",
    "root_pos_w",
    "root_quat_w",
    "command_speed",
    "desired_base_height",
)
COLORS = {
    "walk": "#2878B5",
    "crouch": "#59A14F",
    "sprint": "#E15759",
}


def _decode(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_")[-1]), name
    except ValueError:
        return 0, name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _summary(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = (quaternion[..., index] for index in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _body_velocity(root: np.ndarray, quaternion: np.ndarray, dt: float) -> np.ndarray:
    velocity_w = np.zeros((len(root), 2), dtype=np.float32)
    if len(root) > 1:
        velocity_w[0] = (root[1, :2] - root[0, :2]) / dt
        velocity_w[-1] = (root[-1, :2] - root[-2, :2]) / dt
    if len(root) > 2:
        velocity_w[1:-1] = (root[2:, :2] - root[:-2, :2]) / (2.0 * dt)
    yaw = _yaw_from_wxyz(quaternion)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.stack(
        (
            cosine * velocity_w[:, 0] + sine * velocity_w[:, 1],
            -sine * velocity_w[:, 0] + cosine * velocity_w[:, 1],
        ),
        axis=-1,
    )


def _selected_offsets(total: int, maximum: int, rng: np.random.Generator) -> np.ndarray:
    count = min(total, maximum)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(total, count, replace=False).astype(np.int64))


def _append(store: dict[str, list[np.ndarray]], key: str, values: np.ndarray) -> None:
    if len(values):
        store[key].append(np.asarray(values))


def _concat(store: dict[str, list[np.ndarray]], key: str, width: int | None = None) -> np.ndarray:
    values = store.get(key, [])
    if values:
        return np.concatenate(values, axis=0)
    return np.empty((0, width), dtype=np.float32) if width is not None else np.empty(0, dtype=np.float32)


def audit(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    dt = 1.0 / args.control_rate_hz
    errors: list[str] = []
    demo_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    demo_info: list[tuple[str, str, int, int]] = []

    with h5py.File(dataset_path, "r") as file:
        if "data" not in file:
            raise KeyError("Dataset has no 'data' group.")
        data = file["data"]
        skill_names = [_decode(value) for value in data.attrs.get("skill_names", [])]
        attrs = {key: _decode(value) for key, value in data.attrs.items() if np.ndim(value) == 0}
        if attrs.get("condition_schema") != "velocity_xyyaw_plus_desired_base_height_v1":
            errors.append("unexpected condition_schema")
        if not attrs.get("convention", "").startswith(EXPECTED_CONVENTION_PREFIX):
            errors.append("unexpected or missing alignment convention")
        if not np.isclose(float(attrs.get("control_rate_hz", "nan")), args.control_rate_hz):
            errors.append("unexpected control rate")
        names = sorted(data.keys(), key=_demo_key)
        for name in names:
            demo = data[name]
            if "obs" not in demo:
                errors.append(f"{name}: missing obs group")
                continue
            obs = demo["obs"]
            missing = [key for key in REQUIRED_OBS_KEYS if key not in obs]
            if missing:
                errors.append(f"{name}: missing obs keys {missing}")
                continue
            if any(key not in demo for key in ("actions", "dones", "skill_idx")):
                errors.append(f"{name}: missing action/done/skill_idx field")
                continue
            length = int(obs["joint_pos"].shape[0])
            indices = demo["skill_idx"][:].astype(np.int64)
            unique = np.unique(indices)
            if len(unique) != 1:
                errors.append(f"{name}: contains a skill transition")
                continue
            skill_id = int(unique[0])
            if not 0 <= skill_id < len(skill_names):
                errors.append(f"{name}: skill_idx {skill_id} outside skill_names")
                continue
            skill = skill_names[skill_id]
            field_lengths = [obs[key].shape[0] for key in REQUIRED_OBS_KEYS]
            field_lengths += [demo[key].shape[0] for key in ("actions", "dones", "skill_idx")]
            if any(value != length for value in field_lengths):
                errors.append(f"{name}: inconsistent field lengths")
                continue
            dones = demo["dones"][:].astype(bool)
            if length == 0 or not dones[-1] or np.any(dones[:-1]):
                errors.append(f"{name}: invalid terminal done convention")
                continue
            demo_counts[skill] += 1
            frame_counts[skill] += length
            valid_windows = max(0, length - args.goal_horizon_steps)
            window_counts[skill] += valid_windows
            demo_info.append((name, skill, length, valid_windows))

    expected = list(args.expected_skills) if args.expected_skills else skill_names
    if skill_names != expected:
        errors.append(f"skill_names {skill_names} != expected {expected}")

    rng = np.random.default_rng(args.seed)
    selected_frames = {
        skill: _selected_offsets(frame_counts[skill], args.max_frames_per_skill, rng)
        for skill in skill_names
    }
    selected_windows = {
        skill: _selected_offsets(window_counts[skill], args.max_windows_per_skill, rng)
        for skill in skill_names
    }
    frame_cursor = Counter()
    window_cursor = Counter()
    sampled: dict[str, dict[str, list[np.ndarray]]] = {
        skill: defaultdict(list) for skill in skill_names
    }

    with h5py.File(dataset_path, "r") as file:
        data = file["data"]
        for name, skill, length, valid_windows in demo_info:
            demo = data[name]
            obs = demo["obs"]
            frame_start = frame_cursor[skill]
            frame_stop = frame_start + length
            global_frames = selected_frames[skill]
            local_frames = global_frames[(global_frames >= frame_start) & (global_frames < frame_stop)] - frame_start

            window_start = window_cursor[skill]
            window_stop = window_start + valid_windows
            global_windows = selected_windows[skill]
            local_windows = global_windows[
                (global_windows >= window_start) & (global_windows < window_stop)
            ] - window_start

            if len(local_frames) or len(local_windows):
                root = obs["root_pos_w"][:].astype(np.float32)
                quaternion = obs["root_quat_w"][:].astype(np.float32)
                if not np.isfinite(root).all() or not np.isfinite(quaternion).all():
                    errors.append(f"{name}: non-finite root pose")
                    continue
            else:
                root = quaternion = None

            if len(local_frames):
                command = obs["command_speed"][:].astype(np.float32)
                projected_gravity = obs["projected_gravity"][:].astype(np.float32)
                actions = demo["actions"][:].astype(np.float32)
                joint_pos = obs["joint_pos"][:].astype(np.float32)
                if not all(np.isfinite(value).all() for value in (command, projected_gravity, actions, joint_pos)):
                    errors.append(f"{name}: non-finite frame data")
                    continue
                body_velocity = _body_velocity(root, quaternion, dt)
                idx = local_frames.astype(np.int64)
                _append(sampled[skill], "command", command[idx])
                _append(sampled[skill], "body_velocity", body_velocity[idx])
                _append(sampled[skill], "height", root[idx, 2])
                _append(sampled[skill], "desired_height", obs["desired_base_height"][idx, 0])
                _append(sampled[skill], "projected_gravity", projected_gravity[idx])
                _append(sampled[skill], "actions", actions[idx])
                _append(sampled[skill], "joint_pos", joint_pos[idx])
                valid_delta = idx[idx > 0]
                if len(valid_delta):
                    _append(sampled[skill], "action_delta", np.linalg.norm(actions[valid_delta] - actions[valid_delta - 1], axis=1))

            if len(local_windows):
                cumulative = cumulative_xy_lengths(root)
                goals = []
                for anchor in local_windows.astype(np.int64):
                    end = int(anchor + args.goal_horizon_steps)
                    goals.append(
                        build_goal_vector(
                            root,
                            cumulative,
                            int(anchor),
                            end,
                            root[anchor],
                            quaternion[anchor],
                            quat_w=quaternion,
                            dt=dt,
                            goal_representation="hindsight_geom_avg12",
                        )
                    )
                _append(sampled[skill], "goals", np.asarray(goals, dtype=np.float32))

            frame_cursor[skill] = frame_stop
            window_cursor[skill] = window_stop

    arrays: dict[str, dict[str, np.ndarray]] = {}
    skill_summaries = {}
    for skill in skill_names:
        values = sampled[skill]
        arrays[skill] = {
            "command": _concat(values, "command", 3),
            "body_velocity": _concat(values, "body_velocity", 2),
            "height": _concat(values, "height"),
            "desired_height": _concat(values, "desired_height"),
            "projected_gravity": _concat(values, "projected_gravity", 3),
            "actions": _concat(values, "actions", 12),
            "joint_pos": _concat(values, "joint_pos", 12),
            "action_delta": _concat(values, "action_delta"),
            "goals": _concat(values, "goals", 12),
        }
        item = arrays[skill]
        goals = item["goals"]
        command = item["command"]
        velocity = item["body_velocity"]
        skill_summaries[skill] = {
            "demos": int(demo_counts[skill]),
            "frames": int(frame_counts[skill]),
            "valid_two_second_windows": int(window_counts[skill]),
            "sampled_frames": int(len(command)),
            "sampled_windows": int(len(goals)),
            "command_vx_mps": _summary(command[:, 0]),
            "command_vy_mps": _summary(command[:, 1]),
            "command_wz_rps": _summary(command[:, 2]),
            "achieved_vx_mps": _summary(velocity[:, 0]),
            "achieved_vy_mps": _summary(velocity[:, 1]),
            "instantaneous_vx_error_mps": _summary(velocity[:, 0] - command[:, 0]),
            "measured_height_m": _summary(item["height"]),
            "desired_height_m": _summary(item["desired_height"]),
            "action_delta_l2": _summary(item["action_delta"]),
            "hindsight_average_speed_mps": _summary(goals[:, 11]),
            "hindsight_terminal_height_m": _summary(goals[:, 10]),
            "hindsight_terminal_distance_m": _summary(np.linalg.norm(goals[:, 6:8], axis=1)),
        }

    nonzero_frames = [frame_counts[skill] for skill in skill_names if frame_counts[skill] > 0]
    frame_balance = float(min(nonzero_frames) / max(nonzero_frames)) if nonzero_frames else 0.0
    report = {
        "dataset": str(dataset_path),
        "bytes": dataset_path.stat().st_size,
        "sha256": _sha256(dataset_path),
        "dataset_attrs": attrs,
        "expected_skills": expected,
        "skill_names": skill_names,
        "contract_valid": not errors,
        "contract_errors": errors,
        "frame_balance_min_over_max": frame_balance,
        "control_rate_hz": args.control_rate_hz,
        "goal_horizon_steps": args.goal_horizon_steps,
        "goal_horizon_s": args.goal_horizon_steps / args.control_rate_hz,
        "skills": skill_summaries,
    }
    return report, arrays


def make_plots(
    report: dict[str, object],
    arrays: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    skills = list(arrays)
    colors = {skill: COLORS.get(skill, f"C{index}") for index, skill in enumerate(skills)}

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for skill in skills:
        item = arrays[skill]
        color = colors[skill]
        axes[0, 0].hist(item["command"][:, 0], bins=55, density=True, histtype="step", linewidth=2, color=color, label=skill)
        axes[0, 1].hist(item["height"], bins=55, density=True, histtype="step", linewidth=2, color=color, label=skill)
        count = min(5000, len(item["command"]))
        axes[0, 2].scatter(item["command"][:count, 0], item["body_velocity"][:count, 0], s=3, alpha=0.12, color=color, label=skill)
        axes[1, 0].scatter(item["command"][:count, 1], item["command"][:count, 2], s=3, alpha=0.15, color=color, label=skill)
        axes[1, 2].hist(item["action_delta"], bins=55, density=True, histtype="step", linewidth=2, color=color, label=skill)

    all_actions = np.concatenate([arrays[skill]["actions"] for skill in skills], axis=0)
    action_mean = np.mean(all_actions, axis=0, keepdims=True)
    covariance = np.cov(all_actions - action_mean, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, -2:]
    for skill in skills:
        actions = arrays[skill]["actions"]
        count = min(8000, len(actions))
        projection = (actions[:count] - action_mean) @ basis
        axes[1, 1].scatter(projection[:, 0], projection[:, 1], s=3, alpha=0.12, color=colors[skill], label=skill)

    axes[0, 0].set(title="Commanded forward-speed support", xlabel="command vx [m/s]", ylabel="density")
    axes[0, 1].set(title="Measured posture support", xlabel="base height [m]", ylabel="density")
    axes[0, 2].plot((-1.0, 1.7), (-1.0, 1.7), "k--", linewidth=1)
    axes[0, 2].set(title="Instantaneous velocity response", xlabel="command vx [m/s]", ylabel="achieved vx [m/s]")
    axes[1, 0].set(title="Steering-command support", xlabel="command vy [m/s]", ylabel="command wz [rad/s]")
    axes[1, 1].set(title="Expert-action manifold (PCA)", xlabel="action PC1", ylabel="action PC2")
    axes[1, 2].set(title="Action temporal smoothness", xlabel="||a[t]-a[t-1]||", ylabel="density")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Raw multi-skill dataset audit", fontsize=16)
    figure.savefig(output / "raw_multiskill_support.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for skill in skills:
        goals = arrays[skill]["goals"]
        color = colors[skill]
        axes[0, 0].hist(goals[:, 11], bins=55, density=True, histtype="step", linewidth=2, color=color, label=skill)
        count = min(8000, len(goals))
        axes[0, 1].scatter(goals[:count, 11], goals[:count, 10], s=4, alpha=0.18, color=color, label=skill)
        axes[1, 0].scatter(goals[:count, 6], goals[:count, 7], s=4, alpha=0.14, color=color, label=skill)
    counts = [int(report["skills"][skill]["frames"]) for skill in skills]
    axes[1, 1].bar(skills, counts, color=[colors[skill] for skill in skills])
    axes[0, 0].set(title="Exact 2 s hindsight speed", xlabel="average achieved path speed [m/s]", ylabel="density")
    axes[0, 1].set(title="Condition identifiability", xlabel="average achieved path speed [m/s]", ylabel="terminal height [m]")
    axes[1, 0].set(title="Terminal XY support in current body frame", xlabel="terminal x [m]", ylabel="terminal y [m]")
    axes[1, 0].axis("equal")
    axes[1, 1].set(title="Frames per skill", ylabel="frames")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if axis is not axes[1, 1]:
            axis.legend(fontsize=8)
    figure.suptitle("Training-condition support before diffusion", fontsize=16)
    figure.savefig(output / "hindsight_multiskill_support.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_skills", nargs="+", default=None)
    parser.add_argument("--max_frames_per_skill", type=int, default=30_000)
    parser.add_argument("--max_windows_per_skill", type=int, default=20_000)
    parser.add_argument("--goal_horizon_steps", type=int, default=100)
    parser.add_argument("--control_rate_hz", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    if args.max_frames_per_skill < 1 or args.max_windows_per_skill < 1:
        raise ValueError("Sampling limits must be positive.")
    if args.goal_horizon_steps < 1 or args.control_rate_hz <= 0.0:
        raise ValueError("Goal horizon and control rate must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report, arrays = audit(args)
    (output / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_plots(report, arrays, output)
    print(json.dumps({
        "contract_valid": report["contract_valid"],
        "frame_balance_min_over_max": report["frame_balance_min_over_max"],
        "skills": report["skills"],
        "output_dir": str(output),
    }, indent=2))
    if not report["contract_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
