#!/usr/bin/env python3
"""Compare two Solo12 RSL-RL checkpoints through contact-level gait fingerprints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# On Windows, load RSL-RL (and its tensordict native extension) before Kit.
# Importing it after AppLauncher can crash CPython during module initialization.
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: F401

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--comparison",
    choices=("walk_bound", "walk_flying_trot"),
    default="walk_bound",
    help="comparison preset; flying-trot uses the compatible 48-D walk observation contract",
)
parser.add_argument("--walk_checkpoint", default="checkpoints/walk_final.pt")
parser.add_argument(
    "--bound_checkpoint", default="checkpoints_iri/checkpoints_bound/bound_v2.pt"
)
parser.add_argument(
    "--flying_trot_checkpoint",
    default="checkpoints_iri/checkpoints_bound/flying_trot.pt",
)
parser.add_argument("--speed", type=float, default=0.8)
parser.add_argument("--duration_s", type=float, default=8.0)
parser.add_argument("--warmup_s", type=float, default=2.0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--contact_threshold_n", type=float, default=1.0)
parser.add_argument(
    "--output_dir",
    default="scripts/reinforcement_learning/rsl_rl/evaluations/walk_vs_bound_v2",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


FOOT_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
FOOT_LABELS = ("FL", "FR", "RL", "RR")


def _comparison_spec() -> tuple[str, tuple[str, str], tuple[int, int]]:
    """Return task, policy labels and observation dimensions for the selected preset."""
    if args.comparison == "walk_flying_trot":
        return "solo12-flying-trot-v0", ("walk_final", "flying_trot"), (48, 48)
    return "solo12-bound-v2", ("walk_final", "bound_v2"), (48, 50)


def _configure_env():
    task, _, _ = _comparison_spec()
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 2 * args.num_envs
    cfg.sim.device = args.device or cfg.sim.device
    cfg.seed = args.seed
    cfg.episode_length_s = args.warmup_s + args.duration_s + 2.0
    cfg.terrain.terrain_type = "plane"
    cfg.terrain.terrain_generator = None
    cfg.events = None
    cfg.enable_observation_corruption = False
    cfg.actuation_delay_range = (0, 0)
    cfg.command_resampling_time_s = cfg.episode_length_s + 10.0
    cfg.standing_env_prob = 0.0
    cfg.opposite_direction_cmd_prob = 0.0
    return cfg


class _CheckpointActor(nn.Module):
    """Minimal deterministic actor matching the repository's legacy PPO MLP."""

    def __init__(self, checkpoint: Path, expected_obs_dim: int, device: torch.device):
        super().__init__()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state = payload["model_state_dict"]
        layer_indices = (0, 2, 4, 6)
        weights = [state[f"actor.{index}.weight"] for index in layer_indices]
        if weights[0].shape[1] != expected_obs_dim or weights[-1].shape[0] != 12:
            raise ValueError(
                f"Unexpected actor contract in {checkpoint}: first={tuple(weights[0].shape)}, "
                f"last={tuple(weights[-1].shape)}"
            )
        modules: list[nn.Module] = []
        for layer_number, index in enumerate(layer_indices):
            weight = state[f"actor.{index}.weight"]
            bias = state[f"actor.{index}.bias"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            modules.append(layer)
            if layer_number < len(layer_indices) - 1:
                modules.append(nn.ELU())
        self.actor = nn.Sequential(*modules)
        self.register_buffer("obs_mean", state["actor_obs_normalizer._mean"].clone())
        self.register_buffer("obs_std", state["actor_obs_normalizer._std"].clone())
        self.to(device).eval()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations - self.obs_mean) / (self.obs_std + 1.0e-2)
        return self.actor(normalized)


def _empty_records() -> dict[str, list[np.ndarray]]:
    return {
        "valid": [],
        "contacts": [],
        "lin_vel_b": [],
        "ang_vel_b": [],
        "base_height": [],
        "projected_gravity_b": [],
        "actions": [],
    }


def _rollout(checkpoints: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]:
    if args.num_envs < 1:
        raise ValueError("--num_envs must be at least one per policy.")
    task, labels, obs_dims = _comparison_spec()
    cfg = _configure_env()
    env = gym.make(task, cfg=cfg)
    raw_env = env.unwrapped
    try:
        device = torch.device(raw_env.device)
        actors = {
            label: _CheckpointActor(checkpoints[label], obs_dim, device)
            for label, obs_dim in zip(labels, obs_dims)
        }
        print(
            f"[INFO] Shared physics: {args.num_envs} {labels[0]} + "
            f"{args.num_envs} {labels[1]} environments",
            flush=True,
        )
        for label, checkpoint in checkpoints.items():
            print(f"[INFO] {label}: {checkpoint}", flush=True)

        foot_ids, resolved_names = raw_env._contact_sensor.find_bodies(
            list(FOOT_NAMES), preserve_order=True
        )
        if resolved_names != list(FOOT_NAMES):
            raise RuntimeError(f"Unexpected foot order: {resolved_names}")

        dt = float(raw_env.step_dt)
        warmup_steps = int(round(args.warmup_s / dt))
        rollout_steps = int(round(args.duration_s / dt))
        command = torch.tensor((args.speed, 0.0, 0.0), device=device)
        total_envs = 2 * args.num_envs
        group_slices = {
            labels[0]: slice(0, args.num_envs),
            labels[1]: slice(args.num_envs, total_envs),
        }
        active = {
            label: torch.ones(args.num_envs, dtype=torch.bool, device=device)
            for label in group_slices
        }
        records = {label: _empty_records() for label in group_slices}

        observations, _ = env.reset()
        raw_env._commands[:, :3] = command
        raw_env._command_steps_left.fill_(warmup_steps + rollout_steps + 10)

        for step in range(warmup_steps + rollout_steps):
            raw_env._commands[:, :3] = command
            raw_env._command_steps_left.fill_(warmup_steps + rollout_steps + 10)
            policy_obs = observations["policy"]
            with torch.inference_mode():
                actions = torch.cat(
                    (
                        actors["walk_final"](policy_obs[group_slices["walk_final"], :48]),
                        actors["bound_v2"](policy_obs[group_slices["bound_v2"], :50]),
                    ),
                    dim=0,
                )
                observations, _, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated

            if step == warmup_steps:
                for group_active in active.values():
                    group_active.fill_(True)
            if step < warmup_steps:
                continue

            forces = raw_env._contact_sensor.data.net_forces_w[:, foot_ids, :]
            contacts = torch.linalg.vector_norm(forces, dim=-1) > args.contact_threshold_n
            base_height = raw_env._robot.data.root_pos_w[:, 2] - raw_env._terrain.env_origins[:, 2]
            tensors = {
                "contacts": contacts,
                "lin_vel_b": raw_env._robot.data.root_lin_vel_b,
                "ang_vel_b": raw_env._robot.data.root_ang_vel_b,
                "base_height": base_height,
                "projected_gravity_b": raw_env._robot.data.projected_gravity_b,
                "actions": actions,
            }
            for label, group_slice in group_slices.items():
                group_dones = dones[group_slice]
                valid = active[label] & ~group_dones
                records[label]["valid"].append(valid.detach().cpu().numpy())
                for key, tensor in tensors.items():
                    records[label][key].append(tensor[group_slice].detach().cpu().numpy())
                active[label] &= ~group_dones

        results = {
            label: {key: np.asarray(value) for key, value in group_records.items()}
            for label, group_records in records.items()
        }
        for result in results.values():
            result["dt"] = np.asarray(dt)
        return results
    finally:
        env.close()


def _masked_flat(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return values[valid]


def _stance_jaccard(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    intersection = np.logical_and(a, b) & valid
    union = np.logical_or(a, b) & valid
    return float(intersection.sum() / max(1, union.sum()))


def _contact_correlation(contacts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flattened = []
    for foot_idx in range(4):
        flattened.append(contacts[..., foot_idx][valid].astype(np.float64))
    matrix = np.eye(4, dtype=np.float64)
    for row in range(4):
        for col in range(row + 1, 4):
            if flattened[row].std() < 1.0e-8 or flattened[col].std() < 1.0e-8:
                value = 0.0
            else:
                value = float(np.corrcoef(flattened[row], flattened[col])[0, 1])
            matrix[row, col] = matrix[col, row] = value
    return matrix


def _summarize(data: dict[str, np.ndarray]) -> dict[str, object]:
    valid = data["valid"].astype(bool)
    contacts = data["contacts"].astype(bool)
    lin_vel = data["lin_vel_b"]
    ang_vel = data["ang_vel_b"]

    front_same = contacts[..., 0] == contacts[..., 1]
    rear_same = contacts[..., 2] == contacts[..., 3]
    front_state = contacts[..., 0] | contacts[..., 1]
    rear_state = contacts[..., 2] | contacts[..., 3]
    bound_pattern = front_same & rear_same & (front_state != rear_state)

    diagonal_a_same = contacts[..., 0] == contacts[..., 3]
    diagonal_b_same = contacts[..., 1] == contacts[..., 2]
    diagonal_a_state = contacts[..., 0] | contacts[..., 3]
    diagonal_b_state = contacts[..., 1] | contacts[..., 2]
    trot_pattern = diagonal_a_same & diagonal_b_same & (diagonal_a_state != diagonal_b_state)
    flight = ~contacts.any(axis=-1)

    pitch_proxy = np.arcsin(np.clip(data["projected_gravity_b"][..., 0], -1.0, 1.0))
    valid_count_per_env = valid.sum(axis=0)
    representative_env = int(np.argmax(valid_count_per_env))

    summary = {
        "valid_samples": int(valid.sum()),
        "survival_rate": float(np.mean(valid[-1])) if len(valid) else 0.0,
        "representative_env": representative_env,
        "mean_forward_speed_mps": float(np.mean(_masked_flat(lin_vel[..., 0], valid))),
        "forward_speed_rmse_mps": float(
            np.sqrt(np.mean(np.square(_masked_flat(lin_vel[..., 0] - args.speed, valid))))
        ),
        "mean_abs_lateral_speed_mps": float(
            np.mean(np.abs(_masked_flat(lin_vel[..., 1], valid)))
        ),
        "mean_abs_yaw_rate_rps": float(np.mean(np.abs(_masked_flat(ang_vel[..., 2], valid)))),
        "base_height_std_m": float(np.std(_masked_flat(data["base_height"], valid))),
        "pitch_rms_rad": float(np.sqrt(np.mean(np.square(_masked_flat(pitch_proxy, valid))))),
        "flight_fraction": float(np.mean(_masked_flat(flight, valid))),
        "bound_pattern_fraction": float(np.mean(_masked_flat(bound_pattern, valid))),
        "trot_pattern_fraction": float(np.mean(_masked_flat(trot_pattern, valid))),
        "front_pair_stance_jaccard": _stance_jaccard(
            contacts[..., 0], contacts[..., 1], valid
        ),
        "rear_pair_stance_jaccard": _stance_jaccard(
            contacts[..., 2], contacts[..., 3], valid
        ),
        "fl_rr_stance_jaccard": _stance_jaccard(contacts[..., 0], contacts[..., 3], valid),
        "fr_rl_stance_jaccard": _stance_jaccard(contacts[..., 1], contacts[..., 2], valid),
        "foot_duty_factors": {
            FOOT_LABELS[idx]: float(np.mean(_masked_flat(contacts[..., idx], valid)))
            for idx in range(4)
        },
        "contact_correlation": _contact_correlation(contacts, valid).tolist(),
    }
    return summary


def _nan_time_stats(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(valid, values, np.nan)
    return np.nanmean(masked, axis=1), np.nanstd(masked, axis=1)


def _plot(
    results: dict[str, dict[str, np.ndarray]],
    summaries: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    _, labels, _ = _comparison_spec()
    colors = {labels[0]: "#2878B5", labels[1]: "#D95319"}

    for col, label in enumerate(labels):
        data = results[label]
        env_idx = int(summaries[label]["representative_env"])
        valid = data["valid"][:, env_idx].astype(bool)
        contacts = data["contacts"][:, env_idx].astype(float)
        contacts[~valid] = np.nan
        time_s = np.arange(len(contacts)) * float(data["dt"])
        axes[0, col].imshow(
            contacts.T,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(0.0, time_s[-1] if len(time_s) else 0.0, 3.5, -0.5),
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, col].set_yticks(range(4), FOOT_LABELS)
        axes[0, col].set_xlabel("time [s]")
        axes[0, col].set_title(f"{label}: foot contacts")

        correlation = np.asarray(summaries[label]["contact_correlation"])
        image = axes[1, col].imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[1, col].set_xticks(range(4), FOOT_LABELS)
        axes[1, col].set_yticks(range(4), FOOT_LABELS)
        axes[1, col].set_title(f"{label}: contact correlation")
        for row in range(4):
            for column in range(4):
                axes[1, col].text(
                    column,
                    row,
                    f"{correlation[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(correlation[row, column]) < 0.55 else "white",
                    fontsize=9,
                )

    speed_ax = axes[0, 2]
    for label in labels:
        data = results[label]
        mean, std = _nan_time_stats(data["lin_vel_b"][..., 0], data["valid"].astype(bool))
        time_s = np.arange(len(mean)) * float(data["dt"])
        speed_ax.plot(time_s, mean, label=label, color=colors[label], linewidth=2)
        speed_ax.fill_between(time_s, mean - std, mean + std, color=colors[label], alpha=0.16)
    speed_ax.axhline(args.speed, color="black", linestyle="--", label="command")
    speed_ax.set_xlabel("time [s]")
    speed_ax.set_ylabel("forward speed [m/s]")
    speed_ax.set_title("Velocity tracking")
    speed_ax.legend()
    speed_ax.grid(alpha=0.25)

    metric_ax = axes[1, 2]
    metric_names = ("bound_pattern_fraction", "trot_pattern_fraction", "flight_fraction")
    metric_labels = ("bound pattern", "trot pattern", "flight")
    x = np.arange(len(metric_names))
    width = 0.36
    for offset, label in zip((-width / 2, width / 2), labels):
        values = [float(summaries[label][name]) for name in metric_names]
        metric_ax.bar(x + offset, values, width, label=label, color=colors[label])
    metric_ax.set_xticks(x, metric_labels, rotation=15)
    metric_ax.set_ylim(0.0, 1.0)
    metric_ax.set_ylabel("fraction of valid samples")
    metric_ax.set_title("Contact-pattern fingerprint")
    metric_ax.legend()
    metric_ax.grid(axis="y", alpha=0.25)

    fig.colorbar(image, ax=axes[1, :2], shrink=0.75, label="Pearson correlation")
    fig.suptitle(
        f"Solo12 gait comparison ({args.comparison}) at {args.speed:.2f} m/s",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, labels, _ = _comparison_spec()
    second_checkpoint = (
        args.flying_trot_checkpoint
        if args.comparison == "walk_flying_trot"
        else args.bound_checkpoint
    )
    checkpoints = {
        labels[0]: Path(args.walk_checkpoint).resolve(),
        labels[1]: Path(second_checkpoint).resolve(),
    }
    results = _rollout(checkpoints)
    summaries = {label: _summarize(results[label]) for label in results}
    report = {
        "speed_command_mps": args.speed,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "num_envs": args.num_envs,
        "contact_threshold_n": args.contact_threshold_n,
        "comparison": args.comparison,
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "policies": summaries,
    }

    with (output_dir / "gait_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    np.savez_compressed(
        output_dir / "gait_timeseries.npz",
        **{
            f"{label}_{key}": value
            for label, data in results.items()
            for key, value in data.items()
        },
    )
    _plot(results, summaries, output_dir / "gait_comparison.png")

    print(json.dumps(report, indent=2), flush=True)
    print(f"[PASS] Plot: {output_dir / 'gait_comparison.png'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
