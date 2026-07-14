#!/usr/bin/env python3
"""Vectorized closed-loop evaluation of the frozen velocity-height baseline."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RSL_RL_DIR = _PROJECT_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate the frozen DiffuseLoco velocity-height baseline.")
parser.add_argument("--task", type=str, default="solo12-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--heights", type=float, nargs="+", default=(0.1705, 0.20, 0.23, 0.26, 0.2932))
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    action="append",
    metavar=("VX", "VY", "WZ"),
    help="Repeat to override the default three-command grid.",
)
parser.add_argument("--repeats", type=int, default=3, help="Stochastic DDPM repeats per grid condition.")
parser.add_argument("--duration_s", type=float, default=8.0)
parser.add_argument("--settling_s", type=float, default=2.0)
parser.add_argument("--warmup_steps", type=int, default=25)
parser.add_argument("--dynamic_segment_s", type=float, default=2.0)
parser.add_argument("--skip_dynamic", action="store_true")
parser.add_argument("--num_inference_steps", type=int, default=None)
parser.add_argument("--exec_horizon", type=int, default=1)
parser.add_argument("--guidance_scale", type=float, default=1.0)
parser.add_argument("--latency_samples", type=int, default=100)
parser.add_argument(
    "--torchscript_denoiser",
    action="store_true",
    help="Trace the denoiser to reduce Windows inference overhead without changing the checkpoint.",
)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.reporting import (
    STATIC_COLUMNS,
    plot_dynamic_height,
    plot_latency,
    plot_static_height,
    plot_static_velocity,
    write_csv,
    write_json,
)
from evaluation.optimization import trace_denoiser
from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
from train.config import resolve_inference_steps
from train.data.obs_utils import proprio_from_env_tensors
from train.runtime.checkpoint import load_training_checkpoint


DEFAULT_COMMANDS = ((0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (0.3, 0.15, 0.2))
DYNAMIC_HEIGHTS = (0.2932, 0.23, 0.20, 0.1705, 0.23, 0.2932)


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    repeat: int
    command: tuple[float, float, float]
    height: float


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_config(config: dict[str, Any], inference_steps: int) -> Solo12DiffusionPolicyConfig:
    model = config["model"]
    dataset = config["dataset"]
    diffusion = config["diffusion"]
    return Solo12DiffusionPolicyConfig(
        proprio_dim=model["proprio_dim"],
        action_hist_dim=model["action_hist_dim"],
        goal_dim=model["goal_dim"],
        action_dim=model["action_dim"],
        history=dataset["history"],
        prediction_horizon=dataset["prediction_horizon"],
        execution_offset=dataset["execution_offset"],
        d_model=model["d_model"],
        nhead=model["nhead"],
        num_layers=model["num_layers"],
        p_drop_emb=model["p_drop_emb"],
        p_drop_attn=model["p_drop_attn"],
        separate_goal_conditioning=model["separate_goal_conditioning"],
        num_train_timesteps=diffusion["num_train_timesteps"],
        beta_start=diffusion["beta_start"],
        beta_end=diffusion["beta_end"],
        beta_schedule=diffusion["beta_schedule"],
        prediction_type=diffusion["prediction_type"],
        variance_type=diffusion["variance_type"],
        clip_sample=diffusion["clip_sample"],
        num_inference_steps=inference_steps,
        cfg_dropout_prob=diffusion["cfg_dropout_prob"],
    )


def get_proprio(raw_env: Any, joint_ids: slice) -> torch.Tensor:
    robot = raw_env._robot
    return proprio_from_env_tensors(
        robot.data.joint_pos[:, joint_ids],
        robot.data.joint_vel[:, joint_ids],
        robot.data.root_ang_vel_b,
        robot.data.projected_gravity_b,
    )


def slide_buffer(buffer: torch.Tensor, value: torch.Tensor) -> None:
    buffer[:, :-1] = buffer[:, 1:].clone()
    buffer[:, -1] = value


def update_history(
    proprio_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
    command_buffer: torch.Tensor,
    proprio: torch.Tensor,
    previous_action: torch.Tensor,
    command: torch.Tensor,
) -> None:
    slide_buffer(proprio_buffer, proprio)
    slide_buffer(action_buffer, previous_action)
    slide_buffer(command_buffer, command)


def reset_buffers(
    raw_env: Any,
    joint_ids: slice,
    proprio_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
    command_buffer: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    proprio = get_proprio(raw_env, joint_ids)
    proprio_buffer[:] = proprio[:, None, :]
    action_buffer.zero_()
    command_buffer[:] = command[:, None, :]
    return torch.zeros((proprio.shape[0], 12), device=proprio.device)


def set_env_commands(raw_env: Any, velocity: torch.Tensor) -> None:
    if hasattr(raw_env, "_commands"):
        raw_env._commands[:, :3] = velocity


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"samples": 0, "mean_ms": math.nan, "p50_ms": math.nan, "p95_ms": math.nan, "p99_ms": math.nan}
    return {
        "samples": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
    }


def benchmark_single_env_latency(
    policy: Solo12DiffusionPolicy,
    proprio_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
    command_buffer: torch.Tensor,
    samples: int,
    guidance_scale: float,
) -> list[float]:
    if samples <= 0:
        return []
    device = proprio_buffer.device
    p = proprio_buffer[:1]
    a = action_buffer[:1]
    g = command_buffer[:1]
    for _ in range(5):
        policy.predict_action_denormalized(p, a, g, guidance_scale=guidance_scale)
    synchronize(device)
    latencies = []
    for _ in range(samples):
        start = time.perf_counter()
        policy.predict_action_denormalized(p, a, g, guidance_scale=guidance_scale)
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def summarize_static(
    scenarios: list[Scenario],
    arrays: dict[str, np.ndarray],
    valid: np.ndarray,
    failures: np.ndarray,
    failure_step: np.ndarray,
    settling_steps: int,
    dt: float,
) -> list[dict[str, Any]]:
    rows = []
    for env_idx, scenario in enumerate(scenarios):
        mask = valid[:, env_idx].copy()
        mask[:settling_steps] = False

        def selected(name: str) -> np.ndarray:
            return arrays[name][:, env_idx][mask]

        vx, vy, wz, height = (selected(name) for name in ("vx", "vy", "wz", "height"))
        tilt = selected("tilt_deg")
        action_delta = selected("action_delta")

        def mean(value: np.ndarray) -> float:
            return float(np.mean(value)) if value.size else math.nan

        def std(value: np.ndarray) -> float:
            return float(np.std(value)) if value.size else math.nan

        def rmse(value: np.ndarray, target: float) -> float:
            return float(np.sqrt(np.mean(np.square(value - target)))) if value.size else math.nan

        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "repeat": scenario.repeat,
                "requested_vx": scenario.command[0],
                "requested_vy": scenario.command[1],
                "requested_wz": scenario.command[2],
                "requested_height": scenario.height,
                "samples": int(mask.sum()),
                "survived": not bool(failures[env_idx]),
                "time_to_failure_s": (
                    float(failure_step[env_idx] * dt) if failures[env_idx] else math.nan
                ),
                "achieved_vx_mean": mean(vx),
                "achieved_vy_mean": mean(vy),
                "achieved_wz_mean": mean(wz),
                "achieved_height_mean": mean(height),
                "achieved_height_std": std(height),
                "vx_rmse": rmse(vx, scenario.command[0]),
                "vy_rmse": rmse(vy, scenario.command[1]),
                "wz_rmse": rmse(wz, scenario.command[2]),
                "height_rmse": rmse(height, scenario.height),
                "tilt_rms_deg": float(np.sqrt(np.mean(np.square(tilt)))) if tilt.size else math.nan,
                "tilt_max_deg": float(np.max(tilt)) if tilt.size else math.nan,
                "action_delta_rms": (
                    float(np.sqrt(np.mean(np.square(action_delta)))) if action_delta.size else math.nan
                ),
            }
        )
    return rows


def make_scenarios() -> list[Scenario]:
    commands = tuple(tuple(float(value) for value in command) for command in (args_cli.command or DEFAULT_COMMANDS))
    if args_cli.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    scenarios = []
    for command in commands:
        for height in args_cli.heights:
            for repeat in range(args_cli.repeats):
                scenarios.append(Scenario(len(scenarios), repeat, command, float(height)))
    return scenarios


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    scenarios = make_scenarios()
    num_envs = len(scenarios)
    checkpoint_path = Path(args_cli.checkpoint).resolve()
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else (
        Path(__file__).resolve().parent / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_training_checkpoint(
        checkpoint_path, device, expected_policy_kind="diffuseloco_velocity_height_ddpm"
    )
    config = checkpoint["config"]
    inference_steps = resolve_inference_steps(args_cli.num_inference_steps, config["diffusion"])
    policy_cfg = model_config(config, inference_steps)
    if policy_cfg.goal_dim != 4:
        raise ValueError(f"Expected goal_dim=4, got {policy_cfg.goal_dim}.")
    future_horizon = policy_cfg.prediction_horizon - policy_cfg.execution_offset
    if not 1 <= args_cli.exec_horizon <= future_horizon:
        raise ValueError(f"exec_horizon must be in [1, {future_horizon}].")

    policy = Solo12DiffusionPolicy(policy_cfg)
    policy.load_state_dict(checkpoint["ema_model_state_dict"])
    policy.set_normalizer_stats(checkpoint["normalizer_stats"])
    policy.to(device).eval()
    if args_cli.torchscript_denoiser:
        trace_denoiser(policy, device)

    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = 1.0e9
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed
    if getattr(env_cfg, "events", None):
        env_cfg.events = None
    if hasattr(env_cfg, "enable_observation_corruption"):
        env_cfg.enable_observation_corruption = False
    if hasattr(env_cfg, "actuation_delay_range"):
        env_cfg.actuation_delay_range = (0, 0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    joint_ids = raw_env._joint_ids
    dt = float(env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt)
    duration_steps = max(1, int(round(args_cli.duration_s / dt)))
    settling_steps = max(0, int(round(args_cli.settling_s / dt)))

    velocity = torch.tensor([scenario.command for scenario in scenarios], dtype=torch.float32, device=device)
    heights = torch.tensor([[scenario.height] for scenario in scenarios], dtype=torch.float32, device=device)
    command = torch.cat([velocity, heights], dim=-1)
    set_env_commands(raw_env, velocity)

    proprio_buffer = torch.zeros((num_envs, policy_cfg.history, policy_cfg.proprio_dim), device=device)
    action_buffer = torch.zeros((num_envs, policy_cfg.history, policy_cfg.action_hist_dim), device=device)
    command_buffer = command[:, None, :].repeat(1, policy_cfg.history, 1)
    previous_action = torch.zeros((num_envs, policy_cfg.action_dim), device=device)
    stand_action = torch.zeros_like(previous_action)

    vec_env.reset()
    reset_buffers(raw_env, joint_ids, proprio_buffer, action_buffer, command_buffer, command)
    for _ in range(max(args_cli.warmup_steps, policy_cfg.history + 1)):
        proprio = get_proprio(raw_env, joint_ids)
        update_history(proprio_buffer, action_buffer, command_buffer, proprio, previous_action, command)
        vec_env.step(stand_action)
        previous_action = stand_action.clone()

    single_latency = benchmark_single_env_latency(
        policy,
        proprio_buffer,
        action_buffer,
        command_buffer,
        args_cli.latency_samples,
        args_cli.guidance_scale,
    )

    traces = {name: [] for name in ("vx", "vy", "wz", "height", "tilt_deg", "action_delta")}
    valid_trace = []
    failures = np.zeros(num_envs, dtype=bool)
    failure_step = np.full(num_envs, -1, dtype=np.int64)
    current_chunk: torch.Tensor | None = None
    chunk_index = 0
    batch_latencies = []
    previous_for_delta = previous_action.clone()
    static_start = time.perf_counter()

    for step in range(duration_steps):
        with torch.inference_mode():
            if current_chunk is None or chunk_index == 0:
                synchronize(device)
                infer_start = time.perf_counter()
                trajectory = policy.predict_action_denormalized(
                    proprio_buffer,
                    action_buffer,
                    command_buffer,
                    guidance_scale=args_cli.guidance_scale,
                )
                synchronize(device)
                batch_latencies.append((time.perf_counter() - infer_start) * 1000.0)
                current_chunk = policy.executable_chunk(trajectory, args_cli.exec_horizon)

            action = current_chunk[:, chunk_index]
            chunk_index = (chunk_index + 1) % args_cli.exec_horizon
            if agent_cfg.clip_actions is not None:
                action = torch.clamp(action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            proprio = get_proprio(raw_env, joint_ids)
            _, _, dones, _ = vec_env.step(action)
            update_history(proprio_buffer, action_buffer, command_buffer, proprio, previous_action, command)
            previous_action = action.clone()

        robot = raw_env._robot.data
        projected = robot.projected_gravity_b
        tilt = torch.rad2deg(torch.asin(torch.clamp(torch.linalg.vector_norm(projected[:, :2], dim=-1), 0.0, 1.0)))
        traces["vx"].append(robot.root_lin_vel_b[:, 0].detach().cpu().numpy())
        traces["vy"].append(robot.root_lin_vel_b[:, 1].detach().cpu().numpy())
        traces["wz"].append(robot.root_ang_vel_b[:, 2].detach().cpu().numpy())
        traces["height"].append(robot.root_pos_w[:, 2].detach().cpu().numpy())
        traces["tilt_deg"].append(tilt.detach().cpu().numpy())
        traces["action_delta"].append(
            torch.sqrt(torch.mean(torch.square(action - previous_for_delta), dim=-1)).detach().cpu().numpy()
        )
        previous_for_delta = action.clone()

        done_np = dones.detach().cpu().numpy().astype(bool)
        valid_trace.append((~failures & ~done_np).copy())
        newly_failed = done_np & ~failures
        failure_step[newly_failed] = step + 1
        failures |= done_np
        if np.any(done_np):
            current_chunk = None
            chunk_index = 0
            done_ids = torch.from_numpy(np.flatnonzero(done_np)).to(device=device, dtype=torch.long)
            current = get_proprio(raw_env, joint_ids)[done_ids]
            proprio_buffer[done_ids] = current[:, None, :]
            action_buffer[done_ids].zero_()
            command_buffer[done_ids] = command[done_ids, None, :]
            previous_action[done_ids].zero_()

    static_elapsed = time.perf_counter() - static_start
    trace_arrays = {name: np.stack(values, axis=0) for name, values in traces.items()}
    valid_array = np.stack(valid_trace, axis=0)
    static_rows = summarize_static(
        scenarios, trace_arrays, valid_array, failures, failure_step, settling_steps, dt
    )
    write_csv(output_dir / "static_summary.csv", static_rows, STATIC_COLUMNS)

    dynamic_rows: list[dict[str, float | bool]] = []
    if not args_cli.skip_dynamic:
        # The direct environment stores the last action tensor. Since policy
        # actions are created under inference_mode, reset must mutate it under
        # the same mode on PyTorch 2.7+.
        with torch.inference_mode():
            vec_env.reset()
        dynamic_velocity = torch.tensor((0.4, 0.0, 0.0), device=device).repeat(num_envs, 1)
        dynamic_command = torch.cat(
            [dynamic_velocity, torch.full((num_envs, 1), DYNAMIC_HEIGHTS[0], device=device)], dim=-1
        )
        set_env_commands(raw_env, dynamic_velocity)
        previous_action = reset_buffers(
            raw_env, joint_ids, proprio_buffer, action_buffer, command_buffer, dynamic_command
        )
        for _ in range(max(args_cli.warmup_steps, policy_cfg.history + 1)):
            proprio = get_proprio(raw_env, joint_ids)
            update_history(
                proprio_buffer, action_buffer, command_buffer, proprio, previous_action, dynamic_command
            )
            vec_env.step(stand_action)
            previous_action = stand_action.clone()

        segment_steps = max(1, int(round(args_cli.dynamic_segment_s / dt)))
        current_chunk = None
        chunk_index = 0
        dynamic_step = 0
        for requested_height in DYNAMIC_HEIGHTS:
            dynamic_command[:, 3] = requested_height
            for _ in range(segment_steps):
                with torch.inference_mode():
                    if current_chunk is None or chunk_index == 0:
                        trajectory = policy.predict_action_denormalized(
                            proprio_buffer,
                            action_buffer,
                            command_buffer,
                            guidance_scale=args_cli.guidance_scale,
                        )
                        current_chunk = policy.executable_chunk(trajectory, args_cli.exec_horizon)
                    action = current_chunk[:, chunk_index]
                    chunk_index = (chunk_index + 1) % args_cli.exec_horizon
                    if agent_cfg.clip_actions is not None:
                        action = torch.clamp(action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
                    proprio = get_proprio(raw_env, joint_ids)
                    _, _, dones, _ = vec_env.step(action)
                    update_history(
                        proprio_buffer,
                        action_buffer,
                        command_buffer,
                        proprio,
                        previous_action,
                        dynamic_command,
                    )
                    previous_action = action.clone()
                robot = raw_env._robot.data
                dynamic_rows.append(
                    {
                        "time_s": dynamic_step * dt,
                        "requested_height": float(requested_height),
                        "achieved_height": float(robot.root_pos_w[0, 2].item()),
                        "requested_vx": 0.4,
                        "achieved_vx": float(robot.root_lin_vel_b[0, 0].item()),
                        "achieved_vy": float(robot.root_lin_vel_b[0, 1].item()),
                        "achieved_wz": float(robot.root_ang_vel_b[0, 2].item()),
                        "done": bool(dones[0].item()),
                    }
                )
                dynamic_step += 1
        write_csv(output_dir / "dynamic_height_response.csv", dynamic_rows)

    deployment_deadline_ms = dt * args_cli.exec_horizon * 1000.0
    latency_summary = percentile_summary(single_latency)
    latency_summary["deadline_ms"] = deployment_deadline_ms
    latency_summary["deadline_misses"] = int(np.sum(np.asarray(single_latency) > deployment_deadline_ms))
    latency_summary["deadline_miss_rate"] = (
        float(latency_summary["deadline_misses"] / len(single_latency)) if single_latency else math.nan
    )
    batch_summary = percentile_summary(batch_latencies)

    dataset_paths = config.get("dataset", {}).get("hdf5_paths", [])
    dataset_hashes = {}
    for dataset_path in dataset_paths:
        path = Path(dataset_path)
        if path.exists():
            dataset_hashes[str(path.resolve())] = sha256(path)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "dataset_sha256": dataset_hashes,
        "git_commit": git_commit(),
        "seed": args_cli.seed,
        "num_scenarios": num_envs,
        "duration_s": args_cli.duration_s,
        "settling_s": args_cli.settling_s,
        "control_dt_s": dt,
        "nominal_control_hz": 1.0 / dt,
        "effective_vectorized_steps_hz": duration_steps / static_elapsed,
        "num_inference_steps": inference_steps,
        "exec_horizon": args_cli.exec_horizon,
        "torchscript_denoiser": args_cli.torchscript_denoiser,
        "single_env_latency": latency_summary,
        "vectorized_batch_latency": batch_summary,
        "survival_rate": float(np.mean(~failures)),
        "config": config,
    }
    write_json(output_dir / "summary.json", summary)
    plot_static_height(static_rows, output_dir / "height_interpolation.png")
    plot_static_velocity(static_rows, output_dir / "velocity_tracking.png")
    plot_dynamic_height(dynamic_rows, output_dir / "dynamic_height_response.png")
    plot_latency(single_latency, deployment_deadline_ms, output_dir / "latency.png")

    print(f"[DONE] Evaluation written to {output_dir.resolve()}")
    print(f"[RESULT] survival={summary['survival_rate']:.3f} latency_p95={latency_summary['p95_ms']:.2f} ms")
    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
