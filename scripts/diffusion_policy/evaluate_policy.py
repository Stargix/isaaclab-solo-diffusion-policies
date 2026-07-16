#!/usr/bin/env python3
"""Vectorized closed-loop evaluation of the path-following Diffusion Policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

parser = argparse.ArgumentParser(description="Evaluate the path-following Diffusion Policy.")
parser.add_argument("--task", type=str, default="solo12-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--speeds", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0))
parser.add_argument("--repeats", type=int, default=3, help="Stochastic repeats per scenario condition.")
parser.add_argument("--duration_s", type=float, default=50.0)
parser.add_argument("--warmup_steps", type=int, default=25)
parser.add_argument("--num_inference_steps", type=int, default=None)
parser.add_argument("--exec_horizon", type=int, default=8)
parser.add_argument("--guidance_scale", type=float, default=1.0)
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
from train.data.normalization import NormalizerStats
from train.conditioning.geometry import cumulative_xy_lengths, quat_wxyz_to_rotmat, yaw_from_rotmat
from train.conditioning.goal_builder import build_goal_from_path, advance_path_progress
from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
from train.config import resolve_inference_steps
from train.data.obs_utils import proprio_from_env_tensors
from train.runtime.checkpoint import load_training_checkpoint


@dataclass
class PathPlan:
    path_w: np.ndarray
    yaws_w: np.ndarray
    cumulative_lengths: np.ndarray
    shape_name: str


@dataclass
class Scenario:
    scenario_id: int
    repeat: int
    path_shape: str
    speed: float


def robot_yaw_w(quat_wxyz: np.ndarray) -> float:
    return yaw_from_rotmat(quat_wxyz_to_rotmat(quat_wxyz))


def build_straight_path(start_pos: np.ndarray, start_quat: np.ndarray, length_m: float = 10.0, num_points: int = 250) -> tuple[np.ndarray, np.ndarray]:
    yaw = robot_yaw_w(start_quat)
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    path_points = []
    yaws = []
    for s in np.linspace(0.0, length_m, num_points):
        x = float(start_pos[0] + cos_y * s)
        y = float(start_pos[1] + sin_y * s)
        path_points.append([x, y, 0.2932])
        yaws.append(yaw)
    return np.array(path_points, dtype=np.float32), np.array(yaws, dtype=np.float32)


def build_s_curve_path(start_pos: np.ndarray, start_quat: np.ndarray, length_m: float = 10.0, num_points: int = 250) -> tuple[np.ndarray, np.ndarray]:
    yaw = robot_yaw_w(start_quat)
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    path_points = []
    for s in np.linspace(0.0, length_m, num_points):
        lx = s
        ly = 0.3 * np.sin(2.0 * np.pi * s / 6.0)  # Amplitude = 0.3m, Period = 6m
        wx = float(start_pos[0] + cos_y * lx - sin_y * ly)
        wy = float(start_pos[1] + sin_y * lx + cos_y * ly)
        path_points.append([wx, wy, 0.2932])
    
    yaws = [yaw]
    for idx in range(1, len(path_points)):
        yaws.append(np.arctan2(path_points[idx][1] - path_points[idx - 1][1], path_points[idx][0] - path_points[idx - 1][0]))
    return np.array(path_points, dtype=np.float32), np.array(yaws, dtype=np.float32)


def build_circle_path(start_pos: np.ndarray, start_quat: np.ndarray, radius: float = 1.5, num_points: int = 250) -> tuple[np.ndarray, np.ndarray]:
    yaw = robot_yaw_w(start_quat)
    # Center of circle is start_pos + [0, radius] in the robot's local frame
    center_x = start_pos[0] - radius * np.sin(yaw)
    center_y = start_pos[1] + radius * np.cos(yaw)
    path_points = []
    for theta in np.linspace(-np.pi/2, 5/2*np.pi, num_points):  # 1.5 loops
        wx = float(center_x + radius * np.cos(yaw + theta + np.pi/2))
        wy = float(center_y + radius * np.sin(yaw + theta + np.pi/2))
        path_points.append([wx, wy, 0.2932])
        
    yaws = [yaw]
    for idx in range(1, len(path_points)):
        yaws.append(np.arctan2(path_points[idx][1] - path_points[idx - 1][1], path_points[idx][0] - path_points[idx - 1][0]))
    return np.array(path_points, dtype=np.float32), np.array(yaws, dtype=np.float32)


def compute_vectorized_goals(
    raw_env: Any,
    path_plans: list[PathPlan],
    speeds: np.ndarray,
    path_progress: np.ndarray,
    goal_horizon_steps: int,
    waypoint_time_offsets_s: tuple[float, float, float],
    dt: float,
    v_req_clip: float,
    device: torch.device,
) -> torch.Tensor:
    robot = raw_env._robot
    pos_w = robot.data.root_pos_w.cpu().numpy()
    quat_w = robot.data.root_quat_w.cpu().numpy()
    goals = []
    for i in range(len(path_plans)):
        plan = path_plans[i]
        start_idx = advance_path_progress(plan.path_w, pos_w[i], int(path_progress[i]))
        path_progress[i] = start_idx
        goals.append(
            build_goal_from_path(
                plan.path_w,
                plan.cumulative_lengths,
                plan.yaws_w,
                pos_w[i],
                quat_w[i],
                goal_horizon_steps=goal_horizon_steps,
                dt=dt,
                speed=speeds[i],
                start_idx=start_idx,
                waypoint_time_offsets_s=waypoint_time_offsets_s,
                v_req_clip=v_req_clip,
            )
        )
    return torch.from_numpy(np.stack(goals, axis=0)).to(device)


def get_proprio_30d(raw_env: Any, joint_ids: slice) -> torch.Tensor:
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
    goal_buffer: torch.Tensor,
    proprio: torch.Tensor,
    previous_action: torch.Tensor,
    goal: torch.Tensor,
) -> None:
    slide_buffer(proprio_buffer, proprio)
    slide_buffer(action_buffer, previous_action)
    slide_buffer(goal_buffer, goal)


def _model_cfg_from_checkpoint(config_dict: dict) -> Solo12DiffusionPolicyConfig:
    model = config_dict["model"]
    return Solo12DiffusionPolicyConfig(
        proprio_dim=model.get("proprio_dim", 30),
        action_hist_dim=model.get("action_hist_dim", 12),
        goal_dim=model.get("goal_dim", 11),
        history=config_dict["dataset"]["history"],
        prediction_horizon=config_dict["dataset"]["prediction_horizon"],
        execution_offset=config_dict["dataset"]["execution_offset"],
        d_model=model["d_model"],
        nhead=model["nhead"],
        num_layers=model["num_layers"],
        p_drop_emb=model.get("p_drop_emb", model.get("dropout", 0.0)),
        p_drop_attn=model.get("p_drop_attn", model.get("dropout", 0.3)),
        separate_goal_conditioning=model.get("separate_goal_conditioning", True),
        num_train_timesteps=config_dict["diffusion"]["num_train_timesteps"],
        beta_start=config_dict["diffusion"]["beta_start"],
        beta_end=config_dict["diffusion"]["beta_end"],
        beta_schedule=config_dict["diffusion"]["beta_schedule"],
        prediction_type=config_dict["diffusion"]["prediction_type"],
        variance_type=config_dict["diffusion"]["variance_type"],
        clip_sample=config_dict["diffusion"]["clip_sample"],
        cfg_dropout_prob=config_dict["diffusion"]["cfg_dropout_prob"],
    )


def make_scenarios() -> list[Scenario]:
    scenarios = []
    shapes = ["straight", "circle", "s_curve"]
    for shape in shapes:
        for speed in args_cli.speeds:
            for r in range(args_cli.repeats):
                scenarios.append(Scenario(len(scenarios), r, shape, speed))
    return scenarios


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    scenarios = make_scenarios()
    num_envs = len(scenarios)
    print(f"[INFO] Evaluating {num_envs} vectorized scenarios in parallel.")

    checkpoint_path = Path(args_cli.checkpoint).resolve()
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else (
        Path(__file__).resolve().parent / "evaluations" / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_training_checkpoint(
        checkpoint_path, device, expected_policy_kind=("spatial_time_preview_ddpm", "spatial_reference_path_ddpm")
    )
    config = checkpoint["config"]
    inference_steps = resolve_inference_steps(args_cli.num_inference_steps, config["diffusion"])
    policy_cfg = _model_cfg_from_checkpoint(config)
    policy_cfg.num_inference_steps = inference_steps

    policy = Solo12DiffusionPolicy(policy_cfg)
    policy.load_state_dict(checkpoint["ema_model_state_dict"])
    policy.set_normalizer_stats(checkpoint["normalizer_stats"])
    policy.to(device).eval()

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
    if hasattr(env_cfg, "base_push_interval_range_s"):
        env_cfg.base_push_interval_range_s = (1.0e9, 1.0e9)

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    joint_ids = raw_env._joint_ids
    dt = float(env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt)
    duration_steps = max(1, int(round(args_cli.duration_s / dt)))

    goal_horizon_steps = config["dataset"]["goal_horizon_steps"]
    waypoint_time_offsets_s = tuple(config["dataset"]["waypoint_time_offsets_s"])
    v_req_clip = config["dataset"].get("v_req_clip", 2.0)

    # Initialize environment and get spawn points
    vec_env.reset()
    start_pos = raw_env._robot.data.root_pos_w.cpu().numpy()
    start_quat = raw_env._robot.data.root_quat_w.cpu().numpy()

    # Generate custom paths per environment scenario
    path_plans: list[PathPlan] = []
    speeds = np.array([s.speed for s in scenarios], dtype=np.float32)
    path_progress = np.zeros(num_envs, dtype=np.int32)

    for i, s in enumerate(scenarios):
        if s.path_shape == "straight":
            pts, yaws = build_straight_path(start_pos[i], start_quat[i])
        elif s.path_shape == "s_curve":
            pts, yaws = build_s_curve_path(start_pos[i], start_quat[i])
        elif s.path_shape == "circle":
            pts, yaws = build_circle_path(start_pos[i], start_quat[i])
        else:
            raise ValueError(f"Unknown shape: {s.path_shape}")
        
        path_plans.append(
            PathPlan(
                path_w=pts,
                yaws_w=yaws,
                cumulative_lengths=cumulative_xy_lengths(pts),
                shape_name=s.path_shape
            )
        )

    # Setup history buffers
    proprio_buffer = torch.zeros((num_envs, policy_cfg.history, policy_cfg.proprio_dim), device=device)
    action_buffer = torch.zeros((num_envs, policy_cfg.history, policy_cfg.action_hist_dim), device=device)
    goal_buffer = torch.zeros((num_envs, policy_cfg.history, policy_cfg.goal_dim), device=device)
    previous_action = torch.zeros((num_envs, policy_cfg.action_dim), device=device)
    stand_action = torch.zeros_like(previous_action)

    # Run Warmup (Stand-hold to fill history buffers)
    print(f"[INFO] Warmup: {args_cli.warmup_steps} steps with stand-hold actions...")
    for _ in range(max(args_cli.warmup_steps, policy_cfg.history + 1)):
        proprio = get_proprio_30d(raw_env, joint_ids)
        goals = compute_vectorized_goals(
            raw_env, path_plans, speeds, path_progress, goal_horizon_steps, waypoint_time_offsets_s, dt, v_req_clip, device
        )
        update_history(proprio_buffer, action_buffer, goal_buffer, proprio, previous_action, goals)
        vec_env.step(stand_action)
        previous_action = stand_action.clone()

    # Track metrics
    trajectories = {i: {"ref": [], "actual": []} for i in range(num_envs)}
    tracking_errors = []
    achieved_speeds = []
    achieved_heights = []
    tilts = []
    action_deltas = []
    failures = np.zeros(num_envs, dtype=bool)
    failure_step = np.full(num_envs, -1, dtype=np.int64)

    current_chunk: torch.Tensor | None = None
    chunk_index = 0
    previous_for_delta = previous_action.clone()

    print("[INFO] Starting closed-loop evaluation...")
    for step in range(duration_steps):
        with torch.inference_mode():
            if current_chunk is None or chunk_index == 0:
                full_trajectory = policy.predict_action_denormalized(
                    proprio_buffer,
                    action_buffer,
                    goal_buffer,
                    guidance_scale=args_cli.guidance_scale,
                )
                current_chunk = policy.executable_chunk(full_trajectory, args_cli.exec_horizon)

            action = current_chunk[:, chunk_index]
            chunk_index = (chunk_index + 1) % args_cli.exec_horizon
            if agent_cfg.clip_actions is not None:
                action = torch.clamp(action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            
            proprio = get_proprio_30d(raw_env, joint_ids)
            goals = compute_vectorized_goals(
                raw_env, path_plans, speeds, path_progress, goal_horizon_steps, waypoint_time_offsets_s, dt, v_req_clip, device
            )
            _, _, dones, _ = vec_env.step(action)
            update_history(proprio_buffer, action_buffer, goal_buffer, proprio, previous_action, goals)
            previous_action = action.clone()

        # Record metrics for active step
        robot = raw_env._robot.data
        pos_w = robot.root_pos_w.cpu().numpy()
        projected = robot.projected_gravity_b
        tilt = torch.rad2deg(torch.asin(torch.clamp(torch.linalg.vector_norm(projected[:, :2], dim=-1), 0.0, 1.0))).cpu().numpy()
        vx = robot.root_lin_vel_b[:, 0].cpu().numpy()
        height = robot.root_pos_w[:, 2].cpu().numpy()
        a_delta = torch.sqrt(torch.mean(torch.square(action - previous_for_delta), dim=-1)).cpu().numpy()
        previous_for_delta = action.clone()

        step_errors = []
        for i in range(num_envs):
            plan = path_plans[i]
            closest_idx = advance_path_progress(plan.path_w, pos_w[i], int(path_progress[i]))
            err = float(np.linalg.norm(pos_w[i, :2] - plan.path_w[closest_idx, :2]))
            step_errors.append(err)

            if not failures[i]:
                # Track actual vs reference relative to spawn
                trajectories[i]["actual"].append(pos_w[i, :2] - start_pos[i, :2])
                ref_xy = plan.path_w[closest_idx, :2] - start_pos[i, :2]
                trajectories[i]["ref"].append(ref_xy)

        tracking_errors.append(step_errors)
        achieved_speeds.append(vx)
        achieved_heights.append(height)
        tilts.append(tilt)
        action_deltas.append(a_delta)

        # Handle resets / failures
        done_np = dones.cpu().numpy().astype(bool)
        newly_failed = done_np & ~failures
        failure_step[newly_failed] = step + 1
        failures |= done_np

    # Process metrics
    tracking_errors = np.stack(tracking_errors, axis=0)  # [steps, num_envs]
    achieved_speeds = np.stack(achieved_speeds, axis=0)  # [steps, num_envs]
    achieved_heights = np.stack(achieved_heights, axis=0)
    tilts = np.stack(tilts, axis=0)
    action_deltas = np.stack(action_deltas, axis=0)

    # Save summary table
    summary_rows = []
    for i, s in enumerate(scenarios):
        mask = np.ones(duration_steps, dtype=bool)
        if failures[i]:
            mask[int(failure_step[i]):] = False  # Clip errors after failure
        
        xy_errs = tracking_errors[:, i][mask]
        speeds_ach = achieved_speeds[:, i][mask]
        hgts = achieved_heights[:, i][mask]
        tls = tilts[:, i][mask]
        adeltas = action_deltas[:, i][mask]

        xy_rmse = float(np.sqrt(np.mean(np.square(xy_errs)))) if xy_errs.size else math.nan
        h_rmse = float(np.sqrt(np.mean(np.square(hgts - 0.2932)))) if hgts.size else math.nan
        survived = not bool(failures[i])

        summary_rows.append({
            "scenario_id": s.scenario_id,
            "repeat": s.repeat,
            "path_shape": s.path_shape,
            "requested_speed": s.speed,
            "survived": survived,
            "time_to_failure_s": float(failure_step[i] * dt) if failures[i] else args_cli.duration_s,
            "xy_rmse": xy_rmse,
            "height_rmse": h_rmse,
            "achieved_speed_mean": float(np.mean(speeds_ach)) if speeds_ach.size else math.nan,
            "achieved_height_mean": float(np.mean(hgts)) if hgts.size else math.nan,
            "tilt_rms_deg": float(np.sqrt(np.mean(np.square(tls)))) if tls.size else math.nan,
            "action_delta_rms": float(np.sqrt(np.mean(np.square(adeltas)))) if adeltas.size else math.nan
        })

    # Write CSV
    csv_cols = list(summary_rows[0].keys())
    csv_path = output_dir / "evaluation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Plot results
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Trajectory comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    shapes = ["straight", "circle", "s_curve"]
    titles = ["Straight Path", "Circle Path", "S-Curve Path"]
    for ax, shape, title in zip(axes, shapes, titles):
        # Plot reference path relative to start pos
        plan = None
        for i, s in enumerate(scenarios):
            if s.path_shape == shape and s.repeat == 0:
                plan = path_plans[i]
                ref_xy = plan.path_w[:, :2] - start_pos[i, :2]
                ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", linewidth=2.0, label="Reference")
                break
        
        # Plot actual paths for different speeds
        for speed in args_cli.speeds:
            for i, s in enumerate(scenarios):
                if s.path_shape == shape and s.speed == speed and s.repeat == 0:
                    act_traj = np.array(trajectories[i]["actual"])
                    ax.plot(act_traj[:, 0], act_traj[:, 1], alpha=0.8, linewidth=1.5, label=f"{speed} m/s")
                    break
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative X [m]")
        ax.set_ylabel("Relative Y [m]")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectories.png", dpi=180)
    plt.close(fig)

    # 2. Tracking error and speed plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"straight": "tab:blue", "circle": "tab:orange", "s_curve": "tab:green"}
    
    # Speed Tracking Accuracy
    axes[0].plot([0.0, 1.2], [0.0, 1.2], "k--", label="Ideal")
    for shape in shapes:
        subset = [r for r in summary_rows if r["path_shape"] == shape]
        speeds_req = sorted(list({r["requested_speed"] for r in subset}))
        speeds_ach = []
        speeds_std = []
        for v in speeds_req:
            vals = [r["achieved_speed_mean"] for r in subset if r["requested_speed"] == v]
            speeds_ach.append(np.mean(vals) if vals else np.nan)
            speeds_std.append(np.std(vals) if vals else np.nan)
        axes[0].errorbar(speeds_req, speeds_ach, yerr=speeds_std, fmt="o-", color=colors[shape], capsize=3, label=shape.capitalize())
    axes[0].set_xlabel("Requested Speed [m/s]", fontsize=10)
    axes[0].set_ylabel("Achieved Speed [m/s]", fontsize=10)
    axes[0].set_title("Speed Tracking Performance", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # XY Tracking Error
    for shape in shapes:
        subset = [r for r in summary_rows if r["path_shape"] == shape]
        speeds_req = sorted(list({r["requested_speed"] for r in subset}))
        errs = []
        errs_std = []
        for v in speeds_req:
            vals = [r["xy_rmse"] for r in subset if r["requested_speed"] == v]
            errs.append(np.mean(vals) if vals else np.nan)
            errs_std.append(np.std(vals) if vals else np.nan)
        axes[1].errorbar(speeds_req, errs, yerr=errs_std, fmt="o-", color=colors[shape], capsize=3, label=shape.capitalize())
    axes[1].set_xlabel("Requested Speed [m/s]", fontsize=10)
    axes[1].set_ylabel("XY Position RMSE [m]", fontsize=10)
    axes[1].set_title("XY Path Tracking Error vs Speed", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "speed_and_tracking_error.png", dpi=180)
    plt.close(fig)

    # 3. Survival rate & stability plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Survival Rate
    for shape in shapes:
        subset = [r for r in summary_rows if r["path_shape"] == shape]
        speeds_req = sorted(list({r["requested_speed"] for r in subset}))
        survival = []
        for v in speeds_req:
            rates = [r["survived"] for r in subset if r["requested_speed"] == v]
            survival.append(sum(rates) / len(rates) * 100.0)
        axes[0].plot(speeds_req, survival, "o-", color=colors[shape], label=shape.capitalize())
    axes[0].set_xlabel("Requested Speed [m/s]", fontsize=10)
    axes[0].set_ylabel("Survival Rate [%]", fontsize=10)
    axes[0].set_ylim(-5, 105)
    axes[0].set_title("Controller Survival Rate vs Speed", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Walking Smoothness (Action Delta RMS)
    for shape in shapes:
        subset = [r for r in summary_rows if r["path_shape"] == shape]
        speeds_req = sorted(list({r["requested_speed"] for r in subset}))
        jerkiness = []
        for v in speeds_req:
            vals = [r["action_delta_rms"] for r in subset if r["requested_speed"] == v]
            jerkiness.append(np.mean(vals) if vals else np.nan)
        axes[1].plot(speeds_req, jerkiness, "o-", color=colors[shape], label=shape.capitalize())
    axes[1].set_xlabel("Requested Speed [m/s]", fontsize=10)
    axes[1].set_ylabel("Action Jerkiness (RMS Delta)", fontsize=10)
    axes[1].set_title("Gait Smoothness vs Speed", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "survival_and_smoothness.png", dpi=180)
    plt.close(fig)

    # Write overall validation JSON summary
    global_survival = sum([r["survived"] for r in summary_rows]) / len(summary_rows)
    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "task": args_cli.task,
        "checkpoint": str(checkpoint_path),
        "policy_kind": checkpoint.get("policy_kind", "unknown"),
        "seed": args_cli.seed,
        "duration_s": args_cli.duration_s,
        "speeds_evaluated": list(args_cli.speeds),
        "overall_survival_rate": global_survival,
        "survival_by_path": {
            shape: sum([r["survived"] for r in summary_rows if r["path_shape"] == shape]) / sum([1 for r in summary_rows if r["path_shape"] == shape])
            for shape in shapes
        },
        "mean_xy_rmse_by_path": {
            shape: float(np.nanmean([r["xy_rmse"] for r in summary_rows if r["path_shape"] == shape and r["survived"]]))
            for shape in shapes
        }
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"[DONE] Vectorized Evaluation complete! Results saved in: {output_dir.resolve()}")
    print(f"[RESULT] Overall survival: {global_survival * 100.0:.1f}%")
    for shape in shapes:
        print(f"  - {shape.capitalize()}: survival={overall_summary['survival_by_path'][shape]*100.0:.1f}%, mean_xy_rmse={overall_summary['mean_xy_rmse_by_path'][shape]:.3f} m")

    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
