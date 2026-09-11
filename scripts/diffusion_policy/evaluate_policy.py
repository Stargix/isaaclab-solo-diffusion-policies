#!/usr/bin/env python3
"""Vectorized closed-loop evaluation of the path-following Diffusion Policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import h5py

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
parser.add_argument(
    "--path_shapes",
    type=str,
    nargs="+",
    choices=("straight", "circle", "s_curve", "right_angle", "random_polyline"),
    default=("straight", "circle", "s_curve", "right_angle", "random_polyline"),
    help="Path families to evaluate. New non-smooth families expose corner and polyline generalization.",
)
parser.add_argument("--repeats", type=int, default=3, help="Stochastic repeats per scenario condition.")
parser.add_argument("--duration_s", type=float, default=50.0)
parser.add_argument(
    "--route_length_m",
    type=float,
    default=None,
    help=(
        "Truncate every analytic route to this arc length. Use 4.0 to match "
        "the finite DPPO training task at every requested speed."
    ),
)
parser.add_argument("--path_height", type=float, default=None,
                    help="Route height in metres. Defaults to the checkpoint dataset's desired_base_height.")
parser.add_argument(
    "--path_heights", type=float, nargs="+", default=None,
    help="Evaluate several constant route heights in one paired run. Mutually exclusive with --path_height.",
)
parser.add_argument(
    "--transition_fractions", type=float, nargs="+", default=None,
    help="Evaluate walk->crouch transitions at these route fractions (e.g. 0.25 0.5 0.75). "
         "Uses 0.2932 m walk and 0.1705 m crouch heights.",
)
parser.add_argument(
    "--height_profile", choices=("constant", "interleaved", "random"), default="constant",
    help="constant uses one route height; interleaved repeats a schedule; random samples a reproducible schedule.",
)
parser.add_argument(
    "--height_segment_m", type=float, default=1.0,
    help="Length in metres of each interleaved height segment.",
)
parser.add_argument(
    "--height_cycle", type=float, nargs="+", default=(0.2932, 0.25, 0.21, 0.1705, 0.21, 0.25),
    help="Ordered base-height requirements [m] repeated by --height_profile interleaved.",
)
parser.add_argument("--warmup_steps", type=int, default=25)
parser.add_argument("--num_inference_steps", type=int, default=None)
parser.add_argument(
    "--exec_horizon",
    type=int,
    default=None,
    help="Executed actions per sample; defaults to the DPPO training value or 8 for Phase A.",
)
parser.add_argument("--guidance_scale", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--deterministic_resets", action="store_true",
    help="Disable reset joint/velocity randomization for paired causal evaluation.",
)
parser.add_argument(
    "--require_empty_output_dir", action="store_true",
    help="Refuse to mix a canonical run with files from an earlier invocation.",
)
parser.add_argument(
    "--save_timeseries", action="store_true",
    help="Save per-step route progress, tangent speed, height and validity for temporal-allocation analysis.",
)
parser.add_argument(
    "--reference_replay_dataset",
    type=str,
    default=None,
    help="Evaluate time-indexed reference_pos_w/reference_yaw_w fragments instead of analytic paths.",
)
parser.add_argument("--reference_replay_demos", type=int, default=12)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train.data.normalization import NormalizerStats
from train.conditioning.geometry import cumulative_xy_lengths, quat_wxyz_to_rotmat, yaw_from_rotmat
from train.conditioning.goal_builder import (
    advance_path_progress,
    build_goal_from_path,
    build_goal_vector,
    build_geometric_hindsight_goal_from_path,
    build_geometric_height_profile_goal_from_path,
    build_holonomic_goal_from_path,
    build_path_guidance_goal_from_path,
)
from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
from evaluation.metrics import (
    compute_first_task_success,
    compute_route_metrics,
    point_at_progress,
    physical_state_valid,
    project_trajectory_to_polyline,
    truncate_path_to_arc_length,
    valid_post_step_mask,
)
from evaluation.plots import plot_route_outcomes
from train.config import resolve_inference_steps
from train.data.obs_utils import proprio_from_env_tensors
from train.runtime.checkpoint import load_training_checkpoint


@dataclass
class PathPlan:
    path_w: np.ndarray
    yaws_w: np.ndarray
    cumulative_lengths: np.ndarray
    shape_name: str
    time_indexed: bool = False
    demo_name: str | None = None
    reference_command_b: np.ndarray | None = None
    guidance_w: np.ndarray | None = None
    guidance_cumulative: np.ndarray | None = None
    terminal_step: int | None = None
    height_schedule: tuple[float, ...] | None = None


@dataclass
class Scenario:
    scenario_id: int
    repeat: int
    path_shape: str
    speed: float
    demo_name: str | None = None
    transition_fraction: float | None = None
    path_height: float | None = None


def load_reference_fragments(
    path: str, count: int, min_steps: int,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    fragments = []
    with h5py.File(path, "r") as file:
        for demo_name in sorted(file["data"]):
            obs = file["data"][demo_name]["obs"]
            if "reference_pos_w" not in obs or "reference_yaw_w" not in obs:
                continue
            position = obs["reference_pos_w"][:].astype(np.float32)
            yaw = obs["reference_yaw_w"][:].reshape(-1).astype(np.float32)
            if len(position) < min_steps:
                continue
            command = obs["reference_command"][:] if "reference_command" in obs else obs["command_speed"][:]
            guidance = obs["guidance_pos_w"][:].astype(np.float32) if "guidance_pos_w" in obs else position.copy()
            speed = float(np.mean(np.linalg.norm(command[:min_steps, :2], axis=1)))
            fragments.append((demo_name, position, yaw, command.astype(np.float32), guidance, speed))
            if len(fragments) >= count:
                break
    if not fragments:
        raise ValueError(f"No reference fragments of at least {min_steps} steps found in {path}.")
    return fragments


def align_reference_fragment(
    position: np.ndarray, yaw: np.ndarray, spawn_pos: np.ndarray, spawn_quat: np.ndarray,
    guidance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    source_yaw = float(yaw[0])
    target_yaw = robot_yaw_w(spawn_quat)
    angle = target_yaw - source_yaw
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    aligned = position.copy()
    aligned[:, :2] = (position[:, :2] - position[0, :2]) @ rotation.T + spawn_pos[:2]
    aligned[:, 2] = position[:, 2]
    aligned_yaw = np.unwrap(yaw.astype(np.float64) - source_yaw + target_yaw).astype(np.float32)
    aligned_guidance = None
    if guidance is not None:
        aligned_guidance = guidance.copy()
        aligned_guidance[:, :2] = (guidance[:, :2] - position[0, :2]) @ rotation.T + spawn_pos[:2]
        aligned_guidance[:, 2] = guidance[:, 2]
    return aligned, aligned_yaw, aligned_guidance


def robot_yaw_w(quat_wxyz: np.ndarray) -> float:
    return yaw_from_rotmat(quat_wxyz_to_rotmat(quat_wxyz))


def mean_abs_path_curvature(path_w: np.ndarray) -> float:
    xy = path_w[:, :2].astype(np.float64)
    if len(xy) < 3:
        return 0.0
    heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    distance = np.maximum(np.linalg.norm(np.gradient(xy, axis=0), axis=1), 1.0e-6)
    return float(np.mean(np.abs(np.gradient(heading) / distance)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_is_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def dataset_hashes(config: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for raw_path in config.get("dataset", {}).get("hdf5_paths", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        if path.is_file():
            output[str(path.resolve())] = sha256_file(path)
    return output


def finite_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = np.asarray([row.get(key, math.nan) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def nanmean_or_nan(values: list[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else math.nan


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
        # theta=-pi/2 is exactly the robot's spawn point.  The previous extra
        # +pi/2 rotated the circle and made every evaluation start 2.12 m OOD.
        wx = float(center_x + radius * np.cos(yaw + theta))
        wy = float(center_y + radius * np.sin(yaw + theta))
        path_points.append([wx, wy, 0.2932])
        
    yaws = [yaw]
    for idx in range(1, len(path_points)):
        yaws.append(np.arctan2(path_points[idx][1] - path_points[idx - 1][1], path_points[idx][0] - path_points[idx - 1][0]))
    return np.array(path_points, dtype=np.float32), np.array(yaws, dtype=np.float32)


def _sample_local_polyline(
    waypoints_xy: np.ndarray,
    *,
    start_pos: np.ndarray,
    start_yaw: float,
    z: float = 0.2932,
    spacing_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample connected local line segments and assign tangent yaw per segment."""

    waypoints = np.asarray(waypoints_xy, dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 2:
        raise ValueError("Polyline waypoints must have shape (N,2), N>=2.")
    cos_y, sin_y = float(np.cos(start_yaw)), float(np.sin(start_yaw))
    rotation = np.asarray([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
    points: list[list[float]] = []
    yaws: list[float] = []
    for segment_idx, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 1.0e-4:
            continue
        count = max(1, int(np.ceil(length / spacing_m)))
        tangent_yaw = float(start_yaw + np.arctan2(delta[1], delta[0]))
        for sample_idx in range(count):
            if segment_idx > 0 and sample_idx == 0:
                continue
            local = a + (b - a) * (sample_idx / count)
            world_xy = start_pos[:2] + rotation @ local
            points.append([float(world_xy[0]), float(world_xy[1]), z])
            yaws.append(tangent_yaw)
    final_world = start_pos[:2] + rotation @ waypoints[-1]
    points.append([float(final_world[0]), float(final_world[1]), z])
    yaws.append(yaws[-1] if yaws else float(start_yaw))
    return np.asarray(points, dtype=np.float32), np.asarray(yaws, dtype=np.float32)


def build_right_angle_path(start_pos: np.ndarray, start_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected 90-degree corners with no circular smoothing."""

    yaw = robot_yaw_w(start_quat)
    waypoints = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.2], [4.0, 1.2], [4.0, -0.2], [6.0, -0.2]],
        dtype=np.float32,
    )
    return _sample_local_polyline(waypoints, start_pos=start_pos, start_yaw=yaw)


def build_random_polyline_path(start_pos: np.ndarray, start_quat: np.ndarray, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic connected random polyline for repeatable generalization tests."""

    yaw = robot_yaw_w(start_quat)
    rng = np.random.default_rng(seed)
    x = 0.0
    y = 0.0
    waypoints = [[x, y]]
    for _ in range(7):
        x += float(rng.uniform(0.65, 1.15))
        y = float(np.clip(y + rng.uniform(-0.85, 0.85), -1.15, 1.15))
        waypoints.append([x, y])
    return _sample_local_polyline(np.asarray(waypoints, dtype=np.float32), start_pos=start_pos, start_yaw=yaw)


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
    goal_representation: str,
    reference_step: int | None = None,
    speed_budget_start_arc: np.ndarray | None = None,
    speed_budget_target_arc: np.ndarray | None = None,
    speed_budget_max_mps: float | None = None,
) -> torch.Tensor:
    robot = raw_env._robot
    pos_w = robot.data.root_pos_w.cpu().numpy()
    quat_w = robot.data.root_quat_w.cpu().numpy()
    goals = []
    for i in range(len(path_plans)):
        plan = path_plans[i]
        if plan.time_indexed:
            state_step = min(int(reference_step or 0), len(plan.path_w) - 1)
            if goal_representation == "path_guidance_se2_36":
                if plan.guidance_w is None or plan.guidance_cumulative is None or plan.terminal_step is None:
                    raise RuntimeError("Path-guidance replay is missing guide/terminal metadata.")
                goals.append(build_goal_vector(
                    plan.path_w,
                    plan.cumulative_lengths,
                    state_step,
                    len(plan.path_w) - 1,
                    pos_w[i],
                    quat_w[i],
                    yaws_w=plan.yaws_w,
                    dt=dt,
                    goal_representation=goal_representation,
                    reference_command_b=plan.reference_command_b,
                    guidance_pos_w=plan.guidance_w,
                    guidance_cumulative_xy=plan.guidance_cumulative,
                    terminal_idx=plan.terminal_step,
                ))
                continue
            goals.append(
                build_goal_vector(
                    plan.path_w,
                    plan.cumulative_lengths,
                    state_step,
                    min(state_step + goal_horizon_steps, len(plan.path_w) - 1),
                    pos_w[i],
                    quat_w[i],
                yaws_w=plan.yaws_w,
                dt=dt,
                waypoint_time_offsets_s=waypoint_time_offsets_s,
                v_req_clip=v_req_clip,
                goal_representation=goal_representation,
                reference_command_b=plan.reference_command_b,
                height_profile_w=(
                    plan.path_w[:, 2] if goal_representation == "hindsight_geom_profile16" else None
                ),
                )
            )
            continue
        start_idx = advance_path_progress(plan.path_w, pos_w[i], int(path_progress[i]))
        path_progress[i] = start_idx
        if goal_representation == "path_guidance_se2_36":
            goals.append(build_path_guidance_goal_from_path(
                plan.path_w,
                plan.cumulative_lengths,
                plan.yaws_w,
                pos_w[i],
                quat_w[i],
                speed=speeds[i],
                start_idx=start_idx,
            ))
        elif goal_representation == "holonomic_se2_32":
            goals.append(
                build_holonomic_goal_from_path(
                    plan.path_w,
                    plan.cumulative_lengths,
                    plan.yaws_w,
                    pos_w[i],
                    quat_w[i],
                    goal_horizon_steps=goal_horizon_steps,
                    dt=dt,
                    speed=speeds[i],
                    start_idx=start_idx,
                )
            )
        elif goal_representation == "hindsight_geom_avg12":
            goals.append(
                build_geometric_hindsight_goal_from_path(
                    plan.path_w,
                    plan.cumulative_lengths,
                    plan.yaws_w,
                    pos_w[i],
                    quat_w[i],
                    goal_horizon_steps=goal_horizon_steps,
                    dt=dt,
                    speed=speeds[i],
                    start_idx=start_idx,
                    v_avg_clip=v_req_clip,
                )
            )
        elif goal_representation == "hindsight_geom_profile16":
            goals.append(
                build_geometric_height_profile_goal_from_path(
                    plan.path_w,
                    plan.cumulative_lengths,
                    plan.yaws_w,
                    pos_w[i],
                    quat_w[i],
                    goal_horizon_steps=goal_horizon_steps,
                    dt=dt,
                    speed=speeds[i],
                    start_idx=start_idx,
                    v_avg_clip=v_req_clip,
                )
            )
        else:
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
    goals_tensor = torch.from_numpy(np.stack(goals, axis=0)).to(device)
    if speed_budget_target_arc is not None or speed_budget_start_arc is not None:
        if speed_budget_target_arc is None or speed_budget_start_arc is None:
            raise ValueError("Both speed-budget arc arrays must be provided together.")
        if goal_representation not in {"hindsight_geom_avg12", "hindsight_geom_profile16"} or any(
            plan.time_indexed for plan in path_plans
        ):
            raise ValueError("Dynamic speed budget currently requires an analytic geometric average-speed route.")
        from scripts.dppo_diffusion_rl.conditioning import remaining_speed_budget

        current_arc = np.asarray(
            [plan.cumulative_lengths[int(path_progress[i])] for i, plan in enumerate(path_plans)],
            dtype=np.float32,
        )
        target_distance = np.maximum(speed_budget_target_arc - speed_budget_start_arc, 0.0)
        remaining_distance = np.maximum(speed_budget_target_arc - current_arc, 0.0)
        elapsed_s = float(reference_step or 0) * float(dt)
        goals_tensor[:, -1] = remaining_speed_budget(
            remaining_distance=torch.as_tensor(remaining_distance, device=device),
            route_length=torch.as_tensor(target_distance, device=device),
            desired_mean_speed=torch.as_tensor(speeds, device=device),
            elapsed_s=torch.full((len(path_plans),), elapsed_s, device=device),
            max_speed=float(speed_budget_max_mps or v_req_clip),
            min_remaining_time_s=dt,
        )
    return goals_tensor


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


def make_scenarios(path_heights: tuple[float, ...]) -> list[Scenario]:
    if args_cli.reference_replay_dataset:
        # For holonomic evaluation, we need at least 76 extra lookahead steps (1.52s) to prevent the goal lookahead horizon check from exceeding the reference endpoint.
        minimum = max(2, int(round(args_cli.duration_s * 50.0)) + 76)
        fragments = load_reference_fragments(
            args_cli.reference_replay_dataset, args_cli.reference_replay_demos, minimum
        )
        return [
            Scenario(index, 0, "reference_replay", speed, demo_name=demo_name)
            for index, (demo_name, _, _, _, _, speed) in enumerate(fragments)
        ]
    scenarios = []
    shapes = list(args_cli.path_shapes)
    transitions = [None] if args_cli.transition_fractions is None else list(args_cli.transition_fractions)
    for fraction in transitions:
        if fraction is not None and not 0.0 < fraction < 1.0:
            raise ValueError(f"Transition fractions must be in (0, 1), got {fraction}.")
    for shape in shapes:
        for speed in args_cli.speeds:
            for height in path_heights:
                for fraction in transitions:
                    for r in range(args_cli.repeats):
                        scenarios.append(
                            Scenario(
                                len(scenarios), r, shape, speed,
                                transition_fraction=fraction,
                                path_height=height,
                            )
                        )
    return scenarios


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    checkpoint_path = Path(args_cli.checkpoint).resolve()
    source_git_dirty = git_is_dirty()
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else (
        Path(__file__).resolve().parent / "evaluations" / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    )
    if args_cli.require_empty_output_dir and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Canonical evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    device = torch.device(requested_device)
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        device,
        expected_policy_kind=(
            "spatial_time_preview_ddpm", "spatial_reference_path_ddpm", "holonomic_reference_path_ddpm",
            "path_guidance_terminal_ddpm", "spatial_hindsight_geometry_ddpm",
            "spatial_hindsight_height_profile_ddpm",
        ),
        allow_dppo=True,
    )
    config = checkpoint["config"]
    dppo_task_contract_version = int(checkpoint.get("dppo_task_contract_version", 0))
    dppo_task_config = checkpoint.get("dppo_task_config", {})
    profile_height_mae_tolerance_m = float(
        dppo_task_config.get("profile_height_mae_tolerance_m", 0.04)
    )
    if args_cli.exec_horizon is None:
        args_cli.exec_horizon = (
            int(checkpoint["dppo_config"]["exec_horizon"])
            if checkpoint.get("algorithm") == "dppo"
            else 8
        )
    if args_cli.path_height is not None and args_cli.path_heights is not None:
        raise ValueError("Use either --path_height or --path_heights, not both.")
    if args_cli.path_height is None and args_cli.path_heights is None:
        path_height = 0.2932
        for raw_dataset in config.get("dataset", {}).get("hdf5_paths", []):
            dataset_path = Path(raw_dataset)
            if not dataset_path.is_absolute():
                dataset_path = _PROJECT_ROOT / dataset_path
            if dataset_path.is_file():
                try:
                    with h5py.File(dataset_path, "r") as dataset:
                        values = []
                        for demo in dataset.get("data", {}).values():
                            if "desired_base_height" in demo["obs"]:
                                values.append(float(np.median(demo["obs"]["desired_base_height"][:])))
                        if values:
                            path_height = float(np.median(values))
                except (OSError, KeyError, ValueError):
                    pass
                break
    elif args_cli.path_height is not None:
        path_height = float(args_cli.path_height)
    else:
        path_height = float(args_cli.path_heights[0])
    path_heights = tuple(float(value) for value in (args_cli.path_heights or (path_height,)))
    if any(not 0.10 <= value <= 0.40 for value in path_heights):
        raise ValueError(f"Route heights must be in [0.10, 0.40], got {path_heights}.")
    if len(set(path_heights)) != len(path_heights):
        raise ValueError(f"Route heights must be unique, got {path_heights}.")
    if args_cli.height_profile in {"interleaved", "random"}:
        if args_cli.transition_fractions is not None:
            raise ValueError("Use either --height_profile interleaved or --transition_fractions, not both.")
        if args_cli.height_segment_m <= 0.0:
            raise ValueError("--height_segment_m must be positive.")
        if len(args_cli.height_cycle) < 2 or any(not 0.10 <= value <= 0.40 for value in args_cli.height_cycle):
            raise ValueError("--height_cycle needs at least two physical heights in [0.10, 0.40].")
    if args_cli.route_length_m is not None:
        if args_cli.route_length_m <= 0.0:
            raise ValueError("--route_length_m must be positive.")
        if args_cli.reference_replay_dataset:
            raise ValueError("--route_length_m is only valid for analytic paths.")
    if len(path_heights) > 1 and (
        args_cli.height_profile != "constant" or args_cli.transition_fractions is not None
    ):
        raise ValueError("--path_heights is only valid for constant-height evaluation without transitions.")
    print(f"[INFO] Evaluation route heights: {list(path_heights)} m")
    if args_cli.height_profile in {"interleaved", "random"}:
        print(f"[INFO] {args_cli.height_profile.capitalize()} height schedule: cycle={list(args_cli.height_cycle)}, "
              f"segment={args_cli.height_segment_m:.2f} m")
    scenarios = make_scenarios(path_heights)
    num_envs = len(scenarios)
    print(f"[INFO] Evaluating {num_envs} vectorized scenarios in parallel.")
    inference_steps = resolve_inference_steps(args_cli.num_inference_steps, config["diffusion"])
    policy_cfg = _model_cfg_from_checkpoint(config)
    policy_cfg.num_inference_steps = inference_steps

    if checkpoint.get("algorithm") == "dppo":
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from scripts.dppo_diffusion_rl.checkpointing import build_inference_policy

        policy = build_inference_policy(
            checkpoint, device, inference_steps=inference_steps
        )
        print(
            f"[INFO] DPPO hybrid sampler: frozen early steps + "
            f"{checkpoint['dppo_config']['finetune_denoising_steps']} fine-tuned steps"
        )
    else:
        policy = Solo12DiffusionPolicy(policy_cfg)
        policy.load_state_dict(checkpoint["ema_model_state_dict"])
        policy.set_normalizer_stats(checkpoint["normalizer_stats"])
        policy.to(device).eval()

    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = str(device)
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
    # IsaacLab's default flat plane references a remote Grid USD.  Generate the
    # physically equivalent plane locally so evaluation does not depend on
    # network/cache state.  This changes neither the robot nor policy contract.
    env_cfg.terrain.terrain_type = "generator"
    env_cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=args_cli.seed,
        curriculum=False,
        size=(20.0, 20.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
    )
    if args_cli.deterministic_resets:
        if hasattr(env_cfg, "reset_base_lin_vel_range"):
            env_cfg.reset_base_lin_vel_range = (0.0, 0.0)
        if hasattr(env_cfg, "reset_base_ang_vel_range"):
            env_cfg.reset_base_ang_vel_range = (0.0, 0.0)
        if hasattr(env_cfg, "flexed_initial_joint_pos_noise_range"):
            env_cfg.flexed_initial_joint_pos_noise_range = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    joint_ids = raw_env._joint_ids
    dt = float(env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt)
    duration_steps = max(1, int(round(args_cli.duration_s / dt)))

    goal_horizon_steps = config["dataset"]["goal_horizon_steps"]
    waypoint_time_offsets_s = tuple(config["dataset"]["waypoint_time_offsets_s"])
    v_req_clip = config["dataset"].get("v_req_clip", 2.0)
    goal_representation = config["dataset"].get("goal_representation", "path11")
    profile_task_active = goal_representation == "hindsight_geom_profile16"
    if args_cli.height_profile in {"interleaved", "random"} and goal_representation not in {
        "path11", "hindsight_geom_avg12", "hindsight_geom_profile16"
    }:
        raise ValueError(
            "The selected height profile requires a goal schema with a terminal height "
            f"(got {goal_representation!r})."
        )
    if goal_representation in {"holonomic_se2_32", "path_guidance_se2_36"} and not args_cli.reference_replay_dataset:
        raise ValueError(
            "The primary SE(2) route evaluation must use --reference_replay_dataset from a separate "
            "collection seed. Analytic tangent paths can be explored in play_policy, but they are "
            "not a distribution-matched benchmark for this SE(2) route contract."
        )

    # Initialize environment and get spawn points
    vec_env.reset()
    start_pos = raw_env._robot.data.root_pos_w.cpu().numpy()
    start_quat = raw_env._robot.data.root_quat_w.cpu().numpy()

    # Generate analytic paths or align stored collection references to each spawn.
    path_plans: list[PathPlan] = []
    speeds = np.array([s.speed for s in scenarios], dtype=np.float32)
    path_progress = np.zeros(num_envs, dtype=np.int32)

    replay_fragments = None
    if args_cli.reference_replay_dataset:
        preview_margin = 0 if goal_representation == "path_guidance_se2_36" else goal_horizon_steps
        minimum = max(2, int(round(args_cli.duration_s / dt)) + preview_margin + 1)
        replay_fragments = load_reference_fragments(
            args_cli.reference_replay_dataset, args_cli.reference_replay_demos, minimum
        )
    for i, s in enumerate(scenarios):
        if replay_fragments is not None:
            demo_name, stored_pos, stored_yaw, stored_command, stored_guidance, _ = replay_fragments[i]
            pts, yaws, aligned_guidance = align_reference_fragment(
                stored_pos, stored_yaw, start_pos[i], start_quat[i], stored_guidance
            )
        elif s.path_shape == "straight":
            pts, yaws = build_straight_path(start_pos[i], start_quat[i])
        elif s.path_shape == "s_curve":
            pts, yaws = build_s_curve_path(start_pos[i], start_quat[i])
        elif s.path_shape == "circle":
            pts, yaws = build_circle_path(start_pos[i], start_quat[i])
        elif s.path_shape == "right_angle":
            pts, yaws = build_right_angle_path(start_pos[i], start_quat[i])
        elif s.path_shape == "random_polyline":
            # Pair the same repeat geometry across speeds/heights while making
            # distinct repeats actual geometry generalization trials.
            pts, yaws = build_random_polyline_path(
                start_pos[i], start_quat[i], seed=args_cli.seed + s.repeat
            )
        else:
            raise ValueError(f"Unknown shape: {s.path_shape}")
        if args_cli.route_length_m is not None and replay_fragments is None:
            pts, yaws = truncate_path_to_arc_length(
                pts, yaws, float(args_cli.route_length_m)
            )
        # Analytic route builders use walk height by default. For transition
        # studies, introduce a deliberate walk->crouch height change at the
        # requested fraction of travelled arc length.
        height_schedule = None
        if args_cli.height_profile in {"interleaved", "random"}:
            arc = cumulative_xy_lengths(pts)
            segment_index = np.floor(arc / args_cli.height_segment_m).astype(np.int64)
            cycle = np.asarray(args_cli.height_cycle, dtype=np.float32)
            num_segments = int(segment_index.max()) + 1
            if args_cli.height_profile == "interleaved":
                height_schedule = tuple(float(cycle[index % len(cycle)]) for index in range(num_segments))
            else:
                # Repeats sample different schedules while path/speed conditions
                # with the same repeat remain paired on the same schedule.
                rng = np.random.default_rng(args_cli.seed + s.repeat)
                sampled: list[float] = []
                for _ in range(num_segments):
                    candidates = cycle if not sampled else cycle[~np.isclose(cycle, sampled[-1])]
                    sampled.append(float(rng.choice(candidates)))
                height_schedule = tuple(sampled)
            pts[:, 2] = np.asarray(height_schedule, dtype=np.float32)[segment_index]
        elif s.transition_fraction is None:
            if s.path_height is None:
                raise RuntimeError("Analytic constant-height scenario is missing path_height.")
            pts[:, 2] = s.path_height
        else:
            arc = cumulative_xy_lengths(pts)
            transition_arc = float(arc[-1] * s.transition_fraction)
            pts[:, 2] = np.where(arc < transition_arc, 0.2932, 0.1705)
        
        path_plans.append(
            PathPlan(
                path_w=pts,
                yaws_w=yaws,
                cumulative_lengths=cumulative_xy_lengths(pts),
                shape_name=s.path_shape,
                time_indexed=replay_fragments is not None,
                demo_name=s.demo_name,
                reference_command_b=stored_command if replay_fragments is not None else None,
                guidance_w=aligned_guidance if replay_fragments is not None else None,
                guidance_cumulative=(
                    cumulative_xy_lengths(aligned_guidance) if replay_fragments is not None else None
                ),
                terminal_step=(
                    min(
                        len(stored_command) - 1,
                        int(np.flatnonzero(np.linalg.norm(stored_command, axis=-1) > 1.0e-4)[-1]) + 1,
                    )
                    if replay_fragments is not None
                    and np.any(np.linalg.norm(stored_command, axis=-1) > 1.0e-4)
                    else 0 if replay_fragments is not None else None
                ),
                height_schedule=height_schedule,
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
            raw_env, path_plans, speeds, path_progress, goal_horizon_steps, waypoint_time_offsets_s, dt, v_req_clip, device,
            goal_representation,
            reference_step=0,
        )
        update_history(proprio_buffer, action_buffer, goal_buffer, proprio, previous_action, goals)
        vec_env.step(stand_action)
        previous_action = stand_action.clone()

    # The requested speed horizon starts after history warm-up.  Keep this
    # state separate from the original spawn so warm-up drift cannot be counted
    # as free route progress.
    rollout_start_pos = raw_env._robot.data.root_pos_w.cpu().numpy().copy()
    dynamic_speed_budget = bool(
        checkpoint.get("algorithm") == "dppo"
        and int(checkpoint.get("dppo_task_contract_version", 1)) >= 3
    )
    speed_budget_max_mps = float(
        checkpoint.get("dppo_task_config", {}).get("speed_budget_max_mps", 0.8)
    )
    speed_budget_start_arc = np.asarray(
        [
            project_trajectory_to_polyline(
                rollout_start_pos[i : i + 1, :2], plan.path_w[:, :2]
            )[0][0]
            for i, plan in enumerate(path_plans)
        ],
        dtype=np.float32,
    )
    speed_budget_target_arc = np.asarray(
        [
            min(
                float(speed_budget_start_arc[i]) + float(speeds[i]) * float(args_cli.duration_s),
                float(plan.cumulative_lengths[-1]),
            )
            for i, plan in enumerate(path_plans)
        ],
        dtype=np.float32,
    )
    if dynamic_speed_budget:
        print(
            "[INFO] Actor conditioning: closed-loop remaining-route speed budget "
            f"(checkpoint task contract v{dppo_task_contract_version})."
        )

    # Track metrics
    trajectories = {i: {"ref": [], "actual": []} for i in range(num_envs)}
    tracking_errors = []
    achieved_speeds = []
    achieved_planar_speeds = []
    achieved_tangent_speeds = []
    positions_xy = []
    progress_m = []
    trace_valid = []
    achieved_heights = []
    achieved_yaws = []
    tilts = []
    action_deltas = []
    commanded_heights = []
    conditioned_height_profiles = []
    required_heights = []
    failures = np.zeros(num_envs, dtype=bool)
    base_failures = np.zeros(num_envs, dtype=bool)
    physical_sanity_failures = np.zeros(num_envs, dtype=bool)
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
                raw_env, path_plans, speeds, path_progress, goal_horizon_steps, waypoint_time_offsets_s, dt, v_req_clip, device,
                goal_representation,
                reference_step=step,
                speed_budget_start_arc=speed_budget_start_arc if dynamic_speed_budget else None,
                speed_budget_target_arc=speed_budget_target_arc if dynamic_speed_budget else None,
                speed_budget_max_mps=speed_budget_max_mps if dynamic_speed_budget else None,
            )
            if goal_representation == "hindsight_geom_avg12":
                commanded_heights.append(goals[:, 10].detach().cpu().numpy())
            elif goal_representation == "hindsight_geom_profile16":
                commanded_heights.append(goals[:, 11].detach().cpu().numpy())
                conditioned_height_profiles.append(
                    goals[:, [2, 5, 8, 11]].detach().cpu().numpy()
                )
            elif goal_representation == "path11":
                commanded_heights.append(goals[:, 8].detach().cpu().numpy())
            _, _, dones, _ = vec_env.step(action)
            done_np = dones.cpu().numpy().astype(bool)
            update_history(proprio_buffer, action_buffer, goal_buffer, proprio, previous_action, goals)
            previous_action = action.clone()

        # Record metrics for active step
        robot = raw_env._robot.data
        pos_w = robot.root_pos_w.cpu().numpy()
        projected = robot.projected_gravity_b
        tilt = torch.rad2deg(torch.asin(torch.clamp(torch.linalg.vector_norm(projected[:, :2], dim=-1), 0.0, 1.0))).cpu().numpy()
        vx = robot.root_lin_vel_b[:, 0].cpu().numpy()
        planar_speed = torch.linalg.vector_norm(robot.root_lin_vel_b[:, :2], dim=-1).cpu().numpy()
        vel_world = robot.root_lin_vel_w[:, :2].cpu().numpy()
        height = robot.root_pos_w[:, 2].cpu().numpy()
        quat = robot.root_quat_w.cpu().numpy()
        yaw = np.asarray([robot_yaw_w(value) for value in quat], dtype=np.float32)
        a_delta = torch.sqrt(torch.mean(torch.square(action - previous_for_delta), dim=-1)).cpu().numpy()
        action_np = action.detach().cpu().numpy()
        previous_for_delta = action.clone()
        physical_invalid = ~physical_state_valid(
            pos_w, planar_speed, tilt, action_np
        )
        step_failed = done_np | physical_invalid
        newly_failed = step_failed & ~failures

        step_errors = []
        tangent_speeds = []
        step_required_heights = []
        step_progress_m = []
        for i in range(num_envs):
            plan = path_plans[i]
            closest_idx = (
                min(step, len(plan.path_w) - 1)
                if plan.time_indexed
                else advance_path_progress(plan.path_w, pos_w[i], int(path_progress[i]))
            )
            err = float(np.linalg.norm(pos_w[i, :2] - plan.path_w[closest_idx, :2]))
            step_errors.append(err)
            step_required_heights.append(float(plan.path_w[closest_idx, 2]))
            step_progress_m.append(float(plan.cumulative_lengths[closest_idx] - plan.cumulative_lengths[0]))
            tangent_idx = min(closest_idx + 1, len(plan.path_w) - 1)
            tangent = plan.path_w[tangent_idx, :2] - plan.path_w[max(0, tangent_idx - 1), :2]
            tangent_norm = float(np.linalg.norm(tangent))
            tangent = tangent / tangent_norm if tangent_norm > 1.0e-6 else np.zeros(2, dtype=np.float32)
            tangent_speeds.append(float(np.dot(vel_world[i], tangent)))

            if not failures[i] and not step_failed[i]:
                # Track actual vs reference relative to spawn
                trajectories[i]["actual"].append(pos_w[i, :2] - start_pos[i, :2])
                ref_xy = plan.path_w[closest_idx, :2] - start_pos[i, :2]
                trajectories[i]["ref"].append(ref_xy)

        tracking_errors.append(step_errors)
        achieved_speeds.append(vx)
        achieved_planar_speeds.append(planar_speed)
        achieved_tangent_speeds.append(np.asarray(tangent_speeds, dtype=np.float32))
        positions_xy.append(pos_w[:, :2].copy())
        progress_m.append(np.asarray(step_progress_m, dtype=np.float32))
        # Once an environment has terminated, IsaacLab may immediately reset
        # it; exclude all subsequent reset states from the causal trace.
        trace_valid.append((~failures) & (~step_failed))
        achieved_heights.append(height)
        achieved_yaws.append(yaw)
        tilts.append(tilt)
        action_deltas.append(a_delta)
        required_heights.append(np.asarray(step_required_heights, dtype=np.float32))

        # Handle resets / failures
        failure_step[newly_failed] = step + 1
        base_failures[newly_failed & done_np] = True
        physical_sanity_failures[newly_failed & physical_invalid] = True
        failures |= done_np
        failures |= physical_invalid

    # Process metrics
    tracking_errors = np.stack(tracking_errors, axis=0)  # [steps, num_envs]
    achieved_speeds = np.stack(achieved_speeds, axis=0)  # [steps, num_envs]
    achieved_planar_speeds = np.stack(achieved_planar_speeds, axis=0)
    achieved_tangent_speeds = np.stack(achieved_tangent_speeds, axis=0)
    positions_xy = np.stack(positions_xy, axis=0)
    progress_m = np.stack(progress_m, axis=0)
    trace_valid = np.stack(trace_valid, axis=0)
    achieved_heights = np.stack(achieved_heights, axis=0)
    achieved_yaws = np.stack(achieved_yaws, axis=0)
    tilts = np.stack(tilts, axis=0)
    action_deltas = np.stack(action_deltas, axis=0)

    if len(commanded_heights) != duration_steps:
        raise RuntimeError(f"No terminal-height command available for goal representation {goal_representation!r}.")
    if len(required_heights) != duration_steps:
        raise RuntimeError("No route-height requirement was recorded during evaluation.")
    preview_target_heights = np.stack(commanded_heights, axis=0)
    # Primary tracking target: height required at the robot's current route
    # progress, matching the online task reward. The terminal-preview command
    # is retained separately to diagnose anticipatory causal response.
    target_heights = np.stack(required_heights, axis=0)
    height_profile_previews = (
        np.stack(conditioned_height_profiles, axis=0)
        if conditioned_height_profiles
        else np.empty((duration_steps, num_envs, 0), dtype=np.float32)
    )

    if args_cli.save_timeseries:
        # Keep the trace opt-in: broad evaluations otherwise only need the
        # compact CSV/JSON summaries.  The path metadata makes this artifact
        # self-contained and avoids reconstructing random-polyline geometry.
        np.savez_compressed(
            output_dir / "timeseries.npz",
            time_s=np.arange(duration_steps, dtype=np.float32) * dt,
            positions_xy=positions_xy,
            progress_m=progress_m,
            tangent_speed_m_s=achieved_tangent_speeds,
            planar_speed_m_s=achieved_planar_speeds,
            achieved_height_m=achieved_heights,
            achieved_yaw_rad=achieved_yaws,
            required_height_m=target_heights,
            commanded_height_m=preview_target_heights,
            conditioned_height_profile_m=height_profile_previews,
            valid=trace_valid,
            requested_speed_m_s=speeds,
            failure_step=failure_step,
            base_failure=base_failures,
            physical_sanity_failure=physical_sanity_failures,
        )
        trace_metadata = {
            "dt_s": dt,
            "scenarios": [
                {
                    "index": i,
                    "path_shape": s.path_shape,
                    "repeat": s.repeat,
                    "requested_speed_m_s": float(s.speed),
                    "path_height_m": s.path_height,
                    "height_schedule_m": list(path_plans[i].height_schedule or []),
                    "path_xy": path_plans[i].path_w[:, :2].tolist(),
                    "path_cumulative_m": path_plans[i].cumulative_lengths.tolist(),
                }
                for i, s in enumerate(scenarios)
            ],
        }
        with open(output_dir / "timeseries_metadata.json", "w", encoding="utf-8") as trace_file:
            json.dump(trace_metadata, trace_file, indent=2)
        print(f"[INFO] Saved temporal-allocation trace to {output_dir / 'timeseries.npz'}")

    # Save summary table
    summary_rows = []
    for i, s in enumerate(scenarios):
        mask = valid_post_step_mask(
            duration_steps, int(failure_step[i]) if failures[i] else None
        )
        
        xy_errs = tracking_errors[:, i][mask]
        speeds_ach = achieved_speeds[:, i][mask]
        planar_speeds_ach = achieved_planar_speeds[:, i][mask]
        tangent_speeds_ach = achieved_tangent_speeds[:, i][mask]
        hgts = achieved_heights[:, i][mask]
        yaws_ach = achieved_yaws[:, i][mask]
        tls = tilts[:, i][mask]
        adeltas = action_deltas[:, i][mask]

        xy_rmse = float(np.sqrt(np.mean(np.square(xy_errs)))) if xy_errs.size else math.nan
        plan = path_plans[i]
        target_h = target_heights[:, i][mask]
        preview_target_h = preview_target_heights[:, i][mask]
        h_rmse = float(np.sqrt(np.mean(np.square(hgts - target_h)))) if hgts.size else math.nan
        height_abs_error = np.abs(hgts - target_h)
        preview_h_rmse = (
            float(np.sqrt(np.mean(np.square(hgts - preview_target_h))))
            if hgts.size else math.nan
        )
        target_change_steps = np.flatnonzero(np.abs(np.diff(target_heights[:, i])) > 1.0e-5) + 1
        preview_change_steps = np.flatnonzero(
            np.abs(np.diff(preview_target_heights[:, i])) > 1.0e-5
        ) + 1
        valid_steps = int(np.count_nonzero(mask))
        survived = not bool(failures[i])
        if plan.time_indexed and plan.terminal_step is not None:
            terminal_index = int(plan.terminal_step)
            target_progress_m = float(plan.cumulative_lengths[terminal_index] - plan.cumulative_lengths[0])
            target_arc = float(plan.cumulative_lengths[terminal_index])
        else:
            # Analytic paths are longer than a finite evaluation episode at low
            # speed.  Compare against the point reachable in v*t, not always
            # against the global endpoint of the polyline.
            rollout_start_progress = float(project_trajectory_to_polyline(
                rollout_start_pos[i : i + 1, :2], plan.path_w[:, :2]
            )[0][0])
            remaining_path = max(
                0.0, float(plan.cumulative_lengths[-1]) - rollout_start_progress
            )
            target_progress_m = min(float(s.speed) * float(args_cli.duration_s), remaining_path)
            target_arc = rollout_start_progress + target_progress_m
            terminal_index = int(np.clip(np.searchsorted(plan.cumulative_lengths, target_arc, side="left"), 0, len(plan.path_w) - 1))
        terminal_index = int(np.clip(terminal_index, 0, len(plan.path_w) - 1))
        if trajectories[i]["actual"]:
            final_xy = np.asarray(trajectories[i]["actual"][-1]) + start_pos[i, :2]
            terminal_target_xy = point_at_progress(plan.path_w[:, :2], target_arc)
            terminal_position_error = float(np.linalg.norm(final_xy - terminal_target_xy))
        else:
            terminal_position_error = math.nan
        terminal_yaw_error = (
            float(abs(np.arctan2(
                np.sin(float(plan.yaws_w[terminal_index]) - float(yaws_ach[-1])),
                np.cos(float(plan.yaws_w[terminal_index]) - float(yaws_ach[-1])),
            )))
            if yaws_ach.size else math.nan
        )
        hold_count = min(50, planar_speeds_ach.size)

        route_metrics = None
        final_relative_xy = np.full(2, np.nan, dtype=np.float64)
        if trajectories[i]["actual"]:
            relative_positions = np.asarray(trajectories[i]["actual"], dtype=np.float64)
            final_relative_xy = relative_positions[-1]
            if not plan.time_indexed:
                route_metrics = compute_route_metrics(
                    relative_positions + start_pos[i, :2],
                    plan.path_w[:, :2],
                    requested_speed_m_s=float(s.speed),
                    dt=dt,
                    horizon_steps=duration_steps,
                    start_position_xy=rollout_start_pos[i, :2],
                )
        route_values = route_metrics.to_dict() if route_metrics is not None else {}
        task_success_applicable = bool(
            route_metrics is not None
            and abs(target_arc - float(plan.cumulative_lengths[-1])) <= 1.0e-4
        )
        task_values: dict[str, float | bool] = {}
        if task_success_applicable:
            task_values = compute_first_task_success(
                relative_positions + start_pos[i, :2],
                plan.path_w[:, :2],
                yaws_rad=yaws_ach,
                heights_m=hgts,
                requested_speed_m_s=float(s.speed),
                target_progress_m=float(route_metrics.target_progress_m),
                target_yaw_rad=float(plan.yaws_w[terminal_index]),
                target_height_m=float(plan.path_w[terminal_index, 2]),
                dt=dt,
                start_position_xy=rollout_start_pos[i, :2],
                target_heights_m=target_h if profile_task_active else None,
                profile_height_mae_tolerance_m=(
                    profile_height_mae_tolerance_m if profile_task_active else None
                ),
            ).to_dict()

        summary_rows.append({
            "scenario_id": s.scenario_id,
            "repeat": s.repeat,
            "path_shape": s.path_shape,
            "reference_demo": s.demo_name or "",
            "requested_speed": s.speed,
            "requested_height": s.path_height if s.path_height is not None else math.nan,
            "transition_fraction": s.transition_fraction,
            "height_profile": args_cli.height_profile,
            "height_segment_m": args_cli.height_segment_m if args_cli.height_profile in {"interleaved", "random"} else math.nan,
            "height_schedule": (
                ";".join(f"{height:.4f}" for height in plan.height_schedule)
                if plan.height_schedule is not None else ""
            ),
            "target_height_changes": int(target_change_steps.size),
            "target_height_changes_before_failure": int(np.count_nonzero(target_change_steps < valid_steps)),
            "preview_height_changes": int(preview_change_steps.size),
            "reference_curvature_abs_mean": mean_abs_path_curvature(path_plans[i].path_w),
            "survived": survived,
            "base_failure": bool(base_failures[i]),
            "physical_sanity_failure": bool(physical_sanity_failures[i]),
            "time_to_failure_s": float(failure_step[i] * dt) if failures[i] else args_cli.duration_s,
            "xy_rmse": xy_rmse,
            "height_rmse": h_rmse,
            "height_abs_error_mean": float(np.mean(np.abs(hgts - target_h))) if hgts.size else math.nan,
            "height_abs_error_p95": float(np.percentile(height_abs_error, 95)) if hgts.size else math.nan,
            "height_within_0p03_fraction": float(np.mean(height_abs_error <= 0.03)) if hgts.size else math.nan,
            "preview_height_rmse": preview_h_rmse,
            "preview_height_abs_error_mean": (
                float(np.mean(np.abs(hgts - preview_target_h))) if hgts.size else math.nan
            ),
            "target_progress_m": target_progress_m,
            "target_path_index": terminal_index,
            "terminal_position_error_m": terminal_position_error,
            "route_target_progress_m": route_values.get("target_progress_m", math.nan),
            "route_final_progress_m": route_values.get("final_progress_m", math.nan),
            "route_furthest_progress_m": route_values.get("furthest_progress_m", math.nan),
            "route_completion_ratio": route_values.get("completion_ratio", math.nan),
            "route_horizon_mean_speed_m_s": route_values.get("horizon_mean_speed_m_s", math.nan),
            "route_horizon_speed_ratio": route_values.get("horizon_speed_ratio", math.nan),
            "route_arrived": route_values.get("arrived", math.nan),
            "route_arrival_time_s": route_values.get("arrival_time_s", math.nan),
            "route_arrival_mean_speed_m_s": route_values.get("arrival_mean_speed_m_s", math.nan),
            "route_arrival_speed_ratio": route_values.get("arrival_speed_ratio", math.nan),
            "route_active_mean_speed_m_s": route_values.get("active_mean_speed_m_s", math.nan),
            "route_active_speed_ratio": route_values.get("active_speed_ratio", math.nan),
            "route_schedule_mae_m": route_values.get("schedule_mae_m", math.nan),
            "route_final_schedule_error_m": route_values.get("final_schedule_error_m", math.nan),
            "route_cross_track_rmse_m": route_values.get("cross_track_rmse_m", math.nan),
            "route_cross_track_p95_m": route_values.get("cross_track_p95_m", math.nan),
            "route_terminal_position_error_m": route_values.get("terminal_position_error_m", math.nan),
            "task_success_applicable": task_success_applicable,
            "task_arrived": task_values.get("arrived", False),
            "task_success": task_values.get("success", False),
            "task_success_time_s": task_values.get("time_s", math.nan),
            "task_success_position_error_m": task_values.get("position_error_m", math.nan),
            "task_success_yaw_error_rad": task_values.get("yaw_error_rad", math.nan),
            "task_success_height_error_m": task_values.get("height_error_m", math.nan),
            "task_success_mean_speed_error_m_s": task_values.get(
                "mean_speed_error_m_s", math.nan
            ),
            "task_success_profile_height_mae_m": task_values.get(
                "profile_height_mae_m", math.nan
            ),
            "task_success_profile_height_within_tolerance_fraction": task_values.get(
                "profile_height_within_tolerance_fraction", math.nan
            ),
            "task_success_yaw_ok": task_values.get("yaw_ok", False),
            "task_success_terminal_height_ok": task_values.get(
                "terminal_height_ok", False
            ),
            "task_success_mean_speed_ok": task_values.get("mean_speed_ok", False),
            "task_success_profile_height_ok": task_values.get(
                "profile_height_ok", False
            ),
            "final_relative_x_m": float(final_relative_xy[0]),
            "final_relative_y_m": float(final_relative_xy[1]),
            "terminal_yaw_error_rad": terminal_yaw_error,
            "final_planar_speed_m_s": float(planar_speeds_ach[-1]) if planar_speeds_ach.size else math.nan,
            "last_1s_planar_speed_mean_m_s": (
                float(np.mean(planar_speeds_ach[-hold_count:])) if hold_count else math.nan
            ),
            "achieved_speed_mean": float(np.mean(speeds_ach)) if speeds_ach.size else math.nan,
            "achieved_planar_speed_mean": float(np.mean(planar_speeds_ach)) if planar_speeds_ach.size else math.nan,
            "achieved_tangent_speed_mean": float(np.mean(tangent_speeds_ach)) if tangent_speeds_ach.size else math.nan,
            "achieved_speed_ratio": (
                float(np.mean(tangent_speeds_ach) / s.speed)
                if tangent_speeds_ach.size and s.speed > 1.0e-6 else math.nan
            ),
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

    # Set premium style defaults
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.color"] = "#333333"
    plt.rcParams["ytick.color"] = "#333333"

    # Canonical finite-horizon metrics are plotted separately from the legacy
    # nearest-point/tangent diagnostics retained below for backward comparison.
    plot_route_outcomes(summary_rows, output_dir / "route_outcomes.png")

    # Color Palette: Sapphire Blue, Coral Pink, Emerald, Amber, Amethyst
    colors_list = ["#0052CC", "#FF5A5F", "#00A86B", "#FFB300", "#7B1FA2"]
    
    # 1. Trajectory comparison plot
    shapes = sorted({scenario.path_shape for scenario in scenarios})
    titles = [shape.replace("_", " ").title() for shape in shapes]

    if replay_fragments is not None:
        # Reference replay evaluation: each scenario has a different path.
        # Plot the first 6 scenarios in a grid to show tracking accuracy per demo.
        num_plots = min(6, len(scenarios))
        cols = min(3, num_plots)
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), facecolor="white")
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i in range(num_plots):
            ax = axes[i]
            ax.set_facecolor("#fafafa")
            plan = path_plans[i]
            ref_xy = (plan.path_w[:, :2] - start_pos[i, :2])[:duration_steps]
            act_traj = np.array(trajectories[i]["actual"])
            
            # Plot reference and actual
            ax.plot(ref_xy[:, 0], ref_xy[:, 1], color="#333333", linestyle="--", linewidth=2.0, label="Reference", zorder=3)
            ax.plot(act_traj[:, 0], act_traj[:, 1], color="#0052CC", alpha=0.9, linewidth=2.0, label="Actual", zorder=4)
            
            ax.set_title(f"{plan.demo_name}\n({scenarios[i].speed:.2f} m/s)", fontsize=11, fontweight="bold", pad=8, color="#222222")
            ax.set_xlabel("Relative X [m]", fontsize=9)
            ax.set_ylabel("Relative Y [m]", fontsize=9)
            ax.grid(True, which="both", color="#e5e5e5", linestyle="-", linewidth=0.5, zorder=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axis("equal")
            ax.legend(fontsize=8, facecolor="white", edgecolor="#eaeaea")
            
        # Hide any unused subplots
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)
            
        fig.tight_layout()
        fig.savefig(output_dir / "trajectories.png", dpi=200)
        plt.close(fig)
    else:
        # Analytic paths evaluation
        fig, axes = plt.subplots(1, len(shapes), figsize=(5.5 * len(shapes), 5), facecolor="white")
        if len(shapes) == 1:
            axes = [axes]
        for ax, shape, title in zip(axes, shapes, titles):
            ax.set_facecolor("#fafafa")
            # Plot reference path relative to start pos
            plan = None
            for i, s in enumerate(scenarios):
                if s.path_shape == shape and s.repeat == 0:
                    plan = path_plans[i]
                    ref_xy = plan.path_w[:, :2] - start_pos[i, :2]
                    ax.plot(ref_xy[:, 0], ref_xy[:, 1], color="#333333", linestyle="--", linewidth=2.0, label="Reference", zorder=3)
                    break
            
            # Plot actual paths and the finite-horizon endpoint for each speed.
            # The complete route remains a faint geometric reference; the
            # marker is the position that must be reached after duration_s.
            speeds_list = sorted(list({scenario.speed for scenario in scenarios}))
            for speed_idx, speed in enumerate(speeds_list):
                for i, s in enumerate(scenarios):
                    if s.path_shape == shape and s.speed == speed and s.repeat == 0:
                        act_traj = np.array(trajectories[i]["actual"])
                        color = colors_list[speed_idx % len(colors_list)]
                        ax.plot(act_traj[:, 0], act_traj[:, 1], color=color, alpha=0.9, linewidth=2.0, label=f"{speed} m/s", zorder=4)
                        plan = path_plans[i]
                        start_progress, _ = project_trajectory_to_polyline(
                            rollout_start_pos[i : i + 1, :2], plan.path_w[:, :2]
                        )
                        target_progress = min(
                            float(start_progress[0]) + speed * args_cli.duration_s,
                            float(plan.cumulative_lengths[-1]),
                        )
                        target_xy = point_at_progress(plan.path_w[:, :2], target_progress) - start_pos[i, :2]
                        ax.scatter(
                            target_xy[0],
                            target_xy[1],
                            marker="X",
                            s=65,
                            color=color,
                            edgecolor="white",
                            linewidth=0.8,
                            zorder=5,
                        )
                        break
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#222222")
            ax.set_xlabel("Relative X [m]", fontsize=10, labelpad=8)
            ax.set_ylabel("Relative Y [m]", fontsize=10, labelpad=8)
            ax.grid(True, which="both", color="#e5e5e5", linestyle="-", linewidth=0.5, zorder=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axis("equal")
            ax.legend(fontsize=9, framealpha=0.9, facecolor="white", edgecolor="#eaeaea")
        fig.tight_layout()
        fig.savefig(output_dir / "trajectories.png", dpi=200)
        plt.close(fig)

    # 2. Height command and tracking. The first panel shows the commanded
    # walk/crouch profile and the measured base height; the second summarizes
    # absolute height error by transition location and requested speed.
    transition_values = sorted({s.transition_fraction for s in scenarios}, key=lambda x: -1.0 if x is None else x)
    transition_values = [value for value in transition_values if value is not None] or [None]
    if args_cli.height_profile == "constant" and transition_values == [None]:
        height_conditions = [(height, None) for height in path_heights]
    else:
        height_conditions = [(None, fraction) for fraction in transition_values]
    time_axis = np.arange(duration_steps, dtype=np.float32) * dt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="white")
    ax = axes[0]
    plotted = False
    reference_shape = "straight" if "straight" in shapes else shapes[0]
    reference_speed = min(args_cli.speeds, key=lambda value: abs(value - 0.4))
    for requested_height, fraction in height_conditions:
        candidates = [
            (i, s) for i, s in enumerate(scenarios)
            if s.path_shape == reference_shape and s.repeat == 0
            and s.speed == reference_speed and s.transition_fraction == fraction
            and (requested_height is None or s.path_height == requested_height)
        ]
        if not candidates:
            continue
        i, scenario = candidates[0]
        valid = valid_post_step_mask(
            duration_steps, int(failure_step[i]) if failures[i] else None
        )
        measured = achieved_heights[:, i].copy()
        measured[~valid] = np.nan
        if args_cli.height_profile in {"interleaved", "random"}:
            label = f"{args_cli.height_profile} every {args_cli.height_segment_m:.1f} m"
        elif requested_height is not None:
            label = f"constant {requested_height:.4f} m"
        else:
            label = "constant" if fraction is None else f"transition at {fraction:.0%}"
        ax.plot(time_axis, target_heights[:, i], linestyle="--", linewidth=2.0, label=f"required ({label})")
        ax.plot(
            time_axis,
            preview_target_heights[:, i],
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=f"policy preview ({label})",
        )
        if height_profile_previews.shape[-1] == 4:
            for slot in range(3):
                ax.plot(
                    time_axis,
                    height_profile_previews[:, i, slot],
                    linestyle=":",
                    linewidth=0.7,
                    alpha=0.28,
                    label=f"profile {25 * (slot + 1)}%" if not plotted else None,
                )
        ax.plot(time_axis, measured, linewidth=1.5, alpha=0.9, label=f"actual ({label})")
        plotted = True
    if plotted:
        ax.set_title(f"Height tracking: {reference_shape}, {reference_speed:.1f} m/s")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Base height [m]")
        ax.axhline(0.2932, color="#888888", linewidth=0.8, alpha=0.5)
        ax.axhline(0.1705, color="#888888", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    speeds_list = sorted({float(s.speed) for s in scenarios})
    x = np.arange(len(height_conditions), dtype=np.float32)
    width = 0.8 / max(1, len(speeds_list))
    for speed_index, speed in enumerate(speeds_list):
        values = []
        for requested_height, fraction in height_conditions:
            rows = [r["height_abs_error_mean"] for r in summary_rows
                    if r["requested_speed"] == speed and r["transition_fraction"] == fraction
                    and (requested_height is None or r["requested_height"] == requested_height)]
            values.append(float(np.nanmean(rows)) if rows else np.nan)
        ax.bar(x + (speed_index - (len(speeds_list) - 1) / 2) * width, values,
               width=width, label=f"{speed:.1f} m/s")
    labels = []
    for requested_height, fraction in height_conditions:
        if args_cli.height_profile in {"interleaved", "random"}:
            labels.append(f"{args_cli.height_profile}\n{args_cli.height_segment_m:.1f} m")
        elif requested_height is not None:
            labels.append(f"{requested_height:.4f} m")
        else:
            labels.append("constant" if fraction is None else f"{fraction:.0%}")
    ax.set_xticks(x, labels)
    ax.set_title("Mean absolute height error")
    ax.set_xlabel("Height-profile condition")
    ax.set_ylabel("Absolute error [m]")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "height_tracking_and_error.png", dpi=200)
    plt.close(fig)

    # 3. Tracking error and speed plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    palette = ("#0052CC", "#FF5A5F", "#00A86B", "#FFB300", "#7B1FA2")
    colors = {shape: palette[index % len(palette)] for index, shape in enumerate(shapes)}
    
    # Speed Tracking Accuracy
    axes[0].set_facecolor("#fafafa")
    axes[0].plot([0.0, 1.2], [0.0, 1.2], color="#777777", linestyle=":", linewidth=1.5, label="Ideal Reference")
    for shape in shapes:
        subset = [r for r in summary_rows if r["path_shape"] == shape]
        speeds_req = sorted(list({r["requested_speed"] for r in subset}))
        speeds_ach = []
        speeds_std = []
        for v in speeds_req:
            vals = np.asarray([
                r["route_arrival_mean_speed_m_s"]
                for r in subset
                if r["requested_speed"] == v
            ], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            speeds_ach.append(np.mean(vals) if vals.size else np.nan)
            speeds_std.append(np.std(vals) if vals.size else np.nan)
        
        speeds_ach = np.array(speeds_ach)
        speeds_std = np.array(speeds_std)
        axes[0].plot(speeds_req, speeds_ach, "o-", color=colors[shape], linewidth=2.0, markersize=6, markeredgecolor="white", markeredgewidth=1.0, label=shape.replace("_", " ").title())
        valid_mask = ~np.isnan(speeds_ach) & ~np.isnan(speeds_std)
        if np.any(valid_mask):
            axes[0].fill_between(
                np.array(speeds_req)[valid_mask],
                (speeds_ach - speeds_std)[valid_mask],
                (speeds_ach + speeds_std)[valid_mask],
                color=colors[shape],
                alpha=0.15
            )
            
    axes[0].set_xlabel("Requested Speed [m/s]", fontsize=10, labelpad=8)
    axes[0].set_ylabel("Mean speed to endpoint [m/s]", fontsize=10, labelpad=8)
    axes[0].set_title("Finite-route arrival speed", fontsize=12, fontweight="bold", pad=12, color="#222222")
    axes[0].grid(True, color="#e5e5e5", linestyle="-", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(fontsize=9, facecolor="white", edgecolor="#eaeaea")

    # XY Tracking Error
    axes[1].set_facecolor("#fafafa")
    for shape in shapes:
        subset = [r for r in summary_rows if r["path_shape"] == shape]
        speeds_req = sorted(list({r["requested_speed"] for r in subset}))
        errs = []
        errs_std = []
        for v in speeds_req:
            vals = [r["xy_rmse"] for r in subset if r["requested_speed"] == v]
            errs.append(np.mean(vals) if vals else np.nan)
            errs_std.append(np.std(vals) if vals else np.nan)
            
        errs = np.array(errs)
        errs_std = np.array(errs_std)
        axes[1].plot(speeds_req, errs, "o-", color=colors[shape], linewidth=2.0, markersize=6, markeredgecolor="white", markeredgewidth=1.0, label=shape.replace("_", " ").title())
        valid_mask = ~np.isnan(errs) & ~np.isnan(errs_std)
        if np.any(valid_mask):
            axes[1].fill_between(
                np.array(speeds_req)[valid_mask],
                (errs - errs_std)[valid_mask],
                (errs + errs_std)[valid_mask],
                color=colors[shape],
                alpha=0.15
            )
            
    axes[1].set_xlabel("Requested Speed [m/s]", fontsize=10, labelpad=8)
    axes[1].set_ylabel("XY Position RMSE [m]", fontsize=10, labelpad=8)
    axes[1].set_title("XY Path Tracking Error vs Speed", fontsize=12, fontweight="bold", pad=12, color="#222222")
    axes[1].grid(True, color="#e5e5e5", linestyle="-", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(fontsize=9, facecolor="white", edgecolor="#eaeaea")
    fig.tight_layout()
    fig.savefig(output_dir / "speed_and_tracking_error.png", dpi=200)
    plt.close(fig)

    # 3. Survival rate & stability plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    
    # Survival Rate
    axes[0].set_facecolor("#fafafa")
    for shape in shapes:
        subset = [r for r in summary_rows if r["path_shape"] == shape]
        speeds_req = sorted(list({r["requested_speed"] for r in subset}))
        survival = []
        for v in speeds_req:
            rates = [r["survived"] for r in subset if r["requested_speed"] == v]
            survival.append(sum(rates) / len(rates) * 100.0)
        axes[0].plot(speeds_req, survival, "o-", color=colors[shape], linewidth=2.0, markersize=6, markeredgecolor="white", markeredgewidth=1.0, label=shape.replace("_", " ").title())
    axes[0].set_xlabel("Requested Speed [m/s]", fontsize=10, labelpad=8)
    axes[0].set_ylabel("Survival Rate [%]", fontsize=10, labelpad=8)
    axes[0].set_ylim(-5, 105)
    axes[0].set_title("Controller Survival Rate vs Speed", fontsize=12, fontweight="bold", pad=12, color="#222222")
    axes[0].grid(True, color="#e5e5e5", linestyle="-", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(fontsize=9, facecolor="white", edgecolor="#eaeaea")

    # Walking Smoothness (Action Delta RMS)
    axes[1].set_facecolor("#fafafa")
    for shape in shapes:
        subset = [r for r in summary_rows if r["path_shape"] == shape]
        speeds_req = sorted(list({r["requested_speed"] for r in subset}))
        jerkiness = []
        for v in speeds_req:
            vals = [r["action_delta_rms"] for r in subset if r["requested_speed"] == v]
            jerkiness.append(np.mean(vals) if vals else np.nan)
        axes[1].plot(speeds_req, jerkiness, "o-", color=colors[shape], linewidth=2.0, markersize=6, markeredgecolor="white", markeredgewidth=1.0, label=shape.replace("_", " ").title())
    axes[1].set_xlabel("Requested Speed [m/s]", fontsize=10, labelpad=8)
    axes[1].set_ylabel("Action Jerkiness (RMS Delta)", fontsize=10, labelpad=8)
    axes[1].set_title("Gait Smoothness vs Speed", fontsize=12, fontweight="bold", pad=12, color="#222222")
    axes[1].grid(True, color="#e5e5e5", linestyle="-", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(fontsize=9, facecolor="white", edgecolor="#eaeaea")
    fig.tight_layout()
    fig.savefig(output_dir / "survival_and_smoothness.png", dpi=200)
    plt.close(fig)

    # Write overall validation JSON summary
    global_survival = sum([r["survived"] for r in summary_rows]) / len(summary_rows)
    route_rows = [
        row for row in summary_rows
        if np.isfinite(float(row["route_target_progress_m"]))
    ]
    task_rows = [row for row in summary_rows if row["task_success_applicable"]]
    arrived_task_rows = [row for row in task_rows if row["task_arrived"]]
    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "task": args_cli.task,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "git_commit": current_git_commit(),
        "git_dirty_before_evaluation": source_git_dirty,
        "config": config,
        "dataset_sha256": dataset_hashes(config),
        "policy_kind": checkpoint.get("policy_kind", "unknown"),
        "dppo_task_contract_version": dppo_task_contract_version,
        "profile_height_mae_tolerance_m": (
            profile_height_mae_tolerance_m if profile_task_active else None
        ),
        "seed": args_cli.seed,
        "duration_s": args_cli.duration_s,
        "route_length_m": args_cli.route_length_m,
        "num_scenarios": len(scenarios),
        "num_inference_steps": inference_steps,
        "exec_horizon": args_cli.exec_horizon,
        "goal_horizon_steps": goal_horizon_steps,
        "goal_representation": goal_representation,
        "dynamic_speed_budget": dynamic_speed_budget,
        "speed_budget_max_mps": speed_budget_max_mps if dynamic_speed_budget else None,
        "waypoint_time_offsets_s": waypoint_time_offsets_s,
        "speeds_evaluated": list(args_cli.speeds),
        "heights_evaluated": list(path_heights),
        "transition_fractions": list(args_cli.transition_fractions) if args_cli.transition_fractions is not None else [],
        "height_profile": args_cli.height_profile,
        "height_segment_m": args_cli.height_segment_m if args_cli.height_profile in {"interleaved", "random"} else None,
        "height_cycle": list(args_cli.height_cycle) if args_cli.height_profile in {"interleaved", "random"} else [],
        "reference_replay_dataset": (
            str(Path(args_cli.reference_replay_dataset).resolve()) if args_cli.reference_replay_dataset else None
        ),
        "deterministic_resets": bool(args_cli.deterministic_resets),
        "overall_survival_rate": global_survival,
        "overall_base_failure_rate": float(np.mean(base_failures)),
        "overall_physical_sanity_failure_rate": float(np.mean(physical_sanity_failures)),
        "overall_route_completion_ratio": finite_mean(summary_rows, "route_completion_ratio"),
        "overall_route_horizon_speed_ratio": finite_mean(summary_rows, "route_horizon_speed_ratio"),
        "overall_route_arrival_rate": (
            sum(bool(row["route_arrived"]) for row in route_rows) / len(route_rows)
            if route_rows else None
        ),
        "overall_route_arrival_speed_ratio": finite_mean(
            summary_rows, "route_arrival_speed_ratio"
        ),
        "overall_task_success_rate": (
            sum(bool(row["task_success"]) for row in task_rows) / len(task_rows)
            if task_rows else None
        ),
        "task_success_scenarios": len(task_rows),
        "overall_task_arrival_rate": (
            len(arrived_task_rows) / len(task_rows) if task_rows else None
        ),
        "arrival_conditional_yaw_success_rate": (
            sum(bool(row["task_success_yaw_ok"]) for row in arrived_task_rows)
            / len(arrived_task_rows)
            if arrived_task_rows else None
        ),
        "arrival_conditional_terminal_height_success_rate": (
            sum(
                bool(row["task_success_terminal_height_ok"])
                for row in arrived_task_rows
            )
            / len(arrived_task_rows)
            if arrived_task_rows else None
        ),
        "arrival_conditional_mean_speed_success_rate": (
            sum(bool(row["task_success_mean_speed_ok"]) for row in arrived_task_rows)
            / len(arrived_task_rows)
            if arrived_task_rows else None
        ),
        "arrival_conditional_profile_height_success_rate": (
            sum(
                bool(row["task_success_profile_height_ok"])
                for row in arrived_task_rows
            )
            / len(arrived_task_rows)
            if arrived_task_rows else None
        ),
        "arrival_profile_height_mae_m": finite_mean(
            arrived_task_rows, "task_success_profile_height_mae_m"
        ),
        "overall_route_cross_track_rmse_m": finite_mean(summary_rows, "route_cross_track_rmse_m"),
        "overall_route_terminal_position_error_m": finite_mean(
            summary_rows, "route_terminal_position_error_m"
        ),
        "survival_by_path": {
            shape: sum([r["survived"] for r in summary_rows if r["path_shape"] == shape]) / sum([1 for r in summary_rows if r["path_shape"] == shape])
            for shape in shapes
        },
        "arrival_rate_by_path": {
            shape: (
                sum(
                    bool(row["route_arrived"])
                    for row in route_rows if row["path_shape"] == shape
                )
                / sum(1 for row in route_rows if row["path_shape"] == shape)
            )
            for shape in shapes
            if any(row["path_shape"] == shape for row in route_rows)
        },
        "mean_xy_rmse_by_path": {
            shape: nanmean_or_nan(
                [
                    r["xy_rmse"]
                    for r in summary_rows
                    if r["path_shape"] == shape and r["survived"]
                ]
            )
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
