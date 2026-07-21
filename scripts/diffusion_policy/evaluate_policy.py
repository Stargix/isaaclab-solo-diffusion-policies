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
parser.add_argument("--path_height", type=float, default=None,
                    help="Route height in metres. Defaults to the checkpoint dataset's desired_base_height.")
parser.add_argument("--warmup_steps", type=int, default=25)
parser.add_argument("--num_inference_steps", type=int, default=None)
parser.add_argument("--exec_horizon", type=int, default=8)
parser.add_argument("--guidance_scale", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
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
    build_holonomic_goal_from_path,
    build_path_guidance_goal_from_path,
)
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
    time_indexed: bool = False
    demo_name: str | None = None
    reference_command_b: np.ndarray | None = None
    guidance_w: np.ndarray | None = None
    guidance_cumulative: np.ndarray | None = None
    terminal_step: int | None = None


@dataclass
class Scenario:
    scenario_id: int
    repeat: int
    path_shape: str
    speed: float
    demo_name: str | None = None


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


def dataset_hashes(config: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for raw_path in config.get("dataset", {}).get("hdf5_paths", []):
        path = Path(raw_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        if path.is_file():
            output[str(path.resolve())] = sha256_file(path)
    return output


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
    if args_cli.reference_replay_dataset:
        # For holonomic evaluation, we need at least 76 extra lookahead steps (1.52s) to prevent the goal lookahead horizon check from exceeding the reference endpoint.
        minimum = max(2, int(round(args_cli.duration_s * 50.0)) + 76)
        fragments = load_reference_fragments(
            args_cli.reference_replay_dataset, args_cli.reference_replay_demos, minimum
        )
        return [
            Scenario(index, 0, "reference_replay", speed, demo_name)
            for index, (demo_name, _, _, _, _, speed) in enumerate(fragments)
        ]
    scenarios = []
    shapes = list(args_cli.path_shapes)
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
        checkpoint_path,
        device,
        expected_policy_kind=(
            "spatial_time_preview_ddpm", "spatial_reference_path_ddpm", "holonomic_reference_path_ddpm",
            "path_guidance_terminal_ddpm", "spatial_hindsight_geometry_ddpm",
        ),
    )
    config = checkpoint["config"]
    if args_cli.path_height is None:
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
    else:
        path_height = float(args_cli.path_height)
    if not 0.10 <= path_height <= 0.40:
        raise ValueError(f"--path_height must be in [0.10, 0.40], got {path_height}.")
    print(f"[INFO] Evaluation route height: {path_height:.4f} m")
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
    goal_representation = config["dataset"].get("goal_representation", "path11")
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
            pts, yaws = build_random_polyline_path(start_pos[i], start_quat[i])
        else:
            raise ValueError(f"Unknown shape: {s.path_shape}")
        # Analytic route builders use walk height by default. Match the
        # posture support of the checkpoint unless explicitly overridden.
        pts[:, 2] = path_height
        
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

    # Track metrics
    trajectories = {i: {"ref": [], "actual": []} for i in range(num_envs)}
    tracking_errors = []
    achieved_speeds = []
    achieved_planar_speeds = []
    achieved_tangent_speeds = []
    achieved_heights = []
    achieved_yaws = []
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
                raw_env, path_plans, speeds, path_progress, goal_horizon_steps, waypoint_time_offsets_s, dt, v_req_clip, device,
                goal_representation,
                reference_step=step,
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
        planar_speed = torch.linalg.vector_norm(robot.root_lin_vel_b[:, :2], dim=-1).cpu().numpy()
        vel_world = robot.root_lin_vel_w[:, :2].cpu().numpy()
        height = robot.root_pos_w[:, 2].cpu().numpy()
        quat = robot.root_quat_w.cpu().numpy()
        yaw = np.asarray([robot_yaw_w(value) for value in quat], dtype=np.float32)
        a_delta = torch.sqrt(torch.mean(torch.square(action - previous_for_delta), dim=-1)).cpu().numpy()
        previous_for_delta = action.clone()

        step_errors = []
        tangent_speeds = []
        for i in range(num_envs):
            plan = path_plans[i]
            closest_idx = (
                min(step, len(plan.path_w) - 1)
                if plan.time_indexed
                else advance_path_progress(plan.path_w, pos_w[i], int(path_progress[i]))
            )
            err = float(np.linalg.norm(pos_w[i, :2] - plan.path_w[closest_idx, :2]))
            step_errors.append(err)
            tangent_idx = min(closest_idx + 1, len(plan.path_w) - 1)
            tangent = plan.path_w[tangent_idx, :2] - plan.path_w[max(0, tangent_idx - 1), :2]
            tangent_norm = float(np.linalg.norm(tangent))
            tangent = tangent / tangent_norm if tangent_norm > 1.0e-6 else np.zeros(2, dtype=np.float32)
            tangent_speeds.append(float(np.dot(vel_world[i], tangent)))

            if not failures[i]:
                # Track actual vs reference relative to spawn
                trajectories[i]["actual"].append(pos_w[i, :2] - start_pos[i, :2])
                ref_xy = plan.path_w[closest_idx, :2] - start_pos[i, :2]
                trajectories[i]["ref"].append(ref_xy)

        tracking_errors.append(step_errors)
        achieved_speeds.append(vx)
        achieved_planar_speeds.append(planar_speed)
        achieved_tangent_speeds.append(np.asarray(tangent_speeds, dtype=np.float32))
        achieved_heights.append(height)
        achieved_yaws.append(yaw)
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
    achieved_planar_speeds = np.stack(achieved_planar_speeds, axis=0)
    achieved_tangent_speeds = np.stack(achieved_tangent_speeds, axis=0)
    achieved_heights = np.stack(achieved_heights, axis=0)
    achieved_yaws = np.stack(achieved_yaws, axis=0)
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
        planar_speeds_ach = achieved_planar_speeds[:, i][mask]
        tangent_speeds_ach = achieved_tangent_speeds[:, i][mask]
        hgts = achieved_heights[:, i][mask]
        yaws_ach = achieved_yaws[:, i][mask]
        tls = tilts[:, i][mask]
        adeltas = action_deltas[:, i][mask]

        xy_rmse = float(np.sqrt(np.mean(np.square(xy_errs)))) if xy_errs.size else math.nan
        plan = path_plans[i]
        target_height = float(plan.path_w[-1, 2])
        h_rmse = float(np.sqrt(np.mean(np.square(hgts - target_height)))) if hgts.size else math.nan
        survived = not bool(failures[i])
        if plan.time_indexed and plan.terminal_step is not None:
            terminal_index = int(plan.terminal_step)
            target_progress_m = float(plan.cumulative_lengths[terminal_index] - plan.cumulative_lengths[0])
        else:
            # Analytic paths are longer than a finite evaluation episode at low
            # speed.  Compare against the point reachable in v*t, not always
            # against the global endpoint of the polyline.
            target_progress_m = min(float(s.speed) * float(args_cli.duration_s), float(plan.cumulative_lengths[-1]))
            target_arc = float(plan.cumulative_lengths[0] + target_progress_m)
            terminal_index = int(np.clip(np.searchsorted(plan.cumulative_lengths, target_arc, side="left"), 0, len(plan.path_w) - 1))
        terminal_index = int(np.clip(terminal_index, 0, len(plan.path_w) - 1))
        if trajectories[i]["actual"]:
            final_xy = np.asarray(trajectories[i]["actual"][-1]) + start_pos[i, :2]
            terminal_position_error = float(np.linalg.norm(final_xy - plan.path_w[terminal_index, :2]))
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

        summary_rows.append({
            "scenario_id": s.scenario_id,
            "repeat": s.repeat,
            "path_shape": s.path_shape,
            "reference_demo": s.demo_name or "",
            "requested_speed": s.speed,
            "reference_curvature_abs_mean": mean_abs_path_curvature(path_plans[i].path_w),
            "survived": survived,
            "time_to_failure_s": float(failure_step[i] * dt) if failures[i] else args_cli.duration_s,
            "xy_rmse": xy_rmse,
            "height_rmse": h_rmse,
            "target_progress_m": target_progress_m,
            "target_path_index": terminal_index,
            "terminal_position_error_m": terminal_position_error,
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
            
            # Plot actual paths for different speeds
            speeds_list = sorted(list({scenario.speed for scenario in scenarios}))
            for speed_idx, speed in enumerate(speeds_list):
                for i, s in enumerate(scenarios):
                    if s.path_shape == shape and s.speed == speed and s.repeat == 0:
                        act_traj = np.array(trajectories[i]["actual"])
                        color = colors_list[speed_idx % len(colors_list)]
                        ax.plot(act_traj[:, 0], act_traj[:, 1], color=color, alpha=0.9, linewidth=2.0, label=f"{speed} m/s", zorder=4)
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

    # 2. Tracking error and speed plot
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
            vals = [r["achieved_tangent_speed_mean"] for r in subset if r["requested_speed"] == v]
            speeds_ach.append(np.mean(vals) if vals else np.nan)
            speeds_std.append(np.std(vals) if vals else np.nan)
        
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
    axes[0].set_ylabel("Achieved Speed [m/s]", fontsize=10, labelpad=8)
    axes[0].set_title("Speed Tracking Accuracy", fontsize=12, fontweight="bold", pad=12, color="#222222")
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
    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "task": args_cli.task,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "git_commit": current_git_commit(),
        "config": config,
        "dataset_sha256": dataset_hashes(config),
        "policy_kind": checkpoint.get("policy_kind", "unknown"),
        "seed": args_cli.seed,
        "duration_s": args_cli.duration_s,
        "num_scenarios": len(scenarios),
        "num_inference_steps": inference_steps,
        "exec_horizon": args_cli.exec_horizon,
        "goal_horizon_steps": goal_horizon_steps,
        "goal_representation": goal_representation,
        "waypoint_time_offsets_s": waypoint_time_offsets_s,
        "speeds_evaluated": list(args_cli.speeds),
        "reference_replay_dataset": (
            str(Path(args_cli.reference_replay_dataset).resolve()) if args_cli.reference_replay_dataset else None
        ),
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
