#!/usr/bin/env python3
"""Plot a single S-curve tracking run at 0.4 m/s."""

import argparse
import sys
from typing import Any
import numpy as np
import torch
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RSL_RL_DIR = _PROJECT_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Plot a single S-curve tracking run at 0.4 m/s.")
parser.add_argument("--task", type=str, default="solo12-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--duration_s", type=float, default=50.0)
parser.add_argument("--warmup_steps", type=int, default=25)
parser.add_argument("--exec_horizon", type=int, default=8)
parser.add_argument("--guidance_scale", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True
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


def robot_yaw_w(quat_wxyz: np.ndarray) -> float:
    return yaw_from_rotmat(quat_wxyz_to_rotmat(quat_wxyz))


def build_s_curve_path(start_pos: np.ndarray, start_quat: np.ndarray, length_m: float = 10.0, num_points: int = 250) -> tuple[np.ndarray, np.ndarray]:
    yaw = robot_yaw_w(start_quat)
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    path_points = []
    for s in np.linspace(0.0, length_m, num_points):
        lx = s
        ly = 0.3 * np.sin(2.0 * np.pi * s / 6.0)
        wx = float(start_pos[0] + cos_y * lx - sin_y * ly)
        wy = float(start_pos[1] + sin_y * lx + cos_y * ly)
        path_points.append([wx, wy, 0.2932])
    
    yaws = [yaw]
    for idx in range(1, len(path_points)):
        yaws.append(np.arctan2(path_points[idx][1] - path_points[idx - 1][1], path_points[idx][0] - path_points[idx - 1][0]))
    return np.array(path_points, dtype=np.float32), np.array(yaws, dtype=np.float32)


def get_proprio_30d(raw_env: Any, joint_ids: slice) -> torch.Tensor:
    robot = raw_env._robot
    return proprio_from_env_tensors(
        robot.data.joint_pos[:, joint_ids],
        robot.data.joint_vel[:, joint_ids],
        robot.data.root_ang_vel_b,
        robot.data.projected_gravity_b,
    )



@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = "cuda:0"
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

    checkpoint_path = Path(args_cli.checkpoint).resolve()
    device = torch.device("cuda")

    checkpoint = load_training_checkpoint(checkpoint_path, device, expected_policy_kind=("spatial_time_preview_ddpm", "spatial_reference_path_ddpm"))
    config = checkpoint["config"]
    policy_cfg = _model_cfg_from_checkpoint(config)
    policy_cfg.num_inference_steps = resolve_inference_steps(None, config["diffusion"])

    policy = Solo12DiffusionPolicy(policy_cfg)
    policy.load_state_dict(checkpoint["ema_model_state_dict"])
    policy.set_normalizer_stats(checkpoint["normalizer_stats"])
    policy.to(device).eval()

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    joint_ids = raw_env._joint_ids
    dt = float(raw_env.step_dt)
    duration_steps = max(1, int(round(args_cli.duration_s / dt)))

    goal_horizon_steps = config["dataset"]["goal_horizon_steps"]
    waypoint_time_offsets_s = tuple(config["dataset"]["waypoint_time_offsets_s"])
    v_req_clip = config["dataset"].get("v_req_clip", 2.0)

    vec_env.reset()
    start_pos = raw_env._robot.data.root_pos_w.cpu().numpy()[0]
    start_quat = raw_env._robot.data.root_quat_w.cpu().numpy()[0]

    pts, yaws = build_s_curve_path(start_pos, start_quat)
    cumulative_lengths = cumulative_xy_lengths(pts)

    proprio_buffer = torch.zeros((1, policy_cfg.history, policy_cfg.proprio_dim), device=device)
    action_buffer = torch.zeros((1, policy_cfg.history, policy_cfg.action_hist_dim), device=device)
    goal_buffer = torch.zeros((1, policy_cfg.history, policy_cfg.goal_dim), device=device)
    previous_action = torch.zeros((1, policy_cfg.action_dim), device=device)
    stand_action = torch.zeros_like(previous_action)

    path_progress = np.zeros(1, dtype=np.int32)

    # Warmup
    for _ in range(max(args_cli.warmup_steps, policy_cfg.history + 1)):
        proprio = get_proprio_30d(raw_env, joint_ids)
        goals = compute_goals(raw_env, pts, yaws, cumulative_lengths, path_progress, goal_horizon_steps, waypoint_time_offsets_s, dt, 0.4, v_req_clip, device)
        update_history(proprio_buffer, action_buffer, goal_buffer, proprio, previous_action, goals)
        vec_env.step(stand_action)
        previous_action = stand_action.clone()

    actual_xy = []
    current_chunk = None
    chunk_index = 0

    for step in range(duration_steps):
        with torch.inference_mode():
            if current_chunk is None or chunk_index == 0:
                full_trajectory = policy.predict_action_denormalized(proprio_buffer, action_buffer, goal_buffer, guidance_scale=args_cli.guidance_scale)
                current_chunk = policy.executable_chunk(full_trajectory, args_cli.exec_horizon)

            action = current_chunk[0, chunk_index]
            chunk_index = (chunk_index + 1) % args_cli.exec_horizon
            if agent_cfg.clip_actions is not None:
                action = torch.clamp(action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            
            proprio = get_proprio_30d(raw_env, joint_ids)
            goals = compute_goals(raw_env, pts, yaws, cumulative_lengths, path_progress, goal_horizon_steps, waypoint_time_offsets_s, dt, 0.4, v_req_clip, device)
            _, _, dones, _ = vec_env.step(action.unsqueeze(0))
            update_history(proprio_buffer, action_buffer, goal_buffer, proprio, previous_action, goals)
            previous_action = action.clone().unsqueeze(0)

        pos_w = raw_env._robot.data.root_pos_w.cpu().numpy()[0]
        if dones[0].item():
            break
        actual_xy.append(pos_w[:2] - start_pos[:2])

    # Plot S-curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ref_xy = pts[:, :2] - start_pos[:2]
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", linewidth=2.5, label="Reference Path")
    
    if actual_xy:
        act_xy = np.array(actual_xy)
        ax.plot(act_xy[:, 0], act_xy[:, 1], color="tab:green", linewidth=2.0, label="Traversed Path (0.4 m/s)")
        
    ax.set_title("Solo12 S-Curve Trajectory Tracking at 0.4 m/s", fontsize=14, fontweight="bold")
    ax.set_xlabel("Relative X [m]", fontsize=12)
    ax.set_ylabel("Relative Y [m]", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    artifact_path = "C:/Users/11ser/.gemini/antigravity-ide/brain/745e3943-ff02-4e96-8788-f76a581bfa11/s_curve_0.4_tracking.png"
    fig.savefig(artifact_path, dpi=200)
    plt.close(fig)
    print(f"[DONE] Trajectory plot saved as: {artifact_path}")
    vec_env.close()


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


def compute_goals(raw_env: Any, path_w: np.ndarray, yaws_w: np.ndarray, cumulative_lengths: np.ndarray, path_progress: np.ndarray, goal_horizon_steps: int, waypoint_time_offsets_s: tuple[float, float, float], dt: float, speed: float, v_req_clip: float, device: torch.device) -> torch.Tensor:
    robot = raw_env._robot
    pos_w = robot.data.root_pos_w.cpu().numpy()
    quat_w = robot.data.root_quat_w.cpu().numpy()
    goals = []
    start_idx = advance_path_progress(path_w, pos_w[0], int(path_progress[0]))
    path_progress[0] = start_idx
    goals.append(build_goal_from_path(path_w, cumulative_lengths, yaws_w, pos_w[0], quat_w[0], goal_horizon_steps=goal_horizon_steps, dt=dt, speed=speed, start_idx=start_idx, waypoint_time_offsets_s=waypoint_time_offsets_s, v_req_clip=v_req_clip))
    return torch.from_numpy(np.stack(goals, axis=0)).to(device)


def update_history(proprio_buffer: torch.Tensor, action_buffer: torch.Tensor, goal_buffer: torch.Tensor, proprio: torch.Tensor, previous_action: torch.Tensor, goal: torch.Tensor) -> None:
    slide_buffer(proprio_buffer, proprio)
    slide_buffer(action_buffer, previous_action)
    slide_buffer(goal_buffer, goal)


def slide_buffer(buffer: torch.Tensor, value: torch.Tensor) -> None:
    buffer[:, :-1] = buffer[:, 1:].clone()
    buffer[:, -1] = value


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
