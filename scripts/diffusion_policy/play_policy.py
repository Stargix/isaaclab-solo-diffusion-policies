# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Closed-loop evaluation of a trained Solo12 Diffusion Policy in Isaac Lab.

Example:
    python scripts/diffusion_policy/play_policy.py --task solo12-v0 \
        --checkpoint scripts/diffusion_policy/runs/walk_crouch_diffusion/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_RSL_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_UPSTREAM_RSL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_RSL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run closed-loop Diffusion Policy evaluation.")
parser.add_argument("--task", type=str, default="solo12-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale (1.0 = faster, no double forward).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_inference_steps", type=int, default=None, help="Denoising steps; default from checkpoint.")
parser.add_argument("--path_file", type=str, default=None, help="Optional .npy path [N,3] or [N,4].")
parser.add_argument("--goal_horizon_steps", type=int, default=None, help="Override the training lookahead horizon.")
parser.add_argument("--v_req_clip", type=float, default=None)
parser.add_argument("--desired_speed", type=float, default=0.4, help="Speed used for spatial goal lookahead.")
parser.add_argument("--warmup_steps", type=int, default=25, help="Stand-hold steps before activating the policy.")
parser.add_argument("--temporal_blend_alpha", type=float, default=1.0, help="1.0 = no chunk blending (less tremor).")
parser.add_argument("--exec_horizon", type=int, default=1, help="RHC: execute N chunk steps before replanning (DiffuseLoco uses 1).")
parser.add_argument("--real_time_viewer", action="store_true", default=True, help="Sleep to match sim dt (GUI).")
parser.add_argument("--no_real_time_viewer", action="store_false", dest="real_time_viewer")
parser.add_argument("--compile_policy", action="store_true", help="Try torch.compile on the policy.")
parser.add_argument("--force_walk_goal", action="store_true", help="Override conditioning goals to force walk commands.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = False if args_cli.headless is None else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
from train.checkpoint_utils import load_training_checkpoint
from train.config import resolve_inference_steps
from train.geometry import cumulative_xy_lengths, quat_wxyz_to_rotmat, yaw_from_rotmat
from train.goal_builder import build_goal_batch_from_path
from train.normalization import NormalizerStats
from train.obs_utils import proprio_from_env_tensors


@dataclass
class PathPlan:
    path_w: np.ndarray
    yaws_w: np.ndarray
    cumulative_lengths: np.ndarray
    is_custom: bool

    def rebuild_from_robot(self, pos_w: np.ndarray, quat_w: np.ndarray, env_idx: int = 0) -> None:
        if self.is_custom:
            return
        self.path_w, self.yaws_w = build_default_path(pos_w[env_idx], quat_w[env_idx])
        self.cumulative_lengths = cumulative_xy_lengths(self.path_w)


def robot_yaw_w(quat_wxyz: np.ndarray) -> float:
    return yaw_from_rotmat(quat_wxyz_to_rotmat(quat_wxyz))


def build_default_path(
    start_pos_w: np.ndarray,
    start_quat_w: np.ndarray,
    *,
    length_m: float = 3.0,
    num_points: int = 60,
    crouch_start_m: float = 1.5,
    walk_z: float = 0.24,
    crouch_z: float = 0.16,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a straight path in the robot's forward (yaw) direction."""

    yaw = robot_yaw_w(start_quat_w)
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    path_points: list[list[float]] = []
    yaws: list[float] = []
    for s in np.linspace(0.0, length_m, num_points):
        x = float(start_pos_w[0] + cos_y * s)
        y = float(start_pos_w[1] + sin_y * s)
        z = walk_z if s < crouch_start_m else crouch_z
        path_points.append([x, y, z])
        yaws.append(yaw)
    return np.array(path_points, dtype=np.float32), np.array(yaws, dtype=np.float32)


def load_or_build_path(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    path_file: str | None,
) -> PathPlan:
    if path_file is not None and os.path.exists(path_file):
        print(f"[INFO] Loading path: {path_file}")
        path_array = np.load(path_file)
        if path_array.shape[1] == 3:
            yaws = [0.0]
            for idx in range(1, len(path_array)):
                yaws.append(
                    np.arctan2(path_array[idx, 1] - path_array[idx - 1, 1], path_array[idx, 0] - path_array[idx - 1, 0])
                )
            yaws_w = np.array(yaws, dtype=np.float32)
            path_w = path_array.astype(np.float32)
        else:
            path_w = path_array[:, :3].astype(np.float32)
            yaws_w = path_array[:, 3].astype(np.float32)
        return PathPlan(path_w, yaws_w, cumulative_xy_lengths(path_w), is_custom=True)

    print("[INFO] Generating default 3 m walk->crouch path aligned to robot yaw.")
    path_w, yaws_w = build_default_path(start_pos[0], start_quat[0])
    return PathPlan(path_w, yaws_w, cumulative_xy_lengths(path_w), is_custom=False)


def get_proprio_30d(raw_env: Any, joint_ids: slice) -> torch.Tensor:
    robot = raw_env._robot
    return proprio_from_env_tensors(
        robot.data.joint_pos[:, joint_ids],
        robot.data.joint_vel[:, joint_ids],
        robot.data.root_ang_vel_b,
        robot.data.projected_gravity_b,
    )


def update_delayed_buffers(
    proprio_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
    goal_buffer: torch.Tensor,
    *,
    proprio_value: torch.Tensor,
    action_value: torch.Tensor,
    goal_value: torch.Tensor,
) -> None:
    slide_buffer(proprio_buffer, proprio_value)
    slide_buffer(action_buffer, action_value)
    slide_buffer(goal_buffer, goal_value)


def run_warmup(
    vec_env: Any,
    raw_env: Any,
    joint_ids: slice,
    proprio_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
    goal_buffer: torch.Tensor,
    path_w: np.ndarray,
    yaws_w: np.ndarray,
    cumulative_lengths: np.ndarray,
    path_progress: np.ndarray,
    *,
    warmup_steps: int,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    v_req_clip: float,
    device: torch.device,
) -> None:
    """Fill delayed history buffers with stand-hold dynamics."""

    path_progress[:] = 0
    print(f"[INFO] Warmup: {warmup_steps} steps with stand-hold actions before DP control.")
    stand_action = torch.zeros((proprio_buffer.shape[0], 12), device=device)
    prev_applied_action = torch.zeros_like(stand_action)
    for _ in range(warmup_steps):
        proprio_value = get_proprio_30d(raw_env, joint_ids)
        goal_value = compute_goals(
            raw_env,
            path_w,
            yaws_w,
            cumulative_lengths,
            path_progress,
            goal_horizon_steps=goal_horizon_steps,
            dt=dt,
            speed=speed,
            v_req_clip=v_req_clip,
            device=device,
        )
        update_delayed_buffers(
            proprio_buffer,
            action_buffer,
            goal_buffer,
            proprio_value=proprio_value,
            action_value=prev_applied_action,
            goal_value=goal_value,
        )
        vec_env.step(stand_action)
        prev_applied_action = stand_action.clone()


def reset_env_history(
    env_idx: int,
    proprio_buffer: torch.Tensor,
    action_buffer: torch.Tensor,
    goal_buffer: torch.Tensor,
    raw_env: Any,
    joint_ids: slice,
    path_w: np.ndarray,
    yaws_w: np.ndarray,
    cumulative_lengths: np.ndarray,
    path_progress: np.ndarray,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    v_req_clip: float,
    device: torch.device,
) -> None:
    """Initialize history buffers for a single environment after reset without stepping the physics simulator."""
    proprio_value = get_proprio_30d(raw_env, joint_ids)[env_idx]
    goal_value = compute_goals(
        raw_env,
        path_w,
        yaws_w,
        cumulative_lengths,
        path_progress,
        goal_horizon_steps=goal_horizon_steps,
        dt=dt,
        speed=speed,
        v_req_clip=v_req_clip,
        device=device,
    )[env_idx]

    proprio_buffer[env_idx, :, :] = proprio_value.unsqueeze(0)
    action_buffer[env_idx, :, :] = 0.0
    goal_buffer[env_idx, :, :] = goal_value.unsqueeze(0)


def slide_buffer(buffer: torch.Tensor, new_value: torch.Tensor) -> None:
    buffer[:, :-1] = buffer[:, 1:].clone()
    buffer[:, -1] = new_value


def goal_zscore(goal: torch.Tensor, stats: NormalizerStats) -> torch.Tensor:
    mean = torch.as_tensor(stats.goal.mean, device=goal.device, dtype=goal.dtype)
    std = torch.as_tensor(stats.goal.std, device=goal.device, dtype=goal.dtype)
    return (goal - mean) / std


def compute_goals(
    raw_env: Any,
    path_w: np.ndarray,
    yaws_w: np.ndarray,
    cumulative_lengths: np.ndarray,
    path_progress: np.ndarray,
    *,
    goal_horizon_steps: int,
    dt: float,
    speed: float,
    v_req_clip: float,
    device: torch.device,
) -> torch.Tensor:
    robot = raw_env._robot
    pos_w = robot.data.root_pos_w.cpu().numpy()
    quat_w = robot.data.root_quat_w.cpu().numpy()
    goals = build_goal_batch_from_path(
        path_w,
        cumulative_lengths,
        yaws_w,
        pos_w,
        quat_w,
        goal_horizon_steps=goal_horizon_steps,
        dt=dt,
        speed=speed,
        path_progress=path_progress,
        v_req_clip=v_req_clip,
    )
    goals_tensor = torch.from_numpy(goals).to(device)
    if getattr(args_cli, "force_walk_goal", False):
        goals_tensor[:, 0] = 0.4
        goals_tensor[:, 1] = 0.0
        goals_tensor[:, 2] = 0.8
        goals_tensor[:, 3] = 0.0
        goals_tensor[:, 4] = 1.2
        goals_tensor[:, 5] = 0.0
        goals_tensor[:, 6] = 0.8
        goals_tensor[:, 7] = 0.0
        goals_tensor[:, 8] = 0.24
        goals_tensor[:, 9] = 0.0
        goals_tensor[:, 10] = 1.0
    return goals_tensor


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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading checkpoint: {checkpoint_path} ({device})")

    checkpoint = load_training_checkpoint(
        checkpoint_path, device, expected_policy_kind="spatial_hindsight_ddpm"
    )
    config_dict = checkpoint["config"]
    normalizer_stats = NormalizerStats.from_dict(checkpoint["normalizer_stats"])
    infer_steps = resolve_inference_steps(args_cli.num_inference_steps, config_dict["diffusion"])
    policy_cfg = _model_cfg_from_checkpoint(config_dict)
    policy_cfg.num_inference_steps = infer_steps
    goal_horizon_steps = args_cli.goal_horizon_steps or config_dict["dataset"]["goal_horizon_steps"]

    policy = Solo12DiffusionPolicy(policy_cfg)
    policy.load_state_dict(checkpoint["ema_model_state_dict"])
    policy.set_normalizer_stats(normalizer_stats)
    policy.to(device)
    policy.eval()

    if args_cli.compile_policy:
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
            print("[INFO] torch.compile enabled.")
        except Exception as exc:
            print(f"[WARN] torch.compile failed: {exc}")

    history_len = policy_cfg.history
    v_req_clip = args_cli.v_req_clip or config_dict["dataset"].get("v_req_clip", 2.0)
    future_horizon = policy_cfg.prediction_horizon - policy_cfg.execution_offset
    if not 1 <= args_cli.exec_horizon <= future_horizon:
        raise ValueError(f"exec_horizon must be in [1, {future_horizon}].")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = 1.0e9
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
    num_envs = args_cli.num_envs
    dt = env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt

    print(
        f"[INFO] Control @ {1 / dt:.0f} Hz | infer_steps={infer_steps} "
        f"| goal_horizon={goal_horizon_steps} | exec_horizon={args_cli.exec_horizon} "
        f"| guidance={args_cli.guidance_scale}"
    )

    proprio_buffer = torch.zeros((num_envs, history_len, policy_cfg.proprio_dim), device=device)
    action_buffer = torch.zeros((num_envs, history_len, policy_cfg.action_hist_dim), device=device)
    goal_buffer = torch.zeros((num_envs, history_len, policy_cfg.goal_dim), device=device)

    vec_env.reset()
    start_pos = raw_env._robot.data.root_pos_w.cpu().numpy()
    start_quat = raw_env._robot.data.root_quat_w.cpu().numpy()
    time.sleep(0.5)

    path_plan = load_or_build_path(start_pos, start_quat, args_cli.path_file)
    path_progress = np.zeros(num_envs, dtype=np.int32)

    run_warmup(
        vec_env,
        raw_env,
        joint_ids,
        proprio_buffer,
        action_buffer,
        goal_buffer,
        path_plan.path_w,
        path_plan.yaws_w,
        path_plan.cumulative_lengths,
        path_progress,
        warmup_steps=max(args_cli.warmup_steps, history_len + 1),
        goal_horizon_steps=goal_horizon_steps,
        dt=dt,
        speed=args_cli.desired_speed,
        v_req_clip=v_req_clip,
        device=device,
    )

    # Benchmark latency over 50 runs.
    print("[INFO] Benchmarking policy latency over 50 passes...")
    times = []
    with torch.inference_mode():
        for _ in range(50):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = policy.predict_action_denormalized(
                proprio_buffer,
                action_buffer,
                goal_buffer,
                guidance_scale=args_cli.guidance_scale,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    avg_latency = sum(times) / len(times)
    max_latency = max(times)
    print(f"[INFO] Inference latency - Avg: {avg_latency:.1f} ms | Max: {max_latency:.1f} ms (budget @50Hz = 20 ms)")
    if avg_latency > 20.0 * max(1, args_cli.exec_horizon):
        print(
            f"[WARN] Inference is slow. Use --num_inference_steps 4 --exec_horizon 4 "
            f"and --no_real_time_viewer so sim control is not wall-clock limited."
        )

    camera_look_at = np.array([0.0, 0.0, 0.35])
    camera_offset = np.array([-2.0, 0.0, 0.8])

    prev_actions_chunk = None
    exec_step_idx = 0
    current_chunk = None
    step_count = 0
    infer_ms = 0.0
    prev_applied_action = torch.zeros((num_envs, 12), device=device)
    current_goal = goal_buffer[:, -1].clone()

    while simulation_app.is_running():
        with torch.inference_mode():
            if exec_step_idx == 0 or current_chunk is None:
                # Verify that the goal is in distribution before running inference
                latest_goal = goal_buffer[:, -1]
                for env_idx in range(num_envs):
                    dx = latest_goal[env_idx, 6].item()
                    target_height = latest_goal[env_idx, 8].item()
                    v_req = latest_goal[env_idx, 10].item()
                    assert abs(dx) < 3.0, f"Env {env_idx}: dx={dx:.2f} está OOD (entrenamiento max ~3m)"
                    assert 0.10 <= target_height <= 0.40, (
                        f"Env {env_idx}: target_height={target_height:.2f} fuera de rango fisico"
                    )
                    assert 0.0 <= v_req <= 2.0, f"Env {env_idx}: v_req={v_req:.2f} fuera de rango"

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                full_trajectory = policy.predict_action_denormalized(
                    proprio_buffer,
                    action_buffer,
                    goal_buffer,
                    guidance_scale=args_cli.guidance_scale,
                )
                actions_chunk = policy.executable_chunk(full_trajectory, args_cli.exec_horizon)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - t0) * 1000.0

                if prev_actions_chunk is not None and args_cli.temporal_blend_alpha < 1.0 and actions_chunk.shape[1] > 1:
                    alpha = args_cli.temporal_blend_alpha
                    actions_chunk[:, 0] = alpha * actions_chunk[:, 0] + (1.0 - alpha) * prev_actions_chunk[:, 1]
                current_chunk = actions_chunk
                prev_actions_chunk = actions_chunk.clone()

            k = min(exec_step_idx, current_chunk.shape[1] - 1)
            action_step = current_chunk[:, k]
            exec_step_idx = (exec_step_idx + 1) % max(1, args_cli.exec_horizon)

            if agent_cfg.clip_actions is not None:
                action_step = torch.clamp(action_step, -agent_cfg.clip_actions, agent_cfg.clip_actions)

            # Capture the same pre-action tuple used by the offline dataset.
            proprio_before_action = get_proprio_30d(raw_env, joint_ids)
            goal_before_action = compute_goals(
                raw_env,
                path_plan.path_w,
                path_plan.yaws_w,
                path_plan.cumulative_lengths,
                path_progress,
                goal_horizon_steps=goal_horizon_steps,
                dt=dt,
                speed=args_cli.desired_speed,
                v_req_clip=v_req_clip,
                device=device,
            )
            _, _, dones, _ = vec_env.step(action_step)

            update_delayed_buffers(
                proprio_buffer,
                action_buffer,
                goal_buffer,
                proprio_value=proprio_before_action,
                action_value=prev_applied_action,
                goal_value=goal_before_action,
            )
            prev_applied_action = action_step.clone()
            current_goal = goal_buffer[:, -1].clone()

            for i in range(num_envs):
                if dones[i]:
                    print(f"[INFO] Env {i} reset.")
                    path_progress[i] = 0
                    exec_step_idx = 0
                    current_chunk = None
                    prev_applied_action[i] = 0.0
                    reset_pos = raw_env._robot.data.root_pos_w.cpu().numpy()
                    reset_quat = raw_env._robot.data.root_quat_w.cpu().numpy()
                    path_plan.rebuild_from_robot(reset_pos, reset_quat, i)
                    reset_env_history(
                        i,
                        proprio_buffer,
                        action_buffer,
                        goal_buffer,
                        raw_env,
                        joint_ids,
                        path_plan.path_w,
                        path_plan.yaws_w,
                        path_plan.cumulative_lengths,
                        path_progress,
                        goal_horizon_steps=goal_horizon_steps,
                        dt=dt,
                        speed=args_cli.desired_speed,
                        v_req_clip=v_req_clip,
                        device=device,
                    )

        try:
            root_pos = raw_env._robot.data.root_pos_w[0].cpu().numpy()
            raw_env.sim.set_camera_view(eye=root_pos + camera_offset, target=root_pos + camera_look_at)
        except Exception:
            pass

        if step_count % 10 == 0:
            robot_pos = raw_env._robot.data.root_pos_w[0].cpu().numpy()
            robot_z = robot_pos[2]
            goal_z = goal_zscore(current_goal[0], normalizer_stats)
            z_max = float(goal_z.abs().max().item())
            target_dx = current_goal[0, 6].item()
            target_height = current_goal[0, 8].item()
            v_req_val = current_goal[0, 10].item()
            mode_str = "CROUCHING" if robot_z < 0.20 else "WALKING"
            dist_rem = float(np.linalg.norm(path_plan.path_w[-1, :2] - robot_pos[:2]))
            infer_per_step = infer_ms / max(1, args_cli.exec_horizon)
            print(
                f"Step {step_count:04d} | {mode_str:^10s} | Z={robot_z:.3f}m | "
                f"dx={target_dx:+.2f} target_z={target_height:+.2f} | v_req={v_req_val:.2f} | "
                f"goal|z|max={z_max:.2f} | infer={infer_ms:.0f}ms (~{infer_per_step:.0f}/step) | "
                f"prog={path_progress[0]} | dist_rem={dist_rem:.2f}m"
            )

        step_count += 1
        if args_cli.real_time_viewer:
            time.sleep(dt)

    print("[INFO] Evaluation finished.")
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
