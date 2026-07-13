#!/usr/bin/env python3
"""Evaluate the command-conditioned DiffuseLoco baseline in closed loop."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RSL_RL_DIR = _PROJECT_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the command-conditioned DiffuseLoco baseline.")
parser.add_argument("--task", type=str, default="solo12-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    metavar=("VX", "VY", "WZ"),
    default=(0.4, 0.0, 0.0),
    help="Body-frame velocity command [vx, vy, yaw_rate].",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_inference_steps", type=int, default=None)
parser.add_argument("--exec_horizon", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=25)
parser.add_argument("--guidance_scale", type=float, default=1.0)
parser.add_argument("--real_time_viewer", action="store_true", default=True)
parser.add_argument("--no_real_time_viewer", action="store_false", dest="real_time_viewer")

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
from train.obs_utils import proprio_from_env_tensors


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
    *,
    proprio: torch.Tensor,
    previous_action: torch.Tensor,
    command: torch.Tensor,
) -> None:
    """Append one pre-action tuple, matching the offline delayed-I/O contract."""

    slide_buffer(proprio_buffer, proprio)
    slide_buffer(action_buffer, previous_action)
    slide_buffer(command_buffer, command)


def model_config(config: dict, inference_steps: int) -> Solo12DiffusionPolicyConfig:
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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_training_checkpoint(
        checkpoint_path, device, expected_policy_kind="diffuseloco_command_ddpm"
    )
    config = checkpoint["config"]
    inference_steps = resolve_inference_steps(args_cli.num_inference_steps, config["diffusion"])
    policy_cfg = model_config(config, inference_steps)
    if policy_cfg.goal_dim != 3:
        raise ValueError(f"Command baseline requires goal_dim=3, got {policy_cfg.goal_dim}.")
    future_horizon = policy_cfg.prediction_horizon - policy_cfg.execution_offset
    if not 1 <= args_cli.exec_horizon <= future_horizon:
        raise ValueError(f"exec_horizon must be in [1, {future_horizon}].")

    policy = Solo12DiffusionPolicy(policy_cfg)
    policy.load_state_dict(checkpoint["ema_model_state_dict"])
    policy.set_normalizer_stats(checkpoint["normalizer_stats"])
    policy.to(device).eval()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = 1.0e9
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
    dt = env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt
    command = torch.tensor(args_cli.command, device=device, dtype=torch.float32).repeat(args_cli.num_envs, 1)
    if hasattr(raw_env, "_commands"):
        raw_env._commands[:, :3] = command

    proprio_buffer = torch.zeros(
        (args_cli.num_envs, policy_cfg.history, policy_cfg.proprio_dim), device=device
    )
    action_buffer = torch.zeros(
        (args_cli.num_envs, policy_cfg.history, policy_cfg.action_hist_dim), device=device
    )
    command_buffer = command[:, None, :].repeat(1, policy_cfg.history, 1)
    previous_action = torch.zeros((args_cli.num_envs, policy_cfg.action_dim), device=device)
    stand_action = torch.zeros_like(previous_action)

    vec_env.reset()
    for _ in range(max(args_cli.warmup_steps, policy_cfg.history + 1)):
        pre_action_proprio = get_proprio(raw_env, joint_ids)
        update_history(
            proprio_buffer,
            action_buffer,
            command_buffer,
            proprio=pre_action_proprio,
            previous_action=previous_action,
            command=command,
        )
        vec_env.step(stand_action)
        previous_action = stand_action.clone()

    print(
        f"[INFO] baseline command={tuple(args_cli.command)} control={1 / dt:.0f}Hz "
        f"K={inference_steps} trajectory={policy_cfg.prediction_horizon} "
        f"execute_from={policy_cfg.execution_offset} exec_horizon={args_cli.exec_horizon}"
    )
    current_chunk: torch.Tensor | None = None
    chunk_index = 0
    step_count = 0
    while simulation_app.is_running():
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

            pre_action_proprio = get_proprio(raw_env, joint_ids)
            _, _, dones, _ = vec_env.step(action)
            update_history(
                proprio_buffer,
                action_buffer,
                command_buffer,
                proprio=pre_action_proprio,
                previous_action=previous_action,
                command=command,
            )
            previous_action = action.clone()

            done_indices = torch.nonzero(dones, as_tuple=False).flatten()
            for env_idx in done_indices.tolist():
                current = get_proprio(raw_env, joint_ids)[env_idx]
                proprio_buffer[env_idx] = current
                action_buffer[env_idx].zero_()
                command_buffer[env_idx] = command[env_idx]
                previous_action[env_idx].zero_()
                current_chunk = None
                chunk_index = 0

        if step_count % 50 == 0:
            root = raw_env._robot.data.root_pos_w[0]
            print(f"[STEP {step_count:05d}] base=({root[0]:+.2f}, {root[1]:+.2f}, {root[2]:+.2f})")
        step_count += 1
        if args_cli.real_time_viewer:
            time.sleep(dt)

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
