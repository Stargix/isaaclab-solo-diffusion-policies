#!/usr/bin/env python3
"""Train the path-conditioned Solo12 diffusion policy with DPPO."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Direct DPPO fine-tuning of a Phase-A diffusion policy.")
parser.add_argument("--task", default="solo12-dppo-diffusion-rl-v0")
parser.add_argument("--checkpoint", required=True, help="Phase-A checkpoint, or a DPPO checkpoint to resume.")
parser.add_argument("--output_dir", default=None)
parser.add_argument("--run_name", default="dppo_path_pose_speed_v1")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=1000, help="Additional PPO iterations to run.")
parser.add_argument("--rollout_chunks", type=int, default=32)
parser.add_argument("--episode_length_s", type=float, default=24.0)
parser.add_argument("--save_interval", type=int, default=25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--inference_steps", type=int, default=10)
parser.add_argument("--finetune_denoising_steps", type=int, default=5)
parser.add_argument("--exec_horizon", type=int, default=4)
parser.add_argument("--min_denoising_std", type=float, default=0.10)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--gamma_denoising", type=float, default=0.99)
parser.add_argument("--actor_lr", type=float, default=1.0e-5)
parser.add_argument("--critic_lr", type=float, default=1.0e-3)
parser.add_argument("--clip_ratio_base", type=float, default=1.0e-3)
parser.add_argument("--clip_ratio", type=float, default=1.0e-2)
parser.add_argument("--target_kl", type=float, default=0.02)
parser.add_argument("--update_epochs", type=int, default=5)
parser.add_argument("--minibatch_size", type=int, default=8192)
parser.add_argument("--critic_minibatch_size", type=int, default=4096)
parser.add_argument("--critic_warmup_iterations", type=int, default=10)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb_project", default="solo12-dppo")
parser.add_argument("--wandb_entity", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from scripts.dppo_diffusion_rl.checkpointing import load_policy_checkpoint, restore_training_state
from scripts.dppo_diffusion_rl.config import DPPOConfig
from scripts.dppo_diffusion_rl.critic import ValueCritic
from scripts.dppo_diffusion_rl.isaaclab_env import CRITIC_FEATURE_DIM
from scripts.dppo_diffusion_rl.ppo import DPPOUpdater
from scripts.dppo_diffusion_rl.trainer import DPPOTrainer


def _output_directory() -> Path:
    if args_cli.output_dir:
        return Path(args_cli.output_dir).resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (Path(__file__).resolve().parent / "runs" / f"{timestamp}_{args_cli.run_name}").resolve()


def _dppo_config() -> DPPOConfig:
    return DPPOConfig(
        inference_steps=args_cli.inference_steps,
        finetune_denoising_steps=args_cli.finetune_denoising_steps,
        exec_horizon=args_cli.exec_horizon,
        min_denoising_std=args_cli.min_denoising_std,
        gamma=args_cli.gamma,
        gae_lambda=args_cli.gae_lambda,
        gamma_denoising=args_cli.gamma_denoising,
        actor_lr=args_cli.actor_lr,
        critic_lr=args_cli.critic_lr,
        clip_ratio_base=args_cli.clip_ratio_base,
        clip_ratio=args_cli.clip_ratio,
        target_kl=args_cli.target_kl,
        update_epochs=args_cli.update_epochs,
        minibatch_size=args_cli.minibatch_size,
        critic_minibatch_size=args_cli.critic_minibatch_size,
        critic_warmup_iterations=args_cli.critic_warmup_iterations,
    )


def main() -> None:
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    checkpoint_path = Path(args_cli.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir = _output_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    dppo_cfg = _dppo_config()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    if args_cli.episode_length_s <= 0.0:
        raise ValueError("episode_length_s must be positive.")
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = torch.device(env.unwrapped.device)
    source_checkpoint, policy, start_iteration = load_policy_checkpoint(
        checkpoint_path, device, dppo_cfg
    )
    critic_input_dim = policy.cfg.history * (
        policy.cfg.proprio_dim + policy.cfg.action_hist_dim + policy.cfg.goal_dim
    ) + CRITIC_FEATURE_DIM
    critic = ValueCritic(critic_input_dim).to(device)
    updater = DPPOUpdater(policy, critic, dppo_cfg)
    restore_training_state(source_checkpoint, critic, updater)

    metadata = {
        "command": vars(args_cli),
        "dppo": dppo_cfg.to_dict(),
        "checkpoint": str(checkpoint_path),
        "start_iteration": start_iteration,
        "critic_input_dim": critic_input_dim,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, default=str)

    logger = None
    if args_cli.wandb:
        import wandb

        logger = wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=args_cli.run_name,
            config=metadata,
            dir=str(output_dir),
        )

    trainable = sum(parameter.numel() for parameter in policy.policy.model.parameters() if parameter.requires_grad)
    print(f"[INFO] DPPO output: {output_dir}")
    print(
        f"[INFO] envs={args_cli.num_envs} chunks/update={args_cli.rollout_chunks} "
        f"exec_horizon={dppo_cfg.exec_horizon} K={dppo_cfg.inference_steps} "
        f"K_finetune={dppo_cfg.finetune_denoising_steps} actor_parameters={trainable:,}"
    )
    trainer = DPPOTrainer(
        env=env,
        policy=policy,
        critic=critic,
        updater=updater,
        source_checkpoint=source_checkpoint,
        source_checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        rollout_chunks=args_cli.rollout_chunks,
        start_iteration=start_iteration,
        save_interval=args_cli.save_interval,
        logger=logger,
    )
    try:
        trainer.run(args_cli.iterations)
    finally:
        env.close()
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
