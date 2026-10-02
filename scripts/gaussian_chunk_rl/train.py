#!/usr/bin/env python3
"""Matched BC+PPO pilot. The environment/reward/sampler are unchanged WC-v3."""

import argparse
import json
from pathlib import Path
import random
import subprocess
import traceback

import numpy as np
import torch

from scripts.gaussian_chunk_rl.config import GaussianPPOConfig
from scripts.gaussian_chunk_rl.policy import load_bc_policy


def parser_with_app_args():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="ppo_wc_deterministic_pilot")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--rollout_chunks", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--adaptive_actor_lr", action="store_true",
                        help="Adapt the next rollout's actor LR using measured joint KL.")
    parser.add_argument("--min_actor_lr", type=float, default=1e-7)
    parser.add_argument("--initial_std", type=float, default=0.04)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--smoke", action="store_true", help="Require actor updates; no critic-only warmup.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="solo12-gaussian-ppo")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main(args, source, policy, cfg, output):
    # Imports requiring Kit happen only after AppLauncher.
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from scripts.dppo_diffusion_rl.checkpointing import sha256_file, task_config_from_env_cfg
    from scripts.dppo_diffusion_rl.critic import ValueCritic
    from scripts.dppo_diffusion_rl.isaaclab_env import CRITIC_FEATURE_DIM
    from scripts.gaussian_chunk_rl.ppo import GaussianPPOUpdater
    from scripts.gaussian_chunk_rl.trainer import GaussianPPOTrainer

    env_cfg = parse_env_cfg("solo12-dppo-diffusion-rl-v0", device=args.device,
                            num_envs=args.num_envs, use_fabric=True)
    # Exact frozen WC-v3 contract: no private Gaussian-only feasibility rules.
    settings = {
        "seed": args.seed, "episode_length_s": 24.0,
        "goal_representation": "hindsight_geom_profile16",
        "route_distribution": "supported_hybrid_v3", "route_stage": 2,
        "height_profile_stage": None, "route_speed_max_mps": 1.0,
        "speed_budget_max_mps": 1.5, "pace_consistent_preview": False,
        "transition_boundary_min_m": 0.8, "transition_boundary_max_m": 3.2,
        "path_reward_weight": 2.0, "route_cte_rmse_tolerance_m": 0.10,
        "profile_height_reward_weight": 0.75, "profile_height_mae_tolerance_m": 0.04,
    }
    for name, value in settings.items():
        setattr(env_cfg, name, value)
    env_cfg.terrain.terrain_type = "generator"
    env_cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=args.seed, curriculum=False, size=(20.0, 20.0), border_width=0.0,
        num_rows=1, num_cols=1,
        sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)})
    env = gym.make("solo12-dppo-diffusion-rl-v0", cfg=env_cfg)
    logger = None
    try:
        if abs(float(env.unwrapped.step_dt) - 0.02) > 1e-8:
            raise ValueError("The matched task must run at 50 Hz.")
        policy = policy.to(env.unwrapped.device)
        critic_dim = policy.cfg.history * (policy.cfg.proprio_dim + policy.cfg.action_hist_dim
                                           + policy.cfg.goal_dim) + CRITIC_FEATURE_DIM
        critic = ValueCritic(critic_dim).to(env.unwrapped.device)
        updater = GaussianPPOUpdater(policy, critic, cfg)
        output.mkdir(parents=True, exist_ok=True)
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        metadata = {"command": vars(args), "gaussian_ppo": cfg.to_dict(),
                    "source_sha256": sha256_file(args.checkpoint),
                    "task": task_config_from_env_cfg(env_cfg), "git_commit": git.stdout.strip(),
                    "git_dirty": bool(dirty.stdout.strip()), "critic_input_dim": critic_dim,
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "device": str(env.unwrapped.device)}
        (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.wandb:
            import wandb
            logger = wandb.init(project=args.wandb_project, name=args.run_name,
                                config=metadata, dir=str(output))
        trainer = GaussianPPOTrainer(env=env, policy=policy, critic=critic, updater=updater,
                    source_checkpoint=source, source_checkpoint_path=args.checkpoint,
                    output_dir=output, rollout_chunks=args.rollout_chunks,
                    save_interval=args.save_interval, logger=logger)
        trainer.run(args.iterations)
        if args.smoke:
            records = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
            if not any(row["Policy/actor_updates"] > 0 for row in records):
                raise RuntimeError("Smoke did not execute an actor update.")
            if not all(row["Policy/critic_updates"] > 0 for row in records):
                raise RuntimeError("Smoke did not execute critic updates.")
            print("[SMOKE_OK] collection, actor and critic updates, checkpoint save", flush=True)
    finally:
        env.close()
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    args = parser_with_app_args().parse_args()
    for name in ("num_envs", "iterations", "rollout_chunks", "save_interval"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite/mix an existing run: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = GaussianPPOConfig(actor_lr=args.actor_lr, initial_std=args.initial_std,
                           clip_ratio=args.clip_ratio, target_kl=args.target_kl,
                           adaptive_actor_lr=args.adaptive_actor_lr, min_actor_lr=args.min_actor_lr,
                           critic_warmup_iterations=0 if args.smoke else 10)
    # Validate source and config before expensive simulation startup.
    source, policy = load_bc_policy(Path(args.checkpoint).resolve(), args.device, cfg)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    try:
        main(args, source, policy, cfg, output)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
