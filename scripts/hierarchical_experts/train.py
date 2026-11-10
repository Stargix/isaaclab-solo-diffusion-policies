#!/usr/bin/env python3
"""Train a high-level PPO controller over frozen walk and crouch experts."""

import argparse
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from scripts.gaussian_chunk_rl.config import GaussianPPOConfig
from scripts.hierarchical_experts.policy import Experts, Hierarchy


def parse_args():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk_checkpoint", required=True)
    parser.add_argument("--crouch_checkpoint", required=True)
    parser.add_argument("--reference_checkpoint", required=True,
                        help="WC-v3 checkpoint used to check the task contract, not to initialize the actor.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--rollout_chunks", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor_lr", type=float, default=0.0001)
    parser.add_argument("--initial_std", type=float, default=0.6)
    parser.add_argument("--command_range_mode", choices=("configured", "shared"), default="configured")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sanity_steps", type=int, default=0,
                        help="Run a scripted tracker for this many 50 Hz steps instead of PPO.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main(args, output, experts, reference):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from scripts.dppo_diffusion_rl.checkpointing import sha256_file, task_config_from_env_cfg, validate_resume_task_config
    from scripts.dppo_diffusion_rl.critic import ValueCritic
    from scripts.dppo_diffusion_rl.isaaclab_env import CRITIC_FEATURE_DIM
    from scripts.gaussian_chunk_rl.ppo import GaussianPPOUpdater
    from scripts.hierarchical_experts.trainer import HierarchyTrainer

    cfg = parse_env_cfg("solo12-dppo-diffusion-rl-v0", device=args.device,
                        num_envs=args.num_envs, use_fabric=True)
    settings = {"seed": args.seed, "episode_length_s": 24.0,
                "goal_representation": "hindsight_geom_profile16", "route_distribution": "supported_hybrid_v3",
                "route_stage": 2, "height_profile_stage": None, "route_speed_max_mps": 1.0,
                "speed_budget_max_mps": 1.5, "pace_consistent_preview": False,
                "transition_boundary_min_m": 0.8, "transition_boundary_max_m": 3.2,
                "path_reward_weight": 2.0, "route_cte_rmse_tolerance_m": 0.10,
                "profile_height_reward_weight": 0.75, "profile_height_mae_tolerance_m": 0.04}
    for key, value in settings.items():
        setattr(cfg, key, value)
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=args.seed, curriculum=False, size=(20., 20.), border_width=0., num_rows=1, num_cols=1,
        sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.)})
    task = task_config_from_env_cfg(cfg)
    validate_resume_task_config(reference, task)
    if (cfg.remove_root_lin_vel_b_from_obs or cfg.policy_model != "simple_mlp"
            or cfg.enable_observation_corruption or cfg.observation_noise_model):
        raise ValueError("Expert adapter requires uncorrupted simple_mlp observations with base linear velocity.")
    ppo = GaussianPPOConfig(initial_std=args.initial_std, min_std=0.05, max_std=1.0,
                            actor_lr=args.actor_lr, critic_warmup_iterations=0)
    ppo.validate(prediction_horizon=4, execution_offset=0)
    hashes = {"walk": sha256_file(args.walk_checkpoint), "crouch": sha256_file(args.crouch_checkpoint)}
    env = gym.make("solo12-dppo-diffusion-rl-v0", cfg=cfg)
    try:
        if abs(float(env.unwrapped.step_dt) - 0.02) > 1e-8:
            raise ValueError("Expected the same 50 Hz task interface as WC-v3.")
        device = env.unwrapped.device
        experts = experts.to(device).eval()
        policy = Hierarchy(ppo, args.command_range_mode).to(device)
        critic = ValueCritic(8 * (30 + 12 + 16) + CRITIC_FEATURE_DIM).to(device)
        updater = GaussianPPOUpdater(policy, critic, ppo)
        output.mkdir(parents=True, exist_ok=False)
        repo = Path(__file__).resolve().parents[2]
        git_args = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
        revision = subprocess.run([*git_args, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        status = subprocess.run([*git_args, "status", "--porcelain"], capture_output=True, text=True, check=True)
        metadata = {"command": vars(args), "ppo": ppo.to_dict(), "task": task,
                    "experts_sha256": hashes, "reference_sha256": sha256_file(args.reference_checkpoint),
                    "reference_role": "task contract only", "torch_version": torch.__version__,
                    "git_commit": revision.stdout.strip(), "git_dirty": bool(status.stdout.strip()),
                    "control": {"expert_step_hz": 50, "high_level_hz": 12.5,
                                "velocity_limits": policy.command_limits.tolist(), "skill_order": ["walk", "crouch"]}}
        (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        trainer = HierarchyTrainer(env=env, policy=policy, critic=critic, updater=updater,
                                   source_checkpoint={}, source_checkpoint_path=args.reference_checkpoint,
                                   output_dir=output, rollout_chunks=args.rollout_chunks,
                                   save_interval=args.save_interval, experts=experts, hashes=hashes)
        if args.smoke or args.sanity_steps:
            env.reset()
            command = torch.zeros(args.num_envs, 3, device=device)
            env.unwrapped._commands.copy_(command)
            expected = env.unwrapped._get_observations()["policy"]
            actual = experts.observation(env.unwrapped, command)
            if expected.shape != actual.shape or not torch.allclose(expected, actual, atol=1e-6):
                raise RuntimeError("Expert adapter disagrees with the native environment observation.")
        if args.sanity_steps:
            # Diagnostic controller, not a trained baseline or benchmark score.
            env.reset()
            done_count = 0
            reward_sum = 0.0
            crouch_count = 0
            speed_sum = 0.0
            height_sum = 0.0
            for _ in range(args.sanity_steps):
                goal = env.unwrapped.get_goal()
                xy = goal[:, :2]
                skill = (goal[:, 12] < 0.23).long()
                crouch_count += int(skill.sum())
                speed_sum += float(env.unwrapped._robot.data.root_lin_vel_b[:, 0].sum())
                height_sum += float(env.unwrapped._base_height().sum())
                speed = goal[:, 15].clamp(0.0, 0.6)
                yaw = torch.atan2(xy[:, 1], xy[:, 0].clamp_min(0.05)).clamp(-1.0, 1.0)
                command = torch.stack((speed * torch.cos(yaw).clamp_min(0),
                                       (0.5 * xy[:, 1]).clamp(-0.5, 0.5), yaw), dim=1)
                action = experts.action(env.unwrapped, skill, command)
                if not torch.isfinite(action).all():
                    raise FloatingPointError("Non-finite expert action in tracker sanity check.")
                _, reward, terminated, truncated, _ = env.step(action)
                done_count += int((terminated | truncated).sum())
                reward_sum += float(reward.sum())
            report = {"steps_per_env": args.sanity_steps, "num_envs": args.num_envs,
                      "done_count": done_count, "reward_sum": reward_sum,
                      "crouch_fraction": crouch_count / (args.num_envs * args.sanity_steps),
                      "mean_forward_speed_mps": speed_sum / (args.num_envs * args.sanity_steps),
                      "mean_height_m": height_sum / (args.num_envs * args.sanity_steps)}
            (output / "sanity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[SANITY_OK] {report}", flush=True)
            return
        trainer.run(args.iterations)
        if args.smoke:
            rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
            if not all(row["Policy/actor_updates"] > 0 and row["Policy/critic_updates"] > 0 for row in rows):
                raise RuntimeError("Smoke failed to execute actor and critic updates.")
            if any(p.requires_grad for p in experts.parameters()):
                raise RuntimeError("Experts must remain frozen.")
            if args.rollout_chunks * ppo.exec_horizon >= env.unwrapped.max_episode_length:
                if sum(row["Episode/completed"] for row in rows) < 1:
                    raise RuntimeError("Full-horizon smoke did not exercise any episode reset.")
            print("[SMOKE_OK] collection, PPO updates and checkpoint save", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    args = parse_args()
    if any(getattr(args, k) < 1 for k in ("num_envs", "iterations", "rollout_chunks", "save_interval")):
        raise ValueError("Environment count, iterations, rollout length and save interval must be positive.")
    if args.sanity_steps < 0 or (args.smoke and args.sanity_steps):
        raise ValueError("Use either --smoke or a positive --sanity_steps, not both.")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to mix runs: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    experts = Experts(args.walk_checkpoint, args.crouch_checkpoint)
    reference = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    try:
        main(args, output, experts, reference)
    finally:
        app.close()
