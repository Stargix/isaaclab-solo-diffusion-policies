#!/usr/bin/env python3
"""Train the versioned conditional-Gaussian hierarchy to a physical-step budget."""

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts.hierarchical_experts.policy import Experts
from scripts.hierarchical_experts.policy_v2 import HierarchyV2
from scripts.hierarchical_experts.hybrid_ppo_v2 import HybridPPOUpdaterV2
from scripts.hierarchical_experts.trainer_v2 import HierarchyTrainerV2


def parse_args():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk_checkpoint", required=True)
    parser.add_argument("--crouch_checkpoint", required=True)
    parser.add_argument("--reference_checkpoint", required=True,
                        help="WC-v3 checkpoint used only to validate the unchanged task contract.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--interaction_budget", type=int, default=80_000_000,
                        help="Total 50 Hz environment transitions summed over parallel environments.")
    parser.add_argument("--curriculum_fraction", type=float, default=0.20)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--rollout_chunks", type=int, default=16,
                        help="High-level decisions per PPO rollout.")
    parser.add_argument("--execution_steps", type=int, default=10,
                        help="Low-level 50 Hz actions held per high-level decision (10 = 5 Hz).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--initial_std", type=float, default=0.6)
    parser.add_argument("--min_std", type=float, default=0.05)
    parser.add_argument("--max_std", type=float, default=1.0)
    parser.add_argument("--command_range_mode", choices=("shared", "configured"), default="shared")
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--minibatch_size", type=int, default=8192)
    parser.add_argument("--critic_minibatch_size", type=int, default=4096)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--discrete_entropy_coef", type=float, default=0.01)
    parser.add_argument("--continuous_entropy_coef", type=float, default=0.0)
    parser.add_argument("--save_interval_updates", type=int, default=10)
    parser.add_argument("--smoke", action="store_true",
                        help="Assert termination, curriculum transition, updates and checkpoints.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _run_settings(args):
    return {"seed": args.seed, "interaction_budget": args.interaction_budget,
            "curriculum_fraction": args.curriculum_fraction, "num_envs": args.num_envs,
            "rollout_chunks": args.rollout_chunks, "execution_steps": args.execution_steps,
            "actor_lr": args.actor_lr, "critic_lr": args.critic_lr,
            "initial_std": args.initial_std, "min_std": args.min_std, "max_std": args.max_std,
            "command_range_mode": args.command_range_mode, "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "critic_minibatch_size": args.critic_minibatch_size,
            "clip_ratio": args.clip_ratio, "target_kl": args.target_kl,
            "discrete_entropy_coef": args.discrete_entropy_coef,
            "continuous_entropy_coef": args.continuous_entropy_coef,
            "save_interval_updates": args.save_interval_updates,
            "gamma": 1.0, "gae_lambda": 0.95, "value_coef": 0.5,
            "max_grad_norm": 1.0}


def main(args):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from scripts.dppo_diffusion_rl.checkpointing import (
        sha256_file, task_config_from_env_cfg, validate_resume_task_config)
    from scripts.dppo_diffusion_rl.critic import ValueCritic
    from scripts.dppo_diffusion_rl.isaaclab_env import CRITIC_FEATURE_DIM

    if args.interaction_budget < 1 or args.num_envs < 1 or args.rollout_chunks < 1:
        raise ValueError("Budget, environment count and rollout length must be positive.")
    if args.execution_steps != 10:
        raise ValueError("The initial V2 experiment is fixed to 10 low-level steps (5 Hz).")
    repo = Path(__file__).resolve().parents[2]
    git_args = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    revision = subprocess.run([*git_args, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run([*git_args, "status", "--porcelain"], capture_output=True, text=True, check=True)
    if status.stdout.strip() and not args.smoke:
        raise RuntimeError("A scientific V2 run requires a clean committed worktree; use --smoke only for wiring checks.")
    output = Path(args.output_dir).resolve()
    resume_path = Path(args.resume_from).resolve() if args.resume_from else None
    if resume_path is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir() or resume_path.parent != output:
        raise ValueError("Resume requires an existing output directory and checkpoint inside it.")

    cfg = parse_env_cfg("solo12-dppo-diffusion-rl-v0", device=args.device,
                        num_envs=args.num_envs, use_fabric=True)
    settings = {"seed": args.seed, "episode_length_s": 24.0,
                "goal_representation": "hindsight_geom_profile16", "route_distribution": "supported_hybrid_v3",
                "route_stage": 2, "height_profile_stage": None, "route_speed_max_mps": 1.0,
                "speed_budget_max_mps": 1.5, "pace_consistent_preview": False,
                "transition_boundary_min_m": 0.8, "transition_boundary_max_m": 3.2,
                "path_reward_weight": 2.0, "route_cte_rmse_tolerance_m": 0.10,
                "profile_height_reward_weight": 0.75, "profile_height_mae_tolerance_m": 0.04,
                "hierarchy_curriculum_enabled": True, "hierarchy_curriculum_stage": 0}
    for key, value in settings.items():
        setattr(cfg, key, value)
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=args.seed, curriculum=False, size=(20., 20.), border_width=0., num_rows=1, num_cols=1,
        sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.)})
    task = task_config_from_env_cfg(cfg)
    reference = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)
    validate_resume_task_config(reference, task)
    if (cfg.remove_root_lin_vel_b_from_obs or cfg.policy_model != "simple_mlp"
            or cfg.enable_observation_corruption or cfg.observation_noise_model):
        raise ValueError("Experts require uncorrupted simple_mlp observations with base linear velocity.")
    hashes = {"walk": sha256_file(args.walk_checkpoint), "crouch": sha256_file(args.crouch_checkpoint)}
    run_config = _run_settings(args)
    run_config["experts_sha256"] = hashes
    run_config["reference_sha256"] = sha256_file(args.reference_checkpoint)
    run_config["task_contract"] = task
    run_config["git_commit"] = revision
    run_config["git_dirty"] = bool(status.stdout.strip())
    if resume_path is None:
        run_config_path = output / "run_config.json"
        run_config_path.write_text(json.dumps({"algorithm": "hierarchical_expert_ppo_v2",
            "architecture": HierarchyV2.architecture, "run": run_config,
            "task_contract": task, "curriculum": {"initial_fraction": args.curriculum_fraction,
                "initial_stage": "straight; balanced constant walk/crouch and both transitions; means 0.25/0.50 m/s",
                "final_stage": "supported_hybrid_v3 original sampler",
                "training_distribution_is_not_the_final_evaluation_distribution": True},
            "expert_control": {"low_level_hz": 50,
                "high_level_hz": 50.0 / args.execution_steps,
                "skill_order": ["walk", "crouch"], "command_limits": [[1., .5, 1.], [1., .5, 1.]]
                    if args.command_range_mode == "shared" else [[2., 1., 1.], [1., .5, 1.]],
            "frozen": True}, "git_commit": revision,
            "git_dirty": bool(status.stdout.strip())}, indent=2), encoding="utf-8")
    else:
        saved_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        if saved_config.get("run") != run_config or saved_config.get("task_contract") != task:
            raise ValueError("Resume run config or task contract differs from the existing run.")

    env = gym.make("solo12-dppo-diffusion-rl-v0", cfg=cfg)
    try:
        raw = env.unwrapped
        if abs(float(raw.step_dt) - 0.02) > 1e-8:
            raise ValueError("Expected 50 Hz low-level control.")
        device = raw.device
        experts = Experts(args.walk_checkpoint, args.crouch_checkpoint).to(device).eval()
        policy = HierarchyV2(execution_steps=args.execution_steps, initial_std=args.initial_std,
            min_std=args.min_std, max_std=args.max_std,
            command_range_mode=args.command_range_mode).to(device)
        actor_batch = args.num_envs * args.rollout_chunks
        if args.minibatch_size > actor_batch or args.critic_minibatch_size > actor_batch:
            raise ValueError("Configured minibatches cannot exceed samples in one rollout.")
        if actor_batch % args.minibatch_size or actor_batch % args.critic_minibatch_size:
            raise ValueError("Samples per rollout must be divisible by actor and critic minibatch sizes.")
        critic = ValueCritic(8 * 58 + CRITIC_FEATURE_DIM).to(device)
        updater = HybridPPOUpdaterV2(policy, critic, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
            critic_minibatch_size=args.critic_minibatch_size, clip_ratio=args.clip_ratio,
            target_kl=args.target_kl, discrete_entropy_coef=args.discrete_entropy_coef,
            continuous_entropy_coef=args.continuous_entropy_coef)
        trainer = HierarchyTrainerV2(env=env, policy=policy, critic=critic, updater=updater,
            source_checkpoint=reference, source_checkpoint_path=args.reference_checkpoint,
            output_dir=output, rollout_chunks=args.rollout_chunks,
            save_interval=args.save_interval_updates, experts=experts, hashes=hashes,
            interaction_budget=args.interaction_budget, curriculum_fraction=args.curriculum_fraction,
            config=run_config)
        if resume_path:
            trainer.load_checkpoint(resume_path, run_config)
            if trainer.total_env_transitions >= args.interaction_budget:
                raise ValueError("The saved run has already reached the interaction budget.")
        if args.smoke:
            env.reset()
            zero_command = torch.zeros(args.num_envs, 3, device=device)
            raw._commands.copy_(zero_command)
            expected = raw._get_observations()["policy"]
            actual = experts.observation(raw, zero_command)
            if expected.shape != actual.shape or not torch.allclose(expected, actual, atol=1e-6):
                raise RuntimeError("Frozen-expert adapter differs from the native observation.")
        trainer.run_budget()
        if args.smoke:
            rows = [json.loads(line) for line in trainer.metrics_path.read_text(encoding="utf-8").splitlines()]
            if not rows or not any(row["Episode/completed"] > 0 for row in rows):
                raise RuntimeError("Smoke did not exercise an episode termination.")
            if not any(row["Curriculum/straight_stage_fraction"] > 0 for row in rows) \
                    or not any(row["Curriculum/final_stage_fraction"] > 0 for row in rows):
                raise RuntimeError("Smoke did not exercise both curriculum stages.")
            if not all(row["Policy/actor_updates"] > 0 and row["Policy/critic_updates"] > 0 for row in rows):
                raise RuntimeError("Smoke did not update both actor and critic.")
            state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            if state["algorithm"] != "hierarchical_expert_ppo_v2" or state["format_version"] != 3:
                raise RuntimeError("Smoke checkpoint has the wrong version.")
            print("[SMOKE_OK] v2 curriculum, terminal handling, PPO updates and versioned checkpoint", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    try:
        main(args)
    finally:
        app.close()
