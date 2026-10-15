#!/usr/bin/env python3
"""Train the single frozen-prior latent-noise PPO pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Isaac Lab's editable install can expose the main checkout's ``scripts`` as a
# second namespace-package location. Pin that namespace before importing any
# project module so the smoke/full run cannot mix clean-branch code with the
# dirty root worktree.
import scripts as _project_scripts

_project_scripts.__path__ = [str(ROOT / "scripts")]
if _project_scripts.__spec__ is not None:
    _project_scripts.__spec__.submodule_search_locations = _project_scripts.__path__

from scripts.latent_noise_rl.checkpointing import (
    load_phase_a_prior, source_semantic_digest,
)
from scripts.latent_noise_rl.config import LatentPolicyConfig, pilot_ppo_config
from scripts.latent_noise_rl.decoder import FrozenDDIMDecoder
from scripts.latent_noise_rl.policy import LatentNoisePolicy


def make_parser():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Pinned WCP Phase-A checkpoint.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="wcp_ddim_ppo_seed42")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--rollout_chunks", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only the fixed 8-env / 2-iteration integration smoke.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="solo12-latent-noise-ppo")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _check_output(path: Path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to mix or overwrite run output: {path}")


def _module_provenance():
    result = {}
    for name in ("scripts.dppo_diffusion_rl", "scripts.diffusion_policy"):
        spec = importlib.util.find_spec(name)
        origin = None if spec is None else (spec.origin or list(spec.submodule_search_locations or []))
        resolved = [Path(item).resolve() for item in
                    ([origin] if isinstance(origin, str) else (origin or []))]
        result[name] = [str(path) for path in resolved]
        if spec is None or not resolved or any(
            path != ROOT and ROOT not in path.parents for path in resolved
        ):
            raise RuntimeError(f"Resolved {name} outside this worktree: {origin!r}")
    return result


def _make_environment(args):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("solo12-dppo-diffusion-rl-v0", device=args.device,
                            num_envs=args.num_envs, use_fabric=True)
    settings = {
        "seed": args.seed,
        "episode_length_s": 24.0,
        "goal_representation": "hindsight_geom_profile16",
        "route_distribution": "supported_hybrid_v3",
        "route_stage": 2,
        "height_profile_stage": None,
        "route_speed_max_mps": 1.0,
        "speed_budget_max_mps": 1.5,
        "pace_consistent_preview": False,
        "transition_boundary_min_m": 0.8,
        "transition_boundary_max_m": 3.2,
        "transition_margin_m": 0.0,
        "path_reward_weight": 2.0,
        "route_cte_rmse_tolerance_m": 0.10,
        "profile_height_reward_weight": 0.75,
        "profile_height_mae_tolerance_m": 0.04,
        "posture_tracking_mode": "nominal_window",
        "posture_window_s": 0.4,
        "vertical_velocity_reward_weight": 0.0,
    }
    for name, value in settings.items():
        setattr(env_cfg, name, value)
    env_cfg.validate_task()
    env_cfg.terrain.terrain_type = "generator"
    env_cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=args.seed, curriculum=False, size=(20.0, 20.0), border_width=0.0,
        num_rows=1, num_cols=1,
        sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
    )
    return gym.make("solo12-dppo-diffusion-rl-v0", cfg=env_cfg), settings


def _run(args, source, policy, source_sha, output):
    from isaaclab.terrains import TerrainGeneratorCfg  # noqa: F401
    from scripts.dppo_diffusion_rl.checkpointing import task_config_from_env_cfg
    from scripts.dppo_diffusion_rl.critic import ValueCritic
    from scripts.latent_noise_rl.ppo import LatentNoisePPOUpdater
    from scripts.latent_noise_rl.trainer import LatentNoisePPOTrainer

    env, settings = _make_environment(args)
    logger = None
    initial_actor = {k: v.detach().clone() for k, v in policy.actor.state_dict().items()}
    initial_std = policy.log_std.detach().clone()
    prior_signature = source_semantic_digest(policy.decoder.prior)
    try:
        if abs(float(env.unwrapped.step_dt) - 0.02) > 1e-8:
            raise ValueError("The frozen task contract requires a 50-Hz control interval.")
        policy.source_sha256 = source_sha
        policy.to(env.unwrapped.device)
        critic_width = policy.cfg.history * (
            policy.cfg.proprio_dim + policy.cfg.action_hist_dim + policy.cfg.goal_dim
        ) + env.unwrapped.get_critic_features().shape[-1]
        critic = ValueCritic(critic_width).to(env.unwrapped.device)
        ppo_cfg = policy.ppo_cfg
        updater = LatentNoisePPOUpdater(policy, critic, ppo_cfg)
        output.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout.strip())
        metadata = {
            "command": vars(args), "algorithm": "latent_noise_ppo",
            "latent_policy": policy.latent_cfg.to_dict(), "latent_ppo": ppo_cfg.to_dict(),
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "source_sha256": source_sha, "frozen_prior_semantic_sha256": prior_signature,
            "task": task_config_from_env_cfg(env.unwrapped.cfg), "task_recipe": settings,
            "git_commit": commit, "git_dirty": git_dirty,
            "resolved_module_paths": _module_provenance(),
            "critic_input_dim": critic_width, "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "device": str(env.unwrapped.device),
            "diffusers_version": __import__("diffusers").__version__,
            "control_transitions_per_iteration": args.num_envs * args.rollout_chunks * ppo_cfg.exec_horizon,
            "total_control_transitions": args.num_envs * args.rollout_chunks * ppo_cfg.exec_horizon * args.iterations,
        }
        (output / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        if args.wandb:
            import wandb
            logger = wandb.init(project=args.wandb_project, name=args.run_name,
                                config=metadata, dir=str(output))
        trainer = LatentNoisePPOTrainer(
            env=env, policy=policy, critic=critic, updater=updater,
            source_checkpoint=source, source_checkpoint_path=args.checkpoint,
            output_dir=output, rollout_chunks=args.rollout_chunks,
            save_interval=args.save_interval, logger=logger,
        )
        trainer.run(args.iterations)
        unchanged = source_semantic_digest(policy.decoder.prior) == prior_signature
        actor_changed = any(not torch.equal(initial_actor[k], v.detach().cpu())
                            for k, v in policy.actor.state_dict().items())
        std_changed = not torch.equal(initial_std, policy.log_std.detach().cpu())
        if not unchanged:
            raise RuntimeError("Frozen prior/normalizers changed during the smoke.")
        if args.smoke:
            rows = [json.loads(line) for line in trainer.metrics_path.read_text(encoding="utf-8").splitlines()]
            if not actor_changed and not std_changed:
                raise RuntimeError("Smoke completed without changing the latent actor.")
            if not any(row.get("Policy/actor_updates", 0) > 0 for row in rows):
                raise RuntimeError("Smoke did not perform a PPO actor update.")
            if not all(row.get("Policy/critic_updates", 0) > 0 for row in rows):
                raise RuntimeError("Smoke did not perform PPO critic updates.")
            required = ("initial.pt", "last.pt", "metrics.jsonl", "run_config.json")
            if not all((output / name).is_file() for name in required):
                raise RuntimeError("Smoke is missing a required checkpoint or metadata file.")
            print("[SMOKE_OK] latent actor and critic updated; prior frozen; checkpoints saved", flush=True)
    finally:
        env.close()
        if logger is not None:
            logger.finish()


def main():
    args = make_parser().parse_args()
    output = Path(args.output_dir).resolve()
    if args.preflight_only and args.smoke:
        raise ValueError("Choose either --preflight-only or --smoke.")
    if args.smoke:
        args.num_envs, args.iterations, args.rollout_chunks = 8, 2, 2
        args.save_interval = 1
        if "smoke" not in output.name.lower():
            raise ValueError("Smoke output directory name must include 'smoke'.")
    else:
        expected = (args.num_envs, args.iterations, args.rollout_chunks, args.seed)
        if expected != (4096, 150, 32, 42):
            raise ValueError("The frozen full pilot is fixed at 4096 envs, 150 iterations, 32 chunks, seed 42.")
    if args.save_interval != (1 if args.smoke else 10):
        raise ValueError("The frozen pilot save interval is 10 (smoke overrides it to 1).")
    _check_output(output)
    # Hidden actor weights must be reproducible too, not only rollout noise.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    source, prior, source_sha = load_phase_a_prior(args.checkpoint, "cpu")
    ppo_cfg = pilot_ppo_config(smoke=args.smoke)
    latent_cfg = LatentPolicyConfig()
    policy = LatentNoisePolicy(FrozenDDIMDecoder(prior, inference_steps=latent_cfg.inference_steps),
                               ppo_cfg, latent_cfg)
    if args.preflight_only:
        print(json.dumps({
            "status": "PREFLIGHT_OK", "source_sha256": source_sha,
            "source_policy_kind": source["policy_kind"], "source_schema": source["schema_version"],
            "output_dir": str(output), "latent_config": latent_cfg.to_dict(),
            "ppo_config": ppo_cfg.to_dict(), "ddim_scheduler": policy.decoder.scheduler_config(),
            "causal_noise_dimensions": latent_cfg.noise_rows * latent_cfg.noise_dim,
            "unused_suffix_rows": policy.cfg.prediction_horizon - latent_cfg.noise_rows,
        }, indent=2, default=str))
        return
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    try:
        _run(args, source, policy, source_sha, output)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
