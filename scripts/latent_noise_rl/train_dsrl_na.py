#!/usr/bin/env python3
"""Train noise-aliased DSRL on a pinned WC or WCP Phase-A prior."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts as _project_scripts
_project_scripts.__path__ = [str(ROOT / "scripts")]
if _project_scripts.__spec__ is not None:
    _project_scripts.__spec__.submodule_search_locations = _project_scripts.__path__

from scripts.dppo_diffusion_rl.checkpointing import model_config_from_checkpoint, task_config_from_env_cfg
from scripts.latent_noise_rl.checkpointing import PHASE_A_SOURCES, load_phase_a_prior, source_semantic_digest
from scripts.latent_noise_rl.decoder import FrozenDDIMDecoder
from scripts.latent_noise_rl.dsrl_na import DSRLNA, DSRLNAConfig
from scripts.latent_noise_rl.dsrl_na_trainer import DSRLNATrainer

# WC reproduces the archived WC-v3 DPPO task, including its original pointwise
# height tracking and transition margin. WCP keeps its nominal-window contract.
TASK_OVERRIDES = {
    "wc": {"posture_tracking_mode": "instantaneous", "transition_margin_m": 0.25,
           "vertical_velocity_reward_weight": 0.025},
    "wcp": {"posture_tracking_mode": "nominal_window", "transition_margin_m": 0.0,
            "vertical_velocity_reward_weight": 0.0},
}


def make_parser():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior", choices=tuple(PHASE_A_SOURCES), default="wc",
                        help="Pinned Phase-A source and matching physical task contract.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1000)
    # Match the official DSRL train_freq=2 transition cadence: collect two
    # vectorized macro-actions, then perform 20 SAC updates.
    parser.add_argument("--rollout_chunks", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gradient_steps", type=int, default=20)
    parser.add_argument("--noise_critic_grad_steps", type=int, default=10)
    parser.add_argument("--replay_capacity", type=int, default=100_000)
    parser.add_argument("--initial_rollout_chunks", type=int, default=2001)
    parser.add_argument("--initial_alpha", type=float, default=1.0,
                        help="SAC initial entropy temperature; 1 matches the reference implementation.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="solo12-dsrl-na")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _check_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to mix or overwrite run output: {path}")


def _run(args, source, prior, source_sha, agent, output):
    from scripts.latent_noise_rl.train import _make_environment, _module_provenance

    env, recipe = _make_environment(args, task_overrides=TASK_OVERRIDES[args.prior])
    logger = None
    try:
        if abs(float(env.unwrapped.step_dt) - 0.02) > 1.0e-8:
            raise ValueError("Pinned DSRL task requires 50-Hz control.")
        output.mkdir(parents=True, exist_ok=True)
        task_config = task_config_from_env_cfg(env.unwrapped.cfg)
        metadata = {
            "algorithm": "dsrl_na", "command": vars(args), "task": task_config,
            "source_prior_family": args.prior,
            "task_recipe": recipe, "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "source_checkpoint_sha256": source_sha,
            "frozen_prior_semantic_sha256": source_semantic_digest(prior),
            "dsrl_na_config": agent.cfg.to_dict(),
            "macro_transitions_per_iteration": args.num_envs * args.rollout_chunks,
            "control_transitions_per_iteration": args.num_envs * args.rollout_chunks * agent.cfg.exec_horizon,
            "optimization_macro_transitions": args.num_envs * args.rollout_chunks * args.iterations,
            "optimization_control_transitions": args.num_envs * args.rollout_chunks * agent.cfg.exec_horizon * args.iterations,
            "initial_rollout_macro_transitions": args.initial_rollout_chunks * args.num_envs,
            "initial_rollout_control_transitions": args.initial_rollout_chunks * args.num_envs * agent.cfg.exec_horizon,
            "initial_rollout_seconds_per_env": args.initial_rollout_chunks * agent.cfg.exec_horizon * float(env.unwrapped.step_dt),
            "total_macro_transitions": args.num_envs * (args.rollout_chunks * args.iterations + args.initial_rollout_chunks),
            "total_control_transitions": args.num_envs * agent.cfg.exec_horizon * (args.rollout_chunks * args.iterations + args.initial_rollout_chunks),
            "sac_gradient_steps_per_iteration": args.gradient_steps,
            "noise_critic_gradient_steps_per_iteration": args.noise_critic_grad_steps,
            "effective_transition_batch_for_updates": args.num_envs * args.rollout_chunks,
            "replay_samples_per_new_transition": args.gradient_steps * args.batch_size / (args.num_envs * args.rollout_chunks),
            "discount_unit": "action_chunk",
            "entropy_coordinates": "normalized_tanh_noise",
            "evaluation_sampling": "stochastic",
            "reward_discount_contract": "undiscounted_finite_route_potentials",
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True, check=True).stdout.strip(),
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                              capture_output=True, text=True, check=True).stdout.strip()),
            "resolved_module_paths": _module_provenance(),
            "prior_is_frozen": True,
            "algorithm_reference": "Wagenmaker et al., CoRL 2025, Algorithm 1 (DSRL-NA)",
        }
        (output / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        if args.wandb:
            import wandb
            logger = wandb.init(project=args.wandb_project, name=args.run_name,
                                config=metadata, dir=str(output))
        trainer = DSRLNATrainer(
            env=env, agent=agent, source=source, source_path=args.checkpoint,
            source_sha=source_sha, output_dir=output,
            rollout_chunks=args.rollout_chunks, save_interval=args.save_interval,
            logger=logger,
        )
        actor_before = {key: value.detach().clone() for key, value in agent.actor.state_dict().items()}
        prior_before = source_semantic_digest(agent.decoder.prior)
        trainer.run(args.iterations)
        if args.smoke:
            actor_changed = any(
                not torch.equal(actor_before[key], value.detach())
                for key, value in agent.actor.state_dict().items()
            )
            if not actor_changed:
                raise RuntimeError("DSRL-NA smoke completed without updating the noise actor.")
            if source_semantic_digest(agent.decoder.prior) != prior_before:
                raise RuntimeError("DSRL-NA smoke changed the frozen diffusion prior.")
            required = ("initial.pt", "last.pt", "metrics.jsonl", "run_config.json")
            if not all((output / name).is_file() for name in required):
                raise RuntimeError("DSRL-NA smoke is missing a checkpoint or run artifact.")
            from scripts.latent_noise_rl.dsrl_na import build_dsrl_na_inference_policy, DSRLNAInferencePolicy
            saved = torch.load(output / "last.pt", map_location=agent.device, weights_only=False)
            restored = build_dsrl_na_inference_policy(saved, agent.device)
            current = DSRLNAInferencePolicy(
                agent.decoder, agent.stats, history=agent.history, proprio_dim=agent.proprio_dim,
                action_hist_dim=agent.action_hist_dim, goal_dim=agent.goal_dim,
                action_dim=agent.action_dim, prediction_horizon=agent.prediction_horizon,
                execution_offset=agent.execution_offset, cfg=agent.cfg, device=agent.device,
            )
            current.actor.load_state_dict(agent.actor.state_dict())
            inputs = (trainer.proprio_history, trainer.action_history, trainer.goal_history)
            generator_a = torch.Generator(device=agent.device).manual_seed(901)
            generator_b = torch.Generator(device=agent.device).manual_seed(901)
            torch.testing.assert_close(
                current.predict_action_denormalized(*inputs, generator=generator_a),
                restored.predict_action_denormalized(*inputs, generator=generator_b), rtol=0, atol=0,
            )
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            if any(row["DSRL/q_noise_updates"] != agent.cfg.noise_critic_grad_steps or
                   row["DSRL/actor_updates"] != agent.cfg.gradient_steps for row in rows):
                raise RuntimeError("DSRL-NA smoke used an incorrect optimizer schedule.")
            print("[SMOKE_OK] DSRL-NA actor and critics updated; prior frozen; checkpoints saved.", flush=True)
    finally:
        env.close()
        if logger is not None:
            logger.finish()


def main():
    args = make_parser().parse_args()
    args.run_name = args.run_name or f"{args.prior}_dsrl_na_seed{args.seed}"
    output = Path(args.output_dir).resolve()
    if args.preflight_only and args.smoke:
        raise ValueError("Choose either --preflight-only or --smoke.")
    if args.smoke:
        args.num_envs, args.iterations, args.rollout_chunks = 8, 2, 2
        args.batch_size, args.gradient_steps, args.noise_critic_grad_steps = 8, 2, 1
        args.replay_capacity, args.initial_rollout_chunks, args.save_interval = 256, 4, 1
        if "smoke" not in output.name.lower():
            raise ValueError("Smoke output directory must include 'smoke'.")
    if args.num_envs < 1 or args.iterations < 1 or args.rollout_chunks < 1 or args.initial_rollout_chunks < 1:
        raise ValueError("Training counts must be positive.")
    _check_output(output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    source, prior, source_sha = load_phase_a_prior(
        args.checkpoint, "cpu", expected_source_sha256=PHASE_A_SOURCES[args.prior]
    )
    model_cfg = model_config_from_checkpoint(source["config"])
    dsrl_cfg = DSRLNAConfig(
        hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers,
        train_freq=args.rollout_chunks,
        batch_size=args.batch_size, gradient_steps=args.gradient_steps,
        noise_critic_grad_steps=args.noise_critic_grad_steps,
        replay_capacity=args.replay_capacity,
        initial_rollout_chunks=args.initial_rollout_chunks,
        initial_alpha=args.initial_alpha,
    )
    # Training must preserve the same endpoint objective as DPPO. Old inference
    # checkpoints with gamma=0.99 remain readable for auditing, not resumable.
    if dsrl_cfg.gamma != 1.0:
        raise ValueError("The Solo12 finite-route reward requires gamma=1.0.")
    agent = DSRLNA(
        FrozenDDIMDecoder(prior, inference_steps=10), prior.normalizer_stats,
        history=model_cfg.history, proprio_dim=model_cfg.proprio_dim,
        action_hist_dim=model_cfg.action_hist_dim, goal_dim=model_cfg.goal_dim,
        action_dim=model_cfg.action_dim, prediction_horizon=model_cfg.prediction_horizon,
        execution_offset=model_cfg.execution_offset, cfg=dsrl_cfg, device="cpu",
    )
    if args.preflight_only:
        print(json.dumps({
            "status": "PREFLIGHT_OK", "algorithm": "DSRL-NA",
            "source_prior_family": args.prior, "task_overrides": TASK_OVERRIDES[args.prior],
            "source_sha256": source_sha, "source_policy_kind": source["policy_kind"],
            "source_schema": source["schema_version"], "output_dir": str(output),
            "dsrl_na_config": dsrl_cfg.to_dict(),
            "noise_action_dim": agent.noise_dim, "denoised_chunk_dim": agent.executed_dim,
            "critic_count": 4, "noise_aliasing_distillation": True,
            "initial_rollout_noise": "standard_normal_frozen_prior",
        }, indent=2))
        return
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    try:
        _run(args, source, prior, source_sha, agent, output)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
