"""Versioned, evaluator-compatible checkpoints for DPPO."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicyConfig
from scripts.diffusion_policy.train.data.normalization import NormalizerStats
from scripts.diffusion_policy.train.runtime.checkpoint import load_training_checkpoint

from .config import DPPOConfig
from .critic import ValueCritic
from .policy import DPPODiffusionPolicy
from .ppo import DPPOUpdater


DPPO_CHECKPOINT_VERSION = 1
DPPO_TASK_CONTRACT_VERSION = 2


def training_resume_state(
    checkpoint: dict[str, Any], *, restart_optimization: bool
) -> tuple[int, int, bool]:
    """Return iteration, physics-step count and whether optimizer state is valid.

    A critic and Adam moments estimate a particular reward/termination return.
    They must not be restored across a task-contract change. Actor weights are
    still loaded by :func:`load_policy_checkpoint` and provide the warm start.
    """

    if checkpoint.get("algorithm") != "dppo":
        if restart_optimization:
            raise ValueError(
                "--restart_optimization is only valid for a DPPO checkpoint; "
                "a Phase-A checkpoint already starts with fresh optimization state."
            )
        return 0, 0, False
    saved_contract = int(checkpoint.get("dppo_task_contract_version", 1))
    if saved_contract != DPPO_TASK_CONTRACT_VERSION and not restart_optimization:
        raise ValueError(
            "This DPPO checkpoint was trained with task contract "
            f"v{saved_contract}, but the current environment uses v{DPPO_TASK_CONTRACT_VERSION}. "
            "Start a new output directory with --restart_optimization to keep the actor "
            "while resetting the critic and Adam states."
        )
    if restart_optimization:
        return 0, 0, False
    return (
        int(checkpoint.get("iteration", -1)) + 1,
        int(checkpoint.get("total_physics_steps", 0)),
        True,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_config_from_checkpoint(config: dict[str, Any]) -> Solo12DiffusionPolicyConfig:
    model = config["model"]
    diffusion = config["diffusion"]
    dataset = config["dataset"]
    return Solo12DiffusionPolicyConfig(
        proprio_dim=model.get("proprio_dim", 30),
        action_hist_dim=model.get("action_hist_dim", 12),
        goal_dim=model.get("goal_dim", 11),
        action_dim=model.get("action_dim", 12),
        history=dataset["history"],
        prediction_horizon=dataset["prediction_horizon"],
        execution_offset=dataset["execution_offset"],
        d_model=model["d_model"],
        nhead=model["nhead"],
        num_layers=model["num_layers"],
        p_drop_emb=model.get("p_drop_emb", model.get("dropout", 0.0)),
        p_drop_attn=model.get("p_drop_attn", model.get("dropout", 0.3)),
        separate_goal_conditioning=model.get("separate_goal_conditioning", True),
        num_train_timesteps=diffusion["num_train_timesteps"],
        beta_start=diffusion["beta_start"],
        beta_end=diffusion["beta_end"],
        beta_schedule=diffusion["beta_schedule"],
        prediction_type=diffusion["prediction_type"],
        variance_type=diffusion["variance_type"],
        clip_sample=diffusion["clip_sample"],
        cfg_dropout_prob=diffusion.get("cfg_dropout_prob", 0.0),
        guidance_scale=1.0,
    )


def load_policy_checkpoint(
    path: str | Path,
    device: torch.device,
    dppo_cfg: DPPOConfig,
) -> tuple[dict[str, Any], DPPODiffusionPolicy, int]:
    checkpoint = load_training_checkpoint(
        path,
        device,
        expected_policy_kind="spatial_hindsight_geometry_ddpm",
        allow_dppo=True,
    )
    config = checkpoint["config"]
    if config["dataset"].get("goal_representation") != "hindsight_geom_avg12":
        raise ValueError("DPPO Phase B requires the geometric hindsight goal12 checkpoint.")
    if not config["dataset"].get("include_padded_starts", False):
        raise ValueError(
            "DPPO Phase B requires a checkpoint trained with include_padded_starts=true; "
            "otherwise its reset-history distribution is undefined."
        )
    if int(config["diffusion"]["num_train_timesteps"]) != dppo_cfg.inference_steps:
        raise ValueError(
            "Exact DDPM DPPO currently requires inference_steps == the checkpoint training steps."
        )
    policy = DPPODiffusionPolicy(model_config_from_checkpoint(config), dppo_cfg).to(device)
    stats = NormalizerStats.from_dict(checkpoint["normalizer_stats"])
    start_iteration = 0
    if checkpoint.get("algorithm") == "dppo":
        if checkpoint.get("dppo_checkpoint_version") != DPPO_CHECKPOINT_VERSION:
            raise ValueError("Unsupported DPPO checkpoint version.")
        saved_cfg = DPPOConfig(**checkpoint["dppo_config"])
        resume_contract = (
            "inference_steps",
            "finetune_denoising_steps",
            "exec_horizon",
            "min_denoising_std",
            "gamma",
            "gae_lambda",
            "gamma_denoising",
        )
        mismatch = [
            name
            for name in resume_contract
            if getattr(saved_cfg, name) != getattr(dppo_cfg, name)
        ]
        if mismatch:
            raise ValueError(
                f"Resume configuration changes DPPO's likelihood/objective contract: {mismatch}."
            )
        policy.load_dppo_actor(
            checkpoint["ema_model_state_dict"],
            checkpoint["dppo_base_model_state_dict"],
            stats,
        )
        start_iteration = int(checkpoint.get("iteration", -1)) + 1
    else:
        policy.load_pretrained(checkpoint["ema_model_state_dict"], stats)
    return checkpoint, policy, start_iteration


def build_inference_policy(
    checkpoint: dict[str, Any],
    device: torch.device,
    *,
    inference_steps: int | None = None,
) -> Solo12DiffusionPolicy | DPPODiffusionPolicy:
    """Instantiate the correct sampler without losing DPPO's frozen early steps."""

    policy_cfg = model_config_from_checkpoint(checkpoint["config"])
    stats = NormalizerStats.from_dict(checkpoint["normalizer_stats"])
    if checkpoint.get("algorithm") != "dppo":
        if inference_steps is not None:
            policy_cfg.num_inference_steps = inference_steps
        policy = Solo12DiffusionPolicy(policy_cfg).to(device)
        policy.load_state_dict(checkpoint["ema_model_state_dict"])
        policy.set_normalizer_stats(stats)
        return policy.eval()

    saved_cfg = DPPOConfig(**checkpoint["dppo_config"])
    if inference_steps is not None and inference_steps != saved_cfg.inference_steps:
        raise ValueError(
            "A DPPO checkpoint must be evaluated with its training denoising schedule; "
            f"requested {inference_steps}, expected {saved_cfg.inference_steps}."
        )
    policy = DPPODiffusionPolicy(policy_cfg, saved_cfg).to(device)
    policy.load_dppo_actor(
        checkpoint["ema_model_state_dict"],
        checkpoint["dppo_base_model_state_dict"],
        stats,
    )
    return policy.eval()


def save_checkpoint(
    path: str | Path,
    *,
    source_checkpoint: dict[str, Any],
    source_path: str | Path,
    policy: DPPODiffusionPolicy,
    critic: ValueCritic,
    updater: DPPOUpdater,
    iteration: int,
    total_physics_steps: int,
    metrics: dict[str, float],
    best_score: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": source_checkpoint["schema_version"],
        "policy_kind": source_checkpoint["policy_kind"],
        "config": source_checkpoint["config"],
        "normalizer_stats": source_checkpoint["normalizer_stats"],
        # Compatibility key for existing Phase-A tooling. DPPO-aware playback
        # must still use the hybrid base/actor sampler recorded below.
        "model_state_dict": policy.actor_policy_state_dict(),
        "ema_model_state_dict": policy.actor_policy_state_dict(),
        "algorithm": "dppo",
        "dppo_checkpoint_version": DPPO_CHECKPOINT_VERSION,
        "dppo_task_contract_version": DPPO_TASK_CONTRACT_VERSION,
        "dppo_config": policy.dppo_cfg.to_dict(),
        "dppo_base_model_state_dict": policy.base_model_state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": updater.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": updater.critic_optimizer.state_dict(),
        "iteration": int(iteration),
        "total_physics_steps": int(total_physics_steps),
        "metrics": dict(metrics),
        "best_score": float(best_score),
        "source_checkpoint": str(Path(source_path)),
        "source_checkpoint_sha256": sha256_file(source_path),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_training_state(
    checkpoint: dict[str, Any], critic: ValueCritic, updater: DPPOUpdater
) -> None:
    if checkpoint.get("algorithm") != "dppo":
        return
    critic.load_state_dict(checkpoint["critic_state_dict"])
    updater.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    updater.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    # ``load_state_dict`` restores parameter-group options too. Re-apply the
    # explicit configuration so a resumed run cannot silently use stale LRs.
    for group in updater.actor_optimizer.param_groups:
        group["lr"] = updater.cfg.actor_lr
        group["weight_decay"] = updater.cfg.actor_weight_decay
    for group in updater.critic_optimizer.param_groups:
        group["lr"] = updater.cfg.critic_lr
        group["weight_decay"] = updater.cfg.critic_weight_decay
