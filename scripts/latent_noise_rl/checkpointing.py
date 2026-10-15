"""Strict Phase-A loading and self-contained latent-policy checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicy
from scripts.diffusion_policy.train.data.normalization import NormalizerStats
from scripts.diffusion_policy.train.runtime.checkpoint import load_training_checkpoint
from scripts.dppo_diffusion_rl.checkpointing import model_config_from_checkpoint, sha256_file
from scripts.gaussian_chunk_rl.config import GaussianPPOConfig

from .config import LatentPolicyConfig
from .decoder import FrozenDDIMDecoder
from .policy import LatentNoisePolicy, ALGORITHM, POLICY_BACKEND, CHECKPOINT_VERSION

SOURCE_SHA256 = "40caf9ad17846797980c96ef9a621474ca5bb730239185edf9b32cc9cc3d7886"
SOURCE_POLICY_KIND = "spatial_hindsight_height_profile_ddpm"


def _state_digest(state: dict[str, torch.Tensor], stats: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    for name, value in sorted(stats.items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def load_phase_a_prior(path: str | Path, device="cpu", *, verify_pinned_sha: bool = True):
    path = Path(path).resolve()
    actual_sha = sha256_file(path)
    if verify_pinned_sha and actual_sha != SOURCE_SHA256:
        raise ValueError(f"Unexpected Phase-A prior SHA256 {actual_sha}; expected {SOURCE_SHA256}.")
    source = load_training_checkpoint(path, device, expected_policy_kind=SOURCE_POLICY_KIND)
    if source.get("algorithm") is not None:
        raise ValueError("Latent steering must start directly from the pinned Phase-A imitation checkpoint.")
    config = source.get("config", {})
    dataset, diffusion = config.get("dataset", {}), config.get("diffusion", {})
    model = config.get("model", {})
    required = {
        "schema_version": source.get("schema_version") == 8,
        "goal_representation": dataset.get("goal_representation") == "hindsight_geom_profile16",
        "padded_starts": bool(dataset.get("include_padded_starts")),
        "dt": float(dataset.get("dt", 0.0)) == 0.02,
        "history": int(dataset.get("history", -1)) == 8,
        "horizon": int(dataset.get("prediction_horizon", -1)) == 16,
        "execution_offset": int(dataset.get("execution_offset", -1)) == 8,
        "action_dim": int(model.get("action_dim", -1)) == 12,
        "goal_dim": int(model.get("goal_dim", -1)) == 16,
        "epsilon": diffusion.get("prediction_type") == "epsilon",
        "fixed_small": diffusion.get("variance_type") == "fixed_small",
        "ten_steps": int(diffusion.get("num_train_timesteps", -1)) == 10,
    }
    failed = [key for key, valid in required.items() if not valid]
    if failed:
        raise ValueError(f"Phase-A checkpoint violates the pinned source contract: {failed}.")
    stats = NormalizerStats.from_dict(source["normalizer_stats"])
    prior = Solo12DiffusionPolicy(model_config_from_checkpoint(config)).to(device)
    prior.load_state_dict(source["ema_model_state_dict"], strict=True)
    prior.set_normalizer_stats(stats)
    for name, value in source["ema_model_state_dict"].items():
        if name.endswith((".mask", ".memory_mask")):
            continue  # validated against the exact causal forms by the decoder
        if not torch.isfinite(value).all():
            raise ValueError(f"Source prior contains non-finite model tensor: {name}.")
    for value in stats.to_dict().values():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("Source prior contains non-finite normalizer values.")
    return source, prior, actual_sha


def source_semantic_digest(prior: Solo12DiffusionPolicy) -> str:
    return _state_digest(prior.state_dict(), prior.normalizer_stats.to_dict())


def assert_finite_state(state: dict[str, torch.Tensor]) -> None:
    for name, tensor in state.items():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite checkpoint tensor: {name}.")


def save_latent_checkpoint(path, *, source, source_path, source_sha, source_digest,
                           policy, critic, updater, iteration, total_control_transitions,
                           metrics, best_score, task_config):
    prior = policy.decoder.prior
    current_digest = source_semantic_digest(prior)
    if current_digest != source_digest:
        raise RuntimeError("Frozen prior weights or normalizers changed during latent PPO.")
    prior_state = {k: v.detach().cpu().clone() for k, v in prior.state_dict().items()}
    actor_state = {k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}
    log_std = policy.log_std.detach().cpu().clone()
    # The frozen transformer's action and memory attention masks intentionally
    # contain -inf for blocked attention. FrozenDDIMDecoder validates their exact
    # causal patterns; all learned prior tensors must still be finite.
    assert_finite_state({name: value for name, value in prior_state.items()
                         if not name.endswith((".mask", ".memory_mask"))})
    assert_finite_state(actor_state)
    assert_finite_state({"log_std": log_std})
    copied = {name: source[name] for name in
              ("schema_version", "policy_kind", "config", "normalizer_stats", "split_manifest")
              if name in source}
    copied.update({
        "algorithm": ALGORITHM,
        "policy_backend": POLICY_BACKEND,
        "latent_checkpoint_version": CHECKPOINT_VERSION,
        "latent_ppo_config": policy.ppo_cfg.to_dict(),
        "latent_policy_config": policy.latent_cfg.to_dict(),
        "ddim_scheduler_config": policy.decoder.scheduler_config(),
        "latent_actor_state_dict": actor_state,
        "latent_log_std": log_std,
        "frozen_prior_state_dict": prior_state,
        "frozen_prior_semantic_sha256": current_digest,
        "normalizer_stats": source["normalizer_stats"],
        "dppo_task_contract_version": 9,
        "dppo_task_config": dict(task_config),
        "critic_state_dict": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
        "actor_optimizer_state_dict": updater.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": updater.critic_optimizer.state_dict(),
        "iteration": int(iteration),
        "total_control_transitions": int(total_control_transitions),
        "metrics": dict(metrics),
        "checkpoint_phase": str(metrics.get("checkpoint_phase", "post_update_unscored")),
        "best_score": float(best_score),
        "source_checkpoint_path": str(Path(source_path).resolve()),
        "source_checkpoint_sha256": source_sha,
        "evaluation_sampling": "stochastic_squashed_latent_ddim_eta0",
        "resume_supported": False,
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(copied, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_inference_policy(checkpoint: dict, device="cpu") -> LatentNoisePolicy:
    if checkpoint.get("algorithm") != ALGORITHM or checkpoint.get("policy_backend") != POLICY_BACKEND:
        raise ValueError("Checkpoint is not a latent_noise_ppo / latent_noise_ddim actor.")
    if checkpoint.get("latent_checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported latent steering checkpoint version.")
    if checkpoint.get("resume_supported") is not False:
        raise ValueError("This pilot does not support optimizer resume.")
    if checkpoint.get("source_checkpoint_sha256") != SOURCE_SHA256:
        raise ValueError("Latent actor records an unexpected Phase-A source SHA.")
    source_config = checkpoint.get("config")
    if not isinstance(source_config, dict) or checkpoint.get("policy_kind") != SOURCE_POLICY_KIND:
        raise ValueError("Latent actor is missing its original Phase-A schema/config.")
    prior = Solo12DiffusionPolicy(model_config_from_checkpoint(source_config)).to(device)
    prior.load_state_dict(checkpoint["frozen_prior_state_dict"], strict=True)
    prior.set_normalizer_stats(NormalizerStats.from_dict(checkpoint["normalizer_stats"]))
    ppo_cfg = GaussianPPOConfig(**checkpoint["latent_ppo_config"])
    latent_cfg = LatentPolicyConfig(**{
        **checkpoint["latent_policy_config"],
        "hidden_dims": tuple(checkpoint["latent_policy_config"]["hidden_dims"]),
    })
    policy = LatentNoisePolicy(FrozenDDIMDecoder(prior, inference_steps=latent_cfg.inference_steps),
                               ppo_cfg, latent_cfg).to(device)
    if policy.decoder.scheduler_config() != checkpoint.get("ddim_scheduler_config"):
        raise ValueError("Saved DDIM scheduler configuration differs from the pinned decoder.")
    policy.actor.load_state_dict(checkpoint["latent_actor_state_dict"], strict=True)
    policy.log_std.data.copy_(checkpoint["latent_log_std"].to(device))
    if source_semantic_digest(prior) != checkpoint["frozen_prior_semantic_sha256"]:
        raise ValueError("Frozen prior or normalizers differ from the saved semantic digest.")
    if not all(torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError("Latent inference checkpoint contains non-finite parameters.")
    return policy.eval()
