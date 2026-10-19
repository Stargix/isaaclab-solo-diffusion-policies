"""Load and validate the exact WC-v3 hybrid DPPO inference contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.dppo_diffusion_rl.checkpointing import (
    build_inference_policy,
    checkpoint_goal_representation,
    sha256_file,
)
from scripts.diffusion_policy.train.runtime.checkpoint import load_training_checkpoint
from scripts.diffusion_policy.runtime.sampler import CachedDPPOSampler


WC_V3_SHA256 = "2e75422000cc22ab5023cfbc820eb8d391a69b4fc008a7e0a6bbb2c1e0b54868"


@dataclass(frozen=True)
class RuntimeContract:
    """Validated model/config tuple; physical environment values stay explicit."""

    path: Path
    sha256: str
    device: torch.device
    checkpoint: dict[str, Any]
    policy: torch.nn.Module
    sampler: CachedDPPOSampler
    values: dict[str, Any]


def validate_checkpoint_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Reject checkpoints that would silently change the WC-v3 deployment."""

    if checkpoint.get("algorithm") != "dppo":
        raise ValueError("Runtime requires the hybrid DPPO sampler, not a Phase-A-only actor.")
    if int(checkpoint.get("schema_version", -1)) != 8:
        raise ValueError(f"Expected checkpoint schema 8, got {checkpoint.get('schema_version')!r}.")
    if int(checkpoint.get("dppo_task_contract_version", -1)) != 8:
        raise ValueError(
            "Expected DPPO task contract v8, got "
            f"{checkpoint.get('dppo_task_contract_version')!r}."
        )
    if checkpoint_goal_representation(checkpoint) != "hindsight_geom_profile16":
        raise ValueError("Runtime requires hindsight_geom_profile16 conditioning.")

    config = checkpoint["config"]
    model = config["model"]
    dataset = config["dataset"]
    diffusion = config["diffusion"]
    dppo = checkpoint.get("dppo_config", {})
    exact = {
        "proprio_dim": (int(model.get("proprio_dim", -1)), 30),
        "action_hist_dim": (int(model.get("action_hist_dim", -1)), 12),
        "goal_dim": (int(model.get("goal_dim", -1)), 16),
        "action_dim": (int(model.get("action_dim", -1)), 12),
        "history": (int(dataset.get("history", -1)), 8),
        "prediction_horizon": (int(dataset.get("prediction_horizon", -1)), 16),
        "execution_offset": (int(dataset.get("execution_offset", -1)), 8),
        "goal_horizon_steps": (int(dataset.get("goal_horizon_steps", -1)), 100),
        "inference_steps": (int(dppo.get("inference_steps", -1)), 10),
        "finetune_denoising_steps": (int(dppo.get("finetune_denoising_steps", -1)), 5),
        "exec_horizon": (int(dppo.get("exec_horizon", -1)), 4),
    }
    mismatches = [f"{key}={actual} (expected {expected})" for key, (actual, expected) in exact.items()
                  if actual != expected]
    if mismatches:
        raise ValueError("WC-v3 runtime contract mismatch: " + ", ".join(mismatches))
    if not math_isclose(float(dataset.get("dt", float("nan"))), 0.02):
        raise ValueError(f"Expected policy dt=0.02 s, got {dataset.get('dt')!r}.")
    if diffusion.get("prediction_type") != "epsilon" or diffusion.get("variance_type") != "fixed_small":
        raise ValueError("Unexpected diffusion prediction/variance type for the pinned sampler.")
    task = checkpoint.get("dppo_task_config", {})
    if task.get("goal_representation") != "hindsight_geom_profile16":
        raise ValueError("DPPO task metadata does not declare profile16 conditioning.")
    if bool(task.get("pace_consistent_preview", True)):
        raise ValueError("WC-v3 requires the historical pace_consistent_preview=False behavior.")
    max_speed = float(task.get("speed_budget_max_mps", float("nan")))
    if not math_isclose(max_speed, 1.5):
        raise ValueError(f"Expected remaining-speed cap 1.5 m/s, got {max_speed!r}.")
    return {
        "schema_version": 8,
        "task_contract_version": 8,
        "goal_representation": "hindsight_geom_profile16",
        "history": 8,
        "proprio_dim": 30,
        "action_hist_dim": 12,
        "goal_dim": 16,
        "action_dim": 12,
        "prediction_horizon": 16,
        "execution_offset": 8,
        "goal_horizon_steps": 100,
        "dt_s": 0.02,
        "inference_steps": 10,
        "finetune_denoising_steps": 5,
        "exec_horizon": 4,
        "pace_consistent_preview": False,
        "speed_budget_max_mps": max_speed,
        "v_req_clip_mps": float(dataset.get("v_req_clip", 2.0)),
        "dppo_config": dict(dppo),
    }


def math_isclose(value: float, expected: float) -> bool:
    return abs(value - expected) <= 1.0e-9


def load_runtime_contract(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda:0",
    *,
    expected_sha256: str | None = WC_V3_SHA256,
) -> RuntimeContract:
    """Load via the repo's DPPO-aware sampler and verify actor identity."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(
            "Checkpoint SHA-256 differs from the pinned WC-v3 actor: "
            f"{digest} != {expected_sha256}. Pass expected_sha256=None only after review."
        )
    target = torch.device(device)
    checkpoint = load_training_checkpoint(
        path,
        target,
        expected_policy_kind="spatial_hindsight_height_profile_ddpm",
        allow_dppo=True,
    )
    values = validate_checkpoint_contract(checkpoint)
    policy = build_inference_policy(checkpoint, target, inference_steps=values["inference_steps"])
    policy.eval()
    sampler = CachedDPPOSampler(policy)
    return RuntimeContract(path, digest, target, checkpoint, policy, sampler, values)
