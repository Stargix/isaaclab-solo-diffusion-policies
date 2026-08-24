from __future__ import annotations

import pytest
import torch

from scripts.dppo_diffusion_rl.checkpointing import (
    DPPO_TASK_CONTRACT_VERSION,
    load_policy_checkpoint,
    training_resume_state,
)
from scripts.dppo_diffusion_rl.config import DPPOConfig
from scripts.diffusion_policy.train.conditioning.goal_builder import (
    GEOMETRIC_HINDSIGHT_GOAL_SCHEMA_NAME,
)
from scripts.diffusion_policy.train.runtime.checkpoint import load_training_checkpoint


def test_generic_loader_rejects_hybrid_dppo_checkpoint(tmp_path) -> None:
    path = tmp_path / "hybrid.pt"
    torch.save(
        {
            "schema_version": 7,
            "policy_kind": "spatial_hindsight_geometry_ddpm",
            "algorithm": "dppo",
            "config": {
                "schema_version": 7,
                "policy_kind": "spatial_hindsight_geometry_ddpm",
                "goal_schema": GEOMETRIC_HINDSIGHT_GOAL_SCHEMA_NAME,
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="hybrid DPPO"):
        load_training_checkpoint(path, "cpu")
    loaded = load_training_checkpoint(path, "cpu", allow_dppo=True)
    assert loaded["algorithm"] == "dppo"


def test_dppo_loader_rejects_checkpoint_without_padded_starts(monkeypatch) -> None:
    checkpoint = {
        "config": {
            "dataset": {
                "goal_representation": "hindsight_geom_avg12",
                "include_padded_starts": False,
            }
        }
    }
    monkeypatch.setattr(
        "scripts.dppo_diffusion_rl.checkpointing.load_training_checkpoint",
        lambda *args, **kwargs: checkpoint,
    )
    with pytest.raises(ValueError, match="include_padded_starts=true"):
        load_policy_checkpoint("unused.pt", torch.device("cpu"), DPPOConfig())


def test_legacy_task_contract_requires_clean_optimization_restart() -> None:
    legacy = {
        "algorithm": "dppo",
        "iteration": 69,
        "total_physics_steps": 1234,
    }
    with pytest.raises(ValueError, match="--restart_optimization"):
        training_resume_state(legacy, restart_optimization=False)
    assert training_resume_state(legacy, restart_optimization=True) == (0, 0, False)


def test_current_task_contract_resumes_optimizer_and_counters() -> None:
    current = {
        "algorithm": "dppo",
        "dppo_task_contract_version": DPPO_TASK_CONTRACT_VERSION,
        "iteration": 12,
        "total_physics_steps": 3456,
    }
    assert training_resume_state(current, restart_optimization=False) == (13, 3456, True)
