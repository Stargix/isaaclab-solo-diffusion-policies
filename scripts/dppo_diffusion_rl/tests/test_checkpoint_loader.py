from __future__ import annotations

import pytest
import torch

from scripts.dppo_diffusion_rl.checkpointing import (
    DPPO_TASK_CONTRACT_VERSION,
    checkpoint_goal_representation,
    load_policy_checkpoint,
    training_resume_state,
    validate_resume_task_config,
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
        "policy_kind": "spatial_hindsight_geometry_ddpm",
        "config": {
            "model": {"goal_dim": 12},
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


@pytest.mark.parametrize(
    ("representation", "policy_kind", "goal_dim"),
    (
        ("hindsight_geom_avg12", "spatial_hindsight_geometry_ddpm", 12),
        ("hindsight_geom_profile16", "spatial_hindsight_height_profile_ddpm", 16),
    ),
)
def test_checkpoint_goal_contract_accepts_both_geometric_schemas(
    representation: str, policy_kind: str, goal_dim: int
) -> None:
    checkpoint = {
        "policy_kind": policy_kind,
        "config": {
            "model": {"goal_dim": goal_dim},
            "dataset": {"goal_representation": representation},
        },
    }
    assert checkpoint_goal_representation(checkpoint) == representation


def test_checkpoint_goal_contract_rejects_dimension_aliasing() -> None:
    checkpoint = {
        "policy_kind": "spatial_hindsight_height_profile_ddpm",
        "config": {
            "model": {"goal_dim": 12},
            "dataset": {"goal_representation": "hindsight_geom_profile16"},
        },
    }
    with pytest.raises(ValueError, match="requires goal_dim=16"):
        checkpoint_goal_representation(checkpoint)


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


def test_optimizer_resume_rejects_changed_height_distribution() -> None:
    saved = {
        "dppo_task_config": {
            "route_stage": 1,
            "height_profile_stage": 1,
            "route_speed_max_mps": 0.5,
        }
    }
    with pytest.raises(ValueError, match="height_profile_stage"):
        validate_resume_task_config(
            saved,
            {
                "route_stage": 1,
                "height_profile_stage": 2,
                "route_speed_max_mps": 0.5,
            },
        )


def test_legacy_height_stage_is_inferred_from_route_stage() -> None:
    config = {"route_stage": 1, "route_speed_max_mps": 0.5}
    validate_resume_task_config({"dppo_task_config": config}, config)
