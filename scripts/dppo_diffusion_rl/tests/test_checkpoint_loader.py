from __future__ import annotations

import pytest
import torch

from scripts.dppo_diffusion_rl.checkpointing import load_policy_checkpoint
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
