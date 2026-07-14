"""Checkpoint helpers for Solo12 Diffusion Policy training and playback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..conditioning.goal_builder import GOAL_SCHEMA_NAME


def load_training_checkpoint(
    path: str | Path,
    device: torch.device | str,
    *,
    expected_policy_kind: str | None = None,
) -> dict[str, Any]:
    """Load a Solo12 diffusion-policy checkpoint saved by ``train.py``.

    PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``, which rejects
    checkpoints that contain numpy arrays inside ``normalizer_stats``. Our
    checkpoints are trusted local training artifacts and include full metadata,
    so we load them with ``weights_only=False``.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    schema_version = checkpoint.get("schema_version", config.get("schema_version"))
    policy_kind = checkpoint.get("policy_kind", config.get("policy_kind"))
    goal_schema = config.get("goal_schema")
    if schema_version != 3:
        raise ValueError(
            f"Checkpoint {path} uses unsupported schema_version={schema_version!r}. "
            "Retrain with the temporal-preview conditioning contract (schema v3)."
        )
    if expected_policy_kind is not None and policy_kind != expected_policy_kind:
        raise ValueError(
            f"Checkpoint {path} has policy_kind={policy_kind!r}; expected {expected_policy_kind!r}."
        )
    if goal_schema != GOAL_SCHEMA_NAME:
        raise ValueError(
            f"Checkpoint {path} has goal_schema={goal_schema!r}; expected {GOAL_SCHEMA_NAME!r}."
        )
    return checkpoint
