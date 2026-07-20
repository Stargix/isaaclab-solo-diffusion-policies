"""Checkpoint helpers for Solo12 Diffusion Policy training and playback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from ..conditioning.goal_builder import (
    GOAL_SCHEMA_NAME,
    REFERENCE_GOAL_SCHEMA_NAME,
    HOLONOMIC_REFERENCE_GOAL_SCHEMA_NAME,
    PATH_GUIDANCE_GOAL_SCHEMA_NAME,
    GEOMETRIC_HINDSIGHT_GOAL_SCHEMA_NAME,
)


def load_training_checkpoint(
    path: str | Path,
    device: torch.device | str,
    *,
    expected_policy_kind: str | Iterable[str] | None = None,
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
    if schema_version not in {3, 4, 5, 6, 7}:
        raise ValueError(
            f"Checkpoint {path} uses unsupported schema_version={schema_version!r}. "
            "Retrain with a supported temporal-preview conditioning contract."
        )
    expected_kinds = None
    if expected_policy_kind is not None:
        expected_kinds = {expected_policy_kind} if isinstance(expected_policy_kind, str) else set(expected_policy_kind)
    if expected_kinds is not None and policy_kind not in expected_kinds:
        raise ValueError(
            f"Checkpoint {path} has policy_kind={policy_kind!r}; expected one of {sorted(expected_kinds)!r}."
        )
    expected_goal_schema = (
        GEOMETRIC_HINDSIGHT_GOAL_SCHEMA_NAME if schema_version == 7
        else PATH_GUIDANCE_GOAL_SCHEMA_NAME if schema_version == 6
        else HOLONOMIC_REFERENCE_GOAL_SCHEMA_NAME if schema_version == 5
        else REFERENCE_GOAL_SCHEMA_NAME if schema_version == 4
        else GOAL_SCHEMA_NAME
    )
    if goal_schema != expected_goal_schema:
        raise ValueError(
            f"Checkpoint {path} has goal_schema={goal_schema!r}; expected {expected_goal_schema!r}."
        )
    return checkpoint
