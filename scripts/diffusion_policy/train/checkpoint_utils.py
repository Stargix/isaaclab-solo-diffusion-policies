"""Checkpoint helpers for Solo12 Diffusion Policy training and playback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_training_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    """Load a Solo12 diffusion-policy checkpoint saved by ``train.py``.

    PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``, which rejects
    checkpoints that contain numpy arrays inside ``normalizer_stats``. Our
    checkpoints are trusted local training artifacts and include full metadata,
    so we load them with ``weights_only=False``.
    """

    return torch.load(path, map_location=device, weights_only=False)
