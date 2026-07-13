"""Normalization utilities for diffusion-policy training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ZScoreStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class MinMaxStats:
    min: np.ndarray
    max: np.ndarray


@dataclass
class NormalizerStats:
    proprio: ZScoreStats
    goal: ZScoreStats
    action: MinMaxStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "proprio_mean": self.proprio.mean,
            "proprio_std": self.proprio.std,
            "goal_mean": self.goal.mean,
            "goal_std": self.goal.std,
            "action_min": self.action.min,
            "action_max": self.action.max,
            # Backward compatibility for older checkpoints.
            "obs_mean": self.proprio.mean,
            "obs_std": self.proprio.std,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "NormalizerStats":
        if "proprio_mean" in values:
            proprio_mean = values["proprio_mean"]
            proprio_std = values["proprio_std"]
        else:
            proprio_mean = values["obs_mean"]
            proprio_std = values["obs_std"]
        return cls(
            proprio=ZScoreStats(np.asarray(proprio_mean, dtype=np.float32), _safe_std(np.asarray(proprio_std))),
            goal=ZScoreStats(
                np.asarray(values["goal_mean"], dtype=np.float32),
                _safe_std(np.asarray(values["goal_std"])),
            ),
            action=MinMaxStats(
                np.asarray(values["action_min"], dtype=np.float32),
                np.asarray(values["action_max"], dtype=np.float32),
            ),
        )


def _safe_std(std: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    return np.maximum(std, eps).astype(np.float32)


def _safe_range(min_values: np.ndarray, max_values: np.ndarray, eps: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    center = (min_values + max_values) * 0.5
    half_range = np.maximum((max_values - min_values) * 0.5, eps)
    return (center - half_range).astype(np.float32), (center + half_range).astype(np.float32)


def normalize_zscore(values: torch.Tensor, stats: ZScoreStats) -> torch.Tensor:
    mean = torch.as_tensor(stats.mean, dtype=values.dtype, device=values.device)
    std = torch.as_tensor(stats.std, dtype=values.dtype, device=values.device)
    return (values - mean) / std


def denormalize_zscore(values: torch.Tensor, stats: ZScoreStats) -> torch.Tensor:
    mean = torch.as_tensor(stats.mean, dtype=values.dtype, device=values.device)
    std = torch.as_tensor(stats.std, dtype=values.dtype, device=values.device)
    return values * std + mean


def normalize_minmax(values: torch.Tensor, stats: MinMaxStats) -> torch.Tensor:
    min_values = torch.as_tensor(stats.min, dtype=values.dtype, device=values.device)
    max_values = torch.as_tensor(stats.max, dtype=values.dtype, device=values.device)
    return 2.0 * (values - min_values) / (max_values - min_values) - 1.0


def denormalize_minmax(values: torch.Tensor, stats: MinMaxStats) -> torch.Tensor:
    min_values = torch.as_tensor(stats.min, dtype=values.dtype, device=values.device)
    max_values = torch.as_tensor(stats.max, dtype=values.dtype, device=values.device)
    return 0.5 * (values + 1.0) * (max_values - min_values) + min_values


def build_stats(
    proprio_values: np.ndarray,
    goal_values: np.ndarray,
    action_values: np.ndarray,
) -> NormalizerStats:
    # DDPM inference clips its normalized sample to [-1, 1].  Exact train-set
    # extrema therefore define the only consistent invertible mapping.  The old
    # percentile mapping produced training targets outside [-1, 1] that the
    # sampler could never reproduce.
    action_min, action_max = _safe_range(action_values.min(axis=0), action_values.max(axis=0))
    return NormalizerStats(
        proprio=ZScoreStats(proprio_values.mean(axis=0).astype(np.float32), _safe_std(proprio_values.std(axis=0))),
        goal=ZScoreStats(goal_values.mean(axis=0).astype(np.float32), _safe_std(goal_values.std(axis=0))),
        action=MinMaxStats(action_min, action_max),
    )
