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
    obs: ZScoreStats
    goal: ZScoreStats
    action: MinMaxStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_mean": self.obs.mean,
            "obs_std": self.obs.std,
            "goal_mean": self.goal.mean,
            "goal_std": self.goal.std,
            "action_min": self.action.min,
            "action_max": self.action.max,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "NormalizerStats":
        return cls(
            obs=ZScoreStats(np.asarray(values["obs_mean"]), np.asarray(values["obs_std"])),
            goal=ZScoreStats(np.asarray(values["goal_mean"]), np.asarray(values["goal_std"])),
            action=MinMaxStats(np.asarray(values["action_min"]), np.asarray(values["action_max"])),
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


def build_stats(obs_values: np.ndarray, goal_values: np.ndarray, action_values: np.ndarray) -> NormalizerStats:
    """Create normalizer stats from flattened observations, goals and actions."""

    action_min, action_max = _safe_range(action_values.min(axis=0), action_values.max(axis=0))
    return NormalizerStats(
        obs=ZScoreStats(obs_values.mean(axis=0).astype(np.float32), _safe_std(obs_values.std(axis=0))),
        goal=ZScoreStats(goal_values.mean(axis=0).astype(np.float32), _safe_std(goal_values.std(axis=0))),
        action=MinMaxStats(action_min, action_max),
    )

