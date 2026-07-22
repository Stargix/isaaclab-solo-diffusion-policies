"""Reward definition for local residual refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 3.0
    path: float = 1.0
    speed: float = 1.0
    yaw: float = 0.6
    height: float = 1.0
    alive: float = 0.05
    residual: float = 0.08
    residual_rate: float = 0.04
    tilt: float = 0.15
    vertical_velocity: float = 0.03
    success: float = 6.0
    failure: float = 4.0


def residual_reward(
    *,
    progress_delta: torch.Tensor,
    cross_track: torch.Tensor,
    speed_error: torch.Tensor,
    yaw_error: torch.Tensor,
    height_error: torch.Tensor,
    residual: torch.Tensor,
    previous_residual: torch.Tensor,
    projected_gravity_xy: torch.Tensor,
    vertical_velocity: torch.Tensor,
    success: torch.Tensor,
    failed: torch.Tensor,
    dt: float,
    weights: RewardWeights = RewardWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dense task reward plus trust-region-like residual regularization."""

    terms = {
        "progress": weights.progress * progress_delta,
        "path": weights.path * torch.exp(-torch.square(cross_track / 0.20)) * dt,
        "speed": weights.speed * torch.exp(-torch.square(speed_error / 0.25)) * dt,
        "yaw": weights.yaw * torch.exp(-torch.square(yaw_error / 0.45)) * dt,
        "height": weights.height * torch.exp(-torch.square(height_error / 0.035)) * dt,
        "alive": torch.full_like(progress_delta, weights.alive * dt),
        "residual": -weights.residual * residual.square().mean(dim=1) * dt,
        "residual_rate": -weights.residual_rate * (residual - previous_residual).square().mean(dim=1) * dt,
        "tilt": -weights.tilt * projected_gravity_xy.square().sum(dim=1) * dt,
        "vertical_velocity": -weights.vertical_velocity * vertical_velocity.square() * dt,
        "success": weights.success * success.float(),
        "failure": -weights.failure * failed.float(),
    }
    return torch.stack(tuple(terms.values())).sum(dim=0), terms

