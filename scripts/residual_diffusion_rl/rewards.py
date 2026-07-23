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
    # A route-completion task must not pay the agent merely for keeping an
    # episode alive.  This small time cost breaks ties between equally accurate
    # trajectories without overriding the nominal-speed objective.
    time: float = 0.05
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
    """Progress-first route reward with error costs and bounded residuals.

    The exponential tracking terms are centred at zero.  Thus perfect tracking
    has no per-second reward, whereas tracking error is a smooth negative cost.
    This prevents longer episodes from receiving a larger return simply for
    remaining alive.
    """

    terms = {
        "progress": weights.progress * progress_delta,
        "path": weights.path * (torch.exp(-torch.square(cross_track / 0.20)) - 1.0) * dt,
        "speed": weights.speed * (torch.exp(-torch.square(speed_error / 0.25)) - 1.0) * dt,
        "yaw": weights.yaw * (torch.exp(-torch.square(yaw_error / 0.45)) - 1.0) * dt,
        "height": weights.height * (torch.exp(-torch.square(height_error / 0.035)) - 1.0) * dt,
        "time": torch.full_like(progress_delta, -weights.time * dt),
        "residual": -weights.residual * residual.square().mean(dim=1) * dt,
        "residual_rate": -weights.residual_rate * (residual - previous_residual).square().mean(dim=1) * dt,
        "tilt": -weights.tilt * projected_gravity_xy.square().sum(dim=1) * dt,
        "vertical_velocity": -weights.vertical_velocity * vertical_velocity.square() * dt,
        "success": weights.success * success.float(),
        "failure": -weights.failure * failed.float(),
    }
    return torch.stack(tuple(terms.values())).sum(dim=0), terms
