"""Reward definition for local residual refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 2.0
    path: float = 1.5
    speed: float = 1.0
    schedule: float = 1.0
    yaw: float = 0.4
    height: float = 0.8
    residual: float = 0.15
    residual_rate: float = 0.08
    tilt: float = 0.25
    vertical_velocity: float = 0.05
    success: float = 10.0
    failure: float = 10.0
    fall_extra: float = 5.0


@dataclass(frozen=True)
class RewardScales:
    """Physical error scales at which the normalized Huber loss changes slope."""

    path_m: float = 0.10
    speed_mps: float = 0.15
    schedule_m: float = 0.20
    yaw_rad: float = 0.35
    height_m: float = 0.035
    progress_speed_mps: float = 0.20


def _normalized_huber(error: torch.Tensor, scale: float) -> torch.Tensor:
    """Quadratic near zero and linear, rather than saturated, for large errors."""

    normalized = error.abs() / scale
    return torch.where(normalized <= 1.0, 0.5 * normalized.square(), normalized - 0.5)


def progress_timing_errors(
    *,
    progress: torch.Tensor,
    route_length: torch.Tensor,
    commanded_speed: torch.Tensor,
    elapsed_s: torch.Tensor,
    min_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute schedule and mean-speed errors from one average-speed command."""

    expected_progress = torch.minimum(commanded_speed * elapsed_s, route_length)
    schedule_error = progress - expected_progress
    mean_speed = torch.where(
        elapsed_s > 0.0,
        progress / elapsed_s.clamp_min(min_dt),
        commanded_speed,
    )
    return schedule_error, mean_speed - commanded_speed


def residual_reward(
    *,
    progress_delta: torch.Tensor,
    cross_track: torch.Tensor,
    speed_error: torch.Tensor,
    schedule_error: torch.Tensor,
    yaw_error: torch.Tensor,
    height_error: torch.Tensor,
    residual: torch.Tensor,
    previous_residual: torch.Tensor,
    projected_gravity_xy: torch.Tensor,
    vertical_velocity: torch.Tensor,
    success: torch.Tensor,
    failed: torch.Tensor,
    fell: torch.Tensor,
    dt: float,
    weights: RewardWeights = RewardWeights(),
    scales: RewardScales = RewardScales(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average-speed-aware route reward for a competent frozen prior.

    Progress is only valuable while instantaneous tangential speed agrees with
    the command.  Non-saturating Huber costs make large speed or schedule
    violations strictly worse, closing the former sprint-to-goal exploit.
    There is deliberately no alive bonus or independent time penalty.
    """

    progress_gate = torch.exp(-0.5 * torch.square(speed_error / scales.progress_speed_mps))
    terms = {
        "progress": weights.progress * progress_delta * progress_gate,
        "path": -weights.path * _normalized_huber(cross_track, scales.path_m) * dt,
        "speed": -weights.speed * _normalized_huber(speed_error, scales.speed_mps) * dt,
        "schedule": -weights.schedule * _normalized_huber(schedule_error, scales.schedule_m) * dt,
        "yaw": -weights.yaw * _normalized_huber(yaw_error, scales.yaw_rad) * dt,
        "height": -weights.height * _normalized_huber(height_error, scales.height_m) * dt,
        "residual": -weights.residual * residual.square().mean(dim=1) * dt,
        "residual_rate": -weights.residual_rate * (residual - previous_residual).square().mean(dim=1) * dt,
        "tilt": -weights.tilt * projected_gravity_xy.square().sum(dim=1) * dt,
        "vertical_velocity": -weights.vertical_velocity * vertical_velocity.square() * dt,
        "success": weights.success * success.float(),
        "failure": -weights.failure * failed.float(),
        "fall_extra": -weights.fall_extra * fell.float(),
    }
    return torch.stack(tuple(terms.values())).sum(dim=0), terms
