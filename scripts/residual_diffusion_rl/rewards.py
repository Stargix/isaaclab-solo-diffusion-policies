"""Reward definition for local residual refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 2.0
    path: float = 1.5
    speed: float = 1.0
    # Schedule error persists over the whole route.  A tenth-scale weight keeps
    # a fully failed 24 s episode in the same return range as the other task
    # terms while retaining the non-saturating gradient against sprinting.
    schedule: float = 0.1
    yaw: float = 0.4
    height: float = 0.8
    terminal_position: float = 1.0
    terminal_stop: float = 0.5
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
    remaining_distance: torch.Tensor,
    terminal_distance: torch.Tensor,
    planar_speed: torch.Tensor,
    terminal_brake_distance: float,
    terminal_position_tolerance: float,
    terminal_stop_speed: float,
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

    Arc-length progress is a bounded potential difference, so moving forward
    always dominates standing still without rewarding episode duration.  The
    schedule term, rather than a progress gate, distinguishes on-time motion
    from sprinting.  Close to the endpoint the cruise-speed objective fades
    out while final-position and stopped-speed costs fade in, matching the
    terminal contract.  There is no alive bonus or independent time penalty.
    """

    if terminal_brake_distance <= 0.0:
        raise ValueError("terminal_brake_distance must be positive")
    terminal_gate = (1.0 - remaining_distance / terminal_brake_distance).clamp(0.0, 1.0)
    # Smoothstep avoids a reward discontinuity at the beginning of the
    # terminal region while leaving the policy free to choose how it brakes.
    terminal_gate = terminal_gate.square() * (3.0 - 2.0 * terminal_gate)
    cruise_gate = 1.0 - terminal_gate
    terms = {
        "progress": weights.progress * progress_delta,
        "path": -weights.path * _normalized_huber(cross_track, scales.path_m) * dt,
        "speed": -weights.speed * cruise_gate * _normalized_huber(speed_error, scales.speed_mps) * dt,
        "schedule": -weights.schedule * _normalized_huber(schedule_error, scales.schedule_m) * dt,
        "yaw": -weights.yaw * _normalized_huber(yaw_error, scales.yaw_rad) * dt,
        "height": -weights.height * _normalized_huber(height_error, scales.height_m) * dt,
        "terminal_position": (
            -weights.terminal_position
            * terminal_gate
            * _normalized_huber(terminal_distance, terminal_position_tolerance)
            * dt
        ),
        "terminal_stop": (
            -weights.terminal_stop
            * terminal_gate
            * _normalized_huber(planar_speed, terminal_stop_speed)
            * dt
        ),
        "residual": -weights.residual * residual.square().mean(dim=1) * dt,
        "residual_rate": -weights.residual_rate * (residual - previous_residual).square().mean(dim=1) * dt,
        "tilt": -weights.tilt * projected_gravity_xy.square().sum(dim=1) * dt,
        "vertical_velocity": -weights.vertical_velocity * vertical_velocity.square() * dt,
        "success": weights.success * success.float(),
        "failure": -weights.failure * failed.float(),
        "fall_extra": -weights.fall_extra * fell.float(),
    }
    return torch.stack(tuple(terms.values())).sum(dim=0), terms
