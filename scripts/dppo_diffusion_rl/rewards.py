"""Task reward for direct online fine-tuning of the diffusion policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskRewardWeights:
    progress: float = 3.0
    path: float = 1.25
    mean_speed: float = 0.25
    yaw: float = 0.25
    height: float = 0.75
    tilt: float = 0.10
    vertical_velocity: float = 0.025
    success: float = 10.0
    task_failure: float = 10.0
    fall_extra: float = 10.0


@dataclass(frozen=True)
class TaskRewardScales:
    path_m: float = 0.12
    mean_speed_mps: float = 0.10
    yaw_rad: float = 0.40
    height_m: float = 0.04


def normalized_huber(error: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("Huber scales must be positive.")
    value = error.abs() / scale
    return torch.where(value <= 1.0, 0.5 * value.square(), value - 0.5)


def average_speed_error(
    progress: torch.Tensor,
    elapsed_s: torch.Tensor,
    desired_speed: torch.Tensor,
    *,
    min_elapsed_s: float = 0.5,
) -> torch.Tensor:
    """Route-average speed error, with a neutral startup interval.

    This is the single timing signal.  There is deliberately no extra clock
    reward and no instantaneous velocity target.
    """

    valid = elapsed_s >= min_elapsed_s
    measured = progress / elapsed_s.clamp_min(min_elapsed_s)
    return torch.where(valid, measured - desired_speed, torch.zeros_like(measured))


def path_task_reward(
    *,
    progress_delta: torch.Tensor,
    cross_track: torch.Tensor,
    mean_speed_error: torch.Tensor,
    yaw_error: torch.Tensor,
    height_error: torch.Tensor,
    projected_gravity_xy: torch.Tensor,
    vertical_velocity: torch.Tensor,
    success: torch.Tensor,
    failed: torch.Tensor,
    fell: torch.Tensor,
    dt: float,
    weights: TaskRewardWeights = TaskRewardWeights(),
    scales: TaskRewardScales = TaskRewardScales(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dense potential shaping plus explicit terminal task outcomes.

    ``progress_delta`` telescopes over a route, so reward cannot be increased
    merely by extending episode duration.  All state-error costs are integrated
    in seconds.  Smooth motion is not imposed by a controller or action-rate
    cost; it must remain a property of the learned diffusion manifold.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    terms = {
        "progress": weights.progress * progress_delta,
        "path": -weights.path * normalized_huber(cross_track, scales.path_m) * dt,
        "mean_speed": -weights.mean_speed
        * normalized_huber(mean_speed_error, scales.mean_speed_mps)
        * dt,
        "yaw": -weights.yaw * normalized_huber(yaw_error, scales.yaw_rad) * dt,
        "height": -weights.height * normalized_huber(height_error, scales.height_m) * dt,
        "tilt": -weights.tilt * projected_gravity_xy.square().sum(dim=-1) * dt,
        "vertical_velocity": -weights.vertical_velocity * vertical_velocity.square() * dt,
        "success": weights.success * success.float(),
        "task_failure": -weights.task_failure * failed.float(),
        "fall_extra": -weights.fall_extra * fell.float(),
    }
    return torch.stack(tuple(terms.values())).sum(dim=0), terms
