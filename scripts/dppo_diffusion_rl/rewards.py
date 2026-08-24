"""Task reward for direct online fine-tuning of the diffusion policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskRewardWeights:
    progress: float = 3.0
    path: float = 1.25
    mean_speed: float = 1.0
    yaw: float = 0.25
    height: float = 0.75
    tilt: float = 0.10
    vertical_velocity: float = 0.025
    terminal_position: float = 2.0
    terminal_yaw: float = 0.5
    terminal_height: float = 1.0
    success: float = 10.0
    arrival_failure: float = 10.0
    timeout_failure: float = 10.0
    hard_failure_margin: float = 2.0
    fall_extra: float = 10.0


@dataclass(frozen=True)
class TaskRewardScales:
    path_m: float = 0.12
    yaw_rad: float = 0.40
    height_m: float = 0.04
    vertical_velocity_mps: float = 0.50
    terminal_position_m: float = 0.15
    terminal_height_m: float = 0.05


def normalized_huber(error: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("Huber scales must be positive.")
    value = error.abs() / scale
    return torch.where(value <= 1.0, 0.5 * value.square(), value - 0.5)


def bounded_huber(error: torch.Tensor, scale: float) -> torch.Tensor:
    """Smooth error cost in ``[0, 1]`` with a Huber-shaped origin."""

    return -torch.expm1(-normalized_huber(error, scale))


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


def schedule_error_improvement(
    *,
    progress: torch.Tensor,
    progress_delta: torch.Tensor,
    elapsed_s: torch.Tensor,
    desired_speed: torch.Tensor,
    route_length: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """One-step change in the negative finite-route schedule error potential.

    Summed over an episode this equals ``-|s_T-v*T|`` because the
    initial progress and schedule are both zero. It therefore supplies a dense
    average-speed signal without an occupancy cost or a separate deadline
    reward.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    current_progress = torch.minimum(progress.clamp_min(0.0), route_length)
    previous_progress = torch.minimum(
        (progress - progress_delta).clamp_min(0.0), route_length
    )
    # Do not cap the schedule at the endpoint: |L-v*t| must remain non-zero
    # when the route is completed late. The environment terminates on the first
    # entry into the goal region, so an early arrival cannot recover this error
    # by waiting at the endpoint.
    current_schedule = desired_speed * elapsed_s.clamp_min(0.0)
    previous_schedule = desired_speed * (elapsed_s - dt).clamp_min(0.0)
    previous_error = (previous_progress - previous_schedule).abs()
    current_error = (current_progress - current_schedule).abs()
    return previous_error - current_error


def terminal_pose_potential(
    *,
    remaining_distance: torch.Tensor,
    actor_lookahead_distance: torch.Tensor,
    terminal_distance: torch.Tensor,
    terminal_yaw_error: torch.Tensor,
    terminal_height_error: torch.Tensor,
    weights: TaskRewardWeights = TaskRewardWeights(),
    scales: TaskRewardScales = TaskRewardScales(),
) -> torch.Tensor:
    """Bounded near-terminal potential for final position and pose.

    The gate activates only after the global endpoint enters the actor's local
    geometric lookahead. The potential is non-positive and zero at the exact
    target, so its one-step difference guides observable final corrections
    without creating an alive/occupancy reward.
    """

    lookahead = actor_lookahead_distance.clamp_min(1.0e-6)
    gate = (1.0 - remaining_distance / lookahead).clamp(0.0, 1.0)
    cost = (
        weights.terminal_position
        * bounded_huber(terminal_distance, scales.terminal_position_m)
        + weights.terminal_yaw * bounded_huber(terminal_yaw_error, scales.yaw_rad)
        + weights.terminal_height
        * bounded_huber(terminal_height_error, scales.terminal_height_m)
    )
    return -gate * cost


def path_task_reward(
    *,
    progress_delta: torch.Tensor,
    schedule_improvement: torch.Tensor,
    terminal_pose_improvement: torch.Tensor,
    route_length: torch.Tensor,
    maximum_schedule_distance: torch.Tensor,
    cross_track: torch.Tensor,
    yaw_error: torch.Tensor,
    height_error: torch.Tensor,
    projected_gravity_xy: torch.Tensor,
    vertical_velocity: torch.Tensor,
    success: torch.Tensor,
    arrival_failure: torch.Tensor,
    timeout_failure: torch.Tensor,
    corridor_failure: torch.Tensor,
    overshoot_failure: torch.Tensor,
    fell: torch.Tensor,
    weights: TaskRewardWeights = TaskRewardWeights(),
    scales: TaskRewardScales = TaskRewardScales(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distance-bounded shaping plus explicit terminal task outcomes.

    Tracking costs are paid per metre of monotone route progress, not per
    second. The pace term is a potential difference. Consequently, standing
    longer cannot accumulate an unbounded negative return and early failure is
    never a shortcut around occupancy costs. Smooth motion is not imposed by a
    controller or action-rate cost; it remains a property of the diffusion
    manifold.
    """

    distance = progress_delta.clamp_min(0.0)
    hard_failure = (corridor_failure | overshoot_failure) & ~fell
    # A hard failure after traversing the full route is still strictly worse
    # than standing until timeout. The bound follows from the only positive
    # dense term (progress) and the finite pace potential.
    hard_failure_cost = (
        weights.timeout_failure
        + weights.hard_failure_margin
        + weights.progress * route_length
        + weights.mean_speed * maximum_schedule_distance
    )
    terms = {
        "progress": weights.progress * distance,
        "path": -weights.path * bounded_huber(cross_track, scales.path_m) * distance,
        "mean_speed": weights.mean_speed * schedule_improvement,
        "terminal_pose": terminal_pose_improvement,
        "yaw": -weights.yaw * bounded_huber(yaw_error, scales.yaw_rad) * distance,
        "height": -weights.height * bounded_huber(height_error, scales.height_m) * distance,
        "tilt": -weights.tilt
        * projected_gravity_xy.square().sum(dim=-1).clamp(0.0, 1.0)
        * distance,
        "vertical_velocity": -weights.vertical_velocity
        * bounded_huber(vertical_velocity, scales.vertical_velocity_mps)
        * distance,
        "success": weights.success * success.float(),
        "arrival_failure": -weights.arrival_failure * arrival_failure.float(),
        "timeout_failure": -weights.timeout_failure * timeout_failure.float(),
        "hard_failure": -hard_failure_cost * hard_failure.float(),
        "fall": -(hard_failure_cost + weights.fall_extra) * fell.float(),
    }
    return torch.stack(tuple(terms.values())).sum(dim=0), terms
