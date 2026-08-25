"""Actor-side task conditioning for the finite route-time objective."""

from __future__ import annotations

import torch


def remaining_speed_budget(
    *,
    remaining_distance: torch.Tensor,
    route_length: torch.Tensor,
    desired_mean_speed: torch.Tensor,
    elapsed_s: torch.Tensor,
    max_speed: float,
    min_remaining_time_s: float,
) -> torch.Tensor:
    """Required mean speed over the remaining route-time budget.

    At reset this is exactly the requested route-average speed.  Moving ahead
    of schedule lowers it; falling behind raises it.  The value is clipped to
    the checkpoint's declared conditioning support and tapers to zero at the
    endpoint, matching the original geometric-hindsight ``v_avg`` semantics.
    """

    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive")
    if min_remaining_time_s <= 0.0:
        raise ValueError("min_remaining_time_s must be positive")
    if not (
        remaining_distance.shape
        == route_length.shape
        == desired_mean_speed.shape
        == elapsed_s.shape
    ):
        raise ValueError("speed-budget tensors must have identical shapes")

    eps = torch.finfo(remaining_distance.dtype).eps
    target_duration = route_length.clamp_min(0.0) / desired_mean_speed.clamp_min(eps)
    remaining_time = (target_duration - elapsed_s).clamp_min(min_remaining_time_s)
    budget = remaining_distance.clamp_min(0.0) / remaining_time
    return budget.clamp(0.0, float(max_speed))
