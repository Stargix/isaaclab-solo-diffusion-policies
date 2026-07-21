"""Research reward: smoothness is learned, not imposed by command interpolation."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class RewardWeights:
    progress: float = 2.0
    cross_track: float = 3.0
    final_pose: float = 4.0
    deadline_miss: float = 6.0
    height_clearance: float = 2.0
    smooth_first: float = 0.08
    smooth_second: float = 0.04
    low_level_risk: float = 0.0
    fall: float = 10.0


def hierarchical_reward(*, progress: torch.Tensor, cross_track: torch.Tensor, final_pose_error: torch.Tensor,
                        terminal_window: torch.Tensor, timed_out: torch.Tensor, goal_reached: torch.Tensor,
                        command_height: torch.Tensor, max_command_height: torch.Tensor,
                        command: torch.Tensor, previous_command: torch.Tensor,
                        previous_previous_command: torch.Tensor, low_level_risk: torch.Tensor,
                        fallen: torch.Tensor, corridor_half_width: float, per_second_dt: float,
                        weights: RewardWeights = RewardWeights()) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute scalar reward and auditable components.

    There is no ramp or action post-processing: the policy must discover gradual changes
    when they improve return. Hard command limits remain a separate safety envelope.
    """
    def sq(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.square(x), dim=-1)

    # A lower clearance value means a lower ceiling.  Penalize only violations;
    # the policy is otherwise free to stay high or crouch for its own reasons.
    clearance = torch.relu(command_height - max_command_height)
    outside_corridor = torch.relu(torch.abs(cross_track) - corridor_half_width)
    first = sq(command - previous_command)
    second = sq(command - 2.0 * previous_command + previous_previous_command)
    final_error = sq(final_pose_error)
    components = {
        "progress": weights.progress * progress,
        "cross_track": -weights.cross_track * torch.square(outside_corridor) * per_second_dt,
        # The task is evaluated near the deadline, not by greedily minimizing
        # final distance throughout the route.
        "final_pose": weights.final_pose * terminal_window.float() * torch.exp(-final_error),
        "deadline_miss": -weights.deadline_miss * timed_out.float() * (~goal_reached).float(),
        "height_clearance": -weights.height_clearance * torch.square(clearance) * per_second_dt,
        "smooth_first": -weights.smooth_first * first,
        "smooth_second": -weights.smooth_second * second,
        "low_level_risk": -weights.low_level_risk * low_level_risk * per_second_dt,
        "fall": -weights.fall * fallen.float(),
    }
    return torch.stack(tuple(components.values()), dim=0).sum(dim=0), components
