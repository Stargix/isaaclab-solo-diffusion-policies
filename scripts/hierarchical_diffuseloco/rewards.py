"""Research reward: smoothness is learned, not imposed by command interpolation."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 2.0
    cross_track: float = 1.5
    final_pose: float = 1.0
    deadline: float = 0.5
    height_clearance: float = 2.0
    smooth_first: float = 0.08
    smooth_second: float = 0.04
    low_level_risk: float = 0.25
    fall: float = 10.0


def hierarchical_reward(*, progress: torch.Tensor, cross_track: torch.Tensor, final_pose_error: torch.Tensor,
                        time_error: torch.Tensor, actual_height: torch.Tensor, required_height: torch.Tensor,
                        command: torch.Tensor, previous_command: torch.Tensor,
                        previous_previous_command: torch.Tensor, low_level_risk: torch.Tensor,
                        fallen: torch.Tensor, weights: RewardWeights = RewardWeights()) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute scalar reward and auditable components.

    There is no ramp or action post-processing: the policy must discover gradual changes
    when they improve return. Hard command limits remain a separate safety envelope.
    """
    def sq(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.square(x), dim=-1)

    clearance = torch.relu(required_height - actual_height)
    first = sq(command - previous_command)
    second = sq(command - 2.0 * previous_command + previous_previous_command)
    components = {
        "progress": weights.progress * progress,
        "cross_track": -weights.cross_track * torch.square(cross_track),
        "final_pose": -weights.final_pose * sq(final_pose_error),
        "deadline": -weights.deadline * torch.square(time_error),
        "height_clearance": -weights.height_clearance * torch.square(clearance),
        "smooth_first": -weights.smooth_first * first,
        "smooth_second": -weights.smooth_second * second,
        "low_level_risk": -weights.low_level_risk * low_level_risk,
        "fall": -weights.fall * fallen.float(),
    }
    return torch.stack(tuple(components.values()), dim=0).sum(dim=0), components
