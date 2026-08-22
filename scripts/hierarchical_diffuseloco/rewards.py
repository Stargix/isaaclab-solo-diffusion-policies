"""Auditable reward for route, posture and mean-speed tracking.

All dense error terms are bounded/normalized and integrated in seconds.  Scheduled
progress is potential-shaped, so catching up is rewarded and falling early cannot
avoid a long stream of arbitrary time penalties.
"""

from __future__ import annotations

import torch

try:
    from isaaclab.utils import configclass
except ImportError:  # Allows reward tests without launching Isaac Sim's Python runtime.
    from dataclasses import dataclass

    configclass = dataclass


@configclass
class RewardWeights:
    schedule_potential: float = 4.0
    cross_track: float = 0.60
    heading: float = 0.25
    height: float = 0.45
    command_delta: float = 0.015
    terminal_pose: float = 4.0
    fall: float = 12.0


def smooth_l1(error: torch.Tensor) -> torch.Tensor:
    """Elementwise Huber loss with unit transition and no hidden reduction."""
    absolute = torch.abs(error)
    return torch.where(absolute < 1.0, 0.5 * error.square(), absolute - 0.5)


def schedule_potential(progress_error_normalized: torch.Tensor) -> torch.Tensor:
    """Potential whose maximum is zero at the desired temporal progress."""
    return -smooth_l1(progress_error_normalized)


def hierarchical_reward(*, previous_schedule_potential: torch.Tensor,
                        current_schedule_potential: torch.Tensor,
                        cross_track_normalized: torch.Tensor,
                        heading_error_normalized: torch.Tensor,
                        height_error_normalized: torch.Tensor,
                        normalized_command_delta: torch.Tensor,
                        terminal_pose_error_normalized: torch.Tensor,
                        terminal: torch.Tensor, fallen: torch.Tensor,
                        discount: float, macro_dt: float,
                        weights: RewardWeights = RewardWeights()) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return reward and components without any action filter or command ramp."""
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1].")
    if macro_dt <= 0.0:
        raise ValueError("macro_dt must be positive.")
    command_delta_cost = normalized_command_delta.square().mean(dim=-1)
    terminal_cost = smooth_l1(terminal_pose_error_normalized).sum(dim=-1)
    components = {
        "schedule_potential": weights.schedule_potential * (
            discount * current_schedule_potential - previous_schedule_potential
        ),
        "cross_track": -weights.cross_track * smooth_l1(cross_track_normalized) * macro_dt,
        "heading": -weights.heading * smooth_l1(heading_error_normalized) * macro_dt,
        "height": -weights.height * smooth_l1(height_error_normalized) * macro_dt,
        # This is a learned preference: the action is still sent directly to DiffuseLoco.
        "command_delta": -weights.command_delta * command_delta_cost,
        "terminal_pose": -weights.terminal_pose * terminal.float() * terminal_cost,
        "fall": -weights.fall * fallen.float(),
    }
    return torch.stack(tuple(components.values()), dim=0).sum(dim=0), components
