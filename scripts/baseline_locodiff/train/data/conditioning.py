"""Versioned goal representations for the LocoDiff experiments."""

from __future__ import annotations

import torch

CONDITION_MODE_COMMAND_SKILL = "command_skill"
CONDITION_MODE_VELOCITY_HEIGHT = "velocity_height"
CONDITION_MODES = (CONDITION_MODE_COMMAND_SKILL, CONDITION_MODE_VELOCITY_HEIGHT)
WALK_HEIGHT = 0.2932
CROUCH_HEIGHT = 0.1705


def validate_condition_mode(condition_mode: str) -> str:
    if condition_mode not in CONDITION_MODES:
        raise ValueError(f"condition_mode must be one of {CONDITION_MODES}, got {condition_mode!r}.")
    return condition_mode


def goal_dim_for(condition_mode: str) -> int:
    validate_condition_mode(condition_mode)
    return 5 if condition_mode == CONDITION_MODE_COMMAND_SKILL else 4


def policy_kind_for(condition_mode: str) -> str:
    validate_condition_mode(condition_mode)
    if condition_mode == CONDITION_MODE_COMMAND_SKILL:
        return "locodiff_sde_command_skill_v1"
    return "locodiff_sde_velocity_height_v1"


def goal_from_velocity_and_height(
    velocity: torch.Tensor, height: torch.Tensor, condition_mode: str
) -> torch.Tensor:
    """Build the exact normalized-later goal representation for deployment."""

    validate_condition_mode(condition_mode)
    if height.ndim != 2 or height.shape[-1] != 1:
        raise ValueError(f"height must have shape [B, 1], got {tuple(height.shape)}.")
    if condition_mode == CONDITION_MODE_VELOCITY_HEIGHT:
        return torch.cat([velocity, height], dim=-1)
    crouch = ((WALK_HEIGHT - height) / (WALK_HEIGHT - CROUCH_HEIGHT)).clamp(0.0, 1.0)
    return torch.cat([velocity, 1.0 - crouch, crouch], dim=-1)
