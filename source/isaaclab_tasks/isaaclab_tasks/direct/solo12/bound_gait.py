# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small, simulator-independent helpers for the Solo12 bound contact clock."""

from __future__ import annotations

import math

import torch


BOUND_PHASE_OFFSETS = (0.0, 0.0, 0.5, 0.5)
"""FL, FR, RL, RR phase offsets for a symmetric bound."""


def advance_phase(phase: torch.Tensor, frequency_hz: float, step_dt: float) -> torch.Tensor:
    """Advance a normalized gait phase while keeping it in ``[0, 1)``."""

    return torch.remainder(phase + float(frequency_hz) * float(step_dt), 1.0)


def phase_observation(phase: torch.Tensor) -> torch.Tensor:
    """Return an unambiguous two-dimensional periodic clock."""

    angle = 2.0 * math.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def bound_leg_phases(phase: torch.Tensor) -> torch.Tensor:
    """Return FL, FR, RL, RR phases for a front/rear alternating bound."""

    offsets = phase.new_tensor(BOUND_PHASE_OFFSETS)
    return torch.remainder(phase.unsqueeze(-1) + offsets, 1.0)


def soft_stance_target(
    leg_phase: torch.Tensor,
    duty_factor: float,
    transition_width: float,
) -> torch.Tensor:
    """Build a differentiable periodic stance target in ``[0, 1]``.

    Stance is centred at phase zero. A duty factor below 0.5 leaves two flight
    windows per cycle between the front and rear stance phases.
    """

    if not 0.0 < duty_factor < 1.0:
        raise ValueError(f"duty_factor must be in (0, 1), got {duty_factor}.")
    if transition_width <= 0.0:
        raise ValueError(f"transition_width must be positive, got {transition_width}.")

    circular_distance = torch.abs(torch.remainder(leg_phase + 0.5, 1.0) - 0.5)
    return torch.sigmoid((0.5 * duty_factor - circular_distance) / transition_width)


def bound_stance_targets(
    phase: torch.Tensor,
    duty_factor: float,
    transition_width: float,
) -> torch.Tensor:
    """Return smooth desired contact states in FL, FR, RL, RR order."""

    return soft_stance_target(bound_leg_phases(phase), duty_factor, transition_width)
