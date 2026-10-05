# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure tensor utilities for the SOLO12 ipsilateral pace reward."""

from __future__ import annotations

import torch


SYNCED_FEET = ((0, 2), (1, 3))
ASYNC_FEET = ((0, 1), (0, 3), (2, 1), (2, 3))


def pace_timing_score(
    current_air_time: torch.Tensor,
    current_contact_time: torch.Tensor,
    *,
    std_s2: float = 0.10,
    max_error_s: float = 0.20,
) -> torch.Tensor:
    """Clock-free synchronization score for FL-RL and FR-RR.

    `std_s2` is the denominator of the exponent and already has units s^2;
    it must not be squared again. Inputs have shape ``(..., 4)`` in FL,FR,RL,RR
    order and contain current contact-sensor timers in seconds.
    """
    if std_s2 <= 0.0 or max_error_s <= 0.0:
        raise ValueError(f"std_s2 and max_error_s must be positive, got {std_s2}, {max_error_s}.")
    if current_air_time.shape != current_contact_time.shape or current_air_time.shape[-1] != 4:
        raise ValueError("Air/contact timers must have identical (..., 4) shapes in FL,FR,RL,RR order.")

    air = current_air_time
    contact = current_contact_time
    max_err_sq = max_error_s**2
    error = torch.zeros_like(air[..., 0])
    for first, second in SYNCED_FEET:
        error = error + torch.clamp((air[..., first] - air[..., second]).square(), max=max_err_sq)
        error = error + torch.clamp(
            (contact[..., first] - contact[..., second]).square(), max=max_err_sq
        )
    for first, second in ASYNC_FEET:
        error = error + torch.clamp((air[..., first] - contact[..., second]).square(), max=max_err_sq)
        error = error + torch.clamp((contact[..., first] - air[..., second]).square(), max=max_err_sq)
    return torch.exp(-error / std_s2)


def pace_contact_topology_score(
    net_forces_w: torch.Tensor,
    *,
    threshold_n: float = 1.0,
    softness_n: float = 0.25,
) -> torch.Tensor:
    """Prefer synchronized ipsilateral support with the opposite side out of phase."""
    if threshold_n < 0.0 or softness_n <= 0.0:
        raise ValueError("Contact threshold must be non-negative and softness positive.")
    if net_forces_w.ndim < 2 or net_forces_w.shape[-2:] != (4, 3):
        raise ValueError("Contact forces must have shape (..., 4, 3) in FL,FR,RL,RR order.")
    force_norm = torch.linalg.vector_norm(net_forces_w, dim=-1)
    probability = torch.sigmoid((force_norm - threshold_n) / softness_n)
    ipsi_sync = 1.0 - 0.5 * (
        torch.abs(probability[..., 0] - probability[..., 2])
        + torch.abs(probability[..., 1] - probability[..., 3])
    )
    side_opposition = torch.abs(
        0.5 * (probability[..., 0] + probability[..., 2])
        - 0.5 * (probability[..., 1] + probability[..., 3])
    )
    return torch.clamp(ipsi_sync * side_opposition, min=0.0, max=1.0)


def pace_gait_reward_components(
    current_air_time: torch.Tensor,
    current_contact_time: torch.Tensor,
    net_forces_w: torch.Tensor,
    commands_xy: torch.Tensor,
    tracking_score_xy: torch.Tensor,
    *,
    timing_std_s2: float = 0.10,
    timing_max_error_s: float = 0.20,
    contact_threshold_n: float = 1.0,
    contact_softness_n: float = 0.25,
    max_mode_time_s: float = 0.50,
    min_moving_speed_mps: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Return bounded factors and the gated gait score (without dt or weight)."""
    if max_mode_time_s <= 0.0 or min_moving_speed_mps < 0.0:
        raise ValueError("max_mode_time_s must be positive and moving threshold non-negative.")
    if commands_xy.shape[-1] != 2 or commands_xy.shape[:-1] != tracking_score_xy.shape:
        raise ValueError("commands_xy must have shape (..., 2) matching tracking_score_xy.")
    if current_air_time.shape != current_contact_time.shape:
        raise ValueError("Air and contact timers must have identical shapes.")

    timing = pace_timing_score(
        current_air_time,
        current_contact_time,
        std_s2=timing_std_s2,
        max_error_s=timing_max_error_s,
    )
    topology = pace_contact_topology_score(
        net_forces_w,
        threshold_n=contact_threshold_n,
        softness_n=contact_softness_n,
    )
    mode_duration = torch.maximum(current_air_time, current_contact_time).amax(dim=-1)
    mode_ok = (mode_duration <= max_mode_time_s).to(dtype=timing.dtype)
    moving = (torch.linalg.vector_norm(commands_xy, dim=-1) > min_moving_speed_mps).to(dtype=timing.dtype)
    tracking_score_xy = torch.clamp(tracking_score_xy, min=0.0, max=1.0)
    gait_score = timing * topology * mode_ok * moving * tracking_score_xy
    return {
        "timing": timing,
        "topology": topology,
        "mode_ok": mode_ok,
        "moving": moving,
        "gait": gait_score,
    }
