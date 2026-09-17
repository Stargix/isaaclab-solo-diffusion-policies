from __future__ import annotations

from dataclasses import replace

import torch

from scripts.dppo_diffusion_rl.route_feasibility import (
    PhaseASupportEnvelopeV2,
    maximum_feasible_mean_speed_v2,
)
from scripts.dppo_diffusion_rl.supported_hybrid_v4 import SupportedHybridV4RouteBank


def _straight_route(height_m: float, *, count: int = 1) -> tuple[torch.Tensor, ...]:
    arc = torch.linspace(0.0, 4.0, 101).expand(count, -1)
    yaw = torch.zeros_like(arc)
    height = torch.full_like(arc, height_m)
    return yaw, height, arc


def test_v4_support_matches_phase_a_straight_endpoints() -> None:
    envelope = PhaseASupportEnvelopeV2()
    walk = maximum_feasible_mean_speed_v2(*_straight_route(envelope.walk_height_m))
    crouch = maximum_feasible_mean_speed_v2(*_straight_route(envelope.crouch_height_m))
    torch.testing.assert_close(walk, torch.tensor([envelope.straight_speed_cap_mps]))
    torch.testing.assert_close(crouch, torch.tensor([envelope.crouch_speed_cap_mps]))


def test_v4_transition_reserve_is_conservative() -> None:
    envelope = PhaseASupportEnvelopeV2()
    yaw, height, arc = _straight_route(envelope.walk_height_m)
    height[:, 50:] = envelope.crouch_height_m
    without_reserve = maximum_feasible_mean_speed_v2(
        yaw, height, arc, envelope=replace(envelope, transition_reserve_m=0.0)
    )
    with_reserve = maximum_feasible_mean_speed_v2(
        yaw, height, arc, envelope=envelope
    )
    assert with_reserve.item() < without_reserve.item()


def test_v4_crosses_geometry_profile_and_pace_tier() -> None:
    torch.manual_seed(42)
    count = 4096
    bank = SupportedHybridV4RouteBank(
        count,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(count), stratified=True, speed_max=1.0)

    family = torch.where(
        bank.geometry_class == bank.SMOOTH,
        bank.smooth_subclass,
        bank.geometry_class + 1,
    )
    cells = family * 8 + bank.profile_class * 2 + bank.pace_tier
    assert torch.bincount(cells, minlength=32).tolist() == [128] * 32
    assert torch.bincount(bank.pace_tier, minlength=2).tolist() == [2048, 2048]

    ceiling = bank.maximum_feasible_mean_speed.clamp(max=1.0).clamp_min(0.2)
    assert torch.all(bank.speed >= 0.2)
    assert torch.all(bank.speed <= ceiling + 1.0e-6)
    upper = bank.pace_tier == bank.PACE_UPPER_FEASIBLE
    upper_floor = 0.2 + 0.75 * (ceiling[upper] - 0.2)
    assert torch.all(bank.speed[upper] >= upper_floor - 1.0e-6)


def test_v4_private_pace_tier_is_not_an_actor_target() -> None:
    bank = SupportedHybridV4RouteBank(
        32,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(32), stratified=True, speed_max=1.0)
    state = bank.update(torch.zeros(32, 2))
    assert bank.speed.shape == (32,)
    assert bank.pace_tier.shape == (32,)
    assert "pace_tier" not in vars(state)


def test_v4_can_audit_full_fast_expert_support_without_parent_cap() -> None:
    bank = SupportedHybridV4RouteBank(
        32,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(32), stratified=True, speed_max=1.5)
    assert torch.all(bank.speed <= bank.maximum_feasible_mean_speed + 1.0e-6)
    assert torch.all(bank.speed <= 1.5)
