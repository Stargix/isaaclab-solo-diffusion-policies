from __future__ import annotations

import torch
import pytest

from scripts.dppo_diffusion_rl.checkpointing import validate_resume_task_config
from scripts.dppo_diffusion_rl.rewards import distance_weighted_rmse
from scripts.dppo_diffusion_rl.route_feasibility import (
    PhaseASupportEnvelope,
    maximum_feasible_mean_speed,
)
from scripts.dppo_diffusion_rl.supported_hybrid_v3 import SupportedHybridV3RouteBank


def _straight_route(height_m: float, *, count: int = 1) -> tuple[torch.Tensor, ...]:
    arc = torch.linspace(0.0, 4.0, 101).expand(count, -1)
    yaw = torch.zeros_like(arc)
    height = torch.full_like(arc, height_m)
    return yaw, height, arc


def test_feasibility_matches_endpoint_support_on_straight_routes() -> None:
    envelope = PhaseASupportEnvelope()
    walk = maximum_feasible_mean_speed(*_straight_route(envelope.walk_height_m))
    crouch = maximum_feasible_mean_speed(*_straight_route(envelope.crouch_height_m))
    torch.testing.assert_close(walk, torch.tensor([envelope.straight_speed_cap_mps]))
    torch.testing.assert_close(crouch, torch.tensor([envelope.crouch_speed_cap_mps]))


def test_feasibility_reduces_deadline_for_turning_and_crouch() -> None:
    envelope = PhaseASupportEnvelope()
    yaw, height, arc = _straight_route(envelope.walk_height_m, count=3)
    yaw[1, 50:] = torch.pi / 2.0
    height[2, 50:] = envelope.crouch_height_m
    speed = maximum_feasible_mean_speed(yaw, height, arc, envelope=envelope)
    assert speed[1] < speed[0]
    assert speed[2] < speed[0]
    assert speed[1] >= envelope.curved_speed_cap_mps
    assert speed[2] >= envelope.crouch_speed_cap_mps


def test_distance_weighted_cte_rmse_is_time_independent() -> None:
    errors = torch.tensor([0.0, 0.1, 0.2])
    distance = torch.tensor([1.0, 2.0, 1.0])
    expected = torch.sqrt(torch.sum(errors.square() * distance) / distance.sum())
    actual = distance_weighted_rmse(
        torch.sum(errors.square() * distance)[None], distance.sum()[None]
    )
    torch.testing.assert_close(actual, expected[None])
    torch.testing.assert_close(
        distance_weighted_rmse(torch.zeros(1), torch.zeros(1)), torch.zeros(1)
    )


def test_v3_stratifies_geometry_profiles_and_transition_contexts() -> None:
    torch.manual_seed(42)
    count = 4096
    bank = SupportedHybridV3RouteBank(
        count,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(count), stratified=True, speed_max=1.0)

    assert torch.bincount(bank.geometry_class, minlength=3).tolist() == [2048, 1024, 1024]
    assert torch.bincount(bank.profile_class, minlength=4).tolist() == [1024] * 4
    transition_context = bank.transition_context[bank.transition_context >= 0]
    assert torch.bincount(transition_context, minlength=3).tolist() == [1024, 512, 512]

    transition = bank.profile_class >= 2
    assert torch.all(bank.transition_boundary[transition] >= 0.8)
    assert torch.all(bank.transition_boundary[transition] <= 3.2 + 1.0e-6)
    assert torch.all(torch.isinf(bank.transition_boundary[~transition]))
    assert torch.all(bank.speed >= 0.2)
    assert torch.all(bank.speed <= 1.0)
    assert torch.all(bank.speed <= bank.maximum_feasible_mean_speed + 1.0e-6)


def test_private_feasibility_is_not_part_of_route_goal_state() -> None:
    bank = SupportedHybridV3RouteBank(
        16,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(16), stratified=True, speed_max=1.0)
    # RouteBank's public task state remains the global mean speed. The private
    # local envelope is retained only as an auditable reset-time diagnostic.
    assert bank.speed.shape == (16,)
    assert bank.maximum_feasible_mean_speed.shape == (16,)
    assert not hasattr(bank, "local_target_speed")


def test_old_task_config_keeps_historical_path_defaults() -> None:
    checkpoint = {
        "dppo_task_config": {
            "route_stage": 2,
            "height_profile_stage": None,
        }
    }
    validate_resume_task_config(
        checkpoint,
        {
            "route_stage": 2,
            "height_profile_stage": None,
            "path_reward_weight": 1.25,
            "route_cte_rmse_tolerance_m": None,
        },
    )


def test_resume_rejects_changed_path_contract() -> None:
    checkpoint = {
        "dppo_task_config": {
            "route_stage": 2,
            "height_profile_stage": None,
            "path_reward_weight": 1.25,
            "route_cte_rmse_tolerance_m": None,
        }
    }
    with pytest.raises(ValueError, match="path_reward_weight"):
        validate_resume_task_config(
            checkpoint,
            {
                "route_stage": 2,
                "height_profile_stage": None,
                "path_reward_weight": 2.0,
                "route_cte_rmse_tolerance_m": 0.10,
            },
        )
