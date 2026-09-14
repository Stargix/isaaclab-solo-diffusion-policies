from __future__ import annotations

import pytest
import torch

from scripts.residual_diffusion_rl.routes import RouteBank
from scripts.dppo_diffusion_rl.procedural_routes import SupportedProceduralRouteBank


def test_height_curriculum_can_be_decoupled_from_geometry() -> None:
    torch.manual_seed(7)
    bank = RouteBank(64, "cpu")
    env_ids = torch.arange(64)

    bank.reset(
        env_ids,
        stage=1,
        height_stage=2,
        stratified=True,
        speed_max=0.5,
    )

    assert int(bank.route_kind.max()) <= 2
    assert float(bank.speed.max()) <= 0.5
    assert torch.any(torch.isclose(bank.height, torch.tensor(0.25)))
    assert torch.any(torch.isclose(bank.height, torch.tensor(0.21)))


def test_invalid_independent_height_stage_is_rejected() -> None:
    bank = RouteBank(1, "cpu")
    with pytest.raises(ValueError, match="height_stage"):
        bank.reset(torch.tensor([0]), stage=1, height_stage=3)


def test_supported_procedural_routes_are_balanced_and_use_only_expert_heights() -> None:
    torch.manual_seed(11)
    bank = SupportedProceduralRouteBank(64, "cpu")
    bank.reset(torch.arange(64), stratified=True, speed_max=0.8)

    assert torch.bincount(bank.profile_class, minlength=4).tolist() == [16, 16, 16, 16]
    unique_heights = torch.unique(bank.height)
    torch.testing.assert_close(
        unique_heights,
        torch.tensor([bank.CROUCH_HEIGHT_M, bank.WALK_HEIGHT_M]),
    )
    transition = bank.profile_class >= 2
    assert torch.all(bank.transition_boundary[transition] >= 1.6)
    assert torch.all(bank.transition_boundary[transition] <= 2.4)
    assert torch.isinf(bank.transition_boundary[~transition]).all()


def test_supported_procedural_geometry_is_continuous_and_not_a_shape_catalogue() -> None:
    torch.manual_seed(13)
    bank = SupportedProceduralRouteBank(128, "cpu")
    bank.reset(torch.arange(128), speed_max=0.8)

    segment = torch.linalg.vector_norm(bank.xy[:, 1:] - bank.xy[:, :-1], dim=2)
    torch.testing.assert_close(segment, torch.full_like(segment, 0.04), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(bank.arc[:, -1], torch.full((128,), 4.0))
    assert float(bank.maximum_abs_curvature.max()) <= 0.8 + 1.0e-6
    # Continuous sampling should not collapse the batch to a few templates.
    assert torch.unique(bank.xy[:, -1], dim=0).shape[0] > 120


def test_supported_speed_sampling_respects_crouch_and_geometry_envelopes() -> None:
    torch.manual_seed(17)
    bank = SupportedProceduralRouteBank(400, "cpu")
    bank.reset(torch.arange(400), stratified=True, speed_max=0.8)

    constant_crouch = bank.profile_class == 1
    assert float(bank.speed[constant_crouch].max()) <= 0.4 + 1.0e-6
    assert float(bank.speed.max()) <= 0.8 + 1.0e-6
    assert float(bank.speed.min()) >= 0.2 - 1.0e-6


def test_supported_transition_margin_excludes_only_boundary_from_plateau_score() -> None:
    torch.manual_seed(19)
    bank = SupportedProceduralRouteBank(4, "cpu")
    bank.reset(torch.arange(4), stratified=True, speed_max=0.8)
    transition_env = 2
    boundary_idx = int(
        torch.argmin((bank.arc[transition_env] - bank.transition_boundary[transition_env]).abs())
    )
    bank.progress_idx[transition_env] = boundary_idx - 1
    bank.progress[transition_env] = bank.arc[transition_env, boundary_idx - 1]
    position = bank.xy[:, 0].clone()
    position[transition_env] = bank.xy[transition_env, boundary_idx]
    bank.update(position)

    assert float(bank.height_tracking_weight[transition_env]) == 0.0
    assert torch.all(bank.height_tracking_weight[:2] == 1.0)
