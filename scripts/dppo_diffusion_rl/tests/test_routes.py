from __future__ import annotations

import pytest
import torch

from scripts.residual_diffusion_rl.routes import RouteBank
from scripts.dppo_diffusion_rl.procedural_routes import SupportedProceduralRouteBank
from scripts.dppo_diffusion_rl.hybrid_routes import (
    SupportedHybridRouteBank,
    _has_proper_self_intersection,
    sample_coherent_smooth_geometry,
    sample_waypoint_geometry,
)


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


def test_hybrid_geometry_has_exact_stratified_mixture_and_preserves_profiles() -> None:
    torch.manual_seed(23)
    bank = SupportedHybridRouteBank(400, "cpu")
    bank.reset(torch.arange(400), stratified=True, speed_max=0.8)

    assert torch.bincount(bank.geometry_class, minlength=3).tolist() == [200, 100, 100]
    assert [int((bank.smooth_subclass == value).sum()) for value in (0, 1)] == [100, 100]
    assert torch.bincount(bank.profile_class, minlength=4).tolist() == [100] * 4
    joint = torch.stack(
        [
            torch.bincount(bank.geometry_class[bank.profile_class == profile], minlength=3)
            for profile in range(4)
        ]
    )
    assert joint.tolist() == [[50, 25, 25]] * 4
    torch.testing.assert_close(
        torch.unique(bank.height),
        torch.tensor([bank.CROUCH_HEIGHT_M, bank.WALK_HEIGHT_M]),
    )


def test_coherent_smooth_geometry_supplies_sustained_bounded_curvature() -> None:
    torch.manual_seed(27)
    geometry = sample_coherent_smooth_geometry(512, "cpu")

    delta = geometry.xy[:, 1:] - geometry.xy[:, :-1]
    segment = torch.linalg.vector_norm(delta, dim=2)
    torch.testing.assert_close(segment, torch.full_like(segment, 0.04), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(geometry.arc[:, -1], torch.full((512,), 4.0))
    assert float(geometry.curvature.abs().max()) <= 0.8 + 1.0e-6
    assert float(geometry.target_peak_curvature.min()) >= 0.25 - 1.0e-6
    assert float(geometry.target_peak_curvature.max()) <= 0.8 + 1.0e-6
    assert float((geometry.target_peak_curvature >= 0.79).float().mean()) < 0.04
    assert float(torch.rad2deg(geometry.yaw.abs()).max()) <= 135.0 + 1.0e-4

    tangent_yaw = torch.atan2(delta[..., 1], delta[..., 0])
    angular_error = torch.atan2(
        torch.sin(tangent_yaw - geometry.yaw[:, :-1]),
        torch.cos(tangent_yaw - geometry.yaw[:, :-1]),
    )
    assert float(angular_error.abs().max()) < 1.0e-5
    # This component is deliberately not another near-straight distribution.
    assert float(geometry.curvature.abs().amax(dim=1).median()) > 0.35


@pytest.mark.parametrize("rounded", [False, True])
def test_waypoint_geometry_is_bounded_non_degenerate_and_non_intersecting(rounded: bool) -> None:
    torch.manual_seed(29)
    geometry = sample_waypoint_geometry(512, "cpu", rounded=rounded)

    segment = torch.linalg.vector_norm(geometry.xy[:, 1:] - geometry.xy[:, :-1], dim=2)
    torch.testing.assert_close(segment, torch.full_like(segment, 0.04), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(geometry.arc[:, -1], torch.full((512,), 4.0))
    assert int(geometry.segment_count.min()) >= 2
    assert int(geometry.segment_count.max()) <= 5
    count_histogram = torch.bincount(geometry.segment_count, minlength=6)[2:6]
    assert int(count_histogram.max() - count_histogram.min()) < 60
    active_segments = torch.arange(5)[None, :] < geometry.segment_count[:, None]
    assert float(geometry.segment_lengths[active_segments].min()) >= 0.55 - 1.0e-6
    active_turns = torch.arange(4)[None, :] < (geometry.segment_count - 1)[:, None]
    turn_deg = torch.rad2deg(geometry.turn_angles[active_turns].abs())
    assert float(turn_deg.min()) >= 15.0 - 1.0e-4
    assert float(turn_deg.max()) <= 90.0 + 1.0e-4
    cumulative_heading_deg = torch.rad2deg(torch.cumsum(geometry.turn_angles, dim=1).abs())
    assert float(cumulative_heading_deg[active_turns].max()) <= 135.0 + 1.0e-4
    assert float(torch.linalg.vector_norm(geometry.xy[:, -1], dim=1).min()) >= 1.0

    # Reconstruct the control polygon from its exact sampled parameters.
    headings = torch.cat(
        (torch.zeros(512, 1), torch.cumsum(geometry.turn_angles, dim=1)), dim=1
    )
    increments = geometry.segment_lengths[..., None] * torch.stack(
        (torch.cos(headings), torch.sin(headings)), dim=2
    )
    vertices = torch.cat((torch.zeros(512, 1, 2), torch.cumsum(increments, dim=1)), dim=1)
    assert not torch.any(_has_proper_self_intersection(vertices, geometry.segment_count))


def test_rounded_and_hard_waypoints_have_distinct_corner_regularities() -> None:
    torch.manual_seed(31)
    rounded = sample_waypoint_geometry(128, "cpu", rounded=True)
    torch.manual_seed(31)
    hard = sample_waypoint_geometry(128, "cpu", rounded=False)

    # Parameter sampling is identical under the same seed; only realization is
    # different.  Hard paths have exact zero radius and larger one-step turns.
    torch.testing.assert_close(rounded.segment_count, hard.segment_count)
    torch.testing.assert_close(rounded.segment_lengths, hard.segment_lengths)
    torch.testing.assert_close(rounded.turn_angles, hard.turn_angles)
    assert torch.all(hard.corner_radii == 0.0)
    assert torch.all(rounded.corner_radii[rounded.corner_radii > 0.0] >= 0.12)
    rounded_step = torch.diff(rounded.yaw, dim=1).abs().amax(dim=1)
    hard_step = torch.diff(hard.yaw, dim=1).abs().amax(dim=1)
    assert torch.all(rounded_step < hard_step)
