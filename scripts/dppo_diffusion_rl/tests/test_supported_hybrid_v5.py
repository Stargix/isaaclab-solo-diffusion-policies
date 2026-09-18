from __future__ import annotations

import torch

from scripts.dppo_diffusion_rl.supported_hybrid_v5 import SupportedHybridV5RouteBank


def _family(bank: SupportedHybridV5RouteBank) -> torch.Tensor:
    return torch.where(
        bank.geometry_class == bank.SMOOTH,
        bank.smooth_subclass,
        bank.geometry_class + 1,
    )


def test_v5_has_exact_initial_coverage_and_supported_targets() -> None:
    torch.manual_seed(42)
    count = 4096
    bank = SupportedHybridV5RouteBank(
        count,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(count), stratified=True, speed_max=1.0)

    assert torch.bincount(bank.coverage_class, minlength=5).tolist() == [
        2048,
        512,
        512,
        512,
        512,
    ]
    family = _family(bank)
    for coverage_class in range(5):
        expected = 512 if coverage_class else 2048
        assert torch.bincount(
            family[bank.coverage_class == coverage_class], minlength=4
        ).tolist() == [expected // 4] * 4

    fast = (bank.coverage_class == bank.FAST_CROUCH_TO_WALK) | (
        bank.coverage_class == bank.FAST_WALK_TO_CROUCH
    )
    repeated = (bank.coverage_class == bank.REPEATED_WALK_START) | (
        bank.coverage_class == bank.REPEATED_CROUCH_START
    )
    assert torch.all(bank.speed[fast] >= bank.FAST_SPEED_MIN_MPS)
    assert torch.all(bank.speed[fast] <= bank.FAST_SPEED_MAX_MPS)
    assert torch.all(
        bank.speed[fast] <= bank.maximum_feasible_mean_speed[fast] + 1.0e-6
    )
    assert torch.all(bank.speed[repeated] >= bank.REPEATED_SPEED_MIN_MPS)
    assert torch.all(bank.speed[repeated] <= bank.REPEATED_SPEED_MAX_MPS)


def test_v5_fast_profiles_have_only_a_short_crouch_section() -> None:
    torch.manual_seed(7)
    count = 2048
    bank = SupportedHybridV5RouteBank(
        count,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(count), stratified=True, speed_max=1.0)

    for coverage_class in (
        bank.FAST_CROUCH_TO_WALK,
        bank.FAST_WALK_TO_CROUCH,
    ):
        mask = bank.coverage_class == coverage_class
        crouch_fraction = torch.isclose(
            bank.height[mask], torch.tensor(bank.CROUCH_HEIGHT_M), atol=1.0e-5
        ).float().mean(dim=1)
        crouch_length = crouch_fraction * bank.length_m
        # One grid point of tolerance accounts for the sampled discontinuity.
        ds = bank.length_m / (bank.points - 1)
        assert torch.all(crouch_length >= bank.RESTRICTED_SECTION_MIN_M - ds)
        assert torch.all(crouch_length <= bank.RESTRICTED_SECTION_MAX_M + ds)


def test_v5_repeated_profiles_alternate_and_mask_every_boundary() -> None:
    torch.manual_seed(11)
    count = 512
    bank = SupportedHybridV5RouteBank(
        count,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(count), stratified=True, speed_max=1.0)

    repeated = torch.nonzero(
        bank.coverage_class == bank.REPEATED_WALK_START, as_tuple=False
    ).squeeze(1)
    assert repeated.numel() > 0
    env_id = int(repeated[0])
    finite_boundaries = bank.profile_boundaries[env_id][
        torch.isfinite(bank.profile_boundaries[env_id])
    ]
    assert 3 <= finite_boundaries.numel() <= bank.MAX_PROFILE_BOUNDARIES

    sampled_height = []
    for start, end in zip(
        torch.cat((torch.tensor([0.0]), finite_boundaries)),
        torch.cat((finite_boundaries, torch.tensor([bank.length_m]))),
    ):
        midpoint = 0.5 * (start + end)
        column = torch.argmin((bank.arc[env_id] - midpoint).abs())
        sampled_height.append(float(bank.height[env_id, column]))
    for first, second in zip(sampled_height, sampled_height[1:]):
        assert abs(first - second) > 0.1

    positions = torch.zeros(count, 2)
    boundary_column = torch.argmin(
        (bank.arc[env_id] - finite_boundaries[1]).abs()
    )
    positions[env_id] = bank.xy[env_id, boundary_column]
    bank.update(positions)
    assert float(bank.height_tracking_weight[env_id]) == 0.0


def test_v5_small_asynchronous_reset_never_admits_infeasible_fast_target() -> None:
    torch.manual_seed(19)
    bank = SupportedHybridV5RouteBank(
        64,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(64), stratified=True, speed_max=1.0)
    # The next schedule block contains fast samples but deliberately resets a
    # small subset, exercising the safe replay fallback.
    env_ids = torch.arange(8)
    bank.reset(env_ids, stratified=True, speed_max=1.0)
    fast = (bank.coverage_class[env_ids] == bank.FAST_CROUCH_TO_WALK) | (
        bank.coverage_class[env_ids] == bank.FAST_WALK_TO_CROUCH
    )
    assert torch.all(
        bank.speed[env_ids][fast]
        <= bank.maximum_feasible_mean_speed[env_ids][fast] + 1.0e-6
    )
