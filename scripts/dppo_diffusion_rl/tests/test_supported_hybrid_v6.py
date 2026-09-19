from __future__ import annotations

import torch

from scripts.dppo_diffusion_rl.supported_hybrid_v6 import SupportedHybridV6RouteBank


def test_v6_preserves_v5_coverage_and_adds_exact_low_fast_anchors() -> None:
    torch.manual_seed(42)
    count = 4096
    bank = SupportedHybridV6RouteBank(
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
    for fast_class in (
        bank.FAST_CROUCH_TO_WALK,
        bank.FAST_WALK_TO_CROUCH,
    ):
        ids = torch.nonzero(bank.coverage_class == fast_class, as_tuple=False).squeeze(1)
        bands = bank.fast_speed_band[ids]
        assert int((bands == bank.FAST_BAND_LOW_ANCHOR).sum()) == 128
        assert int((bands == bank.FAST_BAND_FRONTIER).sum()) == 384

        low = ids[bands == bank.FAST_BAND_LOW_ANCHOR]
        frontier = ids[bands == bank.FAST_BAND_FRONTIER]
        assert torch.all(bank.speed[low] >= bank.FAST_SPEED_MIN_MPS)
        assert torch.all(bank.speed[low] <= bank.LOW_FAST_MAX_MPS)
        assert torch.all(bank.speed[low] <= bank.maximum_feasible_mean_speed[low])

        feasible = bank.maximum_feasible_mean_speed[frontier].clamp(
            max=bank.FAST_SPEED_MAX_MPS
        )
        frontier_low = bank.FAST_SPEED_MIN_MPS + bank.FAST_FRONTIER_FRACTION * (
            feasible - bank.FAST_SPEED_MIN_MPS
        )
        assert torch.all(bank.speed[frontier] >= frontier_low - 1.0e-6)


def test_v6_update_groups_are_private_sampler_labels() -> None:
    torch.manual_seed(7)
    bank = SupportedHybridV6RouteBank(
        128,
        "cpu",
        transition_boundary_min_m=0.8,
        transition_boundary_max_m=3.2,
    )
    bank.reset(torch.arange(128), stratified=True, speed_max=1.0)

    assert bank.coverage_class.shape == (128,)
    assert bank.fast_speed_band.shape == (128,)
    # Goal dimensionality remains schema-8; neither private label is appended.
    goals = bank.geometric_height_profile_goal(
        torch.zeros(128, 2), torch.zeros(128), horizon_s=2.0, v_clip=2.0
    )
    assert goals.shape == (128, 16)
