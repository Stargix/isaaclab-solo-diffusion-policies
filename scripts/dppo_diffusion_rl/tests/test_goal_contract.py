from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.diffusion_policy.train.conditioning.goal_builder import (
    build_geometric_height_profile_goal_from_path,
)
from scripts.residual_diffusion_rl.routes import RouteBank


def test_route_goal16_matches_phase_a_builder() -> None:
    bank = RouteBank(1, "cpu", points=101, length_m=4.0)
    bank.reset(torch.tensor([0]), stage=0)
    bank.speed[:] = 0.4
    bank.height[0] = torch.where(bank.arc[0] < 0.4, 0.2932, 0.1705)
    position = torch.zeros(1, 2)
    bank.update(position)

    actual = bank.geometric_height_profile_goal(
        position,
        torch.zeros(1),
        horizon_s=2.0,
        v_clip=2.0,
    )[0]
    path = np.column_stack((bank.xy[0].numpy(), bank.height[0].numpy()))
    expected = build_geometric_height_profile_goal_from_path(
        path,
        bank.arc[0].numpy(),
        bank.yaw[0].numpy(),
        np.asarray([0.0, 0.0, 0.2932], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        goal_horizon_steps=100,
        dt=0.02,
        speed=0.4,
        start_idx=0,
        v_avg_clip=2.0,
    )

    assert tuple(actual.shape) == (16,)
    torch.testing.assert_close(
        actual,
        torch.from_numpy(expected),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        actual[[2, 5, 8, 11]],
        torch.tensor([0.2932, 0.1705, 0.1705, 0.1705]),
    )
    assert float(actual[12]) == pytest.approx(0.2932, abs=1.0e-5)
