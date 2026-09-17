from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.diffusion_policy.train.conditioning.goal_builder import (
    build_geometric_height_profile_goal_batch_from_path,
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


def test_preview_speed_changes_geometry_and_vavg_consistently() -> None:
    bank = RouteBank(1, "cpu", points=101, length_m=4.0)
    bank.arc[0] = torch.linspace(0.0, 4.0, 101)
    bank.xy[0, :, 0] = bank.arc[0]
    bank.xy[0, :, 1] = 0.0
    bank.yaw[0] = 0.0
    bank.height[0] = 0.2932
    bank.speed[0] = 0.4
    position = torch.zeros(1, 2)
    bank.update(position)

    nominal = bank.geometric_height_profile_goal(
        position, torch.zeros(1), horizon_s=2.0, v_clip=2.0
    )[0]
    recovered = bank.geometric_height_profile_goal(
        position,
        torch.zeros(1),
        horizon_s=2.0,
        v_clip=2.0,
        preview_speed=torch.tensor([0.8]),
    )[0]

    assert float(recovered[9]) > float(nominal[9])
    # Spatial discretization may select the next 4 cm route point. The key
    # contract is exact internal agreement between selected preview length and
    # the scalar average pace, rather than either matching the raw request.
    assert float(nominal[-1]) == pytest.approx(float(nominal[9]) / 2.0, abs=1.0e-5)
    assert float(recovered[-1]) == pytest.approx(float(recovered[9]) / 2.0, abs=1.0e-5)


def test_batch_goal_accepts_per_environment_preview_pace() -> None:
    cumulative = np.linspace(0.0, 4.0, 101, dtype=np.float32)
    path = np.column_stack(
        (cumulative, np.zeros_like(cumulative), np.full_like(cumulative, 0.2932))
    )
    goals = build_geometric_height_profile_goal_batch_from_path(
        path,
        cumulative,
        np.zeros_like(cumulative),
        np.zeros((2, 3), dtype=np.float32),
        np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1)),
        goal_horizon_steps=100,
        dt=0.02,
        speed=np.asarray([0.4, 0.8], dtype=np.float32),
        path_progress=np.zeros(2, dtype=np.int32),
        v_avg_clip=2.0,
    )
    assert goals[1, 9] > goals[0, 9]
    np.testing.assert_allclose(goals[:, -1], goals[:, 9] / 2.0, atol=1.0e-5)
