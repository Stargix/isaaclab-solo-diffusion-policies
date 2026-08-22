from __future__ import annotations

import torch

from scripts.dppo_diffusion_rl.rewards import average_speed_error, path_task_reward


def _reward(**overrides):
    batch = 2
    values = {
        "progress_delta": torch.zeros(batch),
        "cross_track": torch.zeros(batch),
        "mean_speed_error": torch.zeros(batch),
        "yaw_error": torch.zeros(batch),
        "height_error": torch.zeros(batch),
        "projected_gravity_xy": torch.zeros(batch, 2),
        "vertical_velocity": torch.zeros(batch),
        "success": torch.zeros(batch, dtype=torch.bool),
        "failed": torch.zeros(batch, dtype=torch.bool),
        "fell": torch.zeros(batch, dtype=torch.bool),
        "dt": 0.02,
    }
    values.update(overrides)
    return path_task_reward(**values)


def test_no_alive_or_clock_reward() -> None:
    reward, terms = _reward()
    torch.testing.assert_close(reward, torch.zeros_like(reward))
    assert "alive" not in terms and "time" not in terms and "schedule" not in terms


def test_progress_is_a_potential_difference() -> None:
    reward, terms = _reward(progress_delta=torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(terms["progress"], torch.tensor([0.3, 0.6]))
    assert torch.all(reward > 0.0)


def test_fall_is_strictly_worse_than_ordinary_failure() -> None:
    ordinary, _ = _reward(failed=torch.ones(2, dtype=torch.bool))
    fall, _ = _reward(
        failed=torch.ones(2, dtype=torch.bool), fell=torch.ones(2, dtype=torch.bool)
    )
    assert torch.all(fall < ordinary)


def test_average_speed_has_neutral_startup_and_correct_units() -> None:
    progress = torch.tensor([0.0, 1.0])
    elapsed = torch.tensor([0.1, 2.0])
    desired = torch.tensor([0.4, 0.4])
    error = average_speed_error(progress, elapsed, desired)
    torch.testing.assert_close(error, torch.tensor([0.0, 0.1]))
