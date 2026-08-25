from __future__ import annotations

import torch

from scripts.dppo_diffusion_rl.conditioning import remaining_speed_budget
from scripts.dppo_diffusion_rl.rewards import (
    average_speed_error,
    bounded_huber,
    path_task_reward,
    schedule_error_improvement,
    terminal_pose_potential,
)
from scripts.dppo_diffusion_rl.config import DPPOConfig


def _speed_budget(progress: list[float], elapsed: list[float]) -> torch.Tensor:
    progress_t = torch.tensor(progress)
    return remaining_speed_budget(
        remaining_distance=4.0 - progress_t,
        route_length=torch.full_like(progress_t, 4.0),
        desired_mean_speed=torch.full_like(progress_t, 0.4),
        elapsed_s=torch.tensor(elapsed),
        max_speed=0.6,
        min_remaining_time_s=0.02,
    )


def test_speed_budget_equals_request_at_reset() -> None:
    torch.testing.assert_close(_speed_budget([0.0], [0.0]), torch.tensor([0.4]))


def test_speed_budget_exposes_ahead_and_behind_schedule() -> None:
    budget = _speed_budget([2.5, 2.0, 1.5], [5.0, 5.0, 5.0])
    torch.testing.assert_close(budget, torch.tensor([0.3, 0.4, 0.5]))


def test_speed_budget_clips_late_debt_and_tapers_at_endpoint() -> None:
    torch.testing.assert_close(
        _speed_budget([3.0, 4.0], [11.0, 11.0]), torch.tensor([0.6, 0.0])
    )


def _reward(**overrides):
    batch = 2
    values = {
        "progress_delta": torch.zeros(batch),
        "schedule_improvement": torch.zeros(batch),
        "terminal_pose_improvement": torch.zeros(batch),
        "route_length": torch.full((batch,), 4.0),
        "maximum_schedule_distance": torch.full((batch,), 4.0),
        "cross_track": torch.zeros(batch),
        "yaw_error": torch.zeros(batch),
        "height_error": torch.zeros(batch),
        "projected_gravity_xy": torch.zeros(batch, 2),
        "vertical_velocity": torch.zeros(batch),
        "success": torch.zeros(batch, dtype=torch.bool),
        "arrival_failure": torch.zeros(batch, dtype=torch.bool),
        "timeout_failure": torch.zeros(batch, dtype=torch.bool),
        "corridor_failure": torch.zeros(batch, dtype=torch.bool),
        "overshoot_failure": torch.zeros(batch, dtype=torch.bool),
        "fell": torch.zeros(batch, dtype=torch.bool),
    }
    values.update(overrides)
    return path_task_reward(**values)


def test_no_alive_or_clock_reward() -> None:
    reward, terms = _reward()
    torch.testing.assert_close(reward, torch.zeros_like(reward))
    assert "alive" not in terms and "time" not in terms and "schedule" not in terms


def test_default_discount_preserves_endpoint_difference_rewards() -> None:
    assert DPPOConfig().gamma == 1.0


def test_progress_is_a_potential_difference() -> None:
    reward, terms = _reward(progress_delta=torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(terms["progress"], torch.tensor([0.3, 0.6]))
    assert torch.all(reward > 0.0)


def test_fall_is_strictly_worse_than_ordinary_failure() -> None:
    ordinary, _ = _reward(corridor_failure=torch.ones(2, dtype=torch.bool))
    fall, _ = _reward(fell=torch.ones(2, dtype=torch.bool))
    assert torch.all(fall < ordinary)


def test_failed_first_arrival_has_same_terminal_cost_as_timeout() -> None:
    arrival, arrival_terms = _reward(
        arrival_failure=torch.ones(2, dtype=torch.bool)
    )
    timeout, timeout_terms = _reward(
        timeout_failure=torch.ones(2, dtype=torch.bool)
    )
    torch.testing.assert_close(arrival, timeout)
    torch.testing.assert_close(
        arrival_terms["arrival_failure"], timeout_terms["timeout_failure"]
    )


def test_average_speed_has_neutral_startup_and_correct_units() -> None:
    progress = torch.tensor([0.0, 1.0])
    elapsed = torch.tensor([0.1, 2.0])
    desired = torch.tensor([0.4, 0.4])
    error = average_speed_error(progress, elapsed, desired)
    torch.testing.assert_close(error, torch.tensor([0.0, 0.1]))


def test_schedule_signal_telescopes_and_is_zero_on_perfect_pace() -> None:
    dt = 0.5
    elapsed = torch.arange(1, 9, dtype=torch.float32) * dt
    perfect_progress = torch.minimum(0.5 * elapsed, torch.tensor(2.0))
    perfect_delta = torch.diff(torch.cat((torch.zeros(1), perfect_progress)))
    perfect = schedule_error_improvement(
        progress=perfect_progress,
        progress_delta=perfect_delta,
        elapsed_s=elapsed,
        desired_speed=torch.full_like(elapsed, 0.5),
        route_length=torch.full_like(elapsed, 2.0),
        dt=dt,
    )
    torch.testing.assert_close(perfect, torch.zeros_like(perfect))

    standing = schedule_error_improvement(
        progress=torch.zeros_like(elapsed),
        progress_delta=torch.zeros_like(elapsed),
        elapsed_s=elapsed,
        desired_speed=torch.full_like(elapsed, 0.5),
        route_length=torch.full_like(elapsed, 2.0),
        dt=dt,
    )
    torch.testing.assert_close(standing.sum(), torch.tensor(-2.0))


def test_schedule_signal_keeps_late_arrival_error_after_nominal_arrival() -> None:
    dt = 1.0
    progress = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    improvement = schedule_error_improvement(
        progress=progress,
        progress_delta=torch.diff(torch.cat((torch.zeros(1), progress))),
        elapsed_s=torch.arange(1, 6, dtype=torch.float32),
        desired_speed=torch.full((5,), 0.5),
        route_length=torch.full((5,), 2.0),
        dt=dt,
    )
    # The route is completed one second late, so the telescoping potential is
    # exactly the negative final schedule error: -0.5 m.
    torch.testing.assert_close(improvement.sum(), torch.tensor(-0.5))


def test_terminal_pose_potential_guides_corrections_without_alive_reward() -> None:
    inactive = terminal_pose_potential(
        remaining_distance=torch.tensor([1.0]),
        actor_lookahead_distance=torch.tensor([0.4]),
        terminal_distance=torch.tensor([1.0]),
        terminal_yaw_error=torch.tensor([1.0]),
        terminal_height_error=torch.tensor([0.1]),
    )
    far = terminal_pose_potential(
        remaining_distance=torch.tensor([0.0]),
        actor_lookahead_distance=torch.tensor([0.4]),
        terminal_distance=torch.tensor([0.30]),
        terminal_yaw_error=torch.tensor([0.8]),
        terminal_height_error=torch.tensor([0.10]),
    )
    close = terminal_pose_potential(
        remaining_distance=torch.tensor([0.0]),
        actor_lookahead_distance=torch.tensor([0.4]),
        terminal_distance=torch.tensor([0.05]),
        terminal_yaw_error=torch.tensor([0.1]),
        terminal_height_error=torch.tensor([0.01]),
    )
    torch.testing.assert_close(inactive, torch.zeros_like(inactive))
    assert far < close < 0.0
    # A correction is rewarded only by the change in potential.
    reward, terms = _reward(terminal_pose_improvement=(close - far).expand(2))
    torch.testing.assert_close(reward, terms["terminal_pose"])
    assert torch.all(reward > 0.0)


def test_no_early_death_incentive() -> None:
    standing_timeout, _ = _reward(
        schedule_improvement=torch.full((2,), -4.0),
        timeout_failure=torch.ones(2, dtype=torch.bool),
    )
    immediate_fall, _ = _reward(fell=torch.ones(2, dtype=torch.bool))
    full_progress_hard_failure, _ = _reward(
        progress_delta=torch.full((2,), 4.0),
        corridor_failure=torch.ones(2, dtype=torch.bool),
    )
    assert torch.all(standing_timeout > immediate_fall)
    assert torch.all(standing_timeout > full_progress_hard_failure)


def test_tracking_costs_are_bounded_and_paid_only_while_advancing() -> None:
    huge = torch.full((2,), 1.0e6)
    stopped, stopped_terms = _reward(cross_track=huge, height_error=huge)
    torch.testing.assert_close(stopped, torch.zeros_like(stopped))
    moving, moving_terms = _reward(
        progress_delta=torch.ones(2), cross_track=huge, height_error=huge
    )
    assert torch.all(stopped_terms["path"] == 0.0)
    assert torch.all(moving_terms["path"] >= -1.25)
    assert torch.all(moving_terms["height"] >= -0.75)
    assert torch.isfinite(moving).all()
    bounded = bounded_huber(huge, 0.1)
    assert torch.all((bounded >= 0.0) & (bounded <= 1.0))
