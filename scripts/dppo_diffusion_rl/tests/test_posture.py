from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.diffusion_policy.evaluation.metrics import compute_first_task_success
from scripts.dppo_diffusion_rl.checkpointing import validate_resume_task_config
from scripts.dppo_diffusion_rl.posture import CausalPostureWindow, causal_posture_trace, posture_window_steps
from scripts.dppo_diffusion_rl.tests.test_reward import _reward
from scripts.dppo_diffusion_rl.rewards import TaskRewardWeights


@pytest.mark.parametrize("mode,steps", [("instantaneous", 1), ("nominal_window", 20)])
def test_window_steps(mode, steps):
    assert posture_window_steps(mode, .4, .02) == steps


@pytest.mark.parametrize("mode,window,dt", [("bad", .4, .02), ("nominal_window", .41, .02), ("instantaneous", float("nan"), .02), ("nominal_window", 0, .02)])
def test_bad_window_rejected(mode, window, dt):
    with pytest.raises(ValueError):
        posture_window_steps(mode, window, dt)


def test_partial_start_and_masked_reset_have_no_fake_history():
    observer = CausalPostureWindow(2, 3, "cpu")
    observer.update(torch.tensor([.3, .2]), torch.tensor([.3, .2]))
    torch.testing.assert_close(observer.mean[:, 0], torch.tensor([.3, .2]))
    observer.update(torch.tensor([.1, .4]), torch.tensor([.3, .2]))
    observer.reset(torch.tensor([0]))
    observer.update(torch.tensor([.5, .6]), torch.tensor([.3, .2]))
    torch.testing.assert_close(observer.mean[:, 0], torch.tensor([.5, .4]))
    assert observer.count.tolist() == [1, 3]
    observer.update(torch.tensor([.7, .8]), torch.tensor([.3, .2]))
    torch.testing.assert_close(observer.mean[:, 0], torch.tensor([.6, .6]))


def test_sinus_is_reduced_without_removing_dc_bias():
    t = np.arange(200) * .02
    raw = .2632 + .035 * np.sin(2 * np.pi * 5 * t)
    target = np.full(200, .2932)
    measured, required = causal_posture_trace(raw, target, mode="nominal_window", window_s=.4, dt=.02)
    assert np.std(measured[20:]) < 1e-6
    assert np.mean(required[20:] - measured[20:]) == pytest.approx(.03, abs=1e-6)
    assert np.max(np.abs(raw - target)) > .05
    assert np.max(np.abs(measured[20:] - required[20:])) < .05


@pytest.mark.parametrize("direction", [1, -1])
def test_matched_transition_is_not_penalized_but_delayed_transition_is(direction):
    target = np.r_[np.full(30, .17), np.full(30, .2932)]
    if direction == -1:
        target = target[::-1].copy()
    measured, required = causal_posture_trace(target, target, mode="nominal_window", window_s=.4, dt=.02)
    np.testing.assert_array_equal(measured, required)
    late = np.r_[np.full(10, target[0]), target[:-10]]
    measured, required = causal_posture_trace(late, target, mode="nominal_window", window_s=.4, dt=.02)
    assert np.max(np.abs(measured - required)) > .06
    assert abs(measured[-1] - target[-1]) < 1e-6


def test_filter_never_mutates_raw_safety_samples():
    raw = np.full(40, .29)
    raw[25] = .55  # hypothetical ceiling breach: remains visible to safety
    before = raw.copy()
    smooth, _ = causal_posture_trace(raw, np.full(40, .29), mode="nominal_window", window_s=.4, dt=.02)
    np.testing.assert_array_equal(raw, before)
    assert raw.max() == .55 and smooth.max() < .31


def test_vz_zero_removes_cost_not_adds_flight_bonus_and_falls_still_cost():
    weights = TaskRewardWeights(vertical_velocity=0)
    still, _ = _reward(progress_delta=torch.ones(2) * .02, weights=weights)
    flying, _ = _reward(progress_delta=torch.ones(2) * .02, vertical_velocity=torch.ones(2) * 2, weights=weights)
    torch.testing.assert_close(still, flying)
    fallen, _ = _reward(fell=torch.ones(2, dtype=torch.bool), weights=weights)
    assert torch.all(fallen < 0)


def test_old_resume_defaults_match_but_semantic_change_rejected():
    old = {"dppo_task_config": {"route_stage": 2}}
    validate_resume_task_config(old, {"route_stage": 2, "posture_tracking_mode": "instantaneous", "posture_window_s": .4, "vertical_velocity_reward_weight": .025})
    with pytest.raises(ValueError, match="posture_tracking_mode"):
        validate_resume_task_config(old, {"route_stage": 2, "posture_tracking_mode": "nominal_window"})


@pytest.mark.parametrize("field,value", [("posture_window_s", .6), ("vertical_velocity_reward_weight", 0.)])
def test_changed_window_or_vertical_cost_rejects_optimizer_resume(field, value):
    with pytest.raises(ValueError, match=field):
        validate_resume_task_config({"dppo_task_config": {"route_stage": 2}}, {"route_stage": 2, field: value})


def test_legacy_trace_keeps_precision():
    raw = np.array([.293200000123, .170500000123])
    measured, target = causal_posture_trace(raw, raw, mode="instantaneous", window_s=.4, dt=.02)
    np.testing.assert_array_equal(raw, measured)
    np.testing.assert_array_equal(raw, target)


def test_evaluation_changes_only_posture_not_first_arrival_time():
    n = 100
    positions = np.column_stack((np.arange(1, n + 1) * .02, np.zeros(n)))
    height = .2632 + .04 * np.sin(2 * np.pi * 5 * np.arange(1, n + 1) * .02)
    # Native-flight trough at first arrival (~94): raw fails, nominal passes.
    height[93:96] = .22
    args = dict(yaws_rad=np.zeros(n), heights_m=height, requested_speed_m_s=1., target_progress_m=2., target_yaw_rad=0., target_height_m=.2932, dt=.02, start_position_xy=np.zeros(2), target_heights_m=np.full(n, .2932), profile_height_mae_tolerance_m=.04)
    raw = compute_first_task_success(positions, np.array([[0., 0.], [2., 0.]]), **args)
    nominal = compute_first_task_success(positions, np.array([[0., 0.], [2., 0.]]), **args, posture_tracking_mode="nominal_window")
    assert raw.time_s == nominal.time_s
    assert not raw.terminal_height_ok and nominal.terminal_height_ok
    assert nominal.success
