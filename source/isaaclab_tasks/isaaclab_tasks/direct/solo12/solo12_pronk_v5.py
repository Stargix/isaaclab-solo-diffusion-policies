# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pronk acquisition with correctly timed, geometry-verified flight intervals."""

import math

import torch
from isaaclab.utils import configclass

from .pronk_gait import IntervalPronkCycleTracker
from .solo12_pronk_v4 import Solo12PronkV4Env, Solo12PronkV4EnvCfg, Solo12PronkV4PPORunnerCfg


@configclass
class Solo12PronkV5EnvCfg(Solo12PronkV4EnvCfg):
    # Time unloading, not only the small portion >3 mm above the ground.
    # Two control samples reject single-frame contact gaps; apex geometry and
    # coordinated, supported landing remain mandatory before any payout.
    pronk_minimum_flight_s = 0.04
    pronk_target_flight_s = 0.12
    pronk_cycle_reward_scale = 3.0
    # Restore the original pronk velocity reward width. V2-v4's 0.5 tolerates
    # yaw/lateral errors similar to the entire requested steering envelope.
    tracking_std = 0.20
    pronk_failure_reward_scale = -5.0
    noise_initial_fraction = 0.25
    initial_max_delay_steps = 0

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.noise_initial_fraction) or not 0.0 <= self.noise_initial_fraction <= 1.0:
            raise ValueError("Initial observation noise fraction must lie in [0, 1].")
        if (not isinstance(self.initial_max_delay_steps, int)
                or not self.actuation_delay_range[0] <= self.initial_max_delay_steps <= self.actuation_delay_range[1]):
            raise ValueError("Initial delay maximum must lie inside the final actuation-delay range.")


class Solo12PronkV5Env(Solo12PronkV4Env):
    cfg: Solo12PronkV5EnvCfg
    cycle_tracker_type = IntervalPronkCycleTracker

    def _noise_fraction(self):
        initial = self.cfg.noise_initial_fraction
        return initial + self._curriculum_fraction() * (1.0 - initial)

    def _maximum_actuation_delay(self):
        low, high = self.cfg.actuation_delay_range
        initial = self.cfg.initial_max_delay_steps
        maximum = math.floor(initial + self._curriculum_fraction() * (high - initial) + 1.0e-6)
        return min(high, max(low, maximum))

    def _maybe_corrupt(self, values, noise_range):
        corrupted = super()._maybe_corrupt(values, noise_range)
        fraction = self._noise_fraction()
        if fraction == 1.0 or not self.cfg.enable_observation_corruption:
            return corrupted
        return values + fraction * (corrupted - values)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        maximum = self._maximum_actuation_delay()
        # At full difficulty retain the parent's exact delay distribution.
        # During acquisition sample uniformly within the current range.
        if maximum < self.cfg.actuation_delay_range[1]:
            delays = torch.randint(self.cfg.actuation_delay_range[0], maximum + 1,
                                   (len(env_ids),), device=self.device, dtype=torch.int)
            self._action_delay_steps[env_ids] = delays
            self._action_delay_buffer.set_time_lag(delays, env_ids)
            self._action_delay_buffer.reset(env_ids)

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.extras["log"].update({
            "Curriculum/pronk_observation_noise_fraction": self._noise_fraction(),
            "Curriculum/pronk_max_actuation_delay_steps": self._maximum_actuation_delay(),
        })
        return reward


@configclass
class Solo12PronkV5PPORunnerCfg(Solo12PronkV4PPORunnerCfg):
    run_name = "pronk_v5_interval_flight_seed42"
