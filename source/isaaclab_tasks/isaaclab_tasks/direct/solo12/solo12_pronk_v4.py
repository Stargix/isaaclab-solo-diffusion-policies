# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Recover physical pronk flight while adapting the original expert to steering."""

from __future__ import annotations

import math

import torch
from isaaclab.utils import configclass

from .pronk_gait import SettledPronkCycleTracker
from .solo12_pronk_env import Solo12PronkEnv, compute_pronk_quadruped_symmetry
from .solo12_pronk_v2 import Solo12PronkV2EnvCfg, Solo12PronkV2PPORunnerCfg


@configclass
class Solo12PronkV4EnvCfg(Solo12PronkV2EnvCfg):
    command_lin_vel_y_range = (-0.25, 0.25)
    command_ang_vel_z_range = (-0.25, 0.25)

    # Original nominal captures show approximately 0.16 s contact-free bouts.
    # A geometrically validated 0.08 s flight gets zero quality; 0.12 s reaches
    # full quality. The inherited 0.20 s upper bound still rejects long jumps.
    pronk_minimum_flight_s = 0.08
    pronk_target_flight_s = 0.12
    pronk_landing_settle_s = 0.06
    pronk_cycle_reward_scale = 2.0  # reward per validated airborne second, not per hop
    pronk_failure_reward_scale = -2.0  # one event per physical termination

    # Preserve impact penalties above the original gait's usual contact loads.
    # This is a soft reward threshold, not a hardware force limit.
    foot_contact_safe_threshold = 30.0

    steering_initial_limit = 0.10
    steering_warmup_steps = 32 * 75
    steering_ramp_steps = 32 * 175  # full command envelope from PPO iteration 250
    push_initial_fraction = 0.25

    def __post_init__(self):
        super().__post_init__()
        if not self.pronk_minimum_flight_s < self.pronk_target_flight_s <= self.pronk_maximum_flight_s:
            raise ValueError("Pronk v4 requires minimum < target <= maximum flight duration.")
        if not math.isfinite(self.pronk_landing_settle_s) or self.pronk_landing_settle_s < self.sim.dt * self.decimation:
            raise ValueError("Pronk v4 settling must span at least one control step.")
        if not math.isfinite(self.pronk_failure_reward_scale) or self.pronk_failure_reward_scale >= 0.0:
            raise ValueError("Pronk v4 physical-failure event reward must be finite and negative.")
        if self.steering_warmup_steps < 0 or self.steering_ramp_steps <= 0:
            raise ValueError("Pronk v4 requires nonnegative warmup and positive curriculum duration.")
        if not 0.0 < self.steering_initial_limit <= min(self.command_lin_vel_y_range[1], self.command_ang_vel_z_range[1]):
            raise ValueError("Initial steering bounds must lie inside the final command envelope.")
        if not 0.0 <= self.push_initial_fraction <= 1.0:
            raise ValueError("Initial push fraction must lie in [0, 1].")
        # Match the local adaptation contract. All other startup randomization,
        # observation corruption, delay, reset noise and pushes are retained.
        if self.events is not None:
            self.events.joint_friction = None


class Solo12PronkV4Env(Solo12PronkEnv):
    cfg: Solo12PronkV4EnvCfg
    cycle_tracker_type = SettledPronkCycleTracker

    def __init__(self, cfg: Solo12PronkV4EnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._pronk_cycle_tracker = self.cycle_tracker_type(
            self.num_envs, self.device, self.step_dt,
            minimum_flight_s=cfg.pronk_minimum_flight_s,
            maximum_flight_s=cfg.pronk_maximum_flight_s,
            landing_window_s=cfg.pronk_landing_window_s,
            support_window_s=cfg.pronk_support_window_s,
            minimum_clearance_m=cfg.pronk_minimum_clearance_m,
            warmup_s=cfg.pronk_warmup_s,
            tracking_std_mps=cfg.tracking_std,
            synchronization_std_s=cfg.pronk_synchronization_std_s,
            target_flight_s=cfg.pronk_target_flight_s,
            landing_settle_s=cfg.pronk_landing_settle_s,
        )
        self._episode_sums["pronk_failure"] = torch.zeros(self.num_envs, device=self.device)

    def _curriculum_fraction(self) -> float:
        return min(1.0, max(0.0, (self.common_step_counter - self.cfg.steering_warmup_steps)
                            / self.cfg.steering_ramp_steps))

    def _resample_commands(self, env_ids, allow_opposite: bool = True):
        super()._resample_commands(env_ids, allow_opposite=allow_opposite)
        fraction = self._curriculum_fraction()
        for column, maximum in ((1, self.cfg.command_lin_vel_y_range[1]),
                                (2, self.cfg.command_ang_vel_z_range[1])):
            limit = self.cfg.steering_initial_limit + fraction * (maximum - self.cfg.steering_initial_limit)
            self._commands[env_ids, column] *= limit / maximum

    def _start_base_pushes(self, env_ids):
        super()._start_base_pushes(env_ids)
        fraction = self.cfg.push_initial_fraction + self._curriculum_fraction() * (1.0 - self.cfg.push_initial_fraction)
        self._base_push_forces_b[env_ids] *= fraction

    def _compute_pronk_reward(self, components):
        self._v4_cycle_components = components
        return components["gait"] * components["flight_duration"] * self.cfg.pronk_cycle_reward_scale

    def _get_rewards(self):
        reward = super()._get_rewards()
        failure_reward = self.reset_terminated.float() * self.cfg.pronk_failure_reward_scale
        self._episode_sums["pronk_failure"] += failure_reward
        self._episode_reward_sums += failure_reward
        components = self._v4_cycle_components
        self.extras["log"].update({
            "RewardsPerStep/pronk_failure": failure_reward.mean().item(),
            "RewardsPerStep/total": (reward + failure_reward).mean().item(),
            # Divide window-averaged validated airtime by window-averaged
            # completed cycles to recover the mean duration without averaging
            # zeros from control steps with no landing.
            "Metrics/pronk_validated_airtime_s_per_step": components["flight_duration"].mean().item(),
            "Curriculum/pronk_steering_fraction": self._curriculum_fraction(),
        })
        return reward + failure_reward


@configclass
class Solo12PronkV4PPORunnerCfg(Solo12PronkV2PPORunnerCfg):
    max_iterations = 1000
    save_interval = 25
    run_name = "pronk_v4_flight_retention_seed42"

    def __post_init__(self):
        super().__post_init__()
        # Initialize from the original jumping expert, not a low-flight pilot.
        self.algorithm.reference_checkpoint = "checkpoints/jumpy_safe.pt"
        self.algorithm.reference_checkpoint_sha256 = "6496d831f4d3b8fec8793eb740351edbe6581198787d713ac7010371a0ff4183"
        self.policy.init_noise_std = 0.10
        self.algorithm.learning_rate = 5.0e-5
        # The reference is not assumed front/back equivariant. Apply the full
        # quadruped group through a mirror loss on transformed action targets;
        # PPO ratios and the reference penalty use actual rollout observations.
        self.algorithm.symmetry_cfg.data_augmentation_func = compute_pronk_quadruped_symmetry
        self.algorithm.symmetry_cfg.use_data_augmentation = False
        self.algorithm.symmetry_cfg.use_mirror_loss = True
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 0.1
