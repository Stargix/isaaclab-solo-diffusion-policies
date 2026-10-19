# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Signed-speed continuation with independent bounded exposure ramps.

Unlike v7, a gait acquisition test cannot prevent training physical robustness.
Full difficulty must still pass a separate frozen evaluation before collection.
"""

import torch
from isaaclab.utils import configclass

from .pronk_schedule import ramp_fraction, slow_flight_multiplier
from .solo12_pronk_env import Solo12PronkEnv
from .solo12_pronk_v5 import Solo12PronkV5Env
from .solo12_pronk_v7 import Solo12PronkV7EnvCfg, Solo12PronkV7PPORunnerCfg


@configclass
class Solo12PronkV8EnvCfg(Solo12PronkV7EnvCfg):
    exposure_schedule_enabled = True
    # Units are control steps: runner uses 32 control steps per PPO iteration.
    steering_start_step = 32 * 100
    steering_end_step = 32 * 1200
    robustness_start_step = 32 * 300
    robustness_end_step = 32 * 1800
    steering_initial_limit = 0.05
    slow_command_probability = 0.30
    slow_command_abs_range = (0.15, 0.25)
    pronk_cycle_reward_scale = 6.0

    def __post_init__(self):
        super().__post_init__()
        ramp_fraction(0, self.steering_start_step, self.steering_end_step)
        ramp_fraction(0, self.robustness_start_step, self.robustness_end_step)
        if not 0 <= self.slow_command_probability <= 1:
            raise ValueError("Slow-command probability must lie in [0, 1].")
        low, high = self.slow_command_abs_range
        if not 0.10 <= low < high <= self.command_lin_vel_x_range[1]:
            raise ValueError("Slow commands must lie inside the signed moving range.")
        # V8 uses V5Env directly. The inherited measured gate is not executed.
        self._set_event_mode()

    def use_full_difficulty(self):
        self.exposure_schedule_enabled = False
        super().use_full_difficulty()


class Solo12PronkV8Env(Solo12PronkV5Env):
    cfg: Solo12PronkV8EnvCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._randomized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._randomization_order = torch.randperm(self.num_envs, device=self.device)
        if not cfg.exposure_schedule_enabled and cfg.events is not None:
            self._randomized.fill_(True)  # full DR was already applied at startup

    def _steering_fraction(self):
        if not self.cfg.exposure_schedule_enabled:
            return 1.0
        return ramp_fraction(self.common_step_counter, self.cfg.steering_start_step,
                             self.cfg.steering_end_step)

    def _curriculum_fraction(self):
        if not self.cfg.exposure_schedule_enabled:
            return 1.0
        return ramp_fraction(self.common_step_counter, self.cfg.robustness_start_step,
                             self.cfg.robustness_end_step)

    def _resample_commands(self, env_ids, allow_opposite=True):
        # Preserve the entire uniform signed vx distribution, adding deliberate
        # coverage of the slow commands that failed. Never overwrite stop holds.
        Solo12PronkEnv._resample_commands(self, env_ids, allow_opposite=allow_opposite)
        commands = self._commands[env_ids].clone()
        moving = commands.abs().amax(dim=-1) > 0
        slow = moving & (torch.rand(len(env_ids), device=self.device) < self.cfg.slow_command_probability)
        magnitude = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.slow_command_abs_range)
        sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)
        commands[slow, 0] = magnitude[slow] * sign[slow]
        for column, maximum in ((1, self.cfg.command_lin_vel_y_range[1]),
                                (2, self.cfg.command_ang_vel_z_range[1])):
            limit = self.cfg.steering_initial_limit + self._steering_fraction() * (maximum - self.cfg.steering_initial_limit)
            commands[:, column] *= limit / maximum
        self._commands[env_ids] = commands

    def _compute_pronk_reward(self, components):
        # V4 logs use this member; retain the same geometry/settled-landing
        # tracker and duration quality rather than paying for force-sensor gaps.
        self._v4_cycle_components = components
        speed = self._commands[:, :2].norm(dim=-1)
        return (components["gait"] * components["flight_duration"]
                * self.cfg.pronk_cycle_reward_scale * slow_flight_multiplier(speed))

    def _get_rewards(self):
        reward = super()._get_rewards()
        commands = self._commands
        speed = commands[:, :2].norm(dim=-1)
        eligible = (speed >= 0.10) & (self._pronk_cycle_tracker.age >= self.cfg.pronk_warmup_s)
        components = self._pronk_cycle_components
        log = self.extras["log"]
        for name, mask in (
            ("slow_forward", eligible & (commands[:, 0] >= 0.10) & (commands[:, 0] <= 0.25)),
            ("slow_reverse", eligible & (commands[:, 0] <= -0.10) & (commands[:, 0] >= -0.25)),
            ("moving", eligible),
        ):
            count = mask.sum().clamp_min(1)
            log[f"Diagnostics/{name}_airtime"] = ((components["flight_duration"] * mask).sum() / (count * self.step_dt)).item()
            log[f"Diagnostics/{name}_cadence_hz"] = ((components["completed"] * mask).sum() / (count * self.step_dt)).item()
            log[f"Diagnostics/{name}_sample_fraction"] = mask.float().mean().item()
        log.update({
            "Curriculum/pronk_steering_fraction": self._steering_fraction(),
            "Curriculum/pronk_robustness_fraction": self._curriculum_fraction(),
            "Curriculum/pronk_randomized_fraction": self._randomized.float().mean().item(),
            "Coverage/pronk_stop_fraction": (commands.abs().amax(dim=-1) == 0).float().mean().item(),
            "Coverage/pronk_reverse_fraction": (commands[:, 0] < -0.1).float().mean().item(),
        })
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        fraction = self._curriculum_fraction()
        fields = ("flexed_initial_joint_pos_noise_range", "reset_base_lin_vel_range", "reset_base_ang_vel_range")
        originals = {name: getattr(self.cfg, name) for name in fields}
        try:
            if self.cfg.exposure_schedule_enabled:
                for name, bounds in originals.items():
                    setattr(self.cfg, name, tuple(value * fraction for value in bounds))
            super()._reset_idx(env_ids)
        finally:
            for name, bounds in originals.items():
                setattr(self.cfg, name, bounds)
        if (hasattr(self, "_randomized") and self.cfg.exposure_schedule_enabled
                and getattr(self, "event_manager", None) is not None):
            selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            selected[self._randomization_order[:round(fraction * self.num_envs)]] = True
            new_ids = env_ids[selected[env_ids] & ~self._randomized[env_ids]]
            if len(new_ids):
                self.event_manager.apply("pronk_robust", env_ids=new_ids)
                self._randomized[new_ids] = True

    def get_training_summary(self):
        return {
            "exposure_schedule_enabled": self.cfg.exposure_schedule_enabled,
            "control_step": self.common_step_counter,
            "steering_fraction": self._steering_fraction(),
            "robustness_fraction": self._curriculum_fraction(),
            "randomized_environment_fraction": self._randomized.float().mean().item(),
            "full_envelope_exposed": self._steering_fraction() == 1 and self._curriculum_fraction() == 1,
            "frozen_evaluation_required": True,
            "acquisition_gate": "none: exposure is not learned competence",
            "longitudinal_range_from_first_episode": self.cfg.command_lin_vel_x_range,
        }


@configclass
class Solo12PronkV8PPORunnerCfg(Solo12PronkV7PPORunnerCfg):
    run_name = "pronk_v8_signed_bounded_exposure_seed42"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.reference_checkpoint = (
            "logs/rsl_rl/solo12_rsl_rl_pronk_runs/"
            "2026-10-11_11-55-16_pronk_v7_bidirectional_from_v6_seed42/model_2999.pt"
        )
        self.algorithm.reference_checkpoint_sha256 = "3073df723a451b1fd042853340f7a4f084f0506916ea0d8241533bde81a06be7"
        # The source tracks signed commands, but its slow-forward style is NOT
        # the target. Keep only a weak regularizer to allow that style to change.
        self.algorithm.reference_loss_coef = 0.01
