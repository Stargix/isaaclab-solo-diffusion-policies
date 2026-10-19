# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Nominal gait acquisition before measured steering/robustness progression."""

import torch
from isaaclab.utils import configclass

from .pronk_curriculum import PronkAcquisitionGate
from .solo12_pronk_v5 import Solo12PronkV5Env, Solo12PronkV5EnvCfg, Solo12PronkV5PPORunnerCfg


@configclass
class Solo12PronkV6EnvCfg(Solo12PronkV5EnvCfg):
    acquisition_enabled = True
    acquisition_window = 256
    acquisition_randomized_window = 64
    acquisition_survival = 0.95
    acquisition_xy_rmse_mps = 0.12
    acquisition_yaw_rmse_rps = 0.15
    acquisition_airtime_fraction = 0.08
    acquisition_cadence_hz = 0.75
    acquisition_initial_speed_mps = 0.5

    noise_initial_fraction = 0.0
    push_initial_fraction = 0.0
    standing_env_prob = 0.0
    command_resampling_time_s = 5.0
    # Stabilize orientation and angular motion without imposing a constant
    # height or penalizing the vertical translation needed for the source gait.
    base_tilt_penalty_reward_scale = -2.0
    ang_vel_xy_reward_scale = -0.05
    # Keep the lower-flight-weight pilot recipe. A joint trial with weight 10
    # and exploration std .05 produced longer flights but worse survival.
    # It does not establish either parameter's isolated causal effect.
    pronk_cycle_reward_scale = 3.0

    def __post_init__(self):
        super().__post_init__()
        if not self.command_lin_vel_x_range[0] <= self.acquisition_initial_speed_mps <= self.command_lin_vel_x_range[1]:
            raise ValueError("Acquisition speed must lie within final forward bounds.")
        PronkAcquisitionGate(
            window=self.acquisition_window, randomized_window=self.acquisition_randomized_window,
            survival=self.acquisition_survival, xy_rmse=self.acquisition_xy_rmse_mps,
            yaw_rmse=self.acquisition_yaw_rmse_rps, airtime=self.acquisition_airtime_fraction,
            cadence=self.acquisition_cadence_hz,
        )
        self._set_event_mode()

    def _set_event_mode(self):
        if self.events is not None:
            for name in ("physics_material", "add_base_mass", "inertia_scale", "base_com"):
                term = getattr(self.events, name, None)
                if term is not None:
                    term.mode = "pronk_robust" if self.acquisition_enabled else "startup"

    def use_full_difficulty(self):
        """Call before environment construction for frozen evaluation/playback."""
        self.acquisition_enabled = False
        self.noise_initial_fraction = 1.0
        self.initial_max_delay_steps = self.actuation_delay_range[1]
        self.push_initial_fraction = 1.0
        self._set_event_mode()


class Solo12PronkV6Env(Solo12PronkV5Env):
    cfg: Solo12PronkV6EnvCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._acquisition = PronkAcquisitionGate(
            window=cfg.acquisition_window, randomized_window=cfg.acquisition_randomized_window,
            survival=cfg.acquisition_survival, xy_rmse=cfg.acquisition_xy_rmse_mps,
            yaw_rmse=cfg.acquisition_yaw_rmse_rps, airtime=cfg.acquisition_airtime_fraction,
            cadence=cfg.acquisition_cadence_hz,
        )
        self._episode_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._randomized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._randomization_order = (torch.randperm(self.num_envs, device=self.device)
                                     if cfg.acquisition_enabled else torch.arange(self.num_envs, device=self.device))
        self._acquisition_sums = torch.zeros((self.num_envs, 6), device=self.device)

    def _steering_fraction(self):
        if not self.cfg.acquisition_enabled:
            return 1.0
        return self._acquisition.steering_fraction if hasattr(self, "_acquisition") else 0.0

    def _curriculum_fraction(self):
        if not self.cfg.acquisition_enabled:
            return 1.0
        return self._acquisition.robustness_fraction if hasattr(self, "_acquisition") else 0.0

    def _resample_commands(self, env_ids, allow_opposite=True):
        # V4 samples from the final bounds, then scales steering using its
        # curriculum. Undo that scale and apply the independent steering stage.
        super()._resample_commands(env_ids, allow_opposite=allow_opposite)
        fraction = self._steering_fraction()
        robust = self._curriculum_fraction()
        for column, maximum in ((1, self.cfg.command_lin_vel_y_range[1]),
                                (2, self.cfg.command_ang_vel_z_range[1])):
            parent_limit = self.cfg.steering_initial_limit + robust * (maximum-self.cfg.steering_initial_limit)
            self._commands[env_ids, column] *= fraction * maximum / parent_limit
        initial = self.cfg.acquisition_initial_speed_mps
        self._commands[env_ids, 0] = initial + fraction * (self._commands[env_ids, 0]-initial)

    def _get_rewards(self):
        reward = super()._get_rewards()
        if hasattr(self, "_acquisition_sums"):
            active = self._acquisition_sums[:, 0]+1 > round(self.cfg.pronk_warmup_s/self.step_dt)
            xy = (self._robot.data.root_lin_vel_b[:, :2]-self._commands[:, :2]).square().sum(-1)
            yaw = (self._robot.data.root_ang_vel_b[:, 2]-self._commands[:, 2]).square()
            components = self._pronk_cycle_components
            values = torch.stack((torch.ones_like(xy), active.float(), xy*active, yaw*active,
                                  components["flight_duration"]*active,
                                  components["completed"].float()*active), dim=-1)
            self._acquisition_sums += values
            self.extras["log"].update({
                "Curriculum/pronk_acquisition_level": self._acquisition.level,
                "Curriculum/pronk_steering_fraction": self._steering_fraction(),
                "Curriculum/pronk_randomized_fraction": self._randomized.float().mean().item(),
            })
        return reward

    def get_training_summary(self):
        return {
            "acquisition_enabled": self.cfg.acquisition_enabled,
            "level": self._acquisition.level,
            "steering_fraction": self._steering_fraction(),
            "robustness_fraction": self._curriculum_fraction(),
            "randomized_environment_fraction": self._randomized.float().mean().item(),
            "full_envelope_reached": self._acquisition.level == 20,
            "recent_nominal_episodes": len(self._acquisition.records[0]),
            "recent_randomized_episodes": len(self._acquisition.records[1]),
            "metric_order": ["survival", "xy_mse", "yaw_mse", "confirmed_airtime", "confirmed_cadence"],
            "recent_nominal_metrics": self._acquisition.statistics(0),
            "recent_randomized_metrics": self._acquisition.statistics(1),
            "full_envelope_acquisition_gate_passed": self._acquisition.level == 20
                and self._acquisition._passes(1, self.cfg.acquisition_randomized_window),
            "frozen_evaluation_required": True,
        }

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        initialized = hasattr(self, "_acquisition_sums")
        if initialized and self.cfg.acquisition_enabled:
            sums = self._acquisition_sums[env_ids]
            seconds = sums[:, 0]*self.step_dt
            failed = self.reset_terminated[env_ids]
            timed_out = self.reset_time_outs[env_ids]
            # Runner may randomize the first episode clock. Its short timeouts
            # cannot count as successful full episodes; actual falls always count.
            qualifies = (sums[:, 0] > 0) & (failed | (timed_out & (seconds >= self.cfg.episode_length_s-0.1)))
            counts = sums[:, 1].clamp_min(1)
            records = torch.stack((~failed & timed_out, sums[:, 2]/counts, sums[:, 3]/counts,
                                   sums[:, 4]/(counts*self.step_dt), sums[:, 5]/(counts*self.step_dt)), dim=-1)
            self._acquisition.add(records[qualifies].cpu().tolist(),
                                  self._episode_level[env_ids][qualifies].cpu().tolist(),
                                  self._randomized[env_ids][qualifies].cpu().tolist())
            if self._acquisition.advance():
                print(f"[Pronk acquisition] passed level {self._acquisition.level-1}; "
                      f"steering={self._steering_fraction():.1f}, robustness={self._curriculum_fraction():.1f}", flush=True)

        fraction = self._curriculum_fraction()
        fields = ("flexed_initial_joint_pos_noise_range", "reset_base_lin_vel_range", "reset_base_ang_vel_range")
        originals = {name: getattr(self.cfg, name) for name in fields}
        # The reset method is synchronous; restore the full declared contract
        # immediately after this call, including on simulator exceptions.
        try:
            if self.cfg.acquisition_enabled:
                for name, bounds in originals.items():
                    setattr(self.cfg, name, tuple(value*fraction for value in bounds))
            super()._reset_idx(env_ids)
        finally:
            for name, bounds in originals.items():
                setattr(self.cfg, name, bounds)

        if initialized:
            self._acquisition_sums[env_ids] = 0
            self._episode_level[env_ids] = self._acquisition.level
            if self.cfg.acquisition_enabled and getattr(self, "event_manager", None) is not None:
                selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                selected[self._randomization_order[:round(fraction*self.num_envs)]] = True
                new_ids = env_ids[selected[env_ids] & ~self._randomized[env_ids]]
                if len(new_ids):
                    self.event_manager.apply("pronk_robust", env_ids=new_ids)
                    self._randomized[new_ids] = True


@configclass
class Solo12PronkV6PPORunnerCfg(Solo12PronkV5PPORunnerCfg):
    max_iterations = 3000
    run_name = "pronk_v6_measured_acquisition_seed42"

    def __post_init__(self):
        super().__post_init__()
        # The source supplies the initial motion, but is not a reliable teacher
        # for every local state. Retain a weak regularizer instead of forcing
        # the adapted actor to reproduce its unstable responses.
        self.algorithm.reference_loss_coef = 0.05
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.01
        self.policy.init_noise_std = 0.20
