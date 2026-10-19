# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Continue the acquired v6 gait across signed speeds, steering and light physical DR."""

from isaaclab.utils import configclass

from .pronk_curriculum import PronkAcquisitionGate
from .solo12_pronk_v5 import Solo12PronkV5Env
from .solo12_pronk_v6 import Solo12PronkV6Env, Solo12PronkV6EnvCfg, Solo12PronkV6PPORunnerCfg


class PronkContinuationGate(PronkAcquisitionGate):
    """Expand steering and robustness together after nominal signed-speed acquisition."""

    @property
    def steering_fraction(self):
        return 0.2 + 0.8 * self.level / 20

    @property
    def robustness_fraction(self):
        return self.level / 20


@configclass
class Solo12PronkV7EnvCfg(Solo12PronkV6EnvCfg):
    pronk_bidirectional_commands = True
    command_lin_vel_x_range = (-0.65, 0.65)
    command_lin_vel_y_range = (-0.25, 0.25)
    command_ang_vel_z_range = (-0.25, 0.25)
    standing_env_prob = 0.08
    acquisition_window = 128

    def __post_init__(self):
        super().__post_init__()
        # Match the established light physical collection profile. The existing
        # Isaac-5.1 contract still excludes joint-friction randomization.
        if self.events is not None:
            self.events.physics_material.params["static_friction_range"] = (0.85, 1.35)
            self.events.physics_material.params["dynamic_friction_range"] = (0.80, 1.30)
            self.events.add_base_mass.params["mass_distribution_params"] = (0.95, 1.10)
            self.events.inertia_scale.params["inertia_distribution_params"] = (0.90, 1.10)
            self.events.base_com.params["com_range"] = {
                "x": (-0.008, 0.008), "y": (-0.006, 0.006), "z": (-0.010, 0.010),
            }


class Solo12PronkV7Env(Solo12PronkV6Env):
    cfg: Solo12PronkV7EnvCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._acquisition = PronkContinuationGate(
            window=cfg.acquisition_window, randomized_window=cfg.acquisition_randomized_window,
            survival=cfg.acquisition_survival, xy_rmse=cfg.acquisition_xy_rmse_mps,
            yaw_rmse=cfg.acquisition_yaw_rmse_rps, airtime=cfg.acquisition_airtime_fraction,
            cadence=cfg.acquisition_cadence_hz,
        )

    def _steering_fraction(self):
        if not self.cfg.acquisition_enabled:
            return 1.0
        return self._acquisition.steering_fraction if hasattr(self, "_acquisition") else 0.2

    def _resample_commands(self, env_ids, allow_opposite=True):
        # Bypass v6's affine compression around vx=.5. The entire signed vx
        # range is sampled even at acquisition level zero, including low speeds.
        Solo12PronkV5Env._resample_commands(self, env_ids, allow_opposite=allow_opposite)
        robust = self._curriculum_fraction()
        for column, maximum in ((1, self.cfg.command_lin_vel_y_range[1]),
                                (2, self.cfg.command_ang_vel_z_range[1])):
            parent_limit = self.cfg.steering_initial_limit + robust * (maximum-self.cfg.steering_initial_limit)
            self._commands[env_ids, column] *= self._steering_fraction() * maximum / parent_limit

    def _get_rewards(self):
        reward = super()._get_rewards()
        vx = self._commands[:, 0]
        self.extras["log"].update({
            "Coverage/pronk_reverse_fraction": (vx < -0.1).float().mean().item(),
            "Coverage/pronk_low_speed_fraction": ((vx.abs() >= 0.1) & (vx.abs() <= 0.25)).float().mean().item(),
            "Coverage/pronk_stop_fraction": (self._commands.abs().amax(dim=-1) == 0).float().mean().item(),
        })
        return reward

    def get_training_summary(self):
        summary = super().get_training_summary()
        summary.update({
            "longitudinal_range_from_first_episode": self.cfg.command_lin_vel_x_range,
            "final_lateral_range": self.cfg.command_lin_vel_y_range,
            "final_yaw_range": self.cfg.command_ang_vel_z_range,
            "physical_randomization_profile": "collection_light_no_joint_friction",
            "progression": "steering_and_robustness_together; vx never compressed",
        })
        return summary


@configclass
class Solo12PronkV7PPORunnerCfg(Solo12PronkV6PPORunnerCfg):
    run_name = "pronk_v7_bidirectional_from_v6_seed42"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.reference_checkpoint = (
            "logs/rsl_rl/solo12_rsl_rl_pronk_runs/"
            "2026-10-11_01-40-56_pronk_v6_measured_acquisition_seed42/model_2999.pt"
        )
        self.algorithm.reference_checkpoint_sha256 = "d73aa994934ade8511723d7c4e5159794fe718c192429eaf95682562a7b27baf"
        self.algorithm.learning_rate = 5.0e-5
        self.policy.init_noise_std = 0.10
