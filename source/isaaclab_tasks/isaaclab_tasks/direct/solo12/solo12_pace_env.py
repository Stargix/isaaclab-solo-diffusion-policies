# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clock-free ipsilateral pace expert, warm-started from the standard walk."""

from __future__ import annotations

import torch
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import Solo12PPORunnerWithSymmetryCfg
from .pace_gait import pace_gait_reward_components
from .solo12_env import Solo12Env
from .solo12_env_cfg import Solo12EnvCfg


PACE_FOOT_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
PACE_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0,
    curriculum=False,
    size=(200.0, 200.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    color_scheme="none",
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class Solo12PaceEnvCfg(Solo12EnvCfg):
    """Velocity expert config with small steering commands and no DR curriculum."""

    command_lin_vel_x_range = (-1.0, 1.0)
    command_lin_vel_y_range = (-0.15, 0.15)
    command_ang_vel_z_range = (-0.35, 0.35)
    command_resampling_time_s = 3.0
    standing_env_prob = 0.10
    opposite_direction_cmd_prob = 0.0
    max_velx_range_curriculum = ()
    forces_applied_to_base_curriculum = (0.0,)
    base_push_force_xy_range = (0.0, 0.0)
    base_push_force_z_range = (0.0, 0.0)
    enable_observation_corruption = False
    events = None
    actuation_delay_range = (0, 0)
    flexed_initial_joint_pos_noise_range = (-0.02, 0.02)
    reset_base_lin_vel_range = (-0.05, 0.05)
    reset_base_ang_vel_range = (-0.05, 0.05)
    tricky_terrain = False

    tracking_std = 0.30
    track_lin_vel_xy_reward_scale = 4.0
    track_ang_vel_z_reward_scale = 0.65
    base_z_desired = 0.2932
    base_height_exp_scale = 1.0 / (0.035**2)
    track_base_height_reward_scale = 0.50
    lin_vel_z_reward_scale = -0.05
    ang_vel_xy_reward_scale = -0.05
    joint_torque_reward_scale = -0.10e-3
    action_rate_reward_scale = -0.02
    undesired_contact_reward_scale = -2.25
    base_tilt_penalty_reward_scale = -0.25
    foot_contact_reward_scale = -0.5e-3
    joint_accel_reward_scale = 0.0
    feet_air_time_reward_scale = 0.0
    force_transmited_through_joints_reward_scale = 0.0

    pace_reward_scale = 1.0
    pace_lateral_reward_scale = 0.65
    pace_lateral_tracking_std_mps = 0.12
    pace_timing_std_s2 = 0.10
    pace_timing_max_error_s = 0.20
    pace_contact_threshold_n = 1.0
    pace_contact_softness_n = 0.25
    pace_max_mode_time_s = 0.50
    pace_min_moving_speed_mps = 0.10
    pace_foot_names = PACE_FOOT_NAMES

    def __post_init__(self):
        super().__post_init__()
        if self.policy_model != "simple_mlp" or self.observation_space != 48:
            raise ValueError("Pace must retain the 48-D simple-MLP walk checkpoint contract.")
        if tuple(self.command_lin_vel_y_range) != (-0.15, 0.15):
            raise ValueError("Pace symmetry assumes vy range [-0.15, 0.15].")
        if tuple(self.command_ang_vel_z_range) != (-0.35, 0.35):
            raise ValueError("Pace symmetry assumes yaw-rate range [-0.35, 0.35].")
        positive = {
            "tracking_std": self.tracking_std,
            "pace_lateral_tracking_std_mps": self.pace_lateral_tracking_std_mps,
            "pace_timing_std_s2": self.pace_timing_std_s2,
            "pace_timing_max_error_s": self.pace_timing_max_error_s,
            "pace_contact_softness_n": self.pace_contact_softness_n,
            "pace_max_mode_time_s": self.pace_max_mode_time_s,
        }
        if any(value <= 0.0 for value in positive.values()):
            raise ValueError(f"Pace reward scales/thresholds must be positive: {positive}")
        if self.pace_reward_scale < 0.0 or self.pace_lateral_reward_scale < 0.0:
            raise ValueError("Pace reward scales cannot be negative.")
        # Build a flat trimesh locally instead of referencing Isaac Nucleus' USD
        # ground asset; the latter is unavailable in some cluster/offline setups.
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = PACE_FLAT_TERRAIN_CFG.copy()
        self.terrain.use_terrain_origins = False
        self.terrain.env_spacing = self.scene.env_spacing
        if "legs" in self.robot.actuators:
            self.robot.actuators["legs"].stiffness = 9.0
            self.robot.actuators["legs"].damping = 0.2


@configclass
class Solo12PacePPORunnerCfg(Solo12PPORunnerWithSymmetryCfg):
    """PPO settings for acquiring pace while preserving the walk actor contract."""

    num_steps_per_env = 32
    max_iterations = 2500
    save_interval = 50
    experiment_name = "solo12_rsl_rl_pace_runs"
    run_name = "pace_v1_seed42"

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.35
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.01
        self.algorithm.entropy_coef = 0.002


class Solo12PaceEnv(Solo12Env):
    """Base velocity task plus clock-free ipsilateral timing/topology rewards."""

    cfg: Solo12PaceEnvCfg

    def __init__(self, cfg: Solo12PaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        foot_ids, resolved_names = self._contact_sensor.find_bodies(
            list(self.cfg.pace_foot_names), preserve_order=True
        )
        if resolved_names != list(self.cfg.pace_foot_names):
            raise RuntimeError(f"Pace needs feet in FL, FR, RL, RR order; sensor returned {resolved_names}.")
        self._pace_foot_ids = foot_ids
        for key in ("pace_gait", "pace_lateral_tracking"):
            if key in self._episode_sums:
                raise RuntimeError(f"Unexpected duplicate episode reward accumulator: {key}")
            self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        reward_base = super()._get_rewards()
        command_xy = self._commands[:, :2]
        velocity_xy = self._robot.data.root_lin_vel_b[:, :2]
        error_xy = command_xy - velocity_xy
        tracking_score_xy = torch.exp(-torch.sum(error_xy.square(), dim=-1) / self.cfg.tracking_std**2)
        lateral_error = self._commands[:, 1] - self._robot.data.root_lin_vel_b[:, 1]
        lateral_score = torch.exp(-lateral_error.square() / self.cfg.pace_lateral_tracking_std_mps**2)

        components = pace_gait_reward_components(
            self._contact_sensor.data.current_air_time[:, self._pace_foot_ids],
            self._contact_sensor.data.current_contact_time[:, self._pace_foot_ids],
            self._contact_sensor.data.net_forces_w[:, self._pace_foot_ids, :],
            command_xy,
            tracking_score_xy,
            timing_std_s2=self.cfg.pace_timing_std_s2,
            timing_max_error_s=self.cfg.pace_timing_max_error_s,
            contact_threshold_n=self.cfg.pace_contact_threshold_n,
            contact_softness_n=self.cfg.pace_contact_softness_n,
            max_mode_time_s=self.cfg.pace_max_mode_time_s,
            min_moving_speed_mps=self.cfg.pace_min_moving_speed_mps,
        )
        gait_reward = components["gait"] * self.cfg.pace_reward_scale * self.step_dt
        lateral_reward = lateral_score * self.cfg.pace_lateral_reward_scale * self.step_dt
        reward_extra = gait_reward + lateral_reward

        self._episode_sums["pace_gait"] += gait_reward
        self._episode_sums["pace_lateral_tracking"] += lateral_reward
        self._episode_reward_sums += reward_extra
        log = self.extras.setdefault("log", {})
        log.update(
            {
                "RewardsPerStep/pace_gait": torch.mean(gait_reward).item(),
                "RewardsPerStep/pace_lateral_tracking": torch.mean(lateral_reward).item(),
                "RewardsPerStep/total": torch.mean(reward_base + reward_extra).item(),
                "Metrics/pace_timing_score": torch.mean(components["timing"]).item(),
                "Metrics/pace_contact_topology_score": torch.mean(components["topology"]).item(),
                "Metrics/pace_mode_guard_fraction": torch.mean(components["mode_ok"]).item(),
                "Metrics/pace_forward_error_abs_mps": torch.mean(torch.abs(error_xy[:, 0])).item(),
                "Metrics/pace_lateral_error_abs_mps": torch.mean(torch.abs(lateral_error)).item(),
                "Metrics/pace_height_m": torch.mean(
                    self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
                ).item(),
            }
        )
        return reward_base + reward_extra
