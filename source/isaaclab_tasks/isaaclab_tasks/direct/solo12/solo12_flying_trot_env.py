# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-first flying-trot task warm-started from the standard Solo12 walk."""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import Solo12PPORunnerWithSymmetryCfg
from .solo12_env import Solo12Env
from .solo12_env_cfg import Solo12EnvCfg


FLYING_TROT_FOOT_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
LOCAL_GROUND_USD_PATH = (
    Path(__file__).parents[4]
    / "borinotIsaacLab/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd"
)


@configclass
class Solo12FlyingTrotEnvCfg(Solo12EnvCfg):
    """Fast diagonal trot with short aerial phases and modest steering authority."""

    # The first stage overlaps the proven walk checkpoint. Later stages shift,
    # rather than merely widen, the distribution towards the fast skill.
    command_lin_vel_x_range = (0.75, 1.00)
    max_velx_range_curriculum = (1.00, 1.25, 1.50)
    flying_trot_min_velx_curriculum = (0.75, 0.85, 1.00)
    command_lin_vel_y_range = (-0.15, 0.15)
    command_ang_vel_z_range = (-0.35, 0.35)
    command_resampling_time_s = 5.0
    standing_env_prob = 0.0
    opposite_direction_cmd_prob = 0.0

    # Forward speed is the task. Lateral/yaw tracking remain independent so an
    # error in a secondary command cannot erase the forward learning signal.
    flying_trot_forward_tracking_std_mps = 0.22
    flying_trot_lateral_tracking_std_mps = 0.12
    flying_trot_yaw_tracking_std_rps = 0.22
    flying_trot_forward_reward_scale = 4.0
    flying_trot_lateral_reward_scale = 0.65
    track_ang_vel_z_reward_scale = 0.65

    # A small topology term protects the diagonal manifold of walk_final. The
    # touchdown-only air-time term promotes longer swing without rewarding a
    # robot for remaining airborne or repeatedly jumping in place.
    flying_trot_diagonal_reward_scale = 0.30
    feet_air_time_reward_scale = 1.50
    feet_air_time_threshold = 0.10
    flying_trot_air_time_max_s = 0.25
    flying_trot_contact_threshold_n = 1.0
    flying_trot_contact_softness_n = 0.5
    flying_trot_foot_names = FLYING_TROT_FOOT_NAMES

    # Soft physical regularization. It refines a moving solution but is too
    # small to make standing preferable to tracking the commanded velocity.
    lin_vel_z_reward_scale = -0.05
    ang_vel_xy_reward_scale = -0.05
    joint_torque_reward_scale = -0.10e-3
    action_rate_reward_scale = -0.02
    undesired_contact_reward_scale = -2.25
    base_tilt_penalty_reward_scale = -0.25
    foot_contact_reward_scale = -0.5e-3
    track_base_height_reward_scale = 0.15
    base_z_desired = 0.28

    # Acquire the new speed regime before robustness fine-tuning. This matches
    # the proven walk prior and prevents exploration plus domain randomization
    # from destroying it during the first PPO updates.
    enable_observation_corruption = False
    events = None
    actuation_delay_range = (0, 0)
    flexed_initial_joint_pos_noise_range = (-0.02, 0.02)
    reset_base_lin_vel_range = (-0.05, 0.05)
    reset_base_ang_vel_range = (-0.05, 0.05)
    forces_applied_to_base_curriculum = (0.0,)
    base_push_force_xy_range = (0.0, 0.0)
    base_push_force_z_range = (0.0, 0.0)
    tricky_terrain = False

    # At most one curriculum transition per full episode is already enforced
    # by Solo12Env. This threshold requires strong tracking before advancing.
    forces_curriculum_threshold_reward = 72.0
    forces_curriculum_smoothing = 0.10

    kp = 9.0
    kd = 0.3

    def __post_init__(self):
        super().__post_init__()
        if self.policy_model != "simple_mlp" or self.observation_space != 48:
            raise ValueError("Flying trot must retain the 48-D simple-MLP walk checkpoint contract.")
        if len(self.flying_trot_min_velx_curriculum) != len(self.max_velx_range_curriculum):
            raise ValueError("Flying-trot curriculum minima and maxima must have equal length.")
        if self.command_lin_vel_y_range != (-0.15, 0.15):
            raise ValueError("Flying-trot mirror training assumes the symmetric vy range [-0.15, 0.15].")
        if self.command_ang_vel_z_range != (-0.35, 0.35):
            raise ValueError("Flying-trot mirror training assumes the symmetric wz range [-0.35, 0.35].")
        if not LOCAL_GROUND_USD_PATH.is_file():
            raise FileNotFoundError(f"Tracked local ground USD is missing: {LOCAL_GROUND_USD_PATH}")
        self.terrain.terrain_type = "usd"
        self.terrain.usd_path = str(LOCAL_GROUND_USD_PATH)
        if "legs" in self.robot.actuators:
            self.robot.actuators["legs"].stiffness = self.kp
            self.robot.actuators["legs"].damping = self.kd


@configclass
class Solo12FlyingTrotPPORunnerCfg(Solo12PPORunnerWithSymmetryCfg):
    """Conservative PPO fine-tuning with all four Solo12 morphology symmetries."""

    num_steps_per_env = 32
    max_iterations = 2500
    save_interval = 50
    experiment_name = "solo12_rsl_rl_flying_trot_runs"
    run_name = "solo12_flying_trot_1p5_v1"

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.15
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.008


class Solo12FlyingTrotEnv(Solo12Env):
    """Solo12 velocity task with a soft, clock-free flying-trot preference."""

    cfg: Solo12FlyingTrotEnvCfg

    def __init__(self, cfg: Solo12FlyingTrotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._flying_trot_feet_ids, names = self._contact_sensor.find_bodies(
            list(self.cfg.flying_trot_foot_names), preserve_order=True
        )
        if names != list(self.cfg.flying_trot_foot_names):
            raise RuntimeError(
                "Flying trot requires feet in FL, FR, RL, RR order; "
                f"contact sensor returned {names}."
            )

        reward_terms = (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "diagonal_trot",
            "feet_air_time",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "dof_torques_l2",
            "action_rate_l2",
            "undesired_contacts",
            "flat_orientation_l2",
            "track_base_height_exp",
            "foot_contact",
        )
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in reward_terms
        }

    def _set_max_velx_range_curriculum_level(self, level_idx: int):
        minima = tuple(float(value) for value in self.cfg.flying_trot_min_velx_curriculum)
        if len(minima) != len(self._max_velx_range_curriculum_values):
            raise ValueError("Flying-trot curriculum minima and maxima must have equal length.")
        min_vel_x = minima[level_idx]
        max_vel_x = self._max_velx_range_curriculum_values[level_idx]
        if min_vel_x <= 0.0 or min_vel_x > max_vel_x:
            raise ValueError(f"Invalid flying-trot velocity stage ({min_vel_x}, {max_vel_x}).")
        self._max_velx_range_curriculum_idx = level_idx
        self.cfg.command_lin_vel_x_range = (min_vel_x, max_vel_x)

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        forward_error = self._commands[:, 0] - root_lin_vel_b[:, 0]
        lateral_error = self._commands[:, 1] - root_lin_vel_b[:, 1]
        yaw_error = self._commands[:, 2] - root_ang_vel_b[:, 2]
        forward_score = torch.exp(
            -torch.square(forward_error / self.cfg.flying_trot_forward_tracking_std_mps)
        )
        lateral_score = torch.exp(
            -torch.square(lateral_error / self.cfg.flying_trot_lateral_tracking_std_mps)
        )
        yaw_score = torch.exp(
            -torch.square(yaw_error / self.cfg.flying_trot_yaw_tracking_std_rps)
        )

        foot_force = torch.linalg.vector_norm(
            self._contact_sensor.data.net_forces_w[:, self._flying_trot_feet_ids, :], dim=-1
        )
        contact_probability = torch.sigmoid(
            (foot_force - self.cfg.flying_trot_contact_threshold_n)
            / self.cfg.flying_trot_contact_softness_n
        )
        diagonal_a = 0.5 * (contact_probability[:, 0] + contact_probability[:, 3])
        diagonal_b = 0.5 * (contact_probability[:, 1] + contact_probability[:, 2])
        diagonal_sync = 1.0 - 0.5 * (
            torch.abs(contact_probability[:, 0] - contact_probability[:, 3])
            + torch.abs(contact_probability[:, 1] - contact_probability[:, 2])
        )
        diagonal_opposition = torch.abs(diagonal_a - diagonal_b)
        diagonal_trot_score = torch.clamp(diagonal_sync * diagonal_opposition, min=0.0, max=1.0)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
            :, self._flying_trot_feet_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._flying_trot_feet_ids]
        touchdown_air_time = torch.clamp(
            last_air_time - self.cfg.feet_air_time_threshold,
            min=-self.cfg.feet_air_time_threshold,
            max=self.cfg.flying_trot_air_time_max_s,
        )
        feet_air_time = torch.sum(touchdown_air_time * first_contact, dim=1)

        z_vel_error = torch.square(root_lin_vel_b[:, 2])
        roll_pitch_rate_error = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(
            torch.square(self._robot.data.applied_torque[:, self._joint_ids]), dim=1
        )
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        base_height_error = torch.square(base_height - self.cfg.base_z_desired)
        undesired_contacts = self._compute_contact_count(
            self._thigh_body_ids, self.cfg.undesired_contact_threshold
        )
        foot_contact = self._compute_foot_contact_penalty()

        rewards = {
            "track_lin_vel_xy_exp": (
                forward_score * self.cfg.flying_trot_forward_reward_scale
                + lateral_score * self.cfg.flying_trot_lateral_reward_scale
            )
            * self.step_dt,
            "track_ang_vel_z_exp": (
                yaw_score * self.cfg.track_ang_vel_z_reward_scale * self.step_dt
            ),
            "diagonal_trot": (
                diagonal_trot_score * self.cfg.flying_trot_diagonal_reward_scale * self.step_dt
            ),
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "ang_vel_xy_l2": roll_pitch_rate_error * self.cfg.ang_vel_xy_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "undesired_contacts": (
                undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt
            ),
            "flat_orientation_l2": (
                flat_orientation * self.cfg.base_tilt_penalty_reward_scale * self.step_dt
            ),
            "track_base_height_exp": (
                torch.exp(-self.cfg.base_height_exp_scale * base_height_error)
                * self.cfg.track_base_height_reward_scale
                * self.step_dt
            ),
            "foot_contact": foot_contact * self.cfg.foot_contact_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(tuple(rewards.values())), dim=0)
        step_log = {
            f"RewardsPerStep/{key}": torch.mean(value).item() for key, value in rewards.items()
        }
        step_log.update(
            {
                "RewardsPerStep/cmd_tracking": (
                    step_log["RewardsPerStep/track_lin_vel_xy_exp"]
                    + step_log["RewardsPerStep/track_ang_vel_z_exp"]
                ),
                "RewardsPerStep/total": torch.mean(reward).item(),
                "Metrics/forward_speed_mps": torch.mean(root_lin_vel_b[:, 0]).item(),
                "Metrics/forward_speed_error_abs_mps": torch.mean(torch.abs(forward_error)).item(),
                "Metrics/lateral_speed_error_abs_mps": torch.mean(torch.abs(lateral_error)).item(),
                "Metrics/yaw_rate_error_abs_rps": torch.mean(torch.abs(yaw_error)).item(),
                "Metrics/diagonal_trot_score": torch.mean(diagonal_trot_score).item(),
                "Metrics/flight_fraction": torch.mean((foot_force < 1.0).all(dim=1).float()).item(),
                "Metrics/base_height_m": torch.mean(base_height).item(),
            }
        )
        self.extras["log"] = step_log
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_reward_sums += reward
        return reward
