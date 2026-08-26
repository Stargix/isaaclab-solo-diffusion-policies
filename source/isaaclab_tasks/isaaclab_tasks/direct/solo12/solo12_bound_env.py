# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward bounding task for the Solo12.

This task intentionally lives beside, rather than inside, the generic velocity
environment. Existing walk/crouch/sprint experiments therefore retain their
exact observation and reward contracts.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from .bound_gait import advance_phase, bound_stance_targets, phase_observation
from .solo12_env import Solo12Env
from .solo12_env_cfg import BASE_OBSERVATION_SPACE, Solo12EnvCfg


BOUND_FOOT_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")


@configclass
class Solo12BoundEnvCfg(Solo12EnvCfg):
    """A phase-conditioned symmetric bound, trained only up to 1.5 m/s."""

    # The base observation plus sin/cos of the gait phase. Unlike a lone sine,
    # this clock is not ambiguous for a memoryless MLP.
    observation_space = BASE_OBSERVATION_SPACE + 2

    # Forward-only curriculum. Solo12Env normally makes every velocity stage
    # symmetric; Solo12BoundEnv overrides that setter to preserve these minima.
    command_lin_vel_x_range = (0.60, 1.00)
    command_lin_vel_y_range = (-0.10, 0.10)
    command_ang_vel_z_range = (-0.25, 0.25)
    max_velx_range_curriculum = (1.00, 1.25, 1.50)
    bound_min_velx_curriculum = (0.60, 0.70, 0.80)
    command_resampling_time_s = 5.0
    standing_env_prob = 0.0
    opposite_direction_cmd_prob = 0.0

    # A fixed 3 Hz clock is within the 2--4 Hz range demonstrated for dynamic
    # structured gaits in Walk These Ways. The 0.40 duty factor creates short
    # flight windows and makes the result visibly different from the trot prior.
    bound_frequency_hz = 3.0
    bound_duty_factor = 0.40
    bound_phase_transition_width = 0.025
    bound_foot_names = BOUND_FOOT_NAMES

    # Positive task reward. Yaw tracking is gated by forward tracking in the
    # environment so standing still cannot collect a useful yaw-only return.
    tracking_std = 0.35
    track_lin_vel_xy_reward_scale = 3.0
    track_ang_vel_z_reward_scale = 0.50

    # Bounded auxiliary costs used inside task_reward * exp(-temperature*cost).
    bound_aux_temperature = 0.50
    bound_contact_schedule_cost_scale = 1.00
    bound_swing_force_cost_scale = 0.15
    bound_stance_slip_cost_scale = 0.25
    bound_roll_cost_scale = 0.20
    bound_pitch_cost_scale = 0.05
    bound_height_cost_scale = 0.06
    bound_action_rate_cost_scale = 0.01
    bound_joint_torque_cost_scale = 2.0e-4
    bound_undesired_contact_cost_scale = 0.30
    bound_vertical_velocity_cost_scale = 0.01

    # Normalizers make every auxiliary term dimensionless and keep the reward
    # weights readable. Pitch has a corridor because pitching is part of a bound.
    bound_contact_threshold_n = 1.0
    bound_contact_softness_n = 0.5
    bound_swing_force_normalizer_n = 12.0
    bound_stance_slip_normalizer_mps = 1.5
    bound_roll_tolerance_rad = math.radians(10.0)
    bound_pitch_free_rad = math.radians(18.0)
    bound_pitch_tolerance_rad = math.radians(12.0)
    bound_height_tolerance_m = 0.05
    bound_cost_clip = 4.0

    # Height is deliberately soft: the clock/contact reward defines the gait,
    # while the body remains free to oscillate vertically and in pitch.
    base_z_desired = 0.25

    # Moderate transfer randomization from the start. Pushes are introduced only
    # after the speed curriculum through the inherited sequential curriculum.
    enable_observation_corruption = True
    base_lin_vel_noise = (-0.05, 0.05)
    base_ang_vel_noise = (-0.10, 0.10)
    projected_gravity_noise = (-0.025, 0.025)
    joint_pos_noise = (-0.005, 0.005)
    joint_vel_noise = (-0.50, 0.50)
    actuation_delay_range = (0, 1)
    forces_applied_to_base_curriculum = (0.0, 2.0)
    base_push_force_z_range = (0.0, 0.0)
    forces_curriculum_threshold_reward = 42.0
    tricky_terrain = False

    kp = 9.0
    kd = 0.3

    def __post_init__(self):
        super().__post_init__()
        if self.policy_model != "simple_mlp":
            raise ValueError("Solo12BoundEnvCfg currently supports only policy_model='simple_mlp'.")

        # Keep startup domain randomization realistic for the low-torque Solo12.
        if self.events is not None:
            self.events.physics_material.params["static_friction_range"] = (0.85, 1.30)
            self.events.physics_material.params["dynamic_friction_range"] = (0.80, 1.20)
            self.events.add_base_mass.params["mass_distribution_params"] = (0.95, 1.08)
            self.events.joint_friction.params["friction_distribution_params"] = (0.01, 0.15)
            self.events.inertia_scale.params["inertia_distribution_params"] = (0.90, 1.10)
            self.events.base_com.params["com_range"] = {
                "x": (-0.007, 0.007),
                "y": (-0.005, 0.005),
                "z": (-0.008, 0.008),
            }

        if "legs" in self.robot.actuators:
            self.robot.actuators["legs"].stiffness = self.kp
            self.robot.actuators["legs"].damping = self.kd


class Solo12BoundEnv(Solo12Env):
    """Solo12 direct environment with an explicit front/rear bound clock."""

    cfg: Solo12BoundEnvCfg

    def __init__(self, cfg: Solo12BoundEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._bound_phase = torch.zeros(self.num_envs, device=self.device)
        self._bound_feet_sensor_ids, sensor_names = self._contact_sensor.find_bodies(
            list(self.cfg.bound_foot_names), preserve_order=True
        )
        self._bound_feet_robot_ids, robot_names = self._robot.find_bodies(
            list(self.cfg.bound_foot_names), preserve_order=True
        )
        expected_names = list(self.cfg.bound_foot_names)
        if sensor_names != expected_names or robot_names != expected_names:
            raise RuntimeError(
                "Bound gait requires feet in FL, FR, RL, RR order; "
                f"contact sensor returned {sensor_names} and robot returned {robot_names}."
            )

        reward_terms = (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "quality_gate",
            "cost_contact_schedule",
            "cost_swing_force",
            "cost_stance_slip",
            "cost_roll",
            "cost_pitch_corridor",
            "cost_height",
            "cost_action_rate",
            "cost_joint_torque",
            "cost_undesired_contact",
            "cost_vertical_velocity",
        )
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_terms
        }

    def _set_max_velx_range_curriculum_level(self, level_idx: int):
        minima = tuple(float(value) for value in self.cfg.bound_min_velx_curriculum)
        if len(minima) != len(self._max_velx_range_curriculum_values):
            raise ValueError(
                "bound_min_velx_curriculum and max_velx_range_curriculum must have equal length, "
                f"got {len(minima)} and {len(self._max_velx_range_curriculum_values)}."
            )
        min_vel_x = minima[level_idx]
        max_vel_x = self._max_velx_range_curriculum_values[level_idx]
        if min_vel_x < 0.0 or min_vel_x > max_vel_x:
            raise ValueError(f"Invalid forward bound curriculum stage ({min_vel_x}, {max_vel_x}).")
        self._max_velx_range_curriculum_idx = level_idx
        self.cfg.command_lin_vel_x_range = (min_vel_x, max_vel_x)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._bound_phase = advance_phase(self._bound_phase, self.cfg.bound_frequency_hz, self.step_dt)
        super()._pre_physics_step(actions)

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        clock = phase_observation(self._bound_phase)
        observations["policy"] = torch.cat((observations["policy"], clock), dim=-1)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        velocity_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        yaw_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        velocity_score = torch.exp(-velocity_error / self.cfg.tracking_std**2)
        yaw_score = torch.exp(-yaw_error / self.cfg.tracking_std**2)

        desired_contact = bound_stance_targets(
            self._bound_phase,
            self.cfg.bound_duty_factor,
            self.cfg.bound_phase_transition_width,
        )
        force_history = self._contact_sensor.data.net_forces_w_history[:, :, self._bound_feet_sensor_ids, :]
        foot_force = torch.max(torch.linalg.vector_norm(force_history, dim=-1), dim=1).values
        contact_probability = torch.sigmoid(
            (foot_force - self.cfg.bound_contact_threshold_n) / self.cfg.bound_contact_softness_n
        )

        contact_schedule_cost = torch.mean(torch.square(contact_probability - desired_contact), dim=1)
        swing_weight = 1.0 - desired_contact
        swing_force = torch.clamp(
            foot_force / self.cfg.bound_swing_force_normalizer_n, min=0.0, max=1.0
        )
        swing_force_cost = torch.sum(swing_weight * torch.square(swing_force), dim=1) / torch.clamp(
            torch.sum(swing_weight, dim=1), min=1.0
        )

        foot_planar_speed = torch.linalg.vector_norm(
            self._robot.data.body_lin_vel_w[:, self._bound_feet_robot_ids, :2], dim=-1
        )
        stance_slip = torch.clamp(
            foot_planar_speed / self.cfg.bound_stance_slip_normalizer_mps, min=0.0, max=1.0
        )
        stance_slip_cost = torch.sum(desired_contact * torch.square(stance_slip), dim=1) / torch.clamp(
            torch.sum(desired_contact, dim=1), min=1.0
        )

        gravity = self._robot.data.projected_gravity_b
        roll_error = torch.abs(gravity[:, 1]) / math.sin(self.cfg.bound_roll_tolerance_rad)
        roll_cost = torch.clamp(torch.square(roll_error), max=self.cfg.bound_cost_clip)
        pitch_excess = torch.relu(
            torch.abs(gravity[:, 0]) - math.sin(self.cfg.bound_pitch_free_rad)
        ) / math.sin(self.cfg.bound_pitch_tolerance_rad)
        pitch_cost = torch.clamp(torch.square(pitch_excess), max=self.cfg.bound_cost_clip)

        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        height_error = (base_height - self.cfg.base_z_desired) / self.cfg.bound_height_tolerance_m
        height_cost = torch.clamp(torch.square(height_error), max=self.cfg.bound_cost_clip)
        action_rate_cost = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_torque_cost = torch.sum(
            torch.square(self._robot.data.applied_torque[:, self._joint_ids]), dim=1
        )
        undesired_contact_cost = self._compute_contact_count(
            self._thigh_body_ids, self.cfg.undesired_contact_threshold
        )
        vertical_velocity_cost = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        weighted_costs = {
            "cost_contact_schedule": contact_schedule_cost * self.cfg.bound_contact_schedule_cost_scale,
            "cost_swing_force": swing_force_cost * self.cfg.bound_swing_force_cost_scale,
            "cost_stance_slip": stance_slip_cost * self.cfg.bound_stance_slip_cost_scale,
            "cost_roll": roll_cost * self.cfg.bound_roll_cost_scale,
            "cost_pitch_corridor": pitch_cost * self.cfg.bound_pitch_cost_scale,
            "cost_height": height_cost * self.cfg.bound_height_cost_scale,
            "cost_action_rate": action_rate_cost * self.cfg.bound_action_rate_cost_scale,
            "cost_joint_torque": joint_torque_cost * self.cfg.bound_joint_torque_cost_scale,
            "cost_undesired_contact": undesired_contact_cost * self.cfg.bound_undesired_contact_cost_scale,
            "cost_vertical_velocity": vertical_velocity_cost * self.cfg.bound_vertical_velocity_cost_scale,
        }
        auxiliary_cost = torch.sum(torch.stack(tuple(weighted_costs.values())), dim=0)
        quality_gate = torch.exp(-self.cfg.bound_aux_temperature * auxiliary_cost)

        # Gating yaw by velocity tracking removes the otherwise attractive
        # "stand still and collect yaw reward" solution.
        linear_term = velocity_score * self.cfg.track_lin_vel_xy_reward_scale
        yaw_term = velocity_score * yaw_score * self.cfg.track_ang_vel_z_reward_scale
        reward = (linear_term + yaw_term) * quality_gate * self.step_dt

        logged_terms = {
            "track_lin_vel_xy_exp": linear_term * self.step_dt,
            "track_ang_vel_z_exp": yaw_term * self.step_dt,
            "quality_gate": quality_gate * self.step_dt,
            **{name: value * self.step_dt for name, value in weighted_costs.items()},
        }
        step_log = {f"RewardsPerStep/{key}": torch.mean(value).item() for key, value in logged_terms.items()}
        step_log["RewardsPerStep/cmd_tracking"] = (
            step_log["RewardsPerStep/track_lin_vel_xy_exp"]
            + step_log["RewardsPerStep/track_ang_vel_z_exp"]
        )
        step_log["RewardsPerStep/total"] = torch.mean(reward).item()

        actual_contact = foot_force > self.cfg.bound_contact_threshold_n
        step_log.update(
            {
                "Metrics/velocity_tracking_score": torch.mean(velocity_score).item(),
                "Metrics/yaw_tracking_score": torch.mean(yaw_score).item(),
                "Metrics/bound_contact_score": torch.mean(1.0 - contact_schedule_cost).item(),
                "Metrics/front_pair_sync": torch.mean(
                    (actual_contact[:, 0] == actual_contact[:, 1]).float()
                ).item(),
                "Metrics/rear_pair_sync": torch.mean(
                    (actual_contact[:, 2] == actual_contact[:, 3]).float()
                ).item(),
                "Metrics/front_rear_opposition": torch.mean(
                    torch.abs(actual_contact[:, :2].float().mean(dim=1) - actual_contact[:, 2:].float().mean(dim=1))
                ).item(),
                "Metrics/base_height_m": torch.mean(base_height).item(),
                "Metrics/foot_planar_speed_mps": torch.mean(foot_planar_speed).item(),
                "Metrics/foot_force_mean_n": torch.mean(foot_force).item(),
                "Metrics/foot_force_max_n": torch.max(foot_force).item(),
                "Metrics/auxiliary_cost": torch.mean(auxiliary_cost).item(),
            }
        )
        self.extras["log"] = step_log

        for key, value in logged_terms.items():
            self._episode_sums[key] += value
        self._episode_reward_sums += reward
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Random phases prevent the policy from overfitting the first stride to a
        # single clock state while keeping the reset pose unchanged.
        self._bound_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
