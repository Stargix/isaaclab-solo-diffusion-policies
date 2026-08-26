# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict straight-line symmetric-bound task for Solo12.

Version 1 remains unchanged for reproducibility.  This environment keeps the
same 50-D observation and 12-D action contracts, but makes straight tracking
and the requested contact topology part of the task reward itself.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from .bound_gait import (
    bound_pair_desynchronization,
    bound_stance_targets,
    diagonal_contact_probability,
)
from .solo12_bound_env import Solo12BoundEnv, Solo12BoundEnvCfg


@configclass
class Solo12BoundV2EnvCfg(Solo12BoundEnvCfg):
    """Forward-only bound with explicit straightness and pair synchronization."""

    # The expert is intentionally one-dimensional.  Turning and lateral motion
    # remain the responsibility of the walk expert in the downstream mixture.
    command_lin_vel_y_range = (0.0, 0.0)
    command_ang_vel_z_range = (0.0, 0.0)

    # Independent scales avoid the weak shared 0.35 tolerance of v1.  The
    # forward tolerance still leaves a learnable signal when initializing from
    # the stable v1 checkpoint, while lateral/yaw errors are much less tolerant.
    bound_forward_tracking_std_mps = 0.30
    bound_lateral_tracking_std_mps = 0.08
    bound_yaw_rate_tracking_std_rps = 0.12
    bound_heading_tracking_std_rad = math.radians(8.0)

    # Gait topology is a primary gate, separate from physical-quality costs.
    # Schedule alignment creates front/rear alternation; XOR and diagonal terms
    # specifically reject left/right splitting and a diagonal trot.
    bound_gait_temperature = 0.80
    bound_contact_schedule_cost_scale = 2.00
    bound_pair_desync_cost_scale = 1.50
    bound_diagonal_contact_cost_scale = 1.00
    bound_pair_height_cost_scale = 0.35
    bound_pair_vertical_velocity_cost_scale = 0.20
    bound_pair_height_tolerance_m = 0.025
    bound_pair_vertical_velocity_tolerance_mps = 0.40

    # Current forces preserve phase information.  The v1 history maximum could
    # smear a touchdown across several control steps and reward the wrong phase.
    bound_contact_softness_n = 0.25

    # Stability remains a multiplicative gate, but no longer hides gait errors
    # inside the same aggregate.
    bound_stability_temperature = 0.50

    # The best possible per-step reward is unchanged (3.5 * dt), making v1/v2
    # learning curves comparable in scale.
    track_lin_vel_xy_reward_scale = 3.0
    track_ang_vel_z_reward_scale = 0.50

    def __post_init__(self):
        super().__post_init__()
        if self.command_lin_vel_y_range != (0.0, 0.0) or self.command_ang_vel_z_range != (0.0, 0.0):
            raise ValueError("solo12-bound-v2 is a straight-only expert: vy and wz ranges must remain zero.")


class Solo12BoundV2Env(Solo12BoundEnv):
    """Phase-conditioned straight bound with measurable contact topology."""

    cfg: Solo12BoundV2EnvCfg

    def __init__(self, cfg: Solo12BoundV2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.command_lin_vel_y_range != (0.0, 0.0) or self.cfg.command_ang_vel_z_range != (0.0, 0.0):
            raise ValueError("solo12-bound-v2 requires fixed zero vy and wz command ranges.")

        reward_terms = (
            # These two names are required by the inherited reset logger.
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "gait_gate",
            "stability_gate",
            "cost_contact_schedule",
            "cost_pair_desync",
            "cost_diagonal_contact",
            "cost_pair_height",
            "cost_pair_vertical_velocity",
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

    @staticmethod
    def _squared_normalized_cost(error: torch.Tensor, tolerance: float, clip: float) -> torch.Tensor:
        if tolerance <= 0.0:
            raise ValueError(f"Reward tolerance must be positive, got {tolerance}.")
        return torch.clamp(torch.square(error / tolerance), max=clip)

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        forward_error = self._commands[:, 0] - root_lin_vel_b[:, 0]
        lateral_error = self._commands[:, 1] - root_lin_vel_b[:, 1]
        yaw_rate_error = self._commands[:, 2] - root_ang_vel_b[:, 2]
        forward_score = torch.exp(
            -torch.square(forward_error / self.cfg.bound_forward_tracking_std_mps)
        )
        lateral_score = torch.exp(
            -torch.square(lateral_error / self.cfg.bound_lateral_tracking_std_mps)
        )
        yaw_rate_score = torch.exp(
            -torch.square(yaw_rate_error / self.cfg.bound_yaw_rate_tracking_std_rps)
        )

        # Body-frame vy=0 and wz=0 alone do not prevent a persistent heading
        # offset.  Penalize heading relative to the configured reset direction.
        quat_w = self._robot.data.root_quat_w
        heading = torch.atan2(
            2.0 * (quat_w[:, 0] * quat_w[:, 3] + quat_w[:, 1] * quat_w[:, 2]),
            1.0 - 2.0 * (torch.square(quat_w[:, 2]) + torch.square(quat_w[:, 3])),
        )
        heading_delta = heading - float(self.cfg.reset_yaw)
        heading_error = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))
        heading_score = torch.exp(
            -torch.square(heading_error / self.cfg.bound_heading_tracking_std_rad)
        )
        command_score = forward_score * lateral_score * yaw_rate_score * heading_score

        desired_contact = bound_stance_targets(
            self._bound_phase,
            self.cfg.bound_duty_factor,
            self.cfg.bound_phase_transition_width,
        )
        current_force = self._contact_sensor.data.net_forces_w[:, self._bound_feet_sensor_ids, :]
        foot_force = torch.linalg.vector_norm(current_force, dim=-1)
        contact_probability = torch.sigmoid(
            (foot_force - self.cfg.bound_contact_threshold_n) / self.cfg.bound_contact_softness_n
        )

        contact_schedule_cost = torch.mean(torch.square(contact_probability - desired_contact), dim=1)
        pair_desync_cost = bound_pair_desynchronization(contact_probability)
        diagonal_cost = diagonal_contact_probability(contact_probability)

        foot_height = self._robot.data.body_pos_w[:, self._bound_feet_robot_ids, 2]
        pair_height_error = torch.stack(
            (foot_height[:, 0] - foot_height[:, 1], foot_height[:, 2] - foot_height[:, 3]), dim=1
        )
        pair_height_cost = torch.mean(
            self._squared_normalized_cost(
                pair_height_error,
                self.cfg.bound_pair_height_tolerance_m,
                self.cfg.bound_cost_clip,
            ),
            dim=1,
        )

        foot_vertical_velocity = self._robot.data.body_lin_vel_w[:, self._bound_feet_robot_ids, 2]
        pair_vertical_velocity_error = torch.stack(
            (
                foot_vertical_velocity[:, 0] - foot_vertical_velocity[:, 1],
                foot_vertical_velocity[:, 2] - foot_vertical_velocity[:, 3],
            ),
            dim=1,
        )
        pair_vertical_velocity_cost = torch.mean(
            self._squared_normalized_cost(
                pair_vertical_velocity_error,
                self.cfg.bound_pair_vertical_velocity_tolerance_mps,
                self.cfg.bound_cost_clip,
            ),
            dim=1,
        )

        weighted_gait_costs = {
            "cost_contact_schedule": contact_schedule_cost * self.cfg.bound_contact_schedule_cost_scale,
            "cost_pair_desync": pair_desync_cost * self.cfg.bound_pair_desync_cost_scale,
            "cost_diagonal_contact": diagonal_cost * self.cfg.bound_diagonal_contact_cost_scale,
            "cost_pair_height": pair_height_cost * self.cfg.bound_pair_height_cost_scale,
            "cost_pair_vertical_velocity": (
                pair_vertical_velocity_cost * self.cfg.bound_pair_vertical_velocity_cost_scale
            ),
        }
        gait_cost = torch.sum(torch.stack(tuple(weighted_gait_costs.values())), dim=0)
        gait_gate = torch.exp(-self.cfg.bound_gait_temperature * gait_cost)

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
        height_cost = self._squared_normalized_cost(
            base_height - self.cfg.base_z_desired,
            self.cfg.bound_height_tolerance_m,
            self.cfg.bound_cost_clip,
        )
        action_rate_cost = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_torque_cost = torch.sum(
            torch.square(self._robot.data.applied_torque[:, self._joint_ids]), dim=1
        )
        undesired_contact_cost = self._compute_contact_count(
            self._thigh_body_ids, self.cfg.undesired_contact_threshold
        )
        vertical_velocity_cost = torch.square(root_lin_vel_b[:, 2])

        weighted_stability_costs = {
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
        stability_cost = torch.sum(torch.stack(tuple(weighted_stability_costs.values())), dim=0)
        stability_gate = torch.exp(-self.cfg.bound_stability_temperature * stability_cost)

        linear_term = command_score * self.cfg.track_lin_vel_xy_reward_scale
        yaw_term = command_score * self.cfg.track_ang_vel_z_reward_scale
        reward = (linear_term + yaw_term) * gait_gate * stability_gate * self.step_dt

        logged_terms = {
            "track_lin_vel_xy_exp": linear_term * self.step_dt,
            "track_ang_vel_z_exp": yaw_term * self.step_dt,
            "gait_gate": gait_gate * self.step_dt,
            "stability_gate": stability_gate * self.step_dt,
            **{name: value * self.step_dt for name, value in weighted_gait_costs.items()},
            **{name: value * self.step_dt for name, value in weighted_stability_costs.items()},
        }
        step_log = {f"RewardsPerStep/{key}": torch.mean(value).item() for key, value in logged_terms.items()}
        step_log["RewardsPerStep/cmd_tracking"] = (
            step_log["RewardsPerStep/track_lin_vel_xy_exp"]
            + step_log["RewardsPerStep/track_ang_vel_z_exp"]
        )
        step_log["RewardsPerStep/total"] = torch.mean(reward).item()

        actual_contact = foot_force > self.cfg.bound_contact_threshold_n
        desired_binary = desired_contact > 0.5
        exact_pattern = torch.all(actual_contact == desired_binary, dim=1)
        front_same = actual_contact[:, 0] == actual_contact[:, 1]
        rear_same = actual_contact[:, 2] == actual_contact[:, 3]
        front_state = torch.any(actual_contact[:, :2], dim=1)
        rear_state = torch.any(actual_contact[:, 2:], dim=1)
        bound_pair_pattern = front_same & rear_same & (front_state != rear_state)
        diagonal_pattern = (
            (actual_contact[:, 0] & actual_contact[:, 3] & ~actual_contact[:, 1] & ~actual_contact[:, 2])
            | (actual_contact[:, 1] & actual_contact[:, 2] & ~actual_contact[:, 0] & ~actual_contact[:, 3])
        )
        front_intersection = actual_contact[:, 0] & actual_contact[:, 1]
        front_union = actual_contact[:, 0] | actual_contact[:, 1]
        rear_intersection = actual_contact[:, 2] & actual_contact[:, 3]
        rear_union = actual_contact[:, 2] | actual_contact[:, 3]
        # Ratio of aggregate stance intersections to unions. Flight is excluded
        # without counting two airborne feet as a successful synchronized stance.
        front_jaccard = torch.sum(front_intersection.float()) / torch.clamp(
            torch.sum(front_union.float()), min=1.0
        )
        rear_jaccard = torch.sum(rear_intersection.float()) / torch.clamp(
            torch.sum(rear_union.float()), min=1.0
        )

        step_log.update(
            {
                "Metrics/velocity_tracking_score": torch.mean(command_score).item(),
                "Metrics/forward_tracking_score": torch.mean(forward_score).item(),
                "Metrics/lateral_tracking_score": torch.mean(lateral_score).item(),
                "Metrics/yaw_rate_tracking_score": torch.mean(yaw_rate_score).item(),
                "Metrics/heading_tracking_score": torch.mean(heading_score).item(),
                "Metrics/forward_speed_mps": torch.mean(root_lin_vel_b[:, 0]).item(),
                "Metrics/lateral_speed_abs_mps": torch.mean(torch.abs(root_lin_vel_b[:, 1])).item(),
                "Metrics/yaw_rate_abs_rps": torch.mean(torch.abs(root_ang_vel_b[:, 2])).item(),
                "Metrics/heading_error_abs_rad": torch.mean(torch.abs(heading_error)).item(),
                "Metrics/bound_contact_score": torch.mean(1.0 - contact_schedule_cost).item(),
                "Metrics/exact_desired_contact_fraction": torch.mean(exact_pattern.float()).item(),
                "Metrics/bound_pair_pattern_fraction": torch.mean(bound_pair_pattern.float()).item(),
                "Metrics/diagonal_pattern_fraction": torch.mean(diagonal_pattern.float()).item(),
                "Metrics/front_pair_sync": front_jaccard.item(),
                "Metrics/rear_pair_sync": rear_jaccard.item(),
                "Metrics/pair_desynchronization": torch.mean(pair_desync_cost).item(),
                "Metrics/front_rear_opposition": torch.mean((front_state != rear_state).float()).item(),
                "Metrics/base_height_m": torch.mean(base_height).item(),
                "Metrics/foot_planar_speed_mps": torch.mean(foot_planar_speed).item(),
                "Metrics/foot_force_mean_n": torch.mean(foot_force).item(),
                "Metrics/foot_force_max_n": torch.max(foot_force).item(),
                "Metrics/gait_cost": torch.mean(gait_cost).item(),
                "Metrics/stability_cost": torch.mean(stability_cost).item(),
            }
        )
        self.extras["log"] = step_log

        for key, value in logged_terms.items():
            self._episode_sums[key] += value
        self._episode_reward_sums += reward
        return reward
