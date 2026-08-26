# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dense-reward, mirror-regularized straight bound task for Solo12."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .solo12_bound_v2_env import Solo12BoundV2Env, Solo12BoundV2EnvCfg


@configclass
class Solo12BoundV3EnvCfg(Solo12BoundV2EnvCfg):
    """Fast-only curriculum and dense directional tracking for a clean bound."""

    # Bounding is the fast expert. Slow recovery remains covered by walk, while
    # the small overlap around 1.0 m/s gives the downstream policy a hand-off.
    command_lin_vel_x_range = (0.90, 1.15)
    max_velx_range_curriculum = (1.15, 1.30, 1.50)
    bound_min_velx_curriculum = (0.90, 1.00, 1.10)

    # Direction-quality weights sum to one. Forward tracking gates all of them,
    # so standing cannot earn heading/yaw reward, but one directional error no
    # longer annihilates the complete PPO learning signal as it did in v2.
    bound_direction_base_weight = 0.55
    bound_lateral_weight = 0.15
    bound_yaw_rate_weight = 0.15
    bound_heading_weight = 0.15

    # A positive-velocity progress component makes acquisition from scratch
    # learnable. The narrow exponential still owns most of the score and makes
    # exact tracking strictly better than underspeed or overspeed motion.
    bound_forward_tracking_weight = 0.65
    bound_forward_progress_weight = 0.35

    def __post_init__(self):
        super().__post_init__()
        direction_weight_sum = (
            self.bound_direction_base_weight
            + self.bound_lateral_weight
            + self.bound_yaw_rate_weight
            + self.bound_heading_weight
        )
        if abs(direction_weight_sum - 1.0) > 1.0e-6:
            raise ValueError(f"Bound v3 direction weights must sum to 1.0, got {direction_weight_sum}.")
        forward_weight_sum = self.bound_forward_tracking_weight + self.bound_forward_progress_weight
        if abs(forward_weight_sum - 1.0) > 1.0e-6:
            raise ValueError(f"Bound v3 forward weights must sum to 1.0, got {forward_weight_sum}.")
        if self.command_lin_vel_x_range[0] <= 0.0:
            raise ValueError("Bound v3 forward-progress shaping requires strictly positive vx commands.")


class Solo12BoundV3Env(Solo12BoundV2Env):
    """Bound v3 environment; left/right equivariance is trained by PPO mirror loss."""

    cfg: Solo12BoundV3EnvCfg

    def _compute_command_tracking_scores(
        self,
        root_lin_vel_b: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tracking = super()._compute_command_tracking_scores(root_lin_vel_b, root_ang_vel_b)
        forward_tracking_exp_score = tracking["forward_score"]
        forward_progress_score = torch.clamp(
            root_lin_vel_b[:, 0] / torch.clamp(self._commands[:, 0], min=1.0e-6),
            min=0.0,
            max=1.0,
        )
        forward_score = (
            self.cfg.bound_forward_tracking_weight * forward_tracking_exp_score
            + self.cfg.bound_forward_progress_weight * forward_progress_score
        )
        direction_quality = (
            self.cfg.bound_direction_base_weight
            + self.cfg.bound_lateral_weight * tracking["lateral_score"]
            + self.cfg.bound_yaw_rate_weight * tracking["yaw_rate_score"]
            + self.cfg.bound_heading_weight * tracking["heading_score"]
        )
        tracking["forward_tracking_exp_score"] = forward_tracking_exp_score
        tracking["forward_progress_score"] = forward_progress_score
        tracking["forward_score"] = forward_score
        tracking["direction_quality"] = direction_quality
        tracking["command_score"] = forward_score * direction_quality
        return tracking

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()

        tracking = self._compute_command_tracking_scores(
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
        )
        sign = self._actions.new_tensor((-1.0, 1.0, 1.0))
        front_mirror_error = torch.mean(
            torch.abs(self._actions[:, 0:3] - self._actions[:, 3:6] * sign), dim=1
        )
        rear_mirror_error = torch.mean(
            torch.abs(self._actions[:, 6:9] - self._actions[:, 9:12] * sign), dim=1
        )
        self.extras["log"].update(
            {
                "Metrics/direction_quality": torch.mean(tracking["direction_quality"]).item(),
                "Metrics/forward_tracking_exp_score": torch.mean(
                    tracking["forward_tracking_exp_score"]
                ).item(),
                "Metrics/forward_progress_score": torch.mean(tracking["forward_progress_score"]).item(),
                "Metrics/front_action_mirror_mae": torch.mean(front_mirror_error).item(),
                "Metrics/rear_action_mirror_mae": torch.mean(rear_mirror_error).item(),
            }
        )
        return reward
