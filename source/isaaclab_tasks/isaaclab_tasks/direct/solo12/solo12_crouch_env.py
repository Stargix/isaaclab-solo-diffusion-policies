# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from isaaclab.utils import configclass
from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
from isaaclab_tasks.direct.solo12.agents.rsl_rl_ppo_cfg import Solo12PPORunnerCfg


@configclass
class Solo12CrouchEnvCfg(Solo12EnvCfg):
    # --- Velocity command ranges ---
    command_lin_vel_x_range = (-0.5, 0.5)
    command_lin_vel_y_range = (-0.2, 0.2)
    command_ang_vel_z_range = (-0.5, 0.5)

    # --- CaT-inspired height termination ---
    # Max allowed height. If exceeded, episode is terminated (after warmup).
    crouch_height_limit = 0.21
    # Target height for a gentle centering guide reward.
    target_base_height = 0.18
    base_height_reward_scale = -20.0  # Softer guide penalty

    # --- Sharper tracking to reward movement ---
    tracking_std = math.sqrt(0.1)
    track_lin_vel_xy_reward_scale = 3.0  # Was 1.5. Doubled.
    track_ang_vel_z_reward_scale = 1.5   # Was 0.75. Doubled.

    # --- Activate feet air time reward ---
    feet_air_time_reward_scale = 2.0
    feet_air_time_threshold = 0.08  # Low threshold: reward short steps typical of crouching

    # --- Reduce movement penalties ---
    action_rate_reward_scale = -0.005    # Was -0.05
    foot_contact_reward_scale = -0.25e-3  # Was -1.0e-3

    # Relax tilt penalty to allow natural pitch while crouched
    base_tilt_penalty_reward_scale = -0.05

    # Train on flat terrain for simplicity
    tricky_terrain = False


@configclass
class Solo12CrouchPPORunnerCfg(Solo12PPORunnerCfg):
    experiment_name = "solo12_rsl_rl_crouch_runs"
    run_name = "solo12_crouch_v4_symmetry"


class Solo12CrouchEnv(Solo12Env):
    cfg: Solo12CrouchEnvCfg

    def __init__(self, cfg: Solo12CrouchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Register the base height reward in episode sums for logging
        self._episode_sums["base_height_l2"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        # Get base rewards computed by Solo12Env
        reward = super()._get_rewards()

        # Calculate specialized base height penalty (gentle centering guide)
        base_height = self._robot.data.root_pos_w[:, 2]
        height_error = torch.square(base_height - self.cfg.target_base_height)
        height_penalty = height_error * self.cfg.base_height_reward_scale * self.step_dt

        # Apply height penalty to step rewards and logs
        reward += height_penalty
        self._episode_sums["base_height_l2"] += height_penalty
        self._episode_reward_sums += height_penalty

        # Inject custom log entries if log exists in extras
        if "log" in self.extras:
            self.extras["log"]["RewardsPerStep/base_height_l2"] = torch.mean(height_penalty).item()
            self.extras["log"]["RewardsPerStep/total"] += torch.mean(height_penalty).item()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Get default terminations (base contact, timeouts...)
        terminated, time_out = super()._get_dones()

        # CaT-style height termination:
        # Robot spawns at 0.35m in USD and falls. Give 30 steps (~0.6s) of warmup
        # to settle and crouch before activating the height limit termination.
        base_height = self._robot.data.root_pos_w[:, 2]
        warmup_done = self.episode_length_buf > 30
        too_tall = (base_height > self.cfg.crouch_height_limit) & warmup_done

        # Combine terminations
        terminated = terminated | too_tall
        return terminated, time_out


