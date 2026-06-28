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

    # --- FIX 1: Sharper tracking gaussian ---
    # sqrt(0.25)=0.5 was too forgiving: robot got 70% reward standing still.
    # sqrt(0.1)=0.316 makes standing still yield only ~40%, forcing locomotion.
    tracking_std = math.sqrt(0.1)

    # --- FIX 2: Activate feet air time reward ---
    # Was 0.0 — no incentive to lift feet. Now rewards stepping.
    feet_air_time_reward_scale = 1.0
    feet_air_time_threshold = 0.25  # Lower threshold for short crouched steps

    # --- FIX 3: Reduce movement penalties ---
    # Crouched gaits require faster, shorter steps that incur more action_rate
    # and foot_contact penalties. Halving them gives room to explore walking.
    action_rate_reward_scale = -0.01   # Was -0.05
    foot_contact_reward_scale = -0.5e-3  # Was -1.0e-3

    # --- FIX 4: Crouch height at 0.17m ---
    # 0.16m was biomechanically too extreme. 0.17m is still very visibly
    # crouched (erect = 0.24m) but gives just enough joint range for steps.
    target_base_height = 0.17
    base_height_reward_scale = -80.0  # Slightly softer to allow gait oscillation

    # Relax tilt penalty to allow natural pitch while crouched
    base_tilt_penalty_reward_scale = -0.1

    # Train on flat terrain for simplicity
    tricky_terrain = False


@configclass
class Solo12CrouchPPORunnerCfg(Solo12PPORunnerCfg):
    experiment_name = "solo12_rsl_rl_crouch_runs"
    run_name = "solo12_crouch_v2"


class Solo12CrouchEnv(Solo12Env):
    cfg: Solo12CrouchEnvCfg

    def __init__(self, cfg: Solo12CrouchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Register the base height reward in episode sums for logging
        self._episode_sums["base_height_l2"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        # Get base rewards computed by Solo12Env
        reward = super()._get_rewards()

        # Calculate specialized base height penalty
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

