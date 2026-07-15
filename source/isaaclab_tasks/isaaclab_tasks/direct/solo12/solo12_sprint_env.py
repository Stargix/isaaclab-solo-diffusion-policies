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
class Solo12SprintEnvCfg(Solo12EnvCfg):
    # --- Velocity command ranges for sprinting (pure forward sprint, consistent with opposite_prob=0.0) ---
    command_lin_vel_x_range = (0.3, 2.0)
    command_lin_vel_y_range = (-0.2, 0.2)
    command_ang_vel_z_range = (-0.2, 0.2)
    opposite_direction_cmd_prob = 0.0

    # Progressive velocity curriculum tailored for Solo12 (prevents initial falls while demanding real tracking)
    max_velx_range_curriculum = [0.6, 1.2, 2.0]
    forces_curriculum_threshold_reward = 35.0

    # Dominant velocity tracking reward to prevent static standing trap
    track_lin_vel_xy_reward_scale = 3.0

    # --- Clean training (Jordi's settings) ---
    enable_observation_corruption = False
    base_push_force_xy_range = (0.0, 0.0)
    base_push_force_z_range = (0.0, 0.0)
    forces_applied_to_base_curriculum = [0.0]
    actuation_delay_range = (0, 0)
    tricky_terrain = False

    # Height tracking reward config (Aractingi et al. 2023 report 0.23-0.25m clearance at high speed)
    track_base_height_reward_scale = 0.12
    base_z_desired = 0.25

    # Gait reward config calibrated for Solo12 natural trotting cadence (0.15s air time)
    feet_air_time_reward_scale = 1.5
    feet_air_time_threshold = 0.15

    # Reduced torque penalty to permit explosive push-off torques for sprinting (Aractingi et al. 2023)
    joint_torque_reward_scale = -0.1e-3

    # --- Actuator gains matching compliant Solo12 hardware validation (Aractingi et al. 2023 / crouch v5) ---
    kp = 9.0
    kd = 0.3

    def __post_init__(self):
        super().__post_init__()
        # Sync actuator gains from config into the robot articulation actuators
        if "legs" in self.robot.actuators:
            self.robot.actuators["legs"].stiffness = self.kp
            self.robot.actuators["legs"].damping = self.kd


@configclass
class Solo12SprintPPORunnerCfg(Solo12PPORunnerCfg):
    experiment_name = "solo12_rsl_rl_sprint_runs"
    run_name = "solo12_sprint_clean"


class Solo12SprintEnv(Solo12Env):
    cfg: Solo12SprintEnvCfg

    def __init__(self, cfg: Solo12SprintEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
