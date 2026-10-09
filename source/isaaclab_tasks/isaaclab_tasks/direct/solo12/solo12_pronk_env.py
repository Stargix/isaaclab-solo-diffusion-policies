# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robust velocity fine-tuning of a warm-started four-foot pronk expert."""

from __future__ import annotations

import math

import torch
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .agents.rsl_rl_ppo_cfg import Solo12PPORunnerWithSymmetryCfg
from .pronk_gait import PronkCycleTracker
from .solo12_env import Solo12Env
from .solo12_env_cfg import Solo12EnvCfg
import solo12_symmetry


PRONK_FOOT_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")


def _load_foot_cylinder_geometry(usd_path: str) -> list[list[tuple[float, ...]]]:
    """Read center/axial/radial vectors from the actual foot colliders in the asset."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    geometry = []
    for body_name in PRONK_FOOT_NAMES:
        calf = stage.GetPrimAtPath(root.GetPath().AppendChild(body_name))
        colliders = [p for p in Usd.PrimRange(calf, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetTypeName() == "Cylinder"]
        if len(colliders) != 1:
            raise ValueError(f"Expected one cylindrical foot collider in {body_name}, got {len(colliders)}.")
        cylinder = UsdGeom.Cylinder(colliders[0])
        matrix, _ = cache.ComputeRelativeTransform(colliders[0], calf)
        axis = "XYZ".index(cylinder.GetAxisAttr().Get())
        basis = [Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)]
        axial = matrix.TransformDir(basis[axis]) * (cylinder.GetHeightAttr().Get() / 2.0)
        radial = [matrix.TransformDir(basis[i]) * cylinder.GetRadiusAttr().Get()
                  for i in range(3) if i != axis]
        geometry.append([tuple(matrix.ExtractTranslation()), tuple(axial), tuple(radial[0]), tuple(radial[1])])
    return geometry


def compute_pronk_left_right_symmetry(env=None, obs=None, actions=None, obs_type: str = "policy"):
    """Keep only identity and left-right reflection from the shared SOLO12 maps."""
    if obs_type != "policy":
        raise ValueError(f"Pronk symmetry only supports obs_type='policy', got {obs_type!r}.")
    if obs is None and actions is None:
        return None, None
    if obs is not None:
        policy_obs = obs if isinstance(obs, torch.Tensor) else obs[obs_type]
        batch_size = policy_obs.shape[0]
    else:
        batch_size = actions.shape[0]

    observations_aug, actions_aug = solo12_symmetry.compute_symmetric_observations_actions(
        env=env, obs=obs, actions=actions, obs_type=obs_type
    )
    keep = 2 * batch_size  # identity, then reflect_x (left-right across the sagittal plane)
    if observations_aug is not None:
        observations_aug = observations_aug[:keep]
    if actions_aug is not None:
        actions_aug = actions_aug[:keep]
    return observations_aug, actions_aug


@configclass
class Solo12PronkEnvCfg(Solo12EnvCfg):
    """Keep the base SOLO12 randomization while restricting commands to useful steering."""

    command_lin_vel_x_range = (0.15, 1.0)
    command_lin_vel_y_range = (-0.15, 0.15)
    command_ang_vel_z_range = (-0.35, 0.35)
    command_resampling_time_s = 3.0
    standing_env_prob = 0.02
    opposite_direction_cmd_prob = 0.0
    max_velx_range_curriculum = ()

    # Keep the same nominal PD and action/observation contract as the walk actor.
    kp = 9.0
    kd = 0.2
    tracking_std = 0.20

    # No instantaneous base-height target: it would suppress the flight phase.
    track_base_height_reward_scale = 0.0
    feet_air_time_reward_scale = 0.0

    # Event bonus, paid once per valid landed cycle (not multiplied by dt).
    pronk_cycle_reward_scale = 0.25
    pronk_minimum_flight_s = 0.025
    pronk_maximum_flight_s = 0.20
    pronk_landing_window_s = 0.04
    pronk_support_window_s = 0.08
    pronk_minimum_clearance_m = 0.003
    pronk_warmup_s = 0.50
    pronk_synchronization_std_s = 0.02
    pronk_max_tilt_deg = 25.0
    pronk_height_range_m = (0.18, 0.45)
    pronk_foot_names = PRONK_FOOT_NAMES

    def __post_init__(self):
        super().__post_init__()
        if self.policy_model != "simple_mlp" or self.observation_space != 48:
            raise ValueError("Pronk fine-tuning requires the compatible 48-D simple-MLP actor contract.")
        if tuple(self.command_lin_vel_x_range) != (0.15, 1.0):
            raise ValueError("Pronk command curriculum expects forward vx in [0.15, 1.0] m/s.")
        if tuple(self.command_lin_vel_y_range) != (-0.15, 0.15):
            raise ValueError("The configured left-right symmetry expects vy in [-0.15, 0.15] m/s.")
        if tuple(self.command_ang_vel_z_range) != (-0.35, 0.35):
            raise ValueError("The configured left-right symmetry expects yaw rate in [-0.35, 0.35] rad/s.")
        if not math.isfinite(self.pronk_cycle_reward_scale) or self.pronk_cycle_reward_scale <= 0.0:
            raise ValueError("pronk_cycle_reward_scale must be finite and positive.")
        if self.tricky_terrain:
            raise ValueError("Pronk geometric flight checks require the configured flat terrain.")
        if not 0.0 < self.pronk_max_tilt_deg < 90.0:
            raise ValueError("pronk_max_tilt_deg must be between 0 and 90 degrees.")
        if not 0.0 < self.pronk_height_range_m[0] < self.pronk_height_range_m[1]:
            raise ValueError("pronk_height_range_m must be a positive ordered interval.")
        # Update contact timers at 200 Hz, even though rewards are read at 50 Hz.
        self.scene.lazy_sensor_update = False
        # Local mesh avoids requiring a Nucleus ground-plane asset download.
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=0, curriculum=False, size=(200.0, 200.0), border_width=0.0,
            num_rows=1, num_cols=1, color_scheme="none",
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        self.terrain.use_terrain_origins = False
        if "legs" in self.robot.actuators:
            self.robot.actuators["legs"].stiffness = self.kp
            self.robot.actuators["legs"].damping = self.kd


@configclass
class Solo12PronkPPORunnerCfg(Solo12PPORunnerWithSymmetryCfg):
    """PPO configuration for actor-only pronk warm-start and robust adaptation."""

    num_steps_per_env = 32
    max_iterations = 2500
    save_interval = 50
    experiment_name = "solo12_rsl_rl_pronk_runs"
    run_name = "pronk_v1_jumpy_safe_seed42"

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.35
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.01
        self.algorithm.entropy_coef = 0.002
        symmetry_cfg = self.algorithm.symmetry_cfg
        if symmetry_cfg is None or not symmetry_cfg.use_data_augmentation:
            raise ValueError("Pronk PPO requires the registered left-right data augmentation.")
        symmetry_cfg.data_augmentation_func = compute_pronk_left_right_symmetry


class Solo12PronkEnv(Solo12Env):
    """Base velocity task plus a bounded bonus for completed physical pronk cycles."""

    cfg: Solo12PronkEnvCfg

    def __init__(self, cfg: Solo12PronkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._pronk_foot_ids, resolved_names = self._contact_sensor.find_bodies(
            list(self.cfg.pronk_foot_names), preserve_order=True
        )
        if resolved_names != list(self.cfg.pronk_foot_names):
            raise RuntimeError(f"Pronk feet must resolve in FL, FR, RL, RR order; got {resolved_names}.")
        self._pronk_robot_foot_ids, robot_names = self._robot.find_bodies(
            list(self.cfg.pronk_foot_names), preserve_order=True
        )
        if robot_names != resolved_names:
            raise RuntimeError("Pronk articulation and sensor foot order must match.")
        self._pronk_foot_geometry = torch.tensor(
            _load_foot_cylinder_geometry(self.cfg.robot.spawn.usd_path), device=self.device, dtype=torch.float32
        )
        self._pronk_cycle_tracker = PronkCycleTracker(
            self.num_envs, self.device, self.step_dt,
            minimum_flight_s=self.cfg.pronk_minimum_flight_s,
            maximum_flight_s=self.cfg.pronk_maximum_flight_s,
            landing_window_s=self.cfg.pronk_landing_window_s,
            support_window_s=self.cfg.pronk_support_window_s,
            minimum_clearance_m=self.cfg.pronk_minimum_clearance_m,
            warmup_s=self.cfg.pronk_warmup_s,
            tracking_std_mps=self.cfg.tracking_std,
            synchronization_std_s=self.cfg.pronk_synchronization_std_s,
        )
        if "pronk_gait" in self._episode_sums:
            raise RuntimeError("Duplicate pronk reward accumulator.")
        self._episode_sums["pronk_gait"] = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_pronk_cycle_tracker"):
            self._pronk_cycle_tracker.reset(env_ids)

    def _get_rewards(self) -> torch.Tensor:
        reward_base = super()._get_rewards()
        command_xy = self._commands[:, :2]
        velocity_xy = self._robot.data.root_lin_vel_b[:, :2]
        data = self._robot.data
        foot_quat = data.body_link_quat_w[:, self._pronk_robot_foot_ids, :]
        geometry_w = quat_apply(
            foot_quat.unsqueeze(-2).expand(-1, -1, 4, -1),
            self._pronk_foot_geometry.unsqueeze(0).expand(self.num_envs, -1, -1, -1),
        )
        center_z = data.body_link_pos_w[:, self._pronk_robot_foot_ids, 2] + geometry_w[:, :, 0, 2]
        extent_z = geometry_w[:, :, 1, 2].abs() + torch.sqrt(
            geometry_w[:, :, 2, 2].square() + geometry_w[:, :, 3, 2].square()
        )
        clearance = center_z - extent_z - self._terrain.env_origins[:, 2:3]
        base_height = data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        valid_pose = (
            (-data.projected_gravity_b[:, 2] >= math.cos(math.radians(self.cfg.pronk_max_tilt_deg)))
            & (base_height >= self.cfg.pronk_height_range_m[0])
            & (base_height <= self.cfg.pronk_height_range_m[1])
            & ~self.reset_terminated
            & (self._compute_contact_count(self._thigh_body_ids, self.cfg.undesired_contact_threshold) == 0)
        )
        contacts = torch.linalg.vector_norm(
            self._contact_sensor.data.net_forces_w[:, self._pronk_foot_ids, :], dim=-1
        ) > self.cfg.contact_sensor.force_threshold
        components = self._pronk_cycle_tracker.update(
            contacts, clearance, self._contact_sensor.data.current_air_time[:, self._pronk_foot_ids],
            command_xy, velocity_xy, valid_pose,
        )
        pronk_reward = components["gait"] * self.cfg.pronk_cycle_reward_scale
        self._episode_sums["pronk_gait"] += pronk_reward
        self._episode_reward_sums += pronk_reward
        completed_mask = components["completed"].float()
        completed_count = completed_mask.sum().clamp_min(1.0)
        log = self.extras.setdefault("log", {})
        log.update(
            {
                "RewardsPerStep/pronk_gait": torch.mean(pronk_reward).item(),
                "RewardsPerStep/total": torch.mean(reward_base + pronk_reward).item(),
                "Metrics/pronk_completed_cycles_per_step": torch.mean(components["completed"].float()).item(),
                "Metrics/pronk_physical_flight_fraction": torch.mean(components["physical_flight"].float()).item(),
                "Metrics/pronk_real_progress": (
                    (components["progress"] * completed_mask).sum() / completed_count
                ).item(),
                "Metrics/pronk_command_tracking": (
                    (components["tracking"] * completed_mask).sum() / completed_count
                ).item(),
                "Metrics/pronk_height_m": torch.mean(
                    self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
                ).item(),
            }
        )
        return reward_base + pronk_reward
