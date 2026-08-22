"""Isaac Lab task: route policy commanding a frozen velocity-height DiffuseLoco.

One external action is four-dimensional.  Internally, one action executes a complete
four-step diffusion chunk over the robot's twelve joints.  These two action spaces
are deliberately stored in different buffers.
"""

from __future__ import annotations

import torch

from .contracts import ActionBounds
from .low_level import FrozenDiffuseLoco
from .rewards import RewardWeights, hierarchical_reward, schedule_potential
from .route import PolylineRoute, build_route_bank, route_bank_from_polyline

try:  # Isaac Lab is intentionally optional for pure-Python tests.
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.utils.buffers import DelayBuffer
    from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
    from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
    from scripts.baseline_diffuseloco.train.data.obs_utils import proprio_from_env_tensors
except ImportError:  # pragma: no cover
    DelayBuffer = None
    Solo12Env = object  # type: ignore[misc,assignment]
    Solo12EnvCfg = object  # type: ignore[misc,assignment]
    proprio_from_env_tensors = None


if Solo12EnvCfg is not object:
    _LOCAL_FLAT_TERRAIN = TerrainGeneratorCfg(
        seed=0, curriculum=False, size=(500.0, 500.0), border_width=0.0,
        num_rows=1, num_cols=1, use_cache=False,
        sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
    )


class HierarchicalSolo12EnvCfg(Solo12EnvCfg):  # type: ignore[misc]
    """Frozen-low-level, forward-route experiment configuration."""

    action_space = 4
    # preview 8*6 + final pose 4 + remaining/schedule/speed 3 + state 13
    observation_space = 68
    episode_length_s = 8.0
    diffuseloco_checkpoint: str = ""
    diffuseloco_inference_steps: int | None = 4
    diffuseloco_exec_horizon: int = 4
    high_level_decimation: int = 4
    low_level_action_clip: float = 100.0

    command_vx_range = (0.0, 0.65)
    command_vy_range = (-0.45, 0.45)
    command_wz_range = (-0.50, 0.50)
    command_height_range = (0.1705, 0.2932)

    route_file: str = ""
    route_bank_size: int = 192
    route_points: int = 65
    route_seed: int = 17
    route_families = ("straight", "s_curve", "right_angle")
    desired_mean_speed_range = (0.30, 0.42)
    route_preview_points: int = 8
    route_preview_distances_m = (0.15, 0.30, 0.45, 0.65, 0.90, 1.20, 1.55, 2.00)
    max_progress_jump_m: float = 0.80
    max_backtrack_m: float = 0.20

    corridor_half_width_m: float = 0.30
    schedule_error_scale_m: float = 0.30
    heading_error_scale_rad: float = 0.50
    goal_position_tolerance_m: float = 0.25
    goal_yaw_tolerance_rad: float = 0.35
    mean_speed_tolerance_mps: float = 0.06
    reward_discount: float = 0.99
    reward_weights: RewardWeights = RewardWeights()

    # The first experiment isolates planning. Robustness/domain randomization is a
    # later ablation and must not silently trigger the inherited Solo12 curriculum.
    events = None
    forces_applied_to_base_curriculum = ()
    max_velx_range_curriculum = ()
    base_push_force_xy_range = (0.0, 0.0)
    base_push_force_z_range = (0.0, 0.0)
    actuation_delay_range = (0, 0)
    tricky_terrain = False
    reset_x_pos = 0.0
    reset_y_pos = 0.0
    reset_yaw = 0.0
    reset_base_lin_vel_range = (0.0, 0.0)
    reset_base_ang_vel_range = (0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()
        # Solo12's post-init derives its own locomotion observation size; restore
        # the high-level contract after that base-class bookkeeping.
        self.action_space = 4
        self.observation_space = 68
        # GroundPlaneCfg references a remote Nucleus asset. A generated two-triangle
        # plane is physically equivalent and makes local/cluster runs self-contained.
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = _LOCAL_FLAT_TERRAIN.copy()
        self.terrain.use_terrain_origins = False


class HierarchicalSolo12Env(Solo12Env):  # type: ignore[misc]
    """Normalized high-level command -> frozen DiffuseLoco -> joint targets."""

    cfg: HierarchicalSolo12EnvCfg

    def __init__(self, cfg: HierarchicalSolo12EnvCfg, render_mode: str | None = None, **kwargs):
        if not cfg.diffuseloco_checkpoint:
            raise ValueError("cfg.diffuseloco_checkpoint is required.")
        if cfg.route_preview_points != len(cfg.route_preview_distances_m):
            raise ValueError("route_preview_points must match route_preview_distances_m.")
        if cfg.high_level_decimation != cfg.diffuseloco_exec_horizon:
            raise ValueError("One high-level action must own exactly one diffusion execution chunk.")
        super().__init__(cfg, render_mode, **kwargs)

        self._bounds = ActionBounds(
            vx=cfg.command_vx_range, vy=cfg.command_vy_range,
            wz=cfg.command_wz_range, height=cfg.command_height_range,
        )
        self._configure_joint_action_buffers()
        self._preview_distances = torch.as_tensor(cfg.route_preview_distances_m, device=self.device)
        route_bank = (
            route_bank_from_polyline(PolylineRoute.from_file(cfg.route_file), episode_duration_s=cfg.episode_length_s)
            if cfg.route_file else
            build_route_bank(
                count=cfg.route_bank_size, points=cfg.route_points,
                episode_duration_s=cfg.episode_length_s,
                speed_range=cfg.desired_mean_speed_range, seed=cfg.route_seed,
                families=tuple(cfg.route_families),
                low_height=cfg.command_height_range[0], high_height=cfg.command_height_range[1],
            )
        )
        self._cache_route_bank(route_bank)
        self._allocate_route_batch(route_bank.xy.shape[1])
        self._assign_routes(torch.arange(self.num_envs, device=self.device))

        self._low_level = FrozenDiffuseLoco(
            cfg.diffuseloco_checkpoint, device=self.device,
            inference_steps=cfg.diffuseloco_inference_steps,
            exec_horizon=cfg.diffuseloco_exec_horizon,
        )
        self._low_level.reset(self.num_envs)
        self._previous_command = torch.zeros(self.num_envs, 4, device=self.device)
        self._progress = torch.zeros(self.num_envs, device=self.device)
        self._cross_track = torch.zeros_like(self._progress)
        self._previous_schedule_potential = torch.zeros_like(self._progress)
        self._high_observation = torch.zeros(self.num_envs, cfg.observation_space, device=self.device)
        self._latest_route_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._latest_final_position_error = torch.zeros_like(self._progress)
        self._latest_mean_speed_error = torch.zeros_like(self._progress)
        self._episode_term_sums = {
            name: torch.zeros_like(self._progress) for name in (
                "schedule_potential", "cross_track", "heading", "height",
                "command_delta", "terminal_pose", "fall",
            )
        }
        self._set_neutral_command(torch.arange(self.num_envs, device=self.device))

    def _configure_joint_action_buffers(self) -> None:
        """Replace the inherited 4-D buffers with the 12-D actuator contract."""
        joint_dim = len(self._joint_ids)
        shape = (self.num_envs, joint_dim)
        self._actions = torch.zeros(shape, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)
        self._delayed_processed_actions = torch.zeros_like(self._actions)
        self._applied_actions = torch.zeros_like(self._actions)
        self._action_delay_buffer = DelayBuffer(self.cfg.actuation_delay_range[1], self.num_envs, device=self.device)
        self._action_delay_steps.zero_()

    def _cache_route_bank(self, bank) -> None:
        self._bank_xy = torch.as_tensor(bank.xy, device=self.device)
        self._bank_yaw = torch.as_tensor(bank.yaw, device=self.device)
        self._bank_height = torch.as_tensor(bank.target_height, device=self.device)
        self._bank_arc = torch.as_tensor(bank.arc_length, device=self.device)
        self._bank_length = torch.as_tensor(bank.length, device=self.device)
        self._bank_speed = torch.as_tensor(bank.desired_mean_speed, device=self.device)
        self._bank_family = tuple(bank.family)

    def _allocate_route_batch(self, points: int) -> None:
        self._route_xy = torch.empty((self.num_envs, points, 2), device=self.device)
        self._route_yaw = torch.empty((self.num_envs, points), device=self.device)
        self._route_height = torch.empty_like(self._route_yaw)
        self._route_arc = torch.empty_like(self._route_yaw)
        self._route_length = torch.empty(self.num_envs, device=self.device)
        self._desired_mean_speed = torch.empty_like(self._route_length)
        self._route_bank_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _assign_routes(self, env_ids: torch.Tensor) -> None:
        if self._bank_xy.shape[0] == 1:
            indices = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        else:
            indices = torch.randint(self._bank_xy.shape[0], (len(env_ids),), device=self.device)
        self._route_bank_index[env_ids] = indices
        self._route_xy[env_ids] = self._bank_xy[indices]
        self._route_yaw[env_ids] = self._bank_yaw[indices]
        self._route_height[env_ids] = self._bank_height[indices]
        self._route_arc[env_ids] = self._bank_arc[indices]
        self._route_length[env_ids] = self._bank_length[indices]
        self._desired_mean_speed[env_ids] = self._bank_speed[indices]

    def _command_limits(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self._bounds.low, device=self.device, dtype=dtype),
            torch.as_tensor(self._bounds.high, device=self.device, dtype=dtype),
        )

    def _normalize_command(self, command: torch.Tensor) -> torch.Tensor:
        low, high = self._command_limits(command.dtype)
        return (2.0 * (command - low) / (high - low) - 1.0).clamp(-1.0, 1.0)

    def _decode_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != (self.num_envs, 4):
            raise ValueError(f"Expected high-level action [{self.num_envs},4], got {tuple(action.shape)}")
        normalized = action.to(self.device).clamp(-1.0, 1.0)
        low, high = self._command_limits(normalized.dtype)
        return low + 0.5 * (normalized + 1.0) * (high - low)

    def _set_neutral_command(self, env_ids: torch.Tensor) -> None:
        low, high = self._command_limits(torch.float32)
        self._previous_command[env_ids] = 0.5 * (low + high)

    def _sample_route(self, arc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        one_value = arc.ndim == 1
        values = arc[:, None] if one_value else arc
        valid = values <= self._route_length[:, None]
        values = torch.maximum(values, torch.zeros_like(values))
        values = torch.minimum(values, self._route_length[:, None])
        index = torch.searchsorted(self._route_arc, values.contiguous(), right=True) - 1
        index = index.clamp(0, self._route_arc.shape[1] - 2)

        def gather_2d(source: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return torch.gather(source, 1, idx)

        s0 = gather_2d(self._route_arc, index)
        s1 = gather_2d(self._route_arc, index + 1)
        alpha = (values - s0) / (s1 - s0).clamp_min(1e-6)
        xy_index = index[..., None].expand(-1, -1, 2)
        xy0 = torch.gather(self._route_xy, 1, xy_index)
        xy1 = torch.gather(self._route_xy, 1, xy_index + 1)
        xy = xy0 + alpha[..., None] * (xy1 - xy0)
        yaw0, yaw1 = gather_2d(self._route_yaw, index), gather_2d(self._route_yaw, index + 1)
        yaw = torch.atan2(
            (1.0 - alpha) * torch.sin(yaw0) + alpha * torch.sin(yaw1),
            (1.0 - alpha) * torch.cos(yaw0) + alpha * torch.cos(yaw1),
        )
        h0, h1 = gather_2d(self._route_height, index), gather_2d(self._route_height, index + 1)
        height = h0 + alpha * (h1 - h0)
        if one_value:
            return xy[:, 0], yaw[:, 0], height[:, 0], valid[:, 0].float()
        return xy, yaw, height, valid.float()

    def _project_route(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self._route_xy[:, :-1]
        segments = self._route_xy[:, 1:] - starts
        length = torch.linalg.vector_norm(segments, dim=-1).clamp_min(1e-6)
        rel = position[:, None, :] - starts
        fraction = ((rel * segments).sum(dim=-1) / length.square()).clamp(0.0, 1.0)
        projection = starts + fraction[..., None] * segments
        distance_sq = (position[:, None, :] - projection).square().sum(dim=-1)
        arc = self._route_arc[:, :-1] + fraction * length
        lower = (self._progress - self.cfg.max_backtrack_m).clamp_min(0.0)[:, None]
        upper = torch.minimum(self._progress + self.cfg.max_progress_jump_m, self._route_length)[:, None]
        allowed = (arc >= lower) & (arc <= upper)
        masked = torch.where(allowed, distance_sq, torch.full_like(distance_sq, torch.inf))
        masked = torch.where(allowed.any(dim=1, keepdim=True), masked, distance_sq)
        index = masked.argmin(dim=-1)
        batch = torch.arange(self.num_envs, device=self.device)
        chosen_arc = arc[batch, index]
        chosen_projection = projection[batch, index]
        chosen_segment = segments[batch, index]
        signed_cross_track = (
            chosen_segment[:, 0] * (position[:, 1] - chosen_projection[:, 1])
            - chosen_segment[:, 1] * (position[:, 0] - chosen_projection[:, 0])
        ) / length[batch, index]
        return chosen_arc, signed_cross_track

    def _robot_yaw(self) -> torch.Tensor:
        quat = self._robot.data.root_quat_w
        return torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
        )

    def _route_features(self):
        position = self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]
        progress, cross_track = self._project_route(position)
        self._progress.copy_(progress)
        self._cross_track.copy_(cross_track)
        preview_s = progress[:, None] + self._preview_distances[None]
        target_xy, target_yaw, target_height, valid = self._sample_route(preview_s)
        robot_yaw = self._robot_yaw()
        c, s = torch.cos(robot_yaw), torch.sin(robot_yaw)
        delta = target_xy - position[:, None]
        local_x = c[:, None] * delta[..., 0] + s[:, None] * delta[..., 1]
        local_y = -s[:, None] * delta[..., 0] + c[:, None] * delta[..., 1]
        local_yaw = torch.atan2(torch.sin(target_yaw - robot_yaw[:, None]), torch.cos(target_yaw - robot_yaw[:, None]))
        low_h, high_h = self.cfg.command_height_range
        normalized_height = 2.0 * (target_height - low_h) / (high_h - low_h) - 1.0
        preview = torch.stack((local_x, local_y, torch.sin(local_yaw), torch.cos(local_yaw), normalized_height, valid), dim=-1)

        final_delta = self._route_xy[:, -1] - position
        final_yaw_error = torch.atan2(
            torch.sin(self._route_yaw[:, -1] - robot_yaw), torch.cos(self._route_yaw[:, -1] - robot_yaw)
        )
        final_pose = torch.stack((
            c * final_delta[:, 0] + s * final_delta[:, 1],
            -s * final_delta[:, 0] + c * final_delta[:, 1],
            torch.sin(final_yaw_error), torch.cos(final_yaw_error),
        ), dim=-1)
        _, route_yaw, current_height, _ = self._sample_route(progress)
        heading_error = torch.atan2(torch.sin(route_yaw - robot_yaw), torch.cos(route_yaw - robot_yaw))
        elapsed = self.episode_length_buf.float() * self.step_dt
        target_progress = torch.minimum(self._desired_mean_speed * elapsed, self._route_length)
        schedule_error = progress - target_progress
        remaining = (self.max_episode_length - self.episode_length_buf).float() * self.step_dt
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        normalized_base_height = (base_height - low_h) / (high_h - low_h)
        tracking_error = torch.cat((
            self._robot.data.root_lin_vel_b[:, :2] - self._previous_command[:, :2],
            self._robot.data.root_ang_vel_b[:, 2:3] - self._previous_command[:, 2:3],
        ), dim=-1)
        state = torch.cat((
            self._robot.data.root_lin_vel_b[:, :2], self._robot.data.root_ang_vel_b[:, 2:3],
            normalized_base_height[:, None], self._robot.data.projected_gravity_b[:, :2],
            tracking_error, self._normalize_command(self._previous_command),
        ), dim=-1)
        return preview, final_pose, remaining, schedule_error, current_height, heading_error, state, cross_track, base_height

    def _get_observations(self) -> dict[str, torch.Tensor]:
        preview, final_pose, remaining, schedule_error, _, _, state, _, _ = self._route_features()
        observation = torch.cat((
            preview.reshape(self.num_envs, -1), final_pose,
            (remaining / self.cfg.episode_length_s).clamp(0.0, 1.0)[:, None],
            (schedule_error / self.cfg.schedule_error_scale_m)[:, None],
            (self._desired_mean_speed / self.cfg.command_vx_range[1])[:, None], state,
        ), dim=-1)
        if observation.shape[1] != self.cfg.observation_space:
            raise RuntimeError(f"Observation contract is {observation.shape[1]}D, config says {self.cfg.observation_space}D.")
        self._high_observation.copy_(observation)
        return {"policy": self._high_observation}

    def _apply_low_level_control(self, command: torch.Tensor, is_rendering: bool) -> None:
        proprio = proprio_from_env_tensors(
            self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs,
            self._robot.data.joint_vel[:, self._joint_ids],
            self._robot.data.root_ang_vel_b, self._robot.data.projected_gravity_b,
        )
        joint_action = self._low_level.act(proprio, command)
        self._actions.copy_(joint_action.clamp(-self.cfg.low_level_action_clip, self.cfg.low_level_action_clip))
        self._processed_actions.copy_(self.cfg.action_scale * self._actions + self._q_offset_action_and_obs)
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._record_base_imu_history_sample()
        self._previous_actions.copy_(self._actions)

    def step(self, action: torch.Tensor):
        """Execute one direct high-level command and one frozen-policy chunk."""
        command = self._decode_action(action)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.high_level_decimation):
            self._apply_low_level_control(command, is_rendering)
        previous_command = self._previous_command.clone()
        self._previous_command.copy_(command)
        self._commands[:, :3] = command[:, :3]

        self.episode_length_buf += self.cfg.high_level_decimation
        self.common_step_counter += self.cfg.high_level_decimation
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards(previous_command)
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _get_rewards(self, previous_command: torch.Tensor) -> torch.Tensor:
        _, final_pose, _, schedule_error, target_height, heading_error, _, cross_track, base_height = self._route_features()
        position_error = torch.linalg.vector_norm(final_pose[:, :2], dim=-1)
        yaw_error = torch.atan2(final_pose[:, 2], final_pose[:, 3])
        elapsed = (self.episode_length_buf.float() * self.step_dt).clamp_min(self.step_dt)
        mean_speed_error = self._progress / elapsed - self._desired_mean_speed
        route_success = (
            self.reset_time_outs & (position_error <= self.cfg.goal_position_tolerance_m)
            & (torch.abs(yaw_error) <= self.cfg.goal_yaw_tolerance_rad)
            & (torch.abs(mean_speed_error) <= self.cfg.mean_speed_tolerance_mps)
        )
        current_potential = schedule_potential(schedule_error / self.cfg.schedule_error_scale_m)
        terminal_error = torch.stack((
            final_pose[:, 0] / self.cfg.goal_position_tolerance_m,
            final_pose[:, 1] / self.cfg.goal_position_tolerance_m,
            yaw_error / self.cfg.goal_yaw_tolerance_rad,
        ), dim=-1)
        macro_dt = self.cfg.high_level_decimation * self.step_dt
        reward, terms = hierarchical_reward(
            previous_schedule_potential=self._previous_schedule_potential,
            current_schedule_potential=current_potential,
            cross_track_normalized=cross_track / self.cfg.corridor_half_width_m,
            heading_error_normalized=heading_error / self.cfg.heading_error_scale_rad,
            height_error_normalized=(base_height - target_height) / (
                self.cfg.command_height_range[1] - self.cfg.command_height_range[0]
            ),
            normalized_command_delta=self._normalize_command(self._previous_command)
            - self._normalize_command(previous_command),
            terminal_pose_error_normalized=terminal_error,
            terminal=self.reset_buf, fallen=self.reset_terminated,
            discount=self.cfg.reward_discount, macro_dt=macro_dt, weights=self.cfg.reward_weights,
        )
        self._previous_schedule_potential.copy_(current_potential)
        self._episode_reward_sums += reward
        for name, value in terms.items():
            self._episode_term_sums[name] += value
        self._latest_route_success.copy_(route_success)
        self._latest_final_position_error.copy_(position_error)
        self._latest_mean_speed_error.copy_(mean_speed_error)
        self.extras["log"] = {
            **{f"RewardsPerStep/{name}": float(value.mean()) for name, value in terms.items()},
            "Metrics/route_progress_fraction": float((self._progress / self._route_length).mean()),
            "Metrics/schedule_error_m": float(schedule_error.abs().mean()),
            "Metrics/cross_track_m": float(cross_track.abs().mean()),
            "Metrics/height_error_m": float((base_height - target_height).abs().mean()),
            "Metrics/mean_speed_error_mps": float(mean_speed_error.abs().mean()),
        }
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        custom_ready = hasattr(self, "_episode_term_sums")
        episode_log = {}
        if custom_ready:
            completed = env_ids[self.episode_length_buf[env_ids] > 0]
            if len(completed) > 0:
                episode_log = {
                    **{f"Episode_RewardHL/{name}": float(value[completed].mean())
                       for name, value in self._episode_term_sums.items()},
                    "Episode_Metrics/route_success_rate": float(self._latest_route_success[completed].float().mean()),
                    "Episode_Metrics/final_position_error_m": float(self._latest_final_position_error[completed].mean()),
                    "Episode_Metrics/mean_speed_error_mps": float(self._latest_mean_speed_error[completed].abs().mean()),
                    "Episode_Termination/route_success": int(self._latest_route_success[completed].sum()),
                }
        super()._reset_idx(env_ids)
        if not custom_ready:
            return
        self.extras.setdefault("log", {}).update(episode_log)
        self._low_level.reset_envs(env_ids)
        self._assign_routes(env_ids)
        self._set_neutral_command(env_ids)
        self._progress[env_ids] = 0.0
        self._cross_track[env_ids] = 0.0
        self._previous_schedule_potential[env_ids] = 0.0
        self._latest_route_success[env_ids] = False
        self._latest_final_position_error[env_ids] = 0.0
        self._latest_mean_speed_error[env_ids] = 0.0
        for value in self._episode_term_sums.values():
            value[env_ids] = 0.0
