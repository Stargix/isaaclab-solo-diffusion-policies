"""Isaac Lab task: a slow route policy commanding a frozen DiffuseLoco policy.

One RL environment step executes one complete diffusion action chunk.  Consequently the
high-level policy cannot change the command inside a chunk, but no controller ever ramps
or filters that command.
"""

from __future__ import annotations

import torch

from .contracts import ActionBounds
from .low_level import FrozenDiffuseLoco
from .rewards import RewardWeights, hierarchical_reward
from .route import PolylineRoute, clearance_route

try:  # Isaac Lab is intentionally optional for the pure-Python tests.
    from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
    from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
    from scripts.baseline_diffuseloco.train.data.obs_utils import proprio_from_env_tensors
except ImportError:  # pragma: no cover
    Solo12Env = object  # type: ignore[misc,assignment]
    Solo12EnvCfg = object  # type: ignore[misc,assignment]
    proprio_from_env_tensors = None


class HierarchicalSolo12EnvCfg(Solo12EnvCfg):  # type: ignore[misc]
    """Configuration for the route/pose/time experiment.

    Height values live in DiffuseLoco command space.  They are not raw geometric roof
    heights until a calibrated body-envelope model or physical ceiling mesh is supplied.
    """

    action_space = 4  # normalized [-1, 1]: mapped affinely to ActionBounds
    # 8 * (x, y, sin_yaw, cos_yaw, max_height, valid) + final pose(4)
    # + remaining time + robot state(17)
    observation_space = 70
    episode_length_s = 8.0
    diffuseloco_checkpoint: str = ""
    diffuseloco_inference_steps: int | None = 4
    diffuseloco_exec_horizon: int = 4
    high_level_decimation: int = 4
    route_length_m: float = 5.0
    route_file: str = ""
    route_preview_points: int = 8
    route_preview_distances_m = (0.20, 0.40, 0.60, 0.90, 1.20, 1.60, 2.00, 2.50)
    max_progress_jump_m: float = 1.0
    max_backtrack_m: float = 0.20
    corridor_half_width_m: float = 0.30
    terminal_window_s: float = 1.0
    goal_position_tolerance_m: float = 0.25
    goal_yaw_tolerance_rad: float = 0.35
    # Align the default robot pose with a route whose first point is (0, 0, 0).
    reset_x_pos = 0.0
    reset_y_pos = 0.0
    reset_yaw = 0.0
    reset_base_lin_vel_range = (0.0, 0.0)
    reset_base_ang_vel_range = (0.0, 0.0)
    reward_weights: RewardWeights = RewardWeights()


class HierarchicalSolo12Env(Solo12Env):  # type: ignore[misc]
    """Normalized high-level action -> frozen DiffuseLoco -> joint targets."""

    cfg: HierarchicalSolo12EnvCfg

    def __init__(self, cfg: HierarchicalSolo12EnvCfg, render_mode: str | None = None, **kwargs):
        if not cfg.diffuseloco_checkpoint:
            raise ValueError("cfg.diffuseloco_checkpoint is required for the frozen low-level experiment.")
        if cfg.route_preview_points != len(cfg.route_preview_distances_m):
            raise ValueError("route_preview_points must match route_preview_distances_m.")
        if cfg.high_level_decimation != cfg.diffuseloco_exec_horizon:
            raise ValueError(
                "high_level_decimation must equal diffuseloco_exec_horizon so a high-level action "
                "always owns exactly one diffusion chunk."
            )
        super().__init__(cfg, render_mode, **kwargs)
        self._bounds = ActionBounds()
        self._route = PolylineRoute.from_file(cfg.route_file) if cfg.route_file else clearance_route(cfg.route_length_m)
        self._cache_route_tensors()
        self._preview_distances = torch.as_tensor(cfg.route_preview_distances_m, device=self.device)
        self._low_level = FrozenDiffuseLoco(
            cfg.diffuseloco_checkpoint,
            device=self.device,
            inference_steps=cfg.diffuseloco_inference_steps,
            exec_horizon=cfg.diffuseloco_exec_horizon,
        )
        self._low_level.reset(self.num_envs)
        self._previous_command = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_previous_command = torch.zeros_like(self._previous_command)
        self._older_command = torch.zeros_like(self._previous_command)
        self._progress = torch.zeros(self.num_envs, device=self.device)
        self._last_progress = torch.zeros_like(self._progress)
        self._cross_track = torch.zeros(self.num_envs, device=self.device)
        self._high_observation = torch.zeros(self.num_envs, cfg.observation_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

    def _cache_route_tensors(self) -> None:
        self._route_xy = torch.as_tensor(self._route.xy, device=self.device)
        self._route_yaw = torch.as_tensor(self._route.yaw, device=self.device)
        self._route_height = torch.as_tensor(self._route.max_command_height, device=self.device)
        self._route_arc = torch.as_tensor(self._route.arc_length, device=self.device)

    def _normalize_command(self, command: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self._bounds.low, device=self.device, dtype=command.dtype)
        high = torch.as_tensor(self._bounds.high, device=self.device, dtype=command.dtype)
        return (2.0 * (command - low) / (high - low) - 1.0).clamp(-1.0, 1.0)

    def _decode_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape != (self.num_envs, 4):
            raise ValueError(f"Expected normalized high-level action [{self.num_envs},4], got {tuple(action.shape)}")
        normalized = action.to(self.device).clamp(-1.0, 1.0)
        low = torch.as_tensor(self._bounds.low, device=self.device, dtype=normalized.dtype)
        high = torch.as_tensor(self._bounds.high, device=self.device, dtype=normalized.dtype)
        return low + 0.5 * (normalized + 1.0) * (high - low)

    def _sample_route(self, arc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the cached polyline at batched arc-length positions."""
        valid = arc <= self._route_arc[-1]
        s = arc.clamp(0.0, self._route_arc[-1])
        index = torch.searchsorted(self._route_arc, s.contiguous(), right=True) - 1
        index = index.clamp(0, len(self._route_arc) - 2)
        s0, s1 = self._route_arc[index], self._route_arc[index + 1]
        alpha = ((s - s0) / (s1 - s0).clamp_min(1e-6)).unsqueeze(-1)
        xy = self._route_xy[index] + alpha * (self._route_xy[index + 1] - self._route_xy[index])
        sin_yaw = (1.0 - alpha.squeeze(-1)) * torch.sin(self._route_yaw[index]) + alpha.squeeze(-1) * torch.sin(self._route_yaw[index + 1])
        cos_yaw = (1.0 - alpha.squeeze(-1)) * torch.cos(self._route_yaw[index]) + alpha.squeeze(-1) * torch.cos(self._route_yaw[index + 1])
        yaw = torch.atan2(sin_yaw, cos_yaw)
        height = (1.0 - alpha.squeeze(-1)) * self._route_height[index] + alpha.squeeze(-1) * self._route_height[index + 1]
        return xy, yaw, height, valid.float()

    def _project_route(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Continuous segment projection, constrained to prevent jumps at path crossings."""
        starts = self._route_xy[:-1]
        segments = self._route_xy[1:] - starts
        length = torch.linalg.vector_norm(segments, dim=-1).clamp_min(1e-6)
        rel = position[:, None, :] - starts[None, :, :]
        fraction = (rel * segments[None, :, :]).sum(dim=-1) / length.square()[None, :]
        fraction = fraction.clamp(0.0, 1.0)
        projection = starts[None, :, :] + fraction[..., None] * segments[None, :, :]
        distance_sq = torch.sum(torch.square(position[:, None, :] - projection), dim=-1)
        arc = self._route_arc[:-1][None, :] + fraction * length[None, :]
        lower = (self._progress - self.cfg.max_backtrack_m).clamp_min(0.0)[:, None]
        upper = (self._progress + self.cfg.max_progress_jump_m).clamp_max(float(self._route.length))[:, None]
        allowed = (arc >= lower) & (arc <= upper)
        distance_sq = torch.where(allowed, distance_sq, torch.full_like(distance_sq, torch.inf))
        index = distance_sq.argmin(dim=-1)
        chosen_arc = arc[torch.arange(self.num_envs, device=self.device), index]
        chosen_projection = projection[torch.arange(self.num_envs, device=self.device), index]
        chosen_segment = segments[index]
        signed_cross_track = (
            chosen_segment[:, 0] * (position[:, 1] - chosen_projection[:, 1])
            - chosen_segment[:, 1] * (position[:, 0] - chosen_projection[:, 0])
        ) / length[index]
        return chosen_arc, signed_cross_track

    def _route_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        position = self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]
        progress, cross_track = self._project_route(position)
        self._progress.copy_(progress)
        self._cross_track.copy_(cross_track)
        preview_s = progress[:, None] + self._preview_distances[None, :]
        target_xy, target_yaw, max_height, valid = self._sample_route(preview_s)
        delta = target_xy - position[:, None, :]
        quat = self._robot.data.root_quat_w
        robot_yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
        )
        c, s = torch.cos(robot_yaw), torch.sin(robot_yaw)
        local_x = c[:, None] * delta[..., 0] + s[:, None] * delta[..., 1]
        local_y = -s[:, None] * delta[..., 0] + c[:, None] * delta[..., 1]
        local_yaw = torch.atan2(torch.sin(target_yaw - robot_yaw[:, None]), torch.cos(target_yaw - robot_yaw[:, None]))
        preview = torch.stack((local_x, local_y, torch.sin(local_yaw), torch.cos(local_yaw), max_height, valid), dim=-1)
        final_delta = self._route_xy[-1][None, :] - position
        final_yaw = torch.atan2(torch.sin(self._route_yaw[-1] - robot_yaw), torch.cos(self._route_yaw[-1] - robot_yaw))
        final_pose = torch.stack((
            c * final_delta[:, 0] + s * final_delta[:, 1],
            -s * final_delta[:, 0] + c * final_delta[:, 1],
            torch.sin(final_yaw),
            torch.cos(final_yaw),
        ), dim=-1)
        current_max_height = self._sample_route(progress)[2]
        remaining = (self.max_episode_length - self.episode_length_buf).float() * self.step_dt
        tracking_error = torch.cat((
            self._robot.data.root_lin_vel_b[:, :2] - self._previous_command[:, :2],
            self._robot.data.root_ang_vel_b[:, 2:3] - self._previous_command[:, 2:3],
        ), dim=-1)
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        state = torch.cat((
            self._robot.data.root_lin_vel_b[:, :2],
            self._robot.data.root_ang_vel_b[:, 2:3],
            base_height[:, None],
            self._robot.data.projected_gravity_b[:, :2],
            tracking_error,
            self._normalize_command(self._previous_command),
            self._normalize_command(self._previous_previous_command),
        ), dim=-1)
        return preview, final_pose, remaining, current_max_height, state, cross_track

    def _get_observations(self) -> dict[str, torch.Tensor]:
        preview, final_pose, remaining, _, state, _ = self._route_features()
        normalized_time = (remaining / self.cfg.episode_length_s).clamp(0.0, 1.0)
        self._high_observation = torch.cat((preview.reshape(self.num_envs, -1), final_pose, normalized_time[:, None], state), dim=-1)
        return {"policy": self._high_observation}

    def _set_high_level_command(self, action: torch.Tensor) -> torch.Tensor:
        command = self._decode_action(action)
        self._older_command.copy_(self._previous_previous_command)
        self._previous_previous_command.copy_(self._previous_command)
        self._previous_command.copy_(command)
        self._commands[:, :3] = command[:, :3]
        return command

    def _apply_low_level_control(self, command: torch.Tensor, is_rendering: bool) -> None:
        self._update_base_push_wrench()
        proprio = proprio_from_env_tensors(
            self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs,
            self._robot.data.joint_vel[:, self._joint_ids],
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
        )
        self._actions = self._low_level.act(proprio, command)
        self._processed_actions = self.cfg.action_scale * self._actions + self._q_offset_action_and_obs
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
        """Execute one high-level action and its complete frozen-policy chunk."""
        command = self._set_high_level_command(action)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.high_level_decimation):
            self._apply_low_level_control(command, is_rendering)

        self.episode_length_buf += self.cfg.high_level_decimation
        self.common_step_counter += self.cfg.high_level_decimation
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.cfg.high_level_decimation * self.step_dt)
        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _get_rewards(self) -> torch.Tensor:
        _, final_pose, remaining, max_height, _, cross_track = self._route_features()
        progress_delta = self._progress - self._last_progress
        self._last_progress.copy_(self._progress)
        position_error = torch.linalg.vector_norm(final_pose[:, :2], dim=-1)
        yaw_error = torch.atan2(final_pose[:, 2], final_pose[:, 3])
        goal_reached = (position_error <= self.cfg.goal_position_tolerance_m) & (
            torch.abs(yaw_error) <= self.cfg.goal_yaw_tolerance_rad
        )
        terminal_window = remaining <= self.cfg.terminal_window_s
        normalized_final_error = torch.stack((
            final_pose[:, 0] / self.cfg.goal_position_tolerance_m,
            final_pose[:, 1] / self.cfg.goal_position_tolerance_m,
            yaw_error / self.cfg.goal_yaw_tolerance_rad,
        ), dim=-1)
        macro_dt = self.cfg.high_level_decimation * self.step_dt
        reward, terms = hierarchical_reward(
            progress=progress_delta,
            cross_track=cross_track,
            final_pose_error=normalized_final_error,
            terminal_window=terminal_window,
            timed_out=self.reset_time_outs,
            goal_reached=goal_reached,
            command_height=self._previous_command[:, 3],
            max_command_height=max_height,
            command=self._normalize_command(self._previous_command),
            previous_command=self._normalize_command(self._previous_previous_command),
            previous_previous_command=self._normalize_command(self._older_command),
            low_level_risk=torch.zeros(self.num_envs, device=self.device),
            fallen=self.reset_terminated,
            corridor_half_width=self.cfg.corridor_half_width_m,
            per_second_dt=macro_dt,
            weights=self.cfg.reward_weights,
        )
        self._episode_reward_sums += reward
        self.extras["log"] = {
            **{f"RewardsPerStep/{name}": float(value.mean()) for name, value in terms.items()},
            "Metrics/route_progress_m": float(self._progress.mean()),
            "Metrics/cross_track_m": float(torch.abs(cross_track).mean()),
            "Metrics/goal_reached": float(goal_reached.float().mean()),
            "Metrics/command_clearance_violation": float(
                torch.relu(self._previous_command[:, 3] - max_height).mean()
            ),
        }
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_low_level"):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._low_level.reset_envs(env_ids)
        # A history of all-zero physical commands would mean an invalid height
        # below the data range and create a fictitious first-step jump penalty.
        neutral = torch.as_tensor((0.0, 0.0, 0.0, self._bounds.high[3]), device=self.device)
        self._previous_command[env_ids] = neutral
        self._previous_previous_command[env_ids] = neutral
        self._older_command[env_ids] = neutral
        self._progress[env_ids] = 0.0
        self._last_progress[env_ids] = 0.0
        self._cross_track[env_ids] = 0.0
