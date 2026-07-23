"""IsaacLab environment for Phase B1 residual PPO."""

from __future__ import annotations

import torch

from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
from scripts.diffusion_policy.train.data.obs_utils import proprio_from_env_tensors

from .contracts import OBSERVATION_DIM, RESIDUAL_ACTION_DIM
from .frozen_policy import FrozenSpatialDiffusion
from .rewards import RewardWeights, progress_timing_errors, residual_reward
from .routes import RouteBank, RouteState


class ResidualDiffusionEnvCfg(Solo12EnvCfg):
    """Flat config fields keep Hydra overrides robust and serializable."""

    action_space = RESIDUAL_ACTION_DIM
    observation_space = OBSERVATION_DIM
    episode_length_s = 24.0
    spatial_diffusion_checkpoint: str = ""
    diffusion_inference_steps: int | None = 10
    diffusion_exec_horizon: int = 4
    # The prior already solves most of the task.  Ten percent leaves PPO enough
    # authority to correct it without cheaply replacing the learned gait.
    residual_scale: float = 0.10
    residual_action_clip: float = 1.0
    route_points: int = 101
    route_length_m: float = 4.0
    height_segment_m: float = 0.8
    corridor_half_width_m: float = 0.45
    route_goal_tolerance_m: float = 0.12
    terminal_height_tolerance_m: float = 0.04
    terminal_mean_speed_tolerance_mps: float = 0.05
    terminal_stop_speed_mps: float = 0.15
    terminal_brake_distance_m: float = 0.60
    terminal_overshoot_m: float = 0.50
    terminal_hold_steps: int = 5
    # B1 starts on its actual target distribution because the frozen prior is
    # already competent there.  The staged option remains an explicit ablation.
    route_stage: int = 2
    use_route_curriculum: bool = False
    stratified_route_sampling: bool = False
    curriculum_stage1_steps: int = 6_400
    curriculum_stage2_steps: int = 12_800
    reset_x_pos = 0.0
    reset_y_pos = 0.0
    reset_yaw = 0.0
    reset_base_lin_vel_range = (0.0, 0.0)
    reset_base_ang_vel_range = (0.0, 0.0)
    actuation_delay_range = (0, 0)
    enable_observation_corruption = False
    flexed_initial_joint_pos_noise_range = (0.0, 0.0)
    events = None

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = OBSERVATION_DIM
        self.state_space = 0


class ResidualDiffusionEnv(Solo12Env):
    """RL action is a bounded correction to the frozen diffusion action."""

    cfg: ResidualDiffusionEnvCfg

    def __init__(self, cfg: ResidualDiffusionEnvCfg, render_mode: str | None = None, **kwargs):
        if not cfg.spatial_diffusion_checkpoint:
            raise ValueError("env.spatial_diffusion_checkpoint is required")
        if not 0.0 <= cfg.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0, 1]")
        if cfg.route_stage not in (0, 1, 2):
            raise ValueError("route_stage must be 0, 1 or 2")
        if cfg.terminal_hold_steps < 1:
            raise ValueError("terminal_hold_steps must be positive")
        if cfg.terminal_brake_distance_m <= cfg.route_goal_tolerance_m:
            raise ValueError("terminal_brake_distance_m must exceed route_goal_tolerance_m")
        if cfg.terminal_overshoot_m <= cfg.route_goal_tolerance_m:
            raise ValueError("terminal_overshoot_m must exceed route_goal_tolerance_m")
        if cfg.curriculum_stage1_steps < 0 or cfg.curriculum_stage2_steps < cfg.curriculum_stage1_steps:
            raise ValueError("curriculum step thresholds must be ordered and non-negative")
        super().__init__(cfg, render_mode, **kwargs)
        self._routes = RouteBank(
            self.num_envs,
            self.device,
            points=cfg.route_points,
            length_m=cfg.route_length_m,
            height_segment_m=cfg.height_segment_m,
        )
        self._prior = FrozenSpatialDiffusion(
            cfg.spatial_diffusion_checkpoint,
            self.num_envs,
            self.device,
            inference_steps=cfg.diffusion_inference_steps,
            exec_horizon=cfg.diffusion_exec_horizon,
        )
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._routes.reset(
            all_ids,
            stage=self._curriculum_stage(),
            stratified=self.cfg.stratified_route_sampling,
        )
        self._prior.reset(all_ids)
        self._residual = torch.zeros(self.num_envs, RESIDUAL_ACTION_DIM, device=self.device)
        self._previous_residual = torch.zeros_like(self._residual)
        self._base_action = torch.zeros_like(self._residual)
        self._base_action_ready = False
        self._route_state: RouteState | None = None
        self._goal = torch.zeros(self.num_envs, 12, device=self.device)
        self._reward_weights = RewardWeights()
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._base_contact_termination = torch.zeros_like(self._success)
        self._corridor_termination = torch.zeros_like(self._success)
        self._terminal_overshoot_termination = torch.zeros_like(self._success)
        self._terminal_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_episode_route_kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_episode_outcome = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._last_episode_mean_speed_error = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_terminal_distance = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_terminal_along_error = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_progress_fraction = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_height_error = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_task_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # RSL-RL may offset episode_length_buf once at training startup to
        # stagger resets.  Task time must remain zero-based for average speed.
        self._task_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _curriculum_stage(self) -> int:
        if not self.cfg.use_route_curriculum:
            return self.cfg.route_stage
        if self.common_step_counter >= self.cfg.curriculum_stage2_steps:
            return 2
        if self.common_step_counter >= self.cfg.curriculum_stage1_steps:
            return 1
        return 0

    def _robot_yaw(self) -> torch.Tensor:
        q = self._robot.data.root_quat_w
        return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                           1.0 - 2.0 * (q[:, 2].square() + q[:, 3].square()))

    def _local_position(self) -> torch.Tensor:
        return self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]

    def _proprio(self) -> torch.Tensor:
        # Phase A was collected, trained and evaluated with measured absolute
        # joint positions.  The Solo12 RL observation convention subtracts the
        # action offset, but doing that here shifts the frozen prior several
        # standard deviations out of its training distribution.
        return proprio_from_env_tensors(
            self._robot.data.joint_pos[:, self._joint_ids],
            self._robot.data.joint_vel[:, self._joint_ids],
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
        )

    def _update_route(self) -> RouteState:
        self._route_state = self._routes.update(self._local_position())
        return self._route_state

    def _timing_errors(self, state: RouteState) -> tuple[torch.Tensor, torch.Tensor]:
        """Return schedule and mean progress-speed errors without a time command."""

        elapsed = self._task_step_buf.float() * self.step_dt
        return progress_timing_errors(
            progress=state.progress,
            route_length=self._routes.length,
            commanded_speed=self._routes.speed,
            elapsed_s=elapsed,
            min_dt=self.step_dt,
        )

    def _tangent_speed(self, state: RouteState) -> torch.Tensor:
        velocity_w = self._robot.data.root_lin_vel_w[:, :2]
        return velocity_w[:, 0] * torch.cos(state.tangent_yaw) + velocity_w[:, 1] * torch.sin(
            state.tangent_yaw
        )

    def _prepare_base_action(self) -> None:
        if self._base_action_ready:
            return
        state = self._update_route()
        self._goal = self._routes.geometric_goal(
            self._local_position(),
            self._robot_yaw(),
            horizon_s=self._prior.goal_horizon_steps * self.step_dt,
            v_clip=self._prior.v_clip,
        )
        self._base_action = self._prior.propose(self._proprio(), self._goal)
        self._base_action_ready = True

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prepare_base_action()
        self._previous_residual.copy_(self._residual)
        self._residual.copy_(actions.to(self.device).clamp(-self.cfg.residual_action_clip,
                                                           self.cfg.residual_action_clip))
        half_action_range = 0.5 * (self._prior.action_max - self._prior.action_min)
        residual_delta = self.cfg.residual_scale * half_action_range * self._residual
        final = self._base_action + residual_delta
        final = torch.maximum(torch.minimum(final, self._prior.action_max), self._prior.action_min)
        self._actions = final
        self._processed_actions = self.cfg.action_scale * final + self._q_offset_action_and_obs
        self._prior.record_executed(final)
        self._base_action_ready = False

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._prepare_base_action()
        state = self._route_state
        yaw_error = torch.atan2(torch.sin(state.tangent_yaw - self._robot_yaw()),
                                torch.cos(state.tangent_yaw - self._robot_yaw()))
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        tangent_speed = self._tangent_speed(state)
        schedule_error, mean_speed_error = self._timing_errors(state)
        route_features = torch.cat((
            self._robot.data.root_lin_vel_b[:, :2],
            self._robot.data.root_ang_vel_b[:, 2:3],
            state.cross_track[:, None],
            state.remaining_fraction[:, None],
            (base_height - state.target_height)[:, None],
            (tangent_speed - self._routes.speed)[:, None],
            torch.sin(yaw_error)[:, None],
            torch.cos(yaw_error)[:, None],
            schedule_error[:, None],
            mean_speed_error[:, None],
        ), dim=1)
        obs = torch.cat((self._proprio(), self._goal, self._base_action,
                         self._residual, route_features), dim=1)
        if obs.shape[1] != OBSERVATION_DIM:
            raise RuntimeError(f"B1 observation contract changed: {obs.shape[1]} != {OBSERVATION_DIM}")
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Solo12.step calls _get_dones immediately before _get_rewards. Reuse
        # that projection so progress_delta is not consumed twice.
        state = self._route_state if self._route_state is not None else self._update_route()
        yaw_error = torch.atan2(torch.sin(state.tangent_yaw - self._robot_yaw()),
                                torch.cos(state.tangent_yaw - self._robot_yaw()))
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        tangent_speed = self._tangent_speed(state)
        planar_speed = torch.linalg.vector_norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        schedule_error, mean_speed_error = self._timing_errors(state)
        # A route timeout is a task failure as well: without this terminal
        # penalty, an agent can avoid the completion bonus by lingering until
        # the horizon.  RSL-RL still receives the timeout flag separately for
        # correct value bootstrapping.
        # The runner randomizes only episode_length_buf at startup to stagger
        # reset times.  Those shortened first rollouts are neutral truncations,
        # not failures of the route policy.
        startup_truncation = self.reset_time_outs & (
            self.episode_length_buf > self._task_step_buf
        )
        failed = (
            self.reset_terminated | (self.reset_time_outs & ~startup_truncation)
        ) & ~self._success
        reward, terms = residual_reward(
            progress_delta=state.progress_delta,
            cross_track=state.cross_track,
            speed_error=tangent_speed - self._routes.speed,
            schedule_error=schedule_error,
            yaw_error=yaw_error,
            height_error=base_height - state.target_height,
            remaining_distance=state.remaining_distance,
            terminal_distance=state.terminal_distance,
            planar_speed=planar_speed,
            terminal_brake_distance=self.cfg.terminal_brake_distance_m,
            terminal_position_tolerance=self.cfg.route_goal_tolerance_m,
            terminal_stop_speed=self.cfg.terminal_stop_speed_mps,
            residual=self._residual,
            previous_residual=self._previous_residual,
            projected_gravity_xy=self._robot.data.projected_gravity_b[:, :2],
            vertical_velocity=self._robot.data.root_lin_vel_b[:, 2],
            success=self._success,
            failed=failed,
            fell=self._base_contact_termination,
            dt=self.step_dt,
            weights=self._reward_weights,
        )
        self._episode_reward_sums += reward
        self.extras["log"] = {
            **{f"RewardsPerStep/{name}": float(value.mean()) for name, value in terms.items()},
            "Metrics/progress_m": float(state.progress.mean()),
            "Metrics/cross_track_abs_m": float(state.cross_track.abs().mean()),
            "Metrics/height_error_abs_m": float((base_height - state.target_height).abs().mean()),
            "Metrics/speed_error_abs_mps": float((tangent_speed - self._routes.speed).abs().mean()),
            "Metrics/schedule_error_abs_m": float(schedule_error.abs().mean()),
            "Metrics/mean_speed_error_abs_mps": float(mean_speed_error.abs().mean()),
            "Metrics/terminal_distance_m": float(state.terminal_distance.mean()),
            "Metrics/terminal_along_error_m": float(state.terminal_along_error.mean()),
            "Metrics/terminal_region_fraction": float(
                (state.remaining_distance <= self.cfg.terminal_brake_distance_m).float().mean()
            ),
            "Metrics/local_goal_average_speed_mps": float(self._goal[:, -1].mean()),
            "Metrics/residual_rms": float(torch.sqrt(self._residual.square().mean())),
            "Metrics/residual_abs_mean": float(self._residual.abs().mean()),
            "Metrics/residual_saturation_frac": float((self._residual.abs() >= 0.95).float().mean()),
            "Metrics/success": float(self._success.float().mean()),
            "Curriculum/stage": float(self._curriculum_stage()),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._task_step_buf += 1
        contact_terminated, time_out = super()._get_dones()
        state = self._update_route()
        corridor_failure = state.cross_track.abs() > self.cfg.corridor_half_width_m
        terminal_overshoot = (
            state.near_terminal
            & (state.terminal_along_error > self.cfg.terminal_overshoot_m)
        )
        yaw_error = torch.atan2(torch.sin(state.tangent_yaw - self._robot_yaw()),
                                torch.cos(state.tangent_yaw - self._robot_yaw()))
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        _, mean_speed_error = self._timing_errors(state)
        planar_speed = torch.linalg.vector_norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        reached_terminal_pose = (
            state.near_terminal
            & (state.terminal_distance <= self.cfg.route_goal_tolerance_m)
            & (yaw_error.abs() <= 0.35)
            & ((base_height - state.target_height).abs() <= self.cfg.terminal_height_tolerance_m)
            & (mean_speed_error.abs() <= self.cfg.terminal_mean_speed_tolerance_mps)
            & (planar_speed <= self.cfg.terminal_stop_speed_mps)
        )
        self._terminal_hold_counter = torch.where(
            reached_terminal_pose,
            self._terminal_hold_counter + 1,
            torch.zeros_like(self._terminal_hold_counter),
        )
        # A contact or corridor failure on the terminal step is a failure, not
        # a successful completion.
        self._success = (
            (self._terminal_hold_counter >= self.cfg.terminal_hold_steps)
            & ~contact_terminated
            & ~corridor_failure
            & ~terminal_overshoot
        )
        self._base_contact_termination.copy_(contact_terminated)
        self._corridor_termination.copy_(corridor_failure)
        self._terminal_overshoot_termination.copy_(terminal_overshoot)
        terminated = (
            contact_terminated | corridor_failure | terminal_overshoot | self._success
        )
        # Terminal outcomes are mutually exclusive.  Only a genuine horizon
        # truncation receives RSL-RL's timeout bootstrapping treatment.
        return terminated, time_out & ~terminated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if hasattr(self, "_prior") and self._route_state is not None:
            capture_ids = (
                torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            )
            capture_ids = capture_ids[self.episode_length_buf[capture_ids] > 0]
            if len(capture_ids) > 0:
                _, mean_speed_error = self._timing_errors(self._route_state)
                self._last_episode_route_kind[capture_ids] = self._routes.route_kind[capture_ids]
                self._last_episode_mean_speed_error[capture_ids] = mean_speed_error[capture_ids]
                self._last_episode_terminal_distance[capture_ids] = self._route_state.terminal_distance[capture_ids]
                self._last_episode_terminal_along_error[capture_ids] = (
                    self._route_state.terminal_along_error[capture_ids]
                )
                self._last_episode_progress_fraction[capture_ids] = (
                    self._route_state.progress[capture_ids]
                    / self._routes.length[capture_ids].clamp_min(1.0e-6)
                )
                base_height = (
                    self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
                )
                self._last_episode_height_error[capture_ids] = (
                    base_height[capture_ids] - self._route_state.target_height[capture_ids]
                )
                self._last_episode_task_steps[capture_ids] = self._task_step_buf[capture_ids]
                outcome = torch.full_like(capture_ids, 3)
                outcome = torch.where(self._success[capture_ids], 0, outcome)
                outcome = torch.where(
                    self._terminal_overshoot_termination[capture_ids], 4, outcome
                )
                outcome = torch.where(self._corridor_termination[capture_ids], 2, outcome)
                outcome = torch.where(self._base_contact_termination[capture_ids], 1, outcome)
                self._last_episode_outcome[capture_ids] = outcome
        super()._reset_idx(env_ids)
        if not hasattr(self, "_prior"):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Solo12Env labels every non-timeout reset as ``base_contact``.  B1
        # introduces route success and corridor exits, so overwrite that
        # inherited aggregate with the mutually interpretable event counts.
        episode_log = self.extras.setdefault("log", {})
        base_count = torch.count_nonzero(self._base_contact_termination[env_ids]).item()
        corridor_count = torch.count_nonzero(self._corridor_termination[env_ids]).item()
        overshoot_count = torch.count_nonzero(
            self._terminal_overshoot_termination[env_ids]
        ).item()
        success_count = torch.count_nonzero(self._success[env_ids]).item()
        completed_count = max(len(env_ids), 1)
        episode_log["Episode_Termination/base_contact"] = base_count
        episode_log["Episode_Termination/corridor_failure"] = corridor_count
        episode_log["Episode_Termination/terminal_overshoot"] = overshoot_count
        episode_log["Episode_Termination/route_success"] = success_count
        episode_log["Episode_TerminationRate/base_contact"] = base_count / completed_count
        episode_log["Episode_TerminationRate/corridor_failure"] = corridor_count / completed_count
        episode_log["Episode_TerminationRate/terminal_overshoot"] = (
            overshoot_count / completed_count
        )
        episode_log["Episode_TerminationRate/route_success"] = success_count / completed_count
        episode_log["Episode_TerminationRate/time_out"] = (
            torch.count_nonzero(self.reset_time_outs[env_ids]).item() / completed_count
        )
        episode_log["Episode_Metrics/final_mean_speed_error_abs_mps"] = float(
            self._last_episode_mean_speed_error[env_ids].abs().mean()
        )
        episode_log["Episode_Metrics/final_position_error_m"] = float(
            self._last_episode_terminal_distance[env_ids].mean()
        )
        episode_log["Episode_Metrics/final_along_error_m"] = float(
            self._last_episode_terminal_along_error[env_ids].mean()
        )
        episode_log["Episode_Metrics/final_progress_fraction"] = float(
            self._last_episode_progress_fraction[env_ids].mean()
        )
        episode_log["Episode_Metrics/final_height_error_abs_m"] = float(
            self._last_episode_height_error[env_ids].abs().mean()
        )
        episode_log["Episode/length_steps"] = float(
            self._last_episode_task_steps[env_ids].float().mean()
        )
        episode_log["Episode/length_seconds"] = (
            episode_log["Episode/length_steps"] * self.step_dt
        )
        self._routes.reset(
            env_ids,
            self._curriculum_stage(),
            stratified=self.cfg.stratified_route_sampling,
        )
        self._prior.reset(env_ids)
        self._residual[env_ids] = 0.0
        self._previous_residual[env_ids] = 0.0
        self._success[env_ids] = False
        self._base_contact_termination[env_ids] = False
        self._corridor_termination[env_ids] = False
        self._terminal_overshoot_termination[env_ids] = False
        self._terminal_hold_counter[env_ids] = 0
        self._task_step_buf[env_ids] = 0
        self._route_state = None
        self._base_action_ready = False
