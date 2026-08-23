"""Direct-action route task used by DPPO rollouts.

The environment contains no policy and no residual controller.  Its action is
the 12-dimensional joint command produced directly by the diffusion actor.
"""

from __future__ import annotations

import torch

from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
from scripts.diffusion_policy.train.data.obs_utils import proprio_from_env_tensors
from scripts.residual_diffusion_rl.routes import RouteBank, RouteState

from .rewards import (
    TaskRewardWeights,
    average_speed_error,
    path_task_reward,
    schedule_error_improvement,
    terminal_pose_potential,
)


CRITIC_FEATURE_DIM = 14


class DPPODiffusionEnvCfg(Solo12EnvCfg):
    """Task-only configuration; DPPO optimizer settings live outside Isaac Lab."""

    episode_length_s = 24.0
    route_points: int = 101
    route_length_m: float = 4.0
    height_segment_m: float = 0.8
    route_stage: int = 2
    route_speed_max_mps: float | None = None
    stratified_route_sampling: bool = True
    corridor_half_width_m: float = 0.60
    route_goal_tolerance_m: float = 0.15
    terminal_height_tolerance_m: float = 0.05
    terminal_yaw_tolerance_rad: float = 0.40
    terminal_mean_speed_tolerance_mps: float = 0.08
    terminal_overshoot_m: float = 0.50
    goal_horizon_steps: int = 100
    v_req_clip: float = 2.0
    reset_x_pos = 0.0
    reset_y_pos = 0.0
    reset_yaw = 0.0
    reset_base_lin_vel_range = (0.0, 0.0)
    reset_base_ang_vel_range = (0.0, 0.0)
    flexed_initial_joint_pos_noise_range = (0.0, 0.0)
    actuation_delay_range = (0, 0)
    enable_observation_corruption = False
    base_push_interval_range_s = (1.0e9, 1.0e9)
    events = None

    def __post_init__(self):
        super().__post_init__()
        self.validate_task()

    def validate_task(self) -> None:
        """Validate task fields, including values overridden after construction."""

        if self.route_stage not in (0, 1, 2):
            raise ValueError("route_stage must be 0, 1 or 2.")
        if self.route_speed_max_mps is not None and self.route_speed_max_mps < 0.2:
            raise ValueError("route_speed_max_mps must be at least 0.2 m/s when set.")
        if self.route_speed_max_mps is not None and self.route_speed_max_mps > self.v_req_clip:
            raise ValueError(
                "route_speed_max_mps cannot exceed v_req_clip: the actor would receive "
                "a clipped speed goal while the task still rewards the unclipped speed."
            )
        if self.route_goal_tolerance_m <= 0.0 or self.corridor_half_width_m <= self.route_goal_tolerance_m:
            raise ValueError("The corridor must be wider than the positive terminal tolerance.")


class DPPODiffusionEnv(Solo12Env):
    """Solo12 physics plus a finite path/pose/average-speed objective."""

    cfg: DPPODiffusionEnvCfg

    def __init__(self, cfg: DPPODiffusionEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra/CLI overrides are applied after ``__post_init__``. Revalidate
        # here so an impossible actor/reward contract cannot reach simulation.
        cfg.validate_task()
        super().__init__(cfg, render_mode, **kwargs)
        self._routes = RouteBank(
            self.num_envs,
            self.device,
            points=cfg.route_points,
            length_m=cfg.route_length_m,
            height_segment_m=cfg.height_segment_m,
        )
        self._reward_weights = TaskRewardWeights()
        self._route_state: RouteState | None = None
        self._goal = torch.zeros(self.num_envs, 12, device=self.device)
        self._task_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._base_contact_failure = torch.zeros_like(self._success)
        self._corridor_failure = torch.zeros_like(self._success)
        self._overshoot_failure = torch.zeros_like(self._success)
        self._last_mean_speed_error = torch.zeros(self.num_envs, device=self.device)
        self._last_terminal_pose_potential = torch.zeros(self.num_envs, device=self.device)
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._routes.reset(
            all_ids,
            cfg.route_stage,
            stratified=cfg.stratified_route_sampling,
            speed_max=cfg.route_speed_max_mps,
        )
        self._update_route_and_goal()

    def _local_position(self) -> torch.Tensor:
        return self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]

    def _robot_yaw(self) -> torch.Tensor:
        q = self._robot.data.root_quat_w
        return torch.atan2(
            2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1.0 - 2.0 * (q[:, 2].square() + q[:, 3].square()),
        )

    def _base_height(self) -> torch.Tensor:
        return self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]

    def _update_route_and_goal(self) -> RouteState:
        self._route_state = self._routes.update(self._local_position())
        self._goal = self._routes.geometric_goal(
            self._local_position(),
            self._robot_yaw(),
            horizon_s=self.cfg.goal_horizon_steps * self.step_dt,
            v_clip=self.cfg.v_req_clip,
        )
        return self._route_state

    def _mean_speed_error(self, state: RouteState) -> torch.Tensor:
        return average_speed_error(
            state.progress,
            self._task_step.float() * self.step_dt,
            self._routes.speed,
        )

    def get_proprioception(self) -> torch.Tensor:
        """The exact 30-D measurement convention used by Phase A."""

        return proprio_from_env_tensors(
            self._robot.data.joint_pos[:, self._joint_ids],
            self._robot.data.joint_vel[:, self._joint_ids],
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
        )

    def get_goal(self) -> torch.Tensor:
        if self._route_state is None:
            self._update_route_and_goal()
        return self._goal

    def get_critic_features(self) -> torch.Tensor:
        state = self._route_state if self._route_state is not None else self._update_route_and_goal()
        yaw_error = torch.atan2(
            torch.sin(state.tangent_yaw - self._robot_yaw()),
            torch.cos(state.tangent_yaw - self._robot_yaw()),
        )
        mean_speed_error = self._mean_speed_error(state)
        tangent_velocity = (
            self._robot.data.root_lin_vel_w[:, 0] * torch.cos(state.tangent_yaw)
            + self._robot.data.root_lin_vel_w[:, 1] * torch.sin(state.tangent_yaw)
        )
        elapsed_fraction = (
            self._task_step.float() / max(float(self.max_episode_length), 1.0)
        ).clamp(0.0, 1.0)
        return torch.cat(
            (
                (state.progress / self._routes.length.clamp_min(1.0e-6))[:, None],
                state.remaining_fraction[:, None],
                state.cross_track[:, None],
                (self._base_height() - state.target_height)[:, None],
                (tangent_velocity - self._routes.speed)[:, None],
                mean_speed_error[:, None],
                torch.sin(yaw_error)[:, None],
                torch.cos(yaw_error)[:, None],
                self._robot.data.root_lin_vel_b[:, 2:3],
                self._robot.data.projected_gravity_b[:, :2],
                state.terminal_distance[:, None],
                self._routes.speed[:, None],
                elapsed_fraction[:, None],
            ),
            dim=1,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._task_step += 1
        contact, timeout = super()._get_dones()
        state = self._update_route_and_goal()
        terminal_yaw_error = torch.atan2(
            torch.sin(self._routes.yaw[:, -1] - self._robot_yaw()),
            torch.cos(self._routes.yaw[:, -1] - self._robot_yaw()),
        )
        mean_speed_error = self._mean_speed_error(state)
        corridor = state.cross_track.abs() > self.cfg.corridor_half_width_m
        overshoot = state.near_terminal & (state.terminal_along_error > self.cfg.terminal_overshoot_m)
        success = (
            state.near_terminal
            & (state.terminal_distance <= self.cfg.route_goal_tolerance_m)
            & (terminal_yaw_error.abs() <= self.cfg.terminal_yaw_tolerance_rad)
            & (
                (self._base_height() - self._routes.height[:, -1]).abs()
                <= self.cfg.terminal_height_tolerance_m
            )
            & (mean_speed_error.abs() <= self.cfg.terminal_mean_speed_tolerance_mps)
            & ~contact
            & ~corridor
            & ~overshoot
        )
        self._success.copy_(success)
        self._base_contact_failure.copy_(contact)
        self._corridor_failure.copy_(corridor)
        self._overshoot_failure.copy_(overshoot)
        self._last_mean_speed_error.copy_(mean_speed_error)
        terminated = contact | corridor | overshoot | success
        return terminated, timeout & ~terminated

    def _get_rewards(self) -> torch.Tensor:
        # ``_reset_idx`` adds this after reward computation. Remove the prior
        # step's event here so consumers never count an episode twice.
        self.extras.pop("dppo_episode", None)
        state = self._route_state if self._route_state is not None else self._update_route_and_goal()
        yaw_error = torch.atan2(
            torch.sin(state.tangent_yaw - self._robot_yaw()),
            torch.cos(state.tangent_yaw - self._robot_yaw()),
        )
        timeout_failure = self.reset_time_outs & ~(
            self._success
            | self._base_contact_failure
            | self._corridor_failure
            | self._overshoot_failure
        )
        schedule_improvement = schedule_error_improvement(
            progress=state.progress,
            progress_delta=state.progress_delta,
            elapsed_s=self._task_step.float() * self.step_dt,
            desired_speed=self._routes.speed,
            route_length=self._routes.length,
            dt=self.step_dt,
        )
        terminal_yaw_error = torch.atan2(
            torch.sin(self._routes.yaw[:, -1] - self._robot_yaw()),
            torch.cos(self._routes.yaw[:, -1] - self._robot_yaw()),
        )
        current_terminal_potential = terminal_pose_potential(
            remaining_distance=state.remaining_distance,
            actor_lookahead_distance=(
                self._routes.speed * self.cfg.goal_horizon_steps * self.step_dt
            ),
            terminal_distance=state.terminal_distance,
            terminal_yaw_error=terminal_yaw_error,
            terminal_height_error=self._base_height() - self._routes.height[:, -1],
            weights=self._reward_weights,
        )
        terminal_pose_improvement = (
            current_terminal_potential - self._last_terminal_pose_potential
        )
        self._last_terminal_pose_potential.copy_(current_terminal_potential)
        reward, terms = path_task_reward(
            progress_delta=state.progress_delta,
            schedule_improvement=schedule_improvement,
            terminal_pose_improvement=terminal_pose_improvement,
            route_length=self._routes.length,
            maximum_schedule_distance=(
                self._routes.speed * float(self.max_episode_length) * self.step_dt
            ),
            cross_track=state.cross_track,
            yaw_error=yaw_error,
            height_error=self._base_height() - state.target_height,
            projected_gravity_xy=self._robot.data.projected_gravity_b[:, :2],
            vertical_velocity=self._robot.data.root_lin_vel_b[:, 2],
            success=self._success,
            timeout_failure=timeout_failure,
            corridor_failure=self._corridor_failure,
            overshoot_failure=self._overshoot_failure,
            fell=self._base_contact_failure,
            weights=self._reward_weights,
        )
        self._episode_reward_sums += reward
        self.extras["log"] = {
            **{f"RewardsPerStep/{name}": float(value.mean()) for name, value in terms.items()},
            "Metrics/progress_fraction": float((state.progress / self._routes.length).mean()),
            "Metrics/cross_track_abs_m": float(state.cross_track.abs().mean()),
            "Metrics/height_error_abs_m": float((self._base_height() - state.target_height).abs().mean()),
            "Metrics/mean_speed_error_abs_mps": float(self._last_mean_speed_error.abs().mean()),
            "Metrics/terminal_distance_m": float(state.terminal_distance.mean()),
            "Metrics/success": float(self._success.float().mean()),
        }
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if hasattr(self, "_routes"):
            self._update_route_and_goal()
        return super()._get_observations()

    def reset_after_action_chunk(self, env_ids: torch.Tensor) -> None:
        """Discard post-terminal padding steps forced by vectorized chunking.

        Isaac Lab auto-resets an environment immediately, while the other
        vectorized environments may still have actions left in their chunk.
        Re-resetting only early-finished environments at the chunk boundary
        prevents the next policy decision from starting after unobserved
        stand-action steps in a new episode.
        """

        if len(env_ids) == 0:
            return
        self.extras.pop("dppo_episode", None)
        # Suppress a second, artificial episode record for the discarded
        # padding fragment. The real terminal event was already consumed.
        self.episode_length_buf[env_ids] = 0
        self._reset_idx(env_ids)
        self._update_route_and_goal()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "_routes"):
            completed = self.episode_length_buf[env_ids] > 0
            completed_ids = env_ids[completed]
            episode_snapshot = None
            episode_event = None
            if len(completed_ids) > 0 and self._route_state is not None:
                episode_snapshot = {
                    "success": float(self._success[completed_ids].float().mean()),
                    "base_contact": float(self._base_contact_failure[completed_ids].float().mean()),
                    "corridor_failure": float(self._corridor_failure[completed_ids].float().mean()),
                    "terminal_overshoot": float(self._overshoot_failure[completed_ids].float().mean()),
                    "mean_speed_error_abs_mps": float(self._last_mean_speed_error[completed_ids].abs().mean()),
                    "terminal_distance_m": float(self._route_state.terminal_distance[completed_ids].mean()),
                    "progress_fraction": float(
                        (self._route_state.progress[completed_ids] / self._routes.length[completed_ids]).mean()
                    ),
                    "count": float(len(completed_ids)),
                }
                episode_event = {
                    "env_ids": completed_ids.clone(),
                    "success": self._success[completed_ids].float().clone(),
                    "base_contact": self._base_contact_failure[completed_ids].float().clone(),
                    "corridor_failure": self._corridor_failure[completed_ids].float().clone(),
                    "terminal_overshoot": self._overshoot_failure[completed_ids].float().clone(),
                    "mean_speed_error_abs_mps": self._last_mean_speed_error[completed_ids].abs().clone(),
                    "terminal_distance_m": self._route_state.terminal_distance[completed_ids].clone(),
                    "progress_fraction": (
                        self._route_state.progress[completed_ids]
                        / self._routes.length[completed_ids].clamp_min(1.0e-6)
                    ).clone(),
                }
        else:
            episode_snapshot = None
            episode_event = None
        super()._reset_idx(env_ids)
        if not hasattr(self, "_routes"):
            return
        if episode_snapshot is not None:
            log = self.extras.setdefault("log", {})
            for name, value in episode_snapshot.items():
                log[f"Episode/{name}"] = value
            self.extras["dppo_episode"] = episode_event
        self._routes.reset(
            env_ids,
            self.cfg.route_stage,
            stratified=self.cfg.stratified_route_sampling,
            speed_max=self.cfg.route_speed_max_mps,
        )
        self._task_step[env_ids] = 0
        self._success[env_ids] = False
        self._base_contact_failure[env_ids] = False
        self._corridor_failure[env_ids] = False
        self._overshoot_failure[env_ids] = False
        self._last_mean_speed_error[env_ids] = 0.0
        self._last_terminal_pose_potential[env_ids] = 0.0
        self._route_state = None
