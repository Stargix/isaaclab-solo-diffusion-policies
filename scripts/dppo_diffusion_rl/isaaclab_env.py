"""Direct-action route task used by DPPO rollouts.

The environment contains no policy and no residual controller.  Its action is
the 12-dimensional joint command produced directly by the diffusion actor.
"""

from __future__ import annotations

import torch

from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
from scripts.diffusion_policy.train.conditioning.goal_builder import goal_dimension
from scripts.diffusion_policy.train.data.obs_utils import proprio_from_env_tensors
from scripts.residual_diffusion_rl.routes import RouteBank, RouteState

from .conditioning import remaining_speed_budget
from .hybrid_routes import HYBRID_ROUTE_CONTRACT_VERSION, SupportedHybridRouteBank
from .procedural_routes import SupportedProceduralRouteBank
from .route_feasibility import FEASIBILITY_CONTRACT_VERSION
from .supported_hybrid_v3 import (
    HYBRID_V3_ROUTE_CONTRACT_VERSION,
    SupportedHybridV3RouteBank,
)
from .rewards import (
    TaskRewardWeights,
    average_speed_error,
    distance_weighted_rmse,
    path_task_reward,
    schedule_error_improvement,
    terminal_pose_potential,
)


CRITIC_FEATURE_DIM = 14
SUPPORTED_ROUTE_DISTRIBUTIONS = {
    "supported_procedural_v1",
    "supported_hybrid_v2",
    "supported_hybrid_v3",
}


class DPPODiffusionEnvCfg(Solo12EnvCfg):
    """Task-only configuration; DPPO optimizer settings live outside Isaac Lab."""

    episode_length_s = 24.0
    route_points: int = 101
    route_length_m: float = 4.0
    height_segment_m: float = 0.8
    route_distribution: str = "legacy"
    procedural_curvature_knots: int = 6
    procedural_max_curvature_rad_m: float = 0.8
    hybrid_route_contract_version: int = HYBRID_ROUTE_CONTRACT_VERSION
    hybrid_v3_route_contract_version: int = HYBRID_V3_ROUTE_CONTRACT_VERSION
    feasibility_contract_version: int = FEASIBILITY_CONTRACT_VERSION
    transition_boundary_min_m: float = 1.6
    transition_boundary_max_m: float = 2.4
    transition_margin_m: float = 0.25
    goal_representation: str = "hindsight_geom_avg12"
    route_stage: int = 2
    # None preserves the historical route-stage coupling. Set to 2 to expose
    # all four height levels while keeping a simpler geometry curriculum.
    height_profile_stage: int | None = None
    route_speed_max_mps: float | None = None
    stratified_route_sampling: bool = True
    corridor_half_width_m: float = 0.60
    route_goal_tolerance_m: float = 0.15
    terminal_height_tolerance_m: float = 0.05
    # Schema-8 only: the task includes the complete route-height profile, not
    # just the terminal posture.  Both quantities are distance-weighted so
    # waiting cannot improve or worsen the result.
    profile_height_mae_tolerance_m: float = 0.04
    profile_height_reward_weight: float = 2.0
    path_reward_weight: float = 1.25
    # ``None`` preserves every historical task contract. V3 enables a
    # distance-weighted full-route precision gate explicitly.
    route_cte_rmse_tolerance_m: float | None = None
    terminal_yaw_tolerance_rad: float = 0.40
    terminal_mean_speed_tolerance_mps: float = 0.08
    terminal_overshoot_m: float = 0.50
    goal_horizon_steps: int = 100
    v_req_clip: float = 2.0
    # Keep headroom above the stage-2 nominal maximum (0.6 m/s): a delayed
    # route at the upper nominal speed must be allowed to recover time.
    speed_budget_max_mps: float = 0.8
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
        valid_distributions = {"legacy", *SUPPORTED_ROUTE_DISTRIBUTIONS}
        if self.route_distribution not in valid_distributions:
            raise ValueError(
                "route_distribution must be 'legacy', 'supported_procedural_v1', "
                "'supported_hybrid_v2' or 'supported_hybrid_v3'."
            )
        if self.height_profile_stage is not None and self.height_profile_stage not in (0, 1, 2):
            raise ValueError("height_profile_stage must be 0, 1, 2 or None.")
        if self.goal_representation not in {
            "hindsight_geom_avg12",
            "hindsight_geom_profile16",
        }:
            raise ValueError(
                "DPPO supports only hindsight_geom_avg12 and hindsight_geom_profile16."
            )
        if self.route_speed_max_mps is not None and self.route_speed_max_mps < 0.2:
            raise ValueError("route_speed_max_mps must be at least 0.2 m/s when set.")
        if self.route_speed_max_mps is not None and self.route_speed_max_mps > self.v_req_clip:
            raise ValueError(
                "route_speed_max_mps cannot exceed v_req_clip: the actor would receive "
                "a clipped speed goal while the task still rewards the unclipped speed."
            )
        if not 0.0 < self.speed_budget_max_mps <= self.v_req_clip:
            raise ValueError("speed_budget_max_mps must be in (0, v_req_clip].")
        sampled_speed_max = self.route_speed_max_mps
        if sampled_speed_max is None:
            sampled_speed_max = (
                0.8
                if self.route_distribution in SUPPORTED_ROUTE_DISTRIBUTIONS
                else 0.4 if self.route_stage <= 0 else 0.5 if self.route_stage == 1 else 0.6
            )
        if sampled_speed_max > self.speed_budget_max_mps:
            raise ValueError(
                "speed_budget_max_mps must cover the maximum sampled route speed."
            )
        if self.route_goal_tolerance_m <= 0.0 or self.corridor_half_width_m <= self.route_goal_tolerance_m:
            raise ValueError("The corridor must be wider than the positive terminal tolerance.")
        if self.profile_height_mae_tolerance_m <= 0.0:
            raise ValueError("profile_height_mae_tolerance_m must be positive.")
        if self.profile_height_reward_weight <= 0.0:
            raise ValueError("profile_height_reward_weight must be positive.")
        if self.path_reward_weight <= 0.0:
            raise ValueError("path_reward_weight must be positive.")
        if (
            self.route_cte_rmse_tolerance_m is not None
            and self.route_cte_rmse_tolerance_m <= 0.0
        ):
            raise ValueError("route_cte_rmse_tolerance_m must be positive when set.")
        if self.route_distribution in SUPPORTED_ROUTE_DISTRIBUTIONS:
            distribution_name = self.route_distribution
            if self.goal_representation != "hindsight_geom_profile16":
                raise ValueError(
                    f"{distribution_name} requires the schema-8 height-profile actor."
                )
            if self.height_profile_stage is not None:
                raise ValueError(
                    f"{distribution_name} owns its support-aware height distribution; "
                    "do not combine it with --height_profile_stage."
                )
            if self.route_length_m != 4.0:
                raise ValueError(
                    f"{distribution_name} is preregistered for 4.0 m routes."
                )
            if sampled_speed_max > 1.2:
                raise ValueError(
                    f"{distribution_name} is bounded by the audited 1.2 m/s fast envelope."
                )
            if not (
                0.0
                < self.transition_boundary_min_m
                <= self.transition_boundary_max_m
                < self.route_length_m
            ):
                raise ValueError("Invalid supported-profile transition boundary range.")
            if self.transition_margin_m < 0.0:
                raise ValueError("transition_margin_m must be non-negative.")
        if (
            self.route_distribution == "supported_hybrid_v2"
            and self.hybrid_route_contract_version != HYBRID_ROUTE_CONTRACT_VERSION
        ):
            raise ValueError(
                "supported_hybrid_v2 requires hybrid route contract version "
                f"{HYBRID_ROUTE_CONTRACT_VERSION}."
            )
        if self.route_distribution == "supported_hybrid_v3":
            if (
                self.hybrid_v3_route_contract_version
                != HYBRID_V3_ROUTE_CONTRACT_VERSION
            ):
                raise ValueError(
                    "supported_hybrid_v3 requires route contract version "
                    f"{HYBRID_V3_ROUTE_CONTRACT_VERSION}."
                )
            if self.feasibility_contract_version != FEASIBILITY_CONTRACT_VERSION:
                raise ValueError(
                    "supported_hybrid_v3 requires feasibility contract version "
                    f"{FEASIBILITY_CONTRACT_VERSION}."
                )
            if self.route_cte_rmse_tolerance_m is None:
                raise ValueError(
                    "supported_hybrid_v3 requires an explicit route CTE RMSE tolerance."
                )


class DPPODiffusionEnv(Solo12Env):
    """Solo12 physics plus a finite path/pose/average-speed objective."""

    cfg: DPPODiffusionEnvCfg

    def __init__(self, cfg: DPPODiffusionEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra/CLI overrides are applied after ``__post_init__``. Revalidate
        # here so an impossible actor/reward contract cannot reach simulation.
        cfg.validate_task()
        super().__init__(cfg, render_mode, **kwargs)
        if cfg.route_distribution == "supported_hybrid_v3":
            self._routes = SupportedHybridV3RouteBank(
                self.num_envs,
                self.device,
                points=cfg.route_points,
                length_m=cfg.route_length_m,
                curvature_knots=cfg.procedural_curvature_knots,
                max_curvature_rad_m=cfg.procedural_max_curvature_rad_m,
                transition_boundary_min_m=cfg.transition_boundary_min_m,
                transition_boundary_max_m=cfg.transition_boundary_max_m,
                transition_margin_m=cfg.transition_margin_m,
            )
        elif cfg.route_distribution == "supported_hybrid_v2":
            self._routes = SupportedHybridRouteBank(
                self.num_envs,
                self.device,
                points=cfg.route_points,
                length_m=cfg.route_length_m,
                curvature_knots=cfg.procedural_curvature_knots,
                max_curvature_rad_m=cfg.procedural_max_curvature_rad_m,
                transition_boundary_min_m=cfg.transition_boundary_min_m,
                transition_boundary_max_m=cfg.transition_boundary_max_m,
                transition_margin_m=cfg.transition_margin_m,
            )
        elif cfg.route_distribution == "supported_procedural_v1":
            self._routes = SupportedProceduralRouteBank(
                self.num_envs,
                self.device,
                points=cfg.route_points,
                length_m=cfg.route_length_m,
                curvature_knots=cfg.procedural_curvature_knots,
                max_curvature_rad_m=cfg.procedural_max_curvature_rad_m,
                transition_boundary_min_m=cfg.transition_boundary_min_m,
                transition_boundary_max_m=cfg.transition_boundary_max_m,
                transition_margin_m=cfg.transition_margin_m,
            )
        else:
            self._routes = RouteBank(
                self.num_envs,
                self.device,
                points=cfg.route_points,
                length_m=cfg.route_length_m,
                height_segment_m=cfg.height_segment_m,
            )
        reward_height = (
            cfg.profile_height_reward_weight
            if cfg.goal_representation == "hindsight_geom_profile16"
            else TaskRewardWeights().height
        )
        self._reward_weights = TaskRewardWeights(
            path=cfg.path_reward_weight,
            height=reward_height,
        )
        self._route_state: RouteState | None = None
        self._goal = torch.zeros(
            self.num_envs,
            goal_dimension(cfg.goal_representation),
            device=self.device,
        )
        self._task_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._position_arrival = torch.zeros_like(self._success)
        self._terminal_yaw_success = torch.zeros_like(self._success)
        self._terminal_height_success = torch.zeros_like(self._success)
        self._terminal_mean_speed_success = torch.zeros_like(self._success)
        self._arrival_failure = torch.zeros_like(self._success)
        self._time_out_failure = torch.zeros_like(self._success)
        self._base_contact_failure = torch.zeros_like(self._success)
        self._corridor_failure = torch.zeros_like(self._success)
        self._overshoot_failure = torch.zeros_like(self._success)
        self._last_mean_speed_error = torch.zeros(self.num_envs, device=self.device)
        self._last_terminal_pose_potential = torch.zeros(self.num_envs, device=self.device)
        self._profile_height_error_distance_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._profile_height_within_tolerance_distance_sum = torch.zeros_like(
            self._profile_height_error_distance_sum
        )
        self._profile_height_distance_sum = torch.zeros_like(
            self._profile_height_error_distance_sum
        )
        self._last_profile_height_mae = torch.zeros_like(
            self._profile_height_error_distance_sum
        )
        self._last_profile_height_within_tolerance = torch.zeros_like(
            self._profile_height_error_distance_sum
        )
        self._profile_height_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._route_cte_sq_distance_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._route_cte_distance_sum = torch.zeros_like(
            self._route_cte_sq_distance_sum
        )
        self._last_route_cte_rmse = torch.zeros_like(
            self._route_cte_sq_distance_sum
        )
        self._route_cte_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_routes(all_ids)
        self._update_route_and_goal()

    def _reset_routes(self, env_ids: torch.Tensor) -> None:
        if self.cfg.route_distribution in SUPPORTED_ROUTE_DISTRIBUTIONS:
            self._routes.reset(
                env_ids,
                stratified=self.cfg.stratified_route_sampling,
                speed_max=(
                    float(self.cfg.route_speed_max_mps)
                    if self.cfg.route_speed_max_mps is not None
                    else 0.8
                ),
            )
            return
        self._routes.reset(
            env_ids,
            self.cfg.route_stage,
            stratified=self.cfg.stratified_route_sampling,
            speed_max=self.cfg.route_speed_max_mps,
            height_stage=self.cfg.height_profile_stage,
        )

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
        goal_builder = (
            self._routes.geometric_height_profile_goal
            if self.cfg.goal_representation == "hindsight_geom_profile16"
            else self._routes.geometric_goal
        )
        self._goal = goal_builder(
            self._local_position(),
            self._robot_yaw(),
            horizon_s=self.cfg.goal_horizon_steps * self.step_dt,
            v_clip=self.cfg.v_req_clip,
        )
        # Preserve the selected Phase-A goal layout and its nominal geometric
        # preview.  Only the final v_avg scalar becomes closed-loop: it is the
        # pace required to meet the route-level first-arrival time from the
        # current progress.  This makes accumulated timing debt observable to
        # the actor without imposing an instantaneous velocity controller.
        self._goal[:, -1] = remaining_speed_budget(
            remaining_distance=self._route_state.remaining_distance,
            route_length=self._routes.length,
            desired_mean_speed=self._routes.speed,
            elapsed_s=self._task_step.float() * self.step_dt,
            max_speed=self.cfg.speed_budget_max_mps,
            min_remaining_time_s=self.step_dt,
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
        height_abs_error = (self._base_height() - state.target_height).abs()
        height_tracking_weight = getattr(
            self._routes,
            "height_tracking_weight",
            torch.ones_like(state.progress),
        )
        height_distance = state.progress_delta.clamp_min(0.0) * height_tracking_weight
        self._profile_height_error_distance_sum += height_abs_error * height_distance
        self._profile_height_within_tolerance_distance_sum += (
            height_abs_error <= self.cfg.profile_height_mae_tolerance_m
        ).float() * height_distance
        self._profile_height_distance_sum += height_distance
        has_profile_distance = self._profile_height_distance_sum > 1.0e-6
        profile_denominator = self._profile_height_distance_sum.clamp_min(1.0e-6)
        self._last_profile_height_mae.copy_(
            torch.where(
                has_profile_distance,
                self._profile_height_error_distance_sum / profile_denominator,
                torch.zeros_like(profile_denominator),
            )
        )
        self._last_profile_height_within_tolerance.copy_(
            torch.where(
                has_profile_distance,
                self._profile_height_within_tolerance_distance_sum
                / profile_denominator,
                torch.zeros_like(profile_denominator),
            )
        )
        profile_height_ok = has_profile_distance & (
            self._last_profile_height_mae
            <= self.cfg.profile_height_mae_tolerance_m
        )
        cte_distance = state.progress_delta.clamp_min(0.0)
        self._route_cte_sq_distance_sum += state.cross_track.square() * cte_distance
        self._route_cte_distance_sum += cte_distance
        has_cte_distance = self._route_cte_distance_sum > 1.0e-6
        self._last_route_cte_rmse.copy_(
            distance_weighted_rmse(
                self._route_cte_sq_distance_sum,
                self._route_cte_distance_sum,
            )
        )
        cte_constraint_active = self.cfg.route_cte_rmse_tolerance_m is not None
        route_cte_ok = (
            has_cte_distance
            & (
                self._last_route_cte_rmse
                <= float(self.cfg.route_cte_rmse_tolerance_m)
            )
            if cte_constraint_active
            else torch.ones_like(has_cte_distance)
        )
        terminal_yaw_error = torch.atan2(
            torch.sin(self._routes.yaw[:, -1] - self._robot_yaw()),
            torch.cos(self._routes.yaw[:, -1] - self._robot_yaw()),
        )
        elapsed_s = self._task_step.float() * self.step_dt
        mean_speed_error = self._mean_speed_error(state)
        corridor = state.cross_track.abs() > self.cfg.corridor_half_width_m
        overshoot = state.near_terminal & (state.terminal_along_error > self.cfg.terminal_overshoot_m)
        arrival = state.near_terminal & (
            state.terminal_distance <= self.cfg.route_goal_tolerance_m
        )
        # The timing objective is route length divided by first-arrival time.
        # Using the full route length avoids redefining the task as a shorter
        # route merely because the goal is represented by a tolerance region.
        arrival_speed_error = (
            self._routes.length / elapsed_s.clamp_min(self.step_dt) - self._routes.speed
        )
        terminal_yaw_ok = (
            terminal_yaw_error.abs() <= self.cfg.terminal_yaw_tolerance_rad
        )
        terminal_height_ok = (
            (self._base_height() - self._routes.height[:, -1]).abs()
            <= self.cfg.terminal_height_tolerance_m
        )
        terminal_mean_speed_ok = (
            arrival_speed_error.abs() <= self.cfg.terminal_mean_speed_tolerance_mps
        )
        profile_constraint_active = (
            self.cfg.goal_representation == "hindsight_geom_profile16"
        )
        success = (
            arrival
            & terminal_yaw_ok
            & terminal_height_ok
            & terminal_mean_speed_ok
            & (profile_height_ok if profile_constraint_active else True)
            & route_cte_ok
            & ~contact
            & ~corridor
            & ~overshoot
        )
        self._profile_height_success.copy_(arrival & profile_height_ok)
        self._route_cte_success.copy_(arrival & route_cte_ok)
        arrival_failure = arrival & ~success & ~contact & ~corridor & ~overshoot
        terminated = contact | corridor | overshoot | arrival
        time_out_failure = timeout & ~terminated
        self._success.copy_(success)
        self._position_arrival.copy_(arrival)
        self._terminal_yaw_success.copy_(arrival & terminal_yaw_ok)
        self._terminal_height_success.copy_(arrival & terminal_height_ok)
        self._terminal_mean_speed_success.copy_(arrival & terminal_mean_speed_ok)
        self._arrival_failure.copy_(arrival_failure)
        self._time_out_failure.copy_(time_out_failure)
        self._base_contact_failure.copy_(contact)
        self._corridor_failure.copy_(corridor)
        self._overshoot_failure.copy_(overshoot)
        self._last_mean_speed_error.copy_(
            torch.where(arrival, arrival_speed_error, mean_speed_error)
        )
        return terminated, time_out_failure

    def _get_rewards(self) -> torch.Tensor:
        # ``_reset_idx`` adds this after reward computation. Remove the prior
        # step's event here so consumers never count an episode twice.
        self.extras.pop("dppo_episode", None)
        state = self._route_state if self._route_state is not None else self._update_route_and_goal()
        yaw_error = torch.atan2(
            torch.sin(state.tangent_yaw - self._robot_yaw()),
            torch.cos(state.tangent_yaw - self._robot_yaw()),
        )
        timeout_failure = self._time_out_failure
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
            # A discontinuous spatial requirement cannot be followed
            # instantaneously. The procedural contract scores the surrounding
            # plateaus and leaves a small boundary interval for a learned
            # physical transition; it does not blend or override actions.
            height_error=(self._base_height() - state.target_height)
            * getattr(
                self._routes,
                "height_tracking_weight",
                torch.ones_like(state.progress),
            ),
            projected_gravity_xy=self._robot.data.projected_gravity_b[:, :2],
            vertical_velocity=self._robot.data.root_lin_vel_b[:, 2],
            success=self._success,
            arrival_failure=self._arrival_failure,
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
            "Metrics/profile_height_mae_m": float(self._last_profile_height_mae.mean()),
            "Metrics/profile_height_within_tolerance_fraction": float(
                self._last_profile_height_within_tolerance.mean()
            ),
            "Metrics/route_cte_rmse_m": float(self._last_route_cte_rmse.mean()),
            "Metrics/desired_mean_speed_mps": float(self._routes.speed.mean()),
            "Metrics/tangent_speed_mps": float(
                (
                    torch.cos(state.tangent_yaw) * self._robot.data.root_lin_vel_w[:, 0]
                    + torch.sin(state.tangent_yaw) * self._robot.data.root_lin_vel_w[:, 1]
                ).mean()
            ),
            "Metrics/transition_fraction": float(
                (getattr(self._routes, "profile_class", torch.zeros_like(self._task_step)) >= 2)
                .float()
                .mean()
            ),
            "Metrics/mean_speed_error_abs_mps": float(self._last_mean_speed_error.abs().mean()),
            "Metrics/terminal_distance_m": float(state.terminal_distance.mean()),
            "Metrics/success": float(self._success.float().mean()),
            "Metrics/arrival_failure": float(self._arrival_failure.float().mean()),
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
                    "position_arrival": float(
                        self._position_arrival[completed_ids].float().mean()
                    ),
                    "terminal_yaw_success": float(
                        self._terminal_yaw_success[completed_ids].float().mean()
                    ),
                    "terminal_height_success": float(
                        self._terminal_height_success[completed_ids].float().mean()
                    ),
                    "terminal_mean_speed_success": float(
                        self._terminal_mean_speed_success[completed_ids].float().mean()
                    ),
                    "arrival_failure": float(self._arrival_failure[completed_ids].float().mean()),
                    "time_out": float(self._time_out_failure[completed_ids].float().mean()),
                    "base_contact": float(self._base_contact_failure[completed_ids].float().mean()),
                    "corridor_failure": float(self._corridor_failure[completed_ids].float().mean()),
                    "terminal_overshoot": float(self._overshoot_failure[completed_ids].float().mean()),
                    "mean_speed_error_abs_mps": float(self._last_mean_speed_error[completed_ids].abs().mean()),
                    "terminal_distance_m": float(self._route_state.terminal_distance[completed_ids].mean()),
                    "progress_fraction": float(
                        (self._route_state.progress[completed_ids] / self._routes.length[completed_ids]).mean()
                    ),
                    "profile_height_constraint_active": float(
                        self.cfg.goal_representation == "hindsight_geom_profile16"
                    ),
                    "profile_height_success": float(
                        self._profile_height_success[completed_ids].float().mean()
                    ),
                    "profile_height_mae_m": float(
                        self._last_profile_height_mae[completed_ids].mean()
                    ),
                    "profile_height_within_tolerance_fraction": float(
                        self._last_profile_height_within_tolerance[completed_ids].mean()
                    ),
                    "route_cte_constraint_active": float(
                        self.cfg.route_cte_rmse_tolerance_m is not None
                    ),
                    "route_cte_success": float(
                        self._route_cte_success[completed_ids].float().mean()
                    ),
                    "route_cte_rmse_m": float(
                        self._last_route_cte_rmse[completed_ids].mean()
                    ),
                    "route_cte_tolerance_m": float(
                        self.cfg.route_cte_rmse_tolerance_m or 0.0
                    ),
                    "count": float(len(completed_ids)),
                }
                episode_event = {
                    "env_ids": completed_ids.clone(),
                    "success": self._success[completed_ids].float().clone(),
                    "position_arrival": self._position_arrival[
                        completed_ids
                    ].float().clone(),
                    "terminal_yaw_success": self._terminal_yaw_success[
                        completed_ids
                    ].float().clone(),
                    "terminal_height_success": self._terminal_height_success[
                        completed_ids
                    ].float().clone(),
                    "terminal_mean_speed_success": self._terminal_mean_speed_success[
                        completed_ids
                    ].float().clone(),
                    "arrival_failure": self._arrival_failure[completed_ids].float().clone(),
                    "time_out": self._time_out_failure[completed_ids].float().clone(),
                    "base_contact": self._base_contact_failure[completed_ids].float().clone(),
                    "corridor_failure": self._corridor_failure[completed_ids].float().clone(),
                    "terminal_overshoot": self._overshoot_failure[completed_ids].float().clone(),
                    "mean_speed_error_abs_mps": self._last_mean_speed_error[completed_ids].abs().clone(),
                    "terminal_distance_m": self._route_state.terminal_distance[completed_ids].clone(),
                    "progress_fraction": (
                        self._route_state.progress[completed_ids]
                        / self._routes.length[completed_ids].clamp_min(1.0e-6)
                    ).clone(),
                    "profile_height_constraint_active": torch.full(
                        (len(completed_ids),),
                        float(self.cfg.goal_representation == "hindsight_geom_profile16"),
                        device=self.device,
                    ),
                    "profile_height_success": self._profile_height_success[
                        completed_ids
                    ].float().clone(),
                    "profile_height_mae_m": self._last_profile_height_mae[
                        completed_ids
                    ].clone(),
                    "profile_height_within_tolerance_fraction": (
                        self._last_profile_height_within_tolerance[completed_ids].clone()
                    ),
                    "route_cte_constraint_active": torch.full(
                        (len(completed_ids),),
                        float(self.cfg.route_cte_rmse_tolerance_m is not None),
                        device=self.device,
                    ),
                    "route_cte_success": self._route_cte_success[
                        completed_ids
                    ].float().clone(),
                    "route_cte_rmse_m": self._last_route_cte_rmse[
                        completed_ids
                    ].clone(),
                    "route_cte_tolerance_m": torch.full(
                        (len(completed_ids),),
                        float(self.cfg.route_cte_rmse_tolerance_m or 0.0),
                        device=self.device,
                    ),
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
        self._reset_routes(env_ids)
        self._task_step[env_ids] = 0
        self._success[env_ids] = False
        self._position_arrival[env_ids] = False
        self._terminal_yaw_success[env_ids] = False
        self._terminal_height_success[env_ids] = False
        self._terminal_mean_speed_success[env_ids] = False
        self._arrival_failure[env_ids] = False
        self._time_out_failure[env_ids] = False
        self._base_contact_failure[env_ids] = False
        self._corridor_failure[env_ids] = False
        self._overshoot_failure[env_ids] = False
        self._last_mean_speed_error[env_ids] = 0.0
        self._last_terminal_pose_potential[env_ids] = 0.0
        self._profile_height_error_distance_sum[env_ids] = 0.0
        self._profile_height_within_tolerance_distance_sum[env_ids] = 0.0
        self._profile_height_distance_sum[env_ids] = 0.0
        self._last_profile_height_mae[env_ids] = 0.0
        self._last_profile_height_within_tolerance[env_ids] = 0.0
        self._profile_height_success[env_ids] = False
        self._route_cte_sq_distance_sum[env_ids] = 0.0
        self._route_cte_distance_sum[env_ids] = 0.0
        self._last_route_cte_rmse[env_ids] = 0.0
        self._route_cte_success[env_ids] = False
        self._route_state = None
