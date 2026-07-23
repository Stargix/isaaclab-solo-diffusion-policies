"""IsaacLab environment for Phase B1 residual PPO."""

from __future__ import annotations

import torch

from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
from scripts.diffusion_policy.train.data.obs_utils import proprio_from_env_tensors

from .contracts import OBSERVATION_DIM, RESIDUAL_ACTION_DIM
from .frozen_policy import FrozenSpatialDiffusion
from .rewards import RewardWeights, residual_reward
from .routes import RouteBank, RouteState


class ResidualDiffusionEnvCfg(Solo12EnvCfg):
    """Flat config fields keep Hydra overrides robust and serializable."""

    action_space = RESIDUAL_ACTION_DIM
    observation_space = OBSERVATION_DIM
    episode_length_s = 24.0
    spatial_diffusion_checkpoint: str = ""
    diffusion_inference_steps: int | None = 10
    diffusion_exec_horizon: int = 4
    residual_scale: float = 0.20
    residual_action_clip: float = 1.0
    route_points: int = 101
    route_length_m: float = 4.0
    height_segment_m: float = 0.8
    corridor_half_width_m: float = 0.45
    route_goal_tolerance_m: float = 0.12
    terminal_height_tolerance_m: float = 0.04
    # DirectRLEnv's common_step_counter counts vector control steps (not the
    # number of environment samples). These thresholds fit a 12k x 32 rollout.
    curriculum_stage1_steps: int = 80_000
    curriculum_stage2_steps: int = 200_000
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
        self._routes.reset(all_ids, stage=0)
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

    def _curriculum_stage(self) -> int:
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
        return proprio_from_env_tensors(
            self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs,
            self._robot.data.joint_vel[:, self._joint_ids],
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
        )

    def _update_route(self) -> RouteState:
        self._route_state = self._routes.update(self._local_position())
        return self._route_state

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
        tangent_speed = self._robot.data.root_lin_vel_b[:, 0]
        route_features = torch.cat((
            self._robot.data.root_lin_vel_b[:, :2],
            self._robot.data.root_ang_vel_b[:, 2:3],
            state.cross_track[:, None],
            state.remaining_fraction[:, None],
            (base_height - state.target_height)[:, None],
            (tangent_speed - self._routes.speed)[:, None],
            torch.sin(yaw_error)[:, None],
            torch.cos(yaw_error)[:, None],
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
        # A route timeout is a task failure as well: without this terminal
        # penalty, an agent can avoid the completion bonus by lingering until
        # the horizon.  RSL-RL still receives the timeout flag separately for
        # correct value bootstrapping.
        failed = (self.reset_terminated | self.reset_time_outs) & ~self._success
        reward, terms = residual_reward(
            progress_delta=state.progress_delta,
            cross_track=state.cross_track,
            speed_error=self._robot.data.root_lin_vel_b[:, 0] - self._routes.speed,
            yaw_error=yaw_error,
            height_error=base_height - state.target_height,
            residual=self._residual,
            previous_residual=self._previous_residual,
            projected_gravity_xy=self._robot.data.projected_gravity_b[:, :2],
            vertical_velocity=self._robot.data.root_lin_vel_b[:, 2],
            success=self._success,
            failed=failed,
            dt=self.step_dt,
            weights=self._reward_weights,
        )
        self._episode_reward_sums += reward
        self.extras["log"] = {
            **{f"RewardsPerStep/{name}": float(value.mean()) for name, value in terms.items()},
            "Metrics/progress_m": float(state.progress.mean()),
            "Metrics/cross_track_abs_m": float(state.cross_track.abs().mean()),
            "Metrics/height_error_abs_m": float((base_height - state.target_height).abs().mean()),
            "Metrics/speed_error_abs_mps": float((self._robot.data.root_lin_vel_b[:, 0] - self._routes.speed).abs().mean()),
            "Metrics/residual_rms": float(torch.sqrt(self._residual.square().mean())),
            "Metrics/residual_abs_mean": float(self._residual.abs().mean()),
            "Metrics/residual_saturation_frac": float((self._residual.abs() >= 0.95).float().mean()),
            "Metrics/success": float(self._success.float().mean()),
            "Curriculum/stage": float(self._curriculum_stage()),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_terminated, time_out = super()._get_dones()
        state = self._update_route()
        corridor_failure = state.cross_track.abs() > self.cfg.corridor_half_width_m
        yaw_error = torch.atan2(torch.sin(state.tangent_yaw - self._robot_yaw()),
                                torch.cos(state.tangent_yaw - self._robot_yaw()))
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        reached_terminal_pose = state.success & (state.cross_track.abs() <= self.cfg.route_goal_tolerance_m) & (
            yaw_error.abs() <= 0.35
        ) & ((base_height - state.target_height).abs() <= self.cfg.terminal_height_tolerance_m)
        # A contact or corridor failure on the terminal step is a failure, not
        # a successful completion.
        self._success = reached_terminal_pose & ~contact_terminated & ~corridor_failure
        self._base_contact_termination.copy_(contact_terminated)
        self._corridor_termination.copy_(corridor_failure)
        return contact_terminated | corridor_failure | self._success, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_prior"):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Solo12Env labels every non-timeout reset as ``base_contact``.  B1
        # introduces route success and corridor exits, so overwrite that
        # inherited aggregate with the mutually interpretable event counts.
        episode_log = self.extras.setdefault("log", {})
        episode_log["Episode_Termination/base_contact"] = torch.count_nonzero(
            self._base_contact_termination[env_ids]
        ).item()
        episode_log["Episode_Termination/corridor_failure"] = torch.count_nonzero(
            self._corridor_termination[env_ids]
        ).item()
        episode_log["Episode_Termination/route_success"] = torch.count_nonzero(self._success[env_ids]).item()
        self._routes.reset(env_ids, self._curriculum_stage())
        self._prior.reset(env_ids)
        self._residual[env_ids] = 0.0
        self._previous_residual[env_ids] = 0.0
        self._success[env_ids] = False
        self._base_contact_termination[env_ids] = False
        self._corridor_termination[env_ids] = False
        self._base_action_ready = False
