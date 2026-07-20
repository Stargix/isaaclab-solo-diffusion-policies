"""Optional Isaac Lab adapter for the frozen-low-level experiment.

Importing this module does not launch Isaac Sim.  The dependency is checked at class
definition time so the route/reward tests remain lightweight.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import torch

from .contracts import ActionBounds
from .low_level import FrozenDiffuseLoco, support_risk
from .rewards import RewardWeights, hierarchical_reward
from .route import PolylineRoute, clearance_route, wrap_angle

try:  # Isaac Lab is intentionally optional for unit tests.
    from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
    from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
    from scripts.baseline_diffuseloco.train.data.obs_utils import proprio_from_env_tensors
except ImportError:  # pragma: no cover - exercised only outside the Isaac environment
    Solo12Env = object  # type: ignore[misc,assignment]
    Solo12EnvCfg = object  # type: ignore[misc,assignment]
    proprio_from_env_tensors = None


class HierarchicalSolo12EnvCfg(Solo12EnvCfg):  # type: ignore[misc]
    """Configuration additions; the base Solo12 scene and actuator setup are reused."""

    action_space = 4
    observation_space = 42  # 8*(x,y,yaw,height) + final pose + time/clearance + 5 state values
    diffuseloco_checkpoint: str = ""
    diffuseloco_inference_steps: int | None = None
    diffuseloco_exec_horizon: int = 1
    route_length_m: float = 5.0
    route_file: str = ""
    route_preview_points: int = 8
    high_level_decimation: int = 4
    reward_weights: RewardWeights = RewardWeights()


class HierarchicalSolo12Env(Solo12Env):  # type: ignore[misc]
    """High-level action -> frozen DiffuseLoco -> joint targets.

    The only command processing is clipping to a documented physical envelope.  In
    particular, no EMA, slew-rate limiter or interpolation is applied to ``height``.
    """

    cfg: HierarchicalSolo12EnvCfg

    def __init__(self, cfg: HierarchicalSolo12EnvCfg, render_mode: str | None = None, **kwargs):
        if not cfg.diffuseloco_checkpoint:
            raise ValueError("cfg.diffuseloco_checkpoint is required for the frozen low-level experiment.")
        super().__init__(cfg, render_mode, **kwargs)
        self._bounds = ActionBounds()
        self._route: PolylineRoute = (PolylineRoute.from_file(cfg.route_file)
                                      if cfg.route_file else clearance_route(cfg.route_length_m))
        self._preview_points = int(cfg.route_preview_points)
        if self._preview_points != 8:
            raise ValueError("The v1 observation contract fixes route_preview_points=8.")
        device = self.device
        self._low_level = FrozenDiffuseLoco(cfg.diffuseloco_checkpoint, device=device,
                                            inference_steps=cfg.diffuseloco_inference_steps,
                                            exec_horizon=cfg.diffuseloco_exec_horizon)
        self._previous_command = torch.zeros(self.num_envs, 4, device=device)
        self._previous_previous_command = torch.zeros_like(self._previous_command)
        self._older_command = torch.zeros_like(self._previous_command)
        self._progress = torch.zeros(self.num_envs, device=device)
        self._last_progress = torch.zeros_like(self._progress)
        self._high_observation = torch.zeros(self.num_envs, cfg.observation_space, device=device)
        self._low_level.reset(self.num_envs)

    def _route_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (torch.as_tensor(self._route.xy, device=self.device), torch.as_tensor(self._route.yaw, device=self.device),
                torch.as_tensor(self._route.required_height, device=self.device))

    def _route_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized straight-route observation; route generation is in ``route.py``."""
        xy, yaw, heights = self._route_tensors()
        origin = self._terrain.env_origins[:, :2]
        position = self._robot.data.root_pos_w[:, :2] - origin
        nearest = torch.sum((position[:, None, :] - xy[None, :, :]) ** 2, dim=-1).argmin(dim=1)
        indices = (nearest[:, None] + torch.arange(1, 9, device=self.device)[None, :]).clamp(max=len(self._route.xy) - 1)
        delta = xy[indices] - position[:, None, :]
        quat = self._robot.data.root_quat_w
        robot_yaw = torch.atan2(2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                                1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
        c, s = torch.cos(robot_yaw), torch.sin(robot_yaw)
        local_x = c[:, None] * delta[..., 0] + s[:, None] * delta[..., 1]
        local_y = -s[:, None] * delta[..., 0] + c[:, None] * delta[..., 1]
        local_yaw = torch.atan2(torch.sin(yaw[indices] - robot_yaw[:, None]), torch.cos(yaw[indices] - robot_yaw[:, None]))
        preview = torch.stack((local_x, local_y, local_yaw, heights[indices]), dim=-1).reshape(self.num_envs, -1)
        final_delta = xy[-1][None, :] - position
        final_pose = torch.stack((c * final_delta[:, 0] + s * final_delta[:, 1],
                                  -s * final_delta[:, 0] + c * final_delta[:, 1],
                                  torch.atan2(torch.sin(yaw[-1] - robot_yaw), torch.cos(yaw[-1] - robot_yaw))), dim=-1)
        required = heights[indices[:, 0]]
        self._progress = (nearest.float() * float(self._route.spacing)).clamp(0.0, self._route.length)
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        state = torch.cat((self._robot.data.root_lin_vel_b[:, :2], self._robot.data.root_ang_vel_b[:, 2:3],
                           base_height[:, None], self._robot.data.projected_gravity_b[:, 0:1]), dim=-1)
        remaining = (self.max_episode_length - self.episode_length_buf).float() * self.step_dt
        return preview, final_pose, remaining, required, state

    def _get_observations(self) -> dict[str, torch.Tensor]:
        preview, final_pose, remaining, required, state = self._route_features()
        self._high_observation = torch.cat((preview, final_pose, remaining[:, None], required[:, None], state), dim=-1)
        return {"policy": self._high_observation}

    def _pre_physics_step(self, actions: torch.Tensor):
        command = actions.to(self.device)
        if command.shape != (self.num_envs, 4):
            raise ValueError(f"Expected high-level action [{self.num_envs},4], got {tuple(command.shape)}")
        low = torch.as_tensor(self._bounds.low, device=self.device, dtype=command.dtype)
        high = torch.as_tensor(self._bounds.high, device=self.device, dtype=command.dtype)
        command = torch.maximum(torch.minimum(command, high), low)
        self._older_command.copy_(self._previous_previous_command)
        self._previous_previous_command.copy_(self._previous_command)
        self._previous_command.copy_(command)
        self._commands[:, :3] = command[:, :3]
        proprio = proprio_from_env_tensors(self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs,
                                            self._robot.data.joint_vel[:, self._joint_ids], self._robot.data.root_ang_vel_b,
                                            self._robot.data.projected_gravity_b)
        joint_action = self._low_level.act(proprio, command)
        self._actions = joint_action
        self._processed_actions = self.cfg.action_scale * self._actions + self._q_offset_action_and_obs

    def _get_rewards(self) -> torch.Tensor:
        preview, final_pose, remaining, required, _ = self._route_features()
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        progress_delta = self._progress - self._last_progress
        self._last_progress.copy_(self._progress)
        time_error = torch.relu(-remaining)
        fallen = self.reset_terminated
        reward, terms = hierarchical_reward(
            progress=progress_delta, cross_track=preview[:, 1], final_pose_error=final_pose,
            time_error=time_error, actual_height=base_height, required_height=required,
            command=self._previous_command, previous_command=self._previous_previous_command,
            previous_previous_command=self._older_command,
            low_level_risk=support_risk(self._previous_command), fallen=fallen,
            weights=self.cfg.reward_weights,
        )
        self.extras["log"] = {f"RewardsPerStep/{name}": float(value.mean()) for name, value in terms.items()}
        return reward * self.step_dt

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._low_level.reset_envs(env_ids)
        self._previous_command[env_ids] = 0.0
        self._previous_previous_command[env_ids] = 0.0
        self._older_command[env_ids] = 0.0
        self._progress[env_ids] = 0.0
        self._last_progress[env_ids] = 0.0
