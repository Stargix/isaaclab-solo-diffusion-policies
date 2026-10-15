"""Inference, delayed histories, geometric profile16 goals and joint targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from scripts.dppo_diffusion_rl.conditioning import remaining_speed_budget
from scripts.diffusion_policy.train.conditioning.goal_builder import (
    advance_path_progress,
    build_geometric_height_profile_goal_from_path,
)

EXPECTED_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)


class RuntimeActionPolicy(Protocol):
    """Small inference interface implemented by both native and cached samplers."""

    def predict_action_denormalized(
        self, proprio_hist: torch.Tensor, action_hist: torch.Tensor, goal_hist: torch.Tensor,
        *, guidance_scale: float = 1.0,
    ) -> torch.Tensor: ...

    def executable_chunk(self, trajectory: torch.Tensor, num_actions: int) -> torch.Tensor: ...


def validate_joint_mapping(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Require the trained SOLO12 action order; never infer alphabetically."""
    ordered = tuple(str(name) for name in names)
    if ordered != EXPECTED_JOINT_NAMES:
        raise ValueError(
            "Joint mapping differs from the trained SOLO12 action order: "
            f"{ordered!r} != {EXPECTED_JOINT_NAMES!r}."
        )
    return ordered


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity.")


@dataclass
class DelayedHistory:
    """Evaluator-compatible buffers; call ``commit_tick`` after each env step."""

    proprio: torch.Tensor
    action: torch.Tensor
    goal: torch.Tensor
    previous_action: torch.Tensor

    @classmethod
    def zeros(
        cls, batch: int, *, device: torch.device, dtype: torch.dtype = torch.float32,
        history: int = 8,
    ) -> "DelayedHistory":
        if batch < 1 or history != 8:
            raise ValueError("WC-v3 history requires a positive batch and exactly 8 frames.")
        return cls(
            torch.zeros(batch, history, 30, device=device, dtype=dtype),
            torch.zeros(batch, history, 12, device=device, dtype=dtype),
            torch.zeros(batch, history, 16, device=device, dtype=dtype),
            torch.zeros(batch, 12, device=device, dtype=dtype),
        )

    def validate(self) -> None:
        batch = self.proprio.shape[0] if self.proprio.ndim == 3 else -1
        expected = ((batch, 8, 30), (batch, 8, 12), (batch, 8, 16), (batch, 12))
        actual = tuple(x.shape for x in (self.proprio, self.action, self.goal, self.previous_action))
        if actual != expected:
            raise ValueError(f"Invalid WC-v3 delayed-history shapes: {actual}, expected {expected}.")
        for name, value in zip(("proprio", "action", "goal", "previous_action"),
                               (self.proprio, self.action, self.goal, self.previous_action)):
            _finite_tensor(name, value)

    def commit_tick(
        self, pre_step_proprio: torch.Tensor, executed_env_action: torch.Tensor,
        current_goal: torch.Tensor,
    ) -> None:
        """Slide pre-step values only after stepping, matching evaluate_policy."""
        self.validate()
        batch = self.proprio.shape[0]
        required = ((pre_step_proprio, (batch, 30), "pre_step_proprio"),
                    (executed_env_action, (batch, 12), "executed_env_action"),
                    (current_goal, (batch, 16), "current_goal"))
        for value, shape, name in required:
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}.")
            _finite_tensor(name, value)
        self.proprio[:, :-1] = self.proprio[:, 1:].clone()
        self.proprio[:, -1] = pre_step_proprio
        self.action[:, :-1] = self.action[:, 1:].clone()
        self.action[:, -1] = self.previous_action
        self.goal[:, :-1] = self.goal[:, 1:].clone()
        self.goal[:, -1] = current_goal
        self.previous_action = executed_env_action.clone()

    def reset(self) -> None:
        self.proprio.zero_()
        self.action.zero_()
        self.goal.zero_()
        self.previous_action.zero_()


@dataclass
class ChunkExecutor:
    """Four-action receding-horizon schedule with exact reset invalidation."""

    policy: RuntimeActionPolicy
    exec_horizon: int = 4
    guidance_scale: float = 1.0
    chunk: torch.Tensor | None = None
    next_slot: int = 0

    def __post_init__(self) -> None:
        if self.exec_horizon != 4:
            raise ValueError("Pinned WC-v3 runtime executes exactly four actions per plan.")
        if self.guidance_scale != 1.0:
            raise ValueError("DPPO hybrid runtime requires guidance_scale=1.")

    @torch.inference_mode()
    def action(self, proprio: torch.Tensor, actions: torch.Tensor, goals: torch.Tensor
               ) -> tuple[torch.Tensor, int, bool]:
        replanned = self.chunk is None or self.next_slot == 0
        if replanned:
            trajectory = self.policy.predict_action_denormalized(
                proprio, actions, goals, guidance_scale=self.guidance_scale,
            )
            self.chunk = self.policy.executable_chunk(trajectory, self.exec_horizon)
            if self.chunk.shape[1] != self.exec_horizon:
                raise ValueError("Policy returned a chunk with an unexpected executable horizon.")
        slot = self.next_slot
        chosen = self.chunk[:, slot].clone()
        self.next_slot = (self.next_slot + 1) % self.exec_horizon
        return chosen, slot, replanned

    def reset(self) -> None:
        self.chunk = None
        self.next_slot = 0


def action_to_joint_target(
    env_action: torch.Tensor,
    *,
    action_scale: float,
    q_offset: torch.Tensor,
    clip_actions: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the reference wrapper clip and Solo12 env position-target map."""
    if env_action.ndim != 2 or env_action.shape[-1] != 12:
        raise ValueError(f"Expected env action (B,12), got {tuple(env_action.shape)}.")
    _finite_tensor("env_action", env_action)
    if not np.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError("action_scale must be finite and positive.")
    if clip_actions is not None and (not np.isfinite(clip_actions) or clip_actions < 0.0):
        raise ValueError("clip_actions must be None or finite and non-negative.")
    if q_offset.shape not in {(12,), (1, 12), tuple(env_action.shape)}:
        raise ValueError(f"q_offset shape {tuple(q_offset.shape)} cannot map {tuple(env_action.shape)}.")
    _finite_tensor("q_offset", q_offset)
    clipped = env_action if clip_actions is None else env_action.clamp(-clip_actions, clip_actions)
    target = float(action_scale) * clipped + q_offset
    _finite_tensor("joint_target", target)
    return clipped, target


@dataclass
class Profile16Route:
    """Materialized route and finite-time state for the supported WC-v3 contract."""

    path_w: np.ndarray
    cumulative_xy: np.ndarray
    yaws_w: np.ndarray
    progress_idx: int
    speed_start_arc: float
    speed_target_arc: float
    desired_mean_speed: float

    def validate(self) -> None:
        path = np.asarray(self.path_w)
        arc = np.asarray(self.cumulative_xy).reshape(-1)
        yaw = np.asarray(self.yaws_w).reshape(-1)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
            raise ValueError("Route must contain at least two XYZ path points.")
        if arc.shape != (len(path),) or yaw.shape != (len(path),):
            raise ValueError("Route arc/yaw arrays must match path length.")
        if not np.isfinite(path).all() or not np.isfinite(arc).all() or not np.isfinite(yaw).all():
            raise ValueError("Route contains nonfinite values.")
        if np.any(np.diff(arc) < -1.0e-7) or float(arc[-1]) <= float(arc[0]):
            raise ValueError("Route arc length must be nondecreasing and nonempty.")
        if not 0 <= int(self.progress_idx) < len(path):
            raise ValueError("Route progress index is outside the materialized path.")
        if self.speed_target_arc < self.speed_start_arc or self.desired_mean_speed <= 0.0:
            raise ValueError("Invalid target arc or desired route-average speed.")

    def build_goal(
        self,
        *,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        step: int,
        dt: float = 0.02,
        goal_horizon_steps: int = 100,
        v_req_clip: float = 2.0,
        speed_budget_max_mps: float = 1.5,
    ) -> np.ndarray:
        """Recreate v3: requested-speed preview + remaining-budget scalar only."""
        self.validate()
        root_pos = np.asarray(root_pos_w, dtype=np.float32).reshape(3)
        quat = np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4)
        if not np.isfinite(root_pos).all() or not np.isfinite(quat).all():
            raise ValueError("Root position/quaternion must be finite.")
        if float(np.linalg.norm(quat)) < 1.0e-8:
            raise ValueError("Root quaternion has near-zero norm.")
        self.progress_idx = advance_path_progress(
            self.path_w, root_pos, int(self.progress_idx)
        )
        # v3 deliberately builds geometry using requested average speed first.
        goal = build_geometric_height_profile_goal_from_path(
            self.path_w, self.cumulative_xy, self.yaws_w, root_pos, quat,
            goal_horizon_steps=goal_horizon_steps, dt=dt,
            speed=float(self.desired_mean_speed), start_idx=self.progress_idx,
            v_avg_clip=v_req_clip,
        )
        remaining_distance = max(float(self.speed_target_arc) - float(self.cumulative_xy[self.progress_idx]), 0.0)
        target_distance = max(float(self.speed_target_arc) - float(self.speed_start_arc), 0.0)
        pace = remaining_speed_budget(
            remaining_distance=torch.tensor([remaining_distance], dtype=torch.float32),
            route_length=torch.tensor([target_distance], dtype=torch.float32),
            desired_mean_speed=torch.tensor([self.desired_mean_speed], dtype=torch.float32),
            elapsed_s=torch.tensor([max(0, int(step)) * float(dt)], dtype=torch.float32),
            max_speed=float(speed_budget_max_mps),
            min_remaining_time_s=float(dt),
        )
        goal[-1] = float(pace.item())
        if goal.shape != (16,) or not np.isfinite(goal).all():
            raise ValueError("Goal builder returned invalid profile16 condition.")
        return goal.astype(np.float32, copy=False)
