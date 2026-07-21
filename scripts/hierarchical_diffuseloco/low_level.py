"""Frozen DiffuseLoco adapter.

The adapter owns only history construction and checkpoint loading.  It never filters or
interpolates the high-level command, preserving the causal research question.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch


class FrozenDiffuseLoco:
    """Batchable inference wrapper around the repository's velocity-height checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cuda",
                 inference_steps: int | None = None, exec_horizon: int = 1):
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.baseline_diffuseloco.model.solo12_diffusion_policy import (
            Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig,
        )
        from scripts.baseline_diffuseloco.train.runtime.checkpoint import load_training_checkpoint

        self.device = torch.device(device)
        checkpoint = load_training_checkpoint(checkpoint, self.device,
                                              expected_policy_kind="diffuseloco_velocity_height_ddpm")
        config = checkpoint["config"]
        model_cfg: dict[str, Any] = dict(config["model"])
        diffusion_cfg: dict[str, Any] = dict(config["diffusion"])
        allowed = set(Solo12DiffusionPolicyConfig.__dataclass_fields__)
        kwargs = {key: value for key, value in {**model_cfg, **diffusion_cfg}.items() if key in allowed}
        if inference_steps is not None:
            kwargs["num_inference_steps"] = int(inference_steps)
        self.policy = Solo12DiffusionPolicy(Solo12DiffusionPolicyConfig(**kwargs))
        self.policy.load_state_dict(checkpoint["ema_model_state_dict"])
        self.policy.set_normalizer_stats(checkpoint["normalizer_stats"])
        self.policy.to(self.device).eval()
        self.exec_horizon = int(exec_horizon)
        future = self.policy.cfg.prediction_horizon - self.policy.cfg.execution_offset
        if not 1 <= self.exec_horizon <= future:
            raise ValueError(f"exec_horizon must be in [1,{future}], got {self.exec_horizon}.")
        self._proprio: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None
        self._goals: torch.Tensor | None = None
        self._chunk: torch.Tensor | None = None
        self._chunk_index = 0
        self._initialized: torch.Tensor | None = None

    @property
    def history(self) -> int:
        return self.policy.cfg.history

    def reset(self, num_envs: int) -> None:
        shape = (num_envs, self.history)
        self._proprio = torch.zeros((*shape, self.policy.cfg.proprio_dim), device=self.device)
        self._actions = torch.zeros((*shape, self.policy.cfg.action_hist_dim), device=self.device)
        self._goals = torch.zeros((*shape, self.policy.cfg.goal_dim), device=self.device)
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._chunk = None
        self._chunk_index = 0

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset selected histories without disturbing other vectorized environments."""
        if self._proprio is None:
            return
        ids = env_ids.to(self.device)
        self._proprio[ids] = 0.0
        self._actions[ids] = 0.0
        self._goals[ids] = 0.0
        self._initialized[ids] = False
        self._chunk = None
        self._chunk_index = 0

    @torch.inference_mode()
    def act(self, proprio: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if proprio.ndim != 2 or proprio.shape[-1] != self.policy.cfg.proprio_dim:
            raise ValueError(f"proprio must be [N,{self.policy.cfg.proprio_dim}], got {tuple(proprio.shape)}")
        if goal.shape != (proprio.shape[0], self.policy.cfg.goal_dim):
            raise ValueError(f"goal must be [N,{self.policy.cfg.goal_dim}], got {tuple(goal.shape)}")
        if self._proprio is None or self._proprio.shape[0] != proprio.shape[0]:
            self.reset(proprio.shape[0])
        proprio, goal = proprio.to(self.device), goal.to(self.device)
        fresh = ~self._initialized
        if torch.any(fresh):
            self._proprio[fresh] = proprio[fresh, None]
            self._goals[fresh] = goal[fresh, None]
            self._initialized[fresh] = True
        active = ~fresh
        if torch.any(active):
            self._proprio[active, :-1] = self._proprio[active, 1:].clone()
            self._proprio[active, -1] = proprio[active]
            self._goals[active, :-1] = self._goals[active, 1:].clone()
            self._goals[active, -1] = goal[active]
        if self._chunk is None or self._chunk_index >= self.exec_horizon:
            trajectory = self.policy.predict_action_denormalized(self._proprio, self._actions, self._goals)
            self._chunk = self.policy.executable_chunk(trajectory, self.exec_horizon)
            self._chunk_index = 0
        action = self._chunk[:, self._chunk_index]
        self._chunk_index += 1
        self._actions[:, :-1] = self._actions[:, 1:].clone()
        self._actions[:, -1] = action
        return action
