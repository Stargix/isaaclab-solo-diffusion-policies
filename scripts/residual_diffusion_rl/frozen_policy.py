"""Adapter that gives residual RL a frozen, stateful diffusion prior."""

from __future__ import annotations

from pathlib import Path

import torch

from scripts.diffusion_policy.model.solo12_diffusion_policy import (
    Solo12DiffusionPolicy,
    Solo12DiffusionPolicyConfig,
)
from scripts.diffusion_policy.train.runtime.checkpoint import load_training_checkpoint

from .contracts import GOAL_DIM, GOAL_SCHEMA, POLICY_KIND, PROPRIO_DIM, RESIDUAL_ACTION_DIM


def _model_config(config: dict, inference_steps: int | None) -> Solo12DiffusionPolicyConfig:
    model = config["model"]
    dataset = config["dataset"]
    diffusion = config["diffusion"]
    return Solo12DiffusionPolicyConfig(
        proprio_dim=model.get("proprio_dim", PROPRIO_DIM),
        action_hist_dim=model.get("action_hist_dim", RESIDUAL_ACTION_DIM),
        goal_dim=model.get("goal_dim", GOAL_DIM),
        history=dataset["history"],
        prediction_horizon=dataset["prediction_horizon"],
        execution_offset=dataset["execution_offset"],
        d_model=model["d_model"],
        nhead=model["nhead"],
        num_layers=model["num_layers"],
        p_drop_emb=model.get("p_drop_emb", model.get("dropout", 0.0)),
        p_drop_attn=model.get("p_drop_attn", model.get("dropout", 0.3)),
        separate_goal_conditioning=model.get("separate_goal_conditioning", True),
        num_train_timesteps=diffusion["num_train_timesteps"],
        beta_start=diffusion["beta_start"],
        beta_end=diffusion["beta_end"],
        beta_schedule=diffusion["beta_schedule"],
        prediction_type=diffusion["prediction_type"],
        variance_type=diffusion["variance_type"],
        clip_sample=diffusion["clip_sample"],
        cfg_dropout_prob=diffusion.get("cfg_dropout_prob", 0.0),
        num_inference_steps=inference_steps,
    )


class FrozenSpatialDiffusion:
    """Maintain the exact delayed-history contract used in Phase A."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        num_envs: int,
        device: torch.device | str,
        *,
        inference_steps: int | None = None,
        exec_horizon: int = 4,
    ):
        self.device = torch.device(device)
        checkpoint = load_training_checkpoint(
            checkpoint_path, self.device, expected_policy_kind=POLICY_KIND
        )
        config = checkpoint["config"]
        if config["dataset"].get("goal_representation") != "hindsight_geom_avg12":
            raise ValueError("Phase B1 requires a hindsight_geom_avg12 checkpoint")
        if config.get("goal_schema") != GOAL_SCHEMA:
            raise ValueError(f"Expected goal schema {GOAL_SCHEMA!r}")
        self.cfg = _model_config(config, inference_steps)
        if self.cfg.proprio_dim != PROPRIO_DIM or self.cfg.goal_dim != GOAL_DIM:
            raise ValueError("Checkpoint tensor dimensions do not match the B1 contract")
        available = self.cfg.prediction_horizon - self.cfg.execution_offset
        if not 1 <= exec_horizon <= available:
            raise ValueError(f"exec_horizon must be in [1, {available}]")
        self.exec_horizon = int(exec_horizon)
        self.goal_horizon_steps = int(config["dataset"]["goal_horizon_steps"])
        self.v_clip = float(config["dataset"].get("v_req_clip", 2.0))

        self.policy = Solo12DiffusionPolicy(self.cfg)
        self.policy.load_state_dict(checkpoint["ema_model_state_dict"])
        self.policy.set_normalizer_stats(checkpoint["normalizer_stats"])
        self.policy.to(self.device).eval()
        self.policy.requires_grad_(False)

        h = self.cfg.history
        self.proprio_history = torch.zeros(num_envs, h, PROPRIO_DIM, device=self.device)
        self.action_history = torch.zeros(num_envs, h, RESIDUAL_ACTION_DIM, device=self.device)
        self.goal_history = torch.zeros(num_envs, h, GOAL_DIM, device=self.device)
        self.last_executed = torch.zeros(num_envs, RESIDUAL_ACTION_DIM, device=self.device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._chunk: torch.Tensor | None = None
        self._chunk_step = self.exec_horizon
        stats = self.policy.normalizer_stats.action
        self.action_min = torch.as_tensor(stats.min, device=self.device)
        self.action_max = torch.as_tensor(stats.max, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.proprio_history[env_ids] = 0.0
        self.action_history[env_ids] = 0.0
        self.goal_history[env_ids] = 0.0
        self.last_executed[env_ids] = 0.0
        self.initialized[env_ids] = False
        # A chunk is batched; invalidate it for all envs if any member resets.
        self._chunk = None
        self._chunk_step = self.exec_horizon

    @staticmethod
    def _append_selected(history: torch.Tensor, value: torch.Tensor, selected: torch.Tensor) -> None:
        """Append through a boolean index without mutating an advanced-index copy."""

        history[selected, :-1] = history[selected, 1:].clone()
        history[selected, -1] = value[selected]

    @torch.no_grad()
    def propose(self, proprio: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        missing = ~self.initialized
        if torch.any(missing):
            self.proprio_history[missing] = proprio[missing, None, :]
            self.goal_history[missing] = goal[missing, None, :]
            self.action_history[missing] = 0.0
            self.initialized[missing] = True
        present = ~missing
        if torch.any(present):
            self._append_selected(self.proprio_history, proprio, present)
            self._append_selected(self.goal_history, goal, present)
            self._append_selected(self.action_history, self.last_executed, present)

        if self._chunk is None or self._chunk_step >= self.exec_horizon:
            trajectory = self.policy.predict_action_denormalized(
                self.proprio_history, self.action_history, self.goal_history
            )
            self._chunk = self.policy.executable_chunk(trajectory, self.exec_horizon)
            self._chunk_step = 0
        action = self._chunk[:, self._chunk_step]
        self._chunk_step += 1
        return action

    def record_executed(self, action: torch.Tensor) -> None:
        self.last_executed.copy_(action)
