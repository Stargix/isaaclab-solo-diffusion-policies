"""Normalized deterministic action-chunk policy with the diffusion runtime API."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from ..train.data.normalization import NormalizerStats, denormalize_minmax, normalize_minmax, normalize_zscore
except ImportError:  # pragma: no cover
    from train.data.normalization import NormalizerStats, denormalize_minmax, normalize_minmax, normalize_zscore

from .transformer_chunk_policy import TransformerChunkPolicy


@dataclass
class Solo12DeterministicChunkPolicyConfig:
    proprio_dim: int = 30
    action_hist_dim: int = 12
    goal_dim: int = 16
    action_dim: int = 12
    history: int = 8
    prediction_horizon: int = 16
    execution_offset: int = 8
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3


class Solo12DeterministicChunkPolicy(torch.nn.Module):
    """Direct BC policy.  Training outputs stay unconstrained; deployment clamps."""

    def __init__(self, cfg: Solo12DeterministicChunkPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = TransformerChunkPolicy(
            proprio_dim=cfg.proprio_dim, action_hist_dim=cfg.action_hist_dim,
            goal_dim=cfg.goal_dim, action_dim=cfg.action_dim, history=cfg.history,
            prediction_horizon=cfg.prediction_horizon, d_model=cfg.d_model,
            nhead=cfg.nhead, num_layers=cfg.num_layers, p_drop_emb=cfg.p_drop_emb,
            p_drop_attn=cfg.p_drop_attn,
        )
        self.normalizer_stats: NormalizerStats | None = None

    def set_normalizer_stats(self, stats: NormalizerStats | dict) -> None:
        self.normalizer_stats = NormalizerStats.from_dict(stats) if isinstance(stats, dict) else stats

    def configure_optimizers(self, *, learning_rate: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(learning_rate=learning_rate, weight_decay=weight_decay, betas=betas)

    def _normalized_prediction(self, proprio_hist: torch.Tensor, action_hist: torch.Tensor, goal_hist: torch.Tensor) -> torch.Tensor:
        if self.normalizer_stats is None:
            raise RuntimeError("Normalizer stats must be set before policy use.")
        return self.model(
            normalize_zscore(proprio_hist, self.normalizer_stats.proprio),
            normalize_minmax(action_hist, self.normalizer_stats.action),
            normalize_zscore(goal_hist, self.normalizer_stats.goal),
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prediction = self._normalized_prediction(batch["proprio_hist"], batch["action_hist"], batch["goal_hist"])
        target = normalize_minmax(batch["actions"], self.normalizer_stats.action)
        loss = F.mse_loss(prediction, target)
        # Diagnostics only: no future-token reweighting or auxiliary objective.
        self.last_loss_metrics = {
            "normalized_mse": float(loss.detach()),
            "executable_future_mse": float(F.mse_loss(prediction[:, self.cfg.execution_offset :], target[:, self.cfg.execution_offset :]).detach()),
        }
        return loss

    @torch.no_grad()
    def predict_action(self, proprio_hist: torch.Tensor, action_hist: torch.Tensor, goal_hist: torch.Tensor, *, guidance_scale: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        if guidance_scale not in (None, 1.0):
            raise ValueError("guidance_scale is only defined for diffusion checkpoints; deterministic_chunk requires 1.0.")
        del generator
        return self._normalized_prediction(proprio_hist, action_hist, goal_hist).clamp(-1.0, 1.0)

    @torch.no_grad()
    def predict_action_denormalized(self, proprio_hist: torch.Tensor, action_hist: torch.Tensor, goal_hist: torch.Tensor, *, guidance_scale: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        return denormalize_minmax(self.predict_action(proprio_hist, action_hist, goal_hist, guidance_scale=guidance_scale, generator=generator), self.normalizer_stats.action)

    def executable_chunk(self, trajectory: torch.Tensor, num_actions: int = 1) -> torch.Tensor:
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1.")
        start, end = self.cfg.execution_offset, self.cfg.execution_offset + num_actions
        if end > self.cfg.prediction_horizon:
            raise ValueError("Requested executable chunk exceeds the prediction horizon.")
        return trajectory[:, start:end]
