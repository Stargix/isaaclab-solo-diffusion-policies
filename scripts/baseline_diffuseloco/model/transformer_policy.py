"""Solo12 wrapper around the tested Diffusion Policy transformer."""

from __future__ import annotations

import torch
from torch import nn

from .transformer_for_diffusion import TransformerForDiffusion


class TransformerDiffusionPolicy(nn.Module):
    """Predict DDPM epsilon for a future action chunk."""

    def __init__(
        self,
        *,
        proprio_dim: int = 30,
        action_hist_dim: int = 12,
        goal_dim: int = 3,
        action_dim: int = 12,
        history: int = 8,
        prediction_horizon: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        n_cond_layers: int = 0,
        separate_goal_conditioning: bool = True,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_hist_dim = action_hist_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.history = history
        self.prediction_horizon = prediction_horizon
        self.io_dim = proprio_dim + action_hist_dim
        if dim_feedforward != 4 * d_model:
            raise ValueError("TransformerForDiffusion uses dim_feedforward = 4 * d_model; set dim_feedforward accordingly.")
        self.model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=prediction_horizon,
            n_obs_steps=history,
            cond_dim=self.io_dim,
            goal_dim=goal_dim,
            n_layer=num_layers,
            n_head=nhead,
            n_emb=d_model,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers,
            separate_goal_conditioning=separate_goal_conditioning,
        )

    def configure_optimizers(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        io_cond = torch.cat([proprio_hist, action_hist], dim=-1)
        return self.model(noisy_actions, timesteps, io_cond, goal_cond=goal_hist)
