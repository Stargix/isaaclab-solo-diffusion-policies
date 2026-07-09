"""Solo12 wrapper around the tested Diffusion Policy transformer."""

from __future__ import annotations

import torch
from torch import nn

from .transformer_for_diffusion import TransformerForDiffusion


class TransformerDiffusionPolicy(nn.Module):
    """Predict DDPM epsilon for a future action chunk.

    This wrapper keeps the Solo12-specific call signature while delegating the
    denoising backbone to `TransformerForDiffusion`, the architecture used by
    Diffusion Policy and reused by DiffuseLoco.
    """

    def __init__(
        self,
        *,
        obs_dim: int = 42,
        goal_dim: int = 11,
        action_dim: int = 12,
        history: int = 8,
        action_horizon: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        causal_attn: bool = True,
        n_cond_layers: int = 0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.history = history
        self.action_horizon = action_horizon
        self.cond_dim = obs_dim + goal_dim
        # TransformerForDiffusion uses feedforward dim = 4 * n_emb internally.
        if dim_feedforward != 4 * d_model:
            raise ValueError("TransformerForDiffusion uses dim_feedforward = 4 * d_model; set dim_feedforward accordingly.")
        self.model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=action_horizon,
            n_obs_steps=history,
            cond_dim=self.cond_dim,
            n_layer=num_layers,
            n_head=nhead,
            n_emb=d_model,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
            causal_attn=causal_attn,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers,
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
        obs_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise with shapes:

        - noisy_actions: ``[B, action_horizon, action_dim]``
        - obs_hist: ``[B, history, obs_dim]``
        - goal_hist: ``[B, history, goal_dim]``
        - timesteps: ``[B]``
        """

        cond = torch.cat([obs_hist, goal_hist], dim=-1)
        return self.model(noisy_actions, timesteps, cond)

