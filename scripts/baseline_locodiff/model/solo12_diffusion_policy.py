"""Solo12 diffusion policy with continuous SDE/EDM formulation (Karras et al. 2022)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from ..train.data.normalization import NormalizerStats, denormalize_minmax, normalize_minmax, normalize_zscore
except ImportError:  # pragma: no cover - script entrypoint
    from train.data.normalization import NormalizerStats, denormalize_minmax, normalize_minmax, normalize_zscore

from .transformer_policy import TransformerDiffusionPolicy


@dataclass
class Solo12DiffusionPolicyConfig:
    proprio_dim: int = 30
    action_hist_dim: int = 12
    goal_dim: int = 4
    action_dim: int = 12
    history: int = 8
    prediction_horizon: int = 16
    execution_offset: int = 8
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3
    separate_goal_conditioning: bool = True
    
    # EDM Hyperparameters (Karras et al. 2022)
    sigma_data: float = 0.5
    sigma_min: float = 0.001
    sigma_max: float = 80.0
    P_mean: float = -1.2
    P_std: float = 1.2
    rho: float = 7.0
    num_inference_steps: int = 5  # default inference step count
    cfg_dropout_prob: float = 0.0
    guidance_scale: float = 1.0


class Solo12DiffusionPolicy(torch.nn.Module):
    """End-to-end Solo12 diffusion policy using continuous SDE/EDM formulation."""

    def __init__(self, cfg: Solo12DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.model = TransformerDiffusionPolicy(
            proprio_dim=cfg.proprio_dim,
            action_hist_dim=cfg.action_hist_dim,
            goal_dim=cfg.goal_dim,
            action_dim=cfg.action_dim,
            history=cfg.history,
            prediction_horizon=cfg.prediction_horizon,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=4 * cfg.d_model,
            p_drop_emb=cfg.p_drop_emb,
            p_drop_attn=cfg.p_drop_attn,
            separate_goal_conditioning=cfg.separate_goal_conditioning,
        )
        self.normalizer_stats: NormalizerStats | None = None
        self.num_inference_steps = cfg.num_inference_steps

    def set_normalizer_stats(self, stats: NormalizerStats | dict) -> None:
        if isinstance(stats, dict):
            self.normalizer_stats = NormalizerStats.from_dict(stats)
        else:
            self.normalizer_stats = stats

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

    def get_scalings(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute preconditioning factors according to Karras et al. 2022."""
        sigma_data = self.cfg.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
        c_in = 1.0 / (sigma**2 + sigma_data**2).sqrt()
        c_noise = 0.25 * torch.log(sigma)
        return c_skip, c_out, c_in, c_noise

    def forward_edm(
        self,
        y: torch.Tensor,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through preconditioned EDM network.
        
        y: noisy actions of shape (B, prediction_horizon, action_dim)
        sigma: noise levels of shape (B, 1, 1) or broadcastable
        """
        c_skip, c_out, c_in, c_noise = self.get_scalings(sigma)
        
        # Flatten noise scaling to (B,) for sinusoidal embedder
        t_input = c_noise.view(-1)
        
        # Model call: scaled actions c_in * y, and conditionings
        model_output = self.model(c_in * y, proprio_hist, action_hist, goal_hist, t_input)
        
        # Preconditioned output D(y)
        return c_skip * y + c_out * model_output

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.normalizer_stats is None:
            raise RuntimeError("Normalizer stats must be set before computing loss.")

        proprio_hist = normalize_zscore(batch["proprio_hist"], self.normalizer_stats.proprio)
        action_hist = normalize_minmax(batch["action_hist"], self.normalizer_stats.action)
        goal_hist = normalize_zscore(batch["goal_hist"], self.normalizer_stats.goal)
        actions = normalize_minmax(batch["actions"], self.normalizer_stats.action)
        batch_size = actions.shape[0]
        device = actions.device

        if self.cfg.cfg_dropout_prob > 0.0:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg.cfg_dropout_prob
            goal_hist = goal_hist.clone()
            goal_hist[drop_mask] = 0.0

        # Sample noise level sigma log-normally
        rnd_normal = torch.randn((batch_size, 1, 1), device=device)
        sigma = torch.exp(rnd_normal * self.cfg.P_std + self.cfg.P_mean)

        # Add normal noise scaled by sigma
        noise = torch.randn_like(actions)
        noisy_actions = actions + noise * sigma

        # Compute preconditioned model prediction
        denoised = self.forward_edm(noisy_actions, proprio_hist, action_hist, goal_hist, sigma)

        # EDM loss weights lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        loss_weight = (sigma**2 + self.cfg.sigma_data**2) / (sigma * self.cfg.sigma_data)**2
        
        loss = (loss_weight * (denoised - actions)**2).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.normalizer_stats is None:
            raise RuntimeError("Normalizer stats must be set before inference.")

        guidance_scale = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        if guidance_scale != 1.0 and self.cfg.cfg_dropout_prob <= 0.0:
            raise ValueError("guidance_scale != 1 requires a checkpoint trained with command CFG dropout.")

        proprio_n = normalize_zscore(proprio_hist, self.normalizer_stats.proprio)
        action_n = normalize_minmax(action_hist, self.normalizer_stats.action)
        goal_n = normalize_zscore(goal_hist, self.normalizer_stats.goal)
        batch_size = proprio_n.shape[0]
        device = proprio_n.device

        steps = self.num_inference_steps
        if steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")

        # Generate step schedule
        if steps == 1:
            sigmas = torch.tensor([self.cfg.sigma_max, self.cfg.sigma_min], device=device)
        else:
            rho = self.cfg.rho
            min_inv = self.cfg.sigma_min ** (1.0 / rho)
            max_inv = self.cfg.sigma_max ** (1.0 / rho)
            step_indices = torch.arange(steps, dtype=torch.float32, device=device)
            sigmas = (max_inv + step_indices / (steps - 1) * (min_inv - max_inv)) ** rho
            sigmas = torch.cat([sigmas, torch.zeros((1,), device=device)])

        # Sample initial actions from N(0, sigma_max^2 * I)
        x = torch.randn(
            (batch_size, self.cfg.prediction_horizon, self.cfg.action_dim),
            device=device,
            dtype=proprio_n.dtype,
            generator=generator,
        ) * sigmas[0]

        # DDIM sampler (Euler-only, matching LocoDiff)
        for i in range(steps):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_curr_tensor = sigma_curr.expand(batch_size, 1, 1)

            # CFG support
            if guidance_scale != 1.0:
                cond_denoised = self.forward_edm(x, proprio_n, action_n, goal_n, sigma_curr_tensor)
                uncond_denoised = self.forward_edm(x, proprio_n, action_n, torch.zeros_like(goal_n), sigma_curr_tensor)
                denoised = uncond_denoised + guidance_scale * (cond_denoised - uncond_denoised)
            else:
                denoised = self.forward_edm(x, proprio_n, action_n, goal_n, sigma_curr_tensor)

            # DDIM / Euler step
            d = (x - denoised) / sigma_curr
            x = x + (sigma_next - sigma_curr) * d

        return x

    @torch.no_grad()
    def predict_action_denormalized(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        actions_n = self.predict_action(
            proprio_hist,
            action_hist,
            goal_hist,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return denormalize_minmax(actions_n, self.normalizer_stats.action)

    def executable_chunk(self, trajectory: torch.Tensor, num_actions: int = 1) -> torch.Tensor:
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1.")
        start = self.cfg.execution_offset
        end = start + num_actions
        if end > self.cfg.prediction_horizon:
            raise ValueError("Requested executable chunk exceeds the prediction horizon.")
        return trajectory[:, start:end]
