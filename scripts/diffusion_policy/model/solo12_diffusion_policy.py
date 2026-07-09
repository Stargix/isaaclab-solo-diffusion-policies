"""Solo12 diffusion policy with diffusers scheduler and literature-style inference.

Training/inference patterns follow Diffusion Policy / DiffuseLoco:
- `TransformerForDiffusion` backbone
- `diffusers.DDPMScheduler` for forward/reverse diffusion
- classifier-free guidance via full-conditioning dropout during training
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    from ..train.normalization import NormalizerStats, denormalize_minmax, normalize_minmax, normalize_zscore
except ImportError:  # pragma: no cover - script entrypoint
    from train.normalization import NormalizerStats, denormalize_minmax, normalize_minmax, normalize_zscore

from .transformer_policy import TransformerDiffusionPolicy


@dataclass
class Solo12DiffusionPolicyConfig:
    obs_dim: int = 42
    goal_dim: int = 11
    action_dim: int = 12
    history: int = 8
    action_horizon: int = 4
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    num_train_timesteps: int = 100
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    variance_type: str = "fixed_small"
    clip_sample: bool = True
    num_inference_steps: int | None = None
    cfg_dropout_prob: float = 0.2
    guidance_scale: float = 1.5


class Solo12DiffusionPolicy(torch.nn.Module):
    """End-to-end Solo12 diffusion policy used for training and inference."""

    def __init__(self, cfg: Solo12DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.model = TransformerDiffusionPolicy(
            obs_dim=cfg.obs_dim,
            goal_dim=cfg.goal_dim,
            action_dim=cfg.action_dim,
            history=cfg.history,
            action_horizon=cfg.action_horizon,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
            variance_type=cfg.variance_type,
            clip_sample=cfg.clip_sample,
        )
        self.normalizer_stats: NormalizerStats | None = None
        self.num_inference_steps = cfg.num_inference_steps or cfg.num_train_timesteps

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

    def _predict_noise(
        self,
        noisy_actions: torch.Tensor,
        obs_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(noisy_actions, obs_hist, goal_hist, timesteps)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.normalizer_stats is None:
            raise RuntimeError("Normalizer stats must be set before computing loss.")

        obs_hist = normalize_zscore(batch["obs_hist"], self.normalizer_stats.obs)
        goal_hist = normalize_zscore(batch["goal_hist"], self.normalizer_stats.goal)
        actions = normalize_minmax(batch["actions"], self.normalizer_stats.action)
        batch_size = actions.shape[0]
        device = actions.device

        if self.cfg.cfg_dropout_prob > 0.0:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg.cfg_dropout_prob
            obs_hist = obs_hist.clone()
            goal_hist = goal_hist.clone()
            obs_hist[drop_mask] = 0.0
            goal_hist[drop_mask] = 0.0

        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = self._predict_noise(noisy_actions, obs_hist, goal_hist, timesteps)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = actions
        else:
            raise ValueError(f"Unsupported prediction_type={self.noise_scheduler.config.prediction_type!r}")
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict_action(
        self,
        obs_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample a normalized action chunk."""
        if self.normalizer_stats is None:
            raise RuntimeError("Normalizer stats must be set before inference.")

        guidance_scale = self.cfg.guidance_scale if guidance_scale is None else guidance_scale
        obs_n = normalize_zscore(obs_hist, self.normalizer_stats.obs)
        goal_n = normalize_zscore(goal_hist, self.normalizer_stats.goal)

        trajectory = torch.randn(
            (obs_n.shape[0], self.cfg.action_horizon, self.cfg.action_dim),
            device=obs_n.device,
            dtype=obs_n.dtype,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=obs_n.device)
        for t in self.noise_scheduler.timesteps:
            if guidance_scale != 1.0:
                cond_pred = self._predict_noise(trajectory, obs_n, goal_n, t)
                uncond_pred = self._predict_noise(
                    trajectory,
                    torch.zeros_like(obs_n),
                    torch.zeros_like(goal_n),
                    t,
                )
                model_output = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                model_output = self._predict_noise(trajectory, obs_n, goal_n, t)
            trajectory = self.noise_scheduler.step(model_output, t, trajectory, generator=generator).prev_sample
        return trajectory

    @torch.no_grad()
    def predict_action_denormalized(
        self,
        obs_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        actions_n = self.predict_action(
            obs_hist,
            goal_hist,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return denormalize_minmax(actions_n, self.normalizer_stats.action)
