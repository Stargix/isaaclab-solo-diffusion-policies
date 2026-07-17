"""Solo12 LocoDiff policy using the paper's score-SDE/EDM formulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from ..train.data.normalization import (
        NormalizerStats,
        denormalize_minmax,
        normalize_minmax,
        normalize_zscore,
    )
except ImportError:  # pragma: no cover - script entrypoint
    from train.data.normalization import (
        NormalizerStats,
        denormalize_minmax,
        normalize_minmax,
        normalize_zscore,
    )

from .transformer_policy import TransformerDiffusionPolicy


@dataclass
class Solo12DiffusionPolicyConfig:
    proprio_dim: int = 33
    action_hist_dim: int = 0
    goal_dim: int = 5
    action_dim: int = 12
    history: int = 8
    prediction_horizon: int = 16
    execution_offset: int = 0
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3
    separate_goal_conditioning: bool = True
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    log_sigma_loc: float = -1.2
    log_sigma_scale: float = 1.2
    noise_distribution: str = "log_logistic"
    sampler: str = "euler"
    num_inference_steps: int = 3

    def __post_init__(self) -> None:
        if self.action_hist_dim != 0:
            raise ValueError("The paper conditions on state history, not action history.")
        if self.goal_dim != 5:
            raise ValueError("Expected [vx, vy, wz, skill_walk, skill_crouch].")
        if self.execution_offset != 0:
            raise ValueError("LocoDiff executes the first predicted future action.")
        if self.noise_distribution != "log_logistic":
            raise ValueError("The published training distribution is log-logistic.")
        if self.sampler not in {"euler", "heun"}:
            raise ValueError("sampler must be 'euler' or 'heun'.")
        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError("Require 0 < sigma_min < sigma_max.")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")


class Solo12DiffusionPolicy(torch.nn.Module):
    """Reward-free version of the published LocoDiff generative controller."""

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
        self.normalizer_stats = NormalizerStats.from_dict(stats) if isinstance(stats, dict) else stats

    def configure_optimizers(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas
        )

    def get_scalings(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma_data = self.cfg.sigma_data
        c_skip = sigma_data**2 / (sigma.square() + sigma_data**2)
        c_out = sigma * sigma_data / (sigma.square() + sigma_data**2).sqrt()
        c_in = 1.0 / (sigma.square() + sigma_data**2).sqrt()
        c_noise = 0.25 * torch.log(sigma)
        return c_skip, c_out, c_in, c_noise

    def forward_edm(
        self,
        noisy_actions: torch.Tensor,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = self.get_scalings(sigma)
        residual = self.model(
            c_in * noisy_actions,
            proprio_hist,
            action_hist,
            goal_hist,
            c_noise.reshape(-1),
        )
        return c_skip * noisy_actions + c_out * residual

    def sample_sigma(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw bounded log-logistic noise levels as stated in the paper."""

        u = torch.rand((batch_size, 1, 1), device=device, dtype=dtype, generator=generator)
        eps = torch.finfo(dtype).eps
        u = u.clamp(eps, 1.0 - eps)
        log_sigma = self.cfg.log_sigma_loc + self.cfg.log_sigma_scale * (
            torch.log(u) - torch.log1p(-u)
        )
        return log_sigma.exp().clamp(self.cfg.sigma_min, self.cfg.sigma_max)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.normalizer_stats is None:
            raise RuntimeError("Normalizer stats must be set before computing loss.")

        proprio = normalize_zscore(batch["proprio_hist"], self.normalizer_stats.proprio)
        action_hist = batch["action_hist"]
        condition = normalize_zscore(batch["goal_hist"], self.normalizer_stats.goal)
        clean_actions = normalize_minmax(batch["actions"], self.normalizer_stats.action)
        sigma = self.sample_sigma(
            clean_actions.shape[0], device=clean_actions.device, dtype=clean_actions.dtype
        )
        noisy_actions = clean_actions + torch.randn_like(clean_actions) * sigma
        denoised = self.forward_edm(noisy_actions, proprio, action_hist, condition, sigma)

        # Equation (5) in LocoDiff: direct x0 denoising error. The old code
        # multiplied it by an EDM weighting that is absent from that objective.
        return F.mse_loss(denoised, clean_actions)

    def sigma_schedule(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        steps = int(self.num_inference_steps)
        ramp = torch.linspace(0, 1, steps, device=device, dtype=dtype)
        max_inv = self.cfg.sigma_max ** (1.0 / self.cfg.rho)
        min_inv = self.cfg.sigma_min ** (1.0 / self.cfg.rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** self.cfg.rho
        return torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])

    @torch.no_grad()
    def predict_action(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.normalizer_stats is None:
            raise RuntimeError("Normalizer stats must be set before inference.")

        proprio = normalize_zscore(proprio_hist, self.normalizer_stats.proprio)
        action_history = action_hist
        condition = normalize_zscore(goal_hist, self.normalizer_stats.goal)
        sigmas = self.sigma_schedule(device=proprio.device, dtype=proprio.dtype)
        x = torch.randn(
            (proprio.shape[0], self.cfg.prediction_horizon, self.cfg.action_dim),
            device=proprio.device,
            dtype=proprio.dtype,
            generator=generator,
        ) * sigmas[0]

        for index in range(self.num_inference_steps):
            sigma = sigmas[index]
            sigma_next = sigmas[index + 1]
            sigma_batch = sigma.expand(proprio.shape[0], 1, 1)
            denoised = self.forward_edm(x, proprio, action_history, condition, sigma_batch)
            derivative = (x - denoised) / sigma
            proposal = x + (sigma_next - sigma) * derivative

            if self.cfg.sampler == "heun" and sigma_next > 0:
                next_batch = sigma_next.expand(proprio.shape[0], 1, 1)
                denoised_next = self.forward_edm(
                    proposal, proprio, action_history, condition, next_batch
                )
                derivative_next = (proposal - denoised_next) / sigma_next
                x = x + (sigma_next - sigma) * 0.5 * (derivative + derivative_next)
            else:
                x = proposal
        return x

    @torch.no_grad()
    def predict_action_denormalized(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        actions = self.predict_action(
            proprio_hist, action_hist, goal_hist, generator=generator
        )
        return denormalize_minmax(actions, self.normalizer_stats.action)

    def executable_chunk(self, trajectory: torch.Tensor, num_actions: int = 1) -> torch.Tensor:
        if not 1 <= num_actions <= self.cfg.prediction_horizon:
            raise ValueError("num_actions must be inside the predicted horizon.")
        return trajectory[:, :num_actions]
