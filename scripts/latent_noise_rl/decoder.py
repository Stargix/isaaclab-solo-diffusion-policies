"""Frozen DDIM mapping from an initial latent to normalized action trajectories."""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler

from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicy
from scripts.diffusion_policy.train.data.normalization import normalize_minmax, normalize_zscore


class FrozenDDIMDecoder(torch.nn.Module):
    def __init__(self, prior: Solo12DiffusionPolicy, *, inference_steps: int = 10):
        super().__init__()
        self.prior = prior
        source = prior.noise_scheduler.config
        self.scheduler = DDIMScheduler(
            num_train_timesteps=int(source.num_train_timesteps),
            beta_start=float(source.beta_start),
            beta_end=float(source.beta_end),
            beta_schedule=str(source.beta_schedule),
            trained_betas=getattr(source, "trained_betas", None),
            clip_sample=bool(source.clip_sample),
            clip_sample_range=float(source.clip_sample_range),
            prediction_type=str(source.prediction_type),
            thresholding=bool(source.thresholding),
            dynamic_thresholding_ratio=float(source.dynamic_thresholding_ratio),
            sample_max_value=float(source.sample_max_value),
            set_alpha_to_one=True,
            steps_offset=0,
            timestep_spacing="leading",
            rescale_betas_zero_snr=False,
        )
        self.inference_steps = int(inference_steps)
        if self.inference_steps != int(source.num_train_timesteps):
            raise ValueError("The pilot pins DDIM inference steps to the checkpoint's 10-step training schedule.")
        self.scheduler.set_timesteps(self.inference_steps)
        if [int(t) for t in self.scheduler.timesteps] != list(range(self.inference_steps - 1, -1, -1)):
            raise ValueError("Unexpected DDIM timestep sequence; refusing an unrecorded schedule.")
        self.validate_causal_mask()
        for parameter in self.prior.parameters():
            parameter.requires_grad_(False)
        self.prior.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.prior.eval()
        return self

    def validate_causal_mask(self) -> None:
        transformer = self.prior.model.model
        mask = getattr(transformer, "mask", None)
        horizon = self.prior.cfg.prediction_horizon
        if mask is None or tuple(mask.shape) != (horizon, horizon):
            raise ValueError("The loaded prior has no compatible action-token causal mask.")
        expected = torch.zeros_like(mask)
        future = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
        expected[future] = float("-inf")
        if not torch.equal(mask, expected):
            raise ValueError("The prior's action-token attention is not the expected causal mask.")

        memory_mask = getattr(transformer, "memory_mask", None)
        history = self.prior.cfg.history
        expected_shape = (horizon, 1 + 2 * history)
        if memory_mask is None or tuple(memory_mask.shape) != expected_shape:
            raise ValueError("The prior has no compatible causal observation/goal memory mask.")
        expected_memory = torch.full_like(memory_mask, float("-inf"))
        expected_memory[:, 0] = 0.0  # diffusion-time condition
        for target_idx in range(horizon):
            visible_history = min(target_idx + 1, history)
            expected_memory[target_idx, 1:1 + visible_history] = 0.0
            goal_start = 1 + history
            expected_memory[target_idx, goal_start:goal_start + visible_history] = 0.0
        if not torch.equal(memory_mask, expected_memory):
            raise ValueError("The prior's observation/goal memory mask is not causal as expected.")

    def scheduler_config(self) -> dict:
        # Diffusers exposes FrozenDict here; copying it with ``dict`` works
        # across its supported versions, while FrozenDict.to_dict is absent.
        return dict(self.scheduler.config)

    @torch.no_grad()
    def decode(self, proprio_hist: torch.Tensor, action_hist: torch.Tensor,
               goal_hist: torch.Tensor, initial_noise: torch.Tensor) -> torch.Tensor:
        """Return clipped normalized x0; scheduler evolution is deterministic."""
        if self.prior.normalizer_stats is None:
            raise RuntimeError("Prior normalizers must be loaded before decoding.")
        expected = (proprio_hist.shape[0], self.prior.cfg.prediction_horizon, self.prior.cfg.action_dim)
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"initial_noise must have shape {expected}, got {tuple(initial_noise.shape)}.")
        if initial_noise.device != proprio_hist.device or initial_noise.dtype != proprio_hist.dtype:
            raise ValueError("Latent and histories must share dtype and device.")
        stats = self.prior.normalizer_stats
        p = normalize_zscore(proprio_hist, stats.proprio)
        a = normalize_minmax(action_hist, stats.action)
        g = normalize_zscore(goal_hist, stats.goal)
        sample = initial_noise.clone()
        self.scheduler.set_timesteps(self.inference_steps, device=sample.device)
        for timestep in self.scheduler.timesteps:
            t_batch = torch.full((sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long)
            eps = self.prior._predict_noise(sample, p, a, g, t_batch)
            sample = self.scheduler.step(
                eps, timestep, sample, eta=0.0, use_clipped_model_output=False,
            ).prev_sample
        if not torch.isfinite(sample).all():
            raise FloatingPointError("DDIM decoder produced non-finite actions.")
        return sample.clamp(-1.0, 1.0)
