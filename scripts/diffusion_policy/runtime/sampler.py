"""Low-overhead, RNG-replayable inference for the pinned DPPO reverse chain."""

from __future__ import annotations

import torch

from scripts.diffusion_policy.train.data.normalization import denormalize_minmax


class CachedDPPOSampler:
    """WC-v3 inference sampler with immutable, device-resident DDPM scalars.

    The checkpoint's ten-step schedule is resolved once at construction. This
    avoids per-denoising-step CUDA scalar reads and scheduler dispatch without
    changing the FP32 reverse equations, random draws, actor, or schedule.
    PPO-only chain/log-probability tensors are not materialized at inference.
    """

    def __init__(self, policy: torch.nn.Module):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.dtype = next(policy.parameters()).dtype
        if policy.dppo_cfg.inference_steps != 10 or policy.dppo_cfg.exec_horizon != 4:
            raise ValueError("Cached sampler is pinned to WC-v3 K=10 / exec_horizon=4.")
        if self.dtype != torch.float32:
            raise ValueError(f"Cached sampler requires FP32, got {self.dtype}.")
        scheduler = policy.policy.noise_scheduler
        scheduler.set_timesteps(policy.dppo_cfg.inference_steps, device=self.device)
        trainable_start = policy.dppo_cfg.inference_steps - policy.dppo_cfg.finetune_denoising_steps
        steps = []
        for index, timestep_tensor in enumerate(scheduler.timesteps):
            timestep = int(timestep_tensor.item())
            previous = int(scheduler.previous_timestep(timestep_tensor).item())
            alpha_t = scheduler.alphas_cumprod[timestep].to(self.device)
            alpha_previous = (
                scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.one
            ).to(self.device)
            variance = scheduler._get_variance(timestep_tensor).to(self.device)
            steps.append((timestep, alpha_t, alpha_previous, variance,
                          index >= trainable_start))
        if len(steps) != policy.dppo_cfg.inference_steps:
            raise RuntimeError("Scheduler returned an unexpected reverse-step count.")
        self.steps = tuple(steps)

    @torch.inference_mode()
    def predict_action_denormalized(
        self,
        proprio_hist: torch.Tensor,
        action_hist: torch.Tensor,
        goal_hist: torch.Tensor,
        *,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if guidance_scale != 1.0:
            raise ValueError("WC-v3 DPPO inference requires guidance_scale=1.")
        if any(value.device != self.device for value in (proprio_hist, action_hist, goal_hist)):
            raise ValueError("Runtime histories and checkpoint must reside on the same device.")
        proprio_n, action_n, goal_n = self.policy._normalized_condition(
            proprio_hist, action_hist, goal_hist
        )
        sample = torch.randn(
            (proprio_n.shape[0], self.policy.cfg.prediction_horizon,
             self.policy.cfg.action_dim),
            device=proprio_n.device,
            dtype=proprio_n.dtype,
            generator=generator,
        )
        scheduler = self.policy.policy.noise_scheduler
        for timestep, alpha_t, alpha_previous, variance, trainable in self.steps:
            timestep_batch = torch.full(
                (sample.shape[0],), timestep, device=sample.device, dtype=torch.long
            )
            model_output = self.policy._model_output(
                sample, timestep_batch, proprio_n, action_n, goal_n, trainable=trainable
            )
            beta_product_t = 1.0 - alpha_t
            beta_product_previous = 1.0 - alpha_previous
            current_alpha = alpha_t / alpha_previous
            current_beta = 1.0 - current_alpha
            predicted_clean = (sample - beta_product_t.sqrt() * model_output) / alpha_t.sqrt()
            if scheduler.config.thresholding:
                predicted_clean = scheduler._threshold_sample(predicted_clean)
            elif scheduler.config.clip_sample:
                limit = float(scheduler.config.clip_sample_range)
                predicted_clean = predicted_clean.clamp(-limit, limit)
            mean = alpha_previous.sqrt() * current_beta / beta_product_t * predicted_clean
            mean = mean + current_alpha.sqrt() * beta_product_previous / beta_product_t * sample
            if timestep > 0:
                noise = torch.randn(
                    sample.shape, device=sample.device, dtype=sample.dtype, generator=generator
                )
                sample = mean + variance.sqrt() * noise
            else:
                sample = mean
        sample = sample.clamp(-1.0, 1.0)
        return denormalize_minmax(sample, self.policy.normalizer_stats.action)

    def executable_chunk(self, trajectory: torch.Tensor, num_actions: int) -> torch.Tensor:
        """Delegate the checkpoint-versioned executable slice to the actor."""
        return self.policy.executable_chunk(trajectory, num_actions)
