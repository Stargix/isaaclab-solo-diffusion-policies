"""Inference-only optimization for the spatial DDPM denoiser."""

from __future__ import annotations

import torch


def trace_denoiser(policy: torch.nn.Module, device: torch.device) -> None:
    cfg = policy.cfg
    example = (
        torch.zeros((1, cfg.prediction_horizon, cfg.action_dim), device=device),
        torch.zeros((1, cfg.history, cfg.proprio_dim), device=device),
        torch.zeros((1, cfg.history, cfg.action_hist_dim), device=device),
        torch.zeros((1, cfg.history, cfg.goal_dim), device=device),
        torch.tensor([cfg.num_train_timesteps - 1], device=device, dtype=torch.long),
    )
    traced = torch.jit.trace(policy.model, example, strict=False, check_trace=False)
    policy.model = torch.jit.optimize_for_inference(traced)

