"""Inference-only optimizations that preserve trained weights and SDE math."""

from __future__ import annotations

import math
import torch


def trace_denoiser(policy: torch.nn.Module, device: torch.device) -> None:
    """Replace the Python denoiser with a dynamic-batch TorchScript graph.

    This removes part of the per-denoising-step Python/module overhead on Windows.
    The probability-flow ODE sampler remains unchanged.
    """

    cfg = policy.cfg
    example = (
        torch.zeros((1, cfg.prediction_horizon, cfg.action_dim), device=device),
        torch.zeros((1, cfg.history, cfg.proprio_dim), device=device),
        torch.zeros((1, cfg.history, cfg.action_hist_dim), device=device),
        torch.zeros((1, cfg.history, cfg.goal_dim), device=device),
        torch.tensor([0.25 * math.log(cfg.sigma_max)], device=device),
    )
    traced = torch.jit.trace(policy.model, example, strict=False, check_trace=False)
    policy.model = torch.jit.optimize_for_inference(traced)
