"""Scalar-only evaluation intervention, independent of Isaac Sim."""
from __future__ import annotations
import torch


def replace_pace_scalar(goals: torch.Tensor, values) -> torch.Tensor:
    """Replace profile16's final scalar without changing the 15 other values."""
    if goals.ndim != 2 or goals.shape[1] != 16:
        raise ValueError("Scalar-only diagnostic requires (N, 16) profile goals")
    values = torch.as_tensor(values, dtype=goals.dtype, device=goals.device)
    if values.shape != goals[:, -1].shape or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Require one finite nonnegative scalar per environment")
    result = goals.clone()
    result[:, -1] = values
    return result
