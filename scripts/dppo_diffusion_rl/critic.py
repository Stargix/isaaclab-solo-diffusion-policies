"""Asymmetric value function for path-conditioned DPPO."""

from __future__ import annotations

import torch
from torch import nn


class ValueCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        if input_dim <= 0 or not hidden_dims:
            raise ValueError("ValueCritic dimensions must be positive.")
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(current, width), nn.Mish()))
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)
