"""GPU rollout storage at the physical action-chunk time scale."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    proprio: torch.Tensor
    action_history: torch.Tensor
    goals: torch.Tensor
    critic_observation: torch.Tensor
    chains: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def compute_gae(
        self,
        last_value: torch.Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.rewards.ndim != 2:
            raise ValueError("Rollout tensors must have shape [chunks, environments].")
        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros_like(last_value)
        for step in reversed(range(self.rewards.shape[0])):
            next_value = last_value if step == self.rewards.shape[0] - 1 else self.old_values[step + 1]
            nonterminal = (~self.dones[step]).float()
            delta = self.rewards[step] + gamma * next_value * nonterminal - self.old_values[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step] = gae
        self.advantages = advantages
        self.returns = advantages + self.old_values

    @property
    def physical_batch_size(self) -> int:
        return int(self.rewards.numel())

    def flatten(self) -> "RolloutBatch":
        if self.advantages is None or self.returns is None:
            raise RuntimeError("compute_gae must be called before flattening the rollout.")
        leading = self.rewards.shape[:2]

        def merge(value: torch.Tensor) -> torch.Tensor:
            if value.shape[:2] != leading:
                raise ValueError("Every rollout tensor must share [chunks, environments].")
            return value.reshape(-1, *value.shape[2:])

        return RolloutBatch(
            proprio=merge(self.proprio),
            action_history=merge(self.action_history),
            goals=merge(self.goals),
            critic_observation=merge(self.critic_observation),
            chains=merge(self.chains),
            old_logprobs=merge(self.old_logprobs),
            old_values=merge(self.old_values),
            rewards=merge(self.rewards),
            dones=merge(self.dones),
            advantages=merge(self.advantages),
            returns=merge(self.returns),
        )
