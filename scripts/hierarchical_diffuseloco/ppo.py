"""Small, dependency-light PPO implementation for the high-level command policy."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Normal

from .contracts import ActionBounds


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, bounds: ActionBounds = ActionBounds(), hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        previous = observation_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        self.body = nn.Sequential(*layers)
        self.actor = nn.Linear(previous, 4)
        self.critic = nn.Linear(previous, 1)
        self.log_std = nn.Parameter(torch.full((4,), -1.0))
        self.register_buffer("low", torch.as_tensor(bounds.low))
        self.register_buffer("high", torch.as_tensor(bounds.high))

    def _distribution(self, observation: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        latent = self.body(observation)
        return Normal(self.actor(latent), self.log_std.clamp(-5.0, 1.0).exp()), latent

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, latent = self._distribution(observation)
        raw = distribution.rsample()
        squashed = torch.tanh(raw)
        action = self.low + 0.5 * (squashed + 1.0) * (self.high - self.low)
        log_prob = distribution.log_prob(raw).sum(-1) - torch.log1p(-squashed.square() + 1e-6).sum(-1)
        return action, log_prob, self.critic(latent).squeeze(-1)

    def evaluate(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, latent = self._distribution(observation)
        normalized = 2.0 * (action - self.low) / (self.high - self.low).clamp_min(1e-6) - 1.0
        squashed = normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(squashed)
        log_prob = distribution.log_prob(raw).sum(-1) - torch.log1p(-squashed.square() + 1e-6).sum(-1)
        entropy = distribution.entropy().sum(-1)
        return log_prob, entropy, self.critic(latent).squeeze(-1)


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 256
    epochs: int = 4
    minibatch_size: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0


def generalized_advantage(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                          last_value: torch.Tensor, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step].float()
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
    return advantages, advantages + values


class PPO:
    def __init__(self, observation_dim: int, bounds: ActionBounds = ActionBounds(), cfg: PPOConfig = PPOConfig(),
                 device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.cfg = cfg
        self.policy = ActorCritic(observation_dim, bounds).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)

    def update(self, observations: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor,
               returns: torch.Tensor, advantages: torch.Tensor) -> dict[str, float]:
        observations, actions = observations.to(self.device), actions.to(self.device)
        old_log_probs, returns, advantages = (old_log_probs.to(self.device), returns.to(self.device), advantages.to(self.device))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = observations.shape[0]
        metrics: dict[str, float] = {}
        for _ in range(self.cfg.epochs):
            for indices in torch.randperm(n, device=self.device).split(self.cfg.minibatch_size):
                log_prob, entropy, value = self.policy.evaluate(observations[indices], actions[indices])
                ratio = (log_prob - old_log_probs[indices]).exp()
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantages[indices]
                actor_loss = -torch.minimum(ratio * advantages[indices], clipped).mean()
                value_loss = 0.5 * torch.square(value - returns[indices]).mean()
                loss = actor_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                metrics = {"loss": float(loss.detach()), "actor_loss": float(actor_loss.detach()),
                           "value_loss": float(value_loss.detach()), "entropy": float(entropy.mean().detach())}
        return metrics
