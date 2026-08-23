"""PPO update over reverse-diffusion transitions and physical-step values."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .buffer import RolloutBatch
from .config import DPPOConfig
from .critic import ValueCritic
from .policy import DPPODiffusionPolicy


@dataclass
class UpdateMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    approximate_kl: float = 0.0
    clip_fraction: float = 0.0
    ratio_mean: float = 1.0
    entropy: float = 0.0
    actor_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    actor_updates: int = 0
    critic_updates: int = 0
    early_stopped: bool = False

    def as_dict(self) -> dict[str, float]:
        actor_divisor = max(self.actor_updates, 1)
        critic_divisor = max(self.critic_updates, 1)
        return {
            "Loss/policy": self.policy_loss / actor_divisor,
            "Loss/value": self.value_loss / critic_divisor,
            "Policy/approximate_kl": self.approximate_kl / actor_divisor,
            "Policy/clip_fraction": self.clip_fraction / actor_divisor,
            "Policy/ratio_mean": self.ratio_mean / actor_divisor,
            "Policy/fixed_transition_entropy": self.entropy / actor_divisor,
            "Policy/actor_grad_norm": self.actor_grad_norm / actor_divisor,
            "Policy/critic_grad_norm": self.critic_grad_norm / critic_divisor,
            "Policy/actor_updates": float(self.actor_updates),
            "Policy/critic_updates": float(self.critic_updates),
            "Policy/early_stopped": float(self.early_stopped),
        }


class DPPOUpdater:
    def __init__(self, policy: DPPODiffusionPolicy, critic: ValueCritic, cfg: DPPOConfig):
        self.policy = policy
        self.critic = critic
        self.cfg = cfg
        self.actor_optimizer = torch.optim.AdamW(
            policy.trainable_parameters,
            lr=cfg.actor_lr,
            weight_decay=cfg.actor_weight_decay,
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=cfg.critic_lr,
            weight_decay=cfg.critic_weight_decay,
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )

    def _normalized_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-8)
        q = self.cfg.advantage_clip_quantile
        if q > 0.0 and len(normalized) > 2:
            low, high = torch.quantile(normalized, torch.tensor([q, 1.0 - q], device=normalized.device))
            normalized = normalized.clamp(low, high)
        return normalized

    @staticmethod
    def _value_loss(
        value: torch.Tensor,
        old_value: torch.Tensor,
        returns: torch.Tensor,
        *,
        value_clip: float | None,
        use_clipping: bool,
    ) -> torch.Tensor:
        loss_unclipped = (value - returns).square()
        if value_clip is None or not use_clipping:
            return 0.5 * loss_unclipped.mean()
        delta = (value - old_value).clamp(-value_clip, value_clip)
        clipped_value = old_value + delta
        loss_clipped = (clipped_value - returns).square()
        return 0.5 * torch.maximum(loss_unclipped, loss_clipped).mean()

    def update(self, rollout: RolloutBatch, *, update_actor: bool = True) -> dict[str, float]:
        batch = rollout.flatten()
        assert batch.advantages is not None and batch.returns is not None
        advantages = self._normalized_advantages(batch.advantages)
        metrics = UpdateMetrics(ratio_mean=0.0 if update_actor else 1.0)
        physical_count = batch.physical_batch_size
        denoising_count = self.cfg.finetune_denoising_steps
        total_actor = physical_count * denoising_count

        self.policy.train()
        for _ in range(self.cfg.update_epochs if update_actor else 0):
            permutation = torch.randperm(total_actor, device=batch.rewards.device)
            for start in range(0, total_actor, self.cfg.minibatch_size):
                flat = permutation[start : start + self.cfg.minibatch_size]
                physical_index = torch.div(flat, denoising_count, rounding_mode="floor")
                denoising_index = flat.remainder(denoising_count)
                new_logprob, entropy = self.policy.transition_logprob(
                    batch.proprio[physical_index],
                    batch.action_history[physical_index],
                    batch.goals[physical_index],
                    batch.chains[physical_index, denoising_index],
                    batch.chains[physical_index, denoising_index + 1],
                    denoising_index,
                )
                old_logprob = batch.old_logprobs[physical_index, denoising_index]
                logratio = new_logprob - old_logprob
                ratio = logratio.exp()
                denoising_discount = self.cfg.gamma_denoising ** (
                    denoising_count - denoising_index - 1
                )
                advantage = advantages[physical_index] * denoising_discount
                clip = self.policy.denoising_clip(denoising_index)
                unclipped = -advantage * ratio
                clipped = -advantage * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                loss = torch.maximum(unclipped, clipped).mean()
                approximate_kl = ((ratio - 1.0) - logratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip).float().mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.policy.model.parameters(), self.cfg.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite DPPO actor gradient.")
                self.actor_optimizer.step()
                metrics.policy_loss += float(loss.detach())
                metrics.approximate_kl += float(approximate_kl.detach())
                metrics.clip_fraction += float(clip_fraction.detach())
                metrics.ratio_mean += float(ratio.mean().detach())
                metrics.entropy += float(entropy.mean().detach())
                metrics.actor_grad_norm += float(grad_norm.detach())
                metrics.actor_updates += 1
                if float(approximate_kl.detach()) > self.cfg.target_kl:
                    metrics.early_stopped = True
                    break
            if metrics.early_stopped:
                break

        for _ in range(self.cfg.update_epochs):
            permutation = torch.randperm(physical_count, device=batch.rewards.device)
            for start in range(0, physical_count, self.cfg.critic_minibatch_size):
                index = permutation[start : start + self.cfg.critic_minibatch_size]
                value = self.critic(batch.critic_observation[index])
                # Never clip critic-only warm-up: with a distant target the
                # clipped branch can otherwise have exactly zero gradient.
                value_loss = self._value_loss(
                    value,
                    batch.old_values[index],
                    batch.returns[index],
                    value_clip=self.cfg.value_clip,
                    use_clipping=update_actor,
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.cfg.value_coef * value_loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite DPPO critic gradient.")
                self.critic_optimizer.step()
                metrics.value_loss += float(value_loss.detach())
                metrics.critic_grad_norm += float(grad_norm.detach())
                metrics.critic_updates += 1
        return metrics.as_dict()
