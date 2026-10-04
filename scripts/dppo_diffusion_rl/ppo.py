"""PPO update over reverse-diffusion transitions and physical-step values."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .buffer import RolloutBatch
from .config import DPPOConfig
from .critic import ValueCritic
from .policy import DPPODiffusionPolicy
from .symmetry import LeftRightAugmentation


@dataclass
class UpdateMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    approximate_kl: float = 0.0
    reference_kl: float = 0.0
    reference_kl_loss: float = 0.0
    clip_fraction: float = 0.0
    ratio_mean: float = 1.0
    entropy: float = 0.0
    actor_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    actor_updates: int = 0
    critic_updates: int = 0
    early_stopped: bool = False
    mirrored_ratio_mean: float = 0.0
    mirrored_logratio_abs: float = 0.0
    mirrored_updates: int = 0

    def as_dict(self) -> dict[str, float]:
        actor_divisor = max(self.actor_updates, 1)
        critic_divisor = max(self.critic_updates, 1)
        return {
            "Loss/policy": self.policy_loss / actor_divisor,
            "Loss/value": self.value_loss / critic_divisor,
            "Policy/approximate_kl": self.approximate_kl / actor_divisor,
            "Policy/reference_kl": self.reference_kl / actor_divisor,
            "Loss/reference_kl": self.reference_kl_loss / actor_divisor,
            "Policy/clip_fraction": self.clip_fraction / actor_divisor,
            "Policy/ratio_mean": self.ratio_mean / actor_divisor,
            "Policy/fixed_transition_entropy": self.entropy / actor_divisor,
            "Policy/actor_grad_norm": self.actor_grad_norm / actor_divisor,
            "Policy/critic_grad_norm": self.critic_grad_norm / critic_divisor,
            "Policy/actor_updates": float(self.actor_updates),
            "Policy/critic_updates": float(self.critic_updates),
            "Policy/early_stopped": float(self.early_stopped),
            "Symmetry/mirrored_ratio_mean": self.mirrored_ratio_mean / max(self.mirrored_updates, 1),
            "Symmetry/mirrored_logratio_abs": self.mirrored_logratio_abs / max(self.mirrored_updates, 1),
            "Symmetry/mirrored_minibatches": float(self.mirrored_updates),
        }


class DPPOUpdater:
    def __init__(self, policy: DPPODiffusionPolicy, critic: ValueCritic, cfg: DPPOConfig):
        self.policy = policy
        self.critic = critic
        self.cfg = cfg
        self.symmetry = (
            LeftRightAugmentation(policy) if cfg.symmetry_augmentation == "mirror" else None
        )
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
    def _reference_kl_for_group(
        reference_kl: torch.Tensor,
        update_groups: torch.Tensor | None,
        required_group: int | None,
    ) -> torch.Tensor:
        """Average reference KL globally or on one private update group.

        Multiplying the empty case by zero keeps a differentiable scalar and
        makes asynchronous minibatches safe.  The group is deliberately not
        part of either policy observation.
        """

        if required_group is None:
            return reference_kl.mean()
        if update_groups is None:
            raise ValueError(
                "reference_kl_update_group was configured, but rollout update groups are missing."
            )
        selected = update_groups == required_group
        if not torch.any(selected):
            return reference_kl.sum() * 0.0
        return reference_kl[selected].mean()

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
        copies = 1 if self.symmetry is None else 2

        self.policy.train()
        for _ in range(self.cfg.update_epochs if update_actor else 0):
            permutation = torch.randperm(total_actor, device=batch.rewards.device)
            for start in range(0, total_actor, self.cfg.minibatch_size):
                flat = permutation[start : start + self.cfg.minibatch_size]
                physical_index = torch.div(flat, denoising_count, rounding_mode="floor")
                denoising_index = flat.remainder(denoising_count)
                self.actor_optimizer.zero_grad(set_to_none=True)
                # Average identity/reflection gradients at the SAME weights,
                # with one optimizer step. Sequential copies avoid doubling
                # peak activation memory or the number of Adam updates.
                for copy_index in range(copies):
                    proprio = batch.proprio[physical_index]
                    action_history = batch.action_history[physical_index]
                    goals = batch.goals[physical_index]
                    previous = batch.chains[physical_index, denoising_index]
                    following = batch.chains[physical_index, denoising_index + 1]
                    if copy_index == 1:
                        assert self.symmetry is not None
                        proprio, action_history, goals = self.symmetry.reflect_conditions(
                            proprio, action_history, goals
                        )
                        previous = self.symmetry.reflect_latent(previous)
                        following = self.symmetry.reflect_latent(following)
                    transition_statistics = self.policy.transition_logprob(
                        proprio, action_history, goals, previous, following,
                        denoising_index,
                        include_reference_kl=self.cfg.reference_kl_coef > 0.0,
                    )
                    if self.cfg.reference_kl_coef > 0.0:
                        new_logprob, entropy, reference_kl = transition_statistics
                    else:
                        new_logprob, entropy = transition_statistics
                        reference_kl = torch.zeros((), device=new_logprob.device)
                    # Eq. 6: original behaviour density/advantage are retained
                    # for the orthogonal reflected transition (Jacobian=1).
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
                    policy_loss = torch.maximum(unclipped, clipped).mean()
                    minibatch_groups = (
                        None if batch.update_groups is None else batch.update_groups[physical_index]
                    )
                    reference_kl_mean = self._reference_kl_for_group(
                        reference_kl, minibatch_groups, self.cfg.reference_kl_update_group
                    )
                    reference_kl_loss = self.cfg.reference_kl_coef * reference_kl_mean
                    ((policy_loss + reference_kl_loss) / copies).backward()
                    clip_fraction = ((ratio - 1.0).abs() > clip).float().mean()
                    if copy_index == 0:
                        # Only real on-policy samples estimate update KL. The
                        # mirrored mismatch is not a policy-update KL and must
                        # not prematurely trip its early-stop threshold.
                        approximate_kl = ((ratio - 1.0) - logratio).mean().detach()
                    else:
                        metrics.mirrored_ratio_mean += float(ratio.mean().detach())
                        metrics.mirrored_logratio_abs += float(logratio.abs().mean().detach())
                        metrics.mirrored_updates += 1
                    metrics.policy_loss += float(policy_loss.detach()) / copies
                    metrics.reference_kl += float(reference_kl_mean.detach()) / copies
                    metrics.reference_kl_loss += float(reference_kl_loss.detach()) / copies
                    metrics.clip_fraction += float(clip_fraction.detach()) / copies
                    metrics.ratio_mean += float(ratio.mean().detach()) / copies
                    metrics.entropy += float(entropy.mean().detach()) / copies
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.policy.model.parameters(), self.cfg.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite DPPO actor gradient.")
                self.actor_optimizer.step()
                metrics.approximate_kl += float(approximate_kl.detach())
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
                self.critic_optimizer.zero_grad(set_to_none=True)
                for copy_index in range(copies):
                    critic_observation = batch.critic_observation[index]
                    if copy_index == 1:
                        assert self.symmetry is not None
                        critic_observation = self.symmetry.reflect_critic(
                            batch.proprio[index], batch.action_history[index],
                            batch.goals[index], critic_observation,
                        )
                    value = self.critic(critic_observation)
                    # Return labels belong to the equivalent augmented behaviour.
                    # Never clip critic-only warm-up, as in historical DPPO.
                    value_loss = self._value_loss(
                        value, batch.old_values[index], batch.returns[index],
                        value_clip=self.cfg.value_clip, use_clipping=update_actor,
                    )
                    (self.cfg.value_coef * value_loss / copies).backward()
                    metrics.value_loss += float(value_loss.detach()) / copies
                grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite DPPO critic gradient.")
                self.critic_optimizer.step()
                metrics.critic_grad_norm += float(grad_norm.detach())
                metrics.critic_updates += 1
        return metrics.as_dict()
