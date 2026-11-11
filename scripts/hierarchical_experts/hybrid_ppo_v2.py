"""Selected-branch PPO update for the V2 hybrid action distribution."""

import math

import torch
from scripts.dppo_diffusion_rl.ppo import DPPOUpdater


class HybridPPOUpdaterV2:
    def __init__(self, policy, critic, *, actor_lr=1e-4, critic_lr=1e-3,
                 update_epochs=5, minibatch_size=8192, critic_minibatch_size=4096,
                 clip_ratio=0.2, target_kl=0.02, max_grad_norm=1.0,
                 value_coef=0.5, discrete_entropy_coef=0.01,
                 continuous_entropy_coef=0.0):
        for name, value in (("actor_lr", actor_lr), ("critic_lr", critic_lr),
                            ("clip_ratio", clip_ratio), ("target_kl", target_kl),
                            ("max_grad_norm", max_grad_norm), ("value_coef", value_coef)):
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (("discrete_entropy_coef", discrete_entropy_coef),
                            ("continuous_entropy_coef", continuous_entropy_coef)):
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0 < clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0, 1).")
        if update_epochs < 1 or minibatch_size < 1 or critic_minibatch_size < 1:
            raise ValueError("PPO epochs and minibatch sizes must be positive.")
        self.policy, self.critic = policy, critic
        self.actor_lr, self.critic_lr = float(actor_lr), float(critic_lr)
        self.update_epochs = int(update_epochs)
        self.minibatch_size, self.critic_minibatch_size = int(minibatch_size), int(critic_minibatch_size)
        self.clip_ratio, self.target_kl = float(clip_ratio), float(target_kl)
        self.max_grad_norm, self.value_coef = float(max_grad_norm), float(value_coef)
        self.discrete_entropy_coef = float(discrete_entropy_coef)
        self.continuous_entropy_coef = float(continuous_entropy_coef)
        self.actor_optimizer = torch.optim.AdamW(policy.parameters(), lr=self.actor_lr, weight_decay=0.0)
        self.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=self.critic_lr, weight_decay=0.0)

    def update(self, batch):
        advantages = batch.advantages
        if advantages is None or batch.returns is None:
            raise RuntimeError("Hybrid rollout must have GAE advantages and returns.")
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        count = batch.rewards.numel()
        if count % self.minibatch_size or count % self.critic_minibatch_size:
            raise ValueError("Rollout size must be divisible by both configured minibatch sizes.")
        stats = {name: 0.0 for name in (
            "loss", "discrete_kl", "continuous_kl", "discrete_clip", "continuous_clip",
            "discrete_ratio", "continuous_ratio", "discrete_entropy", "continuous_entropy",
            "actor_grad", "critic_loss", "critic_grad", "observed_discrete_kl",
            "observed_continuous_kl", "max_discrete_kl", "max_continuous_kl")}
        actor_updates = critic_updates = 0
        observed_kl_batches = 0
        early_stopped = False
        self.policy.train()
        for _ in range(self.update_epochs):
            order = torch.randperm(count, device=batch.rewards.device)
            for start in range(0, count, self.minibatch_size):
                idx = order[start:start + self.minibatch_size]
                new_d, new_c, entropy_d, entropy_c = self.policy.logprobs(
                    batch.proprio[idx], batch.action_history[idx], batch.goals[idx],
                    batch.skills[idx], batch.command_latents[idx])
                logratio_d = new_d - batch.old_discrete_logprobs[idx]
                logratio_c = new_c - batch.old_continuous_logprobs[idx]
                if not all(torch.isfinite(value).all() for value in (new_d, new_c, logratio_d, logratio_c)):
                    raise FloatingPointError("Non-finite hybrid PPO log probability.")
                ratio_d, ratio_c = logratio_d.exp(), logratio_c.exp()
                if not torch.isfinite(ratio_d).all() or not torch.isfinite(ratio_c).all():
                    raise FloatingPointError("Non-finite hybrid PPO ratio.")
                kl_d = ((ratio_d - 1.0) - logratio_d).mean()
                kl_c = ((ratio_c - 1.0) - logratio_c).mean()
                if not torch.isfinite(kl_d) or not torch.isfinite(kl_c):
                    raise FloatingPointError("Non-finite hybrid PPO approximate KL.")
                stats["observed_discrete_kl"] += float(kl_d.detach())
                stats["observed_continuous_kl"] += float(kl_c.detach())
                stats["max_discrete_kl"] = max(stats["max_discrete_kl"], float(kl_d.detach()))
                stats["max_continuous_kl"] = max(stats["max_continuous_kl"], float(kl_c.detach()))
                observed_kl_batches += 1
                if kl_d.item() > self.target_kl or kl_c.item() > self.target_kl:
                    early_stopped = True
                    break
                clipped_d = ratio_d.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
                clipped_c = ratio_c.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
                loss_d = -torch.minimum(ratio_d * advantages[idx], clipped_d * advantages[idx]).mean()
                loss_c = -torch.minimum(ratio_c * advantages[idx], clipped_c * advantages[idx]).mean()
                loss = (loss_d + loss_c
                        - self.discrete_entropy_coef * entropy_d.mean()
                        - self.continuous_entropy_coef * entropy_c.mean())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite hybrid PPO actor loss.")
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if not torch.isfinite(grad):
                    raise FloatingPointError("Non-finite hybrid PPO actor gradient.")
                self.actor_optimizer.step()
                self.policy.project_std()
                for key, value in (("loss", loss), ("discrete_kl", kl_d), ("continuous_kl", kl_c),
                                   ("discrete_clip", ((ratio_d - 1).abs() > self.clip_ratio).float().mean()),
                                   ("continuous_clip", ((ratio_c - 1).abs() > self.clip_ratio).float().mean()),
                                   ("discrete_ratio", ratio_d.mean()), ("continuous_ratio", ratio_c.mean()),
                                   ("discrete_entropy", entropy_d.mean()), ("continuous_entropy", entropy_c.mean()),
                                   ("actor_grad", grad)):
                    stats[key] += float(value.detach())
                actor_updates += 1
            if early_stopped:
                break

        for _ in range(self.update_epochs):
            order = torch.randperm(count, device=batch.rewards.device)
            for start in range(0, count, self.critic_minibatch_size):
                idx = order[start:start + self.critic_minibatch_size]
                value = self.critic(batch.critic_observation[idx])
                loss = DPPOUpdater._value_loss(value, batch.old_values[idx], batch.returns[idx],
                                               value_clip=None, use_clipping=False)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * loss).backward()
                grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                if not torch.isfinite(grad):
                    raise FloatingPointError("Non-finite hybrid PPO critic gradient.")
                self.critic_optimizer.step()
                stats["critic_loss"] += float(loss.detach())
                stats["critic_grad"] += float(grad.detach())
                critic_updates += 1

        actor_divisor, critic_divisor = max(actor_updates, 1), max(critic_updates, 1)
        result = {"Policy/actor_updates": float(actor_updates), "Policy/critic_updates": float(critic_updates),
                  "Policy/early_stopped": float(early_stopped),
                  "Loss/policy": stats["loss"] / actor_divisor,
                  "Loss/value": stats["critic_loss"] / critic_divisor,
                  "Policy/discrete_approximate_kl": stats["discrete_kl"] / actor_divisor,
                  "Policy/continuous_approximate_kl": stats["continuous_kl"] / actor_divisor,
                  "Policy/discrete_kl_observed_mean": stats["observed_discrete_kl"] / max(observed_kl_batches, 1),
                  "Policy/continuous_kl_observed_mean": stats["observed_continuous_kl"] / max(observed_kl_batches, 1),
                  "Policy/discrete_kl_observed_max": stats["max_discrete_kl"],
                  "Policy/continuous_kl_observed_max": stats["max_continuous_kl"],
                  "Policy/kl_observed_minibatches": float(observed_kl_batches),
                  "Policy/discrete_clip_fraction": stats["discrete_clip"] / actor_divisor,
                  "Policy/continuous_clip_fraction": stats["continuous_clip"] / actor_divisor,
                  "Policy/discrete_ratio_mean": stats["discrete_ratio"] / actor_divisor,
                  "Policy/continuous_ratio_mean": stats["continuous_ratio"] / actor_divisor,
                  "Policy/discrete_entropy": stats["discrete_entropy"] / actor_divisor,
                  "Policy/continuous_entropy": stats["continuous_entropy"] / actor_divisor,
                  "Policy/actor_grad_norm": stats["actor_grad"] / actor_divisor,
                  "Policy/critic_grad_norm": stats["critic_grad"] / critic_divisor,
                  "Policy/command_std_walk_mean": self.policy.log_stds[0].exp().mean().item(),
                  "Policy/command_std_crouch_mean": self.policy.log_stds[1].exp().mean().item(),
                  "Policy/discrete_entropy_coef": self.discrete_entropy_coef,
                  "Policy/continuous_entropy_coef": self.continuous_entropy_coef}
        return result


__all__ = ["HybridPPOUpdaterV2"]
