"""PPO on the joint likelihood of the executable Gaussian chunk."""

import torch
from torch.distributions import Normal, kl_divergence
from scripts.dppo_diffusion_rl.ppo import DPPOUpdater, UpdateMetrics


class GaussianPPOUpdater:
    def __init__(self, policy, critic, cfg):
        self.policy, self.critic, self.cfg = policy, critic, cfg
        self.actor_optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.actor_lr, weight_decay=0.0)
        self.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)

    def _adapt_actor_lr(self, measured_kl, *, early_stopped):
        """Adjust the NEXT rollout's LR; this is not a hard trust region.

        The starting LR is the ceiling, so low KL cannot undo the conservative
        initialization. An early stop never triggers an LR increase.
        """
        lr = self.actor_optimizer.param_groups[0]["lr"]
        if measured_kl > 2.0 * self.cfg.target_kl:
            lr = max(self.cfg.min_actor_lr, lr / 1.5)
        elif not early_stopped and 0.0 < measured_kl < self.cfg.target_kl / 2.0:
            lr = min(self.cfg.actor_lr, lr * 1.5)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr
        return lr

    def update(self, rollout, *, update_actor=True):
        batch = rollout.flatten()
        advantage = batch.advantages
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        metrics = UpdateMetrics(ratio_mean=0.0 if update_actor else 1.0)
        count = batch.physical_batch_size
        max_observed_kl = 0.0
        observed_kl_sum, observed_batches = 0.0, 0
        actor_lr = self.actor_optimizer.param_groups[0]["lr"]
        self.policy.train()
        probe_index, old_probe = None, None
        if update_actor and self.cfg.adaptive_actor_lr:
            # A fixed, evenly spaced subset spans the whole flattened rollout.
            # Cache the OLD Gaussian, not just sampled-action log probabilities.
            probe_index = torch.linspace(0, count - 1, min(count, self.cfg.kl_probe_size),
                                         device=batch.rewards.device).long()
            with torch.no_grad():
                _, distribution = self.policy.distribution(batch.proprio[probe_index],
                    batch.action_history[probe_index], batch.goals[probe_index])
                old_probe = Normal(distribution.loc.detach().clone(), distribution.scale.detach().clone())
        for _ in range(self.cfg.update_epochs if update_actor else 0):
            permutation = torch.randperm(count, device=batch.rewards.device)
            for start in range(0, count, self.cfg.minibatch_size):
                index = permutation[start:start + self.cfg.minibatch_size]
                new, entropy = self.policy.logprob(batch.proprio[index], batch.action_history[index],
                                                    batch.goals[index], batch.chains[index, 0])
                logratio = new - batch.old_logprobs[index, 0]
                ratio = logratio.exp()
                approximate_kl = ((ratio - 1.0) - logratio).mean()
                if not torch.isfinite(ratio).all() or not torch.isfinite(approximate_kl):
                    raise FloatingPointError("Non-finite Gaussian PPO likelihood ratio.")
                max_observed_kl = max(max_observed_kl, approximate_kl.item())
                observed_kl_sum += approximate_kl.item()
                observed_batches += 1
                # Stop BEFORE another update once the measured change is too large.
                if approximate_kl.item() > self.cfg.target_kl:
                    metrics.early_stopped = True
                    break
                clipped = ratio.clamp(1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
                loss = -torch.minimum(ratio * advantage[index], clipped * advantage[index]).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite Gaussian PPO actor gradient.")
                self.actor_optimizer.step()
                self.policy.project_std()
                metrics.policy_loss += loss.item()
                metrics.approximate_kl += approximate_kl.item()
                metrics.clip_fraction += ((ratio - 1).abs() > self.cfg.clip_ratio).float().mean().item()
                metrics.ratio_mean += ratio.mean().item()
                metrics.entropy += entropy.mean().item()
                metrics.actor_grad_norm += norm.item()
                metrics.actor_updates += 1
            if metrics.early_stopped:
                break
        final_exact_kl = None
        if old_probe is not None:
            with torch.no_grad():
                _, distribution = self.policy.distribution(batch.proprio[probe_index],
                    batch.action_history[probe_index], batch.goals[probe_index])
                # KL(old || new), SUM over the executable 4x12 coordinates.
                exact = kl_divergence(old_probe, distribution).sum(dim=(-1, -2)).mean()
                if not torch.isfinite(exact):
                    raise FloatingPointError("Non-finite post-update Gaussian KL.")
                final_exact_kl = max(0.0, exact.item())
            # Retain the sampled guard as well: a benign probe must not cause
            # an LR increase after a large observed change elsewhere.
            self._adapt_actor_lr(max(final_exact_kl, max_observed_kl),
                                 early_stopped=metrics.early_stopped)
        for _ in range(self.cfg.update_epochs):
            permutation = torch.randperm(count, device=batch.rewards.device)
            for start in range(0, count, self.cfg.critic_minibatch_size):
                index = permutation[start:start + self.cfg.critic_minibatch_size]
                value = self.critic(batch.critic_observation[index])
                loss = DPPOUpdater._value_loss(value, batch.old_values[index], batch.returns[index],
                                               value_clip=None, use_clipping=False)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.cfg.value_coef * loss).backward()
                norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite Gaussian PPO critic gradient.")
                self.critic_optimizer.step()
                metrics.value_loss += loss.item()
                metrics.critic_grad_norm += norm.item()
                metrics.critic_updates += 1
        result = metrics.as_dict()
        result["Policy/accepted_minibatch_kl"] = result["Policy/approximate_kl"]
        # Include the rejected minibatch: otherwise a one-step update reports
        # exactly zero KL while its NEXT minibatch exceeds the stopping limit.
        result["Policy/approximate_kl"] = observed_kl_sum / max(observed_batches, 1)
        result["Policy/kl_observed_batches"] = float(observed_batches)
        result["Policy/actor_lr"] = actor_lr
        result["Policy/actor_lr_next"] = self.actor_optimizer.param_groups[0]["lr"]
        if final_exact_kl is not None:
            result["Policy/post_update_exact_kl"] = final_exact_kl
        result["Policy/gaussian_entropy"] = result.pop("Policy/fixed_transition_entropy")
        result["Policy/std_mean"] = self.policy.log_std.detach().exp().mean().item()
        result["Policy/std_min"] = self.policy.log_std.detach().exp().min().item()
        result["Policy/std_max"] = self.policy.log_std.detach().exp().max().item()
        result["Policy/max_observed_kl"] = max_observed_kl
        return result
