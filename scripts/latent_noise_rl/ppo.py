"""PPO update adapter over raw latent Gaussian coordinates."""

import torch

from scripts.gaussian_chunk_rl.ppo import GaussianPPOUpdater


class LatentNoisePPOUpdater(GaussianPPOUpdater):
    def __init__(self, policy, critic, cfg):
        super().__init__(policy, critic, cfg)
        actor_parameters = list(policy.actor.parameters()) + [policy.log_std]
        self.actor_optimizer = torch.optim.AdamW(actor_parameters, lr=cfg.actor_lr, weight_decay=0.0)
        optimized = {id(p) for group in self.actor_optimizer.param_groups for p in group["params"]}
        expected = {id(p) for p in actor_parameters}
        if optimized != expected:
            raise RuntimeError("Latent PPO optimizer contains parameters outside the noise actor.")

    def update(self, rollout, *, update_actor=True):
        metrics = super().update(rollout, update_actor=update_actor)
        metrics["Policy/raw_noise_entropy"] = metrics.pop("Policy/gaussian_entropy")
        metrics["Policy/raw_noise_std_mean"] = metrics.pop("Policy/std_mean")
        metrics["Policy/raw_noise_std_min"] = metrics.pop("Policy/std_min")
        metrics["Policy/raw_noise_std_max"] = metrics.pop("Policy/std_max")
        return metrics
