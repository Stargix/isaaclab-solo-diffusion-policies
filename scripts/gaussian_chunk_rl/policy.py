"""Gaussian exploration around the existing BC mean; exact raw-action density."""

from dataclasses import fields
import math

import torch
from torch.distributions import Normal

from scripts.diffusion_policy.model.solo12_deterministic_chunk_policy import (
    Solo12DeterministicChunkPolicy, Solo12DeterministicChunkPolicyConfig,
)
from scripts.diffusion_policy.train.data.normalization import (
    denormalize_minmax, normalize_minmax, normalize_zscore,
)
from scripts.diffusion_policy.train.runtime.checkpoint import load_training_checkpoint
from scripts.dppo_diffusion_rl.policy import DPPOSample
from .config import GaussianPPOConfig

POLICY_KIND = "spatial_hindsight_height_profile_deterministic_chunk_bc"
ALGORITHM = "gaussian_chunk_ppo"


def model_config(config: dict) -> Solo12DeterministicChunkPolicyConfig:
    values = {**config["model"], **{
        name: config["dataset"][name]
        for name in ("history", "prediction_horizon", "execution_offset")
    }}
    values = {f.name: values[f.name] for f in fields(Solo12DeterministicChunkPolicyConfig)
              if f.name in values}
    return Solo12DeterministicChunkPolicyConfig(**values)


def load_bc_policy(path, device, cfg: GaussianPPOConfig):
    source = load_training_checkpoint(path, device, expected_policy_kind=POLICY_KIND)
    if source.get("algorithm") != "deterministic_chunk_bc":
        raise ValueError("This pilot must start from pure deterministic BC, not an RL actor.")
    data = source["config"]["dataset"]
    if (data.get("goal_representation") != "hindsight_geom_profile16"
            or not data.get("include_padded_starts") or float(data.get("dt", 0)) != 0.02):
        raise ValueError("Expected the padded-start profile16, 50-Hz Phase-A contract.")
    mean = Solo12DeterministicChunkPolicy(model_config(source["config"]))
    if (mean.cfg.proprio_dim, mean.cfg.goal_dim, mean.cfg.action_dim, mean.cfg.action_hist_dim) != (30, 16, 12, 12):
        raise ValueError("Checkpoint observation/action dimensions do not match Solo12.")
    mean.load_state_dict(source["ema_model_state_dict"], strict=True)
    mean.set_normalizer_stats(source["normalizer_stats"])
    stats = mean.normalizer_stats
    for array in (stats.proprio.mean, stats.proprio.std, stats.goal.mean, stats.goal.std,
                  stats.action.min, stats.action.max):
        if not torch.isfinite(torch.as_tensor(array)).all():
            raise ValueError("Non-finite checkpoint normalizers.")
    if not torch.all(torch.as_tensor(stats.action.max) > torch.as_tensor(stats.action.min)):
        raise ValueError("Degenerate action normalizer range.")
    if any(not torch.isfinite(p).all() for p in mean.parameters()):
        raise ValueError("Non-finite BC parameters.")
    return source, GaussianChunkPolicy(mean, cfg).to(device)


class GaussianChunkPolicy(torch.nn.Module):
    def __init__(self, mean_policy, ppo_cfg: GaussianPPOConfig):
        super().__init__()
        self.policy = mean_policy
        self.ppo_cfg = ppo_cfg
        ppo_cfg.validate(prediction_horizon=self.cfg.prediction_horizon,
                         execution_offset=self.cfg.execution_offset)
        # One learned scale per joint, shared across the four executable times.
        self.log_std = torch.nn.Parameter(torch.full((self.cfg.action_dim,), math.log(ppo_cfg.initial_std)))
        self.train()

    @property
    def cfg(self):
        return self.policy.cfg

    @property
    def dppo_cfg(self):
        """Collector protocol alias only: physical horizon and GAE, not denoising."""
        return self.ppo_cfg

    def train(self, mode=True):
        super().train(mode)
        # PPO's density must not contain untracked attention-dropout randomness.
        self.policy.eval()
        return self

    def distribution(self, proprio, action, goals):
        trajectory = self.policy._normalized_prediction(proprio, action, goals)
        mean = self.executable_chunk(trajectory)
        std = self.log_std.clamp(math.log(self.ppo_cfg.min_std), math.log(self.ppo_cfg.max_std)).exp()
        return trajectory, Normal(mean, std)

    def executable_chunk(self, trajectory):
        return self.policy.executable_chunk(trajectory, self.ppo_cfg.exec_horizon)

    @torch.no_grad()
    def sample(self, proprio, action, goals):
        trajectory, distribution = self.distribution(proprio, action, goals)
        raw = distribution.sample()
        logprob = distribution.log_prob(raw).sum(dim=(-1, -2))
        executed = trajectory.clone()
        start = self.cfg.execution_offset
        executed[:, start:start + self.ppo_cfg.exec_horizon] = raw
        bounded = executed.clamp(-1.0, 1.0)
        # The common collector stores this single latent sample in its chains
        # field. There is NO diffusion chain. Keep raw draws, NOT clipped actions.
        return DPPOSample(bounded, denormalize_minmax(bounded, self.policy.normalizer_stats.action),
                          raw[:, None], logprob[:, None])

    def logprob(self, proprio, action, goals, raw):
        _, distribution = self.distribution(proprio, action, goals)
        return (distribution.log_prob(raw).sum(dim=(-1, -2)),
                distribution.entropy().sum(dim=(-1, -2)))

    def critic_condition(self, proprio, action, goals, privileged):
        stats = self.policy.normalizer_stats
        return torch.cat((normalize_zscore(proprio, stats.proprio).flatten(1),
                          normalize_minmax(action, stats.action).flatten(1),
                          normalize_zscore(goals, stats.goal).flatten(1), privileged), dim=1)

    @torch.no_grad()
    def project_std(self):
        self.log_std.clamp_(math.log(self.ppo_cfg.min_std), math.log(self.ppo_cfg.max_std))
