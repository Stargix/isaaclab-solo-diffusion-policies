"""PPO policy over the causal prefix of the frozen DDIM initial noise."""

from __future__ import annotations

import math

import torch
from torch.distributions import Normal

from scripts.diffusion_policy.train.data.normalization import (
    denormalize_minmax, normalize_minmax, normalize_zscore,
)
from scripts.dppo_diffusion_rl.policy import DPPOSample
from scripts.gaussian_chunk_rl.config import GaussianPPOConfig

from .config import LatentPolicyConfig
from .decoder import FrozenDDIMDecoder

ALGORITHM = "latent_noise_ppo"
POLICY_BACKEND = "latent_noise_ddim"
CHECKPOINT_VERSION = 1
REFERENCE_MODES = ("checkpoint", "gaussian", "initial")


class LatentNoisePolicy(torch.nn.Module):
    def __init__(self, decoder: FrozenDDIMDecoder, ppo_cfg: GaussianPPOConfig,
                 latent_cfg: LatentPolicyConfig, *, reference_mode: str = "checkpoint"):
        super().__init__()
        self.decoder = decoder
        self.ppo_cfg = ppo_cfg
        self.latent_cfg = latent_cfg
        self.cfg = decoder.prior.cfg
        self.dppo_cfg = ppo_cfg  # Physical collector protocol only.
        latent_cfg.validate(prediction_horizon=self.cfg.prediction_horizon,
                            execution_offset=self.cfg.execution_offset,
                            action_dim=self.cfg.action_dim)
        obs_dim = self.cfg.history * (self.cfg.proprio_dim + self.cfg.action_hist_dim + self.cfg.goal_dim)
        output_dim = latent_cfg.noise_rows * latent_cfg.noise_dim
        widths = latent_cfg.hidden_dims
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, widths[0]), torch.nn.Tanh(),
            torch.nn.Linear(widths[0], widths[1]), torch.nn.Tanh(),
            torch.nn.Linear(widths[1], output_dim),
        )
        torch.nn.init.zeros_(self.actor[-1].weight)
        torch.nn.init.zeros_(self.actor[-1].bias)
        self.log_std = torch.nn.Parameter(torch.full(
            (latent_cfg.noise_rows, latent_cfg.noise_dim), math.log(ppo_cfg.initial_std)
        ))
        self.reference_mode = "checkpoint"
        self.set_reference_mode(reference_mode)
        self.decoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def set_reference_mode(self, mode: str) -> None:
        if mode not in REFERENCE_MODES:
            raise ValueError(f"reference mode must be one of {REFERENCE_MODES}.")
        self.reference_mode = mode

    @property
    def normalizer_stats(self):
        return self.decoder.prior.normalizer_stats

    def _features(self, proprio, action_history, goals):
        stats = self.normalizer_stats
        if stats is None:
            raise RuntimeError("Prior normalizers must be loaded before policy use.")
        return torch.cat((normalize_zscore(proprio, stats.proprio).flatten(1),
                          normalize_minmax(action_history, stats.action).flatten(1),
                          normalize_zscore(goals, stats.goal).flatten(1)), dim=1)

    def actor_distribution(self, proprio, action_history, goals):
        mean = self.actor(self._features(proprio, action_history, goals))
        mean = mean.reshape(-1, self.latent_cfg.noise_rows, self.latent_cfg.noise_dim)
        std = self.log_std.clamp(math.log(self.ppo_cfg.min_std),
                                 math.log(self.ppo_cfg.max_std)).exp()
        return mean, Normal(mean, std.expand_as(mean))

    def distribution(self, proprio, action_history, goals):
        return self.actor_distribution(proprio, action_history, goals)

    def critic_condition(self, proprio, action_history, goals, privileged):
        return torch.cat((self._features(proprio, action_history, goals), privileged), dim=1)

    def _noise(self, proprio, action_history, goals, *, raw_u=None, generator=None):
        mean, dist = self.actor_distribution(proprio, action_history, goals)
        if raw_u is None:
            eps = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            raw_u = mean + dist.scale * eps
        if tuple(raw_u.shape) != tuple(mean.shape):
            raise ValueError("raw latent sample shape does not match the actor distribution.")
        if self.reference_mode == "gaussian":
            # Unbounded standard-normal initialization used by the diffusion sampler.
            prefix = raw_u
        else:
            prefix = self.latent_cfg.noise_scale * torch.tanh(raw_u)
        full = torch.zeros((len(prefix), self.cfg.prediction_horizon, self.cfg.action_dim),
                           dtype=prefix.dtype, device=prefix.device)
        full[:, :self.latent_cfg.noise_rows] = prefix
        return raw_u, full, mean, dist

    @torch.no_grad()
    def sample(self, proprio, action_history, goals, *, generator=None):
        if self.reference_mode == "gaussian":
            raw_u = torch.randn((len(proprio), self.latent_cfg.noise_rows, self.latent_cfg.noise_dim),
                                dtype=proprio.dtype, device=proprio.device, generator=generator)
            prefix = raw_u
            mean, dist = torch.zeros_like(raw_u), Normal(
                torch.zeros_like(raw_u), torch.ones_like(raw_u))
        elif self.reference_mode == "initial":
            raw_u = torch.randn((len(proprio), self.latent_cfg.noise_rows, self.latent_cfg.noise_dim),
                                dtype=proprio.dtype, device=proprio.device, generator=generator)
            prefix = self.latent_cfg.noise_scale * torch.tanh(raw_u)
            mean, dist = torch.zeros_like(raw_u), Normal(
                torch.zeros_like(raw_u), torch.full_like(raw_u, self.latent_cfg.initial_std))
        else:
            raw_u, _, mean, dist = self._noise(proprio, action_history, goals, generator=generator)
            prefix = self.latent_cfg.noise_scale * torch.tanh(raw_u)
        full = torch.zeros((len(prefix), self.cfg.prediction_horizon, self.cfg.action_dim),
                           dtype=prefix.dtype, device=prefix.device)
        full[:, :self.latent_cfg.noise_rows] = prefix
        trajectory = self.decoder.decode(proprio, action_history, goals, full)
        logprob = dist.log_prob(raw_u).sum(dim=(-1, -2))
        start = self.cfg.execution_offset
        end = start + self.ppo_cfg.exec_horizon
        if end > self.cfg.prediction_horizon:
            raise ValueError("Executed action chunk exceeds the source prediction horizon.")
        # One latent draw occupies the collector's single-chain slot.
        return DPPOSample(trajectory, denormalize_minmax(trajectory, self.normalizer_stats.action),
                          raw_u[:, None], logprob[:, None])

    def logprob(self, proprio, action_history, goals, raw_u):
        _, dist = self.actor_distribution(proprio, action_history, goals)
        return (dist.log_prob(raw_u).sum(dim=(-1, -2)),
                dist.entropy().sum(dim=(-1, -2)))

    def executable_chunk(self, trajectory, num_actions=None):
        requested = self.ppo_cfg.exec_horizon if num_actions is None else num_actions
        if requested != self.ppo_cfg.exec_horizon:
            raise ValueError("Latent PPO must execute its trained four-action horizon.")
        start = self.cfg.execution_offset
        return trajectory[:, start:start + requested]

    def predict_action(self, proprio, action_history, goals, *, guidance_scale=None, generator=None):
        if guidance_scale not in (None, 1.0):
            raise ValueError("Latent DDIM inference is pinned to guidance_scale=1.")
        if generator is None:
            raise ValueError("Latent inference requires an explicit seeded torch.Generator.")
        return self.sample(proprio, action_history, goals, generator=generator).normalized_trajectory

    @torch.no_grad()
    def predict_action_denormalized(self, proprio, action_history, goals, *, guidance_scale=None, generator=None):
        if guidance_scale not in (None, 1.0):
            raise ValueError("Latent DDIM inference is pinned to guidance_scale=1.")
        if generator is None:
            raise ValueError("Latent inference requires an explicit seeded torch.Generator.")
        return self.sample(proprio, action_history, goals, generator=generator).denormalized_trajectory

    @torch.no_grad()
    def project_std(self):
        self.log_std.clamp_(math.log(self.ppo_cfg.min_std), math.log(self.ppo_cfg.max_std))
