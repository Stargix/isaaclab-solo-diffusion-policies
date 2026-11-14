"""Versioned hybrid actor with a categorical expert and conditional commands."""

from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .policy import Experts, Hierarchy


@dataclass
class HybridSample:
    skill: torch.Tensor
    command_latent: torch.Tensor
    discrete_logprob: torch.Tensor
    continuous_logprob: torch.Tensor


class HierarchyV2(nn.Module):
    """Shared state encoder, categorical selector, and per-skill Gaussian heads."""

    architecture = "hierarchy_categorical_conditional_gaussian_v2"
    format_version = 3

    def __init__(self, *, execution_steps=10, initial_std=0.6, min_std=0.05,
                 max_std=1.0, command_range_mode="shared"):
        super().__init__()
        if execution_steps < 1 or not 0 < min_std <= initial_std <= max_std:
            raise ValueError("Invalid execution interval or Gaussian standard-deviation bounds.")
        if command_range_mode not in ("shared", "configured"):
            raise ValueError("command_range_mode must be shared or configured.")
        self.cfg = SimpleNamespace(history=8, proprio_dim=30, action_hist_dim=12,
                                   goal_dim=16, action_dim=12)
        self.execution_steps = int(execution_steps)
        self.min_std, self.max_std = float(min_std), float(max_std)
        self.command_range_mode = command_range_mode
        self.network = nn.Sequential(nn.Linear(8 * 58, 512), nn.ELU(),
                                     nn.Linear(512, 256), nn.ELU(),
                                     nn.Linear(256, 128), nn.ELU())
        self.skill_head = nn.Linear(128, 2)
        self.command_heads = nn.ModuleList(nn.Linear(128, 3) for _ in range(2))
        self.log_stds = nn.Parameter(torch.full((2, 3), float(initial_std)).log())
        limits = [[1., .5, 1.], [1., .5, 1.]]
        if command_range_mode == "configured":
            limits = [[2., 1., 1.], [1., .5, 1.]]
        self.register_buffer("command_limits", torch.tensor(limits))

    @staticmethod
    def features(proprio, actions, goals):
        # Preserve precisely the V1 feature transform and observation ordering.
        return Hierarchy.features(proprio, actions, goals)

    def critic_condition(self, proprio, actions, goals, privileged):
        return torch.cat((self.features(proprio, actions, goals), privileged), dim=-1)

    def _selected_means(self, feature, skill):
        selected = feature.new_zeros((feature.shape[0], 3))
        for index, head in enumerate(self.command_heads):
            rows = torch.nonzero(skill.long() == index, as_tuple=False).squeeze(-1)
            if rows.numel():
                selected = selected.index_copy(0, rows, head(feature[rows]))
        return selected

    def parameters_for(self, proprio, actions, goals, skill):
        feature = self.network(self.features(proprio, actions, goals))
        logits = self.skill_head(feature)
        selected_mean = self._selected_means(feature, skill)
        selected_log_std = self.log_stds[skill.long()]
        return logits, Normal(selected_mean, selected_log_std.exp())

    def sample(self, proprio, actions, goals):
        feature = self.network(self.features(proprio, actions, goals))
        categorical = Categorical(logits=self.skill_head(feature))
        skill = categorical.sample()
        mean = self._selected_means(feature, skill)
        distribution = Normal(mean, self.log_stds[skill].exp())
        latent = distribution.sample()
        return HybridSample(skill, latent, categorical.log_prob(skill),
                            distribution.log_prob(latent).sum(-1))

    def logprobs(self, proprio, actions, goals, skill, command_latent):
        logits, conditional = self.parameters_for(proprio, actions, goals, skill)
        categorical = Categorical(logits=logits)
        return (categorical.log_prob(skill.long()),
                conditional.log_prob(command_latent).sum(-1),
                categorical.entropy(), conditional.entropy().sum(-1))

    def deterministic(self, proprio, actions, goals):
        feature = self.network(self.features(proprio, actions, goals))
        skill = self.skill_head(feature).argmax(-1)
        latent = self._selected_means(feature, skill)
        return skill, latent

    def deterministic_with_diagnostics(self, proprio, actions, goals):
        """Return the deterministic action and passive selector/command diagnostics."""
        feature = self.network(self.features(proprio, actions, goals))
        logits = self.skill_head(feature)
        probabilities = logits.softmax(-1)
        skill = logits.argmax(-1)
        latent = self._selected_means(feature, skill)
        command = self.command(skill, latent)
        return skill, latent, {
            "skill_probabilities": probabilities,
            "command_latent": latent,
            "command": command,
            "command_saturated": torch.tanh(latent).abs() >= 0.98,
        }

    def command(self, skill, command_latent):
        return command_latent.tanh() * self.command_limits[skill.long()]

    @torch.no_grad()
    def project_std(self):
        self.log_stds.clamp_(min=math.log(self.min_std), max=math.log(self.max_std))


__all__ = ["Experts", "HierarchyV2", "HybridSample"]
