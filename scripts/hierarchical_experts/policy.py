"""High-level stochastic policy and frozen low-level expert interface."""

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class Expert(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)["model_state_dict"]
        layers = []
        for n, i in enumerate((0, 2, 4, 6)):
            weight, bias = state[f"actor.{i}.weight"], state[f"actor.{i}.bias"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
            layers.append(layer)
            if n < 3:
                layers.append(nn.ELU())
        if layers[0].in_features != 48 or layers[-1].out_features != 12:
            raise ValueError("The expert must map 48 observations to 12 joint actions.")
        self.actor = nn.Sequential(*layers)
        self.register_buffer("mean", state["actor_obs_normalizer._mean"].clone())
        self.register_buffer("std", state["actor_obs_normalizer._std"].clone())
        self.requires_grad_(False)
        self.eval()

    def forward(self, obs):
        return self.actor((obs - self.mean) / (self.std + 0.01))


class Experts(nn.Module):
    def __init__(self, walk, crouch):
        super().__init__()
        self.walk, self.crouch = Expert(walk), Expert(crouch)

    @staticmethod
    def observation(env, command):
        data = env._robot.data
        action = env._applied_actions if env.cfg.action_obs_is_last_executed else env._actions
        return torch.cat((data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
                          command, data.joint_pos[:, env._joint_ids] - env._q_offset_action_and_obs,
                          data.joint_vel[:, env._joint_ids], action), dim=-1)

    @torch.no_grad()
    def action(self, env, skill, command, capture_observation=False):
        env._commands.copy_(command)
        env._command_steps_left.fill_(2)
        obs = self.observation(env, command)
        if capture_observation:
            self.last_observation = obs
        return torch.where(skill[:, None].bool(), self.crouch(obs), self.walk(obs))


class Hierarchy(nn.Module):
    def __init__(self, ppo_cfg, command_range_mode="configured"):
        super().__init__()
        if ppo_cfg.adaptive_actor_lr:
            raise ValueError("Adaptive actor learning rate is not implemented for the hybrid policy.")
        self.cfg = SimpleNamespace(history=8, proprio_dim=30, action_hist_dim=12, goal_dim=16, action_dim=12)
        self.ppo_cfg = self.dppo_cfg = ppo_cfg
        dim = 8 * (30 + 12 + 16)
        self.network = nn.Sequential(nn.Linear(dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                     nn.Linear(256, 128), nn.ELU())
        self.velocity_head = nn.Linear(128, 3)
        self.skill_head = nn.Linear(128, 2)
        self.log_std = nn.Parameter(torch.full((3,), float(ppo_cfg.initial_std)).log())
        if command_range_mode not in ("configured", "shared"):
            raise ValueError("Unknown command range mode.")
        limits = [[2., 1., 1.], [1., 0.5, 1.]] if command_range_mode == "configured" else [[1., 0.5, 1.]] * 2
        self.register_buffer("command_limits", torch.tensor(limits))

    @staticmethod
    def features(proprio, actions, goals):
        p = proprio.clone()
        p[..., :12] /= 3.14
        p[..., 12:24] /= 20.0
        p[..., 24:27] /= 5.0
        g = goals.clone()
        # Four local waypoints followed by current height, heading and speed.
        for i in range(4):
            g[..., 3*i:3*i+2] /= 4.0
            g[..., 3*i+2] /= 0.3
        g[..., 12] /= 0.3
        g[..., 15] /= 1.5
        return torch.cat((p.flatten(1), (actions / 2.0).flatten(1), g.flatten(1)), dim=-1).clamp(-10, 10)

    def critic_condition(self, proprio, actions, goals, privileged):
        return torch.cat((self.features(proprio, actions, goals), privileged), dim=-1)

    def distribution(self, proprio, actions, goals):
        features = self.network(self.features(proprio, actions, goals))
        return Normal(self.velocity_head(features), self.log_std.exp().expand(features.shape[0], -1)), Categorical(logits=self.skill_head(features))

    def sample(self, proprio, actions, goals):
        normal, categorical = self.distribution(proprio, actions, goals)
        velocity, skill = normal.sample(), categorical.sample()
        latent = torch.cat((velocity, skill[:, None].float()), dim=-1)
        logprob = normal.log_prob(velocity).sum(-1) + categorical.log_prob(skill)
        return latent, logprob

    def logprob(self, proprio, actions, goals, latent):
        normal, categorical = self.distribution(proprio, actions, goals)
        logprob = normal.log_prob(latent[..., :3]).sum(-1) + categorical.log_prob(latent[..., 3].long())
        return logprob, normal.entropy().sum(-1) + categorical.entropy()

    def command(self, latent):
        skill = latent[..., 3].long()
        return skill, latent[..., :3].tanh() * self.command_limits[skill]

    def project_std(self):
        with torch.no_grad():
            self.log_std.clamp_(min=float(torch.tensor(self.ppo_cfg.min_std).log()),
                                max=float(torch.tensor(self.ppo_cfg.max_std).log()))
