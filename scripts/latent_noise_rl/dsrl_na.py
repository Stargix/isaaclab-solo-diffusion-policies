"""Noise-aliased DSRL (DSRL-NA) around a frozen Solo12 DDIM prior.

This follows Algorithm 1 of Wagenmaker et al. (2025): an SAC actor chooses
initial diffusion noise, an action-space twin critic is trained from replay,
and a second twin critic is distilled on fresh standard-normal noise through
the frozen diffusion decoder. The optimizer never differentiates through the
decoder; actor gradients flow through the distilled noise critic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .decoder import FrozenDDIMDecoder


@dataclass(frozen=True)
class DSRLNAConfig:
    hidden_dim: int = 512
    hidden_layers: int = 3
    # Official DSRL-NA uses the normalized environment action bounds [-1,1]
    # as its noise bounds; action_magnitude=1.5 is used by DSRL-SAC's wrapper.
    action_magnitude: float = 1.0
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    # Number of vectorized macro-steps collected before the official 20 SAC updates.
    train_freq: int = 2
    gradient_steps: int = 20
    noise_critic_grad_steps: int = 10
    replay_capacity: int = 100_000
    initial_rollout_chunks: int = 2001
    target_entropy: float = 0.0
    initial_alpha: float = 1.0
    critic_backup_combine_type: str = "min"
    exec_horizon: int = 4
    inference_steps: int = 10

    def validate(self):
        if self.hidden_dim < 8 or self.hidden_layers < 1:
            raise ValueError("DSRL networks require positive hidden dimensions.")
        if self.action_magnitude <= 0 or self.batch_size < 2:
            raise ValueError("Invalid DSRL action magnitude or batch size.")
        if self.gradient_steps < 1 or self.noise_critic_grad_steps < 1:
            raise ValueError("DSRL requires positive actor-critic and noise-distillation updates.")
        if self.train_freq < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("Invalid DSRL replay/training frequency.")
        if not 0 < self.tau <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid DSRL discount or target update rate.")
        if self.exec_horizon != 4 or self.inference_steps != 10:
            raise ValueError("This adapter pins the four-action chunk and the 10-step prior.")
        if self.initial_rollout_chunks < 1 or self.initial_alpha <= 0:
            raise ValueError("DSRL requires positive initial rollouts and entropy temperature.")
        if self.critic_backup_combine_type not in {"min", "mean"}:
            raise ValueError("critic_backup_combine_type must be 'min' or 'mean'.")

    def to_dict(self):
        return asdict(self)


def _mlp(input_dim: int, output_dim: int | None, cfg: DSRLNAConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(cfg.hidden_layers):
        layers.extend((nn.Linear(width, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()))
        width = cfg.hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianNoiseActor(nn.Module):
    """SAC actor whose bounded action is a vector of initial-noise values."""

    def __init__(self, observation_dim: int, noise_dim: int, cfg: DSRLNAConfig):
        super().__init__()
        self.trunk = _mlp(observation_dim, None, cfg)
        self.mean = nn.Linear(cfg.hidden_dim, noise_dim)
        self.log_std = nn.Linear(cfg.hidden_dim, noise_dim)
        self.action_magnitude = float(cfg.action_magnitude)

    def distribution(self, observation: torch.Tensor):
        features = self.trunk(observation)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20.0, 2.0)
        return mean, log_std.exp()

    def sample(self, observation: torch.Tensor, *, deterministic=False, generator=None):
        mean, std = self.distribution(observation)
        if deterministic:
            pre_tanh = mean
        else:
            epsilon = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            pre_tanh = mean + std * epsilon
        squashed = torch.tanh(pre_tanh)
        action = self.action_magnitude * squashed
        normal = Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh)
        log_prob -= torch.log(1.0 - squashed.square() + 1.0e-6)
        # SB3 optimizes entropy in tanh-normalized coordinates. Scaling to
        # physical decoder noise must not shift target_entropy=0 by D*log(scale).
        return action, log_prob.sum(dim=-1, keepdim=True)


class TwinQ(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, cfg: DSRLNAConfig):
        super().__init__()
        self.q1 = _mlp(observation_dim + action_dim, 1, cfg)
        self.q2 = _mlp(observation_dim + action_dim, 1, cfg)

    def forward(self, observation, action):
        joined = torch.cat((observation, action), dim=-1)
        return self.q1(joined), self.q2(joined)


class DSRLNoiseReplay:
    """GPU replay of macro transitions with both raw histories and decoded actions."""

    def __init__(self, capacity, proprio_shape, action_history_shape, goal_shape,
                 executed_action_dim, device):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.proprio = torch.empty((capacity, *proprio_shape), device=device)
        self.action_history = torch.empty((capacity, *action_history_shape), device=device)
        self.goals = torch.empty((capacity, *goal_shape), device=device)
        self.next_proprio = torch.empty_like(self.proprio)
        self.next_action_history = torch.empty_like(self.action_history)
        self.next_goals = torch.empty_like(self.goals)
        self.actions = torch.empty((capacity, executed_action_dim), device=device)
        self.rewards = torch.empty((capacity, 1), device=device)
        self.dones = torch.empty((capacity, 1), device=device)
        self.position = 0
        self.size = 0

    @torch.no_grad()
    def add_batch(self, proprio, action_history, goals, actions, rewards, dones,
                  next_proprio, next_action_history, next_goals):
        count = len(proprio)
        if rewards.shape not in {(count,), (count, 1)} or dones.shape not in {(count,), (count, 1)}:
            raise ValueError("Replay rewards and dones must be one scalar per transition.")
        rewards, dones = rewards.reshape(count, 1), dones.reshape(count, 1)
        if count >= self.capacity:
            start = count - self.capacity
            proprio, action_history, goals = proprio[start:], action_history[start:], goals[start:]
            actions, rewards, dones = actions[start:], rewards[start:], dones[start:]
            next_proprio, next_action_history, next_goals = (
                next_proprio[start:], next_action_history[start:], next_goals[start:]
            )
            count = self.capacity
        indices = (torch.arange(count, device=self.device) + self.position) % self.capacity
        for destination, source in (
            (self.proprio, proprio), (self.action_history, action_history), (self.goals, goals),
            (self.actions, actions), (self.rewards, rewards), (self.dones, dones.float()),
            (self.next_proprio, next_proprio), (self.next_action_history, next_action_history),
            (self.next_goals, next_goals),
        ):
            destination[indices] = source
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise RuntimeError("DSRL replay does not yet contain one complete minibatch.")
        idx = torch.randint(self.size, (batch_size,), device=self.device)
        return {name: getattr(self, name)[idx] for name in (
            "proprio", "action_history", "goals", "actions", "rewards", "dones",
            "next_proprio", "next_action_history", "next_goals",
        )}


class DSRLNA:
    """DSRL-NA updates; only the noise actor and critics are trainable."""

    def __init__(self, decoder: FrozenDDIMDecoder, normalizer_stats, *,
                 history: int, proprio_dim: int, action_hist_dim: int, goal_dim: int,
                 action_dim: int, prediction_horizon: int, execution_offset: int,
                 cfg: DSRLNAConfig, device):
        cfg.validate()
        self.decoder = decoder.to(device).eval()
        self.stats = normalizer_stats
        self.cfg = cfg
        self.history, self.proprio_dim = history, proprio_dim
        self.action_hist_dim, self.goal_dim, self.action_dim = action_hist_dim, goal_dim, action_dim
        self.prediction_horizon, self.execution_offset = prediction_horizon, execution_offset
        self.device = torch.device(device)
        if execution_offset + cfg.exec_horizon > prediction_horizon:
            raise ValueError("Executed action chunk exceeds the pinned diffusion horizon.")
        self.observation_dim = history * (proprio_dim + action_hist_dim + goal_dim)
        # A causal decoder's action at row t cannot depend on initial noise at
        # rows > t. Thus rows through the final executed target are controllable.
        self.noise_rows = execution_offset + cfg.exec_horizon
        self.noise_dim = self.noise_rows * action_dim
        self.executed_dim = cfg.exec_horizon * action_dim
        self.actor = SquashedGaussianNoiseActor(self.observation_dim, self.noise_dim, cfg).to(device)
        self.q_action = TwinQ(self.observation_dim, self.executed_dim, cfg).to(device)
        self.q_action_target = TwinQ(self.observation_dim, self.executed_dim, cfg).to(device)
        self.q_action_target.load_state_dict(self.q_action.state_dict())
        self.q_noise = TwinQ(self.observation_dim, self.noise_dim, cfg).to(device)
        for parameter in self.q_action_target.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_action_optimizer = torch.optim.Adam(self.q_action.parameters(), lr=cfg.critic_lr)
        self.q_noise_optimizer = torch.optim.Adam(self.q_noise.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(math.log(cfg.initial_alpha), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    def to(self, device):
        self.device = torch.device(device)
        self.decoder.to(self.device)
        self.actor.to(self.device)
        self.q_action.to(self.device)
        self.q_action_target.to(self.device)
        self.q_noise.to(self.device)
        self.log_alpha.data = self.log_alpha.data.to(self.device)
        return self

    def eval(self):
        self.decoder.eval()
        self.actor.eval()
        self.q_action.eval()
        self.q_action_target.eval()
        self.q_noise.eval()
        return self

    def observation(self, proprio, action_history, goals):
        from scripts.diffusion_policy.train.data.normalization import (
            normalize_minmax, normalize_zscore,
        )
        return torch.cat((normalize_zscore(proprio, self.stats.proprio).flatten(1),
                          normalize_minmax(action_history, self.stats.action).flatten(1),
                          normalize_zscore(goals, self.stats.goal).flatten(1)), dim=1)

    def decode_noise(self, proprio, action_history, goals, flat_noise):
        if flat_noise.shape[-1] != self.noise_dim:
            raise ValueError(f"Expected latent width {self.noise_dim}, got {flat_noise.shape[-1]}.")
        full = torch.zeros((len(flat_noise), self.prediction_horizon, self.action_dim),
                           device=flat_noise.device, dtype=flat_noise.dtype)
        full[:, :self.noise_rows] = flat_noise.reshape(-1, self.noise_rows, self.action_dim)
        trajectory = self.decoder.decode(proprio, action_history, goals, full)
        start, end = self.execution_offset, self.execution_offset + self.cfg.exec_horizon
        from scripts.diffusion_policy.train.data.normalization import denormalize_minmax
        return denormalize_minmax(trajectory[:, start:end], self.stats.action)

    @torch.no_grad()
    def act(self, proprio, action_history, goals, *, deterministic=False, generator=None):
        observation = self.observation(proprio, action_history, goals)
        noise, log_prob = self.actor.sample(observation, deterministic=deterministic,
                                            generator=generator)
        chunk = self.decode_noise(proprio, action_history, goals, noise)
        return noise, chunk, log_prob

    @torch.no_grad()
    def base_noise(self, batch_size, *, generator=None):
        return torch.randn((batch_size, self.noise_dim), device=self.device, generator=generator)

    @torch.no_grad()
    def _decode_flat(self, batch, noise):
        return self.decode_noise(batch["proprio"], batch["action_history"], batch["goals"], noise)

    def _combine(self, values):
        stacked = torch.cat(values, dim=-1)
        if self.cfg.critic_backup_combine_type == "min":
            return stacked.min(dim=-1, keepdim=True).values
        return stacked.mean(dim=-1, keepdim=True)

    def update(self, replay: DSRLNoiseReplay):
        metrics = {name: [] for name in (
            "q_action_loss", "q_noise_loss", "actor_loss", "alpha", "alpha_loss", "q_target"
        )}
        for _ in range(self.cfg.gradient_steps):
            batch = replay.sample(self.cfg.batch_size)
            obs = self.observation(batch["proprio"], batch["action_history"], batch["goals"])
            next_obs = self.observation(batch["next_proprio"], batch["next_action_history"], batch["next_goals"])
            actor_noise, log_prob = self.actor.sample(obs)
            # Match official SAC ordering, including using the pre-update alpha
            # in this iteration's actor and Bellman objectives.
            alpha = self.log_alpha.detach().exp()
            alpha_loss = -(self.log_alpha * (log_prob + self.cfg.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                next_noise, next_log_prob = self.actor.sample(next_obs)
                next_batch = {**batch, "proprio": batch["next_proprio"],
                              "action_history": batch["next_action_history"],
                              "goals": batch["next_goals"]}
                next_action = self._decode_flat(next_batch, next_noise).flatten(1)
                target_values = self._combine(self.q_action_target(next_obs, next_action))
                target = batch["rewards"] + (1.0 - batch["dones"]) * self.cfg.gamma * (
                    target_values - alpha * next_log_prob
                )
            current_q = self.q_action(obs, batch["actions"])
            action_loss = 0.5 * sum(F.mse_loss(value, target) for value in current_q)
            self.q_action_optimizer.zero_grad(set_to_none=True)
            action_loss.backward()
            self.q_action_optimizer.step()
            metrics["q_action_loss"].append(float(action_loss.detach()))
            metrics["q_target"].append(float(target.mean()))

            # Freeze critic parameters without detaching Q_W from its latent
            # input: the pathwise SAC gradient must reach the noise actor.
            for parameter in self.q_noise.parameters():
                parameter.requires_grad_(False)
            actor_q = self._combine(self.q_noise(obs, actor_noise / self.cfg.action_magnitude))
            actor_loss = (alpha * log_prob - actor_q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for parameter in self.q_noise.parameters():
                parameter.requires_grad_(True)
            metrics["actor_loss"].append(float(actor_loss.detach()))
            metrics["alpha_loss"].append(float(alpha_loss.detach()))
            metrics["alpha"].append(float(self.log_alpha.detach().exp()))

            with torch.no_grad():
                for source, target_param in zip(self.q_action.parameters(), self.q_action_target.parameters()):
                    target_param.lerp_(source, self.cfg.tau)
        # Official DSRL-NA distills AFTER the complete SAC update block, once
        # per training call (10 updates total, not 10 per SAC gradient step).
        for _ in range(self.cfg.noise_critic_grad_steps):
            batch = replay.sample(self.cfg.batch_size)
            obs = self.observation(batch["proprio"], batch["action_history"], batch["goals"])
            gaussian_noise = torch.randn((self.cfg.batch_size, self.noise_dim), device=self.device)
            with torch.no_grad():
                decoded_action = self._decode_flat(batch, gaussian_noise).flatten(1)
                teacher_values = self.q_action(obs, decoded_action)
            student_values = self.q_noise(obs, gaussian_noise / self.cfg.action_magnitude)
            noise_loss = 0.5 * sum(
                F.mse_loss(student, teacher) for student, teacher in zip(student_values, teacher_values)
            )
            self.q_noise_optimizer.zero_grad(set_to_none=True)
            noise_loss.backward()
            self.q_noise_optimizer.step()
            metrics["q_noise_loss"].append(float(noise_loss.detach()))
        result = {f"DSRL/{name}": sum(values) / len(values) for name, values in metrics.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("Non-finite DSRL-NA update; refusing to continue training.")
        result.update({"DSRL/actor_updates": self.cfg.gradient_steps,
                       "DSRL/q_action_updates": self.cfg.gradient_steps,
                       "DSRL/q_noise_updates": self.cfg.noise_critic_grad_steps})
        return result


class DSRLNAInferencePolicy:
    """Small actor+decoder-only adapter; critics are not loaded for inference."""

    def __init__(self, decoder, normalizer_stats, *, history, proprio_dim, action_hist_dim,
                 goal_dim, action_dim, prediction_horizon, execution_offset, cfg, device):
        cfg.validate()
        self.decoder = decoder.to(device).eval()
        self.stats = normalizer_stats
        self.cfg = cfg
        self.history, self.proprio_dim = history, proprio_dim
        self.action_hist_dim, self.goal_dim, self.action_dim = action_hist_dim, goal_dim, action_dim
        self.prediction_horizon, self.execution_offset = prediction_horizon, execution_offset
        self.noise_rows = execution_offset + cfg.exec_horizon
        self.noise_dim = self.noise_rows * action_dim
        self.device = torch.device(device)
        if self.noise_rows >= prediction_horizon:
            raise ValueError("DSRL-NA's controllable causal prefix must leave a non-empty suffix.")
        self.actor = SquashedGaussianNoiseActor(
            history * (proprio_dim + action_hist_dim + goal_dim), self.noise_dim, cfg
        ).to(device)
        self.sampling_mode = "stochastic"

    def to(self, device):
        self.device = torch.device(device)
        self.decoder.to(self.device)
        self.actor.to(self.device)
        return self

    def eval(self):
        self.decoder.eval()
        self.actor.eval()
        return self

    def set_reference_mode(self, mode):
        if mode != "checkpoint":
            raise ValueError("DSRL-NA checkpoints only support their learned noise actor.")

    def set_sampling_mode(self, mode):
        if mode not in {"stochastic", "deterministic"}:
            raise ValueError("DSRL-NA sampling must be stochastic or deterministic.")
        self.sampling_mode = mode

    def executable_chunk(self, trajectory, num_actions=None):
        horizon = self.cfg.exec_horizon if num_actions is None else num_actions
        if horizon != self.cfg.exec_horizon:
            raise ValueError("DSRL-NA must execute its saved four-action horizon.")
        start = self.execution_offset
        return trajectory[:, start:start + horizon]

    def observation(self, proprio, action_history, goals):
        from scripts.diffusion_policy.train.data.normalization import (
            normalize_minmax, normalize_zscore,
        )
        return torch.cat((normalize_zscore(proprio, self.stats.proprio).flatten(1),
                          normalize_minmax(action_history, self.stats.action).flatten(1),
                          normalize_zscore(goals, self.stats.goal).flatten(1)), dim=1)

    @torch.no_grad()
    def predict_action(self, proprio, action_history, goals, *, guidance_scale=None, generator=None):
        if guidance_scale not in (None, 1.0):
            raise ValueError("DSRL-NA uses the pinned DDIM sampler with guidance=1.")
        if self.sampling_mode == "stochastic" and generator is None:
            raise ValueError("Stochastic DSRL-NA inference requires an explicit seeded generator.")
        # The official evaluation samples the learned noise distribution.
        # Deterministic actor-mean evaluation is an explicit secondary diagnostic.
        noise, _ = self.actor.sample(
            self.observation(proprio, action_history, goals),
            deterministic=self.sampling_mode == "deterministic", generator=generator,
        )
        full = torch.zeros((len(noise), self.prediction_horizon, self.action_dim),
                           device=noise.device, dtype=noise.dtype)
        full[:, :self.noise_rows] = noise.reshape(-1, self.noise_rows, self.action_dim)
        return self.decoder.decode(proprio, action_history, goals, full)

    @torch.no_grad()
    def predict_action_denormalized(self, proprio, action_history, goals, *, guidance_scale=None, generator=None):
        normalized = self.predict_action(proprio, action_history, goals,
                                         guidance_scale=guidance_scale, generator=generator)
        from scripts.diffusion_policy.train.data.normalization import denormalize_minmax
        return denormalize_minmax(normalized, self.stats.action)


def save_dsrl_na_checkpoint(path, *, source, source_path, source_sha, source_digest,
                            agent, iteration, total_control_transitions, metrics, task_config):
    from .checkpointing import assert_finite_state, source_semantic_digest

    prior = agent.decoder.prior
    if source_semantic_digest(prior) != source_digest:
        raise RuntimeError("Frozen source prior changed during DSRL-NA training.")
    prior_state = {name: tensor.detach().cpu().clone() for name, tensor in prior.state_dict().items()}
    payload = {
        "schema_version": source["schema_version"],
        "policy_kind": source["policy_kind"],
        "config": source["config"],
        "normalizer_stats": source["normalizer_stats"],
        "model_state_dict": prior_state,
        "ema_model_state_dict": prior_state,
        "algorithm": "dsrl_na",
        "policy_backend": "frozen_ddim_dsrl_na",
        "dsrl_checkpoint_version": 2,
        "entropy_coordinates": "normalized_tanh_noise",
        "discount_unit": "action_chunk",
        "dppo_task_contract_version": 9,
        "dppo_task_config": dict(task_config),
        "dsrl_na_config": agent.cfg.to_dict(),
        "latent_policy_config": {
            "noise_rows": agent.noise_rows,
            "noise_dim": agent.action_dim,
            "noise_scale": agent.cfg.action_magnitude,
            "inference_steps": agent.cfg.inference_steps,
            "eta": 0.0,
        },
        "dsrl_actor_state_dict": {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()},
        "dsrl_q_action_state_dict": {k: v.detach().cpu().clone() for k, v in agent.q_action.state_dict().items()},
        "dsrl_q_action_target_state_dict": {
            k: v.detach().cpu().clone() for k, v in agent.q_action_target.state_dict().items()
        },
        "dsrl_q_noise_state_dict": {k: v.detach().cpu().clone() for k, v in agent.q_noise.state_dict().items()},
        "dsrl_log_alpha": agent.log_alpha.detach().cpu().clone(),
        "frozen_prior_state_dict": prior_state,
        "frozen_prior_semantic_sha256": source_digest,
        "source_checkpoint_path": str(Path(source_path).resolve()),
        "source_checkpoint_sha256": source_sha,
        "ddim_scheduler_config": agent.decoder.scheduler_config(),
        "iteration": int(iteration),
        "total_control_transitions": int(total_control_transitions),
        "metrics": dict(metrics),
        "evaluation_sampling": "stochastic_squashed_gaussian_noise_ddim_eta0",
        "resume_supported": False,
    }
    for name in ("dsrl_actor_state_dict", "dsrl_q_action_state_dict",
                 "dsrl_q_action_target_state_dict", "dsrl_q_noise_state_dict"):
        assert_finite_state(payload[name])
    assert_finite_state({"log_alpha": payload["dsrl_log_alpha"]})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_dsrl_na_inference_policy(checkpoint, device="cpu"):
    from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicy
    from scripts.diffusion_policy.train.data.normalization import NormalizerStats
    from scripts.dppo_diffusion_rl.checkpointing import model_config_from_checkpoint
    from .checkpointing import SOURCE_SHA256, source_semantic_digest

    if checkpoint.get("algorithm") != "dsrl_na" or checkpoint.get("policy_backend") != "frozen_ddim_dsrl_na":
        raise ValueError("Checkpoint is not a supported DSRL-NA policy.")
    if checkpoint.get("dsrl_checkpoint_version") != 2 or checkpoint.get("resume_supported") is not False:
        raise ValueError("Unsupported DSRL-NA checkpoint version or resume contract.")
    if checkpoint.get("entropy_coordinates") != "normalized_tanh_noise" or checkpoint.get("discount_unit") != "action_chunk":
        raise ValueError("Incompatible DSRL-NA entropy/discount contract.")
    if checkpoint.get("source_checkpoint_sha256") != SOURCE_SHA256:
        raise ValueError("DSRL-NA checkpoint is not based on the pinned Phase-A prior.")
    if checkpoint.get("policy_kind") != "spatial_hindsight_height_profile_ddpm":
        raise ValueError("DSRL-NA checkpoint does not identify the expected Phase-A policy.")
    cfg = DSRLNAConfig(**checkpoint["dsrl_na_config"])
    model_cfg = model_config_from_checkpoint(checkpoint["config"])
    prior = Solo12DiffusionPolicy(model_cfg).to(device)
    prior.load_state_dict(checkpoint["frozen_prior_state_dict"], strict=True)
    stats = NormalizerStats.from_dict(checkpoint["normalizer_stats"])
    prior.set_normalizer_stats(stats)
    expected_latent = {
        "noise_rows": model_cfg.execution_offset + cfg.exec_horizon,
        "noise_dim": model_cfg.action_dim, "noise_scale": cfg.action_magnitude,
        "inference_steps": cfg.inference_steps, "eta": 0.0,
    }
    if any(checkpoint["latent_policy_config"].get(key) != value
           for key, value in expected_latent.items()):
        raise ValueError("Saved DSRL-NA noise geometry or sampler settings are inconsistent.")
    decoder = FrozenDDIMDecoder(prior, inference_steps=cfg.inference_steps)
    if decoder.scheduler_config() != checkpoint.get("ddim_scheduler_config"):
        raise ValueError("Saved DSRL-NA DDIM scheduler differs from the reconstructed sampler.")
    policy = DSRLNAInferencePolicy(
        decoder, stats, history=model_cfg.history, proprio_dim=model_cfg.proprio_dim,
        action_hist_dim=model_cfg.action_hist_dim, goal_dim=model_cfg.goal_dim,
        action_dim=model_cfg.action_dim, prediction_horizon=model_cfg.prediction_horizon,
        execution_offset=model_cfg.execution_offset, cfg=cfg, device=device,
    )
    policy.actor.load_state_dict(checkpoint["dsrl_actor_state_dict"], strict=True)
    if source_semantic_digest(prior) != checkpoint.get("frozen_prior_semantic_sha256"):
        raise ValueError("Frozen DSRL-NA prior or normalizers do not match the saved digest.")
    if not all(torch.isfinite(parameter).all() for parameter in policy.actor.parameters()):
        raise ValueError("DSRL-NA actor checkpoint contains non-finite parameters.")
    return policy.eval()
