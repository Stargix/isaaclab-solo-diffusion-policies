from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicyConfig
from scripts.diffusion_policy.train.data.normalization import MinMaxStats, NormalizerStats, ZScoreStats
from scripts.dppo_diffusion_rl.config import DPPOConfig
from scripts.dppo_diffusion_rl.buffer import RolloutBatch
from scripts.dppo_diffusion_rl.critic import ValueCritic
from scripts.dppo_diffusion_rl.policy import DPPODiffusionPolicy
from scripts.dppo_diffusion_rl.ppo import DPPOUpdater


def _policy() -> DPPODiffusionPolicy:
    policy_cfg = Solo12DiffusionPolicyConfig(
        proprio_dim=3,
        action_hist_dim=2,
        goal_dim=2,
        action_dim=2,
        history=2,
        prediction_horizon=4,
        execution_offset=2,
        d_model=16,
        nhead=2,
        num_layers=1,
        p_drop_attn=0.3,
        num_train_timesteps=4,
        num_inference_steps=4,
    )
    dppo_cfg = DPPOConfig(
        inference_steps=4,
        finetune_denoising_steps=2,
        exec_horizon=2,
        min_denoising_std=0.1,
        minibatch_size=4,
        critic_minibatch_size=2,
    )
    policy = DPPODiffusionPolicy(policy_cfg, dppo_cfg)
    stats = NormalizerStats(
        proprio=ZScoreStats(np.zeros(3, np.float32), np.ones(3, np.float32)),
        goal=ZScoreStats(np.zeros(2, np.float32), np.ones(2, np.float32)),
        action=MinMaxStats(-np.ones(2, np.float32), np.ones(2, np.float32)),
    )
    policy.load_pretrained(policy.policy.state_dict(), stats)
    return policy


def test_behavior_logprob_is_reproduced_exactly() -> None:
    torch.manual_seed(3)
    policy = _policy()
    proprio = torch.randn(3, 2, 3)
    actions = torch.randn(3, 2, 2).clamp(-1, 1)
    goals = torch.randn(3, 2, 2)
    sample = policy.sample(proprio, actions, goals)
    batch = torch.arange(3).repeat_interleave(2)
    denoise = torch.arange(2).repeat(3)
    new_logprob, _ = policy.transition_logprob(
        proprio[batch],
        actions[batch],
        goals[batch],
        sample.chain[batch, denoise],
        sample.chain[batch, denoise + 1],
        denoise,
    )
    old_logprob = sample.old_logprobs[batch, denoise]
    torch.testing.assert_close(new_logprob, old_logprob, atol=2.0e-6, rtol=2.0e-6)


def test_base_is_frozen_and_dropout_is_disabled() -> None:
    policy = _policy()
    assert not policy.policy.model.training
    assert not policy.base_model.training
    assert all(not parameter.requires_grad for parameter in policy.base_model.parameters())
    assert all(parameter.requires_grad for parameter in policy.policy.model.parameters())
    policy.train()
    assert not policy.policy.model.training
    assert not policy.base_model.training


def test_denoising_clip_is_monotone() -> None:
    policy = _policy()
    clips = policy.denoising_clip(torch.arange(2))
    assert torch.all(clips[1:] >= clips[:-1])
    torch.testing.assert_close(clips[0], torch.tensor(policy.dppo_cfg.clip_ratio_base))
    torch.testing.assert_close(clips[-1], torch.tensor(policy.dppo_cfg.clip_ratio))


def test_reverse_transition_matches_diffusers_scheduler() -> None:
    policy = _policy()
    scheduler = policy.policy.noise_scheduler
    scheduler.set_timesteps(4)
    timestep = scheduler.timesteps[0]
    current = torch.randn(2, 4, 2)
    predicted_noise = torch.randn_like(current)
    mean, std = policy._transition_mean_std(
        current, predicted_noise, timestep, apply_exploration_floor=False
    )
    generator_reference = torch.Generator().manual_seed(91)
    generator_ours = torch.Generator().manual_seed(91)
    expected = scheduler.step(
        predicted_noise, timestep, current, generator=generator_reference
    ).prev_sample
    actual = mean + std * torch.randn(
        current.shape, dtype=current.dtype, device=current.device, generator=generator_ours
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_sample_contract() -> None:
    policy = _policy()
    sample = policy.sample(torch.zeros(2, 2, 3), torch.zeros(2, 2, 2), torch.zeros(2, 2, 2))
    assert sample.chain.shape == (2, 3, 4, 2)
    assert sample.old_logprobs.shape == (2, 2)
    assert sample.denormalized_trajectory.shape == (2, 4, 2)
    assert torch.isfinite(sample.chain).all()
    assert torch.isfinite(sample.old_logprobs).all()
    with pytest.raises(ValueError, match="same horizon"):
        policy.executable_chunk(sample.denormalized_trajectory, 1)


def test_one_synthetic_ppo_update_is_finite_and_preserves_base() -> None:
    torch.manual_seed(11)
    policy = _policy()
    policy.dppo_cfg.update_epochs = 1
    policy.dppo_cfg.minibatch_size = 4
    policy.dppo_cfg.critic_minibatch_size = 2
    proprio = torch.randn(2, 2, 3)
    actions = torch.randn(2, 2, 2).clamp(-1, 1)
    goals = torch.randn(2, 2, 2)
    sample = policy.sample(proprio, actions, goals)
    critic = ValueCritic(7, hidden_dims=(16, 8))
    critic_obs = torch.randn(2, 7)
    with torch.no_grad():
        values = critic(critic_obs)
    rollout = RolloutBatch(
        proprio=proprio.unsqueeze(0),
        action_history=actions.unsqueeze(0),
        goals=goals.unsqueeze(0),
        critic_observation=critic_obs.unsqueeze(0),
        chains=sample.chain.unsqueeze(0),
        old_logprobs=sample.old_logprobs.unsqueeze(0),
        old_values=values.unsqueeze(0),
        rewards=torch.tensor([[1.0, -0.5]]),
        dones=torch.tensor([[True, True]]),
    )
    rollout.compute_gae(torch.zeros(2), gamma=0.99, gae_lambda=0.95)
    base_before = {name: value.clone() for name, value in policy.base_model.state_dict().items()}
    metrics = DPPOUpdater(policy, critic, policy.dppo_cfg).update(rollout)
    assert all(np.isfinite(value) for value in metrics.values())
    for name, value in policy.base_model.state_dict().items():
        torch.testing.assert_close(value, base_before[name])
