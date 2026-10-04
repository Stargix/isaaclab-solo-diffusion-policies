from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicyConfig
from scripts.diffusion_policy.train.data.normalization import (
    MinMaxStats, NormalizerStats, ZScoreStats, denormalize_minmax, normalize_minmax,
)
from scripts.dppo_diffusion_rl.buffer import RolloutBatch
from scripts.dppo_diffusion_rl.config import DPPOConfig
from scripts.dppo_diffusion_rl.critic import ValueCritic
from scripts.dppo_diffusion_rl.policy import DPPODiffusionPolicy
from scripts.dppo_diffusion_rl.ppo import DPPOUpdater
from scripts.dppo_diffusion_rl.symmetry import LeftRightAugmentation
from scripts.dppo_diffusion_rl.checkpointing import training_resume_state


def _policy(mode="mirror", goal_dim=16):
    cfg = Solo12DiffusionPolicyConfig(
        proprio_dim=30, action_hist_dim=12, goal_dim=goal_dim, action_dim=12,
        history=2, prediction_horizon=4, execution_offset=2,
        d_model=16, nhead=2, num_layers=1, p_drop_attn=0.0,
        num_train_timesteps=4, num_inference_steps=4,
    )
    dppo = DPPOConfig(
        inference_steps=4, finetune_denoising_steps=2, exec_horizon=2,
        symmetry_augmentation=mode, minibatch_size=2, critic_minibatch_size=2,
        update_epochs=1, target_kl=100.0,
    )
    policy = DPPODiffusionPolicy(cfg, dppo)
    # Asymmetric input statistics are allowed: reflection happens in RAW space.
    stats = NormalizerStats(
        ZScoreStats(np.arange(30, dtype=np.float32) * .01, np.ones(30, np.float32)),
        ZScoreStats(np.arange(goal_dim, dtype=np.float32) * .02, np.ones(goal_dim, np.float32)),
        MinMaxStats(-np.ones(12, np.float32), np.ones(12, np.float32)),
    )
    policy.load_pretrained(policy.policy.state_dict(), stats)
    return policy


@pytest.mark.parametrize("goal_dim", [12, 16])
def test_raw_conditions_and_latents_are_involutions(goal_dim):
    policy = _policy(goal_dim=goal_dim)
    symmetry = LeftRightAugmentation(policy)
    p, a, g = torch.randn(3, 2, 30), torch.randn(3, 2, 12), torch.randn(3, 2, goal_dim)
    reflected = symmetry.reflect_conditions(p, a, g)
    twice = symmetry.reflect_conditions(*reflected)
    for original, recovered in zip((p, a, g), twice):
        torch.testing.assert_close(original, recovered, atol=0, rtol=0)
    x = torch.randn(3, 3, 4, 12)
    torch.testing.assert_close(x, symmetry.reflect_latent(symmetry.reflect_latent(x)))
    if goal_dim == 16:
        torch.testing.assert_close(g[..., 15], reflected[2][..., 15])
        torch.testing.assert_close(g[..., 12], reflected[2][..., 12])
        torch.testing.assert_close(g[..., 13], -reflected[2][..., 13])
        torch.testing.assert_close(g[..., 14], reflected[2][..., 14])
        torch.testing.assert_close(g[..., :12].reshape(3, 2, 4, 3)[..., 2],
                                   reflected[2][..., :12].reshape(3, 2, 4, 3)[..., 2])


def test_action_normalization_clipping_and_noise_commute_with_reflection():
    policy = _policy()
    symmetry = LeftRightAugmentation(policy)
    x = torch.randn(3, 4, 12) * 2
    stats = policy.normalizer_stats.action
    physical = denormalize_minmax(x, stats)
    torch.testing.assert_close(
        normalize_minmax(symmetry.reflect_latent(physical), stats), symmetry.reflect_latent(x)
    )
    torch.testing.assert_close(
        symmetry.reflect_latent(x.clamp(-1, 1)), symmetry.reflect_latent(x).clamp(-1, 1)
    )
    mean, noise = torch.randn_like(x), torch.randn_like(x)
    torch.testing.assert_close(
        symmetry.reflect_latent(mean + .1 * noise),
        symmetry.reflect_latent(mean) + .1 * symmetry.reflect_latent(noise),
    )


def test_reflection_rejects_non_equivariant_action_ranges():
    policy = _policy()
    policy.normalizer_stats.action.max[0] = 2.0
    with pytest.raises(ValueError, match="reflection-closed"):
        LeftRightAugmentation(policy)


@pytest.mark.parametrize("final", [False, True])
def test_reflected_gaussian_scores_preserve_the_causal_cone(final):
    policy = _policy()
    mirror = LeftRightAugmentation.reflect_latent
    mean, following = torch.randn(3, 4, 12), torch.randn(3, 4, 12)
    std = torch.full((3, 1, 1), .2)
    original = policy._reward_relevant_logprob(mean, std, following, final)
    transformed = policy._reward_relevant_logprob(mirror(mean), std, mirror(following), final)
    torch.testing.assert_close(original, transformed)


def test_privileged_critic_features_are_reflected_and_involutive():
    policy = _policy()
    symmetry = LeftRightAugmentation(policy)
    p, a, g = torch.randn(3, 2, 30), torch.randn(3, 2, 12), torch.randn(3, 2, 16)
    features = torch.randn(3, 14)
    observation = policy.critic_condition(p, a, g, features)
    reflected = symmetry.reflect_critic(p, a, g, observation)
    expected = features.clone()
    expected[:, [6, 10]] *= -1
    torch.testing.assert_close(reflected[:, -14:], expected)
    rp, ra, rg = symmetry.reflect_conditions(p, a, g)
    torch.testing.assert_close(symmetry.reflect_critic(rp, ra, rg, reflected), observation)
    # Normalization is NOT applied to signed normalized observation slices.
    torch.testing.assert_close(reflected, policy.critic_condition(rp, ra, rg, expected))


def _rollout(policy, critic):
    p = torch.randn(3, 2, 30) * .1
    a = torch.randn(3, 2, 12) * .1
    g = torch.randn(3, 2, 16) * .1
    g[..., 1] = .3
    with torch.no_grad():
        sample = policy.sample(p, a, g)
        obs = policy.critic_condition(p, a, g, torch.randn(3, 14))
        values = critic(obs)
    batch = RolloutBatch(
        proprio=p[None], action_history=a[None], goals=g[None],
        critic_observation=obs[None], chains=sample.chain[None],
        old_logprobs=sample.old_logprobs[None], old_values=values[None],
        rewards=torch.tensor([[1., -1., .4]]), dones=torch.ones(1, 3, dtype=torch.bool),
    )
    batch.compute_gae(torch.zeros(3), gamma=1., gae_lambda=.95)
    return batch


def test_mirror_update_is_finite_and_preserves_optimizer_step_budget():
    torch.manual_seed(5)
    policy = _policy()
    critic = ValueCritic(130, hidden_dims=(16,))
    rollout = _rollout(policy, critic)
    original_policy, original_critic = copy.deepcopy(policy), copy.deepcopy(critic)
    original_policy.dppo_cfg.symmetry_augmentation = "none"
    baseline = DPPOUpdater(original_policy, original_critic, original_policy.dppo_cfg)
    updater = DPPOUpdater(policy, critic, policy.dppo_cfg)
    # Same permutation schedule in both arms.
    torch.manual_seed(19)
    baseline_metrics = baseline.update(rollout)
    torch.manual_seed(19)
    metrics = updater.update(rollout)
    assert metrics["Policy/actor_updates"] == baseline_metrics["Policy/actor_updates"] == 3
    assert metrics["Policy/critic_updates"] == baseline_metrics["Policy/critic_updates"] == 2
    assert metrics["Symmetry/mirrored_minibatches"] == 3
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(torch.isfinite(p).all() for p in policy.trainable_parameters)


def test_augmented_old_density_is_original_and_not_update_kl(monkeypatch):
    policy = _policy()
    policy.dppo_cfg.target_kl = .001
    critic = ValueCritic(130, hidden_dims=(16,))
    rollout = _rollout(policy, critic)
    rollout.old_logprobs.zero_()
    parameter = next(policy.trainable_parameters)
    calls = []

    def likelihood(p, a, g, previous, following, index, **kwargs):
        mirrored = g[:, 0, 1] < 0
        calls.append(mirrored.all().item())
        value = mirrored.float() * 2.0 + parameter.sum() * 0.0
        return value, torch.zeros_like(value)

    monkeypatch.setattr(policy, "transition_logprob", likelihood)
    metrics = DPPOUpdater(policy, critic, policy.dppo_cfg).update(rollout)
    assert metrics["Policy/actor_updates"] == 3
    assert metrics["Policy/approximate_kl"] == 0
    assert metrics["Policy/early_stopped"] == 0
    assert metrics["Symmetry/mirrored_ratio_mean"] == pytest.approx(np.exp(2.))
    assert calls == [False, True] * 3


def test_critic_only_warmup_has_no_actor_updates():
    policy = _policy()
    critic = ValueCritic(130, hidden_dims=(16,))
    rollout = _rollout(policy, critic)
    before = copy.deepcopy(policy.policy.model.state_dict())
    metrics = DPPOUpdater(policy, critic, policy.dppo_cfg).update(rollout, update_actor=False)
    assert metrics["Policy/actor_updates"] == 0
    assert metrics["Symmetry/mirrored_minibatches"] == 0
    for name, value in before.items():
        torch.testing.assert_close(value, policy.policy.model.state_dict()[name], atol=0, rtol=0)


def test_full_quadruped_is_rejected_for_directed_yaw_task():
    with pytest.raises(ValueError, match="left/right"):
        DPPOConfig(symmetry_augmentation="quadruped").validate(prediction_horizon=16, execution_offset=8)


def test_pre_update_best_resume_does_not_skip_unapplied_update():
    state = {"algorithm": "dppo", "dppo_task_contract_version": 8,
             "iteration": 7, "total_physics_steps": 100,
             "metrics": {"checkpoint_phase": "pre_update"}}
    assert training_resume_state(state, restart_optimization=False) == (7, 100, True)
    state["metrics"]["checkpoint_phase"] = "post_update_unscored"
    assert training_resume_state(state, restart_optimization=False) == (8, 100, True)
