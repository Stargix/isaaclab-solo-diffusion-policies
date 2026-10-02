"""CPU tests of density, gradients, collection and pre-update selection."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from scripts.diffusion_policy.model.solo12_deterministic_chunk_policy import (
    Solo12DeterministicChunkPolicy, Solo12DeterministicChunkPolicyConfig,
)
from scripts.diffusion_policy.train.data.normalization import NormalizerStats, MinMaxStats, ZScoreStats
from scripts.dppo_diffusion_rl.buffer import RolloutBatch
from scripts.dppo_diffusion_rl.critic import ValueCritic
from scripts.gaussian_chunk_rl.config import GaussianPPOConfig
from scripts.gaussian_chunk_rl.policy import GaussianChunkPolicy
from scripts.gaussian_chunk_rl.ppo import GaussianPPOUpdater
from scripts.gaussian_chunk_rl.trainer import GaussianPPOTrainer

torch.set_num_threads(1)


def make_policy(**settings):
    torch.manual_seed(7)
    mean = Solo12DeterministicChunkPolicy(Solo12DeterministicChunkPolicyConfig(
        d_model=16, nhead=4, num_layers=1, p_drop_attn=0.3))
    mean.set_normalizer_stats(NormalizerStats(
        ZScoreStats(np.zeros(30, np.float32), np.ones(30, np.float32)),
        ZScoreStats(np.zeros(16, np.float32), np.ones(16, np.float32)),
        MinMaxStats(-np.ones(12, np.float32), np.ones(12, np.float32))))
    return GaussianChunkPolicy(mean, GaussianPPOConfig(**settings))


def inputs(batch=4):
    return torch.randn(batch, 8, 30), torch.zeros(batch, 8, 12), torch.randn(batch, 8, 16)


def test_bc_mean_unchanged_and_dropout_disabled():
    actor = make_policy()
    x = inputs()
    actor.train()
    first, _ = actor.distribution(*x)
    second, _ = actor.distribution(*x)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first.clamp(-1, 1), actor.policy.predict_action(*x))
    assert not actor.policy.model.training


def test_raw_likelihood_is_sum_of_executable_48_dimensions_not_clipped():
    actor = make_policy()
    with torch.no_grad():
        actor.policy.model.head.weight.zero_()
        actor.policy.model.head.bias.fill_(2.0)
    x = inputs()
    sample = actor.sample(*x)
    recomputed, _ = actor.logprob(*x, sample.chain[:, 0])
    torch.testing.assert_close(recomputed, sample.old_logprobs[:, 0])
    expected = Normal(torch.full((4, 4, 12), 2.0), 0.04).log_prob(sample.chain[:, 0]).sum((-1, -2))
    torch.testing.assert_close(recomputed, expected)
    assert sample.chain.shape == (4, 1, 4, 12)
    assert (sample.chain[:, 0] > 1).all()
    assert (actor.executable_chunk(sample.denormalized_trajectory) == 1).all()
    clipped_density, _ = actor.logprob(*x, sample.chain[:, 0].clamp(-1, 1))
    assert not torch.allclose(recomputed, clipped_density)


def rollout(actor, batch=4):
    x = inputs(batch)
    sample = actor.sample(*x)
    critic_x = actor.critic_condition(*x, torch.zeros(batch, 14))
    data = RolloutBatch(proprio=x[0][None], action_history=x[1][None], goals=x[2][None],
        critic_observation=critic_x[None], chains=sample.chain[None],
        old_logprobs=sample.old_logprobs[None], old_values=torch.zeros(1, batch),
        rewards=torch.arange(batch, dtype=torch.float32)[None], dones=torch.ones(1, batch, dtype=torch.bool))
    data.compute_gae(torch.zeros(batch), gamma=1.0, gae_lambda=0.95)
    return data


def test_update_changes_actor_and_std_with_finite_gradients():
    actor = make_policy(update_epochs=1, minibatch_size=4, critic_minibatch_size=4)
    critic = ValueCritic(8 * 58 + 14, hidden_dims=(16,))
    updater = GaussianPPOUpdater(actor, critic, actor.ppo_cfg)
    data = rollout(actor)
    before = actor.policy.model.head.weight.detach().clone()
    old_std = actor.log_std.detach().clone()
    metrics = updater.update(data)
    assert metrics["Policy/actor_updates"] == 1
    assert metrics["Policy/critic_updates"] == 1
    assert not torch.equal(before, actor.policy.model.head.weight)
    assert not torch.equal(old_std, actor.log_std)
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["Policy/ratio_mean"] == pytest.approx(1, abs=1e-5)


def test_critic_warmup_does_not_change_actor_or_std():
    actor = make_policy(update_epochs=1)
    critic = ValueCritic(8 * 58 + 14, hidden_dims=(16,))
    updater = GaussianPPOUpdater(actor, critic, actor.ppo_cfg)
    data = rollout(actor)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    metrics = updater.update(data, update_actor=False)
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert metrics["Policy/actor_updates"] == 0 and metrics["Policy/critic_updates"] > 0


def test_kl_guard_stops_before_optimizer_step():
    actor = make_policy(update_epochs=2)
    critic = ValueCritic(8 * 58 + 14, hidden_dims=(16,))
    data = rollout(actor)
    data.old_logprobs -= 5
    before = actor.log_std.detach().clone()
    metrics = GaussianPPOUpdater(actor, critic, actor.ppo_cfg).update(data)
    assert metrics["Policy/early_stopped"] == 1 and metrics["Policy/actor_updates"] == 0
    assert metrics["Policy/approximate_kl"] > actor.ppo_cfg.target_kl
    assert metrics["Policy/accepted_minibatch_kl"] == 0
    assert metrics["Policy/kl_observed_batches"] == 1
    torch.testing.assert_close(actor.log_std, before)


@pytest.mark.parametrize("settings", [{"initial_std": 0}, {"gamma": .99}, {"exec_horizon": 9},
                                    {"actor_lr": float("nan")}, {"clip_ratio": 1.0}])
def test_invalid_configuration_rejected(settings):
    with pytest.raises(ValueError):
        make_policy(**settings)


@pytest.mark.parametrize("settings", [{"adaptive_actor_lr": 1}, {"min_actor_lr": 0},
    {"adaptive_actor_lr": True, "min_actor_lr": .01}, {"kl_probe_size": 0}])
def test_invalid_adaptive_configuration_rejected(settings):
    with pytest.raises(ValueError):
        make_policy(**settings)


def test_adaptive_lr_bounds_and_no_increase_after_early_stop():
    actor = make_policy(adaptive_actor_lr=True, actor_lr=1e-5, min_actor_lr=1e-7)
    updater = GaussianPPOUpdater(actor, ValueCritic(8 * 58 + 14, hidden_dims=(16,)), actor.ppo_cfg)
    assert updater._adapt_actor_lr(.1, early_stopped=True) == pytest.approx(1e-5 / 1.5)
    lowered = updater.actor_optimizer.param_groups[0]["lr"]
    assert updater._adapt_actor_lr(.001, early_stopped=True) == lowered
    assert updater._adapt_actor_lr(.001, early_stopped=False) == pytest.approx(1e-5)
    assert updater._adapt_actor_lr(.001, early_stopped=False) == pytest.approx(1e-5)
    for _ in range(100):
        updater._adapt_actor_lr(.1, early_stopped=True)
    assert updater.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-7)


def test_adaptive_warmup_does_not_change_learning_rate():
    actor = make_policy(adaptive_actor_lr=True, actor_lr=1e-5, update_epochs=1)
    updater = GaussianPPOUpdater(actor, ValueCritic(8 * 58 + 14, hidden_dims=(16,)), actor.ppo_cfg)
    metrics = updater.update(rollout(actor), update_actor=False)
    assert metrics["Policy/actor_lr"] == metrics["Policy/actor_lr_next"] == 1e-5
    assert "Policy/post_update_exact_kl" not in metrics


def test_post_update_exact_kl_is_joint_48_coordinate_kl():
    actor = make_policy(adaptive_actor_lr=True, actor_lr=1e-5, update_epochs=1, minibatch_size=4)
    with torch.no_grad():
        actor.policy.model.head.weight.zero_()
        actor.policy.model.head.bias.zero_()
    updater = GaussianPPOUpdater(actor, ValueCritic(8 * 58 + 14, hidden_dims=(16,)), actor.ppo_cfg)
    def controlled_step():
        with torch.no_grad():
            actor.policy.model.head.bias.add_(.001)
    updater.actor_optimizer.step = controlled_step
    metrics = updater.update(rollout(actor))
    assert metrics["Policy/post_update_exact_kl"] == pytest.approx(48 * .001**2 / (2 * .04**2), rel=1e-3)
    assert metrics["Policy/actor_lr_next"] == 1e-5


def test_adaptive_lr_reduces_on_rejected_minibatch_even_if_probe_unchanged():
    actor = make_policy(adaptive_actor_lr=True, actor_lr=1e-5, update_epochs=1)
    updater = GaussianPPOUpdater(actor, ValueCritic(8 * 58 + 14, hidden_dims=(16,)), actor.ppo_cfg)
    data = rollout(actor)
    data.old_logprobs -= 5
    metrics = updater.update(data)
    assert metrics["Policy/actor_updates"] == 0
    assert metrics["Policy/post_update_exact_kl"] == pytest.approx(0, abs=1e-6)
    assert metrics["Policy/actor_lr_next"] == pytest.approx(1e-5 / 1.5)


def test_std_projection():
    actor = make_policy()
    with torch.no_grad():
        actor.log_std[0] = -100
        actor.log_std[1] = 100
    actor.project_std()
    assert actor.log_std.exp().min().item() == pytest.approx(.01)
    assert actor.log_std.exp().max().item() == pytest.approx(.2)


def test_chunk_terminal_mask_uses_shared_collector(tmp_path):
    class Env:
        num_envs, device = 2, "cpu"
        unwrapped = property(lambda self: self)
        def __init__(self):
            self.counter = 0
            self.resets = []
        def get_proprioception(self): return torch.zeros(2, 30)
        def get_goal(self): return torch.zeros(2, 16)
        def get_critic_features(self): return torch.zeros(2, 14)
        def step(self, action):
            done = torch.tensor([self.counter == 0, False])
            self.counter += 1
            return {}, torch.ones(2), done, torch.zeros(2, dtype=torch.bool), {}
        def reset_after_action_chunk(self, ids): self.resets.append(ids)
    actor = make_policy()
    env = Env()
    # Skip only the saving-specific source-hash setup; collect is exactly shared.
    trainer = object.__new__(GaussianPPOTrainer)
    from scripts.dppo_diffusion_rl.trainer import DPPOTrainer
    DPPOTrainer.__init__(trainer, env=env, policy=actor,
        critic=ValueCritic(8 * 58 + 14, hidden_dims=(16,)), updater=None,
        source_checkpoint={}, source_checkpoint_path="unused.pt", output_dir=tmp_path,
        rollout_chunks=1)
    trainer._seed_history()
    batch, _ = trainer.collect()
    torch.testing.assert_close(batch.rewards[0], torch.tensor([1., 4.]))
    assert batch.chains.shape == (1, 2, 1, 4, 12)
    assert batch.dones[0].tolist() == [True, False]
    assert env.resets[0].tolist() == [0]


def test_selection_saves_actor_before_update(tmp_path):
    trainer = object.__new__(GaussianPPOTrainer)
    actor = make_policy(critic_warmup_iterations=0)
    trainer.policy = actor
    trainer.env = SimpleNamespace(reset=lambda: None)
    trainer._seed_history = lambda: None
    trainer.num_envs, trainer.rollout_chunks, trainer.save_interval = 4, 1, 10
    trainer.best_score = -math.inf
    data = rollout(actor)
    trainer.collect = lambda: (data, {"Episode/success_rate": .5, "Episode/base_contact_rate": 0.,
                                      "Episode/mean_speed_error_abs_mps": .1})
    trainer._selection_score = lambda metrics: .5
    trainer.updater = GaussianPPOUpdater(actor, ValueCritic(8 * 58 + 14, hidden_dims=(16,)), actor.ppo_cfg)
    snapshots = {}
    trainer._save = lambda name, iteration, metrics: snapshots.update({name: actor.policy.model.head.weight.detach().clone()})
    trainer._write_metrics = lambda *args: None
    initial = actor.policy.model.head.weight.detach().clone()
    trainer.run(1)
    torch.testing.assert_close(snapshots["best.pt"], initial, rtol=0, atol=0)
    assert not torch.equal(snapshots["last.pt"], initial)


def test_evaluator_uses_gaussian_horizon_and_dynamic_budget():
    source = Path("scripts/diffusion_policy/evaluate_policy.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'checkpoint["gaussian_ppo_config"]["exec_horizon"]' in source
    assert 'checkpoint.get("algorithm") in {"dppo", "gaussian_chunk_ppo"}' in source
    assert 'not in {"deterministic_chunk_bc", "gaussian_chunk_ppo"}' in source
