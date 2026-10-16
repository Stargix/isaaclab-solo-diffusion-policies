"""Regression checks for the DSRL-NA replay and official SAC update semantics."""
from dataclasses import asdict, replace
import math

import pytest
import torch

from scripts.latent_noise_rl.dsrl_na import (
    DSRLNA, DSRLNAConfig, DSRLNoiseReplay, SquashedGaussianNoiseActor,
    build_dsrl_na_inference_policy, save_dsrl_na_checkpoint,
)
from scripts.latent_noise_rl.dsrl_na_trainer import DSRLNATrainer
from scripts.latent_noise_rl.checkpointing import PHASE_A_SOURCES, source_semantic_digest
from scripts.latent_noise_rl.tests.test_latent_noise import _fixture


def _agent(**overrides):
    prior_policy, inputs = _fixture()
    cfg = DSRLNAConfig(hidden_dim=16, hidden_layers=2, batch_size=4,
                       gradient_steps=3, noise_critic_grad_steps=2, replay_capacity=32,
                       initial_rollout_chunks=4, **overrides)
    model = prior_policy.cfg
    agent = DSRLNA(
        prior_policy.decoder, prior_policy.normalizer_stats,
        history=model.history, proprio_dim=model.proprio_dim,
        action_hist_dim=model.action_hist_dim, goal_dim=model.goal_dim,
        action_dim=model.action_dim, prediction_horizon=model.prediction_horizon,
        execution_offset=model.execution_offset, cfg=cfg, device="cpu",
    )
    replay = DSRLNoiseReplay(32, (8, 30), (8, 12), (8, 16), 48, "cpu")
    p, a, g = inputs
    replay.add_batch(p, a, g, agent.act(p, a, g)[1].flatten(1),
                     torch.ones(4), torch.zeros(4, dtype=torch.bool), p, a, g)
    return agent, replay, inputs


def test_replay_accepts_scalar_columns_and_wraps_without_broadcasting():
    r = DSRLNoiseReplay(5, (1, 1), (1, 1), (1, 1), 1, "cpu")
    for start, column in ((0, False), (3, True)):
        values = torch.arange(start, start + 3, dtype=torch.float32)
        history = values.reshape(3, 1, 1)
        scalar = values[:, None] if column else values
        r.add_batch(history, history, history, values[:, None], scalar, scalar > 0,
                    history, history, history)
    assert r.size == 5
    assert sorted(r.rewards[:, 0].tolist()) == [1, 2, 3, 4, 5]
    sample = r.sample(3)
    assert sample["rewards"].shape == sample["dones"].shape == (3, 1)
    assert torch.equal(r.rewards[:, 0], r.proprio[:, 0, 0])


def test_entropy_uses_normalized_coordinates_independently_of_noise_scale():
    cfg = DSRLNAConfig(hidden_dim=16, hidden_layers=2)
    unit = SquashedGaussianNoiseActor(5, 12, cfg)
    scaled = SquashedGaussianNoiseActor(5, 12, replace(cfg, action_magnitude=1.5))
    scaled.load_state_dict(unit.state_dict())
    obs = torch.randn(4, 5)
    w1, lp1 = unit.sample(obs, generator=torch.Generator().manual_seed(123))
    w2, lp2 = scaled.sample(obs, generator=torch.Generator().manual_seed(123))
    torch.testing.assert_close(w2, 1.5 * w1)
    torch.testing.assert_close(lp1, lp2, rtol=0, atol=0)


def test_updates_follow_official_block_order_and_leave_prior_frozen():
    agent, replay, _ = _agent()
    events = []
    for name, optimizer in (("actor", agent.actor_optimizer),
                            ("action", agent.q_action_optimizer),
                            ("noise", agent.q_noise_optimizer)):
        step = optimizer.step
        def counted(*args, _step=step, _name=name, **kwargs):
            events.append(_name)
            return _step(*args, **kwargs)
        optimizer.step = counted
    prior_before = source_semantic_digest(agent.decoder.prior)
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    metrics = agent.update(replay)
    assert events == ["action", "actor"] * 3 + ["noise"] * 2
    assert metrics["DSRL/q_noise_updates"] == 2
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(actor_before, agent.actor.parameters()))
    assert source_semantic_digest(agent.decoder.prior) == prior_before
    assert all(p.grad is None for p in agent.decoder.prior.parameters())


def test_bellman_target_discounts_once_per_chunk_and_masks_terminals():
    agent, replay, _ = _agent()
    agent.cfg = replace(agent.cfg, gamma=0.9, gradient_steps=1)
    replay.rewards[:4] = 3.0
    replay.dones[:4] = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    batch = {key: getattr(replay, key)[:4] for key in replay.sample(4)}
    replay.sample = lambda _: batch
    sample_actor = agent.actor.sample
    def zero_entropy(*args, **kwargs):
        noise, log_prob = sample_actor(*args, **kwargs)
        return noise, log_prob * 0.0
    agent.actor.sample = zero_entropy
    class ConstantTarget(torch.nn.Module):
        def forward(self, obs, action):
            value = torch.full((len(obs), 1), 2.0, device=obs.device)
            return value, value
    agent.q_action_target = ConstantTarget()
    metrics = agent.update(replay)
    assert metrics["DSRL/q_target"] == pytest.approx((3.0 + 3.0 + 1.8) / 2.0)


def test_default_discount_preserves_finite_route_potential_objective():
    from scripts.dppo_diffusion_rl.rewards import schedule_error_improvement
    assert DSRLNAConfig().gamma == 1.0
    # Equal endpoints reached by different intermediate paths must have the
    # same return for a pure potential difference, including chunk aggregation.
    for potential in (torch.tensor([0., -2., -1., -3., -1.]),
                      torch.tensor([0.,  1.,  2.,  0., -1.])):
        chunks = torch.diff(potential).reshape(2, 2).sum(-1)
        assert chunks.sum().item() == pytest.approx(-1.)
    # Exercise the actual task function: same duration and endpoint, different
    # progress histories. Their undiscounted chunk returns must agree.
    returns = []
    discounted_returns = []
    for progress in (torch.tensor([0., .5, 1., 1.5, 2.]),
                     torch.tensor([.5, 1., 1.5, 2., 2.])):
        rewards = schedule_error_improvement(
            progress=progress, progress_delta=torch.diff(progress, prepend=torch.zeros(1)),
            elapsed_s=torch.arange(1, 6, dtype=torch.float32),
            desired_speed=torch.full((5,), .5), route_length=torch.full((5,), 2.), dt=1.,
        )
        returns.append(rewards.sum().item())
        discounted_returns.append((rewards * .99 ** torch.arange(5)).sum().item())
    assert returns == pytest.approx([-.5, -.5])
    assert abs(discounted_returns[0] - discounted_returns[1]) > .01


def test_training_rejects_discounted_endpoint_rewards(tmp_path):
    agent, _, _ = _agent(gamma=0.99)
    class Env:
        @property
        def unwrapped(self):
            return self
    with pytest.raises(ValueError, match="require gamma=1.0"):
        DSRLNATrainer(env=Env(), agent=agent, source={}, source_path="unused",
                      source_sha="unused", output_dir=tmp_path)


def test_logged_bellman_components_reconstruct_mean_target():
    agent, replay, _ = _agent()
    replay.dones[:4] = torch.tensor([[0.], [1.], [0.], [1.]])
    metrics = agent.update(replay)
    assert metrics["DSRL/q_target"] == pytest.approx(
        metrics["DSRL/reward"] + metrics["DSRL/bootstrap_contribution"] + metrics["DSRL/entropy_contribution"],
        abs=1.e-5,
    )


def test_collector_sums_chunk_rewards_and_stops_terminal_rows(tmp_path):
    agent, _, _ = _agent()
    class Env:
        num_envs, device = 4, "cpu"
        def __init__(self):
            self.unwrapped = self
            self.steps = 0
            self.boundary_resets = []
        def get_proprioception(self):
            return torch.full((4, 30), float(self.steps))
        def get_goal(self):
            return torch.zeros(4, 16)
        def step(self, action):
            self.steps += 1
            done = torch.zeros(4, dtype=torch.bool)
            if self.steps == 2:
                done[0] = True
            return None, torch.ones(4), done, torch.zeros_like(done), {}
        def reset_after_action_chunk(self, ids):
            self.boundary_resets.extend(ids.tolist())
    env = Env()
    trainer = DSRLNATrainer(env=env, agent=agent, source={}, source_path="unused",
                           source_sha="unused", output_dir=tmp_path, rollout_chunks=2)
    trainer._seed_history()
    trainer.collect_batch()
    torch.testing.assert_close(trainer.replay.rewards[:8, 0], torch.tensor([2., 4., 4., 4., 4., 4., 4., 4.]))
    assert trainer.replay.dones[:8, 0].sum() == 1
    assert env.boundary_resets == [0]
    assert trainer.control_steps == 32


@pytest.mark.parametrize("family", ["wc", "wcp"])
@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_checkpoint_and_eval_api_preserve_seeded_executable_actions(tmp_path, family, gamma):
    agent, _, inputs = _agent(gamma=gamma)
    flat = asdict(agent.decoder.prior.cfg)
    source = {
        "schema_version": 8, "policy_kind": "spatial_hindsight_height_profile_ddpm",
        "config": {"model": flat, "diffusion": flat,
                   "dataset": {**flat, "goal_representation": "hindsight_geom_profile16"}},
        "normalizer_stats": agent.stats.to_dict(),
    }
    path = tmp_path / "dsrl.pt"
    save_dsrl_na_checkpoint(
        path, source=source, source_path=path, source_sha=PHASE_A_SOURCES[family],
        source_digest=source_semantic_digest(agent.decoder.prior), agent=agent,
        iteration=0, total_control_transitions=16, metrics={}, task_config={},
    )
    saved = torch.load(path, weights_only=False)
    assert saved["source_prior_family"] == family
    loaded = build_dsrl_na_inference_policy(saved)
    expected = agent.act(*inputs, generator=torch.Generator().manual_seed(700))[1]
    full = loaded.predict_action_denormalized(*inputs, generator=torch.Generator().manual_seed(700))
    torch.testing.assert_close(loaded.executable_chunk(full, 4), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="horizon"):
        loaded.executable_chunk(full, 8)
    with pytest.raises(ValueError, match="seeded generator"):
        loaded.predict_action_denormalized(*inputs)
    loaded.set_sampling_mode("deterministic")
    torch.testing.assert_close(loaded.predict_action(*inputs), loaded.predict_action(*inputs), rtol=0, atol=0)
    saved["dsrl_checkpoint_version"] = 1
    with pytest.raises(ValueError, match="version"):
        build_dsrl_na_inference_policy(saved)


def test_explicit_wc_selector_rejects_wcp_prior_before_loading(monkeypatch):
    from scripts.latent_noise_rl import checkpointing
    monkeypatch.setattr(checkpointing, "sha256_file", lambda _: PHASE_A_SOURCES["wcp"])
    with pytest.raises(ValueError, match=PHASE_A_SOURCES["wc"]):
        checkpointing.load_phase_a_prior("unused.pt", expected_source_sha256=PHASE_A_SOURCES["wc"])
