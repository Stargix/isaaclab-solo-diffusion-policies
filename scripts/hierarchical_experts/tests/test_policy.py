from types import SimpleNamespace

import torch
import pytest

from scripts.gaussian_chunk_rl.config import GaussianPPOConfig
from scripts.hierarchical_experts.policy import Hierarchy, Experts
from scripts.dppo_diffusion_rl.buffer import RolloutBatch
from scripts.dppo_diffusion_rl.critic import ValueCritic
from scripts.gaussian_chunk_rl.ppo import GaussianPPOUpdater


def test_adaptive_lr_rejected_for_hybrid_policy():
    cfg = GaussianPPOConfig(initial_std=0.6, min_std=0.05, max_std=1.0,
                            adaptive_actor_lr=True)
    with pytest.raises(ValueError, match="not implemented for the hybrid policy"):
        Hierarchy(cfg)


def test_joint_logprob_and_command_limits():
    torch.manual_seed(7)
    cfg = GaussianPPOConfig(initial_std=0.6, min_std=0.05, max_std=1.0)
    policy = Hierarchy(cfg)
    p, a, g = torch.randn(8, 8, 30), torch.randn(8, 8, 12), torch.randn(8, 8, 16)
    latent, logprob = policy.sample(p, a, g)
    replayed, _ = policy.logprob(p, a, g, latent)
    assert torch.allclose(logprob, replayed)
    skill, cmd = policy.command(latent)
    assert set(skill.tolist()).issubset({0, 1})
    assert torch.all(cmd.abs() <= policy.command_limits[skill])
    (-replayed.mean()).backward()
    assert policy.skill_head.weight.grad.abs().sum() > 0
    assert policy.velocity_head.weight.grad.abs().sum() > 0


def test_longer_execution_interval_is_part_of_policy_config():
    cfg = GaussianPPOConfig(exec_horizon=12, initial_std=0.6, min_std=0.05, max_std=1.0)
    cfg.validate(prediction_horizon=12, execution_offset=0)
    policy = Hierarchy(cfg, "shared")
    assert policy.dppo_cfg.exec_horizon == 12
    latent, _ = policy.sample(torch.zeros(2, 8, 30), torch.zeros(2, 8, 12), torch.zeros(2, 8, 16))
    assert policy.command(latent)[1].shape == (2, 3)


def test_expert_observation_order():
    batch = 2
    data = SimpleNamespace(root_lin_vel_b=torch.full((batch, 3), 1.), root_ang_vel_b=torch.full((batch, 3), 2.),
                           projected_gravity_b=torch.full((batch, 3), 3.), joint_pos=torch.full((batch, 12), 5.),
                           joint_vel=torch.full((batch, 12), 6.))
    env = SimpleNamespace(_robot=SimpleNamespace(data=data), _joint_ids=list(range(12)),
                          _q_offset_action_and_obs=torch.ones(batch, 12), _actions=torch.full((batch, 12), 7.),
                          _applied_actions=torch.full((batch, 12), 8.),
                          cfg=SimpleNamespace(action_obs_is_last_executed=False))
    obs = Experts.observation(env, torch.full((batch, 3), 4.))
    assert obs.shape == (batch, 48)
    assert torch.equal(obs[0], torch.tensor([1.]*3 + [2.]*3 + [3.]*3 + [4.]*3 + [4.]*12 + [6.]*12 + [7.]*12))
    env.cfg.action_obs_is_last_executed = True
    assert (Experts.observation(env, torch.zeros(batch, 3))[:, -12:] == 8.).all()


def test_ppo_updates_joint_actor_and_critic():
    torch.manual_seed(9)
    cfg = GaussianPPOConfig(initial_std=0.6, min_std=0.05, max_std=1.0,
                            update_epochs=1, minibatch_size=8, critic_minibatch_size=8)
    policy = Hierarchy(cfg)
    critic = ValueCritic(478, hidden_dims=(32,))
    p, a, g = torch.randn(2, 4, 8, 30), torch.randn(2, 4, 8, 12), torch.randn(2, 4, 8, 16)
    with torch.no_grad():
        latent, old = policy.sample(p.flatten(0, 1), a.flatten(0, 1), g.flatten(0, 1))
    rollout = RolloutBatch(proprio=p, action_history=a, goals=g, critic_observation=torch.randn(2, 4, 478),
                           chains=latent.reshape(2, 4, 1, 4), old_logprobs=old.reshape(2, 4, 1),
                           old_values=torch.zeros(2, 4), rewards=torch.randn(2, 4),
                           dones=torch.zeros(2, 4, dtype=torch.bool))
    rollout.compute_gae(torch.zeros(4), gamma=1.0, gae_lambda=0.95)
    before_skill = policy.skill_head.weight.detach().clone()
    before_velocity = policy.velocity_head.weight.detach().clone()
    metrics = GaussianPPOUpdater(policy, critic, cfg).update(rollout)
    assert metrics["Policy/actor_updates"] > 0 and metrics["Policy/critic_updates"] > 0
    assert not torch.equal(before_skill, policy.skill_head.weight)
    assert not torch.equal(before_velocity, policy.velocity_head.weight)


def test_categorical_entropy_is_independent_of_velocity_head():
    cfg = GaussianPPOConfig(initial_std=0.6, min_std=0.05, max_std=1.0)
    policy = Hierarchy(cfg, "shared")
    p, a, g = torch.randn(8, 8, 30), torch.randn(8, 8, 12), torch.randn(8, 8, 16)
    with torch.no_grad():
        policy.skill_head.bias.copy_(torch.tensor([2.0, -2.0]))
    _, categorical = policy.distribution(p, a, g)
    (-categorical.entropy().mean()).backward()
    assert policy.skill_head.bias.grad.abs().sum() > 0
    assert policy.velocity_head.weight.grad is None
    assert torch.equal(policy.command_limits[0], policy.command_limits[1])


def test_ppo_entropy_bonus_changes_skill_without_velocity_gradient():
    cfg = GaussianPPOConfig(initial_std=0.6, min_std=0.05, max_std=1.0,
                            update_epochs=1, minibatch_size=8, critic_minibatch_size=8)
    policy = Hierarchy(cfg)
    with torch.no_grad():
        policy.skill_head.bias.copy_(torch.tensor([2.0, -2.0]))
    critic = ValueCritic(478, hidden_dims=(32,))
    p, a, g = torch.randn(2, 4, 8, 30), torch.randn(2, 4, 8, 12), torch.randn(2, 4, 8, 16)
    with torch.no_grad():
        latent, old = policy.sample(p.flatten(0, 1), a.flatten(0, 1), g.flatten(0, 1))
    rollout = RolloutBatch(proprio=p, action_history=a, goals=g, critic_observation=torch.randn(2, 4, 478),
                           chains=latent.reshape(2, 4, 1, 4), old_logprobs=old.reshape(2, 4, 1),
                           old_values=torch.zeros(2, 4), rewards=torch.zeros(2, 4),
                           dones=torch.zeros(2, 4, dtype=torch.bool), advantages=torch.zeros(2, 4),
                           returns=torch.zeros(2, 4))
    skill_before = policy.skill_head.bias.detach().clone()
    velocity_before = policy.velocity_head.weight.detach().clone()
    GaussianPPOUpdater(policy, critic, cfg, categorical_entropy_coef=0.01).update(rollout)
    assert not torch.equal(skill_before, policy.skill_head.bias)
    assert torch.equal(velocity_before, policy.velocity_head.weight)


def test_evaluation_adapter_is_deterministic_and_checks_hashes(tmp_path):
    from scripts.hierarchical_experts.evaluation import EvaluationAdapter
    from scripts.dppo_diffusion_rl.checkpointing import sha256_file
    state = {"actor_obs_normalizer._mean": torch.zeros(1, 48),
             "actor_obs_normalizer._std": torch.ones(1, 48)}
    for index, (inputs, outputs) in zip((0, 2, 4, 6), ((48, 4), (4, 4), (4, 4), (4, 12))):
        state[f"actor.{index}.weight"] = torch.zeros(outputs, inputs)
        state[f"actor.{index}.bias"] = torch.zeros(outputs)
    expert = tmp_path / "expert.pt"
    torch.save({"model_state_dict": state}, expert)
    cfg = GaussianPPOConfig(initial_std=0.6, min_std=0.05, max_std=1.0)
    actor = Hierarchy(cfg)
    checkpoint = tmp_path / "hierarchy.pt"
    torch.save({"algorithm": "hierarchical_expert_ppo", "ppo_config": cfg.to_dict(),
                "actor_state_dict": actor.state_dict(), "dppo_task_config": {},
                "expert_sha256": {"walk": sha256_file(expert), "crouch": sha256_file(expert)}}, checkpoint)
    adapter = EvaluationAdapter(checkpoint, {"dppo_task_config": {}}, expert, expert, "cpu", 4)
    p, a, g = torch.zeros(2, 8, 30), torch.zeros(2, 8, 12), torch.zeros(2, 8, 16)
    assert torch.equal(adapter.predict_action_denormalized(p, a, g),
                       adapter.predict_action_denormalized(p, a, g))
    with pytest.raises(ValueError, match="decision interval"):
        EvaluationAdapter(checkpoint, {"dppo_task_config": {}}, expert, expert, "cpu", 8)
    long_cfg = GaussianPPOConfig(exec_horizon=12, initial_std=0.6, min_std=0.05, max_std=1.0)
    payload = torch.load(checkpoint, weights_only=False)
    payload["ppo_config"] = long_cfg.to_dict()
    torch.save(payload, checkpoint)
    long_adapter = EvaluationAdapter(checkpoint, {"dppo_task_config": {}}, expert, expert, "cpu", 12)
    assert long_adapter.predict_action_denormalized(p, a, g).shape == (2, 12, 4)
    torch.save({"changed": True}, checkpoint.with_name("wrong.pt"))
    # Keep the actor loadable but deliberately change the recorded hash.
    payload = torch.load(checkpoint, weights_only=False)
    payload["expert_sha256"]["walk"] = "incorrect"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="hashes"):
        EvaluationAdapter(checkpoint, {"dppo_task_config": {}}, expert, expert, "cpu", 4)
