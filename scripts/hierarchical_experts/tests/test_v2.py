from types import SimpleNamespace

import pytest
import torch

from scripts.dppo_diffusion_rl.critic import ValueCritic
from scripts.dppo_diffusion_rl.supported_hybrid_v3 import SupportedHybridV3RouteBank
from scripts.hierarchical_experts.policy_v2 import HierarchyV2
from scripts.hierarchical_experts.hybrid_ppo_v2 import HybridPPOUpdaterV2
from scripts.hierarchical_experts.trainer_v2 import HybridRolloutV2, validate_v2_resume_contract


def _inputs(batch=16):
    return (torch.randn(batch, 8, 30), torch.randn(batch, 8, 12), torch.randn(batch, 8, 16))


def test_v2_samples_and_replays_two_conditional_log_probabilities():
    torch.manual_seed(13)
    policy = HierarchyV2()
    p, a, g = _inputs()
    sampled = policy.sample(p, a, g)
    d, c, _, _ = policy.logprobs(p, a, g, sampled.skill, sampled.command_latent)
    assert torch.equal(d, sampled.discrete_logprob)
    assert torch.equal(c, sampled.continuous_logprob)
    command = policy.command(sampled.skill, sampled.command_latent)
    assert (command.abs() <= policy.command_limits[sampled.skill]).all()
    assert policy.log_stds.shape == (2, 3)


def test_deterministic_diagnostics_do_not_change_policy_outputs_or_rng():
    torch.manual_seed(31)
    policy = HierarchyV2().eval()
    p, a, g = _inputs(11)
    rng_before = torch.random.get_rng_state().clone()
    expected_skill, expected_latent = policy.deterministic(p, a, g)
    rng_after_plain = torch.random.get_rng_state().clone()
    actual_skill, actual_latent, diagnostics = policy.deterministic_with_diagnostics(p, a, g)
    assert torch.equal(actual_skill, expected_skill)
    assert torch.equal(actual_latent, expected_latent)
    assert torch.equal(torch.random.get_rng_state(), rng_after_plain)
    assert diagnostics["skill_probabilities"].shape == (11, 2)
    assert torch.allclose(diagnostics["skill_probabilities"].sum(-1), torch.ones(11))
    assert torch.equal(diagnostics["command"], policy.command(actual_skill, actual_latent))


def test_unselected_continuous_head_does_not_affect_selected_likelihood_or_gradient():
    policy = HierarchyV2()
    p, a, g = _inputs(8)
    skill = torch.zeros(8, dtype=torch.long)
    latent = torch.randn(8, 3)
    before = policy.logprobs(p, a, g, skill, latent)[1].detach()
    with torch.no_grad():
        policy.command_heads[1].weight.add_(100.0)
    after = policy.logprobs(p, a, g, skill, latent)[1]
    assert torch.equal(before, after)
    after.sum().backward()
    assert policy.command_heads[0].weight.grad is not None
    assert policy.command_heads[1].weight.grad is None
    assert policy.log_stds.grad[1].abs().sum() == 0


def test_v2_selector_and_selected_head_receive_gradients_and_ppo_updates():
    torch.manual_seed(17)
    policy = HierarchyV2()
    critic = ValueCritic(478, hidden_dims=(32,))
    p, a, g = _inputs(8)
    sampled = policy.sample(p, a, g)
    d, c, _, _ = policy.logprobs(p, a, g, sampled.skill, sampled.command_latent)
    assert torch.equal(d, sampled.discrete_logprob)
    assert torch.equal(c, sampled.continuous_logprob)
    batch = SimpleNamespace(proprio=p, action_history=a, goals=g,
        skills=sampled.skill, command_latents=sampled.command_latent,
        old_discrete_logprobs=d.detach(), old_continuous_logprobs=c.detach(),
        critic_observation=torch.randn(8, 478), old_values=torch.zeros(8),
        rewards=torch.randn(8), dones=torch.zeros(8, dtype=torch.bool),
        advantages=torch.tensor([-2., -1., -.5, -.1, .1, .5, 1., 2.]),
        returns=torch.randn(8))
    updater = HybridPPOUpdaterV2(policy, critic, update_epochs=1,
        minibatch_size=8, critic_minibatch_size=8)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = updater.update(batch)
    assert metrics["Policy/actor_updates"] == 1
    assert metrics["Policy/critic_updates"] == 1
    assert abs(metrics["Policy/discrete_ratio_mean"] - 1.0) < 1e-6
    assert abs(metrics["Policy/continuous_ratio_mean"] - 1.0) < 1e-6
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))


def test_entropy_only_update_changes_selector_not_conditional_action_heads():
    torch.manual_seed(23)
    policy = HierarchyV2()
    critic = ValueCritic(478, hidden_dims=(32,))
    with torch.no_grad():
        policy.skill_head.bias.copy_(torch.tensor([2.0, -2.0]))
    p, a, g = _inputs(8)
    sampled = policy.sample(p, a, g)
    d, c, _, _ = policy.logprobs(p, a, g, sampled.skill, sampled.command_latent)
    batch = SimpleNamespace(proprio=p, action_history=a, goals=g,
        skills=sampled.skill, command_latents=sampled.command_latent,
        old_discrete_logprobs=d.detach(), old_continuous_logprobs=c.detach(),
        critic_observation=torch.randn(8, 478), old_values=torch.zeros(8),
        rewards=torch.zeros(8), dones=torch.zeros(8, dtype=torch.bool),
        advantages=torch.zeros(8), returns=torch.zeros(8))
    selector = policy.skill_head.bias.detach().clone()
    heads = [head.weight.detach().clone() for head in policy.command_heads]
    stds = policy.log_stds.detach().clone()
    updater = HybridPPOUpdaterV2(policy, critic, update_epochs=1,
        minibatch_size=8, critic_minibatch_size=8, discrete_entropy_coef=0.05)
    updater.update(batch)
    assert not torch.equal(selector, policy.skill_head.bias)
    assert all(torch.equal(old, head.weight) for old, head in zip(heads, policy.command_heads))
    assert torch.equal(stds, policy.log_stds)


def test_straight_curriculum_sampler_balances_profiles_and_speed_and_updates_full_route():
    bank = SupportedHybridV3RouteBank(16, "cpu", points=101, length_m=4.0)
    ids = torch.arange(16)
    profiles = bank.apply_straight_curriculum(ids)
    assert torch.bincount(profiles, minlength=4).tolist() == [4, 4, 4, 4]
    assert torch.equal(bank.xy[:, :, 1], torch.zeros(16, 101))
    assert torch.allclose(bank.xy[:, :, 0], bank.arc)
    assert set(bank.speed.tolist()) == {0.25, 0.5}
    assert torch.isinf(bank.transition_boundary[(profiles < 2)]).all()
    assert torch.equal(bank.transition_boundary[(profiles >= 2)], torch.full((8,), 2.0))
    assert torch.all(bank.height[0] == bank.WALK_HEIGHT_M)
    assert torch.all(bank.height[1] == bank.CROUCH_HEIGHT_M)
    assert (bank.height[2, 50] == bank.CROUCH_HEIGHT_M)
    assert (bank.height[3, 50] == bank.WALK_HEIGHT_M)
    saved = bank.curriculum_state_dict()
    restored = SupportedHybridV3RouteBank(16, "cpu", points=101, length_m=4.0)
    restored.load_curriculum_state_dict(saved)
    assert restored.curriculum_state_dict()["straight_curriculum_counter"] == 0
    assert torch.equal(restored.straight_curriculum_sample_counts,
                       bank.straight_curriculum_sample_counts)


def test_ten_step_macro_gae_masks_terminal_bootstrap():
    batch = HybridRolloutV2(
        proprio=torch.zeros(2, 1, 8, 30), action_history=torch.zeros(2, 1, 8, 12),
        goals=torch.zeros(2, 1, 8, 16), critic_observation=torch.zeros(2, 1, 478),
        skills=torch.zeros(2, 1, dtype=torch.long), command_latents=torch.zeros(2, 1, 3),
        old_discrete_logprobs=torch.zeros(2, 1), old_continuous_logprobs=torch.zeros(2, 1),
        old_values=torch.zeros(2, 1), rewards=torch.tensor([[1.], [2.]]),
        dones=torch.tensor([[False], [True]]))
    batch.compute_gae(torch.tensor([100.]), gae_lambda=0.95)
    assert torch.equal(batch.advantages, torch.tensor([2.9, 2.0]))
    assert torch.equal(batch.returns, torch.tensor([2.9, 2.0]))


def test_resume_contract_rejects_changed_experts_task_budget_and_preupdate():
    state = {"algorithm": "hierarchical_expert_ppo_v2", "format_version": 3,
        "architecture": "v2", "expert_sha256": {"walk": "w", "crouch": "c"},
        "dppo_task_config": {"route": 3}, "run_config": {"seed": 42},
        "planned_interaction_budget": 100, "route_sampler_curriculum_state": {},
        "metrics": {"checkpoint_phase": "post_update_unscored"}}
    args = dict(architecture="v2", expert_hashes={"walk": "w", "crouch": "c"},
        task_config={"route": 3}, run_config={"seed": 42}, interaction_budget=100)
    validate_v2_resume_contract(state, **args)
    for field, value in (("expert_hashes", {"walk": "x", "crouch": "c"}),
                         ("task_config", {"route": 2}), ("run_config", {"seed": 43}),
                         ("interaction_budget", 101)):
        changed = dict(args)
        changed[field] = value
        with pytest.raises(ValueError):
            validate_v2_resume_contract(state, **changed)
    pre_update = {**state, "metrics": {"checkpoint_phase": "pre_update"}}
    with pytest.raises(ValueError, match="post-update"):
        validate_v2_resume_contract(pre_update, **args)


def test_resume_recovers_checkpointed_metrics_and_rejects_newer_log_rows(tmp_path):
    from scripts.hierarchical_experts.trainer_v2 import HierarchyTrainerV2
    trainer = HierarchyTrainerV2.__new__(HierarchyTrainerV2)
    trainer.metrics_path = tmp_path / "metrics.jsonl"
    trainer.update_steps = 3
    trainer.total_env_transitions = 120
    trainer.metrics_path.write_text('{"update_steps": 2}\n', encoding="utf-8")
    trainer._reconcile_metrics({"iteration": 2, "metrics": {"metric": 7.0}})
    rows = [line for line in trainer.metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert '"total_50hz_env_transitions": 120' in rows[-1]
    trainer.metrics_path.write_text('{"update_steps": 4}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="newer than the resume checkpoint"):
        trainer._reconcile_metrics({"iteration": 2, "metrics": {"metric": 7.0}})


def test_v2_checkpoint_roundtrip_adapter_is_deterministic_and_version_checked(tmp_path):
    from scripts.dppo_diffusion_rl.checkpointing import sha256_file
    from scripts.hierarchical_experts.evaluation import EvaluationAdapter

    expert_state = {"actor_obs_normalizer._mean": torch.zeros(1, 48),
                    "actor_obs_normalizer._std": torch.ones(1, 48)}
    for index, (inputs, outputs) in zip((0, 2, 4, 6), ((48, 4), (4, 4), (4, 4), (4, 12))):
        expert_state[f"actor.{index}.weight"] = torch.zeros(outputs, inputs)
        expert_state[f"actor.{index}.bias"] = torch.zeros(outputs)
    expert_path = tmp_path / "expert.pt"
    torch.save({"model_state_dict": expert_state}, expert_path)
    policy = HierarchyV2()
    checkpoint_path = tmp_path / "v2.pt"
    torch.save({"algorithm": "hierarchical_expert_ppo_v2", "format_version": 3,
        "architecture": policy.architecture, "actor_state_dict": policy.state_dict(),
        "run_config": {"execution_steps": 10, "initial_std": .6, "min_std": .05,
                       "max_std": 1.0, "command_range_mode": "shared"},
        "dppo_task_config": {},
        "expert_sha256": {"walk": sha256_file(expert_path), "crouch": sha256_file(expert_path)}},
        checkpoint_path)
    adapter = EvaluationAdapter(checkpoint_path, {"dppo_task_config": {}}, expert_path,
                                expert_path, "cpu", 10)
    p, a, g = _inputs(3)
    first = adapter.predict_action_denormalized(p, a, g)
    second = adapter.predict_action_denormalized(p, a, g)
    assert first.shape == (3, 10, 4)
    assert torch.equal(first, second)
    bad = torch.load(checkpoint_path, weights_only=False)
    bad["architecture"] = "unversioned"
    torch.save(bad, checkpoint_path)
    with pytest.raises(ValueError, match="schema/architecture"):
        EvaluationAdapter(checkpoint_path, {"dppo_task_config": {}}, expert_path,
                          expert_path, "cpu", 10)
