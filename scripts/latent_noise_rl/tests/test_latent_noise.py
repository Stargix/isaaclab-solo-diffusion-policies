"""Small CPU tests for the latent steering math and frozen DDIM mapping."""

import numpy as np
import pytest
import torch

from scripts.diffusion_policy.model.solo12_diffusion_policy import (
    Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig,
)
from scripts.diffusion_policy.train.data.normalization import (
    MinMaxStats, NormalizerStats, ZScoreStats,
)
from scripts.diffusion_policy.train.runtime.checkpoint import load_training_checkpoint
from scripts.dppo_diffusion_rl.buffer import RolloutBatch
from scripts.dppo_diffusion_rl.critic import ValueCritic
from scripts.gaussian_chunk_rl.config import GaussianPPOConfig
from scripts.latent_noise_rl.checkpointing import (
    SOURCE_SHA256, build_inference_policy, save_latent_checkpoint, source_semantic_digest,
)
from scripts.latent_noise_rl.config import LatentPolicyConfig
from scripts.latent_noise_rl.decoder import FrozenDDIMDecoder
from scripts.latent_noise_rl.policy import LatentNoisePolicy
from scripts.latent_noise_rl.ppo import LatentNoisePPOUpdater


def _fixture():
    torch.manual_seed(12)
    cfg = Solo12DiffusionPolicyConfig(
        proprio_dim=30, action_hist_dim=12, goal_dim=16, action_dim=12,
        history=8, prediction_horizon=16, execution_offset=8,
        d_model=16, nhead=4, num_layers=1, p_drop_emb=0.0, p_drop_attn=0.0,
        num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", clip_sample=True,
    )
    prior = Solo12DiffusionPolicy(cfg)
    stats = NormalizerStats(
        ZScoreStats(np.zeros(30, np.float32), np.ones(30, np.float32)),
        ZScoreStats(np.zeros(16, np.float32), np.ones(16, np.float32)),
        MinMaxStats(-np.ones(12, np.float32), np.ones(12, np.float32)),
    )
    prior.set_normalizer_stats(stats)
    latent = LatentPolicyConfig()
    ppo = GaussianPPOConfig(
        initial_std=1.0, min_std=.1, max_std=2.0, actor_lr=1e-5,
        adaptive_actor_lr=True, critic_warmup_iterations=0,
        update_epochs=1, minibatch_size=8, critic_minibatch_size=4,
        kl_probe_size=8,
    )
    policy = LatentNoisePolicy(FrozenDDIMDecoder(prior, inference_steps=10), ppo, latent)
    x = (torch.randn(4, 8, 30), torch.randn(4, 8, 12), torch.randn(4, 8, 16))
    return policy, x


def test_ddim_matches_direct_loop_is_deterministic_and_does_not_draw_rng():
    policy, x = _fixture()
    decoder = policy.decoder
    p, a, g = x
    noise = torch.randn(4, 16, 12)
    rng_before = torch.random.get_rng_state().clone()
    first = decoder.decode(p, a, g, noise)
    second = decoder.decode(p, a, g, noise)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    stats = decoder.prior.normalizer_stats
    from scripts.diffusion_policy.train.data.normalization import normalize_minmax, normalize_zscore
    pn = normalize_zscore(p, stats.proprio)
    an = normalize_minmax(a, stats.action)
    gn = normalize_zscore(g, stats.goal)
    expected = noise.clone()
    decoder.scheduler.set_timesteps(10, device=noise.device)
    for timestep in decoder.scheduler.timesteps:
        t = torch.full((len(noise),), int(timestep), dtype=torch.long)
        eps = decoder.prior._predict_noise(expected, pn, an, gn, t)
        expected = decoder.scheduler.step(eps, timestep, expected, eta=0.0,
                                          use_clipped_model_output=False).prev_sample
    torch.testing.assert_close(first, expected.clamp(-1, 1), rtol=1e-6, atol=1e-6)


def test_future_noise_suffix_cannot_change_executable_chunk():
    policy, x = _fixture()
    first = torch.randn(4, 16, 12)
    second = first.clone()
    second[:, 12:] = torch.randn_like(second[:, 12:]) * 100
    out_a = policy.decoder.decode(*x, first)[:, 8:12]
    out_b = policy.decoder.decode(*x, second)[:, 8:12]
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=1e-6)


def test_action_mapping_and_fixed_execution_slice():
    policy, x = _fixture()
    sample = policy.sample(*x, generator=torch.Generator().manual_seed(19))
    stats = policy.normalizer_stats.action
    expected = 0.5 * (sample.normalized_trajectory + 1.0) * (stats.max - stats.min) + stats.min
    torch.testing.assert_close(sample.denormalized_trajectory, expected)
    chunk = policy.executable_chunk(sample.denormalized_trajectory)
    torch.testing.assert_close(chunk, sample.denormalized_trajectory[:, 8:12])
    with pytest.raises(ValueError, match="four-action horizon"):
        policy.executable_chunk(sample.denormalized_trajectory, 8)


def test_raw_joint_logprob_ratio_and_tanh_jacobian_cancellation():
    policy, x = _fixture()
    mean, dist = policy.distribution(*x)
    raw = torch.randn_like(mean)
    lp_old = dist.log_prob(raw).sum((-1, -2))
    lp_new = policy.logprob(*x, raw)[0]
    torch.testing.assert_close((lp_new - lp_old).exp(), torch.ones_like(lp_old))
    mean2 = mean + .02
    new_dist = torch.distributions.Normal(mean2, dist.scale)
    raw_ratio = (new_dist.log_prob(raw) - dist.log_prob(raw)).sum((-1, -2)).exp()
    w = policy.latent_cfg.noise_scale * torch.tanh(raw)
    log_jac = (np.log(policy.latent_cfg.noise_scale) + 2 * (np.log(2.0) - raw - torch.nn.functional.softplus(-2 * raw))).sum((-1, -2))
    transformed_ratio = ((new_dist.log_prob(raw).sum((-1, -2)) - log_jac)
                         - (dist.log_prob(raw).sum((-1, -2)) - log_jac)).exp()
    torch.testing.assert_close(raw_ratio, transformed_ratio, rtol=2e-5, atol=2e-5)
    assert torch.isfinite(policy.logprob(*x, raw * 100)[0]).all()
    assert w.shape == (4, 12, 12)


def test_gaussian_and_initial_reference_modes_are_distinct():
    policy, x = _fixture()
    gen_a = torch.Generator().manual_seed(55)
    gen_b = torch.Generator().manual_seed(55)
    policy.set_reference_mode("gaussian")
    gaussian = policy.sample(*x, generator=gen_a).normalized_trajectory
    policy.set_reference_mode("initial")
    initial = policy.sample(*x, generator=gen_b).normalized_trajectory
    assert not torch.allclose(gaussian, initial)


def test_latent_update_changes_actor_but_not_frozen_prior():
    policy, x = _fixture()
    device = x[0].device
    priv = torch.randn(4, 6)
    critic_obs = policy.critic_condition(*x, priv)
    critic = ValueCritic(critic_obs.shape[-1], hidden_dims=(32, 16))
    updater = LatentNoisePPOUpdater(policy, critic, policy.ppo_cfg)
    prior_before = source_semantic_digest(policy.decoder.prior)
    actor_before = {k: v.detach().clone() for k, v in policy.actor.state_dict().items()}
    sample = policy.sample(*x, generator=torch.Generator().manual_seed(100))
    observations = tuple(value.reshape(1, 4, *value.shape[1:]).expand(2, -1, *([-1] * (value.ndim - 1))).contiguous()
                        for value in x)
    critic_rollout = critic_obs.reshape(1, 4, -1).expand(2, -1, -1).contiguous()
    latent_rollout = sample.chain.reshape(1, 4, *sample.chain.shape[1:]).expand(2, -1, *([-1] * (sample.chain.ndim - 1))).contiguous()
    old_lp = sample.old_logprobs.reshape(1, 4, 1).expand(2, -1, -1).contiguous()
    old_values = critic(critic_obs).detach().reshape(1, 4).expand(2, -1).contiguous()
    rollout = RolloutBatch(
        proprio=observations[0], action_history=observations[1], goals=observations[2],
        critic_observation=critic_rollout, chains=latent_rollout, old_logprobs=old_lp,
        old_values=old_values, rewards=torch.randn(2, 4), dones=torch.zeros(2, 4, dtype=torch.bool),
    )
    rollout.compute_gae(torch.zeros(4), gamma=1.0, gae_lambda=.95)
    metrics = updater.update(rollout, update_actor=True)
    assert metrics["Policy/actor_updates"] > 0
    assert any(not torch.equal(actor_before[k], v.detach()) for k, v in policy.actor.state_dict().items())
    assert source_semantic_digest(policy.decoder.prior) == prior_before
    optimizer_ids = {id(p) for group in updater.actor_optimizer.param_groups for p in group["params"]}
    assert optimizer_ids == {id(p) for p in policy.actor.parameters()} | {id(policy.log_std)}


def test_generic_loader_rejects_latent_actor_by_default(tmp_path):
    path = tmp_path / "latent.pt"
    torch.save({"algorithm": "latent_noise_ppo"}, path)
    with pytest.raises(ValueError, match="latent-noise PPO actor"):
        load_training_checkpoint(path, "cpu")


def test_latent_checkpoint_roundtrip_preserves_seeded_inference(tmp_path):
    policy, x = _fixture()
    critic = ValueCritic(policy.cfg.history * (30 + 12 + 16) + 6, hidden_dims=(32, 16))
    updater = LatentNoisePPOUpdater(policy, critic, policy.ppo_cfg)
    source = {
        "schema_version": 8,
        "policy_kind": "spatial_hindsight_height_profile_ddpm",
        "config": {
            "model": {
                "proprio_dim": 30, "action_hist_dim": 12, "goal_dim": 16,
                "action_dim": 12, "d_model": 16, "nhead": 4,
                "num_layers": 1, "p_drop_emb": 0.0, "p_drop_attn": 0.0,
                "separate_goal_conditioning": True,
            },
            "dataset": {
                "history": 8, "prediction_horizon": 16, "execution_offset": 8,
                "goal_representation": "hindsight_geom_profile16",
            },
            "diffusion": {
                "num_train_timesteps": 10, "beta_start": 0.0001, "beta_end": 0.02,
                "beta_schedule": "squaredcos_cap_v2", "prediction_type": "epsilon",
                "variance_type": "fixed_small", "clip_sample": True,
                "cfg_dropout_prob": 0.0,
            },
        },
        "normalizer_stats": policy.normalizer_stats.to_dict(),
        "ema_model_state_dict": policy.decoder.prior.state_dict(),
        "split_manifest": {},
    }
    policy.source_sha256 = SOURCE_SHA256
    digest = source_semantic_digest(policy.decoder.prior)
    path = tmp_path / "latent.pt"
    save_latent_checkpoint(
        path, source=source, source_path=path, source_sha=SOURCE_SHA256,
        source_digest=digest, policy=policy, critic=critic, updater=updater,
        iteration=0, total_control_transitions=0, metrics={}, best_score=0.0,
        task_config={"posture_tracking_mode": "nominal_window"},
    )
    restored = torch.load(path, map_location="cpu", weights_only=False)
    loaded_policy = build_inference_policy(restored, "cpu")
    generator_a = torch.Generator().manual_seed(777)
    generator_b = torch.Generator().manual_seed(777)
    expected = policy.eval().predict_action(*x, generator=generator_a)
    actual = loaded_policy.predict_action(*x, generator=generator_b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored["resume_supported"] is False


def test_eval_play_dispatch_is_explicit_and_paired_recipe_is_complete():
    from scripts.latent_noise_rl.experiments.wcp_pilot import evaluation_commands

    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    for name in ("evaluate_policy.py", "play_policy.py"):
        source = (root / "scripts" / "diffusion_policy" / name).read_text(encoding="utf-8")
        compile(source, name, "exec")
        assert 'checkpoint.get("algorithm") in {"latent_noise_ppo", "dsrl_na"}' in source
        assert 'predict_kwargs["generator"] = latent_generator' in source
    commands = evaluation_commands(
        launcher="./isaaclab.sh", checkpoint="/models/latent.pt",
        output_root="/tmp/latent_eval",
    )
    assert len(commands) == 9  # 3 reference modes x upright/crouch/transitions
    assert all("--require_empty_output_dir" in command for command in commands)
    assert all("--exec_horizon 4" in command and "--num_inference_steps 10" in command
               for command in commands)
    assert all("wct_final_evaluation_v1/route_banks/wct_id_routes_v1.npz"
               in command.replace("\\", "/") for command in commands)
    assert all("--task solo12-v0" in command for command in commands)


def test_training_seed_applies_before_actor_initialization(monkeypatch, tmp_path, capsys):
    import copy
    from types import SimpleNamespace
    from scripts.latent_noise_rl import train

    fixture, _ = _fixture()
    args = SimpleNamespace(
        checkpoint="synthetic.pt", output_dir=str(tmp_path / "preflight"),
        preflight_only=True, smoke=False, num_envs=4096, iterations=150,
        rollout_chunks=32, seed=42, save_interval=10,
    )
    monkeypatch.setattr(train, "make_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(train, "load_phase_a_prior", lambda *_: (
        {"policy_kind": "spatial_hindsight_height_profile_ddpm", "schema_version": 8},
        copy.deepcopy(fixture.decoder.prior), SOURCE_SHA256,
    ))
    weights = []
    real_policy = train.LatentNoisePolicy

    def capture_policy(*args, **kwargs):
        policy = real_policy(*args, **kwargs)
        weights.append(policy.actor[0].weight.detach().clone())
        return policy

    monkeypatch.setattr(train, "LatentNoisePolicy", capture_policy)
    torch.manual_seed(123)
    train.main()
    torch.manual_seed(456)
    train.main()
    torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    assert capsys.readouterr().out.count("PREFLIGHT_OK") == 2


@pytest.mark.parametrize("frames", [20, 225])
def test_analyzer_keeps_insufficient_cases_and_labels_unloading(tmp_path, frames):
    import csv
    import json
    from scripts.latent_noise_rl.analyze import analyze_runs

    directory = tmp_path / "trace"
    directory.mkdir()
    np.savez(
        directory / "timeseries.npz", time_s=np.arange(frames) * .02,
        valid=np.ones((frames, 1), dtype=bool), positions_xy=np.zeros((frames, 1, 2)),
        progress_m=np.zeros((frames, 1)), tangent_speed_m_s=np.zeros((frames, 1)),
        achieved_height_m=np.full((frames, 1), .2932),
        required_height_m=np.full((frames, 1), .2932),
        requested_speed_m_s=np.asarray([.2]), foot_contact=np.ones((frames, 1, 4), dtype=bool),
    )
    metadata = {
        "dt_s": .02, "feet_body_names": ["FL_FOOT", "FR_FOOT", "RL_FOOT", "RR_FOOT"],
        "scenarios": [{"path_xy": [[0., 0.], [4., 0.]], "path_cumulative_m": [0., 4.],
                       "path_shape": "procedural", "repeat": 0, "path_height_m": .2932}],
    }
    (directory / "timeseries_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "analysis"
    analyze_runs({"checkpoint": directory}, output)
    with (output / "latent_gait_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    if frames == 20:
        assert rows[0]["gait_data_status"] == "insufficient"
        assert rows[0]["height_mae_m"] == ""
    else:
        assert rows[0]["gait_data_status"] == "sufficient"
        assert float(rows[0]["unloaded_fraction"]) == 0.0
    assert "flight_fraction" not in rows[0]
