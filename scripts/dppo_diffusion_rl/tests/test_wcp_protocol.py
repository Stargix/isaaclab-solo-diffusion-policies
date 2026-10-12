from pathlib import Path

import pytest

from scripts.dppo_diffusion_rl.experiments.wcp_nominal_posture_v1 import train


def test_frozen_protocol_keeps_gait_agnostic_task_and_phase_a_start():
    args = train.training_arguments("cuda:0", False)
    def value(flag):
        return args[args.index(flag) + 1]
    assert "--require_phase_a_source" in args and "--restart_optimization" not in args
    assert value("--posture_tracking_mode") == "nominal_window"
    assert value("--vertical_velocity_reward_weight") == "0.0"
    assert value("--transition_margin_m") == "0.0"
    assert value("--profile_height_reward_weight") == "0.75"
    assert value("--route_distribution") == "supported_hybrid_v3"
    assert "--pace_consistent_preview" not in args
    assert value("--reference_kl_coef") == "0.0"
    assert value("--symmetry_augmentation") == "none"
    assert value("--iterations") == "150"
    assert "--select_rollout_actor" in args
    assert "--wandb" not in args
    assert "--wandb" in train.training_arguments("cuda:0", True)


def test_preflight_rejects_missing_wrong_prior_and_populated_output(tmp_path, monkeypatch):
    prior = tmp_path / "best.pt"
    output = tmp_path / "output"
    monkeypatch.setattr(train, "PRIOR", prior)
    monkeypatch.setattr(train, "OUTPUT", output)
    with pytest.raises(FileNotFoundError):
        train.preflight()
    prior.write_bytes(b"not the frozen prior")
    with pytest.raises(ValueError, match="Wrong Phase-A"):
        train.preflight()
    import hashlib
    monkeypatch.setattr(train, "PRIOR_SHA256", hashlib.sha256(prior.read_bytes()).hexdigest())
    train.preflight()
    output.mkdir()
    (output / "metrics.jsonl").write_text("{}")
    with pytest.raises(FileExistsError):
        train.preflight()
