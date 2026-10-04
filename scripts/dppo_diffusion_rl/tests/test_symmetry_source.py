"""Recover the paired Phase-A prior, never the fine-tuned historical actor."""

import pytest
import torch

from scripts.dppo_diffusion_rl.experiments.symmetry_lr_v1.prepare_source import recover_phase_a


def _archive():
    return {
        "algorithm": "dppo", "source_checkpoint_sha256": "source-sha",
        "schema_version": 8, "policy_kind": "profile16", "config": {},
        "normalizer_stats": {}, "split_manifest": {"seed": 42},
        "dppo_base_model_state_dict": {"weight": torch.tensor([1.0])},
        "ema_model_state_dict": {"model.weight": torch.tensor([7.0])},
        "critic_state_dict": {}, "actor_optimizer_state_dict": {}, "iteration": 85,
    }


def test_recovery_uses_only_immutable_phase_a_weights():
    recovered = recover_phase_a(_archive(), "source-sha")
    for key in ("model_state_dict", "ema_model_state_dict"):
        torch.testing.assert_close(recovered[key]["model.weight"], torch.tensor([1.0]))
    assert recovered["recovered_phase_a_source_sha256"] == "source-sha"
    assert recovered["split_manifest"] == {"seed": 42}
    assert not set(recovered) & {
        "algorithm", "dppo_config", "critic_state_dict", "actor_optimizer_state_dict", "iteration"
    }


def test_recovery_rejects_wrong_prior_or_partial_model():
    with pytest.raises(ValueError, match="required Phase A"):
        recover_phase_a(_archive(), "wrong-sha")
    archive = _archive()
    archive["ema_model_state_dict"]["model.missing"] = torch.ones(1)
    with pytest.raises(ValueError, match="exactly cover"):
        recover_phase_a(archive, "source-sha")
