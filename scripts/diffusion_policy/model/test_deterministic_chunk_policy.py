"""Focused CPU tests for the matched deterministic action-chunk baseline."""

from __future__ import annotations

import ast
import numpy as np
from pathlib import Path
import pytest
import torch

from model.solo12_deterministic_chunk_policy import (
    Solo12DeterministicChunkPolicy,
    Solo12DeterministicChunkPolicyConfig,
)
from train.data.normalization import MinMaxStats, NormalizerStats, ZScoreStats


def _stats() -> NormalizerStats:
    return NormalizerStats(
        proprio=ZScoreStats(np.zeros(30, dtype=np.float32), np.ones(30, dtype=np.float32)),
        goal=ZScoreStats(np.zeros(16, dtype=np.float32), np.ones(16, dtype=np.float32)),
        action=MinMaxStats(-np.ones(12, dtype=np.float32), np.ones(12, dtype=np.float32)),
    )


def _policy() -> Solo12DeterministicChunkPolicy:
    policy = Solo12DeterministicChunkPolicy(
        Solo12DeterministicChunkPolicyConfig(d_model=32, nhead=4, num_layers=2, p_drop_attn=0.0)
    )
    policy.set_normalizer_stats(_stats())
    return policy


@pytest.mark.parametrize("batch_size", [1, 4])
def test_shapes_loss_gradients_and_condition_dependence(batch_size: int) -> None:
    torch.manual_seed(0)
    policy = _policy().train()
    batch = {
        "proprio_hist": torch.randn(batch_size, 8, 30),
        "action_hist": torch.randn(batch_size, 8, 12).clamp(-1, 1),
        "goal_hist": torch.randn(batch_size, 8, 16),
        "actions": torch.randn(batch_size, 16, 12).clamp(-1, 1),
    }
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in (policy.model.io_encoder[0].weight, policy.model.query_emb, policy.model.decoder.layers[0].self_attn.in_proj_weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    policy.eval()
    first = policy.predict_action(batch["proprio_hist"], batch["action_hist"], batch["goal_hist"])
    changed_goal = batch["goal_hist"].clone()
    changed_goal[..., 0] += 1.0
    second = policy.predict_action(batch["proprio_hist"], batch["action_hist"], changed_goal)
    assert first.shape == (batch_size, 16, 12)
    assert not torch.equal(first, second)


def test_determinism_clamp_and_execution_offset() -> None:
    policy = _policy().eval()
    proprio, action, goal = torch.randn(1, 8, 30), torch.zeros(1, 8, 12), torch.randn(1, 8, 16)
    first = policy.predict_action(proprio, action, goal)
    second = policy.predict_action(proprio, action, goal)
    assert torch.equal(first, second)
    assert torch.all(first <= 1.0) and torch.all(first >= -1.0)
    with torch.no_grad():
        policy.model.head.bias.fill_(3.0)
    raw = policy._normalized_prediction(proprio, action, goal)
    deployed = policy.predict_action(proprio, action, goal)
    assert raw.max() > 1.0 and deployed.max() == 1.0
    trajectory = torch.arange(16 * 12, dtype=torch.float32).reshape(1, 16, 12)
    assert torch.equal(policy.executable_chunk(trajectory, 4), trajectory[:, 8:12])
    with pytest.raises(ValueError):
        policy.executable_chunk(trajectory, 9)
    with pytest.raises(ValueError):
        policy.predict_action(proprio, action, goal, guidance_scale=1.1)


def test_state_round_trip_reproduces_prediction(tmp_path) -> None:
    policy = _policy().eval()
    inputs = (torch.randn(1, 8, 30), torch.zeros(1, 8, 12), torch.randn(1, 8, 16))
    expected = policy.predict_action(*inputs)
    path = tmp_path / "deterministic.pt"
    torch.save({"ema_model_state_dict": policy.state_dict(), "normalizer_stats": _stats().to_dict()}, path)
    restored = _policy().eval()
    saved = torch.load(path, weights_only=False)
    restored.load_state_dict(saved["ema_model_state_dict"])
    restored.set_normalizer_stats(saved["normalizer_stats"])
    assert torch.equal(expected, restored.predict_action(*inputs))


def test_evaluator_dispatch_helpers_remain_top_level() -> None:
    evaluator = Path(__file__).resolve().parents[1] / "evaluate_policy.py"
    tree = ast.parse(evaluator.read_text(encoding="utf-8"))
    top_level_functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_deterministic_model_cfg_from_checkpoint" in top_level_functions
    scenarios = top_level_functions["make_scenarios"]
    assert any(isinstance(node, ast.Return) for node in ast.walk(scenarios))
