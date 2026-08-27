"""Strict actor-only warm starts for RSL-RL fine-tuning."""

from __future__ import annotations

from pathlib import Path

import torch


_ACTOR_PREFIXES = ("actor.", "actor_obs_normalizer.")


def load_actor_warm_start(policy: torch.nn.Module, checkpoint_path: str) -> dict:
    """Load actor and actor-normalizer tensors while leaving critic/Adam fresh."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_state = payload.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {path}")

    target_state = policy.state_dict()
    expected_keys = {key for key in target_state if key.startswith(_ACTOR_PREFIXES)}
    source_keys = {key for key in source_state if key.startswith(_ACTOR_PREFIXES)}
    if source_keys != expected_keys:
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        raise ValueError(f"Incompatible actor checkpoint; missing={missing}, extra={extra}")

    actor_state = {}
    for key in sorted(expected_keys):
        source_tensor = source_state[key]
        if source_tensor.shape != target_state[key].shape:
            raise ValueError(
                f"Warm-start tensor {key} has shape {tuple(source_tensor.shape)}, "
                f"expected {tuple(target_state[key].shape)}."
            )
        actor_state[key] = source_tensor

    incompatible = torch.nn.Module.load_state_dict(policy, actor_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected warm-start keys: {incompatible.unexpected_keys}")
    print(
        f"[INFO]: Warm-started actor from {path}; critic, exploration noise, optimizer and iteration are fresh."
    )
    return payload.get("infos", {})
