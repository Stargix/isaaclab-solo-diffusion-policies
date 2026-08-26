# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clock-aware left/right symmetry for the 50-D Solo12 bound policy."""

from __future__ import annotations

from typing import Any

import torch

try:
    from tensordict import TensorDict
except Exception:  # pragma: no cover - optional outside RSL-RL
    TensorDict = None


BOUND_OBSERVATION_SIZE = 50
BOUND_ACTION_SIZE = 12
BOUND_CLOCK_SLICE = slice(48, 50)

_VECTOR_REFLECT_LR = torch.tensor((1.0, -1.0, 1.0))
_PSEUDOVECTOR_REFLECT_LR = torch.tensor((-1.0, 1.0, -1.0))
_COMMAND_REFLECT_LR = torch.tensor((1.0, -1.0, -1.0))
_JOINT_LR_PERM = torch.tensor((3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8), dtype=torch.long)
_JOINT_LR_SIGN = torch.tensor((-1.0, 1.0, 1.0) * 4)


def _device_tensor(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return values.to(device=reference.device, dtype=reference.dtype)


def reflect_bound_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Swap left/right legs and reflect their abduction coordinate."""

    if actions.shape[-1] != BOUND_ACTION_SIZE:
        raise ValueError(f"Expected {BOUND_ACTION_SIZE} bound actions, got {actions.shape[-1]}.")
    permutation = _JOINT_LR_PERM.to(device=actions.device)
    sign = _device_tensor(_JOINT_LR_SIGN, actions)
    return actions[..., permutation] * sign


def reflect_bound_observations_left_right(observations: torch.Tensor) -> torch.Tensor:
    """Reflect the 48-D proprioceptive state while preserving the gait clock."""

    if observations.shape[-1] != BOUND_OBSERVATION_SIZE:
        raise ValueError(
            f"Expected {BOUND_OBSERVATION_SIZE} bound observations, got {observations.shape[-1]}."
        )

    reflected = observations.clone()
    reflected[..., 0:3] *= _device_tensor(_VECTOR_REFLECT_LR, reflected)
    reflected[..., 3:6] *= _device_tensor(_PSEUDOVECTOR_REFLECT_LR, reflected)
    reflected[..., 6:9] *= _device_tensor(_VECTOR_REFLECT_LR, reflected)
    reflected[..., 9:12] *= _device_tensor(_COMMAND_REFLECT_LR, reflected)
    reflected[..., 12:24] = reflect_bound_actions_left_right(reflected[..., 12:24])
    reflected[..., 24:36] = reflect_bound_actions_left_right(reflected[..., 24:36])
    reflected[..., 36:48] = reflect_bound_actions_left_right(reflected[..., 36:48])
    # FL/FR share a phase and RL/RR share a phase, so left/right reflection
    # leaves [sin(phase), cos(phase)] unchanged.
    reflected[..., BOUND_CLOCK_SLICE] = observations[..., BOUND_CLOCK_SLICE]
    return reflected


def _augment_tensor(observations: torch.Tensor) -> torch.Tensor:
    return torch.cat((observations, reflect_bound_observations_left_right(observations)), dim=0)


def _augment_observation_container(observations: Any, obs_type: str):
    if isinstance(observations, torch.Tensor):
        return _augment_tensor(observations)
    if TensorDict is None or not isinstance(observations, TensorDict):
        raise TypeError(f"Expected a torch.Tensor or TensorDict, got {type(observations)!r}.")
    if obs_type not in observations.keys(include_nested=False):
        raise KeyError(f"Expected observation key '{obs_type}', got {list(observations.keys())}.")

    batch_size = observations.batch_size[0]
    augmented = {}
    for key in observations.keys(include_nested=False):
        value = observations[key]
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == BOUND_OBSERVATION_SIZE:
            augmented[key] = _augment_tensor(value)
        elif isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
            augmented[key] = torch.cat((value, value), dim=0)
        else:
            augmented[key] = value
    return TensorDict(source=augmented, batch_size=[2 * batch_size], device=observations.device)


@torch.no_grad()
def compute_bound_left_right_symmetry(
    env: Any = None,
    obs: Any = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[Any, torch.Tensor | None]:
    """Return identity and left/right reflection for RSL-RL symmetry PPO."""

    del env  # The 50-D contract uniquely identifies this task.
    if obs_type != "policy":
        raise ValueError(f"Bound symmetry only supports obs_type='policy', got {obs_type!r}.")

    obs_augmented = None if obs is None else _augment_observation_container(obs, obs_type)
    actions_augmented = None
    if actions is not None:
        actions_augmented = torch.cat((actions, reflect_bound_actions_left_right(actions)), dim=0)
    return obs_augmented, actions_augmented
