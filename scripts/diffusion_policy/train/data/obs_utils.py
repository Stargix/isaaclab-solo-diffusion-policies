"""Shared observation / IO helpers for training and deployment."""

from __future__ import annotations

import h5py
import numpy as np
import torch

PROPRIO_KEYS = ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity")
PROPRIO_DIM = 30
ACTION_HIST_DIM = 12
GOAL_DIM = 11
IO_DIM = PROPRIO_DIM + ACTION_HIST_DIM


def read_proprio_vector(obs_group: h5py.Group | dict) -> np.ndarray:
    """Return proprioception without ``last_action`` (DiffuseLoco-style state)."""

    pieces = []
    for key in PROPRIO_KEYS:
        if isinstance(obs_group, h5py.Group):
            pieces.append(obs_group[key][:])
        else:
            pieces.append(obs_group[key])
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def proprio_from_env_tensors(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([joint_pos, joint_vel, base_ang_vel, projected_gravity], dim=-1)


def delayed_io_windows(
    proprio: np.ndarray,
    actions: np.ndarray,
    step: int,
    history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build delayed proprio/action histories for anchor ``step``.

    Matches DiffuseLoco notation at prediction time ``step``:
    - proprio: ``s_{step-history:step}`` (ends at ``step-1``)
    - actions: ``a_{step-history-1:step-1}`` (ends at ``step-2``)
    """

    proprio_hist = proprio[step - history : step].astype(np.float32)
    action_hist = np.zeros((history, ACTION_HIST_DIM), dtype=np.float32)
    for k in range(history):
        action_idx = step - history + k - 1
        if action_idx >= 0:
            action_hist[k] = actions[action_idx]
    return proprio_hist, action_hist
