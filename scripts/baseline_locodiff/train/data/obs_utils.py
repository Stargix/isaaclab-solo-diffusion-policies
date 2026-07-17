"""Shared observation / IO helpers for training and deployment."""

from __future__ import annotations

import h5py
import numpy as np
import torch

PROPRIO_KEYS = ("joint_pos", "joint_vel", "base_lin_vel", "base_ang_vel", "projected_gravity")
PROPRIO_DIM = 33
ACTION_HIST_DIM = 0
GOAL_DIM = 5  # [vx, vy, wz, skill_walk, skill_crouch]
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
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([joint_pos, joint_vel, base_lin_vel, base_ang_vel, projected_gravity], dim=-1)


def delayed_io_windows(
    proprio: np.ndarray,
    actions: np.ndarray,
    step: int,
    history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the state history used by the paper-aligned SDE policy.

    The paper conditions on orientation, twist, joint position and joint
    velocity histories.  It does not condition on past actions, so the second
    tensor deliberately has a zero-width feature dimension.
    """

    proprio_hist = proprio[step - history : step].astype(np.float32)
    action_hist = np.zeros((history, ACTION_HIST_DIM), dtype=np.float32)
    return proprio_hist, action_hist
