"""Helpers for plotting only physically observed route samples."""

import numpy as np


def active_trajectory_xy(positions, active_steps):
    """Return an (N, 2) array, including for failed/empty episodes."""
    if active_steps < 0:
        raise ValueError("active_steps must be nonnegative")
    points = np.asarray(positions[:active_steps], dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("trajectory samples must have shape (N, 2)")
    return points
