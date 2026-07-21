"""Versioned interfaces between the high-level policy and frozen DiffuseLoco."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ActionBounds:
    """Physical command limits used only as a safety envelope (not a ramp controller)."""

    vx: tuple[float, float] = (-0.75, 0.75)
    vy: tuple[float, float] = (-0.50, 0.50)
    wz: tuple[float, float] = (-0.50, 0.50)
    # These are the two posture commands present in the current walk+crouch
    # dataset.  Intermediate values are deliberately allowed as interpolation
    # probes, but the high-level policy never starts below the collected range.
    height: tuple[float, float] = (0.1705, 0.2932)

    @property
    def low(self) -> np.ndarray:
        return np.asarray([self.vx[0], self.vy[0], self.wz[0], self.height[0]], dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.asarray([self.vx[1], self.vy[1], self.wz[1], self.height[1]], dtype=np.float32)

    def clip(self, command: np.ndarray) -> np.ndarray:
        value = np.asarray(command, dtype=np.float32)
        if value.shape[-1] != 4:
            raise ValueError(f"Expected [...,4] command, got {value.shape}.")
        return np.clip(value, self.low, self.high)


@dataclass(frozen=True)
class RouteObservation:
    """Route-local observation presented to the high-level policy.

    ``preview`` is robot-frame ``(x, y, sin(yaw), cos(yaw), max_height, valid)``
    samples. ``max_height`` is a command-space clearance constraint, not a desired
    posture. The final pose is ``(x, y, sin(yaw), cos(yaw))`` in the same frame.
    """

    preview: np.ndarray
    final_pose: np.ndarray
    remaining_time: float
    state: np.ndarray
    clearance: float

    def as_vector(self) -> np.ndarray:
        preview = np.asarray(self.preview, dtype=np.float32)
        final_pose = np.asarray(self.final_pose, dtype=np.float32)
        state = np.asarray(self.state, dtype=np.float32)
        if preview.ndim != 2 or preview.shape[1] != 6:
            raise ValueError(f"preview must have shape [N,6], got {preview.shape}.")
        if final_pose.shape != (4,) or state.ndim != 1:
            raise ValueError("final_pose must be [4] and state must be a flat vector.")
        if not math.isfinite(float(self.remaining_time)) or not math.isfinite(float(self.clearance)):
            raise ValueError("Route observation contains a non-finite scalar.")
        # ``clearance`` is already present in every preview point as
        # ``max_height``.  Keeping it out of the flat vector prevents a stale
        # duplicate feature when preview construction changes.
        return np.concatenate(
            (preview.reshape(-1), final_pose, np.asarray([self.remaining_time]), state)
        ).astype(np.float32)
