"""Deterministic route geometry and clearance profiles for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import RouteObservation


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class PolylineRoute:
    """A route sampled at fixed arc-length spacing.

    ``required_height`` is an environmental clearance profile, not an action target.  It
    makes crouching necessary in a known section rather than rewarding crouch everywhere.
    """

    xy: np.ndarray
    yaw: np.ndarray
    required_height: np.ndarray
    spacing: float = 0.10

    def __post_init__(self) -> None:
        xy, yaw, height = map(lambda x: np.asarray(x, dtype=np.float32), (self.xy, self.yaw, self.required_height))
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("xy must have shape [N,2] with N>=2.")
        if yaw.shape != (len(xy),) or height.shape != (len(xy),):
            raise ValueError("yaw and required_height must have one value per route point.")
        if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(yaw)) or not np.all(np.isfinite(height)):
            raise ValueError("Route contains non-finite values.")
        if np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1) < 1e-5):
            raise ValueError("Adjacent route points must be distinct.")
        if self.spacing <= 0.0:
            raise ValueError("spacing must be positive.")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "yaw", yaw)
        object.__setattr__(self, "required_height", height)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())

    def observation(self, position_xy: np.ndarray, yaw: float, *, preview_points: int = 8,
                    remaining_time: float = 5.0, state: np.ndarray | None = None) -> RouteObservation:
        position_xy = np.asarray(position_xy, dtype=np.float32)
        if position_xy.shape != (2,):
            raise ValueError("position_xy must have shape [2].")
        nearest = int(np.argmin(np.sum((self.xy - position_xy) ** 2, axis=1)))
        indices = np.minimum(nearest + np.arange(1, preview_points + 1), len(self.xy) - 1)
        delta = self.xy[indices] - position_xy[None, :]
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), axis=-1)
        local_yaw = wrap_angle(self.yaw[indices] - yaw).astype(np.float32)
        preview = np.concatenate((local, local_yaw[:, None], self.required_height[indices, None]), axis=1)
        final_delta = self.xy[-1] - position_xy
        final_pose = np.asarray((c * final_delta[0] + s * final_delta[1], -s * final_delta[0] + c * final_delta[1],
                                 wrap_angle(self.yaw[-1] - yaw)), dtype=np.float32)
        if state is None:
            state = np.zeros(0, dtype=np.float32)
        return RouteObservation(preview, final_pose, float(remaining_time), np.asarray(state),
                                float(self.required_height[indices[0]]))

    @classmethod
    def from_file(cls, path: str, *, spacing: float = 0.10, default_height: float = 0.2932) -> "PolylineRoute":
        """Load an ``.npy`` route with columns ``x,y[,yaw[,required_height]]``."""
        values = np.asarray(np.load(path), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] not in (2, 3, 4):
            raise ValueError("Route file must have shape [N,2], [N,3] or [N,4].")
        xy = values[:, :2]
        delta = np.diff(xy, axis=0, append=xy[-1:])
        derived_yaw = np.arctan2(delta[:, 1], delta[:, 0])
        yaw = values[:, 2] if values.shape[1] >= 3 else derived_yaw
        height = values[:, 3] if values.shape[1] == 4 else np.full(len(xy), default_height, dtype=np.float32)
        return cls(xy, yaw, height, spacing)


def clearance_route(length: float = 5.0, spacing: float = 0.10, *, low_height: float = 0.17,
                    high_height: float = 0.29) -> PolylineRoute:
    """Straight benchmark route with a central low-clearance section."""

    if length <= 1.0:
        raise ValueError("length must exceed 1m for the clearance benchmark.")
    x = np.arange(0.0, length + 0.5 * spacing, spacing, dtype=np.float32)
    xy = np.stack((x, np.zeros_like(x)), axis=1)
    h = np.full_like(x, high_height)
    h[(x >= 0.40 * length) & (x <= 0.60 * length)] = low_height
    return PolylineRoute(xy, np.zeros_like(x), h, spacing)
