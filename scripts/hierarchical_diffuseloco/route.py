"""Deterministic route geometry and clearance profiles for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import RouteObservation


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class PolylineRoute:
    """A route with an explicit maximum allowed command height along arc length.

    ``max_command_height`` is a clearance proxy in the coordinate system used by
    DiffuseLoco's height command. A lower value means a lower ceiling.
    """

    xy: np.ndarray
    yaw: np.ndarray
    max_command_height: np.ndarray
    spacing: float = 0.10

    def __post_init__(self) -> None:
        xy, yaw, height = map(lambda x: np.asarray(x, dtype=np.float32), (self.xy, self.yaw, self.max_command_height))
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("xy must have shape [N,2] with N>=2.")
        if yaw.shape != (len(xy),) or height.shape != (len(xy),):
            raise ValueError("yaw and max_command_height must have one value per route point.")
        if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(yaw)) or not np.all(np.isfinite(height)):
            raise ValueError("Route contains non-finite values.")
        if np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1) < 1e-5):
            raise ValueError("Adjacent route points must be distinct.")
        if self.spacing <= 0.0:
            raise ValueError("spacing must be positive.")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "yaw", yaw)
        object.__setattr__(self, "max_command_height", height)
        arc_length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))).astype(np.float32)
        object.__setattr__(self, "arc_length", arc_length)

    @property
    def length(self) -> float:
        return float(self.arc_length[-1])

    def project(self, position_xy: np.ndarray) -> tuple[float, float, int]:
        """Return continuous arc progress, signed cross-track error and segment index."""
        position = np.asarray(position_xy, dtype=np.float32)
        starts = self.xy[:-1]
        segments = self.xy[1:] - starts
        lengths = np.linalg.norm(segments, axis=1)
        fractions = np.clip(np.sum((position[None] - starts) * segments, axis=1) / np.square(lengths), 0.0, 1.0)
        projections = starts + fractions[:, None] * segments
        index = int(np.argmin(np.sum(np.square(position[None] - projections), axis=1)))
        signed_cross_track = float(
            (segments[index, 0] * (position[1] - projections[index, 1])
             - segments[index, 1] * (position[0] - projections[index, 0])) / lengths[index]
        )
        return float(self.arc_length[index] + fractions[index] * lengths[index]), signed_cross_track, index

    def sample(self, arc_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate route values at arbitrary arc-length positions."""
        requested = np.asarray(arc_positions, dtype=np.float32)
        valid = requested <= self.length
        s = np.clip(requested, 0.0, self.length)
        xy = np.stack([np.interp(s, self.arc_length, self.xy[:, axis]) for axis in range(2)], axis=-1)
        yaw = np.interp(s, self.arc_length, np.unwrap(self.yaw))
        height = np.interp(s, self.arc_length, self.max_command_height)
        return xy.astype(np.float32), wrap_angle(yaw).astype(np.float32), height.astype(np.float32), valid.astype(np.float32)

    def observation(self, position_xy: np.ndarray, yaw: float, *, preview_points: int = 8,
                    remaining_time: float = 5.0, state: np.ndarray | None = None) -> RouteObservation:
        position_xy = np.asarray(position_xy, dtype=np.float32)
        if position_xy.shape != (2,):
            raise ValueError("position_xy must have shape [2].")
        progress, _, _ = self.project(position_xy)
        preview_s = progress + np.linspace(0.2, 2.0, preview_points, dtype=np.float32)
        target_xy, target_yaw, max_height, valid = self.sample(preview_s)
        delta = target_xy - position_xy[None, :]
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), axis=-1)
        local_yaw = wrap_angle(target_yaw - yaw).astype(np.float32)
        preview = np.concatenate((local, np.sin(local_yaw)[:, None], np.cos(local_yaw)[:, None],
                                  max_height[:, None], valid[:, None]), axis=1)
        final_delta = self.xy[-1] - position_xy
        final_yaw = float(wrap_angle(self.yaw[-1] - yaw))
        final_pose = np.asarray((c * final_delta[0] + s * final_delta[1], -s * final_delta[0] + c * final_delta[1],
                                 np.sin(final_yaw), np.cos(final_yaw)), dtype=np.float32)
        if state is None:
            state = np.zeros(0, dtype=np.float32)
        return RouteObservation(preview, final_pose, float(remaining_time), np.asarray(state),
                                float(self.sample(np.asarray([progress], dtype=np.float32))[2][0]))

    @classmethod
    def from_file(cls, path: str, *, spacing: float = 0.10, default_height: float = 0.2932) -> "PolylineRoute":
        """Load ``x,y[,yaw[,max_command_height]]`` and resample it by arc length."""
        values = np.asarray(np.load(path), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] not in (2, 3, 4):
            raise ValueError("Route file must have shape [N,2], [N,3] or [N,4].")
        xy = values[:, :2]
        delta = np.diff(xy, axis=0, append=(xy[-1:] - xy[-2:-1]))
        derived_yaw = np.arctan2(delta[:, 1], delta[:, 0])
        yaw = values[:, 2] if values.shape[1] >= 3 else derived_yaw
        height = values[:, 3] if values.shape[1] == 4 else np.full(len(xy), default_height, dtype=np.float32)
        raw = cls(xy, yaw, height, spacing)
        samples = np.arange(0.0, raw.length, spacing, dtype=np.float32)
        samples = np.append(samples, raw.length).astype(np.float32)
        sampled_xy, sampled_yaw, sampled_height, _ = raw.sample(samples)
        return cls(sampled_xy, sampled_yaw, sampled_height, spacing)


def clearance_route(length: float = 5.0, spacing: float = 0.10, *, low_height: float = 0.1705,
                    high_height: float = 0.2932) -> PolylineRoute:
    """Straight benchmark route with a central low-ceiling section."""

    if length <= 1.0:
        raise ValueError("length must exceed 1m for the clearance benchmark.")
    x = np.arange(0.0, length + 0.5 * spacing, spacing, dtype=np.float32)
    xy = np.stack((x, np.zeros_like(x)), axis=1)
    h = np.full_like(x, high_height)
    h[(x >= 0.40 * length) & (x <= 0.60 * length)] = low_height
    return PolylineRoute(xy, np.zeros_like(x), h, spacing)
