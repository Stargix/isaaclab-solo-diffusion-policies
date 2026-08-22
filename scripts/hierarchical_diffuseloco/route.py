"""Deterministic route geometry and seeded training banks."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import RouteObservation


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class PolylineRoute:
    """A route with a desired base-height profile along arc length.

    The field keeps its historical name so existing ``.npy`` route files remain
    loadable; new code accesses it through :attr:`target_height`.
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

    @property
    def target_height(self) -> np.ndarray:
        """Desired base-height profile (compatibility alias for old route files)."""
        return self.max_command_height

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
        """Load ``x,y[,yaw[,target_height]]`` and resample it by arc length."""
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


@dataclass(frozen=True)
class RouteBank:
    """Dense, equal-resolution routes that can be gathered by environment id."""

    xy: np.ndarray
    yaw: np.ndarray
    target_height: np.ndarray
    arc_length: np.ndarray
    length: np.ndarray
    desired_mean_speed: np.ndarray
    family: tuple[str, ...]

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value, dtype=np.float32) for value in (
            self.xy, self.yaw, self.target_height, self.arc_length,
            self.length, self.desired_mean_speed,
        ))
        xy, yaw, height, arc, length, speed = arrays
        count, points = xy.shape[:2]
        if xy.shape != (count, points, 2) or yaw.shape != (count, points):
            raise ValueError("RouteBank xy/yaw dimensions are inconsistent.")
        if height.shape != yaw.shape or arc.shape != yaw.shape:
            raise ValueError("RouteBank height/arc dimensions are inconsistent.")
        if length.shape != (count,) or speed.shape != (count,) or len(self.family) != count:
            raise ValueError("RouteBank scalar metadata dimensions are inconsistent.")
        if points < 2 or np.any(np.diff(arc, axis=1) <= 0.0):
            raise ValueError("Every route must contain strictly increasing arc samples.")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("RouteBank contains non-finite values.")
        for name, value in zip(
            ("xy", "yaw", "target_height", "arc_length", "length", "desired_mean_speed"), arrays
        ):
            object.__setattr__(self, name, value)

    @property
    def size(self) -> int:
        return int(self.xy.shape[0])


def _resample_xy_by_arc(raw_xy: np.ndarray, length: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    raw_arc = np.concatenate((
        np.zeros(1, dtype=np.float32),
        np.cumsum(np.linalg.norm(np.diff(raw_xy, axis=0), axis=1), dtype=np.float32),
    ))
    if raw_arc[-1] <= 1e-6:
        raise ValueError("Degenerate procedural route.")
    scale = length / float(raw_arc[-1])
    raw_xy = raw_xy * scale
    raw_arc = raw_arc * scale
    arc = np.linspace(0.0, length, points, dtype=np.float32)
    xy = np.stack([np.interp(arc, raw_arc, raw_xy[:, axis]) for axis in range(2)], axis=-1).astype(np.float32)
    delta = np.gradient(xy, axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)
    return xy, yaw


def _procedural_xy(family: str, length: float, points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if family == "straight":
        arc = np.linspace(0.0, length, points, dtype=np.float32)
        return np.stack((arc, np.zeros_like(arc)), axis=-1), np.zeros_like(arc)
    if family == "s_curve":
        u = np.linspace(0.0, 1.0, 513, dtype=np.float32)
        amplitude = float(rng.uniform(0.20, 0.38))
        raw = np.stack((u, amplitude * np.sin(2.0 * np.pi * u) * np.sin(np.pi * u)), axis=-1)
        return _resample_xy_by_arc(raw, length, points)
    if family == "right_angle":
        turn_sign = float(rng.choice((-1.0, 1.0)))
        radius = float(rng.uniform(0.28, 0.40))
        turn_length = 0.5 * np.pi * radius
        straight_length = 0.5 * (length - turn_length)
        if straight_length <= 0.20:
            raise ValueError("Route is too short for the requested right-angle radius.")
        arc = np.linspace(0.0, length, points, dtype=np.float32)
        xy = np.empty((points, 2), dtype=np.float32)
        yaw = np.empty(points, dtype=np.float32)
        before = arc <= straight_length
        during = (arc > straight_length) & (arc < straight_length + turn_length)
        after = ~(before | during)
        xy[before, 0], xy[before, 1], yaw[before] = arc[before], 0.0, 0.0
        angle = (arc[during] - straight_length) / radius
        xy[during, 0] = straight_length + radius * np.sin(angle)
        xy[during, 1] = turn_sign * radius * (1.0 - np.cos(angle))
        yaw[during] = turn_sign * angle
        tail = arc[after] - straight_length - turn_length
        xy[after, 0] = straight_length + radius
        xy[after, 1] = turn_sign * (radius + tail)
        yaw[after] = turn_sign * 0.5 * np.pi
        return xy, yaw
    raise ValueError(f"Unknown route family: {family!r}.")


def _height_profile(u: np.ndarray, mode: int, rng: np.random.Generator,
                    low: float, high: float) -> np.ndarray:
    """Interleave posture targets without filtering the policy command."""
    height = np.full_like(u, high, dtype=np.float32)
    if mode == 0:
        start = float(rng.uniform(0.28, 0.40))
        end = float(rng.uniform(0.60, 0.72))
        height[(u >= start) & (u <= end)] = low
    elif mode == 1:
        mid = 0.5 * (low + high)
        height[(u >= 0.20) & (u < 0.36)] = mid
        height[(u >= 0.36) & (u < 0.62)] = low
        height[(u >= 0.62) & (u < 0.78)] = mid
    else:
        height[(u >= 0.18) & (u < 0.34)] = low
        height[(u >= 0.52) & (u < 0.70)] = low
    return height


def build_route_bank(*, count: int = 192, points: int = 65, episode_duration_s: float = 8.0,
                     speed_range: tuple[float, float] = (0.30, 0.42), seed: int = 17,
                     families: tuple[str, ...] = ("straight", "s_curve", "right_angle"),
                     low_height: float = 0.1705, high_height: float = 0.2932) -> RouteBank:
    """Build the single, seeded training distribution for the first experiment."""
    if count < len(families) or points < 8 or episode_duration_s <= 0.0:
        raise ValueError("Route-bank size, resolution and duration must be positive and non-trivial.")
    if not 0.0 < speed_range[0] <= speed_range[1]:
        raise ValueError("speed_range must be positive and ordered.")
    rng = np.random.default_rng(seed)
    xy_all, yaw_all, height_all, arc_all = [], [], [], []
    lengths, speeds, names = [], [], []
    for index in range(count):
        family = families[index % len(families)]
        requested_speed = float(rng.uniform(*speed_range))
        requested_length = requested_speed * episode_duration_s
        xy, yaw = _procedural_xy(family, requested_length, points, rng)
        arc = np.concatenate((
            np.zeros(1, dtype=np.float32),
            np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1), dtype=np.float32),
        )).astype(np.float32)
        height = _height_profile(arc / arc[-1], index % 3, rng, low_height, high_height)
        xy_all.append(xy)
        yaw_all.append(yaw)
        height_all.append(height)
        arc_all.append(arc)
        lengths.append(float(arc[-1]))
        speeds.append(float(arc[-1]) / episode_duration_s)
        names.append(family)
    return RouteBank(
        np.stack(xy_all), np.stack(yaw_all), np.stack(height_all), np.stack(arc_all),
        np.asarray(lengths), np.asarray(speeds), tuple(names),
    )


def route_bank_from_polyline(route: PolylineRoute, *, episode_duration_s: float) -> RouteBank:
    """Adapt an explicit evaluation route to the same batched contract."""
    return RouteBank(
        route.xy[None], route.yaw[None], route.target_height[None], route.arc_length[None],
        np.asarray([route.length]), np.asarray([route.length / episode_duration_s]), ("file",),
    )
