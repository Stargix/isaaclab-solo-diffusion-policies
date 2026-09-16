"""Versioned, materialized route banks for paired policy evaluation.

The benchmark unit is a geometry draw, not an episode condition.  A route in
this module is stored once in robot-local coordinates and can then be replayed
with several speeds, height profiles and checkpoints without regenerating its
geometry.  The compressed format contains only numeric/string arrays and does
not require pickle when loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from scripts.dppo_diffusion_rl.hybrid_routes import (
    sample_coherent_smooth_geometry,
    sample_waypoint_geometry,
)


ROUTE_BANK_SCHEMA_VERSION = 1
ID_FAMILIES = (
    "procedural",
    "coherent_smooth",
    "rounded_waypoint",
    "hard_waypoint",
)
OOD_FAMILIES = ("ood_arc", "ood_s_curve", "ood_corner")


@dataclass(frozen=True)
class MaterializedRoute:
    """One fixed-length route in a frame whose start pose is ``(0, 0, 0)``."""

    family: str
    repeat: int
    seed: int
    split: str
    xy: np.ndarray
    yaw: np.ndarray

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy)
        yaw = np.asarray(self.yaw)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("Route XY must have shape (N, 2), N >= 2.")
        if yaw.shape != (len(xy),):
            raise ValueError("Route yaw must have one value per XY point.")
        if not np.isfinite(xy).all() or not np.isfinite(yaw).all():
            raise ValueError("Route coordinates must be finite.")
        if np.linalg.norm(xy[0]) > 1.0e-6 or abs(float(yaw[0])) > 1.0e-5:
            raise ValueError("Materialized routes must start at local pose (0, 0, 0).")
        segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if np.any(segment_lengths <= 1.0e-7):
            raise ValueError("Route contains a zero-length segment.")

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integrate_curvature(curvature: np.ndarray, length_m: float) -> tuple[np.ndarray, np.ndarray]:
    curvature = np.asarray(curvature, dtype=np.float64).reshape(-1)
    if curvature.size < 2:
        raise ValueError("Curvature profile must contain at least two intervals.")
    ds = float(length_m) / float(curvature.size)
    interval_yaw = np.concatenate(([0.0], np.cumsum(curvature * ds)))
    increments = ds * np.stack(
        (np.cos(interval_yaw[:-1]), np.sin(interval_yaw[:-1])), axis=1
    )
    xy = np.vstack((np.zeros((1, 2)), np.cumsum(increments, axis=0)))
    yaw = interval_yaw
    yaw[-1] = yaw[-2]
    return xy.astype(np.float32), yaw.astype(np.float32)


def _procedural_route(seed: int, *, points: int, length_m: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    intervals = points - 1
    interval_arc = np.linspace(0.0, length_m, intervals, endpoint=False)
    knot_arc = np.linspace(0.0, length_m, 6)
    amplitude = float(rng.uniform(0.0, 0.8))
    knot_values = rng.uniform(-amplitude, amplitude, size=6)
    curvature = np.interp(interval_arc, knot_arc, knot_values)
    return _integrate_curvature(curvature, length_m)


def _hybrid_route(
    family: str, seed: int, *, points: int, length_m: float
) -> tuple[np.ndarray, np.ndarray]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if family == "coherent_smooth":
            geometry = sample_coherent_smooth_geometry(
                1, "cpu", points=points, length_m=length_m
            )
        elif family == "rounded_waypoint":
            geometry = sample_waypoint_geometry(
                1, "cpu", rounded=True, points=points, length_m=length_m
            )
        elif family == "hard_waypoint":
            geometry = sample_waypoint_geometry(
                1, "cpu", rounded=False, points=points, length_m=length_m
            )
        else:
            raise ValueError(f"Unknown hybrid family: {family}")
    return (
        geometry.xy[0].cpu().numpy().astype(np.float32),
        geometry.yaw[0].cpu().numpy().astype(np.float32),
    )


def _ood_arc(seed: int, *, points: int, length_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-curvature stress beyond the train heading/curvature envelope."""

    rng = np.random.default_rng(seed)
    magnitude = float(rng.uniform(0.58, 0.74))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return _integrate_curvature(np.full(points - 1, sign * magnitude), length_m)


def _ood_s_curve(seed: int, *, points: int, length_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Higher-frequency/higher-curvature S family than the hybrid train bank."""

    rng = np.random.default_rng(seed)
    peak = float(rng.uniform(0.85, 1.10))
    cycles = float(rng.uniform(1.6, 2.4))
    phase = float(rng.uniform(-np.pi, np.pi))
    midpoint_arc = (np.arange(points - 1, dtype=np.float64) + 0.5) * length_m / (points - 1)
    curvature = peak * np.sin(2.0 * np.pi * cycles * midpoint_arc / length_m + phase)
    return _integrate_curvature(curvature, length_m)


def _sample_polyline(points_xy: np.ndarray, *, points: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(points_xy, dtype=np.float64)
    segments = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    total_length = float(lengths.sum())
    samples_xy: list[np.ndarray] = []
    samples_yaw: list[float] = []
    for index, (start, segment, length) in enumerate(zip(vertices[:-1], segments, lengths)):
        intervals = max(1, int(round((points - 1) * float(length) / total_length)))
        tangent_yaw = float(np.arctan2(segment[1], segment[0]))
        for fraction in np.arange(intervals, dtype=np.float64) / intervals:
            # Keep every hard vertex exactly.  Uniform global resampling would
            # cut across a corner and shorten the supposedly 4 m reference.
            samples_xy.append(start + fraction * segment)
            samples_yaw.append(0.0 if not samples_yaw else tangent_yaw)
    samples_xy.append(vertices[-1])
    samples_yaw.append(samples_yaw[-1])
    return np.asarray(samples_xy, dtype=np.float32), np.asarray(samples_yaw, dtype=np.float32)


def _ood_corner(seed: int, *, points: int, length_m: float) -> tuple[np.ndarray, np.ndarray]:
    """One continuously randomized 90--120 degree hard corner."""

    rng = np.random.default_rng(seed)
    first_length = float(rng.uniform(0.35, 0.65) * length_m)
    turn = np.deg2rad(float(rng.uniform(90.0, 120.0)))
    if rng.random() < 0.5:
        turn *= -1.0
    remaining = length_m - first_length
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [first_length, 0.0],
            [first_length + remaining * np.cos(turn), remaining * np.sin(turn)],
        ],
        dtype=np.float64,
    )
    return _sample_polyline(vertices, points=points)


def generate_route(
    family: str,
    *,
    repeat: int,
    seed: int,
    split: str,
    points: int = 101,
    length_m: float = 4.0,
) -> MaterializedRoute:
    """Generate one deterministic route from a declared benchmark family."""

    if points < 16 or length_m <= 0.0:
        raise ValueError("Require points >= 16 and length_m > 0.")
    if family == "procedural":
        xy, yaw = _procedural_route(seed, points=points, length_m=length_m)
    elif family in ID_FAMILIES[1:]:
        xy, yaw = _hybrid_route(family, seed, points=points, length_m=length_m)
    elif family == "ood_arc":
        xy, yaw = _ood_arc(seed, points=points, length_m=length_m)
    elif family == "ood_s_curve":
        xy, yaw = _ood_s_curve(seed, points=points, length_m=length_m)
    elif family == "ood_corner":
        xy, yaw = _ood_corner(seed, points=points, length_m=length_m)
    else:
        raise ValueError(f"Unknown route-bank family: {family}")
    return MaterializedRoute(family, int(repeat), int(seed), split, xy, yaw)


def generate_bank(
    families: Iterable[str],
    *,
    routes_per_family: int,
    base_seed: int,
    split: str,
    points: int = 101,
    length_m: float = 4.0,
) -> list[MaterializedRoute]:
    """Generate independent, deterministically indexed geometry draws."""

    families = tuple(families)
    if routes_per_family < 1 or not families:
        raise ValueError("The bank needs at least one family and one route per family.")
    routes: list[MaterializedRoute] = []
    for family_index, family in enumerate(families):
        for repeat in range(routes_per_family):
            seed = int(base_seed + family_index * 100_003 + repeat)
            routes.append(
                generate_route(
                    family,
                    repeat=repeat,
                    seed=seed,
                    split=split,
                    points=points,
                    length_m=length_m,
                )
            )
    return routes


def save_route_bank(path: Path, routes: Iterable[MaterializedRoute]) -> dict[str, object]:
    """Write a compressed, pickle-free bank and return its manifest entry."""

    path = Path(path)
    route_list = list(routes)
    if not route_list:
        raise ValueError("Cannot save an empty route bank.")
    keys = [(route.family, route.repeat) for route in route_list]
    if len(keys) != len(set(keys)):
        raise ValueError("Route-bank (family, repeat) keys must be unique.")
    offsets = [0]
    for route in route_list:
        offsets.append(offsets[-1] + len(route.xy))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray([ROUTE_BANK_SCHEMA_VERSION], dtype=np.int32),
        xy=np.concatenate([route.xy for route in route_list]).astype(np.float32),
        yaw=np.concatenate([route.yaw for route in route_list]).astype(np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        family=np.asarray([route.family for route in route_list], dtype="U32"),
        repeat=np.asarray([route.repeat for route in route_list], dtype=np.int32),
        seed=np.asarray([route.seed for route in route_list], dtype=np.int64),
        split=np.asarray([route.split for route in route_list], dtype="U16"),
    )
    families = sorted({route.family for route in route_list})
    counts = {family: sum(route.family == family for route in route_list) for family in families}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": ROUTE_BANK_SCHEMA_VERSION,
        "routes": len(route_list),
        "families": counts,
        "splits": sorted({route.split for route in route_list}),
        "length_m_min": min(route.length_m for route in route_list),
        "length_m_max": max(route.length_m for route in route_list),
    }


def load_route_bank(path: Path) -> dict[tuple[str, int], MaterializedRoute]:
    """Load and validate a materialized bank without enabling object arrays."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        version = int(np.asarray(data["schema_version"]).reshape(-1)[0])
        if version != ROUTE_BANK_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported route-bank schema {version}; expected {ROUTE_BANK_SCHEMA_VERSION}."
            )
        xy = np.asarray(data["xy"], dtype=np.float32)
        yaw = np.asarray(data["yaw"], dtype=np.float32)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        family = np.asarray(data["family"]).astype(str)
        repeat = np.asarray(data["repeat"], dtype=np.int32)
        seed = np.asarray(data["seed"], dtype=np.int64)
        split = np.asarray(data["split"]).astype(str)
    count = len(family)
    if not (
        len(repeat) == len(seed) == len(split) == count
        and offsets.shape == (count + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(xy) == len(yaw)
        and np.all(np.diff(offsets) >= 2)
    ):
        raise ValueError("Route-bank arrays are inconsistent.")
    routes: dict[tuple[str, int], MaterializedRoute] = {}
    for index in range(count):
        lower, upper = int(offsets[index]), int(offsets[index + 1])
        route = MaterializedRoute(
            str(family[index]),
            int(repeat[index]),
            int(seed[index]),
            str(split[index]),
            xy[lower:upper].copy(),
            yaw[lower:upper].copy(),
        )
        key = (route.family, route.repeat)
        if key in routes:
            raise ValueError(f"Duplicate route-bank key {key}.")
        routes[key] = route
    return routes


def align_route_to_spawn(
    route: MaterializedRoute,
    start_pos: np.ndarray,
    start_yaw: float,
    *,
    z: float = 0.2932,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly align a local route to one Isaac environment spawn pose."""

    rotation = np.asarray(
        [[np.cos(start_yaw), -np.sin(start_yaw)], [np.sin(start_yaw), np.cos(start_yaw)]],
        dtype=np.float32,
    )
    world_xy = np.asarray(start_pos[:2], dtype=np.float32) + route.xy @ rotation.T
    path = np.column_stack((world_xy, np.full(len(world_xy), z, dtype=np.float32)))
    return path.astype(np.float32), (route.yaw + float(start_yaw)).astype(np.float32)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
