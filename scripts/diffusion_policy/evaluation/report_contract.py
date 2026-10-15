"""Versioned reporting diagnostics; never replace frozen task_success fields.

The adherence gate below is time-weighted, unlike the training distance gate.
Spatial reconstruction on saved XY traces is also not an exact train replay.
"""
from __future__ import annotations

import math
import numpy as np

REPORT_CONTRACT_VERSION = "active_report_v1"
CTE_TOLERANCE_M = 0.10


def project_saved_trace(positions, path):
    """Vectorized distances with the historical projection's same local cursor.

    Equivalent search_back=5/search_forward=50; do not use a global nearest
    branch shortcut on folded routes. Tests compare against the original.
    """
    xy=np.asarray(positions,dtype=float); path=np.asarray(path,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not len(xy) or path.ndim!=2 or path.shape[1]!=2 or len(path)<2:
        raise ValueError('Require (T,2) positions and (N,2) path')
    if not np.isfinite(xy).all() or not np.isfinite(path).all(): raise ValueError('Nonfinite XY')
    segments=np.diff(path,axis=0); lengths=np.linalg.norm(segments,axis=1)
    if np.any(lengths<=1e-9): raise ValueError('Zero-length segment')
    arc=np.r_[0.,np.cumsum(lengths)]
    fraction=np.clip(np.sum((xy[:,None]-path[:-1])*segments,axis=2)/lengths**2,0,1)
    distance_sq=np.sum((xy[:,None]-(path[:-1]+fraction[:,:,None]*segments))**2,axis=2)
    cursor=0; best=np.empty(len(xy),dtype=int)
    for i in range(len(xy)):
        low=max(0,cursor-5); high=min(len(segments),cursor+51)
        best[i]=low+np.argmin(distance_sq[i,low:high]); cursor=max(cursor,best[i])
    indices=np.arange(len(xy))
    return ((arc[best]+fraction[indices,best]*lengths[best]).astype(np.float32),
            np.sqrt(distance_sq[indices,best]).astype(np.float32))


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}


def number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def classify_outcome(row: dict) -> dict:
    """Safety ends at first positional arrival; nonarrivals use full horizon.

    A failed arrival remains a failed task even if its active safety is valid.
    Equal failure/arrival times are conservatively classified as prearrival.
    """
    arrived = truth(row.get("task_arrived"))
    success = truth(row.get("task_success"))
    full_safe = truth(row.get("survived"))
    arrival = number(row.get("task_success_time_s"))
    failure = number(row.get("time_to_failure_s"))
    if arrived and not (math.isfinite(arrival) and arrival > 0):
        raise ValueError("An arrival requires a finite positive arrival time")
    if success and not arrived:
        raise ValueError("Success without first arrival")
    initialization = truth(row.get('initialization_failure',False))
    if not full_safe and not (math.isfinite(failure) and (failure > 0 or (initialization and failure == 0))):
        raise ValueError("A physical failure requires a valid failure time")
    post = bool(arrived and not full_safe and failure > arrival)
    active_safe = bool(full_safe or post)
    if success and not active_safe:
        raise ValueError("Archived success precedes neither physical failure nor reset")
    cte = number(row.get("active_cross_track_rmse_m"))
    adherence = math.isfinite(cte) and 0 <= cte <= CTE_TOLERANCE_M
    return dict(
        archived_success=success, task_arrived=arrived,
        active_survival=active_safe, full_horizon_survival=full_safe,
        prearrival_physical_failure=not active_safe,
        postarrival_physical_failure=post,
        active_temporal_cte_rmse_m=cte,
        adherence_success=bool(success and adherence),
    )


def spatial_tracking(progress, cte, height, required, boundaries, *, margin_m=0.25):
    """First-passage spatial weights reconstructed from continuous projection.

    progress is relative to the active rollout origin; boundary positions are
    in that same frame. Zero travelled distance yields unavailable, not zero.
    Margin coverage is exposed explicitly, including on failed partial routes.
    """
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1)
              for v in (progress, cte, height, required)]
    if not arrays[0].size or len({a.size for a in arrays}) != 1:
        raise ValueError("Require matching nonempty trace arrays")
    if not all(np.isfinite(a).all() for a in arrays) or margin_m < 0:
        raise ValueError("Nonfinite trace or invalid margin")
    p, error, h, target = arrays
    p = np.maximum.accumulate(np.maximum(p, 0))
    ds = np.diff(np.r_[0., p])
    boundaries = np.asarray(boundaries, dtype=np.float64).reshape(-1)
    if not np.isfinite(boundaries).all():
        raise ValueError("Nonfinite boundary")
    keep = np.ones(len(p), dtype=bool)
    if len(boundaries) and margin_m > 0:
        keep = np.min(np.abs(p[:, None] - boundaries), axis=1) >= margin_m
    scored = ds * keep
    total, scored_total = float(ds.sum()), float(scored.sum())
    return dict(
        travelled_progress_m=total,
        spatial_cte_rmse_m=float(np.sqrt(np.sum(error**2 * ds) / total)) if total > 1e-6 else math.nan,
        spatial_height_mae_m=float(np.sum(np.abs(h-target) * scored) / scored_total) if scored_total > 1e-6 else math.nan,
        height_scored_distance_m=scored_total,
        height_scored_fraction=scored_total / total if total > 1e-6 else math.nan,
    )


def paired_route_bootstrap(differences: dict[str, np.ndarray], *, seed=20261001, draws=4000):
    """Stratify geometry clusters by family; actors and conditions are fixed.

    Each array contains one mean paired difference per independent geometry.
    Intervals condition on the observed actors, not population train variance.
    """
    groups = [np.asarray(v, dtype=float) for _, v in sorted(differences.items())]
    if not groups or any(g.ndim != 1 or not g.size or not np.isfinite(g).all() for g in groups):
        raise ValueError("Need finite nonempty route differences in each stratum")
    rng = np.random.default_rng(seed)
    total = sum(len(g) for g in groups)
    sampled = np.zeros(draws)
    for g in groups:
        sampled += g[rng.integers(0, len(g), size=(draws, len(g)))].sum(axis=1) / total
    return dict(mean=float(sum(g.sum() for g in groups) / total),
                ci_low=float(np.quantile(sampled, .025)),
                ci_high=float(np.quantile(sampled, .975)), independent_routes=total)
