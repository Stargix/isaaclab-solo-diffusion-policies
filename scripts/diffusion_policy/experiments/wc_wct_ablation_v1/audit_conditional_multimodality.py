#!/usr/bin/env python3
"""Audit whether Phase-A demonstrations are multimodal after conditioning.

The audit uses the exact SpatialHindsightDataset contract from training.  It
matches samples in normalized (proprio history, delayed action history, goal
history) space, excludes trivial temporal neighbours, and compares executable
future action chunks.  Skill labels are diagnostic only and are never added to
the conditioning vector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_DIFFUSION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DIFFUSION_ROOT))

from train.data.dataset import SpatialHindsightDataset  # noqa: E402


@dataclass(frozen=True)
class AuditConfig:
    history: int = 8
    prediction_horizon: int = 16
    execution_offset: int = 8
    goal_horizon_steps: int = 100
    temporal_exclusion_steps: int = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {}
    return {
        "mean": float(np.mean(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
    }


def _skill_for_sample(dataset: SpatialHindsightDataset, sample_index: int) -> str:
    sample = dataset.samples[int(sample_index)]
    skill_idx = dataset.demos[sample.demo_idx].skill_idx
    if skill_idx is None:
        raise ValueError("The multimodality audit requires diagnostic skill_idx labels.")
    unique = np.unique(skill_idx)
    if len(unique) != 1:
        raise ValueError(f"{dataset.demos[sample.demo_idx].demo_name}: mixed-skill episode.")
    return dataset.skill_names[int(unique[0])]


def _stratified_sample_indices(
    dataset: SpatialHindsightDataset, maximum_per_skill: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    # Reservoir sampling avoids materializing millions of Python integers.  The
    # demo label is cached: calling unique() on a full episode per sample would
    # turn this otherwise linear pass into a prohibitively expensive one.
    reservoirs: dict[str, list[int]] = {skill: [] for skill in dataset.skill_names}
    seen = {skill: 0 for skill in dataset.skill_names}
    demo_skills = []
    for demo in dataset.demos:
        if demo.skill_idx is None:
            raise ValueError("The multimodality audit requires diagnostic skill_idx labels.")
        unique = np.unique(demo.skill_idx)
        if len(unique) != 1:
            raise ValueError(f"{demo.demo_name}: mixed-skill episode.")
        demo_skills.append(dataset.skill_names[int(unique[0])])
    for index in range(len(dataset.samples)):
        skill = demo_skills[dataset.samples[index].demo_idx]
        seen[skill] += 1
        reservoir = reservoirs[skill]
        if len(reservoir) < maximum_per_skill:
            reservoir.append(index)
        else:
            replacement = int(rng.integers(seen[skill]))
            if replacement < maximum_per_skill:
                reservoir[replacement] = index
    return {skill: np.sort(np.asarray(values, dtype=np.int64)) for skill, values in reservoirs.items()}


def _extract_samples(
    dataset: SpatialHindsightDataset,
    selected: dict[str, np.ndarray],
    cfg: AuditConfig,
) -> dict[str, np.ndarray]:
    context, future, skills, demo_ids, anchors = [], [], [], [], []
    for skill_id, skill in enumerate(dataset.skill_names):
        for sample_index in selected[skill]:
            sample = dataset.samples[int(sample_index)]
            proprio, action_hist, goals, actions = dataset._raw_sample(sample)
            context.append(
                np.concatenate(
                    (proprio.numpy().reshape(-1), action_hist.numpy().reshape(-1), goals.numpy().reshape(-1))
                )
            )
            future.append(actions.numpy()[cfg.execution_offset :].reshape(-1))
            skills.append(skill_id)
            demo_ids.append(sample.demo_idx)
            anchors.append(sample.anchor_step)
    return {
        "context": np.asarray(context, dtype=np.float32),
        "future": np.asarray(future, dtype=np.float32),
        "skill": np.asarray(skills, dtype=np.int16),
        "demo": np.asarray(demo_ids, dtype=np.int32),
        "anchor": np.asarray(anchors, dtype=np.int32),
    }


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1.0e-4)
    return ((values - mean) / scale).astype(np.float32), mean, scale


def _projection(values: np.ndarray, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    dimensions = min(int(dimensions), values.shape[1])
    matrix = rng.normal(size=(values.shape[1], dimensions)).astype(np.float32)
    matrix /= np.sqrt(float(dimensions))
    projected = values @ matrix
    projected_scale = np.maximum(np.std(projected, axis=0), 1.0e-4)
    return projected / projected_scale


def _valid_candidate(
    query: int,
    candidate: int,
    demo: np.ndarray,
    anchor: np.ndarray,
    temporal_exclusion_steps: int,
) -> bool:
    if query == candidate:
        return False
    return not (
        demo[query] == demo[candidate]
        and abs(int(anchor[query]) - int(anchor[candidate])) <= temporal_exclusion_steps
    )


def _nearest_pairs(
    context: np.ndarray,
    projected: np.ndarray,
    skills: np.ndarray,
    demo: np.ndarray,
    anchor: np.ndarray,
    *,
    same_skill: bool,
    fixed_target_skill: int | None = None,
    temporal_exclusion_steps: int,
    candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    pair = np.full(len(context), -1, dtype=np.int64)
    distance = np.full(len(context), np.nan, dtype=np.float32)
    unique_skills = np.unique(skills)
    trees = {
        int(skill): (np.flatnonzero(skills == skill), cKDTree(projected[skills == skill]))
        for skill in unique_skills
    }
    for query in range(len(context)):
        target_skills = [int(skills[query])] if same_skill else [int(s) for s in unique_skills if s != skills[query]]
        if fixed_target_skill is not None:
            target_skills = [int(fixed_target_skill)] if int(fixed_target_skill) != int(skills[query]) else []
        best_candidate, best_distance = -1, np.inf
        for target_skill in target_skills:
            indices, tree = trees[target_skill]
            k = min(max(2, candidates), len(indices))
            _, local = tree.query(projected[query], k=k)
            for candidate in indices[np.atleast_1d(local)]:
                candidate = int(candidate)
                if not _valid_candidate(query, candidate, demo, anchor, temporal_exclusion_steps):
                    continue
                exact = float(np.sqrt(np.mean((context[query] - context[candidate]) ** 2)))
                if exact < best_distance:
                    best_candidate, best_distance = candidate, exact
        if best_candidate >= 0:
            pair[query] = best_candidate
            distance[query] = best_distance
    return pair, distance


def _future_distances(future: np.ndarray, pair: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.full(len(pair), np.nan, dtype=np.float32)
    raw = np.full(len(pair), np.nan, dtype=np.float32)
    valid = pair >= 0
    delta = future[valid] - future[pair[valid]]
    raw[valid] = np.sqrt(np.mean(delta**2, axis=1))
    normalized[valid] = np.sqrt(np.mean((delta / scale) ** 2, axis=1))
    return normalized, raw


def _block_distances(context: np.ndarray, pair: np.ndarray, cfg: AuditConfig) -> dict[str, np.ndarray]:
    valid = pair >= 0
    p_end = cfg.history * 30
    a_end = p_end + cfg.history * 12
    blocks = {"proprio": (0, p_end), "action_history": (p_end, a_end), "goal": (a_end, context.shape[1])}
    output: dict[str, np.ndarray] = {}
    for name, (start, end) in blocks.items():
        values = np.full(len(pair), np.nan, dtype=np.float32)
        delta = context[valid, start:end] - context[pair[valid], start:end]
        values[valid] = np.sqrt(np.mean(delta**2, axis=1))
        output[name] = values
    return output


def audit_dataset(path: Path, args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]], dict[str, np.ndarray]]:
    cfg = AuditConfig(temporal_exclusion_steps=args.temporal_exclusion_steps)
    dataset = SpatialHindsightDataset(
        [str(path)],
        history=cfg.history,
        prediction_horizon=cfg.prediction_horizon,
        execution_offset=cfg.execution_offset,
        goal_horizon_steps=cfg.goal_horizon_steps,
        goal_source="achieved",
        goal_representation="hindsight_geom_profile16",
        include_padded_starts=True,
        symmetry_mode="none",
    )
    rng = np.random.default_rng(args.seed)
    selected = _stratified_sample_indices(dataset, args.samples_per_skill, rng)
    arrays = _extract_samples(dataset, selected, cfg)
    normalized_context, _, _ = _standardize(arrays["context"])
    future_scale = np.maximum(np.std(arrays["future"], axis=0), 1.0e-4)
    projected = _projection(normalized_context, args.projection_dimensions, rng)

    same_pair, same_context = _nearest_pairs(
        normalized_context,
        projected,
        arrays["skill"],
        arrays["demo"],
        arrays["anchor"],
        same_skill=True,
        fixed_target_skill=None,
        temporal_exclusion_steps=cfg.temporal_exclusion_steps,
        candidates=args.candidate_neighbors,
    )
    cross_pair, cross_context = _nearest_pairs(
        normalized_context,
        projected,
        arrays["skill"],
        arrays["demo"],
        arrays["anchor"],
        same_skill=False,
        fixed_target_skill=None,
        temporal_exclusion_steps=cfg.temporal_exclusion_steps,
        candidates=args.candidate_neighbors,
    )
    same_action, same_action_raw = _future_distances(arrays["future"], same_pair, future_scale)
    cross_action, cross_action_raw = _future_distances(arrays["future"], cross_pair, future_scale)

    valid_same = np.isfinite(same_context) & np.isfinite(same_action)
    strict_context_gate = float(np.percentile(same_context[valid_same], 50))
    context_gate = float(np.percentile(same_context[valid_same], 95))
    action_gate = float(np.percentile(same_action[valid_same], 95))
    eligible = np.isfinite(cross_context) & (cross_context <= context_gate)
    strict_eligible = np.isfinite(cross_context) & (cross_context <= strict_context_gate)
    divergent = eligible & (cross_action > action_gate)
    strict_divergent = strict_eligible & (cross_action > action_gate)

    random_pair = np.full(len(arrays["skill"]), -1, dtype=np.int64)
    for index, skill in enumerate(arrays["skill"]):
        candidates = np.flatnonzero(arrays["skill"] != skill)
        random_pair[index] = int(rng.choice(candidates))
    random_context = np.sqrt(np.mean((normalized_context - normalized_context[random_pair]) ** 2, axis=1))
    random_action, random_action_raw = _future_distances(arrays["future"], random_pair, future_scale)

    skill_names = dataset.skill_names
    pair_rows: list[dict[str, object]] = []
    for source_id, source in enumerate(skill_names):
        source_mask = arrays["skill"] == source_id
        for target_id, target in enumerate(skill_names):
            if source_id == target_id:
                mask, pairs, context_distance, action_distance = source_mask, same_pair, same_context, same_action
                action_raw = same_action_raw
                kind = "within"
            else:
                pairs, context_distance = _nearest_pairs(
                    normalized_context,
                    projected,
                    arrays["skill"],
                    arrays["demo"],
                    arrays["anchor"],
                    same_skill=False,
                    fixed_target_skill=target_id,
                    temporal_exclusion_steps=cfg.temporal_exclusion_steps,
                    candidates=args.candidate_neighbors,
                )
                action_distance, action_raw = _future_distances(arrays["future"], pairs, future_scale)
                mask = source_mask & (pairs >= 0)
                kind = "cross"
            count = int(np.sum(mask & np.isfinite(context_distance)))
            if not count:
                continue
            pair_rows.append(
                {
                    "kind": kind,
                    "source_skill": source,
                    "target_skill": target,
                    "pairs": count,
                    "context_distance_mean": float(np.nanmean(context_distance[mask])),
                    "context_distance_p50": float(np.nanmedian(context_distance[mask])),
                    "future_action_normalized_mean": float(np.nanmean(action_distance[mask])),
                    "future_action_normalized_p50": float(np.nanmedian(action_distance[mask])),
                    "future_action_raw_rms_rad_mean": float(np.nanmean(action_raw[mask])),
                    "eligible_context_fraction": float(np.mean(context_distance[mask] <= context_gate)) if kind == "cross" else 1.0,
                    "strict_context_fraction": float(np.mean(context_distance[mask] <= strict_context_gate)) if kind == "cross" else 1.0,
                    "conditional_divergence_fraction": float(np.mean((context_distance[mask] <= context_gate) & (action_distance[mask] > action_gate))) if kind == "cross" else 0.0,
                    "strict_conditional_divergence_fraction": float(np.mean((context_distance[mask] <= strict_context_gate) & (action_distance[mask] > action_gate))) if kind == "cross" else 0.0,
                }
            )

    same_blocks = _block_distances(normalized_context, same_pair, cfg)
    cross_blocks = _block_distances(normalized_context, cross_pair, cfg)
    report: dict[str, object] = {
        "dataset": str(path.resolve()),
        "sha256": _sha256(path),
        "skills": skill_names,
        "samples_per_skill": {skill: int(len(selected[skill])) for skill in skill_names},
        "conditioning_dimension": int(normalized_context.shape[1]),
        "executable_future_dimension": int(arrays["future"].shape[1]),
        "matching": {
            "temporal_exclusion_steps": cfg.temporal_exclusion_steps,
            "projection_dimensions": args.projection_dimensions,
            "candidate_neighbors": args.candidate_neighbors,
            "exact_reranking": True,
        },
        "within_skill": {
            "context_distance": _percentiles(same_context),
            "future_action_normalized_rms": _percentiles(same_action),
            "future_action_raw_rms_rad": _percentiles(same_action_raw),
            "block_context_distance": {key: _percentiles(value) for key, value in same_blocks.items()},
        },
        "cross_skill": {
            "context_distance": _percentiles(cross_context),
            "future_action_normalized_rms": _percentiles(cross_action),
            "future_action_raw_rms_rad": _percentiles(cross_action_raw),
            "block_context_distance": {key: _percentiles(value) for key, value in cross_blocks.items()},
        },
        "random_cross_skill_control": {
            "context_distance": _percentiles(random_context),
            "future_action_normalized_rms": _percentiles(random_action),
            "future_action_raw_rms_rad": _percentiles(random_action_raw),
        },
        "data_driven_gates": {
            "strict_similar_context_within_skill_p50": strict_context_gate,
            "similar_context_within_skill_p95": context_gate,
            "divergent_action_within_skill_p95": action_gate,
        },
        "cross_skill_similar_context_fraction": float(np.mean(eligible)),
        "cross_skill_strict_similar_context_fraction": float(np.mean(strict_eligible)),
        "cross_skill_conditional_divergence_fraction": float(np.mean(divergent)),
        "conditional_divergence_given_similar_context": float(np.mean(divergent[eligible])) if np.any(eligible) else 0.0,
        "cross_skill_strict_conditional_divergence_fraction": float(np.mean(strict_divergent)),
        "conditional_divergence_given_strict_similar_context": float(np.mean(strict_divergent[strict_eligible])) if np.any(strict_eligible) else 0.0,
        "interpretation_note": (
            "Cross-skill divergence is evidence of conditional multimodality only when context overlap is non-trivial. "
            "Separated skills imply identifiable conditioning, not multiple valid actions for one condition."
        ),
    }
    diagnostics = {
        "skill": arrays["skill"],
        "same_context": same_context,
        "same_action": same_action,
        "cross_context": cross_context,
        "cross_action": cross_action,
        "eligible": eligible,
        "strict_eligible": strict_eligible,
        "divergent": divergent,
        "strict_divergent": strict_divergent,
        "random_context": random_context,
        "random_action": random_action,
    }
    return report, pair_rows, diagnostics


def _write_plots(results: list[tuple[str, dict[str, object], dict[str, np.ndarray]]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(results), 2, figsize=(12, 5 * len(results)), squeeze=False, constrained_layout=True)
    for row, (label, report, values) in enumerate(results):
        axes[row, 0].hist(values["same_context"], bins=60, density=True, alpha=0.6, label="within skill")
        axes[row, 0].hist(values["cross_context"], bins=60, density=True, alpha=0.6, label="cross skill")
        axes[row, 0].axvline(report["data_driven_gates"]["similar_context_within_skill_p95"], color="black", linestyle="--", label="similarity gate")
        axes[row, 0].set(title=f"{label}: conditioning overlap", xlabel="normalized context RMS", ylabel="density")
        axes[row, 0].legend()

        valid = np.isfinite(values["cross_context"]) & np.isfinite(values["cross_action"])
        axes[row, 1].scatter(values["cross_context"][valid], values["cross_action"][valid], s=4, alpha=0.12)
        axes[row, 1].axvline(report["data_driven_gates"]["similar_context_within_skill_p95"], color="black", linestyle="--")
        axes[row, 1].axhline(report["data_driven_gates"]["divergent_action_within_skill_p95"], color="red", linestyle="--")
        axes[row, 1].set(title=f"{label}: cross-skill conditional branching", xlabel="normalized context RMS", ylabel="normalized future-action RMS")
        axes[row, 1].grid(alpha=0.2)
    figure.savefig(output / "conditional_multimodality.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wc_dataset", required=True)
    parser.add_argument("--wct_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples_per_skill", type=int, default=5000)
    parser.add_argument("--projection_dimensions", type=int, default=64)
    parser.add_argument("--candidate_neighbors", type=int, default=48)
    parser.add_argument("--temporal_exclusion_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_skill < 100:
        raise ValueError("--samples_per_skill must be at least 100.")
    if args.projection_dimensions < 2 or args.candidate_neighbors < 2:
        raise ValueError("Projection and candidate-neighbour counts must be at least 2.")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_reports: dict[str, object] = {"schema": "conditional_multimodality_audit_v1", "seed": args.seed, "datasets": {}}
    rows: list[dict[str, object]] = []
    plot_values = []
    for label, raw_path in (("WC", args.wc_dataset), ("WCT", args.wct_dataset)):
        report, pair_rows, diagnostics = audit_dataset(Path(raw_path), args)
        all_reports["datasets"][label] = report
        rows.extend({"dataset_label": label, **row} for row in pair_rows)
        plot_values.append((label, report, diagnostics))
    (output / "conditional_multimodality.json").write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    if rows:
        with (output / "skill_pair_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _write_plots(plot_values, output)
    np.savez_compressed(
        output / "diagnostics.npz",
        **{f"{label.lower()}_{key}": value for label, _, diagnostic in plot_values for key, value in diagnostic.items()},
    )
    print(json.dumps({label: {
        "similar_context_fraction": report["cross_skill_similar_context_fraction"],
        "strict_similar_context_fraction": report["cross_skill_strict_similar_context_fraction"],
        "conditional_divergence_fraction": report["cross_skill_conditional_divergence_fraction"],
        "divergence_given_similar_context": report["conditional_divergence_given_similar_context"],
        "strict_divergence_fraction": report["cross_skill_strict_conditional_divergence_fraction"],
        "divergence_given_strict_similar_context": report["conditional_divergence_given_strict_similar_context"],
    } for label, report, _ in plot_values}, indent=2))


if __name__ == "__main__":
    main()
