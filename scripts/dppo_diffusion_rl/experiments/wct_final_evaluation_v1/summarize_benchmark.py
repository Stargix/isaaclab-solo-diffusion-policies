#!/usr/bin/env python3
"""Cluster-aware summary of the frozen WCT evaluation suites.

Episode conditions sharing ``(path_shape, repeat)`` are reduced to one route
observation before confidence intervals are computed.  This prevents speeds,
height directions and transition fractions from being counted as independent
geometry draws.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


METRICS: dict[str, tuple[str, Callable[[dict[str, str]], float]]] = {
    "task_success_rate": ("Task success", lambda row: _boolean(row["task_success"])),
    "survival_rate": ("Full-horizon survival", lambda row: _boolean(row["survived"])),
    "strict_arrival_rate": ("Strict arrival", lambda row: _boolean(row["route_arrived"])),
    "base_failure_rate": ("Base failure", lambda row: _boolean(row["base_failure"])),
    "active_cross_track_rmse_m": (
        "Active CTE RMSE [m]",
        lambda row: _number(row["active_cross_track_rmse_m"]),
    ),
    "active_cross_track_p95_m": (
        "Active CTE p95 [m]",
        lambda row: _number(row["active_cross_track_p95_m"]),
    ),
    "active_cross_track_max_m": (
        "Active CTE max [m]",
        lambda row: _number(row["active_cross_track_max_m"]),
    ),
    "active_height_mae_m": (
        "Active height MAE [m]",
        lambda row: _number(row["active_height_mae_m"]),
    ),
    "arrival_speed_abs_error_m_s": (
        "Arrival speed absolute error, arrivals only [m/s]",
        lambda row: abs(
            _number(row["route_arrival_mean_speed_m_s"]) - _number(row["requested_speed"])
        ),
    ),
}


def _boolean(value: str) -> float:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return 1.0
    if normalized in {"false", "0", "no"}:
        return 0.0
    return float("nan")


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def _load_suite(name: str, directory: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    csv_path = directory / "evaluation_summary.csv"
    json_path = directory / "evaluation_summary.json"
    if not csv_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Suite {name!r} is incomplete under {directory}.")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    for row in rows:
        row["suite"] = name
    return rows, metadata


def _suite_label(name: str) -> str:
    prefix = "OOD" if name.startswith("ood_") else "ID"
    body = name.removeprefix("ood_").removeprefix("id_")
    body = body.replace("crouch_to_walk", "C->W").replace("walk_to_crouch", "W->C")
    return prefix + "\n" + body.replace("_", " ")


def _plot_overview(
    rows: list[dict[str, object]], output_path: Path, title: str
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="white")
    plot_metrics = (
        "task_success_rate",
        "active_cross_track_p95_m",
        "active_height_mae_m",
        "arrival_speed_abs_error_m_s",
    )
    x = np.arange(len(rows))
    colors = ["#E67E22" if str(row["suite"]).startswith("ood_") else "#0052CC" for row in rows]
    for axis, metric in zip(axes.flat, plot_metrics):
        means = np.asarray([float(row[metric]) for row in rows])
        lower = np.asarray([float(row[f"{metric}_ci95_low"]) for row in rows])
        upper = np.asarray([float(row[f"{metric}_ci95_high"]) for row in rows])
        errors = np.vstack((means - lower, upper - means))
        axis.bar(x, means, color=colors, alpha=0.85)
        axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="#222222", capsize=3)
        rotation = 0 if len(rows) <= 4 else 12
        alignment = "center" if rotation == 0 else "right"
        axis.set_xticks(
            x,
            [_suite_label(str(row["suite"])) for row in rows],
            rotation=rotation,
            ha=alignment,
        )
        axis.set_title(METRICS[metric][0])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Named evaluator output; repeat for every completed suite.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=916)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise ValueError("--bootstrap_draws must be at least 100.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    suite_metadata: dict[str, dict[str, object]] = {}
    for specification in args.suite:
        if "=" not in specification:
            raise ValueError(f"Expected NAME=DIR, got {specification!r}.")
        name, raw_directory = specification.split("=", 1)
        rows, metadata = _load_suite(name, Path(raw_directory).resolve())
        all_rows.extend(rows)
        suite_metadata[name] = metadata

    route_rows: list[dict[str, object]] = []
    route_groups: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in all_rows:
        key = (row["suite"], row["path_shape"], int(row["repeat"]))
        route_groups.setdefault(key, []).append(row)
    for (suite, family, repeat), conditions in sorted(route_groups.items()):
        output: dict[str, object] = {
            "suite": suite,
            "path_shape": family,
            "repeat": repeat,
            "conditions": len(conditions),
        }
        for metric, (_, extractor) in METRICS.items():
            values = np.asarray([extractor(row) for row in conditions], dtype=np.float64)
            finite = values[np.isfinite(values)]
            output[metric] = float(np.mean(finite)) if finite.size else float("nan")
        route_rows.append(output)

    rng = np.random.default_rng(args.seed)
    aggregate_rows: list[dict[str, object]] = []
    aggregate_groups = sorted({(str(row["suite"]), str(row["path_shape"])) for row in route_rows})
    aggregate_groups += sorted({(str(row["suite"]), "__all__") for row in route_rows})
    for suite, family in aggregate_groups:
        selected = [
            row for row in route_rows
            if row["suite"] == suite and (family == "__all__" or row["path_shape"] == family)
        ]
        aggregate: dict[str, object] = {
            "suite": suite,
            "path_shape": family,
            "independent_routes": len(selected),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            lower, upper = _bootstrap_ci(finite, rng, args.bootstrap_draws)
            aggregate[metric] = float(np.mean(finite)) if finite.size else float("nan")
            aggregate[f"{metric}_ci95_low"] = lower
            aggregate[f"{metric}_ci95_high"] = upper
        aggregate_rows.append(aggregate)

    with (output_dir / "route_level_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(route_rows[0]))
        writer.writeheader()
        writer.writerows(route_rows)
    with (output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    report = {
        "protocol": "wct_final_evaluation_v1",
        "statistical_unit": "one materialized (path_shape, repeat) geometry",
        "bootstrap_draws": args.bootstrap_draws,
        "suite_metadata": suite_metadata,
        "route_level_count": len(route_rows),
        "aggregates": aggregate_rows,
        "interpretation": (
            "Conditions on the same route are averaged before bootstrap resampling. "
            "ID, geometric OOD and fast boundary suites must not be pooled into one success rate."
        ),
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")

    order = {
        "id_constant": 0,
        "id_transition": 1,
        "ood_constant": 2,
        "ood_transition": 3,
        "fast_crouch_to_walk": 4,
        "fast_walk_to_crouch": 5,
        "ood_fast_crouch_to_walk": 6,
        "ood_fast_walk_to_crouch": 7,
    }
    overall = sorted(
        [row for row in aggregate_rows if row["path_shape"] == "__all__"],
        key=lambda row: order.get(str(row["suite"]), 999),
    )
    navigation = [row for row in overall if "fast" not in str(row["suite"])]
    fast = [row for row in overall if "fast" in str(row["suite"])]
    _plot_overview(
        overall,
        output_dir / "benchmark_overview.png",
        "WCT benchmark - route-clustered 95% bootstrap intervals",
    )
    _plot_overview(
        navigation,
        output_dir / "benchmark_overview_navigation.png",
        "Navigation and posture composition (ID and OOD reported separately)",
    )
    _plot_overview(
        fast,
        output_dir / "benchmark_overview_fast.png",
        "Fast-section capability and geometric OOD stress tests",
    )
    print(json.dumps({"routes": len(route_rows), "aggregates": len(aggregate_rows)}, indent=2))


if __name__ == "__main__":
    main()
