"""Paired route-level summary for evaluation-v2 pace interventions.

The same materialized routes are evaluated under dynamic, frozen and shuffled
pace conditioning.  Repeated speed, height and direction conditions are first
averaged within a route; bootstrap intervals then resample routes, stratified
by route family.  These intervals describe route uncertainty, not training-seed
uncertainty.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "survival_rate": ("survived", 1.0, "Survival rate"),
    "task_success_rate": ("task_success", 1.0, "Task success rate"),
    "cte_p95_cm": ("cte_p95_m", 100.0, "CTE p95 [cm]"),
    "height_mae_cm": ("height_mae_m", 100.0, "Height MAE [cm]"),
    "speed_error_m_s": ("speed_abs_error_mean_m_s", 1.0, "Mean |speed error| [m/s]"),
    "abs_final_schedule_debt_m": ("final_schedule_debt_m", -1.0, "|Final schedule debt| [m]"),
}


def _parse_bool(value: str) -> float:
    lowered = value.strip().lower()
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    return np.nan


def _read_route_metrics(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["path_shape"], int(row["repeat"]))
            for output_name, (source_name, scale, _) in METRICS.items():
                raw = row.get(source_name, "")
                if source_name in {"survived", "task_success"}:
                    value = _parse_bool(raw)
                else:
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        value = np.nan
                    if scale < 0.0:
                        value = abs(value)
                        scale = abs(scale)
                    value *= scale
                if np.isfinite(value):
                    grouped[key][output_name].append(value)
    return {
        key: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def _stratified_bootstrap(
    values: dict[tuple[str, int], float], *, rng: np.random.Generator, samples: int
) -> np.ndarray:
    by_family: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in values:
        by_family[key[0]].append(key)
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected: list[float] = []
        for keys in by_family.values():
            indices = rng.integers(0, len(keys), size=len(keys))
            selected.extend(values[keys[index]] for index in indices)
        draws[sample] = float(np.mean(selected))
    return draws


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, help="LABEL=trace-directory (repeat for each intervention)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap_samples must be at least 100")

    conditions: dict[str, Path] = {}
    for specification in args.condition:
        if "=" not in specification:
            raise ValueError(f"Invalid --condition {specification!r}; expected LABEL=DIR")
        label, directory = specification.split("=", 1)
        path = Path(directory).resolve() / "diagnostic_v2" / "route_diagnostics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        conditions[label] = path
    if "dynamic" not in conditions:
        raise ValueError("A dynamic=DIR reference condition is required")

    route_values = {label: _read_route_metrics(path) for label, path in conditions.items()}
    common_routes = set.intersection(*(set(values) for values in route_values.values()))
    if not common_routes:
        raise ValueError("Conditions contain no common materialized routes")
    rng = np.random.default_rng(args.seed)
    aggregate_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []

    for label, values in route_values.items():
        for key in sorted(common_routes):
            route_rows.append({"condition": label, "path_shape": key[0], "repeat": key[1], **values[key]})
        for metric in METRICS:
            selected = {key: values[key][metric] for key in common_routes if metric in values[key]}
            draws = _stratified_bootstrap(selected, rng=rng, samples=args.bootstrap_samples)
            aggregate_rows.append({
                "condition": label,
                "metric": metric,
                "estimate": float(np.mean(list(selected.values()))),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "n_routes": len(selected),
            })

    reference = route_values["dynamic"]
    for label, values in route_values.items():
        if label == "dynamic":
            continue
        for metric in METRICS:
            differences = {
                key: reference[key][metric] - values[key][metric]
                for key in common_routes
                if metric in reference[key] and metric in values[key]
            }
            draws = _stratified_bootstrap(differences, rng=rng, samples=args.bootstrap_samples)
            paired_rows.append({
                "comparison": f"dynamic_minus_{label}",
                "metric": metric,
                "estimate": float(np.mean(list(differences.values()))),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "n_paired_routes": len(differences),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "pace_intervention_route_metrics.csv", route_rows)
    _write_csv(args.output_dir / "pace_intervention_aggregate.csv", aggregate_rows)
    _write_csv(args.output_dir / "pace_intervention_paired_differences.csv", paired_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), facecolor="white", squeeze=False)
    labels = list(conditions)
    colors = {"dynamic": "#1b9e77", "frozen": "#d95f02", "shuffled": "#7570b3"}
    for axis, (metric, (_, _, title)) in zip(axes.flat, METRICS.items()):
        rows = [row for row in aggregate_rows if row["metric"] == metric]
        for position, label in enumerate(labels):
            row = next(item for item in rows if item["condition"] == label)
            estimate = float(row["estimate"])
            axis.errorbar(position, estimate, yerr=[[estimate - float(row["ci95_low"])], [float(row["ci95_high"]) - estimate]], fmt="o", capsize=4, color=colors.get(label, "#333333"))
        axis.set_xticks(range(len(labels)), labels, rotation=20)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    for axis in axes.flat[len(METRICS):]:
        axis.set_visible(False)
    figure.suptitle("Pace-conditioning interventions\nroute-stratified bootstrap 95% intervals", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(args.output_dir / "pace_intervention_summary.png", dpi=190)
    plt.close(figure)

    print(f"Wrote paired pace summary for {len(common_routes)} routes to {args.output_dir}")


if __name__ == "__main__":
    main()
