#!/usr/bin/env python3
"""Compare the exec-horizon runs of the Phase A causal preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "path_shape",
    "repeat",
    "requested_speed",
    "requested_height",
    "survived",
    "height_abs_error_mean",
    "achieved_height_mean",
    "route_completion_ratio",
    "route_horizon_speed_ratio",
    "route_cross_track_rmse_m",
    "route_terminal_position_error_m",
}


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary_path = path / "evaluation_summary.json"
    csv_path = path / "evaluation_summary.csv"
    if not summary_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"Missing evaluation_summary.json/csv under {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No scenarios found in {csv_path}")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"{csv_path} predates causal-audit metrics; missing {sorted(missing)}")
    if summary.get("git_dirty_before_evaluation") is not False:
        raise ValueError(f"{path} was not evaluated from a clean committed worktree.")
    if not summary.get("deterministic_resets", False):
        raise ValueError(f"{path} did not use --deterministic_resets.")
    return summary, rows


def _as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _height_causality(rows: list[dict[str, str]]) -> dict[str, float | int]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["path_shape"], row["requested_speed"], row["repeat"])
        groups[key].append(row)
    deltas = []
    correct = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda row: _as_float(row, "requested_height"))
        if len(ordered) < 2:
            continue
        observed_delta = _as_float(ordered[-1], "achieved_height_mean") - _as_float(
            ordered[0], "achieved_height_mean"
        )
        requested_delta = _as_float(ordered[-1], "requested_height") - _as_float(
            ordered[0], "requested_height"
        )
        deltas.append(observed_delta / requested_delta if requested_delta > 0.0 else math.nan)
        correct += int(observed_delta >= 0.02)
    return {
        "pairs": len(deltas),
        "correct_direction_fraction": correct / len(deltas) if deltas else math.nan,
        "mean_response_slope": _mean(deltas),
    }


def summarize(path: Path) -> dict[str, Any]:
    metadata, rows = _load_run(path)
    survived = [row["survived"].lower() == "true" for row in rows]
    completion = [_as_float(row, "route_completion_ratio") for row in rows]
    speed_error = [abs(_as_float(row, "route_horizon_speed_ratio") - 1.0) for row in rows]
    terminal_error = [_as_float(row, "route_terminal_position_error_m") for row in rows]
    cross_track = [_as_float(row, "route_cross_track_rmse_m") for row in rows]
    height_error = [_as_float(row, "height_abs_error_mean") for row in rows]
    route_success = [
        is_alive and 0.90 <= progress <= 1.10 and terminal <= 0.25
        for is_alive, progress, terminal in zip(survived, completion, terminal_error)
    ]
    return {
        "path": str(path.resolve()),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "git_commit": metadata["git_commit"],
        "exec_horizon": int(metadata["exec_horizon"]),
        "scenarios": len(rows),
        "survival_rate": sum(survived) / len(survived),
        "route_success_rate": sum(route_success) / len(route_success),
        "mean_completion_ratio": _mean(completion),
        "mean_absolute_horizon_speed_ratio_error": _mean(speed_error),
        "mean_terminal_position_error_m": _mean(terminal_error),
        "mean_cross_track_rmse_m": _mean(cross_track),
        "mean_height_absolute_error_m": _mean(height_error),
        "height_causality": _height_causality(rows),
    }


def decide(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    if first["checkpoint_sha256"] != second["checkpoint_sha256"]:
        raise ValueError("The two runs use different checkpoints.")
    candidates = [first, second]
    viable = [
        row for row in candidates
        if row["survival_rate"] >= 0.80
        and row["height_causality"]["correct_direction_fraction"] >= 0.90
    ]
    if not viable:
        return {
            "status": "base_not_ready",
            "reason": "Neither execution horizon passes the 80% safety and 90% causal-response preflight.",
        }
    best_survival = max(row["survival_rate"] for row in viable)
    safety_equivalent = [row for row in viable if best_survival - row["survival_rate"] <= 0.02]
    winner = min(
        safety_equivalent,
        key=lambda row: (
            row["mean_absolute_horizon_speed_ratio_error"],
            row["mean_terminal_position_error_m"],
            row["mean_cross_track_rmse_m"],
        ),
    )
    return {
        "status": "winner_selected",
        "exec_horizon": winner["exec_horizon"],
        "rule": "Safety/causality gate, then speed error, terminal error and cross-track error.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec1", type=Path, required=True)
    parser.add_argument("--exec4", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    runs = [summarize(args.exec1), summarize(args.exec4)]
    report = {"runs": runs, "decision": decide(runs[0], runs[1])}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
