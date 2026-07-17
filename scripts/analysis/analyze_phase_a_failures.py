#!/usr/bin/env python3
"""Relate Phase-A time to failure to tracking, tilt, curvature and speed ratio."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "xy_rmse",
    "tilt_rms_deg",
    "reference_curvature_abs_mean",
    "achieved_speed_ratio",
    "action_delta_rms",
)


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="evaluation_summary.csv files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))

    result = {"rows": len(rows), "correlations_with_time_to_failure_s": {}}
    time_values = np.asarray([float(row["time_to_failure_s"]) for row in rows], dtype=np.float64)
    for field in FIELDS:
        pairs = []
        for index, row in enumerate(rows):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value) and np.isfinite(time_values[index]):
                pairs.append((time_values[index], value))
        if not pairs:
            continue
        time, value = map(np.asarray, zip(*pairs))
        result["correlations_with_time_to_failure_s"][field] = {
            "n": len(time),
            "pearson": correlation(time, value),
            "spearman": correlation(rank(time), rank(value)),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
