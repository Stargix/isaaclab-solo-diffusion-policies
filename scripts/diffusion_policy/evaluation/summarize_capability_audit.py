#!/usr/bin/env python3
"""Validate and summarize the frozen-policy Phase A capability audit."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .summarize_preflight import (
    _as_float,
    _expected_checkpoint_hash,
    _height_causality,
    _load_run,
    _mean,
)


def _expected_scenarios(run: dict[str, Any]) -> int:
    heights = len(run.get("path_heights", [run.get("path_height", 0.2932)]))
    transitions = len(run.get("transition_fractions", [None]))
    return (
        len(run["path_shapes"])
        * len(run["speeds"])
        * heights
        * transitions
        * int(run["repeats"])
    )


def _validate_run(
    metadata: dict[str, Any],
    rows: list[dict[str, str]],
    expected: dict[str, Any],
    checkpoint_hash: str,
) -> None:
    comparisons: dict[str, tuple[Any, Any]] = {
        "checkpoint_sha256": (str(metadata.get("checkpoint_sha256", "")).upper(), checkpoint_hash),
        "exec_horizon": (int(metadata.get("exec_horizon", -1)), int(expected["exec_horizon"])),
        "num_inference_steps": (
            int(metadata.get("num_inference_steps", -1)), int(expected["num_inference_steps"])
        ),
        "seed": (int(metadata.get("seed", -1)), int(expected["seed"])),
        "duration_s": (float(metadata.get("duration_s", math.nan)), float(expected["duration_s"])),
        "scenario_count": (len(rows), _expected_scenarios(expected)),
        "path_shapes": (
            sorted({row["path_shape"] for row in rows}), sorted(expected["path_shapes"])
        ),
        "speeds": (
            sorted(float(value) for value in metadata.get("speeds_evaluated", [])),
            sorted(float(value) for value in expected["speeds"]),
        ),
        "height_profile": (
            str(metadata.get("height_profile", "constant")), str(expected.get("height_profile", "constant"))
        ),
    }
    if "path_heights" in expected:
        comparisons["heights"] = (
            sorted(float(value) for value in metadata.get("heights_evaluated", [])),
            sorted(float(value) for value in expected["path_heights"]),
        )
    if "height_segment_m" in expected:
        comparisons["height_segment_m"] = (
            float(metadata.get("height_segment_m", math.nan)), float(expected["height_segment_m"])
        )
    if "height_cycle" in expected:
        comparisons["height_cycle"] = (
            [float(value) for value in metadata.get("height_cycle", [])],
            [float(value) for value in expected["height_cycle"]],
        )
    mismatches = {
        name: {"actual": actual, "expected": wanted}
        for name, (actual, wanted) in comparisons.items()
        if actual != wanted
    }
    if mismatches:
        raise ValueError(f"Evaluation does not match the capability protocol: {mismatches}")


def _survival_rate(rows: list[dict[str, str]]) -> float:
    return sum(row["survived"].lower() == "true" for row in rows) / len(rows)


def _route_success(row: dict[str, str]) -> bool:
    return (
        row["survived"].lower() == "true"
        and 0.90 <= _as_float(row, "route_completion_ratio") <= 1.10
        and _as_float(row, "route_terminal_position_error_m") <= 0.25
    )


def _condition_breakdown(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["path_shape"], _as_float(row, "requested_speed"))].append(row)
    output = []
    for (shape, speed), group in sorted(groups.items()):
        output.append({
            "path_shape": shape,
            "requested_speed_m_s": speed,
            "scenarios": len(group),
            "survival_rate": _survival_rate(group),
            "route_success_rate": sum(_route_success(row) for row in group) / len(group),
            "mean_absolute_speed_ratio_error": _mean([
                abs(_as_float(row, "route_horizon_speed_ratio") - 1.0) for row in group
            ]),
            "mean_terminal_position_error_m": _mean([
                _as_float(row, "route_terminal_position_error_m") for row in group
            ]),
            "mean_cross_track_rmse_m": _mean([
                _as_float(row, "route_cross_track_rmse_m") for row in group
            ]),
            "mean_height_absolute_error_m": _mean([
                _as_float(row, "height_abs_error_mean") for row in group
            ]),
        })
    return output


def summarize_run(
    path: Path, *, expected: dict[str, Any], checkpoint_hash: str
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata, rows = _load_run(path)
    _validate_run(metadata, rows, expected, checkpoint_hash)
    summary = {
        "id": expected["id"],
        "path": str(path.resolve()),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "git_commit": metadata["git_commit"],
        "scenarios": len(rows),
        "survival_rate": _survival_rate(rows),
        "route_success_rate": sum(_route_success(row) for row in rows) / len(rows),
        "mean_absolute_speed_ratio_error": _mean([
            abs(_as_float(row, "route_horizon_speed_ratio") - 1.0) for row in rows
        ]),
        "mean_terminal_position_error_m": _mean([
            _as_float(row, "route_terminal_position_error_m") for row in rows
        ]),
        "mean_cross_track_rmse_m": _mean([
            _as_float(row, "route_cross_track_rmse_m") for row in rows
        ]),
        "mean_height_absolute_error_m": _mean([
            _as_float(row, "height_abs_error_mean") for row in rows
        ]),
        "height_causality": _height_causality(rows),
        "by_route_and_speed": _condition_breakdown(rows),
    }
    return summary, rows


def decide(core: dict[str, Any], random_height: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    worst_group_survival = min(
        group["survival_rate"] for group in core["by_route_and_speed"]
    )
    checks = {
        "overall_survival": core["survival_rate"] >= gates["minimum_overall_survival_rate"],
        "worst_route_speed_survival": (
            worst_group_survival >= gates["minimum_route_speed_group_survival_rate"]
        ),
        "path_tracking": (
            core["mean_cross_track_rmse_m"] <= gates["maximum_mean_cross_track_rmse_m"]
        ),
        "constant_height_tracking": (
            core["mean_height_absolute_error_m"] <= gates["maximum_mean_height_absolute_error_m"]
        ),
        "height_causality": (
            core["height_causality"]["correct_direction_fraction"]
            >= gates["minimum_height_causal_direction_fraction"]
        ),
        "mean_speed_tracking": (
            core["mean_absolute_speed_ratio_error"]
            <= gates["maximum_mean_absolute_speed_ratio_error"]
        ),
        "finite_route_success": core["route_success_rate"] >= gates["minimum_route_success_rate"],
        "random_height_survival": (
            random_height["survival_rate"] >= gates["minimum_random_height_survival_rate"]
        ),
        "random_height_tracking": (
            random_height["mean_height_absolute_error_m"]
            <= gates["maximum_random_height_absolute_error_m"]
        ),
    }
    foundation = all(checks[key] for key in (
        "overall_survival",
        "worst_route_speed_survival",
        "path_tracking",
        "constant_height_tracking",
        "height_causality",
        "random_height_survival",
        "random_height_tracking",
    ))
    timing = checks["mean_speed_tracking"] and checks["finite_route_success"]
    if foundation and timing:
        status = "base_task_sufficient"
        recommendation = "Do not add RL unless a harder held-out test exposes a specific failure."
    elif foundation:
        status = "command_timing_correction_candidate"
        recommendation = (
            "Correct progress/velocity in command space first; joint-space residual RL is not justified by this failure mode."
        )
    else:
        status = "base_envelope_not_ready"
        recommendation = (
            "Repair or restrict the frozen base envelope before online optimization; RL must not be asked to recover unsafe or unrepresented skills."
        )
    return {
        "status": status,
        "checks": checks,
        "worst_route_speed_survival_rate": worst_group_survival,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected = {run["id"]: run for run in protocol["runs"]}
    checkpoint_hash = _expected_checkpoint_hash(protocol)
    core, _ = summarize_run(
        args.results_root / "core_envelope",
        expected=expected["core_envelope"],
        checkpoint_hash=checkpoint_hash,
    )
    random_height, _ = summarize_run(
        args.results_root / "random_height_challenge",
        expected=expected["random_height_challenge"],
        checkpoint_hash=checkpoint_hash,
    )
    if core["git_commit"] != random_height["git_commit"]:
        raise ValueError("Capability runs were produced from different code commits.")
    report = {
        "protocol_id": protocol["protocol_id"],
        "runs": [core, random_height],
        "decision": decide(core, random_height, protocol["task_gates"]),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
