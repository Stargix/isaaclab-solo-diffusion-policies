#!/usr/bin/env python3
"""Render single-line Phase A evaluation commands from a versioned protocol."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("Boolean flags are emitted separately.")
    return str(value)


def build_command(run: dict, *, platform: str, checkpoint: str, results_root: str) -> str:
    launcher = r".\isaaclab.bat" if platform == "powershell" else "./isaaclab.sh"
    output_dir = str(Path(results_root) / run["id"]).replace("\\", "/")
    tokens = [launcher, "-p", "scripts/diffusion_policy/evaluate_policy.py"]
    arguments: dict[str, object] = {
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "path_shapes": run["path_shapes"],
        "speeds": run["speeds"],
        "repeats": run["repeats"],
        "duration_s": run["duration_s"],
        "num_inference_steps": run["num_inference_steps"],
        "exec_horizon": run["exec_horizon"],
        "seed": run["seed"],
    }
    optional_arguments = (
        "path_height",
        "path_heights",
        "height_profile",
        "height_segment_m",
        "height_cycle",
        "transition_fractions",
        "warmup_steps",
    )
    for key in optional_arguments:
        if key in run:
            arguments[key] = run[key]
    for key, value in arguments.items():
        tokens.append(f"--{key}")
        if isinstance(value, list):
            tokens.extend(_format_value(item) for item in value)
        else:
            tokens.append(_format_value(value))
    for flag in ("deterministic_resets", "require_empty_output_dir", "headless"):
        if run.get(flag, False):
            tokens.append(f"--{flag}")
    if platform == "powershell":
        command = " ".join(f'"{token}"' if " " in token else token for token in tokens)
        return (
            f"{command}; if (-not (Test-Path '{output_dir}/evaluation_summary.json')) "
            "{ throw 'Evaluation did not produce evaluation_summary.json' }"
        )
    return f"{shlex.join(tokens)} && test -f {shlex.quote(output_dir + '/evaluation_summary.json')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--platform", choices=("linux", "powershell"), required=True)
    parser.add_argument("--checkpoint", default="checkpoints_iri/real_walk_crouch_hindsight.pt")
    parser.add_argument(
        "--results_root",
        default="scripts/diffusion_policy/evaluations/phase_a_causal_audit_v1",
    )
    parser.add_argument("--runs", nargs="+", default=None, help="Optional subset of run ids.")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    requested = set(args.runs or [])
    known = {run["id"] for run in protocol["runs"]}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown run ids: {sorted(unknown)}")
    for run in protocol["runs"]:
        if requested and run["id"] not in requested:
            continue
        print(build_command(
            run,
            platform=args.platform,
            checkpoint=args.checkpoint,
            results_root=args.results_root,
        ))


if __name__ == "__main__":
    main()
