#!/usr/bin/env python3
"""Sweep execution horizons and DDPM inference steps for the frozen baseline.

This is an orchestration benchmark, intentionally separate from the Isaac
evaluator.  Each point is evaluated by the same closed-loop evaluator and the
resulting JSON/CSV files are aggregated into one reproducible report.  The
comparison therefore measures the real trade-off between feedback freshness,
latency and locomotion quality instead of timing the Transformer in isolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_HORIZONS = (1, 2, 4, 6, 8)
DEFAULT_INFERENCE_STEPS = (10, 5)
QUALITY_FIELDS = (
    "vx_rmse",
    "vy_rmse",
    "wz_rmse",
    "height_rmse",
    "tilt_rms_deg",
    "action_delta_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark execution-horizon and DDPM-step trade-offs."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", default="solo12-v0")
    parser.add_argument("--exec_horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--inference_steps", type=int, nargs="+", default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--duration_s", type=float, default=8.0)
    parser.add_argument("--settling_s", type=float, default=2.0)
    parser.add_argument("--latency_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_dynamic",
        action="store_true",
        help="Also run the six-segment height sequence for every point (slower).",
    )
    torchscript_group = parser.add_mutually_exclusive_group()
    torchscript_group.add_argument(
        "--torchscript_denoiser",
        dest="use_torchscript",
        action="store_true",
        help="Use the validated TorchScript denoiser (default).",
    )
    torchscript_group.add_argument(
        "--no_torchscript",
        dest="use_torchscript",
        action="store_false",
        help="Benchmark eager PyTorch instead of TorchScript.",
    )
    parser.set_defaults(use_torchscript=True)
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Reuse a run directory when its summary.json already exists.",
    )
    return parser.parse_args()


def mean_column(rows: list[dict[str, str]], field: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


def load_point(run_dir: Path, *, horizon: int, inference_steps: int, dt: float) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with (run_dir / "static_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    latency = summary["single_env_latency"]
    deadline_ms = float(latency["deadline_ms"])
    p95_ms = float(latency["p95_ms"])
    point: dict[str, Any] = {
        "exec_horizon": horizon,
        "inference_steps": inference_steps,
        "survival_rate": float(summary["survival_rate"]),
        "num_scenarios": int(summary["num_scenarios"]),
        "control_hz": 1.0 / dt,
        "replanning_hz": 1.0 / (dt * horizon),
        "chunk_duration_ms": deadline_ms,
        "latency_mean_ms": float(latency["mean_ms"]),
        "latency_p95_ms": p95_ms,
        "latency_p99_ms": float(latency["p99_ms"]),
        "latency_max_ms": float(latency["max_ms"]),
        "deadline_margin_ms": deadline_ms - p95_ms,
        "deadline_miss_rate": float(latency["deadline_miss_rate"]),
        "amortized_p95_ms_per_action": p95_ms / horizon,
        "dynamic_rows": 0,
        "dynamic_done_rows": 0,
    }
    for field in QUALITY_FIELDS:
        point[f"mean_{field}"] = mean_column(rows, field)

    dynamic_path = run_dir / "dynamic_height_response.csv"
    if dynamic_path.exists():
        with dynamic_path.open(newline="", encoding="utf-8") as stream:
            dynamic_rows = list(csv.DictReader(stream))
        point["dynamic_rows"] = len(dynamic_rows)
        point["dynamic_done_rows"] = sum(row.get("done", "False") == "True" for row in dynamic_rows)
    return point


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted({int(row["inference_steps"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(steps))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for color, inference_steps in zip(colors, steps):
        subset = sorted(
            (row for row in rows if int(row["inference_steps"]) == inference_steps),
            key=lambda row: int(row["exec_horizon"]),
        )
        x = [int(row["exec_horizon"]) for row in subset]
        axes[0].plot(x, [row["latency_p95_ms"] for row in subset], "o-", color=color, label=f"K={inference_steps} p95")
        axes[0].plot(x, [row["chunk_duration_ms"] for row in subset], "--", color=color, alpha=0.65, label=f"K={inference_steps} deadline")
        axes[1].plot(x, [row["amortized_p95_ms_per_action"] for row in subset], "o-", color=color, label=f"K={inference_steps}")
    axes[0].set(xlabel="execution horizon (actions)", ylabel="milliseconds", title="Latency vs feedback horizon")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axhline(20.0, color="black", linestyle=":", label="20 ms/action")
    axes[1].set(xlabel="execution horizon (actions)", ylabel="p95 / action (ms)", title="Amortized inference cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(path / "runtime_tradeoff.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for color, inference_steps in zip(colors, steps):
        subset = sorted(
            (row for row in rows if int(row["inference_steps"]) == inference_steps),
            key=lambda row: int(row["exec_horizon"]),
        )
        x = [int(row["exec_horizon"]) for row in subset]
        values = [
            ([100.0 * row["survival_rate"] for row in subset], "survival (%)"),
            ([row["mean_vx_rmse"] for row in subset], "vx RMSE (m/s)"),
            ([row["mean_vy_rmse"] for row in subset], "vy RMSE (m/s)"),
            ([row["mean_wz_rmse"] for row in subset], "wz RMSE (rad/s)"),
            ([row["mean_height_rmse"] for row in subset], "height RMSE (m)"),
            ([row["mean_action_delta_rms"] for row in subset], "action delta RMS"),
        ]
        for axis, (series, ylabel) in zip(axes.flat, values):
            axis.plot(x, series, "o-", color=color, label=f"K={inference_steps}")
            axis.set(xlabel="execution horizon", ylabel=ylabel)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    fig.suptitle("Motion quality and stability trade-offs")
    fig.savefig(path / "quality_tradeoff.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if any(h < 1 or h > 8 for h in args.exec_horizons):
        raise ValueError("All execution horizons must be in [1, 8].")
    if any(k < 1 for k in args.inference_steps):
        raise ValueError("Inference steps must be positive.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).resolve().parent / "evaluate_policy.py"
    points: list[dict[str, Any]] = []
    for inference_steps in args.inference_steps:
        for horizon in args.exec_horizons:
            run_dir = output_dir / f"k{inference_steps}_h{horizon}"
            summary_path = run_dir / "summary.json"
            if not (args.reuse_existing and summary_path.exists()):
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(evaluator),
                    "--task",
                    args.task,
                    "--checkpoint",
                    str(Path(args.checkpoint).resolve()),
                    "--output_dir",
                    str(run_dir.resolve()),
                    "--exec_horizon",
                    str(horizon),
                    "--num_inference_steps",
                    str(inference_steps),
                    "--repeats",
                    str(args.repeats),
                    "--duration_s",
                    str(args.duration_s),
                    "--settling_s",
                    str(args.settling_s),
                    "--latency_samples",
                    str(args.latency_samples),
                    "--seed",
                    str(args.seed),
                    "--headless",
                ]
                if not args.include_dynamic:
                    command.append("--skip_dynamic")
                if args.use_torchscript:
                    command.append("--torchscript_denoiser")
                log_path = run_dir / "run.log"
                print(f"[SWEEP] K={inference_steps} H={horizon}")
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                log_path.write_text(completed.stdout, encoding="utf-8")
                if completed.returncode != 0:
                    tail = "\n".join(completed.stdout.splitlines()[-30:])
                    raise RuntimeError(f"Evaluation failed for K={inference_steps}, H={horizon}:\n{tail}")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            points.append(
                load_point(
                    run_dir,
                    horizon=horizon,
                    inference_steps=inference_steps,
                    dt=float(summary["control_dt_s"]),
                )
            )

    write_csv(output_dir / "runtime_quality_summary.csv", points)
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "exec_horizons": args.exec_horizons,
        "inference_steps": args.inference_steps,
        "repeats": args.repeats,
        "duration_s": args.duration_s,
        "settling_s": args.settling_s,
        "include_dynamic": args.include_dynamic,
        "torchscript_denoiser": args.use_torchscript,
        "points": len(points),
    }
    (output_dir / "benchmark.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_plots(output_dir, points)
    print(f"[DONE] Benchmark written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
