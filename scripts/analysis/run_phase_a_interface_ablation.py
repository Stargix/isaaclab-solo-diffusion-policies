#!/usr/bin/env python3
"""Run the fixed Phase-A duration/chunk matrix with one seed and checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--durations", type=float, nargs="+", default=(15.0, 50.0))
    parser.add_argument("--exec_horizons", type=int, nargs="+", default=(1, 4, 8))
    parser.add_argument("--speeds", type=float, nargs="+", default=(0.2, 0.4, 0.6))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reuse_existing", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    evaluator = root / "scripts" / "diffusion_policy" / "evaluate_policy.py"
    output = Path(args.output_dir)
    runs = []
    for duration in args.durations:
        for horizon in args.exec_horizons:
            run_dir = output / f"duration_{duration:g}s_h{horizon}"
            summary = run_dir / "evaluation_summary.json"
            command = [
                sys.executable, str(evaluator), "--checkpoint", args.checkpoint,
                "--output_dir", str(run_dir), "--duration_s", str(duration),
                "--exec_horizon", str(horizon), "--seed", str(args.seed),
                "--repeats", str(args.repeats), "--speeds", *map(str, args.speeds), "--headless",
            ]
            if not (args.reuse_existing and summary.exists()):
                subprocess.run(command, cwd=root, check=True)
            runs.append({"duration_s": duration, "exec_horizon": horizon, "summary": str(summary.resolve())})
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation_manifest.json").write_text(json.dumps({"seed": args.seed, "runs": runs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
