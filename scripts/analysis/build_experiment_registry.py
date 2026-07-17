#!/usr/bin/env python3
"""Build one traceable registry from every committed evaluation summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_NAMES = {"summary.json", "evaluation_summary.json"}


def _nested(values: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = values
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _family(path: Path) -> str:
    parts = set(path.parts)
    for name in ("baseline_diffuseloco", "baseline_locodiff", "diffusion_policy"):
        if name in parts:
            return name
    return "unknown"


def _dataset_hashes(summary: dict[str, Any]) -> str:
    hashes = summary.get("dataset_sha256", {})
    if not isinstance(hashes, dict):
        return ""
    return " | ".join(f"{Path(path).name}:{digest}" for path, digest in sorted(hashes.items()))


def _row(path: Path, root: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    survival = summary.get("survival_rate", summary.get("overall_survival_rate"))
    k_value = summary.get("num_inference_steps", _nested(config, "diffusion", "num_inference_steps"))
    exec_horizon = summary.get("exec_horizon")
    checkpoint = str(summary.get("checkpoint", ""))
    row = {
        "family": _family(path),
        "evaluation": str(path.parent.relative_to(root)),
        "timestamp": summary.get("timestamp", ""),
        "policy_kind": summary.get("policy_kind", _nested(config, "policy_kind", default="")),
        "checkpoint": checkpoint,
        "checkpoint_sha256": summary.get("checkpoint_sha256", ""),
        "git_commit": summary.get("git_commit", ""),
        "dataset_sha256": _dataset_hashes(summary),
        "task": summary.get("task", ""),
        "seed": summary.get("seed", ""),
        "duration_s": summary.get("duration_s", ""),
        "num_scenarios": summary.get("num_scenarios", ""),
        "num_inference_steps": k_value if k_value is not None else "",
        "exec_horizon": exec_horizon if exec_horizon is not None else "",
        "survival_rate": survival if survival is not None else "",
        "source_file": str(path.relative_to(root)),
    }
    required = ("checkpoint_sha256", "git_commit", "dataset_sha256", "seed", "duration_s")
    row["missing_provenance"] = ",".join(key for key in required if row[key] in ("", None))
    return row


def discover(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "scripts").rglob("*.json")):
        if path.name not in SUMMARY_NAMES or "evaluations" not in path.parts:
            continue
        try:
            rows.append(_row(path, root))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            rows.append({
                "family": _family(path),
                "evaluation": str(path.parent.relative_to(root)),
                "source_file": str(path.relative_to(root)),
                "missing_provenance": f"invalid_summary:{exc}",
            })
    return rows


def write_registry(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "family", "evaluation", "timestamp", "policy_kind", "checkpoint",
        "checkpoint_sha256", "git_commit", "dataset_sha256", "task", "seed",
        "duration_s", "num_scenarios", "num_inference_steps", "exec_horizon",
        "survival_rate", "missing_provenance", "source_file",
    )
    normalized = [{field: row.get(field, "") for field in fields} for row in rows]
    with (output_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    (output_dir / "experiment_registry.json").write_text(
        json.dumps({"experiments": normalized}, indent=2), encoding="utf-8"
    )

    lines = [
        "# Registro de evaluaciones",
        "",
        "Generado a partir de los `summary.json` existentes. Una celda vacía indica que el run antiguo no guardó ese dato.",
        "",
        "| Familia | Evaluación | Duración | K/H | Supervivencia | Trazabilidad pendiente |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in normalized:
        kh = f"{row['num_inference_steps']}/{row['exec_horizon']}"
        lines.append(
            f"| {row['family']} | `{row['evaluation']}` | {row['duration_s']} | {kh} | "
            f"{row['survival_rate']} | {row['missing_provenance']} |"
        )
    (output_dir / "experiment_registry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output_dir", type=Path, default=Path("scripts/analysis/registry"))
    parser.add_argument("--strict", action="store_true", help="Fail if a summary lacks core provenance.")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    rows = discover(root)
    if not rows:
        raise ValueError(f"No evaluation summaries found below {root / 'scripts'}.")
    write_registry(rows, output)
    missing = [row for row in rows if row.get("missing_provenance")]
    print(f"[DONE] {len(rows)} evaluations indexed in {output}")
    print(f"[INFO] {len(missing)} legacy summaries have incomplete provenance.")
    if args.strict and missing:
        raise ValueError("Incomplete provenance: inspect experiment_registry.csv.")


if __name__ == "__main__":
    main()

