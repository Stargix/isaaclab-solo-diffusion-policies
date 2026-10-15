"""Decompose the declared endpoint screen without changing its thresholds."""
import argparse
import json
from pathlib import Path
import numpy as np
from .reporting import save_json
from .convert import sha


def audit(path, output):
    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {"source_sha256": sha(path), "engines": {},
              "scope": "Post-first-arrival endpoint screen, not hardware certification",
              "limitations": ["Raw 20-ms base motion includes gait oscillation; no settling grace.",
                              "Component failure counts overlap and are not additive.",
                              "Nonarrivals are unassessed, not safe; invalid windows fail.",
                              "No perturbation, transport, watchdog or emergency-stop test."]}
    for engine, attempts in rows.items():
        assessed = [r["hold"] for r in attempts if r["hold"]["assessed"]]
        components = {}
        for key, threshold in (("position_p95_m", .2), ("speed_p95_mps", .15), ("height_mae_m", .03)):
            values = [h[key] for h in assessed if key in h]
            if any(not np.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite hold statistic: {engine}, {key}")
            components[key] = {"threshold": threshold, "finite_windows": len(values),
                               "failures": sum(v > threshold for v in values),
                               "median": float(np.median(values)) if values else None,
                               "p95": float(np.percentile(values, 95)) if values else None}
        result["engines"][engine] = {"attempts": len(attempts), "assessed": len(assessed),
                                     "passes": sum(h["pass"] for h in assessed),
                                     "invalid_windows": sum("failure" in h for h in assessed),
                                     "components": components}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.attempts, args.output), indent=2))


if __name__ == "__main__":
    main()
