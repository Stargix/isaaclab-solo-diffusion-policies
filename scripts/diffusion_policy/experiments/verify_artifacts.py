#!/usr/bin/env python3
"""Verify immutable experiment artifacts declared in a JSON manifest.

This utility deliberately depends only on the Python standard library.  It can
therefore run on a workstation or a cluster login node without importing
Isaac Sim or PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve_repository_file(raw_path: str) -> Path:
    path = (REPOSITORY_ROOT / raw_path).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"Artifact escapes repository root: {raw_path}") from error
    return path


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema_version')!r}")

    results: list[dict[str, Any]] = []
    for artifact in manifest.get("artifacts", []):
        path = resolve_repository_file(str(artifact["path"]))
        required = bool(artifact.get("required", True))
        row: dict[str, Any] = {
            "role": artifact["role"],
            "path": str(path),
            "required": required,
            "exists": path.is_file(),
        }
        if not path.is_file():
            row["valid"] = not required
            row["error"] = "missing"
            results.append(row)
            continue

        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        row.update(
            {
                "actual_bytes": actual_size,
                "expected_bytes": int(artifact["bytes"]),
                "actual_sha256": actual_hash,
                "expected_sha256": str(artifact["sha256"]).upper(),
            }
        )
        row["valid"] = (
            actual_size == row["expected_bytes"]
            and actual_hash == row["expected_sha256"]
        )
        results.append(row)

    return {
        "experiment_id": manifest.get("experiment_id"),
        "manifest": str(manifest_path.resolve()),
        "repository_root": str(REPOSITORY_ROOT),
        "valid": bool(results) and all(row["valid"] for row in results),
        "artifacts": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    report = verify_manifest(args.manifest.resolve())
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
