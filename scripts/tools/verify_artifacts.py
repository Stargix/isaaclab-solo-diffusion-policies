"""Verify external research artifacts without importing torch or a simulator.

The manifest contains expected identities, not download credentials. A matching
hash establishes file identity relative to that trusted manifest, not authorship.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "research_artifacts/reproduction/manifest.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Artifact path must be a normalized relative path: {value!r}")
    return path


def load_manifest(path: Path) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(data, dict) or not isinstance(data.get("release_id"), str):
        raise ValueError("Manifest needs an object with a release_id")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported artifact-manifest schema")
    artifacts, profiles = data.get("artifacts"), data.get("profiles")
    if not isinstance(artifacts, list) or not isinstance(profiles, dict) or not profiles:
        raise ValueError("Manifest needs artifacts and named profiles")
    identities, paths = set(), set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError("Artifact entries must be objects")
        identity = item.get("id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError(f"Missing or duplicate artifact ID: {identity!r}")
        identities.add(identity)
        relative = str(safe_relative(item.get("path")))
        # Bundles must also be unambiguous on case-insensitive Windows filesystems.
        if relative.casefold() in paths:
            raise ValueError(f"Duplicate artifact destination: {relative}")
        paths.add(relative.casefold())
        if "sha256" not in item:
            raise ValueError(f"Missing sha256 field for {identity}")
        digest = item["sha256"]
        if digest is not None and (not isinstance(digest, str) or not SHA256.fullmatch(digest)):
            raise ValueError(f"Invalid SHA256 for {identity}")
        size = item.get("size_bytes")
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError(f"Invalid byte size for {identity}")
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or not isinstance(profile.get("artifacts"), list):
            raise ValueError(f"Invalid profile: {name}")
        required = profile["artifacts"]
        if not required or any(not isinstance(x, str) for x in required):
            raise ValueError(f"Empty or invalid artifact list in profile {name}")
        if len(required) != len(set(required)) or not set(required) <= identities:
            raise ValueError(f"Duplicate or unknown artifact IDs in profile {name}")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest: dict, root: Path, profile: str) -> dict:
    if profile not in manifest["profiles"]:
        raise ValueError(f"Unknown profile {profile!r}; choose {', '.join(manifest['profiles'])}")
    root = root.resolve()
    required = set(manifest["profiles"][profile]["artifacts"])
    outcomes = []
    for item in manifest["artifacts"]:
        if item["id"] not in required:
            continue
        path = root.joinpath(*safe_relative(item["path"]).parts)
        result = {"id": item["id"], "path": item["path"]}
        try:
            path.resolve().relative_to(root)
            if not path.is_file():
                result["status"] = "missing"
            elif item["sha256"] is None:
                result["status"] = "identity_not_pinned"
            else:
                before = path.stat()
                if item.get("size_bytes") is not None and before.st_size != item["size_bytes"]:
                    result["status"] = "size_mismatch"
                else:
                    digest = file_sha256(path)
                    after = path.stat()
                    result["observed_sha256"] = digest
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        result["status"] = "changed_during_verification"
                    else:
                        result["status"] = "verified" if digest == item["sha256"] else "hash_mismatch"
        except ValueError:
            result["status"] = "path_escapes_root"
        except OSError as exc:
            result.update(status="unreadable", error=type(exc).__name__)
        outcomes.append(result)
    passed = all(row["status"] == "verified" for row in outcomes)
    return {"schema_version": 1, "release_id": manifest["release_id"], "profile": profile,
            "complete": passed, "verified": sum(x["status"] == "verified" for x in outcomes),
            "required": len(required), "artifacts": outcomes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, required=True, help="Extracted artifact directory")
    parser.add_argument("--profile", default="demo")
    parser.add_argument("--report", type=Path, help="Write a new JSON report; never overwrite a file")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = verify(manifest, args.root, args.profile)
        if args.report is not None:
            forbidden = {args.manifest.resolve()}
            forbidden.update((args.root / x["path"]).resolve() for x in manifest["artifacts"])
            if args.report.resolve() in forbidden:
                raise ValueError("Report path would overwrite a manifest or artifact")
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(f"{args.profile}: {report['verified']}/{report['required']} files verified")
        for row in report["artifacts"]:
            if row["status"] != "verified":
                print(f"  {row['status']}: {row['path']}")
        return 0 if report["complete"] else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Artifact verification error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
