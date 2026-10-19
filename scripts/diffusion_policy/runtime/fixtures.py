"""Opt-in, non-pickle oracle snapshots for same-backend sampler replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


FIXTURE_VERSION = 1


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.hasobject:
        raise TypeError("Runtime fixtures do not permit pickle/object arrays.")
    return result.copy()


def capture_rng_state(device: torch.device) -> dict[str, np.ndarray]:
    """Read current RNG state without sampling or advancing either generator."""
    state = {"rng_cpu": torch.get_rng_state().cpu().numpy().copy()}
    if device.type == "cuda":
        state["rng_cuda"] = torch.cuda.get_rng_state(device).cpu().numpy().copy()
    else:
        state["rng_cuda"] = np.empty((0,), dtype=np.uint8)
    return state


def restore_rng_state(record: dict[str, np.ndarray], device: torch.device) -> None:
    cpu = np.asarray(record["rng_cpu"], dtype=np.uint8)
    torch.set_rng_state(torch.from_numpy(cpu.copy()))
    cuda = np.asarray(record.get("rng_cuda", np.empty(0, dtype=np.uint8)), dtype=np.uint8)
    if device.type == "cuda":
        if cuda.size == 0:
            raise ValueError("CUDA replay requires the captured CUDA RNG state.")
        torch.cuda.set_rng_state(torch.from_numpy(cuda.copy()), device)


def select_capture_steps(duration_steps: int, exec_horizon: int, max_samples: int) -> tuple[int, ...]:
    """Spread snapshots across a rollout, snapping them to chunk boundaries."""
    if duration_steps < 1 or exec_horizon != 4 or max_samples < 1:
        raise ValueError("Expected positive duration/sample count and WC-v3 exec_horizon=4.")
    replans = (duration_steps + exec_horizon - 1) // exec_horizon
    count = min(replans, max_samples)
    slots = np.linspace(0, replans - 1, count, dtype=np.int64)
    steps = tuple(dict.fromkeys(int(slot * exec_horizon) for slot in slots))
    if not steps or steps[0] != 0:
        raise RuntimeError("Snapshot schedule must include the initial policy inference.")
    return steps


class RuntimeFixtureWriter:
    """Collect detached copies in memory, then write one NPZ+JSON fixture pair."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        metadata: dict[str, Any],
        duration_steps: int,
        exec_horizon: int,
        max_samples: int,
        num_envs: int,
        sequential_steps: int = 0,
    ) -> None:
        if num_envs != 1:
            raise ValueError("Oracle fixture capture is intentionally batch=1; use one scenario.")
        self.output_dir = Path(output_dir)
        self.metadata = dict(metadata)
        if sequential_steps < 0:
            raise ValueError("sequential_steps must be non-negative.")
        if sequential_steps:
            if sequential_steps < exec_horizon * 2:
                raise ValueError("Sequential capture must include at least two complete action chunks.")
            self.steps = set(range(min(duration_steps, sequential_steps)))
            if len(self.steps) != sequential_steps:
                raise ValueError("Evaluation duration is shorter than the requested sequential fixture.")
            self.capture_mode = "sequential"
        else:
            self.steps = set(select_capture_steps(duration_steps, exec_horizon, max_samples))
            self.capture_mode = "distributed_snapshots"
        self.records: list[dict[str, np.ndarray]] = []
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Refusing to mix fixture output with existing files: {self.output_dir}")

    def should_capture(self, step: int) -> bool:
        return int(step) in self.steps

    def begin(self, **fields: Any) -> dict[str, np.ndarray]:
        return {str(key): _array(value) for key, value in fields.items()}

    def finish(self, record: dict[str, np.ndarray], **fields: Any) -> None:
        if len(self.records) >= len(self.steps):
            raise RuntimeError("More oracle snapshots received than the declared capture schedule.")
        for key, value in fields.items():
            if key in record:
                raise ValueError(f"Duplicate runtime fixture field {key!r}.")
            record[str(key)] = _array(value)
        if not record:
            raise ValueError("Cannot write an empty runtime snapshot.")
        self.records.append(record)

    def close(self) -> tuple[Path, Path]:
        if not self.records:
            raise RuntimeError("No runtime snapshots were captured; fixture files not written.")
        if len(self.records) != len(self.steps):
            raise RuntimeError(
                f"Captured {len(self.records)} snapshots, expected {len(self.steps)} from the schedule."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        manifest = []
        for index, record in enumerate(self.records):
            prefix = f"sample_{index:03d}__"
            names = []
            for key, value in record.items():
                archive_key = prefix + key
                arrays[archive_key] = value
                names.append(key)
            manifest.append({"index": index, "fields": sorted(names)})
        metadata = {
            "fixture_version": FIXTURE_VERSION,
            "capture_mode": self.capture_mode,
            "metadata": self.metadata,
            "samples": manifest,
        }
        npz_path = self.output_dir / "oracle_snapshots.npz"
        json_path = self.output_dir / "oracle_metadata.json"
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        return npz_path, json_path


def load_fixture_bundle(path: str | Path) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    """Read versioned snapshots with ``allow_pickle=False``."""
    directory = Path(path)
    json_path = directory / "oracle_metadata.json"
    npz_path = directory / "oracle_snapshots.npz"
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"Expected oracle_metadata.json and oracle_snapshots.npz in {directory}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError(f"Unsupported runtime fixture version {metadata.get('fixture_version')!r}.")
    records = []
    with np.load(npz_path, allow_pickle=False) as archive:
        for sample in metadata.get("samples", []):
            index = int(sample["index"])
            prefix = f"sample_{index:03d}__"
            record = {}
            for field in sample["fields"]:
                key = prefix + field
                if key not in archive:
                    raise ValueError(f"Fixture manifest references missing array {key!r}.")
                value = archive[key].copy()
                if value.dtype.hasobject:
                    raise TypeError(f"Unsafe object array in fixture field {key!r}.")
                record[field] = value
            records.append(record)
    if len(records) != len(metadata.get("samples", [])) or not records:
        raise ValueError("Fixture bundle has no consistent samples.")
    return metadata, records
