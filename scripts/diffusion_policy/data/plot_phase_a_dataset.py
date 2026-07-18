#!/usr/bin/env python3
"""Plot the fixed routes, achieved motion and teacher corrections in Phase A.

This is intentionally independent of the training DataLoader.  It visualises
the actual HDF5 contract so a dataset can be rejected for a geometric or
alignment problem before a GPU train starts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


def _decode(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_")[-1]), name
    except ValueError:
        return 0, name


def _load_demo(data: h5py.Group, name: str) -> dict[str, np.ndarray | str]:
    demo = data[name]
    obs = demo["obs"]
    family = _decode(demo.attrs.get("route_family", "unknown"))
    return {
        "name": name,
        "family": family,
        "root": obs["root_pos_w"][:].astype(np.float32),
        "reference": obs["reference_pos_w"][:].astype(np.float32),
        "command": obs["command_speed"][:].astype(np.float32),
        "reference_command": obs["reference_command"][:].astype(np.float32),
        "error": obs["tracking_error_frenet"][:].astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_demos", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as file:
        data = file["data"]
        names = sorted(data.keys(), key=_demo_key)
        if not names:
            raise ValueError("Dataset contains no demonstrations.")
        rng = np.random.default_rng(args.seed)
        selected_idx = np.linspace(0, len(names) - 1, min(args.num_demos, len(names)), dtype=int)
        selected = [_load_demo(data, names[int(i)]) for i in selected_idx]
        all_demos = [_load_demo(data, name) for name in names]

    families = sorted({str(d["family"]) for d in all_demos})
    colors = {family: f"C{index % 10}" for index, family in enumerate(families)}
    counts = Counter(str(d["family"]) for d in all_demos)
    del rng  # selection is deterministic and covers the complete demo index range

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # All paths are translated to the measured robot start.  A visible offset
    # between the two curves is expected and is the recovery signal, not noise.
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    labelled: set[str] = set()
    for demo in selected:
        family = str(demo["family"])
        root = np.asarray(demo["root"])
        reference = np.asarray(demo["reference"])
        origin = root[0, :2]
        label = family if family not in labelled else None
        axes[0, 0].plot(reference[:, 0] - origin[0], reference[:, 1] - origin[1],
                         color=colors[family], alpha=0.7, label=f"reference {label}" if label else None)
        axes[0, 0].plot(root[:, 0] - origin[0], root[:, 1] - origin[1],
                         color=colors[family], linestyle="--", alpha=0.45,
                         label=f"achieved {label}" if label else None)
        labelled.add(family)
    axes[0, 0].set_title("Fixed reference vs achieved XY")
    axes[0, 0].set_xlabel("x relative to robot start [m]")
    axes[0, 0].set_ylabel("y relative to robot start [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=8, ncol=2)

    example = selected[0]
    root = np.asarray(example["root"])
    reference = np.asarray(example["reference"])
    origin = root[0, :2]
    axes[0, 1].plot(reference[:, 0] - origin[0], reference[:, 1] - origin[1],
                     label="reference", linewidth=2.2, color="tab:blue")
    axes[0, 1].plot(root[:, 0] - origin[0], root[:, 1] - origin[1],
                     label="achieved", linewidth=1.5, color="tab:orange")
    axes[0, 1].scatter(reference[0, 0] - origin[0], reference[0, 1] - origin[1], marker="o", color="tab:blue")
    axes[0, 1].scatter(root[0, 0] - origin[0], root[0, 1] - origin[1], marker="x", color="tab:orange")
    axes[0, 1].set_title(f"Example {example['name']} ({example['family']})")
    axes[0, 1].set_xlabel("x relative to robot start [m]")
    axes[0, 1].set_ylabel("y relative to robot start [m]")
    axes[0, 1].axis("equal")
    axes[0, 1].legend()

    time = np.arange(len(np.asarray(example["error"])), dtype=np.float32) * 0.02
    error = np.asarray(example["error"])
    command = np.asarray(example["command"])
    axes[1, 0].plot(time, error[:, 0], label="longitudinal", color="C0")
    axes[1, 0].plot(time, error[:, 1], label="lateral", color="C1")
    axes[1, 0].plot(time, error[:, 2], label="heading", color="C2")
    axes[1, 0].set_title("Teacher tracking error (example)")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("error [m / rad]")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(time, command[:, 0], label="teacher vx")
    axes[1, 1].plot(time, command[:, 1], label="teacher vy")
    axes[1, 1].plot(time, command[:, 2], label="teacher wz")
    axes[1, 1].plot(time, np.asarray(example["reference_command"])[:, 0], "--", alpha=0.7, label="nominal vx")
    axes[1, 1].set_title("Closed-loop command vs nominal route")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("command [m/s, rad/s]")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    figure.savefig(output / "phase_a_routes_overview.png", dpi=180)
    plt.close(figure)

    counts_json = {family: int(counts[family]) for family in families}
    (output / "route_family_counts.json").write_text(json.dumps(counts_json, indent=2), encoding="utf-8")
    print(json.dumps({"selected_demos": len(selected), "families": counts_json,
                      "plot": str((output / "phase_a_routes_overview.png").resolve())}, indent=2))


if __name__ == "__main__":
    main()
