# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Merge several collected HDF5 datasets into one for Diffusion Policy training.

The output re-indexes demos sequentially (``demo_0``, ``demo_1``, ...) and, crucially,
RE-MAPS the per-step ``skill_idx`` of every source demo onto a unified ``skill_names``
table. Without this remap, merging e.g. ``walk_raw.hdf5`` (skill_names=["walk"]) and
``crouch_raw.hdf5`` (skill_names=["crouch"]) would leave ``skill_idx=0`` meaning both
"walk" and "crouch" in the combined file -- ambiguous and a silent bug for the
DataLoader / conditioning.

Example usage::

    python scripts/diffusion_policy/data/merge_datasets.py \
        --inputs scripts/diffusion_policy/data/datasets/walk_raw.hdf5 \
                 scripts/diffusion_policy/data/datasets/crouch_raw.hdf5 \
        --output scripts/diffusion_policy/data/datasets/combined_A.hdf5
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

# Keep file locking off on Windows to avoid spurious lock errors.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

REQUIRED_OBS_KEYS = (
    "joint_pos",
    "joint_vel",
    "base_ang_vel",
    "projected_gravity",
    "last_action",
    "root_pos_w",
    "root_quat_w",
    "command_speed",
    "desired_base_height",
)

EXPECTED_CONVENTION_PREFIX = "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]"


def _read_skill_names(data_group) -> list[str]:
    raw = data_group.attrs.get("skill_names", None)
    if raw is None:
        return []
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _validate_source_file(path: str) -> None:
    """Reject legacy or malformed datasets before writing a misleading merged file."""
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{path}: missing required 'data' group.")

        data = f["data"]
        condition_schema = _decode_attr(data.attrs.get("condition_schema", ""))
        if condition_schema != "velocity_xyyaw_plus_desired_base_height_v1":
            raise ValueError(
                f"{path}: expected velocity-plus-height condition schema, got {condition_schema!r}. "
                "Regenerate it with the current collector."
            )
        convention = data.attrs.get("convention", None)
        if convention is None:
            raise ValueError(
                f"{path}: missing data.attrs['convention']. "
                "Regenerate with the current collect_data.py before merging."
            )
        if not _decode_attr(convention).startswith(EXPECTED_CONVENTION_PREFIX):
            raise ValueError(
                f"{path}: unsupported convention {convention!r}. "
                "Regenerate with the current collect_data.py before merging."
            )

        skill_names = _read_skill_names(data)
        if not skill_names:
            raise ValueError(
                f"{path}: missing data.attrs['skill_names']. "
                "This looks like a legacy dataset; regenerate it before merging."
            )

        demo_names = list(data.keys())
        if not demo_names:
            raise ValueError(f"{path}: contains no demos under 'data'.")

        for demo_name in demo_names:
            demo = data[demo_name]
            if "obs" not in demo:
                raise KeyError(f"{path}/{demo_name}: missing 'obs' group.")
            obs = demo["obs"]
            missing_obs = [key for key in REQUIRED_OBS_KEYS if key not in obs]
            if missing_obs:
                raise KeyError(
                    f"{path}/{demo_name}: missing obs keys {missing_obs}. "
                    "This is likely a pre-alignment dataset."
                )
            for key in ("actions", "dones", "skill_idx"):
                if key not in demo:
                    raise KeyError(f"{path}/{demo_name}: missing required dataset '{key}'.")

            num_samples = obs["joint_pos"].shape[0]
            expected_dims = {
                "joint_pos": 12,
                "joint_vel": 12,
                "base_ang_vel": 3,
                "projected_gravity": 3,
                "last_action": 12,
                "root_pos_w": 3,
                "root_quat_w": 4,
                "command_speed": 3,
                "desired_base_height": 1,
            }
            for key, dim in expected_dims.items():
                if obs[key].ndim != 2 or obs[key].shape[1] != dim:
                    raise ValueError(f"{path}/{demo_name}/obs/{key}: expected shape (T, {dim}), got {obs[key].shape}.")
                if obs[key].shape[0] != num_samples:
                    raise ValueError(f"{path}/{demo_name}/obs/{key}: length does not match joint_pos.")
            if demo["actions"].shape[0] != num_samples:
                raise ValueError(f"{path}/{demo_name}: actions length does not match obs length.")
            if demo["actions"].ndim != 2 or demo["actions"].shape[1] != 12:
                raise ValueError(f"{path}/{demo_name}: expected actions shape (T, 12), got {demo['actions'].shape}.")
            if demo["dones"].shape[0] != num_samples:
                raise ValueError(f"{path}/{demo_name}: dones length does not match obs length.")
            if demo["skill_idx"].shape[0] != num_samples:
                raise ValueError(f"{path}/{demo_name}: skill_idx length does not match obs length.")
            skill_idx = demo["skill_idx"][:]
            if skill_idx.min() < 0 or skill_idx.max() >= len(skill_names):
                raise ValueError(f"{path}/{demo_name}: skill_idx contains values outside data.attrs['skill_names'].")


def _remap_skill_idx(src_demo, src_skill_names: list[str],
                     name_to_unified: dict[str, int]) -> np.ndarray | None:
    if "skill_idx" not in src_demo:
        return None
    src_idx = src_demo["skill_idx"][:]
    remap = np.array(
        [name_to_unified[name] for name in src_skill_names], dtype=np.int16,
    )
    # Clip out-of-range indices defensively (shouldn't happen with valid data).
    src_idx = np.clip(src_idx.astype(np.int16), 0, len(src_skill_names) - 1)
    return remap[src_idx]


def merge_hdf5_files(
    input_files: list[str],
    output_file: str,
    *,
    shuffle_demos: bool = True,
    shuffle_seed: int = 42,
) -> None:
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: build the unified skill_names table in first-seen order.
    unified_skill_names: list[str] = []
    per_source_skills: list[list[str]] = []
    for path in input_files:
        _validate_source_file(path)
        with h5py.File(path, "r") as f:
            names = _read_skill_names(f["data"])
        per_source_skills.append(names)
        for n in names:
            if n not in unified_skill_names:
                unified_skill_names.append(n)

    name_to_unified = {n: i for i, n in enumerate(unified_skill_names)}
    print(f"[INFO] Unified skill_names: {unified_skill_names}")

    demo_entries: list[tuple[str, str, list[str]]] = []
    for path, src_skills in zip(input_files, per_source_skills):
        with h5py.File(path, "r") as in_f:
            in_data = in_f["data"]
            for demo_name in sorted(in_data.keys(), key=_demo_sort_key):
                demo_entries.append((path, demo_name, src_skills))

    if shuffle_demos and len(demo_entries) > 1:
        order = np.random.default_rng(shuffle_seed).permutation(len(demo_entries))
        demo_entries = [demo_entries[int(i)] for i in order]
        print(f"[INFO] Shuffled {len(demo_entries)} demos before writing (seed={shuffle_seed}).")

    # Pass 2: copy demos, remapping skill_idx and writing unified metadata.
    demo_counter = 0
    with h5py.File(out_path, "w") as out_f:
        out_data = out_f.create_group("data")
        out_data.attrs["skill_names"] = np.array(unified_skill_names, dtype="S32")
        # Preserve the alignment convention documented in collect_data.py.
        out_data.attrs["convention"] = (
            "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]; "
            "last_action[t] = actions[t-1] (last_action[0] = 0)."
        )
        out_data.attrs["control_rate_hz"] = 50.0
        out_data.attrs["condition_schema"] = "velocity_xyyaw_plus_desired_base_height_v1"
        out_data.attrs["merged_from"] = np.array(
            [str(Path(p).name) for p in input_files], dtype="S64",
        )
        if shuffle_demos:
            out_data.attrs["demo_order"] = "shuffled"
            out_data.attrs["shuffle_seed"] = shuffle_seed
        else:
            out_data.attrs["demo_order"] = "sequential"

        open_files: dict[str, h5py.File] = {}
        try:
            for path, demo_name, src_skills in demo_entries:
                if path not in open_files:
                    open_files[path] = h5py.File(path, "r")
                in_f = open_files[path]
                in_data = in_f["data"]
                new_name = f"demo_{demo_counter}"
                new_grp = out_data.create_group(new_name)

                in_f.copy(f"data/{demo_name}/obs", new_grp, name="obs")
                in_f.copy(f"data/{demo_name}/actions", new_grp, name="actions")
                in_f.copy(f"data/{demo_name}/dones", new_grp, name="dones")

                remapped = _remap_skill_idx(
                    in_data[demo_name], src_skills, name_to_unified,
                )
                if remapped is not None:
                    new_grp.create_dataset(
                        "skill_idx", data=remapped.astype(np.int8),
                    )

                for attr_name in ("num_samples", "skills_sequence"):
                    if attr_name in in_data[demo_name].attrs:
                        new_grp.attrs[attr_name] = in_data[demo_name].attrs[attr_name]
                if "obs" in new_grp:
                    new_grp.attrs["num_samples"] = new_grp["obs"]["joint_pos"].shape[0]

                demo_counter += 1
        finally:
            for handle in open_files.values():
                handle.close()

    print(f"[SUCCESS] Merged {len(input_files)} file(s) into {out_path}. "
          f"Total demos: {demo_counter}. Skills: {unified_skill_names}")


def _demo_sort_key(name: str) -> int:
    """Sort demo_0, demo_1, ..., demo_10 numerically (not lexicographically)."""
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple HDF5 datasets into one.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to source HDF5 files.")
    parser.add_argument("--output", required=True, help="Path to the merged HDF5 file.")
    parser.add_argument(
        "--no_shuffle_demos",
        action="store_true",
        help="Keep source-file order instead of shuffling demos before writing.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Seed used when shuffling demos during merge.",
    )
    args = parser.parse_args()

    for p in args.inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input dataset not found: {p}")

    merge_hdf5_files(
        args.inputs,
        args.output,
        shuffle_demos=not args.no_shuffle_demos,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()
