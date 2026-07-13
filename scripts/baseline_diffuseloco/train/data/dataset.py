"""Command-conditioned, episode-aware dataset for the DiffuseLoco baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import NormalizerStats, build_stats_with_action_range
from .obs_utils import PROPRIO_DIM, delayed_io_windows, read_proprio_vector
from .symmetry import apply_symmetry, symmetry_count

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCHEMA_VERSION = 3
EXPECTED_CONVENTION_PREFIX = "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]"
REQUIRED_OBS = (
    "joint_pos",
    "joint_vel",
    "base_ang_vel",
    "projected_gravity",
    "last_action",
    "command_speed",
    "desired_base_height",
)


@dataclass(frozen=True)
class DemoSequence:
    source_file: str
    demo_name: str
    proprio: np.ndarray
    actions: np.ndarray
    commands: np.ndarray


@dataclass(frozen=True)
class CommandSample:
    demo_idx: int
    anchor_step: int


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_")[-1]), name
    except ValueError:
        return 0, name


def _validate_file(path: str) -> None:
    with h5py.File(path, "r") as file:
        if "data" not in file:
            raise KeyError(f"{path}: missing data group.")
        data = file["data"]
        condition_schema = _decode(data.attrs.get("condition_schema", ""))
        if condition_schema != "velocity_xyyaw_plus_desired_base_height_v1":
            raise ValueError(
                f"{path}: expected velocity-plus-height condition schema; got {condition_schema!r}. "
                "Regenerate the dataset with the current collector."
            )
        convention = data.attrs.get("convention")
        if convention is None or not _decode(convention).startswith(EXPECTED_CONVENTION_PREFIX):
            raise ValueError(f"{path}: unsupported alignment convention {convention!r}.")
        if not np.isclose(float(data.attrs.get("control_rate_hz", 0.0)), 50.0):
            raise ValueError(f"{path}: baseline requires control_rate_hz=50.")
        for demo_name in data:
            demo = data[demo_name]
            if "obs" not in demo or "actions" not in demo or "dones" not in demo:
                raise KeyError(f"{path}/{demo_name}: incomplete demo.")
            obs = demo["obs"]
            missing = [key for key in REQUIRED_OBS if key not in obs]
            if missing:
                raise KeyError(f"{path}/{demo_name}: missing obs keys {missing}.")
            length = int(demo["actions"].shape[0])
            if length == 0:
                raise ValueError(f"{path}/{demo_name}: empty demonstration.")
            if any(int(obs[key].shape[0]) != length for key in REQUIRED_OBS):
                raise ValueError(f"{path}/{demo_name}: observation length mismatch.")
            if obs["command_speed"].ndim != 2 or obs["command_speed"].shape[1] != 3:
                raise ValueError(f"{path}/{demo_name}: command_speed must have shape (T, 3).")
            if obs["desired_base_height"].ndim != 2 or obs["desired_base_height"].shape[1] != 1:
                raise ValueError(f"{path}/{demo_name}: desired_base_height must have shape (T, 1).")
            dones = demo["dones"][:]
            if len(dones) != length or (length and not bool(dones[-1])) or np.any(dones[:-1]):
                raise ValueError(f"{path}/{demo_name}: invalid dones convention.")
            actions_ds = demo["actions"]
            last_action_ds = obs["last_action"]
            for start in range(0, length, 65_536):
                end = min(start + 65_536, length)
                actions = actions_ds[start:end]
                if not np.isfinite(actions).all():
                    raise ValueError(f"{path}/{demo_name}: non-finite actions.")
                for key in REQUIRED_OBS:
                    if not np.isfinite(obs[key][start:end]).all():
                        raise ValueError(f"{path}/{demo_name}: non-finite obs/{key}.")
                expected = np.empty_like(actions)
                if start == 0:
                    expected[0] = 0.0
                    expected[1:] = actions[:-1]
                else:
                    expected[0] = actions_ds[start - 1]
                    expected[1:] = actions[:-1]
                if not np.allclose(last_action_ds[start:end], expected, atol=1.0e-6, rtol=0.0):
                    raise ValueError(f"{path}/{demo_name}: last_action is not actions[t-1].")


class DiffuseLocoCommandDataset(Dataset):
    """Return the exact delayed conditioning and 16-token action trajectory.

    At anchor ``t`` the target is ``a[t-H:t+F]`` and token ``H`` is the action
    executed at deployment.  Commands are read from the expert data; no path,
    terminal pose, height relabeling or hindsight is present in this baseline.
    """

    def __init__(
        self,
        hdf5_paths: list[str] | tuple[str, ...],
        *,
        history: int = 8,
        prediction_horizon: int = 16,
        execution_offset: int = 8,
        step_stride: int = 1,
        symmetry_mode: str = "none",
    ) -> None:
        if history < 1:
            raise ValueError("history must be >= 1.")
        if execution_offset != history:
            raise ValueError("DiffuseLoco schema v2 requires execution_offset == history.")
        if prediction_horizon <= execution_offset:
            raise ValueError("prediction_horizon must include future actions.")
        if step_stride < 1:
            raise ValueError("step_stride must be >= 1.")

        self.history = history
        self.prediction_horizon = prediction_horizon
        self.execution_offset = execution_offset
        self.future_horizon = prediction_horizon - execution_offset
        self.step_stride = step_stride
        self.symmetry_mode = symmetry_mode
        self._symmetry_count = symmetry_count(symmetry_mode)
        self.source_files = [str(Path(path).resolve()) for path in hdf5_paths]
        self.demos: list[DemoSequence] = []
        self.samples: list[CommandSample] = []

        for path in hdf5_paths:
            self._load(path)
        if not self.samples:
            raise ValueError("No valid command windows in the selected datasets.")

    def _load(self, path: str) -> None:
        _validate_file(path)
        resolved = str(Path(path).resolve())
        with h5py.File(path, "r") as file:
            for demo_name in sorted(file["data"].keys(), key=_sort_key):
                demo = file["data"][demo_name]
                obs = demo["obs"]
                demo_idx = len(self.demos)
                self.demos.append(
                    DemoSequence(
                        source_file=resolved,
                        demo_name=demo_name,
                        proprio=read_proprio_vector(obs),
                        actions=demo["actions"][:].astype(np.float32),
                        commands=np.concatenate(
                            [obs["command_speed"][:], obs["desired_base_height"][:]], axis=-1,
                        ).astype(np.float32),
                    )
                )
                first_anchor = self.history + 1
                last_anchor_exclusive = len(demo["actions"]) - self.future_horizon + 1
                for anchor in range(first_anchor, last_anchor_exclusive, self.step_stride):
                    self.samples.append(CommandSample(demo_idx, anchor))

    def _raw_sample(self, sample: CommandSample) -> tuple[torch.Tensor, ...]:
        demo = self.demos[sample.demo_idx]
        proprio, action_hist = delayed_io_windows(
            demo.proprio, demo.actions, sample.anchor_step, self.history
        )
        command_hist = demo.commands[sample.anchor_step - self.history : sample.anchor_step]
        target_start = sample.anchor_step - self.execution_offset
        target_end = target_start + self.prediction_horizon
        return (
            torch.from_numpy(proprio.copy()),
            torch.from_numpy(action_hist.copy()),
            torch.from_numpy(command_hist.copy()),
            torch.from_numpy(demo.actions[target_start:target_end].copy()),
        )

    def sample_indices_for_demos(self, demo_indices: Iterable[int]) -> list[int]:
        selected = set(int(index) for index in demo_indices)
        output: list[int] = []
        for sample_idx, sample in enumerate(self.samples):
            if sample.demo_idx in selected:
                base = sample_idx * self._symmetry_count
                output.extend(range(base, base + self._symmetry_count))
        if not output:
            raise ValueError("Episode split produced no command windows.")
        return output

    def build_normalizer_stats(
        self,
        demo_indices: Iterable[int],
        *,
        max_stats_samples: int = 20_000,
        seed: int = 0,
    ) -> NormalizerStats:
        selected = set(int(index) for index in demo_indices)
        demos = [demo for index, demo in enumerate(self.demos) if index in selected]
        if not demos:
            raise ValueError("Cannot fit normalizer without training episodes.")
        proprio, actions, commands = self._sample_rows(demos, max_stats_samples, seed)
        if self._symmetry_count > 1:
            proprio, actions, commands = self._augment_stats(proprio, actions, commands)
        action_min, action_max = self._exact_action_range(demos)
        return build_stats_with_action_range(proprio, commands, action_min, action_max)

    @staticmethod
    def _sample_rows(
        demos: list[DemoSequence], max_samples: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max_samples < 1:
            raise ValueError("max_stats_samples must be >= 1.")
        lengths = np.asarray([len(demo.actions) for demo in demos], dtype=np.int64)
        boundaries = np.cumsum(lengths)
        count = min(int(max_samples), int(boundaries[-1]))
        chosen = np.sort(np.random.default_rng(seed).choice(boundaries[-1], count, replace=False))
        demo_ids = np.searchsorted(boundaries, chosen, side="right")
        starts = np.concatenate(([0], boundaries[:-1]))
        proprio, actions, commands = [], [], []
        for demo_idx in np.unique(demo_ids):
            local = chosen[demo_ids == demo_idx] - starts[demo_idx]
            demo = demos[int(demo_idx)]
            proprio.append(demo.proprio[local])
            actions.append(demo.actions[local])
            commands.append(demo.commands[local])
        return np.concatenate(proprio), np.concatenate(actions), np.concatenate(commands)

    def _exact_action_range(self, demos: list[DemoSequence]) -> tuple[np.ndarray, np.ndarray]:
        action_min = np.full(12, np.inf, dtype=np.float32)
        action_max = np.full(12, -np.inf, dtype=np.float32)
        for demo in demos:
            actions = torch.from_numpy(demo.actions)
            proprio = torch.zeros((len(actions), PROPRIO_DIM), dtype=actions.dtype)
            commands = torch.zeros((len(actions), 3), dtype=actions.dtype)
            for index in range(self._symmetry_count):
                _, _, _, transformed = apply_symmetry(
                    proprio,
                    actions,
                    commands,
                    actions,
                    index=index,
                    mode=self.symmetry_mode,
                )
                values = transformed.numpy()
                action_min = np.minimum(action_min, values.min(axis=0))
                action_max = np.maximum(action_max, values.max(axis=0))
        return action_min, action_max

    def _augment_stats(
        self, proprio: np.ndarray, actions: np.ndarray, commands: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_t, a_t, c_t = torch.from_numpy(proprio), torch.from_numpy(actions), torch.from_numpy(commands)
        p_all, a_all, c_all = [], [], []
        for index in range(self._symmetry_count):
            p, ah, c, target = apply_symmetry(
                p_t, a_t, c_t, a_t, index=index, mode=self.symmetry_mode
            )
            p_all.append(p.numpy())
            a_all.append(target.numpy())
            c_all.append(c.numpy())
        return np.concatenate(p_all), np.concatenate(a_all), np.concatenate(c_all)

    def manifest_for_demos(self, demo_indices: Iterable[int]) -> list[dict[str, Any]]:
        return [
            {
                "demo_index": int(index),
                "source_file": self.demos[int(index)].source_file,
                "demo_name": self.demos[int(index)].demo_name,
                "length": int(len(self.demos[int(index)].actions)),
            }
            for index in demo_indices
        ]

    def __len__(self) -> int:
        return len(self.samples) * self._symmetry_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_index = index // self._symmetry_count
        symmetry_index = index % self._symmetry_count
        proprio, action_hist, commands, actions = self._raw_sample(self.samples[sample_index])
        proprio, action_hist, commands, actions = apply_symmetry(
            proprio,
            action_hist,
            commands,
            actions,
            index=symmetry_index,
            mode=self.symmetry_mode,
        )
        return {
            "proprio_hist": proprio,
            "action_hist": action_hist,
            "goal_hist": commands,
            "actions": actions,
        }
