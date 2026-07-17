"""Episode-aware command-and-skill dataset for the LocoDiff baseline."""

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
from .conditioning import (
    CONDITION_MODE_COMMAND_SKILL,
    CONDITION_MODE_VELOCITY_HEIGHT,
    goal_dim_for,
    validate_condition_mode,
)
from .obs_utils import PROPRIO_DIM, delayed_io_windows, read_proprio_vector
from .symmetry import apply_symmetry, symmetry_count

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCHEMA_VERSION = 4
SKILL_NAMES = ("walk", "crouch")
EXPECTED_CONVENTION_PREFIX = "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]"
REQUIRED_OBS = (
    "joint_pos",
    "joint_vel",
    "base_lin_vel",
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
    conditions: np.ndarray


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
        dataset_skills = tuple(_decode(value) for value in data.attrs.get("skill_names", ()))
        if not set(SKILL_NAMES).issubset(dataset_skills):
            raise ValueError(f"{path}: expected walk and crouch demonstrations; got {dataset_skills}.")
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
            skills = tuple(_decode(value) for value in demo.attrs.get("skills_sequence", ()))
            if len(skills) != 1 or skills[0] not in SKILL_NAMES:
                raise ValueError(
                    f"{path}/{demo_name}: each LocoDiff expert episode must have exactly one "
                    f"known skill, got {skills}."
                )
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


class LocoDiffCommandSkillDataset(Dataset):
    """Return state history, a versioned goal and a future action trajectory.

    At anchor ``t`` the target is ``a[t:t+P]``.  The conditioning is the state
    history and either the paper-faithful ``[vx, vy, wz, walk, crouch]`` goal or
    the continuous-height ablation ``[vx, vy, wz, desired_base_height]``. There
    is no reward, return or hindsight term.
    """

    def __init__(
        self,
        hdf5_paths: list[str] | tuple[str, ...],
        *,
        history: int = 8,
        prediction_horizon: int = 16,
        execution_offset: int = 0,
        step_stride: int = 1,
        symmetry_mode: str = "none",
        condition_mode: str = CONDITION_MODE_COMMAND_SKILL,
    ) -> None:
        if history < 1:
            raise ValueError("history must be >= 1.")
        if execution_offset != 0:
            raise ValueError("LocoDiff predicts a purely future trajectory; execution_offset must be 0.")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be >= 1.")
        if step_stride < 1:
            raise ValueError("step_stride must be >= 1.")

        self.history = history
        self.prediction_horizon = prediction_horizon
        self.execution_offset = execution_offset
        self.future_horizon = prediction_horizon
        self.step_stride = step_stride
        self.symmetry_mode = symmetry_mode
        self.condition_mode = validate_condition_mode(condition_mode)
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
                if self.condition_mode == CONDITION_MODE_COMMAND_SKILL:
                    skill_name = _decode(demo.attrs["skills_sequence"][0])
                    skill_one_hot = np.zeros((len(demo["actions"]), len(SKILL_NAMES)), dtype=np.float32)
                    skill_one_hot[:, SKILL_NAMES.index(skill_name)] = 1.0
                    conditions = np.concatenate(
                        [obs["command_speed"][:].astype(np.float32), skill_one_hot], axis=-1,
                    )
                else:
                    conditions = np.concatenate(
                        [obs["command_speed"][:].astype(np.float32), obs["desired_base_height"][:]], axis=-1,
                    ).astype(np.float32)
                demo_idx = len(self.demos)
                self.demos.append(
                    DemoSequence(
                        source_file=resolved,
                        demo_name=demo_name,
                        proprio=read_proprio_vector(obs),
                        actions=demo["actions"][:].astype(np.float32),
                        conditions=conditions,
                    )
                )
                first_anchor = self.history
                last_anchor_exclusive = len(demo["actions"]) - self.future_horizon + 1
                for anchor in range(first_anchor, last_anchor_exclusive, self.step_stride):
                    self.samples.append(CommandSample(demo_idx, anchor))

    def _raw_sample(self, sample: CommandSample) -> tuple[torch.Tensor, ...]:
        demo = self.demos[sample.demo_idx]
        proprio, action_hist = delayed_io_windows(
            demo.proprio, demo.actions, sample.anchor_step, self.history
        )
        condition_hist = demo.conditions[sample.anchor_step - self.history : sample.anchor_step]
        target_start = sample.anchor_step
        target_end = target_start + self.prediction_horizon
        return (
            torch.from_numpy(proprio.copy()),
            torch.from_numpy(action_hist.copy()),
            torch.from_numpy(condition_hist.copy()),
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
        proprio, actions, conditions = self._sample_rows(demos, max_stats_samples, seed)
        if self._symmetry_count > 1:
            proprio, actions, conditions = self._augment_stats(proprio, actions, conditions)
        action_min, action_max = self._exact_action_range(demos)
        return build_stats_with_action_range(proprio, conditions, action_min, action_max)

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
        proprio, actions, conditions = [], [], []
        for demo_idx in np.unique(demo_ids):
            local = chosen[demo_ids == demo_idx] - starts[demo_idx]
            demo = demos[int(demo_idx)]
            proprio.append(demo.proprio[local])
            actions.append(demo.actions[local])
            conditions.append(demo.conditions[local])
        return np.concatenate(proprio), np.concatenate(actions), np.concatenate(conditions)

    def _exact_action_range(self, demos: list[DemoSequence]) -> tuple[np.ndarray, np.ndarray]:
        action_min = np.full(12, np.inf, dtype=np.float32)
        action_max = np.full(12, -np.inf, dtype=np.float32)
        for demo in demos:
            actions = torch.from_numpy(demo.actions)
            proprio = torch.zeros((len(actions), PROPRIO_DIM), dtype=actions.dtype)
            conditions = torch.zeros((len(actions), goal_dim_for(self.condition_mode)), dtype=actions.dtype)
            for index in range(self._symmetry_count):
                _, _, _, transformed = apply_symmetry(
                    proprio,
                    actions,
                    conditions,
                    actions,
                    index=index,
                    mode=self.symmetry_mode,
                )
                values = transformed.numpy()
                action_min = np.minimum(action_min, values.min(axis=0))
                action_max = np.maximum(action_max, values.max(axis=0))
        return action_min, action_max

    def _augment_stats(
        self, proprio: np.ndarray, actions: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_t, a_t, c_t = torch.from_numpy(proprio), torch.from_numpy(actions), torch.from_numpy(conditions)
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
        proprio, action_hist, conditions, actions = self._raw_sample(self.samples[sample_index])
        proprio, action_hist, conditions, actions = apply_symmetry(
            proprio,
            action_hist,
            conditions,
            actions,
            index=symmetry_index,
            mode=self.symmetry_mode,
        )
        return {
            "proprio_hist": proprio,
            "action_hist": action_hist,
            "goal_hist": conditions,
            "actions": actions,
        }


# Kept as an import alias for small downstream scripts. New checkpoints use the
# schema-v4 name and are not compatible with the old height-conditioned model.
DiffuseLocoCommandDataset = LocoDiffCommandSkillDataset
