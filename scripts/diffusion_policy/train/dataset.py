"""Episode-aware hindsight dataset for the spatial Solo12 diffusion policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import cumulative_xy_lengths
from .goal_builder import build_goal_vector
from .normalization import NormalizerStats, build_stats_with_action_range
from .obs_utils import GOAL_DIM, PROPRIO_DIM, delayed_io_windows, read_proprio_vector
from .symmetry import apply_symmetry, symmetry_count

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCHEMA_VERSION = 2
EXPECTED_CONVENTION_PREFIX = "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]"
HDF5_OBS_KEYS = (
    "joint_pos",
    "joint_vel",
    "base_ang_vel",
    "projected_gravity",
    "last_action",
    "root_pos_w",
    "root_quat_w",
    "command_speed",
)


@dataclass(frozen=True)
class DemoSequence:
    source_file: str
    demo_name: str
    proprio: np.ndarray
    actions: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    cumulative_xy: np.ndarray
    skill_idx: np.ndarray | None


@dataclass(frozen=True)
class HindsightSample:
    demo_idx: int
    anchor_step: int


def _decode_attr(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_")[-1]), name
    except ValueError:
        return 0, name


def _validate_hdf5(path: str) -> None:
    with h5py.File(path, "r") as file:
        if "data" not in file:
            raise KeyError(f"{path}: missing 'data' group.")
        data = file["data"]
        convention = data.attrs.get("convention")
        if convention is None or not _decode_attr(convention).startswith(EXPECTED_CONVENTION_PREFIX):
            raise ValueError(f"{path}: unsupported or missing alignment convention {convention!r}.")
        if "skill_names" not in data.attrs:
            raise ValueError(f"{path}: missing data.attrs['skill_names'].")
        rate = float(data.attrs.get("control_rate_hz", 0.0))
        if not np.isclose(rate, 50.0):
            raise ValueError(f"{path}: expected control_rate_hz=50, got {rate}.")
        if not data.keys():
            raise ValueError(f"{path}: no demonstrations found.")

        for demo_name in data:
            demo = data[demo_name]
            if "obs" not in demo:
                raise KeyError(f"{path}/{demo_name}: missing obs group.")
            obs = demo["obs"]
            missing = [key for key in HDF5_OBS_KEYS if key not in obs]
            if missing:
                raise KeyError(f"{path}/{demo_name}: missing obs keys {missing}.")
            if "actions" not in demo or "dones" not in demo:
                raise KeyError(f"{path}/{demo_name}: missing actions or dones.")
            length = int(demo["actions"].shape[0])
            for key in HDF5_OBS_KEYS:
                if int(obs[key].shape[0]) != length:
                    raise ValueError(f"{path}/{demo_name}: obs/{key} length mismatch.")
            if int(demo["dones"].shape[0]) != length:
                raise ValueError(f"{path}/{demo_name}: dones length mismatch.")
            if length and not bool(demo["dones"][-1]):
                raise ValueError(f"{path}/{demo_name}: final dones entry must be True.")


class SpatialHindsightDataset(Dataset):
    """Build delayed conditioning and a full DiffuseLoco-style trajectory.

    At anchor ``t``:

    - proprio condition: ``s[t-H:t]`` (latest state is ``s[t-1]``);
    - action condition: ``a[t-H-1:t-1]`` (latest action is ``a[t-2]``);
    - goal condition: rolling achieved-future goals aligned with each state;
    - denoising target: ``a[t-H:t+F]``;
    - deployment executes target token ``H``, corresponding to ``a[t]``.

    Rolling hindsight matches deployment: each historical goal is the plan that
    was available with its corresponding historical observation.  It avoids the
    previous train/deploy mismatch where every history token pointed to one fixed
    endpoint while deployment stored moving lookahead goals.
    """

    def __init__(
        self,
        hdf5_paths: list[str] | tuple[str, ...],
        *,
        history: int = 8,
        prediction_horizon: int = 16,
        execution_offset: int = 8,
        goal_horizon_steps: int = 100,
        step_stride: int = 1,
        dt: float = 0.02,
        waypoint_distances: tuple[float, float, float] = (0.4, 0.8, 1.2),
        v_req_clip: float = 2.0,
        symmetry_mode: str = "none",
    ) -> None:
        if history < 1:
            raise ValueError("history must be >= 1.")
        if execution_offset != history:
            raise ValueError("Schema v2 requires execution_offset == history.")
        if prediction_horizon <= execution_offset:
            raise ValueError("prediction_horizon must include at least one future action.")
        if goal_horizon_steps < 1:
            raise ValueError("goal_horizon_steps must be >= 1.")
        if step_stride < 1:
            raise ValueError("step_stride must be >= 1.")

        self.history = history
        self.prediction_horizon = prediction_horizon
        self.execution_offset = execution_offset
        self.future_horizon = prediction_horizon - execution_offset
        self.goal_horizon_steps = goal_horizon_steps
        self.step_stride = step_stride
        self.dt = dt
        self.waypoint_distances = waypoint_distances
        self.v_req_clip = v_req_clip
        self.symmetry_mode = symmetry_mode
        self._symmetry_count = symmetry_count(symmetry_mode)

        self.demos: list[DemoSequence] = []
        self.samples: list[HindsightSample] = []
        self.skill_names: list[str] = []
        self.source_files = [str(Path(path).resolve()) for path in hdf5_paths]

        for path in hdf5_paths:
            self._load_file(path)
        if not self.samples:
            raise ValueError("No valid samples: check episode lengths, history and goal horizon.")

    def _load_file(self, path: str) -> None:
        _validate_hdf5(path)
        resolved = str(Path(path).resolve())
        with h5py.File(path, "r") as file:
            data = file["data"]
            names = [_decode_attr(value) for value in data.attrs["skill_names"]]
            if not self.skill_names:
                self.skill_names = names
            elif self.skill_names != names:
                raise ValueError(f"{path}: skill_names {names} do not match {self.skill_names}.")

            for demo_name in sorted(data.keys(), key=_demo_sort_key):
                demo = data[demo_name]
                obs = demo["obs"]
                demo_idx = len(self.demos)
                self.demos.append(
                    DemoSequence(
                        source_file=resolved,
                        demo_name=demo_name,
                        proprio=read_proprio_vector(obs),
                        actions=demo["actions"][:].astype(np.float32),
                        root_pos_w=obs["root_pos_w"][:].astype(np.float32),
                        root_quat_w=obs["root_quat_w"][:].astype(np.float32),
                        cumulative_xy=cumulative_xy_lengths(obs["root_pos_w"][:].astype(np.float32)),
                        skill_idx=demo["skill_idx"][:].astype(np.int16) if "skill_idx" in demo else None,
                    )
                )
                self._index_demo(demo_idx)

    def _index_demo(self, demo_idx: int) -> None:
        length = len(self.demos[demo_idx].actions)
        first_anchor = self.history + 1
        # Latest goal history token is at t-1 and looks goal_horizon_steps ahead.
        last_for_goal_exclusive = length - self.goal_horizon_steps + 1
        last_for_actions_exclusive = length - self.future_horizon + 1
        last_anchor_exclusive = min(last_for_goal_exclusive, last_for_actions_exclusive)
        for anchor in range(first_anchor, last_anchor_exclusive, self.step_stride):
            self.samples.append(HindsightSample(demo_idx, anchor))

    def _goal_for_state(self, demo: DemoSequence, state_step: int) -> np.ndarray:
        end_step = state_step + self.goal_horizon_steps
        return build_goal_vector(
            demo.root_pos_w,
            demo.cumulative_xy,
            state_step,
            end_step,
            demo.root_pos_w[state_step],
            demo.root_quat_w[state_step],
            quat_w=demo.root_quat_w,
            dt=self.dt,
            waypoint_distances=self.waypoint_distances,
            v_req_clip=self.v_req_clip,
        )

    def _goal_history(self, sample: HindsightSample) -> np.ndarray:
        demo = self.demos[sample.demo_idx]
        start = sample.anchor_step - self.history
        return np.stack([self._goal_for_state(demo, state_step) for state_step in range(start, sample.anchor_step)])

    def _raw_sample(self, sample: HindsightSample) -> tuple[torch.Tensor, ...]:
        demo = self.demos[sample.demo_idx]
        proprio_hist, action_hist = delayed_io_windows(
            demo.proprio, demo.actions, sample.anchor_step, self.history
        )
        target_start = sample.anchor_step - self.execution_offset
        target_end = target_start + self.prediction_horizon
        return (
            torch.from_numpy(proprio_hist.copy()),
            torch.from_numpy(action_hist.copy()),
            torch.from_numpy(self._goal_history(sample).astype(np.float32)),
            torch.from_numpy(demo.actions[target_start:target_end].copy()),
        )

    def sample_indices_for_demos(self, demo_indices: Iterable[int]) -> list[int]:
        selected = set(int(index) for index in demo_indices)
        indices: list[int] = []
        for sample_idx, sample in enumerate(self.samples):
            if sample.demo_idx in selected:
                base = sample_idx * self._symmetry_count
                indices.extend(range(base, base + self._symmetry_count))
        if not indices:
            raise ValueError("Episode split produced no windows.")
        return indices

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
        proprio, actions = self._sample_state_action_rows(demos, max_stats_samples, seed)
        candidate_positions = np.flatnonzero(
            np.fromiter((sample.demo_idx in selected for sample in self.samples), dtype=bool)
        )
        rng = np.random.default_rng(seed)
        count = min(max_stats_samples, len(candidate_positions))
        chosen = rng.choice(candidate_positions, size=count, replace=False)
        goals = np.concatenate([self._goal_history(self.samples[int(i)]) for i in chosen], axis=0)

        if self._symmetry_count > 1:
            proprio, actions, goals = self._augment_stats(proprio, actions, goals)
        action_min, action_max = self._exact_action_range(demos)
        return build_stats_with_action_range(proprio, goals, action_min, action_max)

    @staticmethod
    def _sample_state_action_rows(
        demos: list[DemoSequence], max_samples: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_samples < 1:
            raise ValueError("max_stats_samples must be >= 1.")
        lengths = np.asarray([len(demo.actions) for demo in demos], dtype=np.int64)
        boundaries = np.cumsum(lengths)
        count = min(int(max_samples), int(boundaries[-1]))
        chosen = np.sort(np.random.default_rng(seed).choice(boundaries[-1], count, replace=False))
        demo_ids = np.searchsorted(boundaries, chosen, side="right")
        starts = np.concatenate(([0], boundaries[:-1]))
        proprio, actions = [], []
        for demo_idx in np.unique(demo_ids):
            local = chosen[demo_ids == demo_idx] - starts[demo_idx]
            demo = demos[int(demo_idx)]
            proprio.append(demo.proprio[local])
            actions.append(demo.actions[local])
        return np.concatenate(proprio), np.concatenate(actions)

    def _exact_action_range(self, demos: list[DemoSequence]) -> tuple[np.ndarray, np.ndarray]:
        action_min = np.full(12, np.inf, dtype=np.float32)
        action_max = np.full(12, -np.inf, dtype=np.float32)
        for demo in demos:
            actions = torch.from_numpy(demo.actions)
            proprio = torch.zeros((len(actions), PROPRIO_DIM), dtype=actions.dtype)
            goals = torch.zeros((len(actions), GOAL_DIM), dtype=actions.dtype)
            for index in range(self._symmetry_count):
                _, _, _, transformed = apply_symmetry(
                    proprio,
                    actions,
                    goals,
                    actions,
                    index=index,
                    mode=self.symmetry_mode,
                )
                values = transformed.numpy()
                action_min = np.minimum(action_min, values.min(axis=0))
                action_max = np.maximum(action_max, values.max(axis=0))
        return action_min, action_max

    def _augment_stats(
        self, proprio: np.ndarray, actions: np.ndarray, goals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        proprio_t = torch.from_numpy(proprio)
        actions_t = torch.from_numpy(actions)
        goals_t = torch.from_numpy(goals)
        proprio_all, actions_all, goals_all = [], [], []
        for index in range(self._symmetry_count):
            p, a, g, _ = apply_symmetry(
                proprio_t, actions_t, goals_t, actions_t, index=index, mode=self.symmetry_mode
            )
            proprio_all.append(p.numpy())
            actions_all.append(a.numpy())
            goals_all.append(g.numpy())
        return np.concatenate(proprio_all), np.concatenate(actions_all), np.concatenate(goals_all)

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
        proprio, action_hist, goals, actions = self._raw_sample(self.samples[sample_index])
        proprio, action_hist, goals, actions = apply_symmetry(
            proprio,
            action_hist,
            goals,
            actions,
            index=symmetry_index,
            mode=self.symmetry_mode,
        )
        return {
            "proprio_hist": proprio,
            "action_hist": action_hist,
            "goal_hist": goals,
            "actions": actions,
        }


# Backward-compatible import name for scripts while rejecting old checkpoints via schema_version.
LocomotionHindsightDataset = SpatialHindsightDataset
