"""HDF5 dataset with hindsight relabeling for Solo12 Diffusion Policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import cumulative_xy_lengths, local_waypoints_by_path_distance, relative_yaw, transform_point_to_local, transform_point_to_yaw_frame

from .normalization import NormalizerStats, build_stats, normalize_minmax, normalize_zscore
from .symmetry import apply_symmetry, symmetry_count

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

EXPECTED_CONVENTION_PREFIX = "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]"
OBS_KEYS = ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "last_action", "root_pos_w", "root_quat_w")
OBS_DIM = 42
GOAL_DIM = 11
ACTION_DIM = 12


@dataclass(frozen=True)
class DemoSequence:
    obs: np.ndarray
    actions: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    cumulative_xy: np.ndarray
    skill_idx: np.ndarray | None


@dataclass(frozen=True)
class HindsightSample:
    demo_idx: int
    step: int
    end_step: int


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _demo_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def _read_obs_vector(obs_group: h5py.Group) -> np.ndarray:
    pieces = [
        obs_group["joint_pos"][:],
        obs_group["joint_vel"][:],
        obs_group["base_ang_vel"][:],
        obs_group["projected_gravity"][:],
        obs_group["last_action"][:],
    ]
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def _validate_hdf5(path: str) -> None:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{path}: missing 'data' group.")
        data = f["data"]
        convention = data.attrs.get("convention", None)
        if convention is None:
            raise ValueError(f"{path}: missing data.attrs['convention']; regenerate or merge with current scripts.")
        if not _decode_attr(convention).startswith(EXPECTED_CONVENTION_PREFIX):
            raise ValueError(f"{path}: unsupported convention {convention!r}.")
        if "skill_names" not in data.attrs:
            raise ValueError(f"{path}: missing data.attrs['skill_names'].")
        if not data.keys():
            raise ValueError(f"{path}: no demos found.")

        for demo_name in data:
            demo = data[demo_name]
            if "obs" not in demo:
                raise KeyError(f"{path}/{demo_name}: missing 'obs' group.")
            obs = demo["obs"]
            missing = [key for key in OBS_KEYS if key not in obs]
            if missing:
                raise KeyError(f"{path}/{demo_name}: missing obs keys {missing}.")
            for key in ("actions", "dones", "skill_idx"):
                if key not in demo:
                    raise KeyError(f"{path}/{demo_name}: missing '{key}'.")
            length = demo["actions"].shape[0]
            for key in ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "last_action"):
                if obs[key].shape[0] != length:
                    raise ValueError(f"{path}/{demo_name}: obs/{key} length mismatch.")


class LocomotionHindsightDataset(Dataset):
    """Load merged HDF5 demonstrations and produce normalized DDPM samples.

    The history is current-inclusive: sample ``t`` returns observations
    ``[t - history + 1, ..., t]`` and actions ``[t, ..., t + action_horizon - 1]``.
    This matches the aligned HDF5 convention where ``obs[t]`` is the state before
    executing ``actions[t]``.
    """

    def __init__(
        self,
        hdf5_paths: list[str] | tuple[str, ...],
        *,
        history: int = 8,
        action_horizon: int = 4,
        min_segment_steps: int = 50,
        max_segment_steps: int = 150,
        segment_stride: int = 10,
        dt: float = 0.02,
        waypoint_distances: tuple[float, float, float] = (0.4, 0.8, 1.2),
        v_req_clip: float = 2.0,
        symmetry_mode: str = "none",
        max_stats_samples: int = 20000,
        normalizer_stats: NormalizerStats | None = None,
    ):
        if history < 1:
            raise ValueError("history must be >= 1.")
        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1.")
        if min_segment_steps < action_horizon:
            raise ValueError("min_segment_steps must be >= action_horizon.")
        if max_segment_steps < min_segment_steps:
            raise ValueError("max_segment_steps must be >= min_segment_steps.")

        self.history = history
        self.action_horizon = action_horizon
        self.min_segment_steps = min_segment_steps
        self.max_segment_steps = max_segment_steps
        self.segment_stride = segment_stride
        self.dt = dt
        self.waypoint_distances = waypoint_distances
        self.v_req_clip = v_req_clip
        self.symmetry_mode = symmetry_mode
        self._symmetry_count = symmetry_count(symmetry_mode)

        self.demos: list[DemoSequence] = []
        self.samples: list[HindsightSample] = []
        self.skill_names: list[str] = []
        self.source_files = [str(Path(p)) for p in hdf5_paths]

        for path in hdf5_paths:
            self._load_file(path)
        if not self.samples:
            raise ValueError("No valid hindsight samples. Check demo length and segment settings.")

        self.normalizer_stats = normalizer_stats or self._build_normalizer_stats(max_stats_samples)

    def _load_file(self, path: str) -> None:
        _validate_hdf5(path)
        with h5py.File(path, "r") as f:
            data = f["data"]
            if not self.skill_names:
                raw_names = data.attrs["skill_names"]
                self.skill_names = [s.decode() if isinstance(s, bytes) else str(s) for s in raw_names]

            for demo_name in sorted(data.keys(), key=_demo_sort_key):
                demo = data[demo_name]
                obs_group = demo["obs"]
                obs = _read_obs_vector(obs_group)
                actions = demo["actions"][:].astype(np.float32)
                root_pos_w = obs_group["root_pos_w"][:].astype(np.float32)
                root_quat_w = obs_group["root_quat_w"][:].astype(np.float32)
                skill_idx = demo["skill_idx"][:].astype(np.int16) if "skill_idx" in demo else None
                demo_idx = len(self.demos)
                self.demos.append(
                    DemoSequence(
                        obs=obs,
                        actions=actions,
                        root_pos_w=root_pos_w,
                        root_quat_w=root_quat_w,
                        cumulative_xy=cumulative_xy_lengths(root_pos_w),
                        skill_idx=skill_idx,
                    )
                )
                self._index_demo(demo_idx, len(actions))

    def _index_demo(self, demo_idx: int, length: int) -> None:
        first_step = self.history - 1
        last_action_start = length - self.action_horizon
        for step in range(first_step, last_action_start + 1):
            for segment_steps in range(self.min_segment_steps, self.max_segment_steps + 1, self.segment_stride):
                end_step = step + segment_steps
                if end_step < length:
                    self.samples.append(HindsightSample(demo_idx, step, end_step))

    def _goal_for_step(self, demo: DemoSequence, step: int, end_step: int) -> np.ndarray:
        pos = demo.root_pos_w
        quat = demo.root_quat_w
        origin_w = pos[step]
        quat_step = quat[step]
        target_w = pos[end_step]

        waypoints = local_waypoints_by_path_distance(
            pos,
            step,
            demo.cumulative_xy,
            origin_w,
            quat_step,
            distances=self.waypoint_distances,
            end_idx=end_step,
        )
        target_rel = transform_point_to_yaw_frame(target_w, origin_w, quat_step)

        dyaw = relative_yaw(quat_step, quat[end_step])
        path_length_remaining = demo.cumulative_xy[end_step] - demo.cumulative_xy[step]
        time_remaining = max((end_step - step) * self.dt, self.dt)
        v_req = np.clip(path_length_remaining / (time_remaining + 1.0e-3), 0.0, self.v_req_clip)

        return np.asarray(
            [
                *waypoints.tolist(),
                float(target_rel[0]),
                float(target_rel[1]),
                float(target_rel[2]),
                float(dyaw),
                float(v_req),
            ],
            dtype=np.float32,
        )

    def _goal_history(self, sample: HindsightSample) -> np.ndarray:
        demo = self.demos[sample.demo_idx]
        start = sample.step - self.history + 1
        return np.stack([self._goal_for_step(demo, step, sample.end_step) for step in range(start, sample.step + 1)])

    def _raw_sample(self, sample: HindsightSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        demo = self.demos[sample.demo_idx]
        obs_hist = demo.obs[sample.step - self.history + 1 : sample.step + 1]
        goal_hist = self._goal_history(sample)
        actions = demo.actions[sample.step : sample.step + self.action_horizon]
        return (
            torch.from_numpy(obs_hist.astype(np.float32)),
            torch.from_numpy(goal_hist.astype(np.float32)),
            torch.from_numpy(actions.astype(np.float32)),
        )

    def _build_normalizer_stats(self, max_stats_samples: int) -> NormalizerStats:
        obs_values = np.concatenate([demo.obs for demo in self.demos], axis=0)
        action_values = np.concatenate([demo.actions for demo in self.demos], axis=0)

        rng = np.random.default_rng(0)
        sample_count = min(max_stats_samples, len(self.samples))
        sample_indices = rng.choice(len(self.samples), size=sample_count, replace=False)
        goal_values = np.concatenate([self._goal_history(self.samples[int(i)]) for i in sample_indices], axis=0)

        if self._symmetry_count > 1:
            obs_values, goal_values, action_values = self._augment_stats_arrays(obs_values, goal_values, action_values)

        return build_stats(obs_values, goal_values, action_values)

    def _augment_stats_arrays(
        self,
        obs_values: np.ndarray,
        goal_values: np.ndarray,
        action_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.from_numpy(obs_values)
        goal_t = torch.from_numpy(goal_values)
        actions_t = torch.from_numpy(action_values)
        obs_aug = []
        goal_aug = []
        action_aug = []
        for idx in range(self._symmetry_count):
            obs_i, goal_i, action_i = apply_symmetry(obs_t, goal_t, actions_t, index=idx, mode=self.symmetry_mode)
            obs_aug.append(obs_i.numpy())
            goal_aug.append(goal_i.numpy())
            action_aug.append(action_i.numpy())
        return np.concatenate(obs_aug), np.concatenate(goal_aug), np.concatenate(action_aug)

    def get_normalizer_stats(self) -> NormalizerStats:
        return self.normalizer_stats

    def __len__(self) -> int:
        return len(self.samples) * self._symmetry_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_index = index // self._symmetry_count
        symmetry_index = index % self._symmetry_count
        obs_hist, goal_hist, actions = self._raw_sample(self.samples[sample_index])
        obs_hist, goal_hist, actions = apply_symmetry(
            obs_hist,
            goal_hist,
            actions,
            index=symmetry_index,
            mode=self.symmetry_mode,
        )

        return {
            "obs_hist": obs_hist,
            "goal_hist": goal_hist,
            "actions": actions,
        }

