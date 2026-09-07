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

from ..conditioning.geometry import cumulative_xy_lengths
from ..conditioning.goal_builder import WAYPOINT_TIME_OFFSETS_S, build_goal_vector, goal_dimension
from .normalization import NormalizerStats, build_stats_with_action_range
from .obs_utils import PROPRIO_DIM, delayed_io_windows, read_proprio_vector
from .symmetry import apply_symmetry, symmetry_count

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

SCHEMA_VERSION = 3
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
REFERENCE_OBS_KEYS = (
    "reference_pos_w",
    "reference_yaw_w",
    "reference_command",
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
    desired_base_height: np.ndarray | None = None
    reference_pos_w: np.ndarray | None = None
    reference_yaw_w: np.ndarray | None = None
    reference_command_b: np.ndarray | None = None
    reference_cumulative_xy: np.ndarray | None = None
    guidance_pos_w: np.ndarray | None = None
    guidance_cumulative_xy: np.ndarray | None = None
    terminal_step: int | None = None


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


def _validate_hdf5(
    path: str,
    *,
    require_reference_path: bool,
    require_holonomic_reference: bool = False,
    require_path_guidance_reference: bool = False,
    require_desired_height_profile: bool = False,
) -> None:
    with h5py.File(path, "r") as file:
        if "data" not in file:
            raise KeyError(f"{path}: missing 'data' group.")
        data = file["data"]
        if require_holonomic_reference:
            route_profile = _decode_attr(data.attrs.get("route_profile", ""))
            reference_schema = _decode_attr(data.attrs.get("reference_schema", ""))
            if route_profile != "phase_a_holonomic" or reference_schema != "fixed_holonomic_se2_route_with_closed_loop_teacher_v3":
                raise ValueError(
                    f"{path}: holonomic_se2_32 requires phase_a_holonomic/v3 reference data, got "
                    f"route_profile={route_profile!r}, reference_schema={reference_schema!r}."
                )
        if require_path_guidance_reference:
            route_profile = _decode_attr(data.attrs.get("route_profile", ""))
            reference_schema = _decode_attr(data.attrs.get("reference_schema", ""))
            if (
                route_profile != "phase_a_path_guidance"
                or reference_schema != "fixed_waypoint_task_with_noisy_guidance_and_terminal_stop_v1"
            ):
                raise ValueError(
                    f"{path}: path_guidance_se2_36 requires phase_a_path_guidance/v1 data, got "
                    f"route_profile={route_profile!r}, reference_schema={reference_schema!r}."
                )
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
            if require_reference_path:
                missing_reference = [key for key in REFERENCE_OBS_KEYS if key not in obs]
                if missing_reference:
                    raise KeyError(
                        f"{path}/{demo_name}: reference goal source requires obs keys {missing_reference}."
                    )
            if require_path_guidance_reference and "guidance_pos_w" not in obs:
                raise KeyError(f"{path}/{demo_name}: path-guidance data require obs/guidance_pos_w.")
            if require_desired_height_profile and "desired_base_height" not in obs:
                raise KeyError(
                    f"{path}/{demo_name}: hindsight_geom_profile16 requires obs/desired_base_height."
                )
            length = int(demo["actions"].shape[0])
            if length == 0:
                raise ValueError(f"{path}/{demo_name}: empty demonstration.")
            for key in HDF5_OBS_KEYS:
                if int(obs[key].shape[0]) != length:
                    raise ValueError(f"{path}/{demo_name}: obs/{key} length mismatch.")
            if require_reference_path:
                for key in REFERENCE_OBS_KEYS:
                    if int(obs[key].shape[0]) != length:
                        raise ValueError(f"{path}/{demo_name}: obs/{key} length mismatch.")
            if require_path_guidance_reference and int(obs["guidance_pos_w"].shape[0]) != length:
                raise ValueError(f"{path}/{demo_name}: obs/guidance_pos_w length mismatch.")
            if require_desired_height_profile and int(obs["desired_base_height"].shape[0]) != length:
                raise ValueError(f"{path}/{demo_name}: obs/desired_base_height length mismatch.")
            dones = demo["dones"][:]
            if len(dones) != length or (length and not bool(dones[-1])) or np.any(dones[:-1]):
                raise ValueError(f"{path}/{demo_name}: dones must be false except at the final sample.")
            actions_ds = demo["actions"]
            last_action_ds = obs["last_action"]
            for start in range(0, length, 65_536):
                end = min(start + 65_536, length)
                actions = actions_ds[start:end]
                if not np.isfinite(actions).all():
                    raise ValueError(f"{path}/{demo_name}: non-finite actions.")
                for key in HDF5_OBS_KEYS:
                    if not np.isfinite(obs[key][start:end]).all():
                        raise ValueError(f"{path}/{demo_name}: non-finite obs/{key}.")
                if require_reference_path:
                    for key in REFERENCE_OBS_KEYS:
                        if not np.isfinite(obs[key][start:end]).all():
                            raise ValueError(f"{path}/{demo_name}: non-finite obs/{key}.")
                if require_path_guidance_reference and not np.isfinite(obs["guidance_pos_w"][start:end]).all():
                    raise ValueError(f"{path}/{demo_name}: non-finite obs/guidance_pos_w.")
                if require_desired_height_profile and not np.isfinite(
                    obs["desired_base_height"][start:end]
                ).all():
                    raise ValueError(f"{path}/{demo_name}: non-finite obs/desired_base_height.")
                expected = np.empty_like(actions)
                if start == 0:
                    expected[0] = 0.0
                    expected[1:] = actions[:-1]
                else:
                    expected[0] = actions_ds[start - 1]
                    expected[1:] = actions[:-1]
                if not np.allclose(last_action_ds[start:end], expected, atol=1.0e-6, rtol=0.0):
                    raise ValueError(f"{path}/{demo_name}: last_action is not actions[t-1].")
                quat_norm = np.linalg.norm(obs["root_quat_w"][start:end], axis=-1)
                if not np.allclose(quat_norm, 1.0, atol=1.0e-3, rtol=0.0):
                    raise ValueError(f"{path}/{demo_name}: root_quat_w is not normalized.")


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
        waypoint_time_offsets_s: tuple[float, float, float] = WAYPOINT_TIME_OFFSETS_S,
        v_req_clip: float = 2.0,
        goal_source: str = "achieved",
        goal_representation: str = "path11",
        include_padded_starts: bool = False,
        startup_sample_multiplier: int = 1,
        symmetry_mode: str = "none",
    ) -> None:
        if history < 1:
            raise ValueError("history must be >= 1.")
        if execution_offset != history:
            raise ValueError("Spatial schema v3 requires execution_offset == history.")
        if prediction_horizon <= execution_offset:
            raise ValueError("prediction_horizon must include at least one future action.")
        if goal_horizon_steps < 1:
            raise ValueError("goal_horizon_steps must be >= 1.")
        if step_stride < 1:
            raise ValueError("step_stride must be >= 1.")
        if goal_source not in {"achieved", "reference"}:
            raise ValueError("goal_source must be 'achieved' or 'reference'.")
        geometric_representations = {"hindsight_geom_avg12", "hindsight_geom_profile16"}
        if goal_representation in geometric_representations and goal_source != "achieved":
            raise ValueError(f"{goal_representation} requires goal_source='achieved'.")
        if goal_source != "reference" and goal_representation not in {"path11", *geometric_representations}:
            raise ValueError("SE(2) route representations require goal_source='reference'.")
        if startup_sample_multiplier < 1:
            raise ValueError("startup_sample_multiplier must be >= 1.")
        if not np.isclose(dt, 0.02):
            raise ValueError("Spatial schema v3 is fixed at 50 Hz and requires dt=0.02.")
        terminal_time_s = goal_horizon_steps * dt
        if max(waypoint_time_offsets_s) >= terminal_time_s:
            raise ValueError(
                "Temporal preview offsets must be earlier than goal_horizon_steps * dt "
                f"({terminal_time_s:.3f}s)."
            )

        self.history = history
        self.prediction_horizon = prediction_horizon
        self.execution_offset = execution_offset
        self.future_horizon = prediction_horizon - execution_offset
        self.goal_horizon_steps = goal_horizon_steps
        self.step_stride = step_stride
        self.dt = dt
        self.waypoint_time_offsets_s = tuple(float(value) for value in waypoint_time_offsets_s)
        self.v_req_clip = v_req_clip
        self.goal_source = goal_source
        self.goal_representation = goal_representation
        self.goal_dim = goal_dimension(goal_representation)
        self.include_padded_starts = bool(include_padded_starts)
        self.startup_sample_multiplier = int(startup_sample_multiplier)
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
        _validate_hdf5(
            path,
            require_reference_path=self.goal_source == "reference",
            require_holonomic_reference=self.goal_representation == "holonomic_se2_32",
            require_path_guidance_reference=self.goal_representation == "path_guidance_se2_36",
            require_desired_height_profile=self.goal_representation == "hindsight_geom_profile16",
        )
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
                reference_pos_w = (
                    obs["reference_pos_w"][:].astype(np.float32)
                    if self.goal_source == "reference"
                    else None
                )
                reference_yaw_w = (
                    obs["reference_yaw_w"][:].astype(np.float32).reshape(-1)
                    if self.goal_source == "reference"
                    else None
                )
                reference_command_b = (
                    obs["reference_command"][:].astype(np.float32)
                    if self.goal_source == "reference"
                    else None
                )
                guidance_pos_w = (
                    obs["guidance_pos_w"][:].astype(np.float32)
                    if self.goal_representation == "path_guidance_se2_36"
                    else None
                )
                desired_base_height = (
                    obs["desired_base_height"][:].astype(np.float32).reshape(-1)
                    if self.goal_representation == "hindsight_geom_profile16"
                    else None
                )
                terminal_step = None
                if self.goal_representation == "path_guidance_se2_36":
                    moving = np.flatnonzero(np.linalg.norm(reference_command_b, axis=-1) > 1.0e-4)
                    terminal_step = min(len(reference_command_b) - 1, int(moving[-1]) + 1) if len(moving) else 0
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
                        desired_base_height=desired_base_height,
                        reference_pos_w=reference_pos_w,
                        reference_yaw_w=reference_yaw_w,
                        reference_command_b=reference_command_b,
                        reference_cumulative_xy=(
                            cumulative_xy_lengths(reference_pos_w) if reference_pos_w is not None else None
                        ),
                        guidance_pos_w=guidance_pos_w,
                        guidance_cumulative_xy=(
                            cumulative_xy_lengths(guidance_pos_w) if guidance_pos_w is not None else None
                        ),
                        terminal_step=terminal_step,
                    )
                )
                self._index_demo(demo_idx)

    def _index_demo(self, demo_idx: int) -> None:
        length = len(self.demos[demo_idx].actions)
        first_anchor = 0 if self.include_padded_starts else self.history + 1
        # Latest goal history token is at t-1 and looks goal_horizon_steps ahead.
        last_for_goal_exclusive = (
            length
            if self.goal_representation == "path_guidance_se2_36"
            else length - self.goal_horizon_steps + 1
        )
        last_for_actions_exclusive = length - self.future_horizon + 1
        last_anchor_exclusive = min(last_for_goal_exclusive, last_for_actions_exclusive)
        for anchor in range(first_anchor, last_anchor_exclusive, self.step_stride):
            repeats = self.startup_sample_multiplier if anchor < self.history else 1
            self.samples.extend(HindsightSample(demo_idx, anchor) for _ in range(repeats))

    def _goal_for_state(self, demo: DemoSequence, state_step: int) -> np.ndarray:
        state_step = max(0, int(state_step))
        end_step = state_step + self.goal_horizon_steps
        if self.goal_source == "reference":
            if (
                demo.reference_pos_w is None
                or demo.reference_yaw_w is None
                or demo.reference_cumulative_xy is None
                or (self.goal_representation == "holonomic_se2_32" and demo.reference_command_b is None)
                or (
                    self.goal_representation == "path_guidance_se2_36"
                    and (
                        demo.guidance_pos_w is None
                        or demo.guidance_cumulative_xy is None
                        or demo.terminal_step is None
                    )
                )
            ):
                raise RuntimeError("Reference goal requested but the demonstration has no reference path.")
            return build_goal_vector(
                demo.reference_pos_w,
                demo.reference_cumulative_xy,
                state_step,
                end_step,
                demo.root_pos_w[state_step],
                demo.root_quat_w[state_step],
                yaws_w=demo.reference_yaw_w,
                dt=self.dt,
                waypoint_time_offsets_s=self.waypoint_time_offsets_s,
                v_req_clip=self.v_req_clip,
                goal_representation=self.goal_representation,
                reference_command_b=demo.reference_command_b,
                guidance_pos_w=demo.guidance_pos_w,
                guidance_cumulative_xy=demo.guidance_cumulative_xy,
                terminal_idx=demo.terminal_step,
            )
        return build_goal_vector(
            demo.root_pos_w,
            demo.cumulative_xy,
            state_step,
            end_step,
            demo.root_pos_w[state_step],
            demo.root_quat_w[state_step],
            quat_w=demo.root_quat_w,
            dt=self.dt,
            waypoint_time_offsets_s=self.waypoint_time_offsets_s,
            v_req_clip=self.v_req_clip,
            goal_representation=self.goal_representation,
            height_profile_w=demo.desired_base_height,
        )

    def _goal_history(self, sample: HindsightSample) -> np.ndarray:
        demo = self.demos[sample.demo_idx]
        start = sample.anchor_step - self.history
        return np.stack([self._goal_for_state(demo, state_step) for state_step in range(start, sample.anchor_step)])

    def goal_for_sample(self, sample_index: int, history_index: int = -1) -> np.ndarray:
        """Return one raw goal token for coverage/preflight analysis."""

        sample = self.samples[int(sample_index)]
        history_index = history_index if history_index >= 0 else self.history + history_index
        if not 0 <= history_index < self.history:
            raise IndexError(f"history_index must be in [-{self.history}, {self.history - 1}].")
        state_step = sample.anchor_step - self.history + history_index
        return self._goal_for_state(self.demos[sample.demo_idx], state_step)

    def _raw_sample(self, sample: HindsightSample) -> tuple[torch.Tensor, ...]:
        demo = self.demos[sample.demo_idx]
        proprio_hist, action_hist = delayed_io_windows(
            demo.proprio,
            demo.actions,
            sample.anchor_step,
            self.history,
            pad_start=self.include_padded_starts,
        )
        target_start = sample.anchor_step - self.execution_offset
        target_end = target_start + self.prediction_horizon
        target_actions = np.zeros((self.prediction_horizon, demo.actions.shape[-1]), dtype=np.float32)
        source_start = max(0, target_start)
        source_end = min(len(demo.actions), target_end)
        destination_start = source_start - target_start
        destination_end = destination_start + max(0, source_end - source_start)
        if source_end > source_start:
            target_actions[destination_start:destination_end] = demo.actions[source_start:source_end]
        return (
            torch.from_numpy(proprio_hist.copy()),
            torch.from_numpy(action_hist.copy()),
            torch.from_numpy(self._goal_history(sample).astype(np.float32)),
            torch.from_numpy(target_actions),
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
            goals = torch.zeros((len(actions), self.goal_dim), dtype=actions.dtype)
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
