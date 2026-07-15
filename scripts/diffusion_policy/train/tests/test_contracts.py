"""Regression tests for spatial hindsight relabeling and temporal alignment."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PACKAGE_ROOT))

from train.data.dataset import SpatialHindsightDataset
from train.data.episode_split import split_episode_indices
from train.conditioning.goal_builder import advance_path_progress, build_goal_vector
from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig


def write_dataset(path: Path, *, demos: int = 3, length: int = 240) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["control_rate_hz"] = 50.0
        data.attrs["convention"] = (
            "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]; "
            "last_action[t] = actions[t-1] (last_action[0] = 0)."
        )
        data.attrs["skill_names"] = np.asarray(["walk"], dtype=h5py.string_dtype())
        for demo_idx in range(demos):
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            offset = float(demo_idx * 1000)
            steps = np.arange(length, dtype=np.float32)[:, None]
            actions = np.repeat(steps + offset, 12, axis=1)
            obs.create_dataset("joint_pos", data=np.repeat(steps + offset, 12, axis=1))
            obs.create_dataset("joint_vel", data=np.repeat(steps, 12, axis=1))
            obs.create_dataset("base_ang_vel", data=np.repeat(steps, 3, axis=1))
            obs.create_dataset("projected_gravity", data=np.repeat(steps, 3, axis=1))
            last_action = np.zeros_like(actions)
            last_action[1:] = actions[:-1]
            obs.create_dataset("last_action", data=last_action)
            obs.create_dataset("command_speed", data=np.repeat(np.array([[0.4, 0.0, 0.0]], np.float32), length, axis=0))
            root_pos = np.zeros((length, 3), dtype=np.float32)
            root_pos[:, 0] = np.arange(length, dtype=np.float32) * 0.02
            root_pos[:, 1] = np.square(np.arange(length, dtype=np.float32)) * 0.0001
            root_pos[:, 2] = 0.20 + np.arange(length, dtype=np.float32) * 0.0002
            root_quat = np.zeros((length, 4), dtype=np.float32)
            root_quat[:, 0] = 1.0
            obs.create_dataset("root_pos_w", data=root_pos)
            obs.create_dataset("root_quat_w", data=root_quat)
            reference_pos = root_pos.copy()
            # Deliberately diverge from achieved motion so the test catches an
            # accidental fallback to hindsight positions.
            reference_pos[:, 1] = np.arange(length, dtype=np.float32) * 0.01
            reference_pos[:, 2] = 0.2932
            obs.create_dataset("reference_pos_w", data=reference_pos)
            obs.create_dataset("reference_yaw_w", data=np.zeros((length, 1), dtype=np.float32))
            obs.create_dataset(
                "reference_command",
                data=np.repeat(np.array([[0.4, 0.0, 0.0]], np.float32), length, axis=0),
            )
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("skill_idx", data=np.zeros(length, dtype=np.int8))
            dones = np.zeros(length, dtype=bool)
            dones[-1] = True
            demo.create_dataset("dones", data=dones)


class SpatialContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "spatial.hdf5"
        write_dataset(self.path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_target_and_rolling_hindsight_alignment(self) -> None:
        dataset = SpatialHindsightDataset(
            [str(self.path)], goal_horizon_steps=100, symmetry_mode="none"
        )
        item = dataset[0]
        np.testing.assert_allclose(item["proprio_hist"][:, 0], np.arange(1, 9))
        np.testing.assert_allclose(item["action_hist"][:, 0], np.arange(0, 8))
        np.testing.assert_allclose(item["actions"][:, 0], np.arange(1, 17))
        self.assertEqual(float(item["actions"][8, 0]), 9.0)
        demo = dataset.demos[0]
        expected_first = build_goal_vector(
            demo.root_pos_w,
            demo.cumulative_xy,
            1,
            101,
            demo.root_pos_w[1],
            demo.root_quat_w[1],
            quat_w=demo.root_quat_w,
        )
        np.testing.assert_allclose(item["goal_hist"][0], expected_first, atol=1e-6)
        self.assertFalse(np.allclose(item["goal_hist"][0], item["goal_hist"][-1]))
        np.testing.assert_allclose(item["goal_hist"][0, [0, 2, 4]], [0.5, 1.0, 1.5], atol=1e-6)

    def test_episode_split_and_train_only_stats(self) -> None:
        dataset = SpatialHindsightDataset(
            [str(self.path)], goal_horizon_steps=100, symmetry_mode="none"
        )
        split = split_episode_indices(len(dataset.demos), 0.34, 3)
        self.assertFalse(set(split.train_demo_indices) & set(split.val_demo_indices))
        self.assertFalse(
            set(dataset.sample_indices_for_demos(split.train_demo_indices))
            & set(dataset.sample_indices_for_demos(split.val_demo_indices))
        )
        stats = dataset.build_normalizer_stats([0], max_stats_samples=2, seed=3)
        self.assertEqual(float(stats.action.max.max()), 239.0)

    def test_progress_search_does_not_jump_to_crossing_branch(self) -> None:
        path = np.zeros((120, 3), dtype=np.float32)
        path[:, 0] = np.arange(120, dtype=np.float32) * 0.05
        path[100, :2] = path[10, :2]
        progress = advance_path_progress(path, path[10, :2], 10, search_forward=20)
        self.assertEqual(progress, 10)

    def test_small_spatial_policy_loss_and_sampling(self) -> None:
        dataset = SpatialHindsightDataset(
            [str(self.path)], goal_horizon_steps=100, symmetry_mode="none"
        )
        policy = Solo12DiffusionPolicy(
            Solo12DiffusionPolicyConfig(
                goal_dim=11,
                d_model=32,
                nhead=4,
                num_layers=1,
                p_drop_attn=0.0,
                num_train_timesteps=2,
                num_inference_steps=2,
            )
        )
        policy.set_normalizer_stats(dataset.build_normalizer_stats([0, 1], max_stats_samples=50))
        batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
        self.assertTrue(torch.isfinite(policy.compute_loss(batch)))
        trajectory = policy.predict_action_denormalized(
            batch["proprio_hist"], batch["action_hist"], batch["goal_hist"]
        )
        self.assertEqual(tuple(trajectory.shape), (1, 16, 12))
        self.assertEqual(tuple(policy.executable_chunk(trajectory, 2).shape), (1, 2, 12))

    def test_reference_goal_and_padded_reset_contract(self) -> None:
        dataset = SpatialHindsightDataset(
            [str(self.path)],
            goal_horizon_steps=100,
            goal_source="reference",
            include_padded_starts=True,
            startup_sample_multiplier=2,
            symmetry_mode="none",
        )
        item = dataset[0]
        # Anchor t=0 executes token H = expert action a[0]. The preceding
        # action trajectory and history are padded zeros, matching deployment.
        np.testing.assert_allclose(item["proprio_hist"][:, 0], 0.0)
        np.testing.assert_allclose(item["action_hist"], 0.0)
        np.testing.assert_allclose(item["actions"][:8], 0.0)
        np.testing.assert_allclose(item["actions"][8, 0], 0.0)
        # The reference curves much more than the achieved trajectory.
        self.assertGreater(float(item["goal_hist"][-1, 7]), 0.5)


if __name__ == "__main__":
    unittest.main()
