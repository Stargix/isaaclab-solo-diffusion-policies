"""Regression tests for the command-baseline temporal and data contracts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT))

from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
from model.transformer_for_diffusion import TransformerForDiffusion
from train.dataset import DiffuseLocoCommandDataset
from train.episode_split import split_episode_indices


def write_dataset(path: Path, *, demos: int = 3, length: int = 140) -> None:
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
            obs.create_dataset("command_speed", data=np.repeat(np.array([[0.4, 0.1, -0.2]], np.float32), length, axis=0))
            demo.create_dataset("actions", data=actions)
            dones = np.zeros(length, dtype=bool)
            dones[-1] = True
            demo.create_dataset("dones", data=dones)


class CommandContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "command.hdf5"
        write_dataset(self.path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_exact_temporal_alignment(self) -> None:
        dataset = DiffuseLocoCommandDataset([str(self.path)], symmetry_mode="none")
        item = dataset[0]
        self.assertEqual(tuple(item["actions"].shape), (16, 12))
        np.testing.assert_allclose(item["proprio_hist"][:, 0], np.arange(1, 9))
        np.testing.assert_allclose(item["action_hist"][:, 0], np.arange(0, 8))
        np.testing.assert_allclose(item["actions"][:, 0], np.arange(1, 17))
        self.assertEqual(float(item["actions"][8, 0]), 9.0)

    def test_episode_split_and_train_only_stats(self) -> None:
        dataset = DiffuseLocoCommandDataset([str(self.path)], symmetry_mode="none")
        split = split_episode_indices(len(dataset.demos), 0.34, 7)
        self.assertFalse(set(split.train_demo_indices) & set(split.val_demo_indices))
        train_windows = dataset.sample_indices_for_demos(split.train_demo_indices)
        val_windows = dataset.sample_indices_for_demos(split.val_demo_indices)
        self.assertFalse(set(train_windows) & set(val_windows))
        stats = dataset.build_normalizer_stats([0])
        self.assertLess(float(stats.action.max.max()), 200.0)

    def test_memory_mask_exposes_full_history_at_execution(self) -> None:
        model = TransformerForDiffusion(
            input_dim=12,
            output_dim=12,
            horizon=16,
            n_obs_steps=8,
            cond_dim=42,
            goal_dim=3,
            n_layer=1,
            n_head=4,
            n_emb=32,
            causal_attn=True,
            separate_goal_conditioning=True,
        )
        self.assertTrue(torch.isfinite(model.memory_mask[8]).all())
        self.assertTrue(torch.isfinite(model.memory_mask[0, [0, 1, 9]]).all())
        self.assertTrue(torch.isneginf(model.memory_mask[0, 2:9]).all())

    def test_small_policy_loss_and_sampling_shapes(self) -> None:
        dataset = DiffuseLocoCommandDataset([str(self.path)], symmetry_mode="none")
        stats = dataset.build_normalizer_stats([0, 1])
        cfg = Solo12DiffusionPolicyConfig(
            goal_dim=3,
            d_model=32,
            nhead=4,
            num_layers=1,
            p_drop_attn=0.0,
            num_train_timesteps=2,
            num_inference_steps=2,
        )
        policy = Solo12DiffusionPolicy(cfg)
        policy.set_normalizer_stats(stats)
        item = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
        self.assertTrue(torch.isfinite(policy.compute_loss(item)))
        trajectory = policy.predict_action_denormalized(
            item["proprio_hist"], item["action_hist"], item["goal_hist"]
        )
        self.assertEqual(tuple(trajectory.shape), (1, 16, 12))
        self.assertEqual(tuple(policy.executable_chunk(trajectory, 2).shape), (1, 2, 12))


if __name__ == "__main__":
    unittest.main()
