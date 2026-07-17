"""Fast checks for the reward-free LocoDiff data and SDE contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.baseline_locodiff.model.solo12_diffusion_policy import (
    Solo12DiffusionPolicy,
    Solo12DiffusionPolicyConfig,
)
from scripts.baseline_locodiff.train.data.dataset import LocoDiffCommandSkillDataset


def write_dataset(path: Path) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["condition_schema"] = "velocity_xyyaw_plus_desired_base_height_v1"
        data.attrs["control_rate_hz"] = 50.0
        data.attrs["convention"] = (
            "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]; "
            "last_action[t] = actions[t-1] (last_action[0] = 0)."
        )
        data.attrs["skill_names"] = np.asarray((b"walk", b"crouch"))
        for demo_index, skill in enumerate(("walk", "crouch")):
            length = 32
            demo = data.create_group(f"demo_{demo_index}")
            demo.attrs["skills_sequence"] = np.asarray([skill.encode()])
            obs = demo.create_group("obs")
            actions = np.arange(length * 12, dtype=np.float32).reshape(length, 12) / 100.0
            previous = np.concatenate([np.zeros((1, 12), np.float32), actions[:-1]])
            obs.create_dataset("joint_pos", data=np.zeros((length, 12), np.float32))
            obs.create_dataset("joint_vel", data=np.zeros((length, 12), np.float32))
            obs.create_dataset("base_lin_vel", data=np.zeros((length, 3), np.float32))
            obs.create_dataset("base_ang_vel", data=np.zeros((length, 3), np.float32))
            obs.create_dataset("projected_gravity", data=np.zeros((length, 3), np.float32))
            obs.create_dataset("last_action", data=previous)
            obs.create_dataset("command_speed", data=np.full((length, 3), 0.1 + demo_index, np.float32))
            height = 0.2932 if skill == "walk" else 0.1705
            obs.create_dataset("desired_base_height", data=np.full((length, 1), height, np.float32))
            demo.create_dataset("actions", data=actions)
            dones = np.zeros(length, dtype=np.bool_)
            dones[-1] = True
            demo.create_dataset("dones", data=dones)


class PaperContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "tiny.hdf5"
        write_dataset(self.path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_dataset_is_future_only_and_uses_one_hot_skill(self) -> None:
        dataset = LocoDiffCommandSkillDataset(
            [str(self.path)], history=4, prediction_horizon=6, symmetry_mode="none"
        )
        sample = dataset[0]
        anchor = dataset.samples[0].anchor_step
        expected = dataset.demos[0].actions[anchor : anchor + 6]
        np.testing.assert_allclose(sample["actions"].numpy(), expected)
        self.assertEqual(tuple(sample["action_hist"].shape), (4, 0))
        np.testing.assert_allclose(sample["goal_hist"][:, 3:].numpy(), [[1.0, 0.0]] * 4)

    def test_empty_action_history_supports_all_quadruped_symmetries(self) -> None:
        dataset = LocoDiffCommandSkillDataset(
            [str(self.path)], history=4, prediction_horizon=6, symmetry_mode="quadruped"
        )
        for symmetry_index in range(4):
            sample = dataset[symmetry_index]
            self.assertEqual(tuple(sample["action_hist"].shape), (4, 0))
            self.assertEqual(tuple(sample["actions"].shape), (6, 12))

    def test_continuous_height_condition_reuses_the_same_hdf5(self) -> None:
        dataset = LocoDiffCommandSkillDataset(
            [str(self.path)],
            history=4,
            prediction_horizon=6,
            symmetry_mode="none",
            condition_mode="velocity_height",
        )
        walk = dataset[0]["goal_hist"]
        crouch = dataset[dataset.sample_indices_for_demos((1,))[0]]["goal_hist"]
        self.assertEqual(tuple(walk.shape), (4, 4))
        np.testing.assert_allclose(walk[:, -1].numpy(), 0.2932)
        np.testing.assert_allclose(crouch[:, -1].numpy(), 0.1705)

    def test_log_logistic_loss_backward_and_three_step_sample(self) -> None:
        dataset = LocoDiffCommandSkillDataset(
            [str(self.path)], history=4, prediction_horizon=6, symmetry_mode="none"
        )
        stats = dataset.build_normalizer_stats((0, 1), max_stats_samples=32)
        samples = [dataset[0], dataset[len(dataset) // 2]]
        batch = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
        policy = Solo12DiffusionPolicy(
            Solo12DiffusionPolicyConfig(
                history=4,
                prediction_horizon=6,
                d_model=32,
                nhead=4,
                num_layers=2,
                p_drop_attn=0.0,
                num_inference_steps=3,
            )
        )
        policy.set_normalizer_stats(stats)
        sigma = policy.sample_sigma(1024, device=torch.device("cpu"), dtype=torch.float32)
        self.assertTrue(torch.isfinite(sigma).all())
        self.assertGreaterEqual(float(sigma.min()), policy.cfg.sigma_min)
        self.assertLessEqual(float(sigma.max()), policy.cfg.sigma_max)
        loss = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in policy.parameters()))
        policy.eval()
        output = policy.predict_action(batch["proprio_hist"], batch["action_hist"], batch["goal_hist"])
        self.assertEqual(tuple(output.shape), (2, 6, 12))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNone(policy.model.model.memory_mask)

        continuous_policy = Solo12DiffusionPolicy(
            Solo12DiffusionPolicyConfig(
                goal_dim=4,
                history=4,
                prediction_horizon=6,
                d_model=32,
                nhead=4,
                num_layers=2,
                p_drop_attn=0.0,
                num_inference_steps=3,
            )
        )
        self.assertEqual(continuous_policy.cfg.goal_dim, 4)


if __name__ == "__main__":
    unittest.main()
