"""Regression tests for spatial hindsight relabeling and temporal alignment."""

from __future__ import annotations

import sys
import tempfile
import unittest
import random
from pathlib import Path

import h5py
import numpy as np
import torch

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PACKAGE_ROOT))
sys.path.insert(0, str(_PACKAGE_ROOT / "data"))

from train.data.dataset import SpatialHindsightDataset
from train.data.episode_split import split_episode_indices
from train.conditioning.goal_builder import advance_path_progress, build_goal_vector
from train.data.symmetry import apply_symmetry
from capability_routes import CapabilityLimits, generate_waypoint_guidance_route
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

    def test_geometric_hindsight_goal_uses_arc_fractions_and_average_speed(self) -> None:
        dataset = SpatialHindsightDataset(
            [str(self.path)],
            goal_horizon_steps=100,
            goal_source="achieved",
            goal_representation="hindsight_geom_avg12",
            symmetry_mode="none",
        )
        item = dataset[0]
        self.assertEqual(tuple(item["goal_hist"].shape), (8, 12))
        demo = dataset.demos[0]
        expected = build_goal_vector(
            demo.root_pos_w,
            demo.cumulative_xy,
            1,
            101,
            demo.root_pos_w[1],
            demo.root_quat_w[1],
            quat_w=demo.root_quat_w,
            goal_representation="hindsight_geom_avg12",
        )
        np.testing.assert_allclose(item["goal_hist"][0], expected, atol=1e-6)
        self.assertGreater(float(item["goal_hist"][0, 11]), 0.0)
        # The 25/50/75% points must advance geometrically along the trajectory.
        distances = np.linalg.norm(item["goal_hist"][0, :8].numpy().reshape(4, 2), axis=-1)
        self.assertTrue(np.all(np.diff(distances) >= -1.0e-5))

        mirrored = apply_symmetry(
            torch.zeros((8, 30)), torch.zeros((8, 12)), item["goal_hist"], torch.zeros((16, 12)),
            index=1, mode="quadruped",
        )[2]
        np.testing.assert_allclose(mirrored[:, 0::2][:, :4], item["goal_hist"][:, 0::2][:, :4], atol=1e-6)
        np.testing.assert_allclose(mirrored[:, 1:8:2], -item["goal_hist"][:, 1:8:2], atol=1e-6)
        np.testing.assert_allclose(mirrored[:, 8], -item["goal_hist"][:, 8], atol=1e-6)
        np.testing.assert_allclose(mirrored[:, 9:], item["goal_hist"][:, 9:], atol=1e-6)

        policy = Solo12DiffusionPolicy(Solo12DiffusionPolicyConfig(
            goal_dim=12, d_model=32, nhead=4, num_layers=1, p_drop_attn=0.0,
            num_train_timesteps=2, num_inference_steps=2,
        ))
        policy.set_normalizer_stats(dataset.build_normalizer_stats([0, 1], max_stats_samples=50))
        batch = {key: value.unsqueeze(0) for key, value in item.items()}
        self.assertTrue(torch.isfinite(policy.compute_loss(batch)))

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

    def test_waypoint_generator_is_bounded_and_has_terminal_hold(self) -> None:
        limits = CapabilityLimits(0.10, 0.45, 0.30, 0.50, 0.90, 0.45)
        route = generate_waypoint_guidance_route(
            start_pos_w=np.array([0.0, 0.0, 0.17], dtype=np.float32),
            start_yaw_w=0.0,
            desired_height=0.1705,
            steps=300,
            dt=0.02,
            limits=limits,
            rng=random.Random(7),
            startup_hold_steps=25,
            terminal_hold_steps=50,
        )
        self.assertEqual(route.pos_w.shape, (300, 3))
        self.assertIsNotNone(route.guidance_pos_w)
        np.testing.assert_allclose(route.nominal_command_b[:25], 0.0, atol=1e-6)
        np.testing.assert_allclose(route.nominal_command_b[-50:], 0.0, atol=1e-6)
        self.assertLessEqual(float(np.max(np.abs(route.nominal_command_b[:, 0]))), limits.vx_max + 1e-5)
        self.assertLessEqual(float(np.max(np.abs(route.nominal_command_b[:, 1]))), limits.vy_abs_max + 1e-5)
        self.assertLessEqual(float(np.max(np.abs(route.nominal_command_b[:, 2]))), limits.wz_abs_max + 1e-5)

    def test_path_guidance_goal_has_no_velocity_and_indexes_terminal_hold(self) -> None:
        with h5py.File(self.path, "r+") as file:
            data = file["data"]
            data.attrs["route_profile"] = "phase_a_path_guidance"
            data.attrs["reference_schema"] = "fixed_waypoint_task_with_noisy_guidance_and_terminal_stop_v1"
            for demo in data.values():
                obs = demo["obs"]
                reference = obs["reference_pos_w"][:]
                reference[190:] = reference[190]
                del obs["reference_pos_w"]
                obs.create_dataset("reference_pos_w", data=reference)
                command = obs["reference_command"][:]
                command[190:] = 0.0
                del obs["reference_command"]
                obs.create_dataset("reference_command", data=command)
                guidance = reference.copy()
                guidance[:, 1] += 0.04 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(guidance)))
                obs.create_dataset("guidance_pos_w", data=guidance)

        dataset = SpatialHindsightDataset(
            [str(self.path)],
            goal_horizon_steps=100,
            goal_source="reference",
            goal_representation="path_guidance_se2_36",
            symmetry_mode="none",
        )
        item = dataset[0]
        self.assertEqual(tuple(item["goal_hist"].shape), (8, 36))
        np.testing.assert_allclose(item["goal_hist"][:, 34], 0.0)
        self.assertTrue(torch.all(item["goal_hist"][:, 35] > 0.0))
        # The path tokens contain direction/log-distance/arc-offset only; the
        # authoritative terminal block begins at 28 and exposes phase/time.
        arc_offsets = item["goal_hist"][0, 3:28:4].numpy()
        self.assertEqual(float(arc_offsets[0]), 0.0)
        self.assertTrue(np.all(np.diff(arc_offsets) >= 0.0))
        last_sample = max(
            range(len(dataset.samples)),
            key=lambda index: dataset.samples[index].anchor_step,
        )
        terminal_goal = dataset.goal_for_sample(last_sample)
        self.assertEqual(float(terminal_goal[33]), 0.0)
        self.assertEqual(float(terminal_goal[34]), 1.0)
        policy = Solo12DiffusionPolicy(Solo12DiffusionPolicyConfig(
            goal_dim=36,
            d_model=32,
            nhead=4,
            num_layers=1,
            p_drop_attn=0.0,
            num_train_timesteps=2,
            num_inference_steps=2,
        ))
        policy.set_normalizer_stats(dataset.build_normalizer_stats([0, 1], max_stats_samples=50))
        batch = {key: value.unsqueeze(0) for key, value in item.items()}
        loss = policy.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
