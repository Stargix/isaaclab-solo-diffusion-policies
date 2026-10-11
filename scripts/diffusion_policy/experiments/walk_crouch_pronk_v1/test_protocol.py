"""No-simulator regressions for temporal consistency and portable linked data."""
import ast
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/diffusion_policy/data"))
sys.path.insert(0, str(ROOT / "scripts/diffusion_policy"))
from merge_datasets import REQUIRED_OBS_KEYS, EXPECTED_CONVENTION_PREFIX, merge_hdf5_files
from prepare_data import PROTOCOL, collection_command
from train.conditioning.geometry import cumulative_xy_lengths
from train.conditioning.goal_builder import build_goal_vector
from train.data.dataset import SpatialHindsightDataset
import torch
from unittest.mock import patch
import reset_row_repair
from run_phase_a import training_command


class ProtocolTests(unittest.TestCase):
    def test_train_is_from_scratch_without_fast_trot_or_config_overrides(self):
        command = training_command(Path("fresh_output"), "cuda:0")
        self.assertEqual(command[command.index("--policy_backend") + 1], "diffusion")
        self.assertEqual(command[command.index("--config") + 1], str(ROOT / PROTOCOL["train_config"]))
        self.assertEqual(command[command.index("--datasets") + 1], str(reset_row_repair.OUT / "walk_crouch_pronk_linked.hdf5"))
        for flag in ("--resume", "--checkpoint", "--epochs", "--goal_horizon_steps", "--symmetry_mode"):
            self.assertNotIn(flag, command)
        self.assertFalse(any("sprint" in token or "fast_trot" in token for token in command))

    def test_exact_reset_recovery_preserves_actions_and_all_other_rows(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as directory:
            base = Path(directory)
            path = base / "walk_full.hdf5"
            actions = np.random.default_rng(42).normal(size=(299, 12)).astype(np.float32)
            with h5py.File(path, "w") as file:
                data = file.create_group("data")
                data.attrs.update(policy_input_timing="synchronous_pre_action_v2", collection_saved_demos=2)
                for index in range(2):
                    demo = data.create_group(f"demo_{index}")
                    obs = demo.create_group("obs")
                    q = np.tile(reset_row_repair.SAFE_POSE, (299, 1))
                    if index == 1:
                        q[0] = 0
                    obs["joint_pos"], obs["joint_vel"] = q, np.zeros_like(q)
                    obs["command_speed"] = np.zeros((299, 3), dtype=np.float32)
                    obs["root_pos_w"] = np.tile([0, 0, .35], (299, 1)).astype(np.float32)
                    demo["actions"] = actions
            before = reset_row_repair.sha256(path)
            with patch.object(reset_row_repair, "OUT", base):
                result = reset_row_repair.repair_dataset(path, expected_original_sha256=before)
                self.assertEqual(result["modified_demo_names"], ["demo_1"])
                self.assertEqual(reset_row_repair.repair_dataset(path, expected_original_sha256=before), result)
            with h5py.File(path) as file:
                for demo in file["data"].values():
                    np.testing.assert_array_equal(demo["actions"][:], actions)
                    np.testing.assert_array_equal(demo["obs/joint_pos"][:], np.tile(reset_row_repair.SAFE_POSE, (299, 1)))
    def test_masked_joint_query_preserves_batch_and_joint_axes(self):
        tree = ast.parse((ROOT / "scripts/diffusion_policy/data/collect_data.py").read_text(encoding="utf-8"))
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == "query_raw_state")
        namespace = {"Any": object, "Dict": dict, "np": np, "torch": torch}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "collector_helper", "exec"), namespace)
        q = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        data = SimpleNamespace(joint_pos=q, joint_vel=q * .1, root_ang_vel_b=torch.zeros(5, 3),
                               projected_gravity_b=torch.zeros(5, 3), root_pos_w=torch.zeros(5, 3),
                               root_quat_w=torch.zeros(5, 4))
        raw = SimpleNamespace(_robot=SimpleNamespace(data=data, device="cpu"), _commands=torch.zeros(5, 3))
        for mask in ([3], [1, 4]):
            state = namespace["query_raw_state"](raw, [1, 4, 6], np.asarray(mask))
            self.assertEqual(state["joint_pos"].shape, (len(mask), 3))
            np.testing.assert_array_equal(state["joint_pos"], q[mask][:, [1, 4, 6]].numpy())
    def test_teacher_inputs_refreshed_after_manual_reset_and_command_write(self):
        tree = ast.parse((ROOT / "scripts/diffusion_policy/data/collect_data.py").read_text(encoding="utf-8"))
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == "refresh_expert_observations")
        namespace = {"Any": object, "Dict": dict, "np": np}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "collector_helper", "exec"), namespace)
        raw = SimpleNamespace(_commands=torch.tensor([[.3, .1, .2]]), position="post-reset")
        vec = SimpleNamespace(get_observations=lambda: (raw.position, raw._commands.clone()))
        state = {"command": np.zeros((1, 3))}
        obs = namespace["refresh_expert_observations"](vec, raw, state)
        self.assertEqual(obs[0], "post-reset")
        np.testing.assert_allclose(state["command"], obs[1].numpy())
        raw._commands.zero_()
        np.testing.assert_allclose(state["command"], [[.3, .1, .2]])
    def test_temporal_and_source_contract(self):
        p = PROTOCOL
        self.assertEqual(p["episode_s"] * p["control_rate_hz"], 300)
        self.assertEqual(p["hindsight_steps"] / p["control_rate_hz"], 2)
        self.assertGreaterEqual(p["command_resample_s"], 2)
        self.assertGreaterEqual(p["stop_hold_s"], 2)
        self.assertEqual(set(p["sources"]), {"walk", "crouch", "pronk"})
        self.assertEqual(p["sources"]["walk"]["height_m"], p["sources"]["pronk"]["height_m"])
        config = json.loads((ROOT / p["train_config"]).read_text())
        self.assertEqual(config["goal_representation"], "hindsight_geom_profile16")
        self.assertEqual(config["goal_horizon_steps"], p["hindsight_steps"])
        for skill in ("walk", "pronk"):
            command = collection_command(skill, Path("pilot.hdf5"), 29900, 128)
            self.assertNotIn("sprint", command)
            self.assertNotIn("--record_expert_diagnostics", command)

    def test_height_labels_do_not_encode_bounce(self):
        n = 101
        pos = np.zeros((n, 3), dtype=np.float32)
        pos[:, 0] = np.arange(n) * .02 * .3
        pos[:, 2] = .27 + .035 * np.sin(np.arange(n) * .2)
        quat = np.zeros((n, 4), dtype=np.float32)
        quat[:, 0] = 1
        goal = build_goal_vector(pos, cumulative_xy_lengths(pos), 0, 100, pos[0], quat[0],
                                 quat_w=quat, dt=.02, goal_representation="hindsight_geom_profile16",
                                 height_profile_w=np.full(n, .2932, dtype=np.float32))
        np.testing.assert_allclose(goal[[2, 5, 8, 11, 12]], .2932)
        self.assertAlmostEqual(float(goal[15]), .3, places=5)

    def test_linked_merge_relocates_and_remaps(self):
        (ROOT / "scratch").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as directory:
            base = Path(directory) / "bundle"
            base.mkdir()
            paths = []
            dims = dict(zip(REQUIRED_OBS_KEYS, [12, 12, 3, 3, 12, 3, 4, 3, 1]))
            for skill in ("walk", "crouch", "pronk"):
                path = base / f"{skill}.hdf5"
                paths.append(str(path))
                with h5py.File(path, "w") as file:
                    data = file.create_group("data")
                    data.attrs.update(skill_names=np.asarray([skill], dtype="S32"),
                                      condition_schema="velocity_xyyaw_plus_desired_base_height_v1",
                                      convention=EXPECTED_CONVENTION_PREFIX)
                    demo = data.create_group("demo_0")
                    obs = demo.create_group("obs")
                    for key, dim in dims.items():
                        obs[key] = np.zeros((299, dim), dtype=np.float32)
                    obs["root_quat_w"][:, 0] = 1
                    obs["root_pos_w"][:, 0] = np.arange(299) * .02 * .3
                    obs["desired_base_height"][:] = .1705 if skill == "crouch" else .2932
                    obs["last_action"][1:] = 1
                    demo["actions"] = np.ones((299, 12), dtype=np.float32)
                    demo["dones"] = np.r_[np.zeros(298, dtype=bool), True]
                    demo["skill_idx"] = np.zeros(299, dtype=np.int8)
            merged = base / "index.hdf5"
            merge_hdf5_files(paths, str(merged), link_data=True)
            relocated = Path(directory) / "relocated"
            base.rename(relocated)
            merged = relocated / "index.hdf5"
            with h5py.File(merged) as file:
                for demo in file["data"].values():
                    self.assertIsInstance(demo.get("obs", getlink=True), h5py.ExternalLink)
                    np.testing.assert_array_equal(demo["actions"][:], 1)
                self.assertEqual({int(d["skill_idx"][0]) for d in file["data"].values()}, {0, 1, 2})
            loader = SpatialHindsightDataset([str(merged)], goal_representation="hindsight_geom_profile16",
                                             include_padded_starts=True, symmetry_mode="quadruped")
            self.assertEqual(len(loader.samples), 3 * 200)
            sample = loader[0]
            self.assertEqual(tuple(sample["goal_hist"].shape), (8, 16))
            self.assertEqual(tuple(sample["actions"].shape), (16, 12))
            self.assertEqual(set(sample), {"proprio_hist", "action_hist", "goal_hist", "actions"})
            with self.assertRaises(FileExistsError):
                merge_hdf5_files(paths, str(merged))


if __name__ == "__main__":
    unittest.main()
