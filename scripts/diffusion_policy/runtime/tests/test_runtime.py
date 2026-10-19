from __future__ import annotations

import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.diffusion_policy.runtime.contract import validate_checkpoint_contract
from scripts.diffusion_policy.runtime.controller import (
    ChunkExecutor,
    DelayedHistory,
    Profile16Route,
    action_to_joint_target,
    EXPECTED_JOINT_NAMES,
    validate_joint_mapping,
)
from scripts.diffusion_policy.runtime.fixtures import (
    RuntimeFixtureWriter,
    load_fixture_bundle,
    select_capture_steps,
)
from scripts.diffusion_policy.runtime.benchmark import run_sequential_replay


def valid_checkpoint() -> dict:
    return {
        "algorithm": "dppo",
        "schema_version": 8,
        "dppo_task_contract_version": 8,
        "policy_kind": "spatial_hindsight_height_profile_ddpm",
        "config": {
            "goal_schema": "hindsight_geometric_height_profile16_v1",
            "model": {"proprio_dim": 30, "action_hist_dim": 12, "goal_dim": 16, "action_dim": 12},
            "dataset": {
                "history": 8, "prediction_horizon": 16, "execution_offset": 8,
                "goal_horizon_steps": 100, "dt": 0.02,
                "goal_representation": "hindsight_geom_profile16", "v_req_clip": 2.0,
            },
            "diffusion": {"prediction_type": "epsilon", "variance_type": "fixed_small"},
        },
        "dppo_config": {"inference_steps": 10, "finetune_denoising_steps": 5, "exec_horizon": 4},
        "dppo_task_config": {
            "goal_representation": "hindsight_geom_profile16", "pace_consistent_preview": False,
            "speed_budget_max_mps": 1.5,
        },
    }


class FakePolicy:
    def __init__(self):
        self.calls = 0

    def predict_action_denormalized(self, proprio, action, goal, *, guidance_scale):
        if guidance_scale != 1.0:
            raise AssertionError("guidance contract changed")
        self.calls += 1
        base = torch.arange(16 * 12, dtype=proprio.dtype, device=proprio.device).reshape(1, 16, 12)
        return base.expand(proprio.shape[0], -1, -1) + 1000.0 * self.calls

    @staticmethod
    def executable_chunk(trajectory, num_actions):
        if num_actions != 4:
            raise ValueError("unexpected chunk")
        return trajectory[:, 8:12]


class SequentialReplayPolicy:
    """Deterministic sampler whose output depends on the captured history."""

    def predict_action_denormalized(self, proprio, action, goal, *, guidance_scale):
        if guidance_scale != 1.0:
            raise AssertionError("guidance contract changed")
        marker = proprio[0, -1, 0]
        sequence = torch.arange(16 * 12, dtype=proprio.dtype, device=proprio.device).reshape(1, 16, 12)
        return sequence * 0.001 + marker

    @staticmethod
    def executable_chunk(trajectory, num_actions):
        if num_actions != 4:
            raise ValueError("unexpected chunk")
        return trajectory[:, 8:12]


def make_sequential_fixture():
    """Build eight exact consecutive records with nonzero initial histories."""
    device = torch.device("cpu")
    x = np.linspace(0.0, 4.0, 101, dtype=np.float32)
    path = np.column_stack((x, np.zeros_like(x), np.full_like(x, 0.2932)))
    yaw = np.zeros_like(x)
    cumulative = x.copy()
    initial = {
        "proprio_history": torch.arange(8 * 30, dtype=torch.float32).reshape(8, 30) + 1,
        "action_history": torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12) + 101,
        "goal_history": torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) + 201,
        "previous_action": torch.arange(12, dtype=torch.float32) + 301,
    }
    history = DelayedHistory(**{
        field: initial[source].unsqueeze(0).clone()
        for field, source in (("proprio", "proprio_history"),
                              ("action", "action_history"),
                              ("goal", "goal_history"),
                              ("previous_action", "previous_action"))
    })
    route = Profile16Route(path.copy(), cumulative.copy(), yaw.copy(), 0, 0.0, 4.0, 0.35)
    policy = SequentialReplayPolicy()
    rng = torch.get_rng_state().numpy().copy()
    metadata = {
        "capture_mode": "sequential",
        "metadata": {
            "guidance_scale": 1.0,
            "goal_horizon_steps": 100,
            "speed_budget_max_mps": 1.5,
            "v_req_clip_mps": 2.0,
            "environment": {
                "policy_dt_s": 0.02,
                "action_scale": 0.25,
                "q_offset_rad": np.zeros(12, dtype=np.float32).tolist(),
                "wrapper_clip_actions": None,
            },
        },
    }
    records = []
    for step in range(8):
        root_pos = np.asarray([step * 0.01, 0.0, 0.2932], dtype=np.float32)
        root_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        before = route.progress_idx
        goal = route.build_goal(
            root_pos_w=root_pos,
            root_quat_wxyz=root_quat,
            step=step,
            dt=0.02,
            goal_horizon_steps=100,
            speed_budget_max_mps=1.5,
        )
        trajectory = policy.predict_action_denormalized(
            history.proprio, history.action, history.goal, guidance_scale=1.0
        ) if step % 4 == 0 else None
        if trajectory is not None:
            chunk = policy.executable_chunk(trajectory, 4)
        action = chunk[:, step % 4].clone()
        proprio = torch.full((1, 30), 10.0 + step, dtype=torch.float32)
        records.append({
            **{name: value[0].cpu().numpy().copy() for name, value in {
                "proprio_history": history.proprio,
                "action_history": history.action,
                "goal_history": history.goal,
                "previous_action": history.previous_action,
            }.items()},
            "step": np.asarray(step, dtype=np.int64),
            "rng_cpu": rng.copy(),
            "rng_cuda": np.empty((0,), dtype=np.uint8),
            "root_pos_pre": root_pos,
            "root_quat_pre": root_quat,
            "path_w": path.copy(),
            "path_yaw_w": yaw.copy(),
            "path_cumulative_xy": cumulative.copy(),
            "path_progress_idx": np.asarray(before, dtype=np.int64),
            "speed_budget_start_arc_m": np.asarray(0.0, dtype=np.float32),
            "speed_budget_target_arc_m": np.asarray(4.0, dtype=np.float32),
            "requested_mean_speed_mps": np.asarray(0.35, dtype=np.float32),
            "full_trajectory": trajectory[0].cpu().numpy().copy() if trajectory is not None else records[-1]["full_trajectory"].copy(),
            "executable_chunk": chunk[0].cpu().numpy().copy(),
            "action_from_chunk": action[0].cpu().numpy().copy(),
            "env_action": action[0].cpu().numpy().copy(),
            "current_goal_pre_step": goal.copy(),
            "proprio_pre_step": proprio[0].cpu().numpy().copy(),
            "path_progress_before_goal": np.asarray(before, dtype=np.int64),
            "path_progress_after_goal": np.asarray(route.progress_idx, dtype=np.int64),
            "conditioning_speed_mps": np.asarray(goal[-1], dtype=np.float32),
            "selected_chunk_slot": np.asarray(step % 4, dtype=np.int64),
            "chunk_slot": np.asarray(step % 4, dtype=np.int64),
            "joint_target_post_step": (0.25 * action[0]).cpu().numpy().copy(),
            "dones": np.asarray(False),
        })
        history.commit_tick(proprio, action, torch.as_tensor(goal).unsqueeze(0))
    return metadata, records


class RuntimeTest(unittest.TestCase):
    def test_sequential_replay_seeds_nonzero_captured_histories(self):
        metadata, records = make_sequential_fixture()
        self.assertTrue(np.any(records[0]["proprio_history"]))
        self.assertTrue(np.any(records[0]["action_history"]))
        self.assertTrue(np.any(records[0]["goal_history"]))
        self.assertTrue(np.any(records[0]["previous_action"]))
        policy = SequentialReplayPolicy()
        contract = SimpleNamespace(
            device=torch.device("cpu"),
            sampler=policy,
            policy=policy,
            sha256="test-checkpoint-sha",
            values={"v_req_clip_mps": 2.0},
        )
        with tempfile.TemporaryDirectory() as temp:
            result = run_sequential_replay(contract, (metadata, records), Path(temp))
        self.assertTrue(result["pass"])
        self.assertEqual(result["step_count"], 8)
        self.assertEqual(result["complete_chunks"], 2)
        self.assertTrue(all(row["history_pass"] for row in result["steps"]))
        self.assertEqual(result["steps"][0]["history_max_abs_error"], 0.0)

    def test_contract_accepts_pinned_and_rejects_horizon_or_preview_changes(self):
        result = validate_checkpoint_contract(valid_checkpoint())
        self.assertEqual(result["exec_horizon"], 4)
        bad = valid_checkpoint()
        bad["dppo_config"]["exec_horizon"] = 8
        with self.assertRaisesRegex(ValueError, "exec_horizon"):
            validate_checkpoint_contract(bad)
        bad = valid_checkpoint()
        bad["dppo_task_config"]["pace_consistent_preview"] = True
        with self.assertRaisesRegex(ValueError, "pace_consistent_preview"):
            validate_checkpoint_contract(bad)

    def test_history_delay_uses_pre_step_and_previous_action(self):
        history = DelayedHistory.zeros(1, device=torch.device("cpu"))
        # Distinct sentinels reveal dimension/order mistakes.
        history.proprio[0, :, 0] = torch.arange(8, dtype=torch.float32)
        history.action[0, :, 0] = torch.arange(10, 18, dtype=torch.float32)
        history.goal[0, :, 0] = torch.arange(20, 28, dtype=torch.float32)
        history.previous_action.fill_(77)
        pre = torch.full((1, 30), 88.0)
        env_action = torch.full((1, 12), 99.0)
        goal = torch.full((1, 16), 111.0)
        history.commit_tick(pre, env_action, goal)
        torch.testing.assert_close(history.proprio[0, :, 0], torch.tensor([1, 2, 3, 4, 5, 6, 7, 88.]))
        torch.testing.assert_close(history.action[0, :, 0], torch.tensor([11, 12, 13, 14, 15, 16, 17, 77.]))
        torch.testing.assert_close(history.goal[0, :, 0], torch.tensor([21, 22, 23, 24, 25, 26, 27, 111.]))
        torch.testing.assert_close(history.previous_action, env_action)

    def test_chunk_executor_and_delayed_history_reset_together(self):
        class RecordingPolicy(FakePolicy):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def predict_action_denormalized(self, proprio, action, goal, *, guidance_scale):
                self.inputs.append((proprio.clone(), action.clone(), goal.clone()))
                return super().predict_action_denormalized(
                    proprio, action, goal, guidance_scale=guidance_scale
                )

        policy = RecordingPolicy()
        executor = ChunkExecutor(policy)
        history = DelayedHistory.zeros(1, device=torch.device("cpu"))
        history.previous_action.fill_(7.0)
        history.action[:, -1].fill_(6.0)
        history.proprio[:, -1].fill_(5.0)
        history.goal[:, -1].fill_(4.0)

        slots, executed_actions = [], []
        for tick in range(5):
            action, slot, replanned = executor.action(
                history.proprio, history.action, history.goal
            )
            slots.append(slot)
            executed_actions.append(action.clone())
            self.assertEqual(replanned, slot == 0)
            pre = torch.full((1, 30), 10.0 + tick)
            goal = torch.full((1, 16), 20.0 + tick)
            history.commit_tick(pre, action, goal)

        self.assertEqual(slots, [0, 1, 2, 3, 0])
        self.assertEqual(policy.calls, 2)
        # The second plan observes values committed after the prior tick, while
        # action history retains the deliberate one-step previous-action delay.
        torch.testing.assert_close(policy.inputs[1][0][0, -1], torch.full((30,), 13.0))
        torch.testing.assert_close(policy.inputs[1][2][0, -1], torch.full((16,), 23.0))
        torch.testing.assert_close(policy.inputs[1][1][0, -1], executed_actions[2][0])

        history.reset()
        executor.reset()
        self.assertEqual(float(history.previous_action.abs().max()), 0.0)
        self.assertEqual(float(history.proprio.abs().max()), 0.0)
        self.assertEqual(float(history.action.abs().max()), 0.0)
        self.assertEqual(float(history.goal.abs().max()), 0.0)
        _, slot, replanned = executor.action(history.proprio, history.action, history.goal)
        self.assertEqual((slot, replanned), (0, True))
        self.assertEqual(policy.calls, 3)
        for value in policy.inputs[-1]:
            self.assertEqual(float(value.abs().max()), 0.0)

    def test_chunk_slots_replan_after_four_and_reset_invalidates(self):
        policy = FakePolicy()
        executor = ChunkExecutor(policy)
        props, acts, goals = torch.zeros(1, 8, 30), torch.zeros(1, 8, 12), torch.zeros(1, 8, 16)
        seen = []
        for expected_slot in range(4):
            action, slot, replanned = executor.action(props, acts, goals)
            seen.append(slot)
            self.assertEqual(slot, expected_slot)
            self.assertEqual(replanned, expected_slot == 0)
            self.assertEqual(float(action[0, 0]), 1000.0 + 8 * 12 + expected_slot * 12)
        _, slot, replanned = executor.action(props, acts, goals)
        self.assertEqual((slot, replanned), (0, True))
        self.assertEqual(policy.calls, 2)
        executor.reset()
        _, slot, replanned = executor.action(props, acts, goals)
        self.assertEqual((slot, replanned), (0, True))
        self.assertEqual(policy.calls, 3)

    def test_action_clip_and_joint_target_map(self):
        raw = torch.tensor([[2.0, -2.0, *([0.5] * 10)]])
        clipped, target = action_to_joint_target(
            raw, action_scale=0.25, q_offset=torch.arange(12, dtype=torch.float32), clip_actions=1.0
        )
        self.assertEqual(float(clipped[0, 0]), 1.0)
        self.assertEqual(float(target[0, 0]), 0.25)
        self.assertEqual(float(target[0, 1]), 0.75)
        with self.assertRaisesRegex(ValueError, "NaN|infinity"):
            action_to_joint_target(torch.full((1, 12), float("nan")), action_scale=.25,
                                   q_offset=torch.zeros(12), clip_actions=None)

    def test_joint_name_order_is_explicit_and_rejects_permutations(self):
        self.assertEqual(validate_joint_mapping(EXPECTED_JOINT_NAMES), EXPECTED_JOINT_NAMES)
        with self.assertRaisesRegex(ValueError, "Joint mapping"):
            validate_joint_mapping(tuple(reversed(EXPECTED_JOINT_NAMES)))

    def test_profile_goal_remaining_budget_and_terminal(self):
        x = np.linspace(0.0, 4.0, 101, dtype=np.float32)
        path = np.column_stack((x, np.zeros_like(x), np.full_like(x, 0.2932)))
        route = Profile16Route(path, x.copy(), np.zeros_like(x), 0, 0.0, 4.0, 0.35)
        goal = route.build_goal(root_pos_w=np.asarray([0, 0, .29]), root_quat_wxyz=np.asarray([1,0,0,0]), step=0)
        self.assertEqual(goal.shape, (16,))
        self.assertAlmostEqual(float(goal[12]), .2932, places=5)
        self.assertAlmostEqual(float(goal[15]), .35, places=5)
        terminal = Profile16Route(path, x.copy(), np.zeros_like(x), 100, 0.0, 4.0, 0.35)
        final_goal = terminal.build_goal(root_pos_w=np.asarray([4, 0, .29]), root_quat_wxyz=np.asarray([1,0,0,0]), step=600)
        self.assertEqual(float(final_goal[15]), 0.0)
        self.assertTrue(np.isfinite(final_goal).all())

    def test_route_quaternion_and_path_validation(self):
        x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        path = np.column_stack((x, np.zeros_like(x), np.full_like(x, .2932)))
        route = Profile16Route(path, x, np.zeros_like(x), 0, 0., 1., .35)
        with self.assertRaisesRegex(ValueError, "quaternion"):
            route.build_goal(root_pos_w=np.zeros(3), root_quat_wxyz=np.zeros(4), step=0)
        route.path_w[2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            route.validate()

    def test_snapshot_schedule_distributed_and_roundtrips_without_pickle(self):
        steps = select_capture_steps(600, 4, 12)
        self.assertEqual(steps[0], 0)
        self.assertEqual(len(steps), 12)
        self.assertGreater(steps[-1], 500)
        state = torch.get_rng_state().numpy().copy()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "nested"
            writer = RuntimeFixtureWriter(output, metadata={"test": "yes"}, duration_steps=40,
                                          exec_horizon=4, max_samples=10, num_envs=1)
            for index, step in enumerate(sorted(writer.steps)):
                record = writer.begin(step=np.asarray(step), **{"rng_cpu": state, "rng_cuda": np.empty(0, np.uint8)})
                writer.finish(record, x=np.arange(4, dtype=np.float32) + index)
            npz, js = writer.close()
            self.assertTrue(npz.exists() and js.exists())
            with self.assertRaises(FileExistsError):
                RuntimeFixtureWriter(output, metadata={}, duration_steps=40, exec_horizon=4,
                                    max_samples=10, num_envs=1)
            metadata, records = load_fixture_bundle(output)
            self.assertEqual(metadata["fixture_version"], 1)
            np.testing.assert_array_equal(records[0]["x"], np.arange(4, dtype=np.float32))
            with np.load(npz, allow_pickle=False) as archive:
                self.assertIn("sample_000__x", archive.files)


if __name__ == "__main__":
    unittest.main()
