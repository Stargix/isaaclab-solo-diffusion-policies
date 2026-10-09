"""Regression tests for motion gating and completed physical pronk cycles."""

import importlib.util
import unittest
from pathlib import Path

import torch

_REWARD_PATH = Path(__file__).resolve().parents[5] / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/pronk_gait.py"
_SPEC = importlib.util.spec_from_file_location("pronk_gait", _REWARD_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load pronk reward from {_REWARD_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PronkCycleTracker = _MODULE.PronkCycleTracker
pronk_cycle_score = _MODULE.pronk_cycle_score


class PronkRewardTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PronkCycleTracker(1, "cpu", 0.02, warmup_s=0.04)

    def tick(self, contacts, *, clearance=0.01, air=0.06, valid=True, velocity=0.6, command=0.6):
        return self.tracker.update(
            torch.tensor([contacts], dtype=torch.bool),
            torch.full((1, 4), clearance),
            torch.full((1, 4), air),
            torch.tensor([[command, 0.0]]),
            torch.tensor([[velocity, 0.0]]),
            torch.tensor([valid]),
        )

    def support(self):
        for _ in range(3):
            self.tick([1, 1, 1, 1], clearance=0.0, air=0.0)

    def flight(self, **kwargs):
        for air in (0.02, 0.04, 0.06):
            result = self.tick([0, 0, 0, 0], air=air, **kwargs)
            self.assertEqual(result["gait"].item(), 0.0)

    def test_one_bonus_only_after_complete_supported_cycle(self):
        self.support()
        self.flight()
        landing = self.tick([1, 1, 1, 1], clearance=0.0, air=0.0)
        self.assertTrue(landing["completed"].item())
        self.assertAlmostEqual(landing["gait"].item(), 1.0, places=5)
        again = self.tick([1, 1, 1, 1], clearance=0.0, air=0.0)
        self.assertEqual(again["gait"].item(), 0.0)

    def test_reset_drop_without_previous_support_gets_no_bonus(self):
        self.flight()
        self.assertEqual(self.tick([1, 1, 1, 1], clearance=0.0)["gait"].item(), 0.0)

    def test_zero_contact_force_without_geometric_clearance_is_not_flight(self):
        self.support()
        self.flight(clearance=0.0)
        self.assertEqual(self.tick([1, 1, 1, 1], clearance=0.0)["gait"].item(), 0.0)

    def test_excessive_flight_and_failed_landing_get_no_bonus(self):
        self.support()
        for _ in range(16):
            self.assertEqual(self.tick([0, 0, 0, 0])["gait"].item(), 0.0)
        self.assertEqual(self.tick([1, 1, 1, 1], clearance=0.0)["gait"].item(), 0.0)

    def test_loss_of_upright_pose_discards_whole_cycle(self):
        self.support()
        self.flight()
        self.tick([0, 0, 0, 0], valid=False)
        self.assertEqual(self.tick([1, 1, 1, 1], clearance=0.0)["gait"].item(), 0.0)

    def test_landing_window_allows_small_offset_but_rejects_large_offset(self):
        self.support()
        self.flight()
        self.tick([1, 1, 0, 0], clearance=0.0)
        result = self.tick([1, 1, 1, 1], clearance=0.0)
        self.assertGreater(result["gait"].item(), 0.0)
        self.assertLess(result["gait"].item(), 1.0)
        self.tracker.reset()
        self.support()
        self.flight()
        for _ in range(5):
            self.tick([1, 0, 0, 0], clearance=0.0)
        self.assertEqual(self.tick([1, 1, 1, 1], clearance=0.0)["gait"].item(), 0.0)

    def test_changed_command_or_reset_cannot_inherit_pending_bonus(self):
        self.support()
        self.flight()
        self.tick([0, 0, 0, 0], command=0.3)
        self.assertEqual(self.tick([1, 1, 1, 1], command=0.3)["gait"].item(), 0.0)
        self.tracker.reset()
        self.support()
        self.flight()
        self.tracker.reset()
        self.assertEqual(self.tick([1, 1, 1, 1])["gait"].item(), 0.0)

    def test_completed_in_place_reverse_or_sideways_jump_gets_zero(self):
        commands = torch.tensor([[0.15, 0.0], [0.15, 0.0], [0.15, 0.0]])
        velocities = torch.tensor([[0.0, 0.0], [-0.15, 0.0], [0.0, 0.15]])
        result = pronk_cycle_score(
            commands, velocities, torch.zeros(3, 4), torch.zeros(3, 4), torch.ones(3, dtype=torch.bool)
        )
        self.assertTrue(torch.equal(result["gait"], torch.zeros(3)))

    def test_diagonal_takeoff_has_lower_score_than_synchronized_takeoff(self):
        result = pronk_cycle_score(
            torch.tensor([[0.6, 0.0], [0.6, 0.0]]),
            torch.tensor([[0.6, 0.0], [0.6, 0.0]]),
            torch.tensor([[0.06, 0.06, 0.06, 0.06], [0.02, 0.12, 0.12, 0.02]]),
            torch.zeros(2, 4), torch.ones(2, dtype=torch.bool),
        )
        self.assertAlmostEqual(result["gait"][0].item(), 1.0, places=5)
        self.assertLess(result["gait"][1].item(), 0.01)

    def test_per_environment_reset_preserves_other_pending_cycle(self):
        tracker = PronkCycleTracker(2, "cpu", 0.02, warmup_s=0.04)
        command = torch.tensor([[0.6, 0.0], [0.6, 0.0]])
        def tick(contact):
            return tracker.update(torch.full((2, 4), contact, dtype=torch.bool),
                                  torch.full((2, 4), 0.0 if contact else 0.01),
                                  torch.full((2, 4), 0.0 if contact else 0.06),
                                  command, command, torch.ones(2, dtype=torch.bool))
        for _ in range(3):
            tick(True)
        for _ in range(3):
            tick(False)
        tracker.reset(torch.tensor([0]))
        result = tick(True)
        self.assertEqual(result["gait"][0].item(), 0.0)
        self.assertGreater(result["gait"][1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
