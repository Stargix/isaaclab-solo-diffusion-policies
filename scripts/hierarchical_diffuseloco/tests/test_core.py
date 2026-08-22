from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.hierarchical_diffuseloco.contracts import ActionBounds
from scripts.hierarchical_diffuseloco.rewards import RewardWeights, hierarchical_reward, schedule_potential
from scripts.hierarchical_diffuseloco.route import build_route_bank


class RouteBankTests(unittest.TestCase):
    def test_seeded_bank_has_all_families_and_correct_mean_speed(self):
        bank = build_route_bank(count=9, points=33, episode_duration_s=8.0, seed=5)
        self.assertEqual(set(bank.family), {"straight", "s_curve", "right_angle"})
        np.testing.assert_allclose(bank.length / 8.0, bank.desired_mean_speed, rtol=1e-5)
        np.testing.assert_allclose(bank.xy[:, 0], 0.0, atol=1e-6)
        self.assertTrue(np.all(np.diff(bank.arc_length, axis=1) > 0.0))
        self.assertTrue(np.any(np.isclose(bank.target_height, 0.1705)))
        self.assertTrue(np.any(np.isclose(bank.target_height, 0.2932)))

    def test_seed_is_reproducible(self):
        first = build_route_bank(count=6, points=17, seed=7)
        second = build_route_bank(count=6, points=17, seed=7)
        np.testing.assert_array_equal(first.xy, second.xy)
        np.testing.assert_array_equal(first.target_height, second.target_height)


class RewardTests(unittest.TestCase):
    def _reward(self, previous_error: float, current_error: float, *, fallen: bool = False):
        zero = torch.zeros(1)
        reward, terms = hierarchical_reward(
            previous_schedule_potential=schedule_potential(torch.tensor([previous_error])),
            current_schedule_potential=schedule_potential(torch.tensor([current_error])),
            cross_track_normalized=zero,
            heading_error_normalized=zero,
            height_error_normalized=zero,
            normalized_command_delta=torch.zeros(1, 4),
            terminal_pose_error_normalized=torch.zeros(1, 3),
            terminal=torch.tensor([fallen]), fallen=torch.tensor([fallen]),
            discount=0.99, macro_dt=0.08, weights=RewardWeights(),
        )
        return float(reward), terms

    def test_standing_behind_schedule_is_worse_than_catching_up(self):
        falling_behind, _ = self._reward(0.0, -1.0)
        catching_up, _ = self._reward(-1.0, 0.0)
        self.assertLess(falling_behind, 0.0)
        self.assertGreater(catching_up, 0.0)

    def test_fall_has_explicit_cost(self):
        safe, _ = self._reward(0.0, 0.0)
        fallen, terms = self._reward(0.0, 0.0, fallen=True)
        self.assertAlmostEqual(fallen - safe, -RewardWeights().fall, places=5)
        self.assertLess(float(terms["fall"]), 0.0)


class ContractTests(unittest.TestCase):
    def test_forward_task_maps_zero_actor_to_forward_motion(self):
        bounds = ActionBounds()
        midpoint = 0.5 * (bounds.low + bounds.high)
        self.assertGreater(float(midpoint[0]), 0.0)
        self.assertEqual(bounds.low.shape, (4,))


if __name__ == "__main__":
    unittest.main()
