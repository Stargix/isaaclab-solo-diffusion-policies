from __future__ import annotations

import unittest

import torch

from scripts.residual_diffusion_rl.contracts import OBSERVATION_DIM
from scripts.residual_diffusion_rl.rewards import residual_reward
from scripts.residual_diffusion_rl.routes import RouteBank


class PhaseB1CoreTests(unittest.TestCase):
    def test_observation_contract_is_versioned(self):
        self.assertEqual(OBSERVATION_DIM, 75)

    def test_straight_route_builds_checkpoint_goal12(self):
        bank = RouteBank(1, "cpu", points=101, length_m=4.0)
        bank.reset(torch.tensor([0]), stage=0)
        bank.speed[:] = 0.4
        bank.height[:] = 0.2932
        position = torch.zeros(1, 2)
        bank.update(position)
        goal = bank.geometric_goal(position, torch.zeros(1), horizon_s=2.0, v_clip=2.0)
        expected = torch.tensor([
            0.2, 0.0, 0.4, 0.0, 0.6, 0.0,
            0.8, 0.0, 0.0, 1.0, 0.2932, 0.4,
        ])
        torch.testing.assert_close(goal[0], expected, atol=1.0e-5, rtol=1.0e-5)

    def test_route_progress_is_monotone(self):
        bank = RouteBank(2, "cpu")
        ids = torch.arange(2)
        bank.reset(ids, stage=2)
        previous = bank.progress.clone()
        for x in torch.linspace(0.0, 3.0, 12):
            state = bank.update(torch.stack((torch.full((2,), x), torch.zeros(2)), dim=1))
            self.assertTrue(torch.all(state.progress >= previous))
            previous = state.progress.clone()

    def test_residual_regularizer_prefers_local_correction(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1),
            speed_error=torch.zeros(1), yaw_error=torch.zeros(1),
            height_error=torch.zeros(1), previous_residual=torch.zeros(1, 12),
            projected_gravity_xy=torch.zeros(1, 2), vertical_velocity=torch.zeros(1),
            success=torch.zeros(1, dtype=torch.bool), failed=torch.zeros(1, dtype=torch.bool),
            dt=0.02,
        )
        local, _ = residual_reward(**common, residual=torch.zeros(1, 12))
        large, _ = residual_reward(**common, residual=torch.ones(1, 12))
        self.assertGreater(float(local), float(large))


if __name__ == "__main__":
    unittest.main()

