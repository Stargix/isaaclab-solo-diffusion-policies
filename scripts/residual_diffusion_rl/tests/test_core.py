from __future__ import annotations

import unittest

import torch

from scripts.residual_diffusion_rl.contracts import OBSERVATION_DIM
from scripts.residual_diffusion_rl.rewards import progress_timing_errors, residual_reward
from scripts.residual_diffusion_rl.routes import RouteBank


class PhaseB1CoreTests(unittest.TestCase):
    def test_observation_contract_is_versioned(self):
        self.assertEqual(OBSERVATION_DIM, 77)

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

    def test_stratified_stage2_sampling_covers_every_route_family(self):
        bank = RouteBank(8, "cpu")
        ids = torch.arange(8)
        bank.reset(ids, stage=2, stratified=True)
        self.assertEqual(bank.route_kind.tolist(), [0, 1, 2, 3, 0, 1, 2, 3])

    def test_residual_regularizer_prefers_local_correction(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1),
            speed_error=torch.zeros(1), schedule_error=torch.zeros(1), yaw_error=torch.zeros(1),
            height_error=torch.zeros(1), previous_residual=torch.zeros(1, 12),
            projected_gravity_xy=torch.zeros(1, 2), vertical_velocity=torch.zeros(1),
            success=torch.zeros(1, dtype=torch.bool), failed=torch.zeros(1, dtype=torch.bool),
            fell=torch.zeros(1, dtype=torch.bool),
            dt=0.02,
        )
        local, _ = residual_reward(**common, residual=torch.zeros(1, 12))
        large, _ = residual_reward(**common, residual=torch.ones(1, 12))
        self.assertGreater(float(local), float(large))

    def test_tracking_is_an_error_cost_not_an_alive_bonus(self):
        common = dict(
            progress_delta=torch.zeros(1), residual=torch.zeros(1, 12),
            previous_residual=torch.zeros(1, 12), projected_gravity_xy=torch.zeros(1, 2),
            vertical_velocity=torch.zeros(1), success=torch.zeros(1, dtype=torch.bool),
            failed=torch.zeros(1, dtype=torch.bool), fell=torch.zeros(1, dtype=torch.bool), dt=0.02,
            schedule_error=torch.zeros(1),
        )
        perfect, perfect_terms = residual_reward(
            **common,
            cross_track=torch.zeros(1), speed_error=torch.zeros(1),
            yaw_error=torch.zeros(1), height_error=torch.zeros(1),
        )
        inaccurate, _ = residual_reward(
            **common,
            cross_track=torch.full((1,), 0.20), speed_error=torch.full((1,), 0.25),
            yaw_error=torch.full((1,), 0.45), height_error=torch.full((1,), 0.035),
        )
        for name in ("path", "speed", "yaw", "height"):
            torch.testing.assert_close(perfect_terms[name], torch.zeros(1))
        torch.testing.assert_close(perfect, torch.zeros(1))
        self.assertGreater(float(perfect), float(inaccurate))

    def test_large_speed_error_does_not_saturate(self):
        common = dict(
            progress_delta=torch.full((1,), 0.04), cross_track=torch.zeros(1),
            schedule_error=torch.zeros(1), yaw_error=torch.zeros(1), height_error=torch.zeros(1),
            residual=torch.zeros(1, 12), previous_residual=torch.zeros(1, 12),
            projected_gravity_xy=torch.zeros(1, 2), vertical_velocity=torch.zeros(1),
            success=torch.zeros(1, dtype=torch.bool), failed=torch.zeros(1, dtype=torch.bool),
            fell=torch.zeros(1, dtype=torch.bool), dt=0.02,
        )
        nominal, nominal_terms = residual_reward(**common, speed_error=torch.zeros(1))
        fast, fast_terms = residual_reward(**common, speed_error=torch.full((1,), 0.8))
        very_fast, very_fast_terms = residual_reward(**common, speed_error=torch.full((1,), 1.6))
        self.assertGreater(float(nominal), float(fast))
        self.assertGreater(float(fast_terms["speed"]), float(very_fast_terms["speed"]))
        self.assertGreater(float(nominal_terms["progress"]), float(fast_terms["progress"]))

    def test_schedule_error_penalizes_sprint_then_wait(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1), speed_error=torch.zeros(1),
            yaw_error=torch.zeros(1), height_error=torch.zeros(1), residual=torch.zeros(1, 12),
            previous_residual=torch.zeros(1, 12), projected_gravity_xy=torch.zeros(1, 2),
            vertical_velocity=torch.zeros(1), success=torch.zeros(1, dtype=torch.bool),
            failed=torch.zeros(1, dtype=torch.bool), fell=torch.zeros(1, dtype=torch.bool), dt=0.02,
        )
        on_schedule, _ = residual_reward(**common, schedule_error=torch.zeros(1))
        ahead, _ = residual_reward(**common, schedule_error=torch.ones(1))
        behind, _ = residual_reward(**common, schedule_error=-torch.ones(1))
        self.assertGreater(float(on_schedule), float(ahead))
        torch.testing.assert_close(ahead, behind)

    def test_schedule_cost_has_episode_compatible_scale(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1),
            speed_error=torch.zeros(1), yaw_error=torch.zeros(1), height_error=torch.zeros(1),
            residual=torch.zeros(1, 12), previous_residual=torch.zeros(1, 12),
            projected_gravity_xy=torch.zeros(1, 2), vertical_velocity=torch.zeros(1),
            success=torch.zeros(1, dtype=torch.bool), failed=torch.zeros(1, dtype=torch.bool),
            fell=torch.zeros(1, dtype=torch.bool), dt=0.02,
        )
        _, terms = residual_reward(**common, schedule_error=torch.full((1,), 4.0))
        # Even the maximum route-scale error remains below 0.05 per step. It is
        # still strictly negative and therefore never becomes reward-neutral.
        self.assertLess(float(terms["schedule"]), 0.0)
        self.assertGreater(float(terms["schedule"]), -0.05)

    def test_average_speed_command_defines_schedule_and_final_time(self):
        schedule_error, mean_speed_error = progress_timing_errors(
            progress=torch.tensor([2.0, 4.0, 4.0]),
            route_length=torch.full((3,), 4.0),
            commanded_speed=torch.full((3,), 0.4),
            elapsed_s=torch.tensor([5.0, 10.0, 5.0]),
            min_dt=0.02,
        )
        torch.testing.assert_close(schedule_error, torch.tensor([0.0, 0.0, 2.0]))
        torch.testing.assert_close(mean_speed_error, torch.tensor([0.0, 0.0, 0.4]))

    def test_terminal_failure_is_penalized(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1),
            speed_error=torch.zeros(1), schedule_error=torch.zeros(1),
            yaw_error=torch.zeros(1), height_error=torch.zeros(1),
            residual=torch.zeros(1, 12), previous_residual=torch.zeros(1, 12),
            projected_gravity_xy=torch.zeros(1, 2), vertical_velocity=torch.zeros(1),
            success=torch.zeros(1, dtype=torch.bool), fell=torch.zeros(1, dtype=torch.bool), dt=0.02,
        )
        continuing, _ = residual_reward(**common, failed=torch.zeros(1, dtype=torch.bool))
        failed, _ = residual_reward(**common, failed=torch.ones(1, dtype=torch.bool))
        self.assertGreater(float(continuing), float(failed))

    def test_fall_costs_more_than_other_failure(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1), speed_error=torch.zeros(1),
            schedule_error=torch.zeros(1), yaw_error=torch.zeros(1), height_error=torch.zeros(1),
            residual=torch.zeros(1, 12), previous_residual=torch.zeros(1, 12),
            projected_gravity_xy=torch.zeros(1, 2), vertical_velocity=torch.zeros(1),
            success=torch.zeros(1, dtype=torch.bool), failed=torch.ones(1, dtype=torch.bool), dt=0.02,
        )
        task_failure, _ = residual_reward(**common, fell=torch.zeros(1, dtype=torch.bool))
        fall, _ = residual_reward(**common, fell=torch.ones(1, dtype=torch.bool))
        self.assertGreater(float(task_failure), float(fall))

    def test_terminal_distance_rejects_longitudinal_overshoot(self):
        bank = RouteBank(1, "cpu", points=101, length_m=4.0)
        bank.reset(torch.tensor([0]), stage=0)
        for x in torch.linspace(0.0, 4.0, 12):
            bank.update(torch.tensor([[x, 0.0]]))
        state = bank.update(torch.tensor([[4.5, 0.0]]))
        self.assertTrue(bool(state.success))
        self.assertAlmostEqual(float(state.cross_track), 0.0, places=6)
        self.assertAlmostEqual(float(state.terminal_distance), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
