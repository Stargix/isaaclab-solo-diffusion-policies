from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from scripts.residual_diffusion_rl.contracts import OBSERVATION_DIM
from scripts.diffusion_policy.train.conditioning.goal_builder import (
    build_geometric_hindsight_goal_from_path,
)
from scripts.residual_diffusion_rl.rewards import progress_timing_errors, residual_reward
from scripts.residual_diffusion_rl.routes import RouteBank


_ACTOR_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "source/isaaclab_tasks/isaaclab_tasks/direct/residual_diffusion_rl/agents/residual_actor_critic.py"
)
_ACTOR_SPEC = importlib.util.spec_from_file_location(
    "residual_actor_critic_under_test", _ACTOR_MODULE_PATH
)
if _ACTOR_SPEC is None or _ACTOR_SPEC.loader is None:
    raise RuntimeError(f"Cannot load residual actor module from {_ACTOR_MODULE_PATH}")
_ACTOR_MODULE = importlib.util.module_from_spec(_ACTOR_SPEC)
_ACTOR_SPEC.loader.exec_module(_ACTOR_MODULE)
ResidualActorCritic = _ACTOR_MODULE.ResidualActorCritic


def _cruise_reward_inputs() -> dict:
    return {
        "remaining_distance": torch.full((1,), 4.0),
        "terminal_distance": torch.full((1,), 4.0),
        "planar_speed": torch.zeros(1),
        "terminal_brake_distance": 0.60,
        "terminal_position_tolerance": 0.12,
        "terminal_stop_speed": 0.15,
    }


class PhaseB1CoreTests(unittest.TestCase):
    def test_observation_contract_is_versioned(self):
        self.assertEqual(OBSERVATION_DIM, 77)

    def test_residual_actor_mean_starts_at_zero(self):
        observations = TensorDict({"policy": torch.randn(8, 77)}, batch_size=[8])
        actor_critic = ResidualActorCritic(
            observations,
            {"policy": ["policy"], "critic": ["policy"]},
            12,
            init_noise_std=0.10,
            noise_std_type="log",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32],
            activation="elu",
        )
        output = actor_critic.act_inference(observations)
        torch.testing.assert_close(output, torch.zeros_like(output))

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

    def test_goal12_matches_phase_a_builder_near_route_end(self):
        bank = RouteBank(1, "cpu", points=101, length_m=4.0)
        bank.reset(torch.tensor([0]), stage=0)
        bank.speed[:] = 0.4
        bank.height[:] = 0.2932
        for x in torch.linspace(0.0, 3.8, 12):
            bank.update(torch.tensor([[x, 0.0]]))

        position = torch.tensor([[3.8, 0.0]])
        bank.update(position)
        actual = bank.geometric_goal(
            position, torch.zeros(1), horizon_s=2.0, v_clip=2.0
        )[0]
        path = np.column_stack((bank.xy[0].numpy(), bank.height[0].numpy()))
        expected = build_geometric_hindsight_goal_from_path(
            path,
            bank.arc[0].numpy(),
            bank.yaw[0].numpy(),
            np.asarray([3.8, 0.0, 0.2932], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            goal_horizon_steps=100,
            dt=0.02,
            speed=0.4,
            start_idx=int(bank.progress_idx[0]),
            v_avg_clip=2.0,
        )
        torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1.0e-5, rtol=1.0e-5)
        self.assertAlmostEqual(float(actual[-1]), 0.10, places=5)

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
            **_cruise_reward_inputs(),
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
            failed=torch.zeros(1, dtype=torch.bool), fell=torch.zeros(1, dtype=torch.bool),
            **_cruise_reward_inputs(), dt=0.02,
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
            fell=torch.zeros(1, dtype=torch.bool), **_cruise_reward_inputs(), dt=0.02,
        )
        nominal, nominal_terms = residual_reward(**common, speed_error=torch.zeros(1))
        fast, fast_terms = residual_reward(**common, speed_error=torch.full((1,), 0.8))
        very_fast, very_fast_terms = residual_reward(**common, speed_error=torch.full((1,), 1.6))
        self.assertGreater(float(nominal), float(fast))
        self.assertGreater(float(fast_terms["speed"]), float(very_fast_terms["speed"]))
        torch.testing.assert_close(nominal_terms["progress"], fast_terms["progress"])
        torch.testing.assert_close(fast_terms["progress"], very_fast_terms["progress"])

    def test_schedule_error_penalizes_sprint_then_wait(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1), speed_error=torch.zeros(1),
            yaw_error=torch.zeros(1), height_error=torch.zeros(1), residual=torch.zeros(1, 12),
            previous_residual=torch.zeros(1, 12), projected_gravity_xy=torch.zeros(1, 2),
            vertical_velocity=torch.zeros(1), success=torch.zeros(1, dtype=torch.bool),
            failed=torch.zeros(1, dtype=torch.bool), fell=torch.zeros(1, dtype=torch.bool),
            **_cruise_reward_inputs(), dt=0.02,
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
            fell=torch.zeros(1, dtype=torch.bool), **_cruise_reward_inputs(), dt=0.02,
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
            success=torch.zeros(1, dtype=torch.bool), fell=torch.zeros(1, dtype=torch.bool),
            **_cruise_reward_inputs(), dt=0.02,
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
            success=torch.zeros(1, dtype=torch.bool), failed=torch.ones(1, dtype=torch.bool),
            **_cruise_reward_inputs(), dt=0.02,
        )
        task_failure, _ = residual_reward(**common, fell=torch.zeros(1, dtype=torch.bool))
        fall, _ = residual_reward(**common, fell=torch.ones(1, dtype=torch.bool))
        self.assertGreater(float(task_failure), float(fall))

    def test_terminal_region_rewards_stopping_not_cruise_speed(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1),
            speed_error=torch.full((1,), -0.4), schedule_error=torch.zeros(1),
            yaw_error=torch.zeros(1), height_error=torch.zeros(1),
            remaining_distance=torch.zeros(1), terminal_distance=torch.zeros(1),
            terminal_brake_distance=0.60, terminal_position_tolerance=0.12,
            terminal_stop_speed=0.15, residual=torch.zeros(1, 12),
            previous_residual=torch.zeros(1, 12), projected_gravity_xy=torch.zeros(1, 2),
            vertical_velocity=torch.zeros(1), success=torch.zeros(1, dtype=torch.bool),
            failed=torch.zeros(1, dtype=torch.bool), fell=torch.zeros(1, dtype=torch.bool),
            dt=0.02,
        )
        stopped, stopped_terms = residual_reward(**common, planar_speed=torch.zeros(1))
        moving, moving_terms = residual_reward(**common, planar_speed=torch.full((1,), 0.4))
        torch.testing.assert_close(stopped_terms["speed"], torch.zeros(1))
        self.assertGreater(float(stopped), float(moving))
        self.assertGreater(float(stopped_terms["terminal_stop"]), float(moving_terms["terminal_stop"]))

    def test_terminal_position_cost_rejects_longitudinal_overshoot(self):
        common = dict(
            progress_delta=torch.zeros(1), cross_track=torch.zeros(1),
            speed_error=torch.zeros(1), schedule_error=torch.zeros(1),
            yaw_error=torch.zeros(1), height_error=torch.zeros(1),
            remaining_distance=torch.zeros(1), planar_speed=torch.zeros(1),
            terminal_brake_distance=0.60, terminal_position_tolerance=0.12,
            terminal_stop_speed=0.15, residual=torch.zeros(1, 12),
            previous_residual=torch.zeros(1, 12), projected_gravity_xy=torch.zeros(1, 2),
            vertical_velocity=torch.zeros(1), success=torch.zeros(1, dtype=torch.bool),
            failed=torch.zeros(1, dtype=torch.bool), fell=torch.zeros(1, dtype=torch.bool),
            dt=0.02,
        )
        endpoint, _ = residual_reward(**common, terminal_distance=torch.zeros(1))
        overshot, _ = residual_reward(**common, terminal_distance=torch.full((1,), 0.5))
        self.assertGreater(float(endpoint), float(overshot))

    def test_terminal_distance_rejects_longitudinal_overshoot(self):
        bank = RouteBank(1, "cpu", points=101, length_m=4.0)
        bank.reset(torch.tensor([0]), stage=0)
        for x in torch.linspace(0.0, 4.0, 12):
            bank.update(torch.tensor([[x, 0.0]]))
        state = bank.update(torch.tensor([[4.5, 0.0]]))
        self.assertTrue(bool(state.near_terminal))
        self.assertAlmostEqual(float(state.cross_track), 0.0, places=6)
        self.assertAlmostEqual(float(state.terminal_distance), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
