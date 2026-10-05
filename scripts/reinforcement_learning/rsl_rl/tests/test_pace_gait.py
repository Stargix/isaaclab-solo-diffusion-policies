"""Unit tests for the simulator-independent pace gait reward helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER_PATH = (
    REPO_ROOT
    / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/pace_gait.py"
)
SPEC = importlib.util.spec_from_file_location("pace_gait_under_test", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load pure pace helper module from {HELPER_PATH}")
PACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACE)


def _phase_timers(contact_state: tuple[int, int, int, int], duration: float = 0.1):
    contact = torch.tensor(contact_state, dtype=torch.float32).unsqueeze(0)
    air_time = (1.0 - contact) * duration
    contact_time = contact * duration
    forces = torch.zeros((1, 4, 3), dtype=torch.float32)
    forces[0, contact.bool()[0], 2] = 5.0
    return air_time, contact_time, forces


class PaceGaitRewardTests(unittest.TestCase):
    def test_clock_free_timer_score_prefers_ipsilateral_alternation(self):
        pace = _phase_timers((1, 0, 1, 0))
        diagonal = _phase_timers((1, 0, 0, 1))
        pace_score = PACE.pace_timing_score(pace[0], pace[1])
        diagonal_score = PACE.pace_timing_score(diagonal[0], diagonal[1])
        self.assertAlmostEqual(float(pace_score.item()), 1.0, places=5)
        # The timer term is intentionally soft (as in Isaac Lab); the separate
        # instantaneous topology term supplies the strong gait-family contrast.
        self.assertLess(float(diagonal_score.item()), 0.60)

    def test_timer_score_matches_the_isaaclab_sync_async_formula(self):
        air = torch.tensor([[0.03, 0.17, 0.07, 0.11]], dtype=torch.float64)
        contact = torch.tensor([[0.19, 0.04, 0.13, 0.08]], dtype=torch.float64)
        max_error_s = 0.20
        std_s2 = 0.10
        error_sum = torch.zeros(1, dtype=torch.float64)
        for first, second in PACE.SYNCED_FEET:
            error_sum += torch.clamp((air[:, first] - air[:, second]).square(), max=max_error_s**2)
            error_sum += torch.clamp(
                (contact[:, first] - contact[:, second]).square(), max=max_error_s**2
            )
        for first, second in PACE.ASYNC_FEET:
            error_sum += torch.clamp((air[:, first] - contact[:, second]).square(), max=max_error_s**2)
            error_sum += torch.clamp((contact[:, first] - air[:, second]).square(), max=max_error_s**2)
        expected = torch.exp(-error_sum / std_s2)
        actual = PACE.pace_timing_score(air, contact, std_s2=std_s2, max_error_s=max_error_s)
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12))

    def test_topology_prefers_pace_over_diagonal_bound_and_static_patterns(self):
        pace = _phase_timers((1, 0, 1, 0))[2]
        diagonal = _phase_timers((1, 0, 0, 1))[2]
        bound = _phase_timers((1, 1, 0, 0))[2]
        all_support = _phase_timers((1, 1, 1, 1))[2]
        all_flight = _phase_timers((0, 0, 0, 0))[2]
        pace_score = float(PACE.pace_contact_topology_score(pace).item())
        self.assertGreater(pace_score, 0.95)
        for non_pace in (diagonal, bound, all_support, all_flight):
            self.assertLess(float(PACE.pace_contact_topology_score(non_pace).item()), 0.05)

    def test_zero_timers_and_static_single_pair_cannot_get_gait_reward(self):
        zeros = torch.zeros((2, 4), dtype=torch.float32)
        contacts = torch.zeros((2, 4, 3), dtype=torch.float32)
        commands = torch.tensor([[0.7, 0.0], [0.7, 0.0]])
        tracking = torch.ones(2)
        components = PACE.pace_gait_reward_components(
            zeros,
            zeros,
            contacts,
            commands,
            tracking,
        )
        self.assertTrue(torch.all(components["gait"] == 0.0))

        long_mode = torch.full((1, 4), 0.8)
        long_mode_forces = _phase_timers((1, 0, 1, 0))[2]
        guarded = PACE.pace_gait_reward_components(
            long_mode,
            long_mode,
            long_mode_forces,
            torch.tensor([[0.7, 0.0]]),
            torch.ones(1),
        )
        self.assertEqual(float(guarded["mode_ok"].item()), 0.0)
        self.assertEqual(float(guarded["gait"].item()), 0.0)

    def test_stopping_disables_only_the_gait_factor(self):
        air, contact, forces = _phase_timers((1, 0, 1, 0))
        components = PACE.pace_gait_reward_components(
            air,
            contact,
            forces,
            torch.zeros((1, 2)),
            torch.ones(1),
        )
        self.assertEqual(float(components["timing"].item()), 1.0)
        self.assertEqual(float(components["gait"].item()), 0.0)

    def test_left_right_and_front_back_reflections_preserve_pace_scores(self):
        air, contact, forces = _phase_timers((1, 0, 1, 0))
        permutations = (torch.tensor([1, 0, 3, 2]), torch.tensor([2, 3, 0, 1]))
        timing = PACE.pace_timing_score(air, contact)
        topology = PACE.pace_contact_topology_score(forces)
        for permutation in permutations:
            self.assertTrue(torch.allclose(timing, PACE.pace_timing_score(air[:, permutation], contact[:, permutation])))
            self.assertTrue(
                torch.allclose(topology, PACE.pace_contact_topology_score(forces[:, permutation]))
            )

    def test_bad_shapes_and_nonpositive_scales_fail_fast(self):
        with self.assertRaises(ValueError):
            PACE.pace_timing_score(torch.zeros((4,)), torch.zeros((4,)), std_s2=0.0)
        with self.assertRaises(ValueError):
            PACE.pace_timing_score(torch.zeros((1, 3)), torch.zeros((1, 3)))
        with self.assertRaises(ValueError):
            PACE.pace_contact_topology_score(torch.zeros((1, 4, 3)), softness_n=0.0)


if __name__ == "__main__":
    unittest.main()
