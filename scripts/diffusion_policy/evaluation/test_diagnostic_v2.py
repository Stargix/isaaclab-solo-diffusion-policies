"""Unit tests for the evaluation-v2 pure post-processing helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.diffusion_policy.evaluation.diagnostic_v2 import (
    TraceBundle,
    _interpolate_first_passage,
    analyze_trace,
    route_descriptors,
)
from scripts.diffusion_policy.evaluation.route_bank import DIAGNOSTIC_FAMILIES, generate_bank


class DiagnosticV2Test(unittest.TestCase):
    def test_route_descriptors_identify_turn_and_straight(self) -> None:
        straight = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        corner = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAlmostEqual(route_descriptors(straight)["max_local_turn_deg"], 0.0)
        self.assertAlmostEqual(route_descriptors(corner)["max_local_turn_deg"], 90.0)

    def test_alignment_is_in_route_distance_not_time(self) -> None:
        steps = 6
        path = np.column_stack((np.linspace(0.0, 2.0, 5), np.zeros(5)))
        progress = np.asarray([[0.0], [0.2], [0.7], [1.0], [1.5], [2.0]])
        required = np.asarray([[0.2932], [0.2932], [0.2932], [0.1705], [0.1705], [0.1705]])
        trace = {
            "time_s": np.arange(steps, dtype=np.float32) * 0.1,
            "positions_xy": np.concatenate((progress[..., None], np.zeros((steps, 1, 1))), axis=2),
            "progress_m": progress,
            "tangent_speed_m_s": np.full((steps, 1), 0.5),
            "achieved_height_m": required.copy(),
            "required_height_m": required,
            "valid": np.ones((steps, 1), dtype=bool),
            "requested_speed_m_s": np.asarray([0.5]),
        }
        metadata = {"scenarios": [{
            "path_shape": "unit", "repeat": 0, "requested_speed_m_s": 0.5,
            "transition_direction": "walk_to_crouch", "transition_fraction": 0.5,
            "path_xy": path.tolist(), "path_cumulative_m": np.linspace(0.0, 2.0, 5).tolist(),
        }]}
        metrics, events, aligned = analyze_trace(
            TraceBundle(trace, metadata), alignment_window_m=(-0.2, 0.2), alignment_step_m=0.1
        )
        self.assertEqual(len(metrics), 1)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(float(events[0]["transition_progress_m"]), 1.0)
        self.assertEqual(aligned["height_error_m"].shape, (1, 5))
        self.assertAlmostEqual(float(metrics[0]["final_schedule_debt_m"]), -1.75)

    def test_first_passage_alignment_does_not_mix_backward_motion(self) -> None:
        progress = np.asarray([0.0, 0.5, 1.0, 0.7, 1.1])
        values = np.asarray([0.0, 0.5, 1.0, 99.0, 1.1])
        interpolated = _interpolate_first_passage(progress, values, np.asarray([0.7]))
        self.assertAlmostEqual(float(interpolated[0]), 0.7)

    def test_repeated_transition_metrics_use_local_plateaus(self) -> None:
        steps = 13
        progress = np.linspace(0.0, 3.0, steps)[:, None]
        required = np.where(progress < 1.0, 0.2932, np.where(progress < 2.0, 0.1705, 0.2932))
        achieved = required.copy()
        achieved[progress[:, 0] >= 2.0, 0] = 0.1705
        path = np.column_stack((np.linspace(0.0, 3.0, steps), np.zeros(steps)))
        trace = {
            "time_s": np.arange(steps, dtype=np.float32) * 0.25,
            "positions_xy": np.concatenate((progress[..., None], np.zeros((steps, 1, 1))), axis=2),
            "progress_m": progress,
            "tangent_speed_m_s": np.ones((steps, 1)),
            "achieved_height_m": achieved,
            "required_height_m": required,
            "valid": np.ones((steps, 1), dtype=bool),
            "requested_speed_m_s": np.asarray([1.0]),
        }
        metadata = {"scenarios": [{
            "path_shape": "unit", "repeat": 0, "requested_speed_m_s": 1.0,
            "height_profile": "interleaved", "path_xy": path.tolist(),
            "path_cumulative_m": np.linspace(0.0, 3.0, steps).tolist(),
        }]}
        _, events, _ = analyze_trace(
            TraceBundle(trace, metadata), alignment_window_m=(-0.2, 0.2), alignment_step_m=0.1
        )
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(float(events[0]["post_height_mae_m"]), 0.0)
        self.assertGreater(float(events[1]["post_height_mae_m"]), 0.1)

    def test_geometry_sweep_families_are_valid_routes(self) -> None:
        routes = generate_bank(
            DIAGNOSTIC_FAMILIES,
            routes_per_family=1,
            base_seed=17,
            split="unit",
        )
        self.assertEqual(len(routes), len(DIAGNOSTIC_FAMILIES))
        for route in routes:
            self.assertAlmostEqual(route.length_m, 4.0, places=4)


if __name__ == "__main__":
    unittest.main()
