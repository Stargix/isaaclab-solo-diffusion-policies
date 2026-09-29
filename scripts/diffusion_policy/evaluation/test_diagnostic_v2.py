"""Unit tests for the evaluation-v2 pure post-processing helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.diffusion_policy.evaluation.diagnostic_v2 import (
    TraceBundle,
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
