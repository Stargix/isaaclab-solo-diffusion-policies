"""Tests for the frozen WCT geometry bank and active-route metrics."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .metrics import compute_active_tracking_metrics
from .route_bank import generate_bank, load_route_bank, save_route_bank
from .analyze_temporal_allocation import _leg_indices


class RouteBankTest(unittest.TestCase):
    def test_solo12_rear_leg_names_are_recognized(self) -> None:
        self.assertEqual(
            _leg_indices(["FL_calf", "FR_calf", "RL_calf", "RR_calf"]),
            {"fl": 0, "fr": 1, "hl": 2, "hr": 3},
        )

    def test_round_trip_is_pickle_free_and_deterministic(self) -> None:
        first = generate_bank(
            ("procedural", "ood_corner"),
            routes_per_family=3,
            base_seed=77,
            split="test",
        )
        second = generate_bank(
            ("procedural", "ood_corner"),
            routes_per_family=3,
            base_seed=77,
            split="test",
        )
        for route_a, route_b in zip(first, second):
            np.testing.assert_array_equal(route_a.xy, route_b.xy)
            np.testing.assert_array_equal(route_a.yaw, route_b.yaw)
            self.assertAlmostEqual(route_a.length_m, 4.0, places=5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.npz"
            manifest = save_route_bank(path, first)
            loaded = load_route_bank(path)
        self.assertEqual(manifest["routes"], 6)
        self.assertEqual(set(loaded), {(family, repeat) for family in ("procedural", "ood_corner") for repeat in range(3)})
        np.testing.assert_array_equal(loaded[("ood_corner", 2)].xy, first[-1].xy)

    def test_active_tracking_stops_at_arrival_and_stratifies_transition(self) -> None:
        path = np.column_stack((np.linspace(0.0, 4.0, 101), np.zeros(101)))
        # The post-arrival outlier must not contaminate active metrics.
        positions = np.asarray(
            [[0.5, 0.02], [1.5, 0.04], [2.0, 0.08], [3.0, 0.03], [4.0, 0.02], [4.0, 2.0]]
        )
        heights = np.asarray([0.29, 0.29, 0.24, 0.17, 0.17, 0.40])
        target = np.asarray([0.29, 0.29, 0.17, 0.17, 0.17, 0.17])
        metrics = compute_active_tracking_metrics(
            positions,
            path,
            heights_m=heights,
            target_heights_m=target,
            start_position_xy=np.asarray([0.0, 0.0]),
            active_steps=5,
            transition_progress_m=2.0,
            transition_margin_m=0.25,
        )
        self.assertEqual(metrics.steps, 5)
        self.assertLess(metrics.cross_track_max_m, 0.1)
        self.assertAlmostEqual(metrics.transition_cross_track_rmse_m, 0.08, places=6)
        self.assertGreater(metrics.transition_height_mae_m, 0.06)


if __name__ == "__main__":
    unittest.main()
