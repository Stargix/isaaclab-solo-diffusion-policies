"""Unit tests for Phase A route metrics; no Isaac Sim dependency."""

from __future__ import annotations

import unittest

import numpy as np

from .metrics import compute_route_metrics, point_at_progress, project_trajectory_to_polyline


class RouteMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    def test_point_at_progress_interpolates(self) -> None:
        np.testing.assert_allclose(point_at_progress(self.path, 1.25), [1.25, 0.0])

    def test_projection_keeps_longitudinal_and_cross_track_separate(self) -> None:
        positions = np.asarray([[0.2, 0.1], [0.7, -0.2], [1.4, 0.05]], dtype=np.float32)
        progress, cross_track = project_trajectory_to_polyline(positions, self.path)
        np.testing.assert_allclose(progress, [0.2, 0.7, 1.4], atol=1.0e-6)
        np.testing.assert_allclose(cross_track, [0.1, 0.2, 0.05], atol=1.0e-6)

    def test_perfect_schedule_scores_one(self) -> None:
        positions = np.asarray([[0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]])
        metrics = compute_route_metrics(
            positions, self.path, requested_speed_m_s=0.5, dt=0.5, horizon_steps=4,
        )
        self.assertAlmostEqual(metrics.completion_ratio, 1.0)
        self.assertAlmostEqual(metrics.horizon_speed_ratio, 1.0)
        self.assertAlmostEqual(metrics.schedule_mae_m, 0.0)
        self.assertAlmostEqual(metrics.terminal_position_error_m, 0.0)

    def test_early_failure_is_not_rewarded_as_correct_speed(self) -> None:
        positions = np.asarray([[0.25, 0.0], [0.5, 0.0]])
        metrics = compute_route_metrics(
            positions, self.path, requested_speed_m_s=0.5, dt=0.5, horizon_steps=4,
        )
        self.assertAlmostEqual(metrics.active_speed_ratio, 1.0)
        self.assertAlmostEqual(metrics.horizon_speed_ratio, 0.5)
        self.assertAlmostEqual(metrics.completion_ratio, 0.5)

    def test_warmup_offset_is_not_free_progress(self) -> None:
        positions = np.asarray([[0.75, 0.0], [1.0, 0.0]])
        metrics = compute_route_metrics(
            positions,
            self.path,
            requested_speed_m_s=0.5,
            dt=0.5,
            horizon_steps=2,
            start_position_xy=np.asarray([0.5, 0.0]),
        )
        self.assertAlmostEqual(metrics.final_progress_m, 0.5)
        self.assertAlmostEqual(metrics.horizon_speed_ratio, 1.0)
        self.assertAlmostEqual(metrics.terminal_position_error_m, 0.0)

    def test_self_crossing_does_not_jump_to_distant_branch(self) -> None:
        path = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [1.0, -1.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        positions = np.asarray([[0.0, 0.0], [0.2, 0.2], [0.5, 0.5]], dtype=np.float32)
        progress, _ = project_trajectory_to_polyline(positions, path, search_forward=2)
        self.assertLess(progress[0], 0.01)
        self.assertTrue(np.all(np.diff(progress) >= -1.0e-6))


if __name__ == "__main__":
    unittest.main()
