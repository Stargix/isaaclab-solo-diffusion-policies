"""Unit tests for Phase A route metrics; no Isaac Sim dependency."""

from __future__ import annotations

import unittest

import numpy as np

from .metrics import (
    compute_first_task_success,
    compute_route_metrics,
    point_at_progress,
    project_trajectory_to_polyline,
    valid_post_step_mask,
)


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

    def test_post_step_mask_excludes_auto_reset_sample(self) -> None:
        np.testing.assert_array_equal(
            valid_post_step_mask(5, 3), [True, True, False, False, False]
        )
        np.testing.assert_array_equal(
            valid_post_step_mask(3, 1), [False, False, False]
        )

    def test_perfect_schedule_scores_one(self) -> None:
        positions = np.asarray([[0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]])
        metrics = compute_route_metrics(
            positions, self.path, requested_speed_m_s=0.5, dt=0.5, horizon_steps=4,
        )
        self.assertAlmostEqual(metrics.completion_ratio, 1.0)
        self.assertAlmostEqual(metrics.horizon_speed_ratio, 1.0)
        self.assertTrue(metrics.arrived)
        self.assertAlmostEqual(metrics.arrival_time_s, 2.0)
        self.assertAlmostEqual(metrics.arrival_speed_ratio, 1.0)
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
        self.assertFalse(metrics.arrived)
        self.assertTrue(np.isnan(metrics.arrival_speed_ratio))

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

    def test_finite_route_endpoint_wait_separates_horizon_and_arrival_speed(self) -> None:
        positions = np.asarray(
            [[1.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]], dtype=np.float32
        )
        metrics = compute_route_metrics(
            positions, self.path, requested_speed_m_s=1.0, dt=1.0, horizon_steps=4,
        )
        self.assertAlmostEqual(metrics.target_progress_m, 2.0)
        self.assertAlmostEqual(metrics.horizon_speed_ratio, 0.5)
        self.assertTrue(metrics.arrived)
        self.assertAlmostEqual(metrics.arrival_time_s, 2.0)
        self.assertAlmostEqual(metrics.arrival_speed_ratio, 1.0)
        self.assertAlmostEqual(metrics.completion_ratio, 1.0)

    def test_arrival_time_interpolates_between_samples(self) -> None:
        positions = np.asarray([[0.4, 0.0], [1.2, 0.0]], dtype=np.float32)
        metrics = compute_route_metrics(
            positions, self.path, requested_speed_m_s=0.5, dt=1.0, horizon_steps=2,
        )
        # Target progress is 1 m: 0.6 / 0.8 of the second interval.
        self.assertAlmostEqual(metrics.arrival_time_s, 1.75, places=6)
        self.assertAlmostEqual(metrics.arrival_mean_speed_m_s, 1.0 / 1.75, places=6)

    def test_self_crossing_does_not_jump_to_distant_branch(self) -> None:
        path = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [1.0, -1.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        positions = np.asarray([[0.0, 0.0], [0.2, 0.2], [0.5, 0.5]], dtype=np.float32)
        progress, _ = project_trajectory_to_polyline(positions, path, search_forward=2)
        self.assertLess(progress[0], 0.01)
        self.assertTrue(np.all(np.diff(progress) >= -1.0e-6))

    def test_task_success_requires_joint_terminal_constraints(self) -> None:
        positions = np.asarray([[0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]])
        result = compute_first_task_success(
            positions,
            self.path,
            yaws_rad=np.zeros(4),
            heights_m=np.full(4, 0.2932),
            requested_speed_m_s=0.5,
            target_progress_m=1.0,
            target_yaw_rad=0.0,
            target_height_m=0.2932,
            dt=0.5,
            start_position_xy=np.asarray([0.0, 0.0]),
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.time_s, 2.0)
        self.assertAlmostEqual(result.mean_speed_error_m_s, 0.0)

        wrong_pose = compute_first_task_success(
            positions,
            self.path,
            yaws_rad=np.ones(4),
            heights_m=np.full(4, 0.2932),
            requested_speed_m_s=0.5,
            target_progress_m=1.0,
            target_yaw_rad=0.0,
            target_height_m=0.2932,
            dt=0.5,
            start_position_xy=np.asarray([0.0, 0.0]),
        )
        self.assertFalse(wrong_pose.success)

    def test_task_success_cannot_recover_from_training_corridor_failure(self) -> None:
        positions = np.asarray([[0.25, 0.7], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]])
        result = compute_first_task_success(
            positions,
            self.path,
            yaws_rad=np.zeros(4),
            heights_m=np.full(4, 0.2932),
            requested_speed_m_s=0.5,
            target_progress_m=1.0,
            target_yaw_rad=0.0,
            target_height_m=0.2932,
            dt=0.5,
            start_position_xy=np.asarray([0.0, 0.0]),
        )
        self.assertFalse(result.success)

    def test_task_success_cannot_recover_from_training_overshoot(self) -> None:
        endpoint_path = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = compute_first_task_success(
            np.asarray([[1.6, 0.0], [1.0, 0.0]]),
            endpoint_path,
            yaws_rad=np.zeros(2),
            heights_m=np.full(2, 0.2932),
            requested_speed_m_s=0.5,
            target_progress_m=1.0,
            target_yaw_rad=0.0,
            target_height_m=0.2932,
            dt=1.0,
            start_position_xy=np.asarray([0.0, 0.0]),
        )
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
