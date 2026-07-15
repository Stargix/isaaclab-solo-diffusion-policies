from __future__ import annotations

import unittest

import numpy as np

from path_tracker import PathCommandTracker, PathTrackerConfig, align_path_to_pose


class PathTrackerTests(unittest.TestCase):
    def test_straight_path_command_and_height_preview(self) -> None:
        path = np.zeros((81, 3), dtype=np.float32)
        path[:, 0] = np.arange(81) * 0.05
        path[:, 2] = np.where(path[:, 0] < 1.0, 0.2932, 0.1705)
        tracker = PathCommandTracker(path, PathTrackerConfig(desired_speed=0.4))
        command = tracker.command(np.asarray((0.0, 0.0, 0.2932)), np.asarray((1.0, 0.0, 0.0, 0.0)))
        self.assertAlmostEqual(float(command[0]), 0.4, places=5)
        self.assertAlmostEqual(float(command[1]), 0.0, places=5)
        self.assertAlmostEqual(float(command[2]), 0.0, places=5)
        self.assertAlmostEqual(float(command[3]), 0.2932, places=5)

        command = tracker.command(np.asarray((0.25, 0.0, 0.2932)), np.asarray((1.0, 0.0, 0.0, 0.0)))
        self.assertAlmostEqual(float(command[3]), 0.1705, places=5)

    def test_progress_cannot_jump_to_far_crossing(self) -> None:
        path = np.zeros((120, 3), dtype=np.float32)
        path[:, 0] = np.arange(120) * 0.05
        path[100, :2] = path[10, :2]
        tracker = PathCommandTracker(path, PathTrackerConfig(search_forward=20))
        tracker.progress[0] = 10
        tracker.command(np.asarray((*path[10, :2], 0.2)), np.asarray((1.0, 0.0, 0.0, 0.0)))
        self.assertLess(int(tracker.progress[0]), 31)

    def test_relative_path_alignment_preserves_height(self) -> None:
        path = np.asarray(((0.0, 0.0, 0.2932), (1.0, 0.0, 0.1705)), dtype=np.float32)
        half = np.sqrt(0.5)
        aligned = align_path_to_pose(path, np.asarray((0.5, 0.5, 0.3)), np.asarray((half, 0, 0, half)))
        np.testing.assert_allclose(aligned[:, :2], ((0.5, 0.5), (0.5, 1.5)), atol=1e-6)
        np.testing.assert_allclose(aligned[:, 2], path[:, 2])


if __name__ == "__main__":
    unittest.main()
