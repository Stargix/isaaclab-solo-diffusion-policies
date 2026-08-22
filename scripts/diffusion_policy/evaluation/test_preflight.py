"""Unit tests for causal preflight selection; no Isaac Sim dependency."""

from __future__ import annotations

import unittest

from .summarize_preflight import _height_causality, decide


def _candidate(exec_horizon: int, *, survival: float, causal: float, speed_error: float) -> dict:
    return {
        "checkpoint_sha256": "ABC",
        "git_commit": "deadbeef",
        "exec_horizon": exec_horizon,
        "survival_rate": survival,
        "mean_absolute_horizon_speed_ratio_error": speed_error,
        "mean_terminal_position_error_m": 0.10,
        "mean_cross_track_rmse_m": 0.03,
        "height_causality": {"correct_direction_fraction": causal},
    }


class PreflightDecisionTest(unittest.TestCase):
    def test_rejects_both_unsafe_contracts(self) -> None:
        first = _candidate(1, survival=0.70, causal=1.0, speed_error=0.1)
        second = _candidate(4, survival=0.90, causal=0.5, speed_error=0.1)
        self.assertEqual(decide(first, second)["status"], "base_not_ready")

    def test_safety_has_priority_over_speed(self) -> None:
        first = _candidate(1, survival=0.95, causal=1.0, speed_error=0.20)
        second = _candidate(4, survival=0.90, causal=1.0, speed_error=0.01)
        self.assertEqual(decide(first, second)["exec_horizon"], 1)

    def test_speed_breaks_safety_equivalent_tie(self) -> None:
        first = _candidate(1, survival=0.95, causal=1.0, speed_error=0.08)
        second = _candidate(4, survival=0.94, causal=1.0, speed_error=0.03)
        self.assertEqual(decide(first, second)["exec_horizon"], 4)

    def test_height_causality_uses_paired_conditions(self) -> None:
        rows = [
            {
                "path_shape": "straight", "requested_speed": "0.4", "repeat": "0",
                "requested_height": "0.1705", "achieved_height_mean": "0.18",
            },
            {
                "path_shape": "straight", "requested_speed": "0.4", "repeat": "0",
                "requested_height": "0.2932", "achieved_height_mean": "0.28",
            },
        ]
        result = _height_causality(rows)
        self.assertEqual(result["pairs"], 1)
        self.assertEqual(result["correct_direction_fraction"], 1.0)
        self.assertGreater(result["mean_response_slope"], 0.8)


if __name__ == "__main__":
    unittest.main()
