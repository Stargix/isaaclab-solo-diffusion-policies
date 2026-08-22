"""Unit tests for Phase A capability classification."""

from __future__ import annotations

import unittest

from .summarize_capability_audit import _expected_scenarios, decide


GATES = {
    "minimum_overall_survival_rate": 0.9,
    "minimum_route_speed_group_survival_rate": 7 / 9,
    "maximum_mean_absolute_speed_ratio_error": 0.2,
    "maximum_mean_cross_track_rmse_m": 0.15,
    "maximum_mean_height_absolute_error_m": 0.03,
    "minimum_height_causal_direction_fraction": 0.9,
    "minimum_route_success_rate": 0.8,
    "minimum_random_height_survival_rate": 0.9,
    "maximum_random_height_absolute_error_m": 0.03,
}


def _run(*, survival: float = 0.95, speed_error: float = 0.1, success: float = 0.9) -> dict:
    return {
        "survival_rate": survival,
        "route_success_rate": success,
        "mean_absolute_speed_ratio_error": speed_error,
        "mean_cross_track_rmse_m": 0.05,
        "mean_height_absolute_error_m": 0.01,
        "height_causality": {"correct_direction_fraction": 1.0},
        "by_route_and_speed": [{"survival_rate": survival}],
    }


class CapabilityAuditTest(unittest.TestCase):
    def test_scenario_count_supports_constant_and_profile_runs(self) -> None:
        self.assertEqual(_expected_scenarios({
            "path_shapes": ["a", "b"], "speeds": [1, 2], "path_heights": [0.2, 0.3], "repeats": 3,
        }), 24)
        self.assertEqual(_expected_scenarios({
            "path_shapes": ["a", "b"], "speeds": [1, 2], "path_height": 0.2, "repeats": 3,
        }), 12)

    def test_sufficient_base_does_not_trigger_rl(self) -> None:
        result = decide(_run(), _run(), GATES)
        self.assertEqual(result["status"], "base_task_sufficient")

    def test_timing_only_failure_selects_command_space(self) -> None:
        result = decide(_run(speed_error=0.5, success=0.1), _run(), GATES)
        self.assertEqual(result["status"], "command_timing_correction_candidate")

    def test_safety_failure_rejects_online_optimization(self) -> None:
        result = decide(_run(survival=0.7), _run(), GATES)
        self.assertEqual(result["status"], "base_envelope_not_ready")


if __name__ == "__main__":
    unittest.main()
