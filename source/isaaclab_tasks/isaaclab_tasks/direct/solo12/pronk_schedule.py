"""Bounded training exposure; progress is NOT a gait acceptance criterion."""

import math


def ramp_fraction(step: int, start: int, end: int) -> float:
    if not 0 <= start < end:
        raise ValueError("A curriculum ramp requires 0 <= start < end.")
    return min(1.0, max(0.0, (step - start) / (end - start)))


def slow_flight_multiplier(speed, low: float = 0.25, high: float = 0.5):
    """Double the settled-flight bonus at slow commands, taper to 1 at .5 m/s.

    Eligibility and valid landing are still enforced by the cycle tracker.
    This cannot reward stops, free falls or aborted flights.
    """
    if not all(math.isfinite(v) for v in (low, high)) or not 0 < low < high:
        raise ValueError("Slow-flight weighting needs 0 < low < high.")
    return 1.0 + ((high - speed) / (high - low)).clamp(0.0, 1.0)
