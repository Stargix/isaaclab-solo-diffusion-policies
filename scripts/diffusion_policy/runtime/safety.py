"""Fail-latched command admission; a simulation stop is not a hardware E-stop."""
from dataclasses import dataclass
import numpy as np


class SafetyFault(RuntimeError):
    pass


@dataclass
class CommandSupervisor:
    lower: np.ndarray
    upper: np.ndarray
    state_age_limit_s: float = .04
    plan_deadline_s: float = .08
    velocity_limit_rad_s: float = 100.
    fault: str | None = None
    last_sequence: int = -1
    last_now: float = float("-inf")

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float).copy()
        self.upper = np.asarray(self.upper, dtype=float).copy()
        if self.lower.shape != (12,) or self.upper.shape != (12,) or not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("Require twelve finite ordered joint limits")
        if not all(np.isfinite(x) and x > 0 for x in (self.state_age_limit_s, self.plan_deadline_s, self.velocity_limit_rad_s)):
            raise ValueError("Safety timing/velocity limits must be positive and finite")

    def _reject(self, reason):
        self.fault = reason
        raise SafetyFault(reason)

    def admit(self, *, q, dq, target, now, state_stamp, plan_stamp, sequence):
        if self.fault is not None:
            raise SafetyFault(f"Latched: {self.fault}; explicit external reset required")
        try:
            now, state_stamp, plan_stamp = float(now), float(state_stamp), float(plan_stamp)
        except (ValueError, TypeError):
            self._reject("invalid_clock_type")
        if not np.isfinite([now, state_stamp, plan_stamp]).all() or now < self.last_now:
            self._reject("nonfinite_or_reversed_clock")
        if state_stamp > now or plan_stamp > now:
            self._reject("future_timestamp")
        if now - state_stamp > self.state_age_limit_s:
            self._reject("stale_state")
        if now - plan_stamp > self.plan_deadline_s:
            self._reject("expired_plan")
        if isinstance(sequence, (bool, np.bool_)) or not isinstance(sequence, (int, np.integer)) or sequence <= self.last_sequence:
            self._reject("repeated_or_reversed_sequence")
        arrays = {}
        for name, value in (("q", q), ("dq", dq), ("target", target)):
            try:
                value = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                self._reject(f"invalid_{name}")
            if value.shape != (12,) or not np.isfinite(value).all():
                self._reject(f"invalid_{name}")
            arrays[name] = value
        q, dq, target = arrays["q"], arrays["dq"], arrays["target"]
        if np.any(np.asarray(target) < self.lower) or np.any(np.asarray(target) > self.upper):
            self._reject("target_outside_joint_limits")
        # Simulator constraint solvers can overshoot slightly; state admission
        # has a stated 0.05-rad tolerance, not a target clamp or hidden clipping.
        if np.any(np.asarray(q) < self.lower-.05) or np.any(np.asarray(q) > self.upper+.05):
            self._reject("state_outside_joint_limits")
        if np.any(np.abs(dq) > self.velocity_limit_rad_s):
            self._reject("joint_velocity_limit")
        self.last_sequence, self.last_now = int(sequence), float(now)
        return np.asarray(target).copy()
