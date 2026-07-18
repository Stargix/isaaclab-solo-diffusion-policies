"""Capability-bounded planar routes and a closed-loop command teacher.

This module deliberately has no Isaac Sim dependency.  A route is generated
before a rollout and is never fitted to the motion subsequently achieved by the
robot.  The tracker is the small high-level teacher used to turn the fixed route
and the current robot pose into the velocity command consumed by an existing
locomotion expert.

Keeping this contract in a standalone module makes it testable and, crucially,
keeps route generation agnostic to the name of the skill/checkpoint.  A caller
supplies the measured capability envelope instead of selecting a hard-coded
``walk`` or ``crouch`` preset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to [-pi, pi)."""

    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Extract yaw from WXYZ quaternions without a simulator dependency."""

    quat = np.asarray(quat_wxyz, dtype=np.float32)
    w, x, y, z = (quat[..., index] for index in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


@dataclass(frozen=True)
class CapabilityLimits:
    """Measured command envelope used by the route generator and tracker."""

    vx_min: float
    vx_max: float
    vy_abs_max: float
    wz_abs_max: float
    curvature_abs_max: float
    acceleration_abs_max: float

    def validate(self) -> None:
        if not (0.0 <= self.vx_min <= self.vx_max):
            raise ValueError("Require 0 <= vx_min <= vx_max for capability routes.")
        if min(self.vy_abs_max, self.wz_abs_max, self.curvature_abs_max, self.acceleration_abs_max) <= 0.0:
            raise ValueError("All capability limits except vx_min must be positive.")


@dataclass(frozen=True)
class TrackerGains:
    """Conservative feedback gains for the velocity-command teacher."""

    lookahead_s: float = 0.45
    longitudinal_gain: float = 0.55
    lateral_gain: float = 1.00
    heading_gain: float = 1.15
    lateral_yaw_gain: float = 0.65

    def validate(self) -> None:
        if self.lookahead_s < 0.0:
            raise ValueError("Tracker lookahead must be non-negative.")
        if min(self.longitudinal_gain, self.lateral_gain, self.heading_gain, self.lateral_yaw_gain) < 0.0:
            raise ValueError("Tracker gains must be non-negative.")


@dataclass(frozen=True)
class ReferenceRoute:
    """Dense, fixed world-frame reference sampled at the control rate."""

    pos_w: np.ndarray  # (T, 3), state at the beginning of each tick
    yaw_w: np.ndarray  # (T,)
    nominal_command_b: np.ndarray  # (T, 3), command that integrates pos[t] -> pos[t+1]
    family: str
    dt: float

    def __post_init__(self) -> None:
        if self.pos_w.ndim != 2 or self.pos_w.shape[1] != 3:
            raise ValueError("pos_w must have shape (T, 3).")
        if self.yaw_w.shape != (len(self.pos_w),):
            raise ValueError("yaw_w must have shape (T,).")
        if self.nominal_command_b.shape != (len(self.pos_w), 3):
            raise ValueError("nominal_command_b must have shape (T, 3).")
        if len(self.pos_w) < 2 or self.dt <= 0.0:
            raise ValueError("A route needs at least two samples and a positive dt.")


ROUTE_FAMILIES = ("straight", "arc_left", "arc_right", "s_curve", "stop_go")


def _sample_family(rng: random.Random) -> str:
    return rng.choices(ROUTE_FAMILIES, weights=(0.28, 0.18, 0.18, 0.26, 0.10), k=1)[0]


def _speed_profile(
    rng: random.Random,
    steps: int,
    dt: float,
    limits: CapabilityLimits,
    startup_hold_steps: int,
    family: str,
) -> np.ndarray:
    """Generate a speed schedule with bounded acceleration and explicit stops."""

    speed = np.zeros(steps, dtype=np.float32)
    target = rng.uniform(limits.vx_min, limits.vx_max)
    ramp = max(1, int(math.ceil(max(target, 1.0e-3) / limits.acceleration_abs_max / dt)))
    start = min(max(0, startup_hold_steps), steps)
    for t in range(start, steps):
        speed[t] = min(target, target * float(t - start + 1) / float(ramp))

    # One controlled braking/restart window supplies useful near-stop examples
    # without asking the expert to execute an arbitrary discontinuity.
    if family == "stop_go" and start + 50 < steps:
        stop_start = rng.randint(start + max(20, steps // 3), max(start + 20, steps - 35))
        stop_len = rng.randint(12, min(32, steps - stop_start - 1))
        speed[stop_start: stop_start + stop_len] = 0.0
        for k in range(min(ramp, steps - stop_start - stop_len)):
            speed[stop_start + stop_len + k] = min(target, target * float(k + 1) / float(ramp))
        if stop_start + stop_len + ramp < steps:
            speed[stop_start + stop_len + ramp:] = target
    return speed


def _curvature_profile(rng: random.Random, steps: int, limits: CapabilityLimits, family: str) -> np.ndarray:
    """Sample smooth curvature.  ``w=v*kappa`` remains capability bounded."""

    max_kappa = limits.curvature_abs_max
    if family == "straight" or family == "stop_go":
        return np.zeros(steps, dtype=np.float32)
    if family in {"arc_left", "arc_right"}:
        sign = 1.0 if family == "arc_left" else -1.0
        value = sign * rng.uniform(0.30 * max_kappa, 0.75 * max_kappa)
        return np.full(steps, value, dtype=np.float32)

    # A single smooth S-turn is preferable to independent heading jumps: it
    # stays differentiable enough for a command-conditioned locomotion expert.
    amplitude = rng.uniform(0.30 * max_kappa, 0.75 * max_kappa)
    phase = np.linspace(0.0, 2.0 * math.pi, num=steps, endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(phase)).astype(np.float32)


def generate_capability_route(
    *,
    start_pos_w: np.ndarray,
    start_yaw_w: float,
    desired_height: float,
    steps: int,
    dt: float,
    limits: CapabilityLimits,
    rng: random.Random,
    startup_hold_steps: int,
    initial_lateral_offset_m: float = 0.0,
    initial_yaw_offset_rad: float = 0.0,
) -> ReferenceRoute:
    """Create one executable route from a generic velocity capability envelope.

    The initial lateral/yaw offsets are intentional recovery states.  They move
    the *reference*, not the robot, so the following teacher provides the only
    valid labels for returning to the external route.
    """

    limits.validate()
    if steps < 2:
        raise ValueError("Route length must be at least two control ticks.")
    if startup_hold_steps < 0:
        raise ValueError("startup_hold_steps must be non-negative.")

    family = _sample_family(rng)
    speed = _speed_profile(rng, steps, dt, limits, startup_hold_steps, family)
    curvature = _curvature_profile(rng, steps, limits, family)
    yaw_offset = rng.uniform(-initial_yaw_offset_rad, initial_yaw_offset_rad)
    lateral_offset = rng.uniform(-initial_lateral_offset_m, initial_lateral_offset_m)

    pos = np.zeros((steps, 3), dtype=np.float32)
    yaw = np.zeros(steps, dtype=np.float32)
    command = np.zeros((steps, 3), dtype=np.float32)
    initial_yaw = float(start_yaw_w + yaw_offset)
    pos[0] = np.asarray(start_pos_w, dtype=np.float32)
    pos[0, 0] += -math.sin(initial_yaw) * lateral_offset
    pos[0, 1] += math.cos(initial_yaw) * lateral_offset
    pos[:, 2] = float(desired_height)
    yaw[0] = initial_yaw

    for t in range(steps):
        v = float(speed[t])
        w = float(np.clip(v * curvature[t], -limits.wz_abs_max, limits.wz_abs_max))
        command[t] = (v, 0.0, w)
        if t + 1 >= steps:
            continue
        cos_yaw, sin_yaw = math.cos(float(yaw[t])), math.sin(float(yaw[t]))
        pos[t + 1, 0] = pos[t, 0] + dt * cos_yaw * v
        pos[t + 1, 1] = pos[t, 1] + dt * sin_yaw * v
        pos[t + 1, 2] = float(desired_height)
        yaw[t + 1] = float(yaw[t] + dt * w)

    return ReferenceRoute(pos, yaw, command, family, float(dt))


def tracking_command(
    route: ReferenceRoute,
    step_idx: int,
    robot_pos_w: np.ndarray,
    robot_quat_w: np.ndarray,
    limits: CapabilityLimits,
    gains: TrackerGains,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the high-level expert command and `[longitudinal,lateral,yaw]` error.

    The target advances in *reference time* with a modest preview.  Consequently
    the stored student condition (the same timed preview) and the teacher action
    are generated from the same task definition; the route itself is not reset
    to the achieved base pose when the robot drifts.
    """

    gains.validate()
    index = int(np.clip(step_idx, 0, len(route.pos_w) - 1))
    lookahead = int(round(gains.lookahead_s / route.dt))
    target_idx = min(index + lookahead, len(route.pos_w) - 1)
    robot_pos = np.asarray(robot_pos_w, dtype=np.float32)
    robot_yaw = float(yaw_from_quat_wxyz(np.asarray(robot_quat_w, dtype=np.float32)))
    delta_w = route.pos_w[target_idx, :2] - robot_pos[:2]
    cos_yaw, sin_yaw = math.cos(robot_yaw), math.sin(robot_yaw)
    longitudinal = cos_yaw * float(delta_w[0]) + sin_yaw * float(delta_w[1])
    lateral = -sin_yaw * float(delta_w[0]) + cos_yaw * float(delta_w[1])
    heading = float(wrap_to_pi(float(route.yaw_w[target_idx]) - robot_yaw))

    nominal = route.nominal_command_b[index]
    vx = float(np.clip(nominal[0] + gains.longitudinal_gain * longitudinal, -limits.vx_max, limits.vx_max))
    vy = float(np.clip(gains.lateral_gain * lateral, -limits.vy_abs_max, limits.vy_abs_max))
    wz = float(np.clip(
        nominal[2] + gains.heading_gain * heading + gains.lateral_yaw_gain * lateral,
        -limits.wz_abs_max,
        limits.wz_abs_max,
    ))
    return np.asarray((vx, vy, wz), dtype=np.float32), np.asarray((longitudinal, lateral, heading), dtype=np.float32)
