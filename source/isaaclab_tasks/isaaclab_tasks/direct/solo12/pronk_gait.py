# Copyright (c) 2022-2026 The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent recognition and scoring of completed pronk cycles."""

from __future__ import annotations

import math

import torch


def pronk_cycle_score(
    commands_xy: torch.Tensor,
    velocity_xy: torch.Tensor,
    takeoff_air_time: torch.Tensor,
    landing_time: torch.Tensor,
    completed: torch.Tensor,
    *,
    tracking_std_mps: float = 0.20,
    synchronization_std_s: float = 0.02,
    minimum_command_speed_mps: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Score one completed hop in [0, 1]; moving in place gets exactly zero.

    Velocity is measured over the cycle. Progress projects it onto the
    commanded direction, excluding sideways drift and reverse travel.
    Flight length does not increase the bonus.
    """
    batch_shape = commands_xy.shape[:-1]
    if commands_xy.shape[-1] != 2 or velocity_xy.shape != commands_xy.shape:
        raise ValueError("Commands and velocities must have identical (..., 2) shapes.")
    if takeoff_air_time.shape != (*batch_shape, 4) or landing_time.shape != takeoff_air_time.shape:
        raise ValueError("Takeoff/landing times must have matching (..., 4) shapes.")
    if completed.shape != batch_shape:
        raise ValueError("Completed-cycle mask must match the batch dimensions.")
    if not all(math.isfinite(v) and v > 0 for v in (tracking_std_mps, synchronization_std_s)):
        raise ValueError("Tracking and synchronization widths must be finite and positive.")

    speed = torch.linalg.vector_norm(commands_xy, dim=-1)
    along_command = torch.sum(velocity_xy * commands_xy, dim=-1) / speed.clamp_min(1.0e-6)
    progress = torch.clamp(along_command / speed.clamp_min(1.0e-6), min=0.0, max=1.0)
    tracking = torch.exp(-torch.sum((commands_xy - velocity_xy).square(), dim=-1) / tracking_std_mps**2)
    timing_error = torch.mean(
        (takeoff_air_time - takeoff_air_time.mean(dim=-1, keepdim=True)).square(), dim=-1
    ) + torch.mean((landing_time - landing_time.mean(dim=-1, keepdim=True)).square(), dim=-1)
    synchronization = torch.exp(-timing_error / synchronization_std_s**2)
    eligible = completed & (speed >= minimum_command_speed_mps)
    score = torch.where(eligible, progress * tracking * synchronization, torch.zeros_like(speed))
    return {"gait": score, "progress": progress, "tracking": tracking, "synchronization": synchronization}


class PronkCycleTracker:
    """Track support -> physical flight -> four-foot landing, once per cycle.

    Inputs are sampled at the control rate; contact timers can come from the
    physics-rate sensor. A reset drop cannot start a cycle before supported
    upright stance. Invalid posture, command changes and excessive duration
    discard the pending cycle instead of paying a flight reward.
    """

    def __init__(
        self, num_envs: int, device: str | torch.device, dt: float, *,
        minimum_flight_s: float = 0.025, maximum_flight_s: float = 0.20,
        landing_window_s: float = 0.04, support_window_s: float = 0.08,
        minimum_clearance_m: float = 0.003, warmup_s: float = 0.50,
        tracking_std_mps: float = 0.20, synchronization_std_s: float = 0.02,
    ):
        values = (dt, minimum_flight_s, maximum_flight_s, landing_window_s,
                  support_window_s, minimum_clearance_m, warmup_s,
                  tracking_std_mps, synchronization_std_s)
        if not all(math.isfinite(v) and v > 0.0 for v in values):
            raise ValueError("Pronk cycle thresholds must be finite and positive.")
        if maximum_flight_s < minimum_flight_s:
            raise ValueError("Maximum flight duration cannot be below its minimum.")
        self.dt = dt
        self.minimum_flight_s = minimum_flight_s
        self.maximum_flight_s = maximum_flight_s
        self.landing_window_s = landing_window_s
        self.support_window_s = support_window_s
        self.minimum_clearance_m = minimum_clearance_m
        self.warmup_s = warmup_s
        self.tracking_std_mps = tracking_std_mps
        self.synchronization_std_s = synchronization_std_s
        self.age = torch.zeros(num_envs, device=device)
        self.support_age = torch.full((num_envs, 4), float("inf"), device=device)
        self.armed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.active = torch.zeros_like(self.armed)
        self.elapsed = torch.zeros_like(self.age)
        self.flight = torch.zeros_like(self.age)
        self.takeoff_air_time = torch.zeros((num_envs, 4), device=device)
        self.landing_time = torch.full((num_envs, 4), float("inf"), device=device)
        self.velocity_sum = torch.zeros((num_envs, 2), device=device)
        self.commands = torch.zeros_like(self.velocity_sum)
        self.geometry_verified = torch.zeros_like(self.active)

    def _flight_mask(self, contacts, clearance_m):
        return ~contacts.any(dim=-1) & (clearance_m >= self.minimum_clearance_m).all(dim=-1)

    def _verification_mask(self):
        # Legacy v1-v4 already verify clearance on each timed airborne sample.
        return torch.ones_like(self.active)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        for value in (self.age, self.elapsed, self.flight, self.takeoff_air_time,
                      self.velocity_sum, self.commands):
            value[env_ids] = 0.0
        self.support_age[env_ids] = float("inf")
        self.landing_time[env_ids] = float("inf")
        self.armed[env_ids] = False
        self.active[env_ids] = False
        self.geometry_verified[env_ids] = False

    def update(self, contacts, clearance_m, air_time, commands_xy, velocity_xy, valid_pose):
        if contacts.shape != self.support_age.shape or clearance_m.shape != contacts.shape or air_time.shape != contacts.shape:
            raise ValueError("Contact, clearance and air-time inputs must have matching (N, 4) shapes.")
        self.age += self.dt
        eligible = valid_pose & (self.age >= self.warmup_s)
        self.support_age = torch.where(contacts, 0.0, self.support_age + self.dt)
        self.armed |= eligible & ~self.active & (self.support_age <= self.support_window_s).all(dim=-1)
        self.armed &= eligible
        airborne = self._flight_mask(contacts, clearance_m)
        geometric_flight = ~contacts.any(dim=-1) & (clearance_m >= self.minimum_clearance_m).all(dim=-1)
        start = self.armed & ~self.active & airborne
        self.active |= start
        self.armed &= ~start
        self.elapsed[start] = 0.0
        self.flight[start] = 0.0
        self.velocity_sum[start] = 0.0
        self.landing_time[start] = float("inf")
        self.takeoff_air_time[start] = air_time[start]
        self.commands[start] = commands_xy[start]
        self.geometry_verified[start] = False
        self.geometry_verified |= self.active & geometric_flight & eligible

        self.elapsed += self.active * self.dt
        self.flight += (self.active & airborne) * self.dt
        self.velocity_sum += self.active.unsqueeze(-1) * velocity_xy * self.dt
        new_landings = self.active.unsqueeze(-1) & contacts & ~torch.isfinite(self.landing_time)
        self.landing_time = torch.where(new_landings, self.elapsed.unsqueeze(-1), self.landing_time)
        first_landing = self.landing_time.amin(dim=-1)
        landed_all = torch.isfinite(self.landing_time).all(dim=-1)
        changed_command = (commands_xy - self.commands).abs().amax(dim=-1) > 1.0e-6
        invalid = self.active & (
            ~eligible | changed_command
            | (self.elapsed > self.maximum_flight_s + self.landing_window_s + 1.0e-6)
            | (self.flight > self.maximum_flight_s + 1.0e-6)
            | (self.elapsed - first_landing > self.landing_window_s + 1.0e-6)
        )
        completed = (self.active & landed_all & ~invalid
                     & (self.flight >= self.minimum_flight_s - 1.0e-6) & self._verification_mask())
        average_velocity = self.velocity_sum / self.elapsed.clamp_min(self.dt).unsqueeze(-1)
        landing_times_finite = torch.where(torch.isfinite(self.landing_time), self.landing_time, 0.0)
        components = pronk_cycle_score(
            self.commands, average_velocity, self.takeoff_air_time, landing_times_finite, completed,
            tracking_std_mps=self.tracking_std_mps, synchronization_std_s=self.synchronization_std_s,
        )
        components.update(completed=completed, physical_flight=geometric_flight & eligible,
                          unloaded_flight=~contacts.any(dim=-1) & eligible,
                          flight_duration=self.flight.clone())
        finished = completed | invalid | (self.active & landed_all)
        self.active &= ~finished
        self.armed &= ~finished
        self.support_age[invalid | ~eligible] = float("inf")
        return components


class SettledPronkCycleTracker(PronkCycleTracker):
    """Confirm a substantive flight only after a supported, upright landing.

    The reward quality ramps between minimum_flight_s and target_flight_s.
    Multiplying it by the returned flight duration rewards validated airborne
    seconds instead of hop counts. Brief contact gaps and extra hop frequency
    therefore do not receive the same bonus as the original spring-like motion.
    """

    def __init__(self, *args, target_flight_s: float = 0.12, landing_settle_s: float = 0.06, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.minimum_flight_s < target_flight_s <= self.maximum_flight_s:
            raise ValueError("Target flight must exceed the minimum and not exceed the maximum.")
        if not math.isfinite(landing_settle_s) or landing_settle_s < self.dt:
            raise ValueError("Landing settling must span at least one control step.")
        self.target_flight_s = target_flight_s
        self.landing_settle_s = landing_settle_s
        self.pending = torch.zeros_like(self.active)
        self.pending_age = torch.zeros_like(self.age)
        self.pending_flight = torch.zeros_like(self.age)
        self.pending_commands = torch.zeros_like(self.commands)
        self.pending_components = {
            name: torch.zeros_like(self.age) for name in ("gait", "progress", "tracking", "synchronization")
        }

    def _duration_quality(self, duration):
        return ((duration - self.minimum_flight_s)
                / (self.target_flight_s - self.minimum_flight_s)).clamp(0.0, 1.0)

    def _bounced_mask(self, contacts, clearance_m):
        return self.active & torch.isfinite(self.landing_time).any(dim=-1) & ~contacts.any(dim=-1)

    def _landing_supported_mask(self, contacts, clearance_m):
        return contacts.any(dim=-1) & (self.support_age <= self.landing_settle_s).all(dim=-1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.pending[env_ids] = False
        for tensor in (self.pending_age, self.pending_flight, self.pending_commands,
                       *self.pending_components.values()):
            tensor[env_ids] = 0.0

    def update(self, contacts, clearance_m, air_time, commands_xy, velocity_xy, valid_pose):
        # Once touchdown has begun, another complete loss of support is a
        # bounced landing, not a longer continuous flight.
        bounced = self._bounced_mask(contacts, clearance_m)
        result = super().update(
            contacts, clearance_m, air_time, commands_xy, velocity_xy, valid_pose & ~bounced
        )
        candidate = result["completed"]
        self.pending |= candidate
        self.pending_age[candidate] = 0.0
        self.pending_flight[candidate] = result["flight_duration"][candidate]
        self.pending_commands[candidate] = commands_xy[candidate]
        for name, tensor in self.pending_components.items():
            tensor[candidate] = result[name][candidate]

        changed_command = (commands_xy - self.pending_commands).abs().amax(dim=-1) > 1.0e-6
        # Use recent per-foot support to tolerate single-frame PhysX force
        # flicker, but never confirm an airborne or failed landing.
        supported = self._landing_supported_mask(contacts, clearance_m)
        invalid = self.pending & (~valid_pose | changed_command | ~supported)
        self.pending &= ~invalid
        self.pending_age += self.pending * self.dt
        confirmed = self.pending & (self.pending_age >= self.landing_settle_s - 1.0e-6)
        flight_quality = self._duration_quality(self.pending_flight)
        for name, tensor in self.pending_components.items():
            result[name] = torch.where(confirmed, tensor, torch.zeros_like(tensor))
        result["gait"] *= flight_quality
        result["completed"] = confirmed
        result["flight_duration"] = torch.where(confirmed, self.pending_flight, torch.zeros_like(self.age))
        result["flight_quality"] = torch.where(confirmed, flight_quality, torch.zeros_like(self.age))
        result["candidate_completed"] = candidate
        self.pending &= ~confirmed
        return result


class IntervalPronkCycleTracker(SettledPronkCycleTracker):
    """Time a complete unloaded interval, verify its geometry at the apex.

    The common airborne interval starts at the first simultaneous unloading and
    ends at the first foot contact. Near-ground portions count towards duration;
    all four foot colliders must nevertheless clear the configured threshold at
    least once during that SAME interval. A contact/bounce interrupts it. Reset
    drops, failed landings, command changes and excessive duration still reject
    payout. This does not reinterpret force-only sensor gaps as genuine hops.
    """

    def _flight_mask(self, contacts, clearance_m):
        # Once the first foot touches down, timing is over. Force chatter
        # during a partial landing must not append a second unloaded interval.
        landing_started = self.active & torch.isfinite(self.landing_time).any(dim=-1)
        return ~contacts.any(dim=-1) & ~landing_started

    def _verification_mask(self):
        return self.geometry_verified

    def _bounced_mask(self, contacts, clearance_m):
        # A one-frame force gap at ground level is not a second physical hop.
        return (super()._bounced_mask(contacts, clearance_m)
                & (clearance_m >= self.minimum_clearance_m).all(dim=-1))

    def _landing_supported_mask(self, contacts, clearance_m):
        recent_support = (self.support_age <= self.landing_settle_s).all(dim=-1)
        support_now = contacts.any(dim=-1) | (clearance_m < self.minimum_clearance_m).any(dim=-1)
        return recent_support & support_now

    def _duration_quality(self, duration):
        # A modest valid hop has a nonzero acquisition signal. Total payout
        # scales as D^3 below target, rather than equally rewarding microhops.
        return (duration / self.target_flight_s).clamp(0.0, 1.0).square()
