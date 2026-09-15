"""Hybrid smooth/waypoint route geometry for the second procedural ablation.

This module intentionally does not change :mod:`procedural_routes`.  The
existing ``supported_procedural_v1`` distribution remains reproducible, while
this candidate adds two continuously randomized waypoint families:

* rounded polylines, whose heading changes over a finite arc interval;
* hard polylines, whose tangent changes at the sampled waypoint.

The sampler is vectorized for the thousands of environments used by DPPO and
keeps every route at the same 4 m arc length and spatial discretization.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch

from .procedural_routes import SupportedProceduralRouteBank


HYBRID_ROUTE_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class WaypointGeometry:
    """A batch of sampled waypoint routes and their auditable parameters."""

    xy: torch.Tensor
    yaw: torch.Tensor
    arc: torch.Tensor
    segment_count: torch.Tensor
    segment_lengths: torch.Tensor
    turn_angles: torch.Tensor
    corner_radii: torch.Tensor


@dataclass(frozen=True)
class SmoothGeometry:
    """A batch of coherent smooth paths and their continuous parameters."""

    xy: torch.Tensor
    yaw: torch.Tensor
    arc: torch.Tensor
    curvature: torch.Tensor
    target_peak_curvature: torch.Tensor
    curvature_bias: torch.Tensor
    curvature_amplitude: torch.Tensor
    curvature_cycles: torch.Tensor
    curvature_phase: torch.Tensor


def sample_coherent_smooth_geometry(
    count: int,
    device: torch.device | str,
    *,
    points: int = 101,
    length_m: float = 4.0,
    max_curvature_rad_m: float = 0.8,
    max_abs_heading_deg: float = 135.0,
) -> SmoothGeometry:
    """Sample continuous arc/S geometry without a finite shape catalogue.

    A constant curvature bias supplies sustained arcs, while a continuously
    parameterized low-frequency sinusoid supplies C/S variation.  Half of the
    hybrid bank still uses the exact v1 smooth process, so this component is
    intentionally biased away from near-zero curvature.
    """

    device = torch.device(device)
    if count < 0:
        raise ValueError("count must be non-negative.")
    if points < 16 or length_m <= 0.0 or max_curvature_rad_m < 0.25:
        raise ValueError("Invalid coherent-smooth geometry dimensions.")
    if not 0.0 < max_abs_heading_deg <= 180.0:
        raise ValueError("max_abs_heading_deg must lie in (0, 180].")

    ds = length_m / float(points - 1)
    midpoint_s = (torch.arange(points - 1, device=device) + 0.5) * ds
    bias = torch.empty(count, device=device).uniform_(-0.55, 0.55)
    amplitude = torch.empty(count, device=device).uniform_(0.25, 0.70)
    cycles = torch.empty(count, device=device).uniform_(0.5, 1.5)
    phase = torch.empty(count, device=device).uniform_(-torch.pi, torch.pi)
    raw_curvature = bias[:, None] + amplitude[:, None] * torch.sin(
        2.0 * torch.pi * cycles[:, None] * midpoint_s[None, :] / length_m
        + phase[:, None]
    )

    # Sample the desired peak explicitly.  Normalizing every raw curve to a
    # continuous target avoids the boundary pile-up created by clipping or by
    # scaling every out-of-range sample exactly to max_curvature_rad_m.
    target_peak_curvature = torch.empty(count, device=device).uniform_(
        0.25, max_curvature_rad_m
    )
    raw_peak = raw_curvature.abs().amax(dim=1).clamp_min(1.0e-6)
    curvature = raw_curvature * (target_peak_curvature / raw_peak)[:, None]

    # Preserve the sampled shape while enforcing the global-heading contract.
    # Scaling (rather than clipping) avoids flat spots in curvature and keeps
    # the route differentiable along arc length.
    interval_yaw = torch.cumsum(curvature * ds, dim=1)
    heading_limit = torch.deg2rad(
        torch.tensor(max_abs_heading_deg, device=device)
    )
    heading_scale = (
        interval_yaw.abs().amax(dim=1) / heading_limit
    ).clamp_min(1.0)
    curvature = curvature / heading_scale[:, None]
    interval_yaw = torch.cumsum(curvature * ds, dim=1)

    point_yaw = torch.cat(
        (torch.zeros(count, 1, device=device), interval_yaw), dim=1
    )
    increments = ds * torch.stack(
        (torch.cos(point_yaw[:, :-1]), torch.sin(point_yaw[:, :-1])), dim=2
    )
    xy = torch.cat(
        (
            torch.zeros(count, 1, 2, device=device),
            torch.cumsum(increments, dim=1),
        ),
        dim=1,
    )
    yaw = point_yaw
    yaw[:, -1] = yaw[:, -2]
    arc = torch.linspace(0.0, length_m, points, device=device).expand(count, -1)
    return SmoothGeometry(
        xy=xy,
        yaw=yaw,
        arc=arc,
        curvature=curvature,
        target_peak_curvature=target_peak_curvature,
        curvature_bias=bias,
        curvature_amplitude=amplitude,
        curvature_cycles=cycles,
        curvature_phase=phase,
    )


def _cross_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _has_proper_self_intersection(vertices: torch.Tensor, segment_count: torch.Tensor) -> torch.Tensor:
    """Return whether open polylines contain a non-adjacent proper crossing."""

    count = vertices.shape[0]
    invalid = torch.zeros(count, dtype=torch.bool, device=vertices.device)
    max_segments = vertices.shape[1] - 1
    for first, second in combinations(range(max_segments), 2):
        if second <= first + 1:
            continue
        active = segment_count > second
        p0, p1 = vertices[:, first], vertices[:, first + 1]
        q0, q1 = vertices[:, second], vertices[:, second + 1]
        pq, pr = p1 - p0, q1 - q0
        side_q0 = _cross_2d(pq, q0 - p0)
        side_q1 = _cross_2d(pq, q1 - p0)
        side_p0 = _cross_2d(pr, p0 - q0)
        side_p1 = _cross_2d(pr, p1 - q0)
        crossing = (side_q0 * side_q1 < 0.0) & (side_p0 * side_p1 < 0.0)
        invalid |= active & crossing
    return invalid


def _sample_polyline_parameters(
    count: int,
    device: torch.device,
    *,
    length_m: float,
    min_segments: int,
    max_segments: int,
    min_segment_length_m: float,
    min_turn_rad: float,
    max_turn_rad: float,
    max_abs_heading_rad: float,
    minimum_endpoint_distance_m: float,
    max_attempts: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample non-degenerate, non-self-intersecting open polylines."""

    if count == 0:
        empty_i = torch.empty(0, dtype=torch.long, device=device)
        return (
            empty_i,
            torch.empty(0, max_segments, device=device),
            torch.empty(0, max_segments - 1, device=device),
            torch.empty(0, max_segments + 1, 2, device=device),
        )

    # Complexity is sampled once.  Rejected candidates retain this segment
    # count while only their lengths and turns are redrawn, so geometric
    # validity cannot silently bias the bank toward simpler paths.
    segment_count = torch.randint(
        min_segments, max_segments + 1, (count,), device=device
    )
    lengths = torch.zeros(count, max_segments, device=device)
    turns = torch.zeros(count, max_segments - 1, device=device)
    vertices = torch.zeros(count, max_segments + 1, 2, device=device)
    pending = torch.ones(count, dtype=torch.bool, device=device)

    segment_ids = torch.arange(max_segments, device=device)[None, :]
    turn_ids = torch.arange(max_segments - 1, device=device)[None, :]
    for _ in range(max_attempts):
        pending_ids = torch.nonzero(pending, as_tuple=False).squeeze(1)
        sample_count = int(pending_ids.numel())
        if sample_count == 0:
            break

        sampled_count = segment_count[pending_ids]
        active_segments = segment_ids < sampled_count[:, None]
        remaining = length_m - sampled_count.float() * min_segment_length_m
        weights = -torch.log(
            torch.rand(sample_count, max_segments, device=device).clamp_min(1.0e-6)
        )
        weights *= active_segments
        sampled_lengths = active_segments * min_segment_length_m
        sampled_lengths += remaining[:, None] * weights / weights.sum(dim=1, keepdim=True)

        active_turns = turn_ids < (sampled_count - 1)[:, None]
        magnitudes = min_turn_rad + torch.rand(
            sample_count, max_segments - 1, device=device
        ) * (max_turn_rad - min_turn_rad)
        signs = torch.where(
            torch.rand(sample_count, max_segments - 1, device=device) < 0.5,
            -torch.ones_like(magnitudes),
            torch.ones_like(magnitudes),
        )
        sampled_turns = magnitudes * signs * active_turns
        headings = torch.cat(
            (
                torch.zeros(sample_count, 1, device=device),
                torch.cumsum(sampled_turns, dim=1),
            ),
            dim=1,
        )
        increments = sampled_lengths[..., None] * torch.stack(
            (torch.cos(headings), torch.sin(headings)), dim=2
        )
        sampled_vertices = torch.cat(
            (
                torch.zeros(sample_count, 1, 2, device=device),
                torch.cumsum(increments, dim=1),
            ),
            dim=1,
        )
        endpoint_rows = torch.arange(sample_count, device=device)
        endpoint = sampled_vertices[endpoint_rows, sampled_count]
        invalid = _has_proper_self_intersection(sampled_vertices, sampled_count)
        invalid |= (headings.abs() > max_abs_heading_rad).any(dim=1)
        invalid |= torch.linalg.vector_norm(endpoint, dim=1) < minimum_endpoint_distance_m

        accepted_local = torch.nonzero(~invalid, as_tuple=False).squeeze(1)
        accepted_global = pending_ids[accepted_local]
        lengths[accepted_global] = sampled_lengths[accepted_local]
        turns[accepted_global] = sampled_turns[accepted_local]
        vertices[accepted_global] = sampled_vertices[accepted_local]
        pending[accepted_global] = False

    if torch.any(pending):
        raise RuntimeError(
            f"Could not sample {int(pending.sum())} valid waypoint routes after "
            f"{max_attempts} attempts."
        )
    return segment_count, lengths, turns, vertices


def sample_waypoint_geometry(
    count: int,
    device: torch.device | str,
    *,
    rounded: bool,
    points: int = 101,
    length_m: float = 4.0,
    min_segments: int = 2,
    max_segments: int = 5,
    min_segment_length_m: float = 0.55,
    min_turn_deg: float = 15.0,
    max_turn_deg: float = 90.0,
    max_abs_heading_deg: float = 135.0,
    minimum_endpoint_distance_m: float = 1.0,
    corner_radius_min_m: float = 0.12,
    corner_radius_max_m: float = 0.30,
) -> WaypointGeometry:
    """Generate fixed-arc waypoint paths for plotting, tests and DPPO resets."""

    device = torch.device(device)
    if count < 0:
        raise ValueError("count must be non-negative.")
    if points < 16 or length_m <= 0.0:
        raise ValueError("points must be >= 16 and length_m must be positive.")
    if not 2 <= min_segments <= max_segments:
        raise ValueError("Expected 2 <= min_segments <= max_segments.")
    if max_segments * min_segment_length_m >= length_m:
        raise ValueError("Minimum segment length leaves no sampling freedom.")
    if not 0.0 < min_turn_deg <= max_turn_deg <= 90.0:
        raise ValueError("Turn magnitudes must lie in (0, 90] degrees.")
    if max_abs_heading_deg < max_turn_deg or max_abs_heading_deg > 180.0:
        raise ValueError("max_abs_heading_deg must lie in [max_turn_deg, 180].")
    if not 0.0 < corner_radius_min_m <= corner_radius_max_m:
        raise ValueError("Invalid corner-radius range.")

    min_turn_rad = torch.deg2rad(torch.tensor(min_turn_deg)).item()
    max_turn_rad = torch.deg2rad(torch.tensor(max_turn_deg)).item()
    ds = length_m / float(points - 1)
    # Each sampled corner is represented on the fixed arc grid.  Reserve the
    # worst-case accumulated half-bin displacement so the *realized* terminal
    # remains outside the requested start-clearance radius.
    control_endpoint_clearance = minimum_endpoint_distance_m + max_segments * 0.5 * ds
    segment_count, lengths, turns, _ = _sample_polyline_parameters(
        count,
        device,
        length_m=length_m,
        min_segments=min_segments,
        max_segments=max_segments,
        min_segment_length_m=min_segment_length_m,
        min_turn_rad=min_turn_rad,
        max_turn_rad=max_turn_rad,
        max_abs_heading_rad=torch.deg2rad(torch.tensor(max_abs_heading_deg)).item(),
        minimum_endpoint_distance_m=control_endpoint_clearance,
    )

    arc_1d = torch.linspace(0.0, length_m, points, device=device)
    arc = arc_1d.expand(count, -1)
    midpoint_s = 0.5 * (arc_1d[:-1] + arc_1d[1:])
    boundaries = torch.cumsum(lengths, dim=1)[:, :-1]
    active_turns = torch.arange(max_segments - 1, device=device)[None, :] < (
        segment_count - 1
    )[:, None]

    if rounded:
        radii = corner_radius_min_m + torch.rand(
            count, max_segments - 1, device=device
        ) * (corner_radius_max_m - corner_radius_min_m)
        radii *= active_turns
        blend_length = (radii * turns.abs()).clamp_min(2.0 * ds)
        blend_length *= active_turns
        # With the stated radii and >=0.55 m segments, adjacent blends do not
        # overlap.  This preserves identifiable straight sections.
        relative = (
            midpoint_s[None, :, None]
            - (boundaries[:, None, :] - 0.5 * blend_length[:, None, :])
        ) / blend_length[:, None, :].clamp_min(1.0e-6)
        blend = relative.clamp(0.0, 1.0)
        blend = blend.square() * (3.0 - 2.0 * blend)
        interval_yaw = torch.sum(
            turns[:, None, :] * blend * active_turns[:, None, :], dim=2
        )
    else:
        radii = torch.zeros(count, max_segments - 1, device=device)
        interval_yaw = torch.sum(
            turns[:, None, :]
            * (midpoint_s[None, :, None] >= boundaries[:, None, :])
            * active_turns[:, None, :],
            dim=2,
        )

    increments = ds * torch.stack(
        (torch.cos(interval_yaw), torch.sin(interval_yaw)), dim=2
    )
    xy = torch.cat(
        (
            torch.zeros(count, 1, 2, device=device),
            torch.cumsum(increments, dim=1),
        ),
        dim=1,
    )
    yaw = torch.cat((interval_yaw, interval_yaw[:, -1:]), dim=1)
    return WaypointGeometry(
        xy=xy,
        yaw=yaw,
        arc=arc,
        segment_count=segment_count,
        segment_lengths=lengths,
        turn_angles=turns,
        corner_radii=radii,
    )


class SupportedHybridRouteBank(SupportedProceduralRouteBank):
    """V2 bank: 50% smooth, 25% rounded and 25% hard paths.

    Geometry is crossed with the four support-aware posture profiles through
    an exact 16-cell schedule whenever stratified sampling is enabled.
    """

    SMOOTH = 0
    ROUNDED = 1
    HARD = 2
    SMOOTH_V1 = 0
    SMOOTH_COHERENT = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.geometry_class = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.smooth_subclass = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.coherent_target_peak_curvature = torch.zeros(
            self.num_envs, device=self.device
        )
        self.waypoint_segment_count = torch.zeros_like(self.geometry_class)
        self.waypoint_turn_angles = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.waypoint_segment_lengths = torch.zeros(
            self.num_envs, 5, device=self.device
        )
        self.waypoint_corner_radii = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._geometry_sample_counter = 0

    def _sample_geometry_class(
        self, count: int, *, stratified: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stratified:
            # The parent profile schedule cycles every four samples.  A
            # 16-cell joint schedule gives every one of its four profile
            # classes 2 smooth, 1 rounded and 1 hard examples, avoiding a
            # geometry/posture shortcut in the actor.
            joint_index = (
                torch.arange(count, device=self.device) + self._geometry_sample_counter
            ) % 16
            quarter = torch.div(joint_index, 4, rounding_mode="floor")
            self._geometry_sample_counter = (self._geometry_sample_counter + count) % 16
        else:
            quarter = torch.randint(0, 4, (count,), device=self.device)
        geometry_class = torch.where(
            quarter < 2,
            torch.zeros_like(quarter),
            quarter - 1,
        )
        smooth_subclass = torch.where(
            quarter < 2, quarter, torch.full_like(quarter, -1)
        )
        return geometry_class, smooth_subclass

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        stratified: bool = True,
        speed_max: float = 0.8,
    ) -> None:
        super().reset(env_ids, stratified=stratified, speed_max=speed_max)
        count = len(env_ids)
        if count == 0:
            return
        geometry_class, smooth_subclass = self._sample_geometry_class(
            count, stratified=stratified
        )
        self.geometry_class[env_ids] = geometry_class
        self.smooth_subclass[env_ids] = smooth_subclass
        self.coherent_target_peak_curvature[env_ids] = 0.0
        self.waypoint_segment_count[env_ids] = 0
        self.waypoint_turn_angles[env_ids] = 0.0
        self.waypoint_segment_lengths[env_ids] = 0.0
        self.waypoint_corner_radii[env_ids] = 0.0

        coherent_local_ids = torch.nonzero(
            smooth_subclass == self.SMOOTH_COHERENT, as_tuple=False
        ).squeeze(1)
        if coherent_local_ids.numel() > 0:
            coherent_global_ids = env_ids[coherent_local_ids]
            smooth = sample_coherent_smooth_geometry(
                int(coherent_local_ids.numel()),
                self.device,
                points=self.points,
                length_m=self.length_m,
                max_curvature_rad_m=self.max_curvature_rad_m,
            )
            self.xy[coherent_global_ids] = smooth.xy
            self.yaw[coherent_global_ids] = smooth.yaw
            self.arc[coherent_global_ids] = smooth.arc
            self.maximum_abs_curvature[coherent_global_ids] = (
                smooth.curvature.abs().amax(dim=1)
            )
            self.coherent_target_peak_curvature[coherent_global_ids] = (
                smooth.target_peak_curvature
            )
            self.route_kind[coherent_global_ids] = 7

            crouch_fraction = torch.isclose(
                self.height[coherent_global_ids],
                torch.tensor(self.CROUCH_HEIGHT_M, device=self.device),
                atol=1.0e-5,
            ).float().mean(dim=1)
            feasible_speed = 1.0 / (
                crouch_fraction / self.crouch_speed_cap_mps
                + (1.0 - crouch_fraction) / self.curved_speed_cap_mps
            )
            speed_hi = feasible_speed.clamp(max=float(speed_max)).clamp_min(0.2)
            self.speed[coherent_global_ids] = 0.2 + torch.rand(
                int(coherent_local_ids.numel()), device=self.device
            ) * (speed_hi - 0.2)

        for route_class, rounded in ((self.ROUNDED, True), (self.HARD, False)):
            local_mask = geometry_class == route_class
            local_ids = torch.nonzero(local_mask, as_tuple=False).squeeze(1)
            if local_ids.numel() == 0:
                continue
            global_ids = env_ids[local_ids]
            geometry = sample_waypoint_geometry(
                int(local_ids.numel()),
                self.device,
                rounded=rounded,
                points=self.points,
                length_m=self.length_m,
            )
            self.xy[global_ids] = geometry.xy
            self.yaw[global_ids] = geometry.yaw
            self.arc[global_ids] = geometry.arc
            self.waypoint_segment_count[global_ids] = geometry.segment_count
            self.waypoint_turn_angles[global_ids] = geometry.turn_angles
            self.waypoint_segment_lengths[global_ids] = geometry.segment_lengths
            self.waypoint_corner_radii[global_ids] = geometry.corner_radii
            self.maximum_abs_curvature[global_ids] = torch.inf

            # Any route containing a concentrated turn uses the established
            # curved-route cap.  The route-level target stays global; the actor
            # is still free to slow at a corner and recover on a straight.
            crouch_fraction = torch.isclose(
                self.height[global_ids],
                torch.tensor(self.CROUCH_HEIGHT_M, device=self.device),
                atol=1.0e-5,
            ).float().mean(dim=1)
            feasible_speed = 1.0 / (
                crouch_fraction / self.crouch_speed_cap_mps
                + (1.0 - crouch_fraction) / self.curved_speed_cap_mps
            )
            speed_hi = feasible_speed.clamp(max=float(speed_max)).clamp_min(0.2)
            self.speed[global_ids] = 0.2 + torch.rand(
                int(local_ids.numel()), device=self.device
            ) * (speed_hi - 0.2)

            self.route_kind[global_ids] = 5 if rounded else 6
