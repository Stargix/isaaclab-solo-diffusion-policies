"""Private Phase-A support model used only to sample feasible route tasks.

The quantities computed here are deliberately not observations, critic
features or reward targets.  They provide a conservative route-level upper
bound for the *mean* speed requested at reset, preventing impossible
path/posture/deadline combinations without prescribing a local speed profile
to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


FEASIBILITY_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class PhaseASupportEnvelope:
    """Audited WCT Phase-A locomotion envelope.

    The endpoint caps are the same ones used by the supported-v1/v2 route
    generators.  Curvature interpolation avoids introducing discrete semantic
    labels such as ``walk`` or ``trot`` into the task.
    """

    crouch_height_m: float = 0.1705
    walk_height_m: float = 0.2932
    crouch_speed_cap_mps: float = 0.4
    curved_speed_cap_mps: float = 0.6
    straight_speed_cap_mps: float = 1.2
    free_curvature_rad_m: float = 0.15
    full_curvature_rad_m: float = 0.8
    turn_support_window_m: float = 0.4

    def validate(self) -> None:
        if not self.crouch_height_m < self.walk_height_m:
            raise ValueError("Expected crouch height below walk height.")
        if not (
            0.0
            < self.crouch_speed_cap_mps
            <= self.curved_speed_cap_mps
            <= self.straight_speed_cap_mps
        ):
            raise ValueError("Expected crouch <= curved <= straight speed caps.")
        if not 0.0 <= self.free_curvature_rad_m < self.full_curvature_rad_m:
            raise ValueError("Invalid curvature support interval.")
        if self.turn_support_window_m <= 0.0:
            raise ValueError("turn_support_window_m must be positive.")


def _validate_route_tensors(
    yaw: torch.Tensor, height: torch.Tensor, arc: torch.Tensor
) -> None:
    if yaw.ndim != 2 or height.shape != yaw.shape or arc.shape != yaw.shape:
        raise ValueError("yaw, height and arc must have identical [batch, points] shapes.")
    if yaw.shape[1] < 3:
        raise ValueError("Routes require at least three points.")
    if torch.any((arc[:, 1:] - arc[:, :-1]) <= 0.0):
        raise ValueError("Route arc coordinates must be strictly increasing.")


def local_turning_demand(
    yaw: torch.Tensor,
    arc: torch.Tensor,
    *,
    support_window_m: float,
) -> torch.Tensor:
    """Return non-negative local turning demand in rad/m.

    A max filter spreads a discrete waypoint turn over a finite locomotion
    support window.  This affects only reset-time feasibility: it is never
    exposed to the actor.
    """

    if yaw.ndim != 2 or arc.shape != yaw.shape:
        raise ValueError("yaw and arc must have identical [batch, points] shapes.")
    if support_window_m <= 0.0:
        raise ValueError("support_window_m must be positive.")
    ds = arc[:, 1:] - arc[:, :-1]
    if torch.any(ds <= 0.0):
        raise ValueError("Route arc coordinates must be strictly increasing.")
    delta_yaw = torch.atan2(
        torch.sin(yaw[:, 1:] - yaw[:, :-1]),
        torch.cos(yaw[:, 1:] - yaw[:, :-1]),
    )
    curvature = (delta_yaw / ds).abs()
    point_curvature = torch.cat((curvature, curvature[:, -1:]), dim=1)

    nominal_ds = float(ds.mean().detach().cpu())
    kernel = max(1, int(round(support_window_m / nominal_ds)))
    if kernel % 2 == 0:
        kernel += 1
    return F.max_pool1d(
        point_curvature[:, None, :], kernel_size=kernel, stride=1, padding=kernel // 2
    ).squeeze(1)


def local_support_speed(
    yaw: torch.Tensor,
    height: torch.Tensor,
    arc: torch.Tensor,
    *,
    envelope: PhaseASupportEnvelope = PhaseASupportEnvelope(),
) -> torch.Tensor:
    """Compute the private local Phase-A support envelope in m/s."""

    envelope.validate()
    _validate_route_tensors(yaw, height, arc)
    turning = local_turning_demand(
        yaw, arc, support_window_m=envelope.turn_support_window_m
    )
    curvature_fraction = (
        (turning - envelope.free_curvature_rad_m)
        / (envelope.full_curvature_rad_m - envelope.free_curvature_rad_m)
    ).clamp(0.0, 1.0)
    upright_cap = envelope.straight_speed_cap_mps + curvature_fraction * (
        envelope.curved_speed_cap_mps - envelope.straight_speed_cap_mps
    )
    height_fraction = (
        (height - envelope.crouch_height_m)
        / (envelope.walk_height_m - envelope.crouch_height_m)
    ).clamp(0.0, 1.0)
    return envelope.crouch_speed_cap_mps + height_fraction * (
        upright_cap - envelope.crouch_speed_cap_mps
    )


def maximum_feasible_mean_speed(
    yaw: torch.Tensor,
    height: torch.Tensor,
    arc: torch.Tensor,
    *,
    envelope: PhaseASupportEnvelope = PhaseASupportEnvelope(),
) -> torch.Tensor:
    """Return route-level feasible mean speed without exposing a local target."""

    support = local_support_speed(yaw, height, arc, envelope=envelope)
    ds = arc[:, 1:] - arc[:, :-1]
    segment_cap = 0.5 * (support[:, :-1] + support[:, 1:])
    minimum_time = torch.sum(ds / segment_cap.clamp_min(1.0e-6), dim=1)
    route_length = arc[:, -1] - arc[:, 0]
    return route_length / minimum_time.clamp_min(1.0e-6)
