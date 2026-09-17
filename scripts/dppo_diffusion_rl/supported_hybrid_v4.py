"""Support-aware hybrid-v4 route distribution.

V4 preserves the audited v2 geometry and v3 posture/transition distribution.
It changes only reset-time task sampling: the Phase-A support estimate includes
the fast upright expert and finite posture-transition reserve, and the sampled
global mean-speed targets deliberately cover the upper feasible range.  These
private quantities are never exposed to the actor, critic or reward.
"""

from __future__ import annotations

import torch

from .route_feasibility import (
    FEASIBILITY_CONTRACT_VERSION_V2,
    PhaseASupportEnvelopeV2,
    maximum_feasible_mean_speed_v2,
)
from .supported_hybrid_v3 import SupportedHybridV3RouteBank


HYBRID_V4_ROUTE_CONTRACT_VERSION = 1


class SupportedHybridV4RouteBank(SupportedHybridV3RouteBank):
    """V3 routes with evidence-bounded and fast-balanced global deadlines."""

    PACE_FULL_RANGE = 0
    PACE_UPPER_FEASIBLE = 1

    def __init__(
        self,
        *args,
        support_envelope_v2: PhaseASupportEnvelopeV2 | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.support_envelope_v2 = support_envelope_v2 or PhaseASupportEnvelopeV2(
            crouch_height_m=self.CROUCH_HEIGHT_M,
            walk_height_m=self.WALK_HEIGHT_M,
            crouch_speed_cap_mps=self.crouch_speed_cap_mps,
            strong_turn_speed_cap_mps=self.curved_speed_cap_mps,
        )
        self.support_envelope_v2.validate()
        self.pace_tier = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._pace_tier_counter = 0

    def _sample_pace_tier(self, count: int, *, stratified: bool) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if stratified:
            # The inherited joint geometry/profile schedule has 16 cells.
            # Repeating it once per pace tier creates an exact 32-cell cross,
            # so fast targets cannot correlate with a geometry or posture.
            joint_index = (
                torch.arange(count, device=self.device) + self._pace_tier_counter
            ) % 32
            self._pace_tier_counter = (self._pace_tier_counter + count) % 32
            return (joint_index >= 16).long()
        return torch.randint(0, 2, (count,), device=self.device)

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        stratified: bool = True,
        speed_max: float = 1.0,
    ) -> None:
        if speed_max < 0.2 or speed_max > self.support_envelope_v2.straight_speed_cap_mps:
            raise ValueError(
                "speed_max must be in [0.2, "
                f"{self.support_envelope_v2.straight_speed_cap_mps}] m/s."
            )
        # V3 remains the sole owner of geometry, profiles and transition
        # contexts. V4 overwrites only its private feasibility estimate and
        # the single route-average speed sampled at reset.
        super().reset(
            env_ids,
            stratified=stratified,
            speed_max=min(speed_max, self.fast_speed_cap_mps),
        )
        count = len(env_ids)
        if count == 0:
            return

        feasible = maximum_feasible_mean_speed_v2(
            self.yaw[env_ids],
            self.height[env_ids],
            self.arc[env_ids],
            envelope=self.support_envelope_v2,
        )
        self.maximum_feasible_mean_speed[env_ids] = feasible
        speed_hi = feasible.clamp(max=float(speed_max)).clamp_min(0.2)
        pace_tier = self._sample_pace_tier(count, stratified=stratified)
        self.pace_tier[env_ids] = pace_tier

        full_range_low = torch.full_like(speed_hi, 0.2)
        upper_quartile_low = 0.2 + 0.75 * (speed_hi - 0.2)
        speed_low = torch.where(
            pace_tier == self.PACE_UPPER_FEASIBLE,
            upper_quartile_low,
            full_range_low,
        )
        self.speed[env_ids] = speed_low + torch.rand(
            count, device=self.device
        ) * (speed_hi - speed_low)


__all__ = [
    "FEASIBILITY_CONTRACT_VERSION_V2",
    "HYBRID_V4_ROUTE_CONTRACT_VERSION",
    "SupportedHybridV4RouteBank",
]
