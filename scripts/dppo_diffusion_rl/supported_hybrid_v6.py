"""Retention-aware consolidation distribution for the final WCT policy.

V6 keeps the complete v5 task definition but repairs its measured speed-
coverage asymmetry.  Within each fast transition direction, one quarter of
the routes are explicit 0.65--0.70 m/s anchors and the other three quarters
retain v5's feasible upper-frontier objective.  This is private reset-time
sampling state: no speed band, gait label, or skill index reaches the policy,
critic, observation, or reward.

The optimizer may use ``coverage_class == REPLAY_V3`` to apply a reference KL
only on the known v3 task.  That label likewise remains private to training.
"""

from __future__ import annotations

import torch

from .supported_hybrid_v5 import (
    FEASIBILITY_CONTRACT_VERSION_V2,
    SupportedHybridV5RouteBank,
)


HYBRID_V6_ROUTE_CONTRACT_VERSION = 1


class SupportedHybridV6RouteBank(SupportedHybridV5RouteBank):
    """V5 coverage with an explicit low-fast calibration anchor."""

    FAST_BAND_NONE = 0
    FAST_BAND_LOW_ANCHOR = 1
    FAST_BAND_FRONTIER = 2

    LOW_FAST_FRACTION = 0.25
    LOW_FAST_MAX_MPS = 0.70

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fast_speed_band = torch.full(
            (self.num_envs,),
            self.FAST_BAND_NONE,
            dtype=torch.long,
            device=self.device,
        )

    def _assign_low_speed_anchors(
        self,
        env_ids: torch.Tensor,
        coverage: torch.Tensor,
        *,
        stratified: bool,
    ) -> None:
        self.fast_speed_band[env_ids] = self.FAST_BAND_NONE
        for fast_class in (
            self.FAST_CROUCH_TO_WALK,
            self.FAST_WALK_TO_CROUCH,
        ):
            local = torch.nonzero(coverage == fast_class, as_tuple=False).squeeze(1)
            if local.numel() == 0:
                continue
            selected_ids = env_ids[local]
            self.fast_speed_band[selected_ids] = self.FAST_BAND_FRONTIER

            anchor_count = (
                int(round(local.numel() * self.LOW_FAST_FRACTION))
                if stratified
                else int(
                    (
                        torch.rand(local.numel(), device=self.device)
                        < self.LOW_FAST_FRACTION
                    ).sum()
                )
            )
            if anchor_count == 0:
                continue
            # Randomizing within the already balanced direction avoids tying
            # the low-speed band to geometry/profile order while preserving an
            # exact 25% count in full stratified resets.
            anchor_local = local[
                torch.randperm(local.numel(), device=self.device)[:anchor_count]
            ]
            anchor_ids = env_ids[anchor_local]
            feasible = self.maximum_feasible_mean_speed[anchor_ids].clamp(
                max=self.LOW_FAST_MAX_MPS
            )
            if torch.any(feasible < self.FAST_SPEED_MIN_MPS):
                raise RuntimeError("An infeasible route reached the v6 low-speed anchor.")
            self.speed[anchor_ids] = self.FAST_SPEED_MIN_MPS + torch.rand(
                anchor_count, device=self.device
            ) * (feasible - self.FAST_SPEED_MIN_MPS)
            self.fast_speed_band[anchor_ids] = self.FAST_BAND_LOW_ANCHOR

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        stratified: bool = True,
        speed_max: float = 1.0,
    ) -> None:
        super().reset(env_ids, stratified=stratified, speed_max=speed_max)
        if len(env_ids) == 0:
            return
        self._assign_low_speed_anchors(
            env_ids,
            self.coverage_class[env_ids],
            stratified=stratified,
        )


__all__ = [
    "FEASIBILITY_CONTRACT_VERSION_V2",
    "HYBRID_V6_ROUTE_CONTRACT_VERSION",
    "SupportedHybridV6RouteBank",
]
