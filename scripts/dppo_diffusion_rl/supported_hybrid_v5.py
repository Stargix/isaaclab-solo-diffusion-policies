"""Coverage-corrected final route distribution for WCT DPPO.

V5 deliberately restores the successful v3 actor observation and reward
contract.  It changes only reset-time task coverage:

* 50% exact v3 replay;
* 25% feasible single transitions with a short (0.8--1.2 m) crouch section
  and a global mean-speed target in the upper quartile of each route's
  feasible [0.65, min(0.85, ceiling)] m/s interval;
* 25% repeated binary height profiles at [0.35, 0.45] m/s.

The feasibility estimate and coverage labels are private sampler state.  No
curvature, local speed target, gait label, or skill index is exposed to the
actor, critic, or reward.
"""

from __future__ import annotations

import torch

from scripts.residual_diffusion_rl.routes import RouteState

from .route_feasibility import (
    FEASIBILITY_CONTRACT_VERSION_V2,
    PhaseASupportEnvelopeV2,
    maximum_feasible_mean_speed_v2,
)
from .supported_hybrid_v3 import SupportedHybridV3RouteBank


HYBRID_V5_ROUTE_CONTRACT_VERSION = 1


class SupportedHybridV5RouteBank(SupportedHybridV3RouteBank):
    """V3 task with targeted coverage of the two measured failure modes."""

    REPLAY_V3 = 0
    FAST_CROUCH_TO_WALK = 1
    FAST_WALK_TO_CROUCH = 2
    REPEATED_WALK_START = 3
    REPEATED_CROUCH_START = 4

    REPEATED_WALK_PROFILE = 4
    REPEATED_CROUCH_PROFILE = 5

    FAST_SPEED_MIN_MPS = 0.65
    FAST_SPEED_MAX_MPS = 0.85
    FAST_FRONTIER_FRACTION = 0.75
    REPEATED_SPEED_MIN_MPS = 0.35
    REPEATED_SPEED_MAX_MPS = 0.45
    RESTRICTED_SECTION_MIN_M = 0.8
    RESTRICTED_SECTION_MAX_M = 1.2
    REPEATED_FIRST_BOUNDARY_MIN_M = 0.65
    REPEATED_FIRST_BOUNDARY_MAX_M = 0.95
    REPEATED_SECTION_MIN_M = 0.70
    REPEATED_SECTION_MAX_M = 0.90
    MAX_PROFILE_BOUNDARIES = 5

    # Eight equally sized 16-cell blocks preserve the inherited exact cross
    # of four geometry families and four v3 profiles.  Four replay blocks plus
    # one block for each targeted case yield 50/25/25 coverage.
    _COVERAGE_BLOCKS = (
        REPLAY_V3,
        REPLAY_V3,
        REPLAY_V3,
        REPLAY_V3,
        FAST_CROUCH_TO_WALK,
        FAST_WALK_TO_CROUCH,
        REPEATED_WALK_START,
        REPEATED_CROUCH_START,
    )

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
        self.coverage_class = torch.full(
            (self.num_envs,), self.REPLAY_V3, dtype=torch.long, device=self.device
        )
        self.profile_boundaries = torch.full(
            (self.num_envs, self.MAX_PROFILE_BOUNDARIES),
            torch.inf,
            device=self.device,
        )
        self._coverage_sample_counter = 0

    def _sample_coverage_class(self, count: int, *, stratified: bool) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        blocks = torch.tensor(self._COVERAGE_BLOCKS, device=self.device)
        if stratified:
            index = (
                torch.arange(count, device=self.device) + self._coverage_sample_counter
            ) % (16 * len(self._COVERAGE_BLOCKS))
            self._coverage_sample_counter = (
                self._coverage_sample_counter + count
            ) % (16 * len(self._COVERAGE_BLOCKS))
            return blocks[torch.div(index, 16, rounding_mode="floor")]
        draw = torch.randint(0, len(self._COVERAGE_BLOCKS), (count,), device=self.device)
        return blocks[draw]

    def _family_index(self, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.where(
            self.geometry_class[env_ids] == self.SMOOTH,
            self.smooth_subclass[env_ids],
            self.geometry_class[env_ids] + 1,
        )

    def _hypothetical_fast_profiles(
        self, env_ids: torch.Tensor, restricted_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        arc = self.arc[env_ids]
        c2w_boundary = restricted_length
        w2c_boundary = self.length_m - restricted_length
        c2w_height = torch.where(
            arc < c2w_boundary[:, None],
            torch.full_like(arc, self.CROUCH_HEIGHT_M),
            torch.full_like(arc, self.WALK_HEIGHT_M),
        )
        w2c_height = torch.where(
            arc < w2c_boundary[:, None],
            torch.full_like(arc, self.WALK_HEIGHT_M),
            torch.full_like(arc, self.CROUCH_HEIGHT_M),
        )
        c2w_ceiling = maximum_feasible_mean_speed_v2(
            self.yaw[env_ids], c2w_height, arc, envelope=self.support_envelope_v2
        )
        w2c_ceiling = maximum_feasible_mean_speed_v2(
            self.yaw[env_ids], w2c_height, arc, envelope=self.support_envelope_v2
        )
        return c2w_height, w2c_height, c2w_ceiling, w2c_ceiling

    def _repair_fast_coverage(
        self,
        env_ids: torch.Tensor,
        coverage: torch.Tensor,
        parent_profile: torch.Tensor,
        c2w_ceiling: torch.Tensor,
        w2c_ceiling: torch.Tensor,
        *,
        stratified: bool,
    ) -> torch.Tensor:
        """Assign fast labels to the best-supported matched replay routes.

        Matching geometry family and inherited v3 profile preserves both
        marginal distributions in full stratified batches. The reassignment affects
        only which already sampled route receives the fast task; it never
        modifies observations. A small asynchronous batch with no matching
        candidate safely falls back to replay rather than admitting an
        impossible deadline.
        """

        repaired = coverage.clone()
        family = self._family_index(env_ids)
        directions = (
            (self.FAST_CROUCH_TO_WALK, c2w_ceiling),
            (self.FAST_WALK_TO_CROUCH, w2c_ceiling),
        )
        if not stratified:
            for fast_class, ceiling in directions:
                ineligible = (repaired == fast_class) & (
                    ceiling < self.FAST_SPEED_MIN_MPS
                )
                repaired[ineligible] = self.REPLAY_V3
            return repaired

        for route_family in range(4):
            for profile in range(4):
                # Alternate assignment order across the 16 fixed cells so
                # neither transition direction systematically receives the
                # highest-ceiling route when both are feasible.
                ordered_directions = (
                    directions
                    if (route_family + profile) % 2 == 0
                    else directions[::-1]
                )
                for fast_class, ceiling in ordered_directions:
                    key = (family == route_family) & (parent_profile == profile)
                    current = torch.nonzero(
                        (repaired == fast_class) & key, as_tuple=False
                    ).squeeze(1)
                    if current.numel() == 0:
                        continue
                    pool = torch.nonzero(
                        ((repaired == self.REPLAY_V3) | (repaired == fast_class))
                        & key
                        & (ceiling >= self.FAST_SPEED_MIN_MPS),
                        as_tuple=False,
                    ).squeeze(1)
                    # Assign fast objectives to the best-supported routes in
                    # each fixed geometry/profile cell. This is the intended
                    # feasibility filter: sharp/low-support instances remain
                    # in v3 replay, while family-level counts stay balanced.
                    keep = min(pool.numel(), current.numel())
                    selected = pool[torch.topk(ceiling[pool], keep).indices]
                    repaired[current] = self.REPLAY_V3
                    repaired[selected] = fast_class
        return repaired

    def _sample_repeated_profile(
        self, arc: torch.Tensor, *, starts_crouched: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = arc.shape[0]
        first = torch.empty(count, device=self.device).uniform_(
            self.REPEATED_FIRST_BOUNDARY_MIN_M,
            self.REPEATED_FIRST_BOUNDARY_MAX_M,
        )
        spacing = torch.empty(count, device=self.device).uniform_(
            self.REPEATED_SECTION_MIN_M, self.REPEATED_SECTION_MAX_M
        )
        rank = torch.arange(self.MAX_PROFILE_BOUNDARIES, device=self.device)
        boundaries = first[:, None] + spacing[:, None] * rank[None, :]
        boundaries = torch.where(
            boundaries <= self.length_m - 2.0 * self.transition_margin_m,
            boundaries,
            torch.full_like(boundaries, torch.inf),
        )
        crossings = (arc[:, :, None] >= boundaries[:, None, :]).sum(dim=2)
        crouched = starts_crouched[:, None] ^ (crossings.remainder(2) == 1)
        height = torch.where(
            crouched,
            torch.full_like(crossings, self.CROUCH_HEIGHT_M, dtype=self.arc.dtype),
            torch.full_like(crossings, self.WALK_HEIGHT_M, dtype=self.arc.dtype),
        )
        return height, boundaries

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        stratified: bool = True,
        speed_max: float = 1.0,
    ) -> None:
        if speed_max < self.FAST_SPEED_MAX_MPS or speed_max > self.fast_speed_cap_mps:
            raise ValueError(
                f"supported_hybrid_v5 requires speed_max in "
                f"[{self.FAST_SPEED_MAX_MPS}, {self.fast_speed_cap_mps}] m/s."
            )
        # This call is the exact successful v3 task. V5 overwrites only the
        # targeted half of samples selected below.
        super().reset(env_ids, stratified=stratified, speed_max=speed_max)
        count = len(env_ids)
        if count == 0:
            return

        parent_profile = self.profile_class[env_ids].clone()
        coverage = self._sample_coverage_class(count, stratified=stratified)
        restricted_length = torch.empty(count, device=self.device).uniform_(
            self.RESTRICTED_SECTION_MIN_M, self.RESTRICTED_SECTION_MAX_M
        )
        c2w_height, w2c_height, c2w_ceiling, w2c_ceiling = (
            self._hypothetical_fast_profiles(env_ids, restricted_length)
        )
        coverage = self._repair_fast_coverage(
            env_ids,
            coverage,
            parent_profile,
            c2w_ceiling,
            w2c_ceiling,
            stratified=stratified,
        )
        self.coverage_class[env_ids] = coverage

        # Represent every profile with the same boundary tensor so plateau
        # scoring excludes all physical transitions, including repeated ones.
        boundaries = torch.full(
            (count, self.MAX_PROFILE_BOUNDARIES), torch.inf, device=self.device
        )
        inherited_boundary = self.transition_boundary[env_ids]
        boundaries[:, 0] = inherited_boundary

        for fast_class, profile_class, height, ceiling, boundary in (
            (
                self.FAST_CROUCH_TO_WALK,
                3,
                c2w_height,
                c2w_ceiling,
                restricted_length,
            ),
            (
                self.FAST_WALK_TO_CROUCH,
                2,
                w2c_height,
                w2c_ceiling,
                self.length_m - restricted_length,
            ),
        ):
            mask = coverage == fast_class
            if not torch.any(mask):
                continue
            selected_ids = env_ids[mask]
            self.height[selected_ids] = height[mask]
            self.profile_class[selected_ids] = profile_class
            self.transition_boundary[selected_ids] = boundary[mask]
            self.transition_context[selected_ids] = self.TRANSITION_UNIFORM
            boundaries[mask, 0] = boundary[mask]
            feasible = ceiling[mask].clamp(max=self.FAST_SPEED_MAX_MPS)
            if torch.any(feasible < self.FAST_SPEED_MIN_MPS):
                raise RuntimeError("An infeasible route reached the v5 fast sampler.")
            self.maximum_feasible_mean_speed[selected_ids] = ceiling[mask]
            frontier_low = self.FAST_SPEED_MIN_MPS + self.FAST_FRONTIER_FRACTION * (
                feasible - self.FAST_SPEED_MIN_MPS
            )
            self.speed[selected_ids] = frontier_low + torch.rand(
                int(mask.sum()), device=self.device
            ) * (feasible - frontier_low)

        repeated_mask = (coverage == self.REPEATED_WALK_START) | (
            coverage == self.REPEATED_CROUCH_START
        )
        if torch.any(repeated_mask):
            repeated_ids = env_ids[repeated_mask]
            starts_crouched = coverage[repeated_mask] == self.REPEATED_CROUCH_START
            repeated_height, repeated_boundaries = self._sample_repeated_profile(
                self.arc[repeated_ids], starts_crouched=starts_crouched
            )
            self.height[repeated_ids] = repeated_height
            boundaries[repeated_mask] = repeated_boundaries
            self.transition_boundary[repeated_ids] = repeated_boundaries[:, 0]
            self.transition_context[repeated_ids] = self.NO_TRANSITION
            self.profile_class[repeated_ids] = torch.where(
                starts_crouched,
                torch.full_like(starts_crouched, self.REPEATED_CROUCH_PROFILE, dtype=torch.long),
                torch.full_like(starts_crouched, self.REPEATED_WALK_PROFILE, dtype=torch.long),
            )
            ceiling = maximum_feasible_mean_speed_v2(
                self.yaw[repeated_ids],
                self.height[repeated_ids],
                self.arc[repeated_ids],
                envelope=self.support_envelope_v2,
            )
            self.maximum_feasible_mean_speed[repeated_ids] = ceiling
            speed_hi = ceiling.clamp(max=self.REPEATED_SPEED_MAX_MPS).clamp_min(
                self.REPEATED_SPEED_MIN_MPS
            )
            self.speed[repeated_ids] = self.REPEATED_SPEED_MIN_MPS + torch.rand(
                len(repeated_ids), device=self.device
            ) * (speed_hi - self.REPEATED_SPEED_MIN_MPS)

        self.profile_boundaries[env_ids] = boundaries
        self.height_tracking_weight[env_ids] = 1.0

    def update(self, position_local: torch.Tensor) -> RouteState:
        state = super().update(position_local)
        distance = (
            state.progress[:, None] - self.profile_boundaries
        ).abs().amin(dim=1)
        self.height_tracking_weight.copy_(
            (distance >= self.transition_margin_m).float()
        )
        return state


__all__ = [
    "FEASIBILITY_CONTRACT_VERSION_V2",
    "HYBRID_V5_ROUTE_CONTRACT_VERSION",
    "SupportedHybridV5RouteBank",
]
