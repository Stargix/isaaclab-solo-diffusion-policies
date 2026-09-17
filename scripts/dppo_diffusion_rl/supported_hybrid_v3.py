"""Support-aware hybrid-v3 route distribution for the final DPPO train.

V3 reuses the audited v2 geometry generators, broadens transition placement,
and samples only a feasible *route-average* speed.  The private support model
is never provided to the actor, critic or reward.
"""

from __future__ import annotations

import torch

from .hybrid_routes import SupportedHybridRouteBank
from .route_feasibility import (
    FEASIBILITY_CONTRACT_VERSION,
    PhaseASupportEnvelope,
    local_turning_demand,
    maximum_feasible_mean_speed,
)


HYBRID_V3_ROUTE_CONTRACT_VERSION = 1


class SupportedHybridV3RouteBank(SupportedHybridRouteBank):
    """V2 geometry plus transition-context coverage and feasible deadlines."""

    TRANSITION_UNIFORM = 0
    TRANSITION_LOW_CURVATURE = 1
    TRANSITION_HIGH_CURVATURE = 2
    NO_TRANSITION = -1

    def __init__(self, *args, support_envelope: PhaseASupportEnvelope | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_envelope = support_envelope or PhaseASupportEnvelope(
            crouch_height_m=self.CROUCH_HEIGHT_M,
            walk_height_m=self.WALK_HEIGHT_M,
            crouch_speed_cap_mps=self.crouch_speed_cap_mps,
            curved_speed_cap_mps=self.curved_speed_cap_mps,
            straight_speed_cap_mps=self.fast_speed_cap_mps,
        )
        self.support_envelope.validate()
        self.transition_context = torch.full(
            (self.num_envs,), self.NO_TRANSITION, dtype=torch.long, device=self.device
        )
        self.maximum_feasible_mean_speed = torch.full(
            (self.num_envs,), self.crouch_speed_cap_mps, device=self.device
        )
        self._transition_context_counter = 0

    def _sample_transition_context(self, count: int, *, stratified: bool) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if stratified:
            # 50% independent/uniform, 25% low and 25% high curvature.
            schedule = torch.tensor(
                [
                    self.TRANSITION_UNIFORM,
                    self.TRANSITION_UNIFORM,
                    self.TRANSITION_LOW_CURVATURE,
                    self.TRANSITION_HIGH_CURVATURE,
                ],
                dtype=torch.long,
                device=self.device,
            )
            index = (
                torch.arange(count, device=self.device) + self._transition_context_counter
            ) % len(schedule)
            self._transition_context_counter = (
                self._transition_context_counter + count
            ) % len(schedule)
            return schedule[index]
        draw = torch.randint(0, 4, (count,), device=self.device)
        return torch.where(draw < 2, torch.zeros_like(draw), draw - 1)

    def _context_boundaries(
        self, env_ids: torch.Tensor, contexts: torch.Tensor
    ) -> torch.Tensor:
        count = len(env_ids)
        boundary = torch.empty(count, device=self.device).uniform_(
            self.transition_boundary_min_m, self.transition_boundary_max_m
        )
        non_uniform = contexts != self.TRANSITION_UNIFORM
        if not torch.any(non_uniform):
            return boundary

        selected_ids = env_ids[non_uniform]
        selected_context = contexts[non_uniform]
        turning = local_turning_demand(
            self.yaw[selected_ids],
            self.arc[selected_ids],
            support_window_m=self.support_envelope.turn_support_window_m,
        )
        candidate_mask = (
            (self.arc[selected_ids] >= self.transition_boundary_min_m)
            & (self.arc[selected_ids] <= self.transition_boundary_max_m)
        )
        candidate_columns = torch.nonzero(candidate_mask[0], as_tuple=False).squeeze(1)
        if candidate_columns.numel() == 0:
            raise RuntimeError("No route samples lie inside the transition range.")
        candidate_turning = turning[:, candidate_columns]
        subset_size = max(1, int(candidate_columns.numel()) // 4)
        low_columns = torch.topk(
            candidate_turning, subset_size, dim=1, largest=False
        ).indices
        high_columns = torch.topk(
            candidate_turning, subset_size, dim=1, largest=True
        ).indices
        choices = torch.where(
            (selected_context == self.TRANSITION_LOW_CURVATURE)[:, None],
            low_columns,
            high_columns,
        )
        random_rank = torch.randint(
            0, subset_size, (len(selected_ids),), device=self.device
        )
        row = torch.arange(len(selected_ids), device=self.device)
        selected_column = candidate_columns[choices[row, random_rank]]
        boundary[non_uniform] = self.arc[selected_ids, selected_column]
        return boundary

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        stratified: bool = True,
        speed_max: float = 1.0,
    ) -> None:
        if speed_max < 0.2 or speed_max > self.fast_speed_cap_mps:
            raise ValueError(
                f"speed_max must be in [0.2, {self.fast_speed_cap_mps}] m/s."
            )
        # V2 owns all geometry/profile generation. V3 only replaces transition
        # placement and route-level deadline sampling after that audited draw.
        super().reset(env_ids, stratified=stratified, speed_max=speed_max)
        count = len(env_ids)
        if count == 0:
            return

        profile_class = self.profile_class[env_ids]
        is_transition = profile_class >= 2
        transition_ids = env_ids[is_transition]
        contexts = self._sample_transition_context(
            int(is_transition.sum()), stratified=stratified
        )
        self.transition_context[env_ids] = self.NO_TRANSITION
        self.transition_context[transition_ids] = contexts

        sampled_boundary = self._context_boundaries(transition_ids, contexts)
        boundary = torch.full((count,), torch.inf, device=self.device)
        boundary[is_transition] = sampled_boundary
        starts_crouched = (profile_class == 1) | (profile_class == 3)
        start_height = torch.where(
            starts_crouched,
            torch.full_like(boundary, self.CROUCH_HEIGHT_M),
            torch.full_like(boundary, self.WALK_HEIGHT_M),
        )
        end_height = torch.where(
            is_transition,
            torch.where(
                starts_crouched,
                torch.full_like(boundary, self.WALK_HEIGHT_M),
                torch.full_like(boundary, self.CROUCH_HEIGHT_M),
            ),
            start_height,
        )
        self.height[env_ids] = torch.where(
            self.arc[env_ids] < boundary[:, None],
            start_height[:, None],
            end_height[:, None],
        )
        self.transition_boundary[env_ids] = boundary

        feasible = maximum_feasible_mean_speed(
            self.yaw[env_ids],
            self.height[env_ids],
            self.arc[env_ids],
            envelope=self.support_envelope,
        )
        self.maximum_feasible_mean_speed[env_ids] = feasible
        speed_hi = feasible.clamp(max=float(speed_max)).clamp_min(0.2)
        self.speed[env_ids] = 0.2 + torch.rand(count, device=self.device) * (
            speed_hi - 0.2
        )
        self.height_tracking_weight[env_ids] = 1.0


__all__ = [
    "FEASIBILITY_CONTRACT_VERSION",
    "HYBRID_V3_ROUTE_CONTRACT_VERSION",
    "SupportedHybridV3RouteBank",
]
