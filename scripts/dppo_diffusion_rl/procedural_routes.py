"""Support-aware procedural routes for the final path/pose DPPO experiment.

The offline profile16 dataset contains constant walk, crouch and fast-expert
episodes.  This bank deliberately composes only the two clearly separated
posture endpoints (walk/crouch) and leaves intermediate measured heights to
emerge during transitions.  Route geometry is sampled continuously per reset;
there is no finite catalogue of polygons for the actor to memorize.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.residual_diffusion_rl.routes import RouteBank, RouteState


class SupportedProceduralRouteBank(RouteBank):
    """Procedural 4 m routes constrained by the Phase-A locomotion envelope.

    Profile classes are stratified as constant walk, constant crouch,
    walk-to-crouch and crouch-to-walk.  Transition boundaries are sampled so
    both plateaus are 1.6--2.4 m on a 4 m route.  A narrow region around the
    discontinuous task boundary is excluded from the *plateau* height score;
    this gives the learned policy room to execute a physical transition but
    neither blends actions nor prescribes a height trajectory.
    """

    WALK_HEIGHT_M = 0.2932
    CROUCH_HEIGHT_M = 0.1705
    PROFILE_CLASS_COUNT = 4

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        points: int = 101,
        length_m: float = 4.0,
        curvature_knots: int = 6,
        max_curvature_rad_m: float = 0.8,
        transition_boundary_min_m: float = 1.6,
        transition_boundary_max_m: float = 2.4,
        transition_margin_m: float = 0.25,
        crouch_speed_cap_mps: float = 0.4,
        curved_speed_cap_mps: float = 0.6,
        fast_speed_cap_mps: float = 1.2,
    ) -> None:
        super().__init__(
            num_envs,
            device,
            points=points,
            length_m=length_m,
            height_segment_m=1.6,
            walk_height=self.WALK_HEIGHT_M,
            crouch_height=self.CROUCH_HEIGHT_M,
        )
        if curvature_knots < 2:
            raise ValueError("curvature_knots must be at least 2.")
        if max_curvature_rad_m <= 0.0:
            raise ValueError("max_curvature_rad_m must be positive.")
        if not 0.0 < transition_boundary_min_m <= transition_boundary_max_m < length_m:
            raise ValueError("Transition boundary range must lie inside the route.")
        if transition_margin_m < 0.0:
            raise ValueError("transition_margin_m must be non-negative.")
        if transition_margin_m >= min(
            transition_boundary_min_m, length_m - transition_boundary_max_m
        ):
            raise ValueError("transition_margin_m would remove an entire plateau.")
        if min(crouch_speed_cap_mps, curved_speed_cap_mps, fast_speed_cap_mps) <= 0.0:
            raise ValueError("Skill-support speed caps must be positive.")
        if not crouch_speed_cap_mps <= curved_speed_cap_mps <= fast_speed_cap_mps:
            raise ValueError("Expected crouch <= curved <= fast speed caps.")

        self.curvature_knots = int(curvature_knots)
        self.max_curvature_rad_m = float(max_curvature_rad_m)
        self.transition_boundary_min_m = float(transition_boundary_min_m)
        self.transition_boundary_max_m = float(transition_boundary_max_m)
        self.transition_margin_m = float(transition_margin_m)
        self.crouch_speed_cap_mps = float(crouch_speed_cap_mps)
        self.curved_speed_cap_mps = float(curved_speed_cap_mps)
        self.fast_speed_cap_mps = float(fast_speed_cap_mps)

        self.profile_class = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.transition_boundary = torch.full((num_envs,), torch.inf, device=self.device)
        self.height_tracking_weight = torch.ones(num_envs, device=self.device)
        self.maximum_abs_curvature = torch.zeros(num_envs, device=self.device)
        self._profile_sample_counter = 0

    def _sample_profile_class(self, count: int, *, stratified: bool) -> torch.Tensor:
        if stratified:
            values = (
                torch.arange(count, device=self.device) + self._profile_sample_counter
            ) % self.PROFILE_CLASS_COUNT
            self._profile_sample_counter = (
                self._profile_sample_counter + count
            ) % self.PROFILE_CLASS_COUNT
            return values
        return torch.randint(0, self.PROFILE_CLASS_COUNT, (count,), device=self.device)

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        stratified: bool = True,
        speed_max: float = 0.8,
    ) -> None:
        count = len(env_ids)
        if count == 0:
            return
        if speed_max < 0.2 or speed_max > self.fast_speed_cap_mps:
            raise ValueError(
                f"speed_max must be in [0.2, {self.fast_speed_cap_mps}] m/s."
            )

        # Smooth random curvature is linearly interpolated between independent
        # knots.  Sampling a per-route amplitude continuously includes nearly
        # straight and strongly curved routes without discrete shape labels.
        knot_amplitude = torch.rand(count, 1, device=self.device) * self.max_curvature_rad_m
        knots = torch.empty(
            count, 1, self.curvature_knots, device=self.device
        ).uniform_(-1.0, 1.0)
        knots *= knot_amplitude[:, :, None]
        curvature = F.interpolate(
            knots, size=self.points - 1, mode="linear", align_corners=True
        ).squeeze(1)
        ds = self.length_m / float(self.points - 1)
        yaw = torch.cat(
            (
                torch.zeros(count, 1, device=self.device),
                torch.cumsum(curvature * ds, dim=1),
            ),
            dim=1,
        )
        # The final path sample has no outgoing segment. Reuse the last real
        # segment tangent instead of integrating one unexecuted curvature bin.
        yaw[:, -1] = yaw[:, -2]
        increments = ds * torch.stack((torch.cos(yaw[:, :-1]), torch.sin(yaw[:, :-1])), dim=2)
        xy = torch.cat(
            (
                torch.zeros(count, 1, 2, device=self.device),
                torch.cumsum(increments, dim=1),
            ),
            dim=1,
        )
        arc = torch.linspace(0.0, self.length_m, self.points, device=self.device).expand(count, -1)

        profile_class = self._sample_profile_class(count, stratified=stratified)
        boundary = torch.empty(count, device=self.device).uniform_(
            self.transition_boundary_min_m, self.transition_boundary_max_m
        )
        is_transition = profile_class >= 2
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
        height = torch.where(
            arc < boundary[:, None], start_height[:, None], end_height[:, None]
        )
        boundary = torch.where(is_transition, boundary, torch.full_like(boundary, torch.inf))

        # The maximum attainable route-average speed is the harmonic mean of
        # the demonstrated crouch and non-crouch envelopes.  Curved portions
        # use the project's already successful 0.6 m/s path envelope; nearly
        # straight routes retain headroom for the fast expert.
        max_abs_curvature = curvature.abs().amax(dim=1)
        non_crouch_cap = torch.where(
            max_abs_curvature <= 0.15,
            torch.full_like(max_abs_curvature, self.fast_speed_cap_mps),
            torch.full_like(max_abs_curvature, self.curved_speed_cap_mps),
        )
        crouch_fraction = torch.isclose(
            height, torch.tensor(self.CROUCH_HEIGHT_M, device=self.device), atol=1.0e-5
        ).float().mean(dim=1)
        feasible_speed = 1.0 / (
            crouch_fraction / self.crouch_speed_cap_mps
            + (1.0 - crouch_fraction) / non_crouch_cap
        )
        speed_hi = feasible_speed.clamp(max=float(speed_max)).clamp_min(0.2)
        speed = 0.2 + torch.rand(count, device=self.device) * (speed_hi - 0.2)

        self.xy[env_ids] = xy
        self.yaw[env_ids] = yaw
        self.height[env_ids] = height
        self.arc[env_ids] = arc
        self.length[env_ids] = self.length_m
        self.speed[env_ids] = speed
        self.route_kind[env_ids] = 4
        self.profile_class[env_ids] = profile_class
        self.transition_boundary[env_ids] = boundary
        self.maximum_abs_curvature[env_ids] = max_abs_curvature
        self.progress_idx[env_ids] = 0
        self.progress[env_ids] = 0.0
        self._last_progress[env_ids] = 0.0
        self.cross_track[env_ids] = 0.0
        self.height_tracking_weight[env_ids] = 1.0

    def update(self, position_local: torch.Tensor) -> RouteState:
        state = super().update(position_local)
        distance_to_boundary = (state.progress - self.transition_boundary).abs()
        self.height_tracking_weight.copy_(
            (distance_to_boundary >= self.transition_margin_m).float()
        )
        return state
