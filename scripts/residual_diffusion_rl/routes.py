"""GPU-vectorized route bank for Phase B1.

Routes are local to each IsaacLab environment origin.  The bank deliberately
uses a small, auditable family: it tests geometric transfer without turning B1
into a general navigation benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RouteState:
    progress: torch.Tensor
    progress_delta: torch.Tensor
    cross_track: torch.Tensor
    tangent_yaw: torch.Tensor
    target_height: torch.Tensor
    remaining_distance: torch.Tensor
    remaining_fraction: torch.Tensor
    terminal_distance: torch.Tensor
    terminal_along_error: torch.Tensor
    near_terminal: torch.Tensor


class RouteBank:
    """Per-environment paths with batched projection and sampling."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        points: int = 101,
        length_m: float = 4.0,
        height_segment_m: float = 0.8,
        walk_height: float = 0.2932,
        crouch_height: float = 0.1705,
    ):
        if points < 16:
            raise ValueError("routes need at least 16 points")
        if length_m <= 0.0 or height_segment_m <= 0.0:
            raise ValueError("route and height segment lengths must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.points = points
        self.length_m = float(length_m)
        self.height_segment_m = float(height_segment_m)
        self.height_values = torch.tensor(
            [walk_height, 0.25, 0.21, crouch_height], device=self.device
        )
        self.xy = torch.zeros(num_envs, points, 2, device=self.device)
        self.yaw = torch.zeros(num_envs, points, device=self.device)
        self.height = torch.full((num_envs, points), walk_height, device=self.device)
        self.arc = torch.zeros(num_envs, points, device=self.device)
        self.length = torch.ones(num_envs, device=self.device)
        self.speed = torch.full((num_envs,), 0.3, device=self.device)
        self.progress_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.progress = torch.zeros(num_envs, device=self.device)
        self._last_progress = torch.zeros_like(self.progress)
        self.cross_track = torch.zeros_like(self.progress)
        self.route_kind = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._kind_sample_counter = 0

    def reset(
        self,
        env_ids: torch.Tensor,
        stage: int,
        *,
        stratified: bool = False,
        speed_max: float | None = None,
        height_stage: int | None = None,
    ) -> None:
        """Sample routes with independently selectable geometry and height stages.

        ``height_stage=None`` preserves the historical coupling to ``stage``.
        Supplying it lets DPPO expose intermediate posture targets without
        simultaneously changing route geometry and speed difficulty.
        """

        count = len(env_ids)
        if count == 0:
            return
        u = torch.linspace(0.0, 1.0, self.points, device=self.device).expand(count, -1)
        max_kind = 0 if stage <= 0 else 2 if stage == 1 else 3
        if stratified:
            kind = (
                torch.arange(count, device=self.device) + self._kind_sample_counter
            ) % (max_kind + 1)
            self._kind_sample_counter = (self._kind_sample_counter + count) % (max_kind + 1)
        else:
            kind = torch.randint(0, max_kind + 1, (count,), device=self.device)
        x = self.length_m * u
        y = torch.zeros_like(x)

        s_curve = kind == 1
        y[s_curve] = 0.42 * torch.sin(2.0 * torch.pi * u[s_curve])

        right_angle = kind == 2
        if torch.any(right_angle):
            ur = u[right_angle]
            x[right_angle] = torch.where(ur <= 0.5, self.length_m * ur, 0.5 * self.length_m)
            y[right_angle] = torch.where(ur <= 0.5, torch.zeros_like(ur), self.length_m * (ur - 0.5))

        random_curve = kind == 3
        if torch.any(random_curve):
            ur = u[random_curve]
            amp1 = torch.empty((int(random_curve.sum()), 1), device=self.device).uniform_(-0.45, 0.45)
            amp2 = torch.empty_like(amp1).uniform_(-0.22, 0.22)
            y[random_curve] = amp1 * torch.sin(torch.pi * ur) + amp2 * torch.sin(3.0 * torch.pi * ur)

        xy = torch.stack((x, y), dim=-1)
        delta = xy[:, 1:] - xy[:, :-1]
        segment = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-6)
        arc = torch.cat((torch.zeros(count, 1, device=self.device), torch.cumsum(segment, dim=1)), dim=1)
        tangent = torch.cat((delta, delta[:, -1:]), dim=1)
        yaw = torch.atan2(tangent[..., 1], tangent[..., 0])

        profile_stage = stage if height_stage is None else int(height_stage)
        if profile_stage not in (0, 1, 2):
            raise ValueError("height_stage must be 0, 1, 2 or None.")
        if profile_stage <= 0:
            values = torch.randint(0, 2, (count, 1), device=self.device) * 3
            height = self.height_values[values].expand(-1, self.points)
        else:
            section = torch.floor(arc / self.height_segment_m).long()
            num_sections = int(torch.ceil(arc.max() / self.height_segment_m).item()) + 1
            if profile_stage == 1:
                choices = torch.randint(0, 2, (count, num_sections), device=self.device) * 3
            else:
                choices = torch.randint(0, len(self.height_values), (count, num_sections), device=self.device)
            height = torch.gather(choices, 1, section.clamp_max(num_sections - 1))
            height = self.height_values[height]

        self.xy[env_ids] = xy
        self.yaw[env_ids] = yaw
        self.height[env_ids] = height
        self.arc[env_ids] = arc
        self.length[env_ids] = arc[:, -1]
        speed_hi = float(speed_max) if speed_max is not None else (
            0.4 if stage <= 0 else 0.5 if stage == 1 else 0.6
        )
        if speed_hi < 0.2:
            raise ValueError("speed_max must be at least 0.2 m/s.")
        self.speed[env_ids] = torch.empty(count, device=self.device).uniform_(0.2, speed_hi)
        self.route_kind[env_ids] = kind
        self.progress_idx[env_ids] = 0
        self.progress[env_ids] = 0.0
        self._last_progress[env_ids] = 0.0
        self.cross_track[env_ids] = 0.0

    def update(self, position_local: torch.Tensor) -> RouteState:
        """Project robot positions while preventing jumps to distant branches."""

        index = torch.arange(self.points, device=self.device)[None, :]
        low = (self.progress_idx - 5).clamp_min(0)[:, None]
        high = (self.progress_idx + 35).clamp_max(self.points - 1)[:, None]
        allowed = (index >= low) & (index <= high)
        distance_sq = torch.sum((self.xy - position_local[:, None, :]) ** 2, dim=-1)
        distance_sq = torch.where(allowed, distance_sq, torch.full_like(distance_sq, torch.inf))
        nearest = distance_sq.argmin(dim=1)
        self.progress_idx = torch.maximum(self.progress_idx, nearest)
        rows = torch.arange(self.num_envs, device=self.device)
        projection = self.xy[rows, self.progress_idx]
        tangent_yaw = self.yaw[rows, self.progress_idx]
        error = position_local - projection
        signed = torch.cos(tangent_yaw) * error[:, 1] - torch.sin(tangent_yaw) * error[:, 0]
        new_progress = self.arc[rows, self.progress_idx]
        delta_progress = (new_progress - self.progress).clamp_min(0.0)
        self._last_progress.copy_(self.progress)
        self.progress.copy_(new_progress)
        self.cross_track.copy_(signed)
        remaining_distance = (self.length - new_progress).clamp_min(0.0)
        remaining = (remaining_distance / self.length.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        terminal_error = position_local - self.xy[:, -1]
        terminal_distance = torch.linalg.vector_norm(terminal_error, dim=1)
        terminal_yaw = self.yaw[:, -1]
        terminal_along_error = (
            torch.cos(terminal_yaw) * terminal_error[:, 0]
            + torch.sin(terminal_yaw) * terminal_error[:, 1]
        )
        near_terminal = remaining <= (0.12 / self.length).clamp_max(0.05)
        return RouteState(
            progress=new_progress,
            progress_delta=delta_progress,
            cross_track=signed,
            tangent_yaw=tangent_yaw,
            target_height=self.height[rows, self.progress_idx],
            remaining_distance=remaining_distance,
            remaining_fraction=remaining,
            terminal_distance=terminal_distance,
            terminal_along_error=terminal_along_error,
            near_terminal=near_terminal,
        )

    def geometric_goal(
        self,
        position_local: torch.Tensor,
        robot_yaw: torch.Tensor,
        *,
        horizon_s: float,
        v_clip: float,
    ) -> torch.Tensor:
        """Torch equivalent of the checkpoint's geometric hindsight goal12."""

        if horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        start_s = self.progress
        end_s = torch.minimum(start_s + self.speed * horizon_s, self.length)
        fractions = torch.tensor((0.25, 0.5, 0.75), device=self.device)
        target_s = start_s[:, None] + fractions[None, :] * (end_s - start_s)[:, None]
        # Float32 cumulative sums can place an exact grid target a few ulps
        # below its route point. NumPy's training builder resolves that point,
        # so use a tiny left tolerance instead of advancing one full sample.
        preview_idx = torch.searchsorted(
            self.arc.contiguous(), (target_s - 1.0e-6).clamp_min(0.0).contiguous()
        ).clamp_max(self.points - 1)
        end_idx = torch.searchsorted(
            self.arc.contiguous(), (end_s[:, None] - 1.0e-6).clamp_min(0.0).contiguous()
        ).squeeze(1).clamp_max(self.points - 1)
        rows = torch.arange(self.num_envs, device=self.device)
        preview_xy = self.xy[rows[:, None], preview_idx]
        terminal_xy = self.xy[rows, end_idx]
        # The Phase-A goal contract stores the average speed of the selected
        # local segment, not the route-level command.  Once the lookahead is
        # clipped by the route end this value must taper to zero.
        local_segment_length = (
            self.arc[rows, end_idx] - self.arc[rows, self.progress_idx]
        ).clamp_min(0.0)
        local_average_speed = (local_segment_length / horizon_s).clamp(0.0, v_clip)
        c, s = torch.cos(robot_yaw), torch.sin(robot_yaw)

        def to_body(points: torch.Tensor) -> torch.Tensor:
            delta = points - position_local[:, None, :] if points.ndim == 3 else points - position_local
            return torch.stack((c[:, None] * delta[..., 0] + s[:, None] * delta[..., 1],
                                -s[:, None] * delta[..., 0] + c[:, None] * delta[..., 1]), dim=-1) if points.ndim == 3 else torch.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), dim=-1)

        preview_b = to_body(preview_xy).reshape(self.num_envs, 6)
        terminal_b = to_body(terminal_xy)
        dyaw = torch.atan2(torch.sin(self.yaw[rows, end_idx] - robot_yaw), torch.cos(self.yaw[rows, end_idx] - robot_yaw))
        return torch.cat((
            preview_b,
            terminal_b,
            torch.sin(dyaw)[:, None],
            torch.cos(dyaw)[:, None],
            self.height[rows, end_idx, None],
            local_average_speed[:, None],
        ), dim=1)

    def geometric_height_profile_goal(
        self,
        position_local: torch.Tensor,
        robot_yaw: torch.Tensor,
        *,
        horizon_s: float,
        v_clip: float,
    ) -> torch.Tensor:
        """Torch equivalent of Phase A's geometric height-profile goal16.

        ``[x25,y25,h25, ..., x100,y100,h100, h_now, sin(yaw), cos(yaw), v_avg]``.

        This intentionally follows the schema-8 NumPy builder exactly.  The
        legacy goal12 uses a small search tolerance retained for checkpoint
        compatibility, whereas schema 8 resolves the terminal route sample
        first and then takes arc fractions of that discrete segment.
        """

        if horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        start_s = self.progress
        requested_end_s = torch.minimum(start_s + self.speed * horizon_s, self.length)
        end_idx = torch.searchsorted(
            self.arc.contiguous(), requested_end_s[:, None].contiguous()
        ).squeeze(1).clamp_max(self.points - 1)
        rows = torch.arange(self.num_envs, device=self.device)
        end_s = self.arc[rows, end_idx]
        fractions = torch.tensor((0.25, 0.5, 0.75), device=self.device)
        target_s = start_s[:, None] + fractions[None, :] * (end_s - start_s)[:, None]
        preview_idx = torch.searchsorted(
            self.arc.contiguous(), target_s.contiguous()
        ).clamp_max(self.points - 1)
        sample_idx = torch.cat((preview_idx, end_idx[:, None]), dim=1)
        required_height = self.height[rows[:, None], sample_idx]
        sampled_xy = self.xy[rows[:, None], sample_idx]
        c, s = torch.cos(robot_yaw), torch.sin(robot_yaw)
        delta = sampled_xy - position_local[:, None, :]
        local_xy = torch.stack(
            (
                c[:, None] * delta[..., 0] + s[:, None] * delta[..., 1],
                -s[:, None] * delta[..., 0] + c[:, None] * delta[..., 1],
            ),
            dim=2,
        )
        dyaw = torch.atan2(
            torch.sin(self.yaw[rows, end_idx] - robot_yaw),
            torch.cos(self.yaw[rows, end_idx] - robot_yaw),
        )
        local_average_speed = (
            (end_s - self.arc[rows, self.progress_idx]).clamp_min(0.0) / horizon_s
        ).clamp(0.0, v_clip)

        spatial_tokens = torch.cat((local_xy, required_height[..., None]), dim=2)
        current_height = self.height[rows, self.progress_idx, None]
        return torch.cat(
            (
                spatial_tokens.reshape(self.num_envs, 12),
                current_height,
                torch.sin(dyaw)[:, None],
                torch.cos(dyaw)[:, None],
                local_average_speed[:, None],
            ),
            dim=1,
        )
