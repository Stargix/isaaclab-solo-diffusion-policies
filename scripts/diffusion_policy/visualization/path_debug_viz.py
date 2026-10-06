"""Lightweight viewport overlays for closed-loop path-policy playback.

The policy never reads these prims: they are debug-only USD/debug-draw objects
created on demand by :mod:`play_policy`.  Keeping them here prevents viewer
instrumentation from leaking into the control/conditioning code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathDebugVizCfg:
    """Options for a single-environment route overlay."""

    show_path: bool = False
    show_goal: bool = False
    show_preview: bool = False
    show_actual_path: bool = False
    show_height: bool = False
    max_actual_points: int = 750
    style: str = "classic"

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.show_path,
                self.show_goal,
                self.show_preview,
                self.show_actual_path,
                self.show_height,
            )
        )


class PathDebugVisualizer:
    """Draw a planned path, current target frame and measured robot trace.

    Classic lines use transient debug draw; the clean preset uses shaded,
    non-colliding geometry. The target sphere
    is a USD marker instancer, which remains visible and selectable in the
    Stage tree under ``/World/Visuals/PathDebug``.  Heading is intentionally
    represented only by the planar arrow, avoiding redundant XYZ axes.
    """

    _REFERENCE_COLOR = (1.0, 0.08, 0.08, 1.0)
    _ACTUAL_COLOR = (0.05, 0.48, 0.96, 1.0)
    _PREVIEW_COLOR = (1.0, 0.55, 0.05, 1.0)
    _HEIGHT_COLOR = (0.42, 0.52, 0.68, 0.50)
    _HEADING_COLOR = (1.0, 0.78, 0.08, 1.0)

    def __init__(self, cfg: PathDebugVizCfg, path_w: np.ndarray, yaws_w: np.ndarray):
        if not cfg.enabled:
            raise ValueError("PathDebugVisualizer requires at least one enabled overlay.")
        if cfg.max_actual_points < 2:
            raise ValueError("max_actual_points must be at least two.")
        if cfg.style not in ("classic", "clean"):
            raise ValueError("Path visualization style must be classic or clean.")

        # These imports must remain lazy: this module is importable in unit
        # tests and CLI parsing without starting an Isaac Sim application.
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.cfg = cfg
        if cfg.style == "clean":
            from .shaded_lines import ShadedLineDraw

            self._draw = ShadedLineDraw()
            self._REFERENCE_COLOR = (0.85, 0.055, 0.04, 1.0)
            self._ACTUAL_COLOR = (0.035, 0.30, 0.74, 1.0)
            self._PREVIEW_COLOR = (0.95, 0.48, 0.06, 1.0)
        else:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self._draw = omni_debug_draw.acquire_debug_draw_interface()
        self._actual_trace: list[np.ndarray] = []
        self._path_w = np.empty((0, 3), dtype=np.float32)
        self._yaws_w = np.empty((0,), dtype=np.float32)
        self.set_plan(path_w, yaws_w, clear_actual=True)

        self._goal_sphere = None
        if cfg.show_goal:
            sphere_cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/PathDebug/goal_point",
                markers={
                    "goal": sim_utils.SphereCfg(
                        radius=0.030,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.9, 0.05), roughness=0.35
                        ),
                    )
                },
            )
            self._goal_sphere = VisualizationMarkers(sphere_cfg)

    @property
    def path_size(self) -> int:
        return len(self._path_w)

    def set_plan(self, path_w: np.ndarray, yaws_w: np.ndarray, *, clear_actual: bool) -> None:
        """Replace the displayed route, e.g. after a default-path reset."""

        path = np.asarray(path_w, dtype=np.float32)
        yaws = np.asarray(yaws_w, dtype=np.float32).reshape(-1)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
            raise ValueError("The debug path must have shape (N>=2, 3).")
        if yaws.shape != (len(path),):
            raise ValueError("The debug yaw vector must have one value per path point.")
        self._path_w = path.copy()
        self._yaws_w = yaws.copy()
        if clear_actual:
            self._actual_trace.clear()

    @staticmethod
    def _draw_polyline(draw: object, points: np.ndarray, color: tuple[float, float, float, float], width: float) -> None:
        if len(points) < 2:
            return
        draw.draw_lines(
            points[:-1].astype(np.float32).tolist(),
            points[1:].astype(np.float32).tolist(),
            [color] * (len(points) - 1),
            [float(width)] * (len(points) - 1),
        )

    def _draw_height_guides(self) -> None:
        guide_points = self._path_w
        if self.cfg.style == "clean":
            changes = np.flatnonzero(np.abs(np.diff(guide_points[:, 2])) > 0.04)
            indices = np.unique(np.concatenate((
                np.linspace(0, len(guide_points) - 1, min(8, len(guide_points))).astype(int),
                changes, changes + 1,
            )))
            guide_points = guide_points[indices]
        floor_points = guide_points.copy()
        floor_points[:, 2] = 0.015
        self._draw.draw_lines(
            floor_points.astype(np.float32).tolist(),
            guide_points.astype(np.float32).tolist(),
            [self._HEIGHT_COLOR] * len(guide_points),
            [1.0] * len(guide_points),
        )

    def _draw_heading_arrow(self, position: np.ndarray, yaw: float) -> None:
        """Draw a short heading arrow anchored at the goal-point center.

        The previous symmetric arrow was centred on the sphere.  Its rear half
        was therefore hidden by the opaque sphere, making the visible arrow
        look detached and shifted forward.  Starting the shaft at the exact
        sphere centre makes the attachment unambiguous; the sphere naturally
        occludes only the part that should be behind it.
        """

        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        lateral = np.array([-heading[1], heading[0], 0.0], dtype=np.float32)
        # Keep the lift small: a large Z offset projects as a visible lateral
        # displacement in the usual oblique Isaac Sim camera view.
        shaft_start = position + np.array([0.0, 0.0, 0.008], dtype=np.float32)
        arrow_length = 0.14
        tip = shaft_start + arrow_length * heading
        head_length = 0.045
        head_width = 0.030
        wing_a = tip - head_length * heading + head_width * lateral
        wing_b = tip - head_length * heading - head_width * lateral
        self._draw.draw_lines(
            [shaft_start.tolist(), tip.tolist(), tip.tolist()],
            [tip.tolist(), wing_a.tolist(), wing_b.tolist()],
            [self._HEADING_COLOR] * 3,
            [3.0, 3.0, 3.0],
        )

    def update(self, *, robot_position_w: np.ndarray, progress_index: int, goal_index: int) -> None:
        """Refresh all enabled overlays for the current control step."""

        robot_position = np.asarray(robot_position_w, dtype=np.float32).reshape(3)
        self._actual_trace.append(robot_position.copy())
        if len(self._actual_trace) > self.cfg.max_actual_points:
            self._actual_trace = self._actual_trace[-self.cfg.max_actual_points :]

        progress = int(np.clip(progress_index, 0, self.path_size - 1))
        goal = int(np.clip(goal_index, progress, self.path_size - 1))

        self._draw.clear_lines()
        if self.cfg.show_path:
            if self.cfg.style == "clean" and self.cfg.show_preview:
                # Each segment is drawn once, avoiding red/orange depth fighting.
                self._draw_polyline(self._draw, self._path_w[: progress + 1], self._REFERENCE_COLOR, width=3.0)
                self._draw_polyline(self._draw, self._path_w[goal:], self._REFERENCE_COLOR, width=3.0)
            else:
                self._draw_polyline(self._draw, self._path_w, self._REFERENCE_COLOR, width=3.0)
        if self.cfg.show_height:
            self._draw_height_guides()
        if self.cfg.show_preview:
            self._draw_polyline(
                self._draw,
                self._path_w[progress : goal + 1],
                self._PREVIEW_COLOR,
                width=4.0,
            )
        if self.cfg.show_actual_path:
            self._draw_polyline(
                self._draw,
                np.asarray(self._actual_trace, dtype=np.float32),
                self._ACTUAL_COLOR,
                width=2.0,
            )

        if self._goal_sphere is not None:
            target_position = self._path_w[goal : goal + 1]
            self._goal_sphere.visualize(translations=target_position)
            self._draw_heading_arrow(target_position[0], float(self._yaws_w[goal]))

    def close(self) -> None:
        """Remove transient lines and hide persistent marker instancers."""

        self._draw.clear_lines()
        for marker in (self._goal_sphere,):
            if marker is not None:
                marker.set_visibility(False)
