"""CPU checks of clean drawing semantics; no Isaac Sim startup required."""
import numpy as np
import pytest

from scripts.diffusion_policy.visualization.path_debug_viz import PathDebugVisualizer, PathDebugVizCfg
from scripts.diffusion_policy.visualization.shaded_lines import tube_geometry


class Draw:
    def __init__(self):
        self.calls = []
    def clear_lines(self):
        self.calls.clear()
    def draw_lines(self, starts, ends, colors, widths):
        self.calls.append((np.array(starts), np.array(ends), colors, widths))


class Marker:
    def visualize(self, *, translations):
        self.position = translations.copy()


def visualizer(style="clean", height=False, preview=True):
    instance = object.__new__(PathDebugVisualizer)
    instance.cfg = PathDebugVizCfg(show_path=True, show_preview=preview, show_height=height, style=style)
    instance._draw = Draw()
    instance._actual_trace = []
    instance._goal_sphere = None
    path = np.array([[0,0,.2932], [1,0,.2932], [2,0,.1705], [3,0,.1705], [4,0,.2932]], dtype=np.float32)
    instance.set_plan(path, np.zeros(len(path)), clear_actual=True)
    return instance


@pytest.mark.parametrize("progress,goal", [(0,0), (0,2), (1,3), (4,4)])
def test_clean_segments_cover_the_reference_exactly_once(progress, goal):
    viz = visualizer()
    viz.update(robot_position_w=[0,0,.29], progress_index=progress, goal_index=goal)
    rendered = [(tuple(a), tuple(b)) for starts, ends, _, _ in viz._draw.calls for a,b in zip(starts,ends)]
    expected = [(tuple(a), tuple(b)) for a,b in zip(viz._path_w[:-1],viz._path_w[1:])]
    assert len(rendered) == len(expected)
    assert set(rendered) == set(expected)


def test_classic_keeps_original_reference_plus_overlay():
    viz = visualizer("classic")
    viz.update(robot_position_w=[0,0,.29], progress_index=1, goal_index=3)
    assert sum(len(call[0]) for call in viz._draw.calls) == 6  # four reference, two preview


def test_goal_center_and_yaw_source_are_not_replaced():
    viz = visualizer()
    viz._goal_sphere = Marker()
    viz.update(robot_position_w=[0,0,.29], progress_index=1, goal_index=3)
    np.testing.assert_array_equal(viz._goal_sphere.position, viz._path_w[3:4])


def test_shaded_geometry_keeps_xyz_centrelines_including_vertical_segments():
    starts = np.array([[0,0,.2932], [1,0,.1705], [2,0,.1705]])
    ends = np.array([[1,0,.2932], [1,0,.2932], [2,0,.1705]])
    points, faces = tube_geometry(starts, ends, [.009,.009,.009])
    rings = points.reshape(2,12,3)  # degenerate final segment omitted
    np.testing.assert_allclose(rings[:,:6].mean(axis=1), starts[:2], atol=1e-6)
    np.testing.assert_allclose(rings[:,6:].mean(axis=1), ends[:2], atol=1e-6)
    assert np.isfinite(points).all()
    assert len(faces) == 2*6*4


def test_optional_clean_height_guides_are_sparse_and_include_height_changes():
    viz = visualizer(height=True)
    x = np.linspace(0,4,201)
    path = np.column_stack([x, np.zeros(len(x)), np.where(x<2,.2932,.1705)]).astype(np.float32)
    viz.set_plan(path,np.zeros(len(x)),clear_actual=True)
    viz._draw_height_guides()
    assert len(viz._draw.calls[0][0]) <= 10
    assert path[99].tolist() in viz._draw.calls[0][1].tolist()
    assert path[100].tolist() in viz._draw.calls[0][1].tolist()


def test_invalid_style_is_rejected_before_simulation_imports():
    with pytest.raises(ValueError,match="classic or clean"):
        PathDebugVisualizer(PathDebugVizCfg(show_path=True,style="invalid"),np.zeros((2,3)),np.zeros(2))
