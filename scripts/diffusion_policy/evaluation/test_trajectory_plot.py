import numpy as np
import pytest
from .trajectory_plot import active_trajectory_xy


def test_empty_and_staging_failure_do_not_plot_unobserved_reset():
    assert active_trajectory_xy([], 0).shape == (0, 2)
    assert active_trajectory_xy([[1, 2]], 0).shape == (0, 2)


def test_only_active_xy_samples_are_drawn():
    np.testing.assert_equal(active_trajectory_xy([[1, 2], [3, 4]], 1), [[1, 2]])
    with pytest.raises(ValueError):
        active_trajectory_xy([[1, 2, 3]], 1)
