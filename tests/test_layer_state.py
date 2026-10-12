"""A layer's state: what the index and the filter select must match the truth."""
import numpy as np
import pytest

from smappy.filter import LocFilter                      # noqa: E402
from smappy.locs import Localizations                    # noqa: E402
from smappy.render import (DisplaySettings, FieldOfView,  # noqa: E402
                            RenderSettings, render_locs)
from smappy.layer_state import ViewState                 # noqa: E402


def _table(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 10000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 8000, n).astype(np.float32),
        "z_nm": rng.uniform(-400, 400, n).astype(np.float32),
        "xy_err_nm": (rng.gamma(4, 3, n) + 5).astype(np.float32),
        "logl_rel": rng.normal(-1, 0.5, n).astype(np.float32),
        "frame": np.arange(n, dtype=np.int64),
    }, {"units": "nm"})


@pytest.mark.parametrize("mode", ["hist", "precision"])
def test_the_index_does_not_change_the_image(mode):
    """Selecting through the spatial index must match rendering everything.

    This is the whole safety property of the index: it may hand back extra
    candidates, but it may never lose one whose blob reaches the view.
    """
    locs = _table()
    state = ViewState(locs, RenderSettings(mode=mode))
    state.set_bound("xy_err_nm", None, 15.0)
    for fov in (FieldOfView.from_range((0, 10000), (0, 8000), 20.0),
                FieldOfView.from_range((2000, 3000), (2000, 3000), 2.0),
                FieldOfView.from_range((-5000, -4000), (0, 1000), 2.0)):
        through_index = state.render(fov)
        everything = render_locs(locs, fov, state.settings,
                                 select=state.filter.indices)
        assert np.allclose(through_index.weight, everything.weight, atol=1e-6)


def test_a_filter_change_is_visible_without_rebuilding_the_index():
    locs = _table()
    state = ViewState(locs)
    index = state.index
    fov = FieldOfView.from_range((0, 10000), (0, 8000), 20.0)
    before = state.render(fov).n_locs
    state.set_bound("xy_err_nm", None, 12.0)
    assert state.render(fov).n_locs < before
    assert state.index is index          # positions did not change, so it stands


def _groupable_table(n=3000, seed=5):
    """A table where each emitter is on for a few consecutive frames."""
    rng = np.random.default_rng(seed)
    emitters = rng.uniform(0, 5000, (n // 3, 2))
    x, y, frame = [], [], []
    for i, (px, py) in enumerate(emitters):
        for k in range(3):
            x.append(px + rng.normal(0, 5)); y.append(py + rng.normal(0, 5))
            frame.append(i % 500 + k)
    n = len(x)
    return Localizations({
        "x_nm": np.array(x, np.float32), "y_nm": np.array(y, np.float32),
        "frame": np.array(frame, np.int64),
        "xy_err_nm": (rng.gamma(4, 3, n) + 5).astype(np.float32),
        "photons": rng.gamma(3, 500, n).astype(np.float32),
        "logl_rel": rng.normal(-1, 0.5, n).astype(np.float32),
    }, {"units": "nm"})


def test_one_bound_holds_on_both_tables_and_keeps_different_rows():
    """The bounds are the layer's: switching to the grouped table keeps them,
    and there they apply to the combined precision, which is smaller."""
    state = ViewState(_groupable_table())
    state.set_bound("xy_err_nm", None, 8.0)
    ungrouped_kept = state.filter.mask.mean()

    state.show_grouped(True)
    assert state.has_grouped
    assert len(state.locs) < len(state.tables.locs)
    assert state.filter.ranges == {"xy_err_nm": (None, 8.0)}
    assert state.filter.mask.mean() > ungrouped_kept  # a blink beats its frames

    state.show_grouped(False)
    assert state.filter.ranges == {"xy_err_nm": (None, 8.0)}
    assert state.filter.mask.mean() == ungrouped_kept


def test_switching_does_not_regroup():
    state = ViewState(_groupable_table())
    state.show_grouped(True)
    grouped = state.tables.peek_grouped()
    state.show_grouped(False)
    state.show_grouped(True)
    assert state.tables.peek_grouped() is grouped    # built once, then reused


def test_each_set_has_its_own_index_and_renders():
    state = ViewState(_groupable_table())
    fov = FieldOfView.from_range((0, 5000), (0, 5000), 20.0)
    ungrouped = state.render(fov).n_locs
    state.show_grouped(True)
    assert state.index is not state.tables.ungrouped.index
    assert state.render(fov).n_locs < ungrouped
