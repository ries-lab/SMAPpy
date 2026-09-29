"""The filter's histogram: drawn over the bulk, and a drag moves one bound."""
import numpy as np
import pytest

from smappy.filter import histogram_range
from smappy.locs import Localizations


def test_the_histogram_spans_the_bulk_of_a_long_tailed_precision():
    rng = np.random.default_rng(0)
    err = np.concatenate([rng.normal(5, 1, 9500), rng.uniform(30, 400, 500)])
    lo, hi = histogram_range(err)
    assert hi < 20                       # the 99th percentile is ~ 360
    assert lo < 3


def test_a_symmetric_column_keeps_its_percentiles():
    x = np.random.default_rng(0).uniform(0, 10_000, 10_000)
    lo, hi = histogram_range(x)
    assert lo == pytest.approx(100, abs=30) and hi == pytest.approx(9900, abs=30)


def test_a_column_of_one_value_still_gets_a_span():
    lo, hi = histogram_range(np.ones(100))
    assert hi > lo


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_dragging_one_end_keeps_a_bound_beyond_the_histogram_at_the_other(app):
    from smappy.gui.render_tab import FilterWidget
    from smappy.session import Session
    rng = np.random.default_rng(0)
    n = 5000
    session = Session(Localizations({
        "x_nm": rng.uniform(0, 1e4, n), "y_nm": rng.uniform(0, 1e4, n),
        "frame": np.arange(n), "photons": rng.uniform(100, 900, n),
        "xy_err_nm": np.concatenate([rng.normal(5, 1, n - 250),
                                     rng.uniform(30, 400, 250)])}, {"units": "nm"}))
    layer = session.layers[0]
    layer.set_bound("xy_err_nm", None, 25.0)
    widget = FilterWidget(session)
    widget.bind(layer)
    widget._select("xy_err_nm")
    widget._show_field()
    _, edges = widget._histogram("xy_err_nm")
    assert edges[-1] < 25                          # the bound is off the histogram
    widget.region.setRegion((3.0, widget.region.getRegion()[1]))
    widget._on_region()
    assert layer.filter.ranges["xy_err_nm"] == (3.0, 25.0)
    widget.region.setRegion((edges[0] - 1, widget.region.getRegion()[1]))
    widget._on_region()                            # dragged to the edge: no bound
    assert layer.filter.ranges["xy_err_nm"] == (None, 25.0)
