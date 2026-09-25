"""What the render window shows: it keeps rendering, and it says how much.

A render that fails -- a table swapped while it was being drawn, during a live
fit -- once left the view waiting forever, so the image stayed black.  And the
count on the window was the table's length, not what the filter and grouping
left on screen, which hid why a picture looked empty.
"""
import numpy as np
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from smappy.io.formats import FileInfo                                # noqa: E402
from smappy.locs import Localizations                                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def table(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "frame": rng.integers(0, 2000, n).astype(np.int64),
        "photons": rng.uniform(100, 900, n).astype(np.float32),
        "xy_err_nm": rng.uniform(5, 60, n).astype(np.float32),
    }, {"units": "nm"})


def settle(app, rounds=40):
    import time
    for _ in range(rounds):
        app.processEvents()
        time.sleep(0.01)


def test_a_failed_render_does_not_leave_the_view_black(app):
    from smappy.gui.render_view import RenderView
    session = Session()
    session.add_file(table(), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    view = RenderView(session)
    view.resize(400, 400)
    view.show()

    real = view.composite
    calls = {"n": 0}

    def once_broken(fov):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("table swapped mid-render")
        return real(fov)

    view.composite = once_broken
    view.tile_valid = False
    view.render()
    settle(app)
    assert not view._busy, "a failed render left the view waiting for good"
    assert "table swapped" in view.last_error

    view.tile_valid = False
    view.render()                                    # the next one goes through
    settle(app)
    assert view.fov is not None and view.image.image is not None
    assert float(np.max(view.image.image)) > 0


def test_the_filter_list_marks_every_bounded_field(app):
    from smappy.gui.render_tab import FilterWidget
    session = Session()
    session.add_file(table(), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    widget = FilterWidget(session)
    layer = session.layers[0]
    widget.bind(layer)

    def item(name):
        return widget.field.findData(name)

    # the default precision bound is marked; an untouched field is not
    assert widget.field.itemText(item("xy_err_nm")).startswith("●")
    assert widget.field.itemData(item("xy_err_nm"), 0x0006).bold()   # FontRole
    assert widget.field.itemText(item("photons")) == "photons"

    # a bound set on a field that is not on show is marked too
    widget._select("x_nm")
    layer.set_bound("photons", 300.0, None)
    widget.refresh()
    assert widget.field.itemText(item("photons")).startswith("●")

    # reading the current field gives the column, never the marked label
    widget._select("xy_err_nm")
    assert widget.current_field() == "xy_err_nm"
    widget._apply(None, None)                        # clearing removes the mark
    assert widget.field.itemText(item("xy_err_nm")) == "xy_err_nm"


def test_the_title_counts_what_is_shown_not_the_table(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    from smappy import config
    config.load(reload=True)
    from smappy.gui.app import ControlWindow, RenderWindow
    session = Session()
    control = ControlWindow(session, RenderWindow(session))
    session.add_file(table(), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    settle(app, 5)
    layer = session.layers[0]
    shown = len(layer.filter)
    assert 0 < shown < len(session.locs) or layer.grouped
    title = control.render_window.windowTitle()
    assert f"{shown:,} of {len(layer.locs):,}" in title and "shown" in title

    layer.set_bound("photons", 800.0, None)
    session.changed("layer")
    assert f"{len(layer.filter):,} of" in control.render_window.windowTitle()


def test_the_filter_fits_the_control_window_with_every_quick_button(app):
    """Eight quick buttons and the drop-down on one row made the filter ~440
    px wide in a 380 px window; the Render tab scrolls only vertically, so the
    histogram's right end -- and the handle of the upper bound -- was cut off."""
    from smappy.gui.render_tab import QUICK_FIELDS, FilterWidget
    from smappy.gui.widgets import CONTROL_WIDTH
    rng = np.random.default_rng(3)
    n = 1000
    columns = {"x_nm": rng.uniform(0, 1e4, n), "y_nm": rng.uniform(0, 1e4, n)}
    for _, names in QUICK_FIELDS:
        columns[names[0]] = rng.uniform(0, 10, n)
    session = Session(Localizations(columns, {}))
    widget = FilterWidget(session)
    widget.bind(session.layers[0])
    assert len(widget.quick) == len(QUICK_FIELDS)
    scrollbar_and_margins = 40
    assert widget.minimumSizeHint().width() <= CONTROL_WIDTH - scrollbar_and_margins
