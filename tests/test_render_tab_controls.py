"""The Render tab's controls as they read: numbers to the digits they mean,
only the settings the render mode uses, the quick buttons side by side, every
bound cleared at once, a histogram whose bounds are easy to take, and an
overview that is the data and nothing around it.  And the About box and the
Help window's contrast."""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.io.formats import FileInfo                                # noqa: E402
from smappy.locs import Localizations                                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def table(n=4000, seed=0, width=10_000.0, height=10_000.0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, width, n).astype(np.float32),
        "y_nm": rng.uniform(0, height, n).astype(np.float32),
        "frame": rng.integers(0, 2000, n).astype(np.int64),
        "photons": rng.uniform(100, 900, n).astype(np.float32),
        "xy_err_nm": rng.uniform(5, 60, n).astype(np.float32),
    }, {"units": "nm"})


def test_filled_in_numbers_are_rounded_to_a_thousandth_of_their_range():
    from smappy.gui.widgets import round_in_span, round_significant
    assert round_in_span(23.481726, 100.0) == 23.5
    assert round_in_span(1234.5678, 5000.0) == 1235.0
    assert round_in_span(0.0123456, 0.05) == 0.01235
    assert round_significant(1234.567) == 1230.0
    assert round_significant(0.0123456) == 0.0123


def test_a_spin_box_leaves_out_the_zeros_after_the_point(app):
    from smappy.gui.widgets import CompactSpinBox
    box = CompactSpinBox(decimals=2, maximum=100)
    box.setValue(3.0)
    assert box.text() == "3"
    box.setValue(2.5)
    assert box.text() == "2.5"
    assert box.valueFromText(box.text()) == 2.5


def test_the_colour_range_auto_is_rounded(app):
    from smappy.gui.render_tab import RenderTab
    session = Session(table())
    tab = RenderTab(session)
    lo, hi = tab._field_range("photons")
    assert lo == round(lo, 1) and hi == round(hi, 1)


def test_more_shows_only_the_widths_the_render_mode_reads(app):
    from smappy.gui.render_tab import RenderTab
    session = Session(table())
    tab = RenderTab(session)
    form = tab._more_form
    tab.set_mode("precision")
    assert form.isRowVisible(tab.factor)
    assert not form.isRowVisible(tab.sigma) and not form.isRowVisible(tab.sigma_y)
    tab.set_mode("gauss")
    # x against y is drawn round: one sigma
    assert form.isRowVisible(tab.sigma) and not form.isRowVisible(tab.sigma_y)
    assert form.labelForField(tab.sigma).text() == "sigma"
    assert not form.isRowVisible(tab.factor)
    tab.set_mode("hist")
    assert not any(form.isRowVisible(w) for w in (tab.sigma, tab.sigma_y, tab.factor))
    assert form.isRowVisible(tab.gamma)


def test_the_quick_buttons_wrap_only_when_the_row_is_full(app):
    from smappy.gui.render_tab import QUICK_FIELDS, FilterWidget
    rng = np.random.default_rng(3)
    columns = {"x_nm": rng.uniform(0, 1e4, 1000), "y_nm": rng.uniform(0, 1e4, 1000)}
    for _, names in QUICK_FIELDS:
        columns[names[0]] = rng.uniform(0, 10, 1000)
    session = Session(Localizations(columns, {}))
    widget = FilterWidget(session)
    widget.bind(session.layers[0])
    widget.resize(1200, 400)
    widget.show()
    app.processEvents()
    tops = {b.geometry().top() for b in widget.quick.values()}
    assert len(tops) == 1                         # room for all: one row
    widget.resize(300, 400)
    app.processEvents()
    buttons = list(widget.quick.values())
    assert len({b.geometry().top() for b in buttons}) > 1
    # in their order, each row filled before the next begins
    places = [(b.geometry().top(), b.geometry().left()) for b in buttons]
    assert places == sorted(places)
    widget.close()


def test_the_cross_beside_the_bounds_clears_every_bound(app):
    from smappy.gui.render_tab import FilterWidget
    session = Session()
    session.add_file(table(), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    widget = FilterWidget(session)
    layer = session.layers[0]
    widget.bind(layer)
    layer.set_bound("photons", 300.0, None)
    widget.refresh()
    assert not widget.clear_all.isHidden()
    widget.clear_all.click()
    assert all(r == (None, None) for r in layer.ranges.values())
    assert widget.clear_all.isHidden()


def test_a_dragged_bound_is_a_round_number(app):
    from smappy.gui.render_tab import FilterWidget
    session = Session(table())
    widget = FilterWidget(session)
    layer = session.layers[0]
    widget.bind(layer)
    widget._select("photons")
    lo, hi = widget.region.getRegion()
    widget.region.setRegion((lo, 512.34567))
    widget._on_region()
    assert layer.ranges["photons"][1] == pytest.approx(512.3)


def test_the_histogram_zooms_out_only_a_little_past_the_data(app):
    from smappy.gui.render_tab import HANDLE_REACH, HIST_MARGIN, FilterWidget
    session = Session(table())
    widget = FilterWidget(session)
    widget.bind(session.layers[0])
    widget._select("photons")
    _, edges = widget._histogram("photons")
    limits = widget.plot.getViewBox().state["limits"]
    span = edges[-1] - edges[0]
    assert limits["xLimits"][0] >= edges[0] - (HIST_MARGIN + 0.01) * span
    assert limits["xLimits"][1] <= edges[-1] + (HIST_MARGIN + 0.01) * span
    # the bounds' lines reach further for the mouse than pyqtgraph's own
    assert all(line._maxMarkerSize == HANDLE_REACH for line in widget.region.lines)


def test_the_overview_is_the_data_s_extent_at_the_panel_s_shape(app):
    from smappy.gui.render_tab import OVERVIEW_MAX, OVERVIEW_MIN, RenderTab
    from smappy.gui.render_view import RenderView
    session = Session()
    view = RenderView(session)
    tab = RenderTab(session, view)
    tab.resize(380, 800)
    tab.show()
    session.add_file(table(width=10_000, height=5_000),
                     FileInfo("t.hdf5", "/t.hdf5", "smappy"))
    app.processEvents()
    overview = tab.overview
    overview.update_image()
    image = overview.image.image
    ny, nx = image.shape[:2]
    assert ny / nx == pytest.approx(0.5, rel=0.05)      # no black margin drawn
    graphics = overview.graphics
    assert OVERVIEW_MIN <= graphics.height() <= OVERVIEW_MAX
    assert graphics.height() == pytest.approx(graphics.width() * 0.5, abs=3)
    tab.close()


def test_a_click_in_the_overview_centres_the_main_view_there(app):
    from PySide6.QtCore import QPointF, QRectF
    from smappy.gui.render_tab import Overview
    from smappy.gui.render_view import RenderView
    session = Session(table())
    view = RenderView(session)
    view.resize(600, 600)
    view.show()
    overview = Overview(session, view)
    overview.resize(380, 300)
    overview.show()
    app.processEvents()
    overview.update_image()
    view.view.setRange(QRectF(0, 0, 1000, 1000), padding=0)
    app.processEvents()
    before = view.view.viewRect()

    class Click:                        # what pyqtgraph hands the item
        accepted = False

        def pos(self):
            # three quarters across and a quarter down, in the image's pixels
            ny, nx = overview.image.image.shape[:2]
            return QPointF(0.75 * nx, 0.25 * ny)

        def accept(self):
            self.accepted = True

    click = Click()
    overview.image.mouseClickEvent(click)
    target = overview.image.mapToView(click.pos())
    app.processEvents()
    after = view.view.viewRect()
    assert click.accepted
    assert after.center().x() == pytest.approx(target.x(), abs=1)
    assert after.center().y() == pytest.approx(target.y(), abs=1)
    assert 6000 < target.x() < 9000 and 1000 < target.y() < 4000
    assert after.width() == pytest.approx(before.width(), rel=1e-3)   # the zoom kept
    overview.close()
    view.close()


def test_the_overview_detached_fills_its_window(app):
    from smappy.gui.render_tab import Overview
    from smappy.gui.render_view import RenderView
    from smappy.gui.widgets import FloatingWindow
    session = Session(table())
    overview = Overview(session, RenderView(session))
    window = FloatingWindow("overview", overview)
    window.resize(700, 700)
    window.show()
    app.processEvents()
    assert overview.graphics.height() > 400
    window.close()


def test_about_says_the_version_the_licence_github_and_what_to_cite():
    from smappy.gui import about
    text = about.about_html()
    assert about.version() in text and "BSD" in text
    assert "github.com/ries-lab/SMAPpy" in text
    assert "10.1038/s41592-020-0938-1" in text


def test_the_help_window_writes_black_on_white_or_white_on_black(app):
    from PySide6.QtGui import QPalette
    from smappy.gui.help_window import HelpWindow
    window = HelpWindow()
    palette = window.browser.palette()
    text = palette.color(QPalette.Text).name()
    base = palette.color(QPalette.Base).name()
    assert (text, base) in (("#000000", "#ffffff"), ("#ffffff", "#000000"))


def test_auto_scales_the_colours_to_what_is_shown(app):
    """A column first chosen takes its range from every localization; *auto*
    from those the picture shows -- here, the half of the field in view."""
    from PySide6.QtCore import QRectF
    from smappy.gui.render_tab import RenderTab
    locs = table()
    locs.columns["photons"] = (locs["x_nm"] / 10).astype(np.float32)   # 0..1000
    session = Session(locs)
    tab = RenderTab(session)
    tab.color.setCurrentIndex(1)
    tab.color_field.setCurrentText("photons")
    lo, hi = session.layers[0].state.settings.color_range
    assert lo < 50 and hi > 950
    tab.filter.view_rect = lambda: QRectF(0, 0, 5000, 10_000)        # the left half
    tab.color_auto.click()
    lo, hi = session.layers[0].state.settings.color_range
    assert lo < 50 and 450 < hi < 510
