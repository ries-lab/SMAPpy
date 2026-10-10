"""The drawn region's outline, and how wide a tab is allowed to make the window."""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.locs import Localizations                                 # noqa: E402
from smappy.regions import Region                                     # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def table(n=2000):
    rng = np.random.default_rng(0)
    return Localizations({
        "x_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "frame": rng.integers(0, 100, n).astype(np.int64),
        "photons": rng.uniform(100, 900, n).astype(np.float32),
        "xy_err_nm": rng.uniform(5, 25, n).astype(np.float32),
    }, {"units": "nm"})


@pytest.mark.parametrize("region", [
    Region.rect(0, 0, 1000, 1000),
    Region.line((0, 0), (1000, 1000), 200),
    Region("polygon", [(0, 0), (1000, 0), (500, 900)]),
])
def test_every_region_gets_the_dark_line_under_its_outline(app, region):
    from smappy.gui.render_view import RenderView
    view = RenderView(Session(table()))
    view._show_roi(region)
    assert view.roi_halo is not None
    assert view.roi_halo.path().elementCount() > 2
    assert view.roi_halo.zValue() < view.roi_item.zValue()

    view.roi_item.sigRegionChanged.emit(view.roi_item)   # dragged: it follows
    assert view.roi_halo.path().elementCount() > 2

    view._show_roi(None)
    assert view.roi_halo is None


def test_the_roi_header_never_widens_the_control_window(app):
    from smappy.gui.roi_tab import ROIHeader
    from smappy.gui.widgets import CONTROL_WIDTH
    header = ROIHeader(Session(table()))
    header.file_count.setText("1. " + "a_long_experiment_name" * 4
                              + "_sml.hdf5 - 128 ROIs")
    assert header.sizeHint().width() <= CONTROL_WIDTH


def test_the_line_width_is_on_the_toolbar_and_moves_the_line(app):
    """It is the number a line ROI exists for, and it lived behind a
    right-click and a modal dialog."""
    from smappy.gui.render_view import RenderToolBar, RenderView

    session = Session(table())
    view = RenderView(session)
    bar = RenderToolBar(view)
    assert bar.line_width.value() == view.line_width

    session.set_roi(Region.line((0, 0), (500, 500), 100.0))
    bar.line_width.setValue(250.0)
    assert session.roi.width == 250.0
    assert view.line_width == 250.0

    # and a line dragged wider on screen moves the spin box back
    session.set_roi(Region.line((0, 0), (500, 500), 77.0))
    assert bar.line_width.value() == 77.0


def test_the_pixel_size_shows_the_zoom_and_typing_one_zooms_to_it(app):
    """SMAP's pixrec: the zoom is a pixel size one can read and type."""
    from smappy.gui.render_view import DEFAULT_PIXEL, RenderToolBar, RenderView

    session = Session(table())
    view = RenderView(session)
    view.resize(400, 300)
    bar = RenderToolBar(view)
    view.reset()
    assert bar.pixel.value() == pytest.approx(view.pixel_size(), abs=0.01)

    centre = view.view.viewRect().center()
    bar.pixel.setValue(4.0)
    assert view.pixel_size() == pytest.approx(4.0, rel=0.01)
    assert view.view.viewRect().center().x() == pytest.approx(centre.x(), abs=1)

    view.view.scaleBy((2, 2))                  # a zoom moves the number
    assert bar.pixel.value() == pytest.approx(8.0, rel=0.02)

    bar.pixel_button.click()
    assert view.pixel_size() == pytest.approx(DEFAULT_PIXEL, rel=0.01)
    assert bar.pixel.suffix() == " nm"


def blinks(n_emitters=300, on=3):
    """Each emitter on for ``on`` consecutive frames, a few nm apart: grouping
    makes exactly one localization of each."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10_000, (n_emitters, 2))
    start = rng.integers(0, 1000, n_emitters) * 10       # blinks never touch
    rows = np.repeat(np.arange(n_emitters), on)
    frame = start[rows] + np.tile(np.arange(on), n_emitters)
    n = len(rows)
    return Localizations({
        "x_nm": (xy[rows, 0] + rng.normal(0, 3, n)).astype(np.float32),
        "y_nm": (xy[rows, 1] + rng.normal(0, 3, n)).astype(np.float32),
        "frame": frame.astype(np.int64),
        "photons": rng.uniform(500, 900, n).astype(np.float32),
        "xy_err_nm": rng.uniform(5, 10, n).astype(np.float32),
    }, {"units": "nm"})


def test_a_grouped_layer_counts_blinks_inside_the_roi_as_it_does_outside(app):
    """The count beside the picture is of what is drawn.  It used to count the
    grouped table without an ROI and the ungrouped one with one, so a region
    round nearly everything held three times the layer's whole count."""
    from smappy.gui.render_tab import FilterWidget
    from smappy.gui.render_view import RenderToolBar, RenderView

    session = Session(blinks())
    session.show_grouped(0, True)
    grouped, _ = session.table(0)
    assert len(grouped) == 300 and len(session.locs) == 900
    view = RenderView(session)
    bar = RenderToolBar(view)
    widget = FilterWidget(session)
    widget.bind(session.layers[0])

    session.set_roi(Region.rect(0, 0, 10_000, 5_000))
    inside = int(((grouped["y_nm"] <= 5_000) & (grouped["y_nm"] >= 0)).sum())
    assert len(session.shown_selection(0)) == inside
    assert f"layer 1: {inside} in" in bar.counts.text()
    widget._update_count()
    assert widget.count.text().endswith(f", {inside} in ROI")
    # what the plugins get is unchanged: the ungrouped table, three per blink
    assert len(session.selection(0)) == pytest.approx(3 * inside, abs=6)

    session.show_grouped(0, False)
    assert len(session.shown_selection(0)) == len(session.selection(0))


def test_layers_side_by_side_get_a_panel_each_and_the_overlay_is_their_sum(app, tmp_path):
    """SMAP's split and comp: a picture per layer, and the added-up one after."""
    from smappy.gui.render_view import RenderView

    session = Session(table())
    session.add_layer()
    view = RenderView(session)
    view.resize(800, 400)
    view.reset()
    assert view.panels == ["overlay"] and not view.mirrors

    view.set_layout_mode("side")
    assert view.panels == [0, 1] and len(view.mirrors) == 1
    view.set_layout_mode("side+overlay")
    assert view.panels == [0, 1, "overlay"] and len(view.mirrors) == 2
    assert [m.label.textItem.toPlainText() for m in view.mirrors] == [
        session.layers[1].name, "overlay"]

    fov = view.current_fov()
    first, second, overlay = view.panel_images(fov)
    summed, _ = view.composite(fov)
    assert np.allclose(overlay, summed)
    assert np.allclose(overlay, np.clip(first + second, 0, 1))

    # the outline of the ROI is on every panel, which pan with the first
    session.set_roi(Region.rect(1000, 1000, 3000, 3000))
    assert all(m.outline[1].isVisible() for m in view.mirrors)
    view.view.translateBy(x=500)
    assert view.mirrors[0].view.viewRect() == view.view.viewRect()

    path = tmp_path / "panels.png"
    view.save_png(path)
    from PIL import Image
    assert Image.open(path).size[0] > 2.5 * fov.nx          # three across

    session.layers[1].visible = False
    session.changed("layer")
    assert view.panels == ["overlay"] and not view.mirrors  # one layer: one panel

    with pytest.raises(ValueError):
        view.set_layout_mode("tiled")


def test_the_panels_go_whichever_way_shows_the_data_larger(app):
    from smappy.gui.render_view import RenderView

    def arranged(stretch_x):
        locs = table()
        locs.columns["x_nm"] = locs.columns["x_nm"] * stretch_x
        session = Session(locs)
        session.add_layer()
        view = RenderView(session)
        view.resize(800, 600)
        view.set_layout_mode("side")
        return view.stacked

    assert arranged(4.0)                 # 40 by 10 µm: one above the other
    assert not arranged(0.5)             # 5 by 10 µm: side by side
