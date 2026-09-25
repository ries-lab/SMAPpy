"""The axes section of the Render tab, and the scale bars that go with it.

Choosing a pair of columns has to produce a picture straight away -- a frame
number is in the thousands and a photon count in the hundred thousands, and at
scale 1 the first render would be one bright pixel -- so the section fits the
scales and the widths itself.  What it must never do is change the ordinary
picture.
"""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.io.formats import FileInfo                                # noqa: E402
from smappy.locs import Localizations                                 # noqa: E402
from smappy.render import FieldOfView                                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "z_nm": rng.uniform(-300, 300, n).astype(np.float32),
        "frame": rng.integers(0, 20_000, n).astype(np.int64),
        "photons": rng.uniform(200, 4000, n).astype(np.float32),
        "xy_err_nm": rng.uniform(5, 25, n).astype(np.float32),
    }, {"units": "nm"})


def loaded():
    session = Session()
    session.add_file(table(), FileInfo("a.hdf5", "/a.hdf5", "smappy"))
    return session


def choose(tab, axis: str, field: str) -> None:
    tab.axes.fields[axis].setCurrentText(field)


def test_choosing_two_columns_fits_them_onto_the_screen(app):
    from smappy.gui.render_tab import RenderTab
    session = loaded()
    tab = RenderTab(session)
    choose(tab, "x", "frame")
    choose(tab, "y", "photons")

    axes = session.axes()
    assert (axes.x, axes.y) == ("frame", "photons")
    # a round number of units per render unit, and both spans on one screen
    for scale in (axes.x_scale, axes.y_scale):
        assert scale > 0 and f"{scale:g}"[0] in "125"
    (x0, x1), (y0, y1) = session.full_view(0.0)
    assert 0.2 < (x1 - x0) / (y1 - y0) < 5


def test_an_axis_with_no_precision_behind_it_is_binned_rather_than_blurred(app):
    from smappy.gui.render_tab import RenderTab
    session = loaded()
    tab = RenderTab(session)
    choose(tab, "x", "x_nm")
    choose(tab, "y", "photons")

    settings = session.layers[0].state.settings
    assert settings.mode == "precision"          # x keeps the localization precision
    assert settings.sigma_y == 0.0               # and photons are counted
    assert tab.sigma_y.value() == 0.0 or tab.sigma_y.text() == ""


def test_two_columns_of_the_same_quantity_keep_one_scale(app):
    from smappy.gui.render_tab import RenderTab
    session = loaded()
    tab = RenderTab(session)
    choose(tab, "x", "x_nm")
    choose(tab, "y", "z_nm")
    assert tab.axes.same.isChecked()             # x against z is a picture
    axes = session.axes()
    assert axes.x_scale == axes.y_scale


def test_reset_puts_the_ordinary_picture_back(app):
    from smappy.gui.render_tab import RenderTab
    session = loaded()
    tab = RenderTab(session)
    choose(tab, "x", "frame")
    choose(tab, "y", "photons")
    assert not session.axes().is_default
    tab.axes.reset.click()
    assert session.axes().is_default
    assert all(l.state.settings.axes.is_default for l in session.layers)


def test_the_scale_bars_are_in_the_axes_own_units(app):
    from smappy.gui.render_tab import RenderTab
    from smappy.gui.render_view import RenderView
    session = loaded()
    view = RenderView(session)
    view.resize(400, 400)
    tab = RenderTab(session, view)

    view._update_guides(view.current_fov(200, 200))
    assert view.scalebar.isVisible() and not view.axis_bars.items[0][0].isVisible()
    assert view.scalebar.text.toPlainText().endswith(("nm", "µm"))

    choose(tab, "x", "frame")
    choose(tab, "y", "photons")
    view._update_guides(view.current_fov(200, 200))
    assert not view.scalebar.isVisible()
    across, up = (text.toPlainText() for _, text in view.axis_bars.items)
    assert across.endswith("frame") and up.endswith("photons")
    assert float(across.split()[0]) > 0 and float(up.split()[0]) > 0
    # and a plot hangs the other way up from a picture of a place
    assert not view.view.yInverted()


def test_the_3d_tripod_says_how_long_each_arm_is(app):
    from smappy.gui.render_tab import RenderTab
    from smappy.gui.view3d import View3D
    session = loaded()
    session.slab_from_roi()
    tab = RenderTab(session)
    view = View3D(session)
    view.resize(400, 400)

    choose(tab, "x", "frame")
    choose(tab, "y", "photons")
    choose(tab, "z", "x_nm")
    view._fov = FieldOfView.from_range((0, 1000), (0, 1000), 5.0)
    view._draw_guides()
    assert not view.scalebar.isVisible()          # a rotated view mixes the axes
    labels = [label.toPlainText() for _, label in view.axes]
    assert labels[0].startswith("frame") and labels[1].startswith("photons")
    assert labels[2].startswith("x_nm") and labels[2].endswith(("nm", "µm"))


def test_a_gaussian_width_survives_a_trip_through_other_axes(app):
    """Zeroing it is right for photons, and forgetting it is not."""
    from smappy.gui.render_tab import RenderTab
    session = loaded()
    tab = RenderTab(session)
    tab.mode.setCurrentText("gauss")
    tab.sigma.setValue(25.0)

    choose(tab, "x", "frame")
    choose(tab, "y", "photons")
    assert session.layers[0].state.settings.sigma == 0.0

    tab.axes.reset.click()
    assert session.layers[0].state.settings.sigma == 25.0
    assert tab.sigma.value() == 25.0
