"""The 3D window: its ROI, its title bar and options, and what it draws over the image."""
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


def table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 5000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 5000, n).astype(np.float32),
        "z_nm": rng.normal(0, 200, n).astype(np.float32),
        "frame": rng.integers(0, 100, n).astype(np.int64),
        "xy_err_nm": rng.uniform(8, 15, n).astype(np.float32),
    }, {})


def test_the_window_opens_on_the_roi_and_is_the_picture_and_a_title_bar(app):
    from smappy.gui.view3d import View3DWindow

    session = Session(table())
    session.set_roi(Region.line((1000, 1000), (3000, 2000), 250.0))
    window = View3DWindow(session)
    window.show()
    app.processEvents()

    assert session.slab.size[1] == pytest.approx(250.0)      # without pressing a button
    # the options drop from the title bar; nothing else opens beside it
    assert not window.options.isVisible() and not window.keys.isVisible()
    texts = [b.text() for b in window.toolbar.findChildren(type(window.toolbar.keys_button))]
    assert {"Save", "top", "front", "side", "fit", "options", "keys"} <= set(texts)
    window.toolbar.options_button.click()
    assert window.options.isVisible()
    window.close()
    assert not window.options.isVisible()


def test_hidden_and_shown_again_the_view_is_where_it_was_left(app):
    """The 2D window's *3D* is a toggle; switching it off and on must not
    throw away the turn and the zoom."""
    from smappy.gui.view3d import View3DWindow

    window = View3DWindow(Session(table()))
    seen = []
    window.shown.connect(seen.append)
    window.show()
    app.processEvents()
    window.view.projection.rotate_view(30.0, 20.0)
    window.view.projection.zoom *= 0.5
    before = window.view.projection.matrix.copy(), window.view.projection.zoom
    window.close()
    window.show()
    app.processEvents()
    assert np.allclose(window.view.projection.matrix, before[0])
    assert window.view.projection.zoom == pytest.approx(before[1])
    assert seen == [True, False, True]
    window.close()


def test_each_look_shows_only_the_fields_it_reads(app):
    from smappy.gui.view3d import View3DWindow

    window = View3DWindow(Session(table()))
    panel, form = window.panel, window.panel._form

    def shown(widget):
        return form.isRowVisible(widget)

    assert panel.look == "auto" == window.view.projection.engine
    assert shown(panel.opacity) and shown(panel.attenuation)
    assert not shown(panel.point_alpha) and not shown(panel.ssao)

    panel.looks.button(1).setChecked(True)                    # points
    assert window.view.projection.engine == "points"
    assert shown(panel.point_size) and shown(panel.point_alpha)
    assert not shown(panel.opacity) and not shown(panel.ssao)

    panel.looks.button(2).setChecked(True)                    # spheres
    assert window.view.projection.engine == "spheres"
    assert shown(panel.ssao) and shown(panel.point_size) and not shown(panel.point_alpha)


def test_perspective_is_a_tick_and_a_distance_in_microns(app):
    from smappy.gui.view3d import View3DWindow

    session = Session(table())
    session.set_roi(Region.rect(1000, 1000, 3000, 2000))
    window = View3DWindow(session)
    panel, proj = window.panel, window.view.projection

    assert not panel.perspective_on.isChecked() and proj.focal is None   # orthographic
    box_nm = float(np.max(session.slab.size))
    assert panel.perspective.value() == pytest.approx(round(10 * box_nm / 1000, 1))
    assert panel.perspective.suffix().strip() == "µm"

    panel.perspective_on.setChecked(True)
    assert proj.focal == pytest.approx(panel.perspective.value() * 1000)
    panel.perspective.setValue(50.0)
    assert proj.focal == pytest.approx(50_000.0)
    # typed by hand: a new box no longer moves it
    session.set_slab(session.slab, follow_roi=False)
    panel.refresh()
    assert panel.perspective.value() == pytest.approx(50.0)


def test_point_alpha_starts_where_a_point_cloud_needs_it(app):
    """A million sprites land on a few hundred thousand pixels; at 0.5 the
    first few saturate it.  And it is nudged in hundredths, not tenths."""
    from smappy.gui.view3d import View3DWindow
    from smappy.view3d import Projection

    window = View3DWindow(Session(table()))
    assert window.panel.point_alpha.value() == pytest.approx(0.05) == Projection.point_alpha
    assert window.panel.point_alpha.singleStep() == pytest.approx(0.01)


def test_the_scale_bar_and_the_axes_follow_the_view(app):
    from smappy.gui.view3d import View3DWindow

    session = Session(table())
    session.set_roi(Region.rect(1000, 1000, 3000, 2000))
    window = View3DWindow(session)
    view = window.view
    view._fov = view.projection.fov(400, 400)

    view.projection.azimuth = view.projection.elevation = 0.0        # looking down z
    view._draw_guides()
    assert view.scalebar.isVisible()
    assert view.scalebar.text.toPlainText().endswith(("nm", "µm"))
    arms = [np.hypot(*(np.diff(a.getData(), axis=1).ravel())) for a, _ in view.axes]
    assert arms[0] > 0 and arms[1] > 0
    assert arms[2] == pytest.approx(0, abs=1e-9)     # z points at the viewer

    view.projection.elevation = 90.0                 # tip it: now z is across the screen
    view._draw_guides()
    tipped = [np.hypot(*(np.diff(a.getData(), axis=1).ravel())) for a, _ in view.axes]
    assert tipped[2] == pytest.approx(arms[0])
    assert tipped[1] == pytest.approx(0, abs=1e-9)

    window.panel.guides.setChecked(False)
    assert not view.scalebar.isVisible() and not view.axes[0][0].isVisible()


def test_the_3d_button_shows_and_hides_the_window_and_follows_it(app):
    from smappy.gui.app import ControlWindow, RenderWindow

    session = Session(table())
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    button = render.toolbar.view3d_button
    try:
        assert button.isCheckable() and not button.isChecked()
        button.click()
        app.processEvents()
        window = control.view3d_window
        assert window.isVisible() and button.isChecked()
        button.click()
        assert not window.isVisible() and not button.isChecked()
        control.toggle_3d()                       # View > 3D view, Ctrl+3
        assert window.isVisible() and button.isChecked()
        assert control.view3d_window is window    # the same window, the same view
        window.close()                            # closed by its own title bar
        assert not button.isChecked()
    finally:
        if control.view3d_window is not None:
            control.view3d_window.close()
        render.close()
        control.close()
