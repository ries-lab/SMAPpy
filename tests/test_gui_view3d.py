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
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QToolButton
    bar = window.toolbar
    texts = {b.text() for b in bar.findChildren(QToolButton)} | \
        {a.text() for a in bar.findChildren(QAction)}
    assert {"top", "front", "side", "fit", "rock", "options", "\u22ef", "Save image...",
            "PNG as displayed...", "Mouse and keys...", "Help"} <= texts
    # the views are pictures, their names for the tooltips and the tours
    assert all(not bar.presets[name].icon().isNull() for name in ("top", "front", "side"))
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


def _wait(app, seconds):
    import time
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_the_rock_turns_the_view_and_eases_back_to_where_it_began(app):
    from smappy.gui.view3d import RELAX_SECONDS, View3DWindow

    window = View3DWindow(Session(table()))
    window.show()
    app.processEvents()
    window._preset("front")
    motion, proj = window.view.motion, window.view.projection
    start = proj.matrix.copy()
    window.toolbar.motion_action.trigger()
    assert motion.active and window.toolbar.motion_action.isChecked()
    _wait(app, 0.4)
    assert not np.allclose(proj.matrix, start)
    # about z from the front: z stays upright on the screen
    assert proj.matrix[1, 2] == pytest.approx(start[1, 2], abs=1e-9)
    window.toolbar.motion_action.trigger()                   # stop: it eases back
    assert not motion.active and not window.toolbar.motion_action.isChecked()
    _wait(app, RELAX_SECONDS + 0.3)
    assert not motion.timer.isActive()
    np.testing.assert_allclose(proj.matrix, start, atol=1e-9)
    window.close()


def test_a_spin_stops_where_it_is_and_a_view_button_takes_over(app):
    from smappy.gui.view3d import View3DWindow

    window = View3DWindow(Session(table()))
    window.show()
    app.processEvents()
    motion, proj = window.view.motion, window.view.projection
    window.toolbar._set_kind(motion, "spin")
    motion.speed = 90.0
    motion.toggle()
    _wait(app, 0.3)
    motion.toggle()
    assert not motion.timer.isActive()                       # no way back for a spin
    turned = proj.matrix.copy()
    _wait(app, 0.1)
    np.testing.assert_allclose(proj.matrix, turned)
    motion.toggle()
    window.toolbar.presets["top"].click()                    # taken by hand: stops at once
    assert not motion.timer.isActive()
    window.close()


def test_make_a_movie_hands_the_rock_to_the_players_view_path(app):
    from smappy.gui.scan_player import NONE, ScanPlayer
    from smappy.gui.view3d import View3DWindow

    session = Session(table())
    player = ScanPlayer(session)
    session.window_openers["view path"] = player.take_path
    window = View3DWindow(session)
    window.show()
    app.processEvents()
    window._preset("front")
    start = window.view.projection.matrix.copy()
    window.view.motion.toggle()
    _wait(app, 0.2)
    window._movie()
    assert not window.view.motion.timer.isActive()
    np.testing.assert_allclose(window.view.projection.matrix, start, atol=1e-9)
    assert len(player.path.keys) == 5 and not player.path.ease
    np.testing.assert_allclose(player.path.keys[0].view.rotation, start, atol=1e-9)
    assert player.coordinate.currentText() == NONE
    assert player.ends.currentData() == "loop" and player.record_from.currentData() == "3D"
    window.close()
