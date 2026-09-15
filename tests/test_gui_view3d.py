"""The 3D window: its ROI, its controls window, and what it draws over the image."""
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
        "loc_precision_nm": rng.uniform(8, 15, n).astype(np.float32),
    }, {})


def test_the_window_opens_on_the_roi_and_its_controls_are_their_own_window(app):
    from smappy.gui.view3d import View3DWindow

    session = Session(table())
    session.set_roi(Region.line((1000, 1000), (3000, 2000), 250.0))
    window = View3DWindow(session)
    window.show()
    app.processEvents()

    assert session.slab.size[1] == pytest.approx(250.0)      # without pressing a button
    assert window.controls.isVisible()                        # beside, not docked
    assert window.controls.window() is not window
    assert window.width() > window.controls.width() * 2       # the image gets the room
    window.close()
    assert not window.controls.isVisible()


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


def test_the_controls_window_can_be_closed_and_asked_back(app):
    """It is a window of its own, so closing it must not be a dead end."""
    from smappy.gui.view3d import View3DWindow

    window = View3DWindow(Session(table()))
    window.show()
    app.processEvents()
    assert window.controls.isVisible() and window.controls_action.isChecked()

    window.controls.close()
    assert not window.controls_action.isChecked()
    window.controls_action.trigger()
    assert window.controls.isVisible()

    window.close()                              # closed together
    assert not window.controls.isVisible()
    window.show()                               # and back together
    app.processEvents()
    assert window.controls.isVisible()
    window.close()
