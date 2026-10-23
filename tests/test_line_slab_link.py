"""A line ROI and the 3D window's box: turned by keys in the 2D view, and
while the box follows the line, each is an edit of the other."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.regions import Region
from smappy.session import Session
from smappy.view3d import Projection, Slab


def table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 5000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 5000, n).astype(np.float32),
        "z_nm": rng.normal(0, 200, n).astype(np.float32),
        "frame": rng.integers(0, 100, n).astype(np.int64),
    }, {})


def test_a_line_turns_about_its_middle_keeping_its_length_and_width():
    line = Region.line((1000, 1000), (3000, 1000), 200.0)
    turned = line.turned(90.0)
    p0, p1 = turned.ends()
    np.testing.assert_allclose(p0, (2000, 0), atol=1e-9)
    np.testing.assert_allclose(p1, (2000, 2000), atol=1e-9)
    assert turned.width == 200.0
    with pytest.raises(ValueError):
        Region.rect(0, 0, 1, 1).turned(5.0)


def test_the_footprint_of_a_slab_is_the_roi_it_was_made_from():
    line = Region.line((1000, 1000), (3000, 2000), 250.0)
    back = Slab.from_region(line, (-300, 300)).footprint("line")
    np.testing.assert_allclose(back.points, line.points, atol=1e-9)
    rect = Region.rect(100, 200, 900, 700)
    np.testing.assert_allclose(Slab.from_region(rect, (0, 1)).footprint("rect").points,
                               rect.points)
    assert Slab([0, 0, 0], [10, 10, 10], angle=30).footprint("rect") is None
    assert Slab([0, 0, 0], [10, 10, 10]).footprint("polygon") is None


def test_a_box_dragged_while_it_follows_a_line_redraws_the_line():
    session = Session(table())
    session.set_roi(Region.line((1000, 1000), (3000, 1000), 200.0))
    slab = session.slab
    slab.set_axis_range(1, 900.0, 1300.0)                     # one side out by 200
    slab.set_axis_range(0, 1000.0, 4000.0)                    # and the end out by 1000
    slab.set_axis_range(2, -50.0, 80.0)
    session.slab_resized(slab)
    assert session.slab_follows_roi
    p0, p1 = session.roi.ends()
    assert session.roi.width == pytest.approx(400.0)
    # the line's middle moves with the side that moved, half as far
    np.testing.assert_allclose(p0, (1000, 1100), atol=1e-6)
    np.testing.assert_allclose(p1, (4000, 1100), atol=1e-6)
    # the z range dragged is kept through the next edit of the line
    session.roi_edited(Region.line((1000, 1500), (3000, 1500), 200.0))
    assert session.slab.axis_range(2) == pytest.approx((-50.0, 80.0))
    session.slab_from_roi(keep_z=False)
    assert session.slab_z is None
    assert session.slab.axis_range(2) != pytest.approx((-50.0, 80.0))


def test_a_box_dragged_without_following_goes_its_own_way():
    session = Session(table())
    session.set_roi(Region.polygon([(0, 0), (1000, 0), (500, 800)]))
    session.slab_resized(session.slab)
    assert not session.slab_follows_roi
    assert session.roi.kind == "polygon"


def test_the_view_follows_a_slab_moved_and_turned():
    proj = Projection(azimuth=40.0, elevation=90.0, zoom=3.0)
    proj.offset[:] = (55.0, -20.0)
    slab = Slab([100, 200, 0], [1000, 200, 400], angle=10.0)
    proj.follow(slab, turned_by=10.0)
    assert proj.azimuth == pytest.approx(30.0)
    np.testing.assert_allclose(proj.pivot, slab.center)
    np.testing.assert_allclose(proj.offset, 0.0)
    assert proj.zoom == 3.0 and proj.elevation == 90.0
    # seen from the front of the slab before, from the front of it after
    front = Projection()
    front.preset("front", slab_angle=0.0)
    front.follow(slab, turned_by=10.0)
    reference = Projection()
    reference.preset("front", slab_angle=10.0)
    assert front.azimuth == pytest.approx(reference.azimuth)


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _key(view, key, shift=False):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    mods = Qt.ShiftModifier if shift else Qt.NoModifier
    view._on_key(QKeyEvent(QEvent.KeyPress, key, mods))


def test_keys_in_the_2d_view_turn_the_line_as_it_is_seen(app):
    from PySide6.QtCore import Qt
    from smappy.gui.render_view import RenderView
    session = Session(table())
    view = RenderView(session)
    session.set_roi(Region.line((1000, 1000), (3000, 1000), 200.0))

    def angle():
        p0, p1 = session.roi.ends()
        return np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

    assert view.view.yInverted()
    _key(view, Qt.Key_BracketLeft)            # anticlockwise on a y-down screen
    assert angle() == pytest.approx(-5.0)
    _key(view, Qt.Key_Greater, shift=True)    # Shift + . on a US keyboard
    assert angle() == pytest.approx(-4.0)
    _key(view, Qt.Key_Semicolon, shift=True)  # Shift + , on a German one
    assert angle() == pytest.approx(-5.0)
    assert session.roi.width == pytest.approx(200.0)
    session.set_roi(Region.rect(0, 0, 100, 100))
    _key(view, Qt.Key_Comma)                  # nothing to turn
    assert session.roi.kind == "rect"


def test_the_3d_view_stays_on_a_line_moved_and_turned_in_2d(app):
    from smappy.gui.view3d import View3DWindow
    session = Session(table())
    session.set_roi(Region.line((1000, 1000), (3000, 1000), 250.0))
    window = View3DWindow(session)
    window.show()
    app.processEvents()
    window.view.preset("front")
    azimuth = window.view.projection.azimuth
    session.roi_edited(Region.line((2000, 3000), (4000, 3000), 250.0))     # dragged
    np.testing.assert_allclose(window.view.projection.pivot[:2], (3000, 3000))
    np.testing.assert_allclose(window.view.projection.offset, 0.0)
    session.set_roi(session.roi.turned(20.0))
    assert window.view.projection.azimuth == pytest.approx((azimuth - 20.0) % 360)
    # the slab moved in depth here, as Ctrl+wheel does, keeps a pan
    window.view.projection.offset[:] = (40.0, -30.0)
    shown = window.view.projection.screen_centre()
    session.slab.center = session.slab.center + [0.0, 0.0, 50.0]
    session.changed("slab")
    np.testing.assert_allclose(window.view.projection.screen_centre()[:2], shown[:2],
                               atol=1e-6)
    window.close()
