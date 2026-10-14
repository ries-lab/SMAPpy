"""The picture's toolbar and the picture: what they show after a zoom."""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.locs import Localizations                                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def uniform(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({"x_nm": rng.uniform(0, 10000, n),
                          "y_nm": rng.uniform(0, 10000, n),
                          "frame": np.arange(n), "xy_err_nm": np.full(n, 10.0)})


def test_the_count_in_view_is_what_the_layer_draws_inside_the_box():
    session = Session(uniform())
    locs = session.layers[0].locs
    x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
    shown = session.shown_selection(0).mask
    inside = shown & (x >= 1000) & (x < 3000) & (y >= 2000) & (y < 7000)
    assert session.count_in_view(0, 1000, 3000, 2000, 7000) == int(inside.sum())
    assert session.count_in_view(0, -1, 1e5, -1, 1e5) == int(shown.sum())


def test_the_toolbar_counts_follow_the_zoom(app):
    from PySide6.QtCore import QRectF
    from smappy.gui.render_view import RenderToolBar, RenderView
    session = Session(uniform())
    view = RenderView(session)
    bar = RenderToolBar(view)
    view.view.setRange(QRectF(0, 0, 2500, 2500), padding=0)
    bar._update_counts("view")             # what the timer does once a zoom settles
    rect = view.view.viewRect()
    expected = session.count_in_view(0, rect.left(), rect.right(), rect.top(), rect.bottom())
    assert expected < len(session.layers[0].filter) / 4
    assert bar.counts.text() == f"{expected:,}"


def test_the_session_colour_keeps_the_buttons_framed(app):
    """The colour used to be the whole style sheet, which undid the buttons."""
    from smappy.gui.render_view import RenderToolBar, RenderView
    bar = RenderToolBar(RenderView(Session(uniform(100))))
    bar.set_border("#ff0000")
    sheet = bar.styleSheet()
    assert "border-bottom: 4px solid #ff0000" in sheet
    assert "QToolButton {" in sheet and "border: 1px solid" in sheet
    bar.set_border("")
    assert "QToolButton {" in bar.styleSheet() and "#ff0000" not in bar.styleSheet()


def test_a_double_click_moves_that_place_to_the_middle(app):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from smappy.gui.render_view import RenderView
    view = RenderView(Session(uniform()))
    view.resize(400, 400)
    view.show()
    app.processEvents()
    from PySide6.QtCore import QRectF
    view.view.setRange(QRectF(0, 0, 4000, 4000), padding=0)
    app.processEvents()
    target = view.view.mapViewToScene(QPointF(1000, 3000))
    at = view.graphics.mapFromScene(target)
    event = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(at), QPointF(at),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    assert view.eventFilter(view.graphics.viewport(), event)
    centre = view.view.viewRect().center()
    assert abs(centre.x() - 1000) < 40 and abs(centre.y() - 3000) < 40
    view.close()
