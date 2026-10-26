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
    assert bar.counts.text() == f"{expected:,} shown in the view"


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


def _click_at(view, kind, button, x, y):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    at = QPointF(view.graphics.mapFromScene(view.view.mapViewToScene(QPointF(x, y))))
    buttons = Qt.NoButton if "Release" in kind.name else button
    return view.eventFilter(view.graphics.viewport(),
                            QMouseEvent(kind, at, at, button, buttons, Qt.NoModifier))


def test_a_right_click_moves_that_place_to_the_middle(app):
    from PySide6.QtCore import QEvent, QRectF, Qt
    from smappy.gui.render_view import RenderView
    view = RenderView(Session(uniform()))
    view.resize(400, 400)
    view.show()
    app.processEvents()
    view.view.setRange(QRectF(0, 0, 4000, 4000), padding=0)
    app.processEvents()
    _click_at(view, QEvent.MouseButtonPress, Qt.RightButton, 1000, 3000)
    _click_at(view, QEvent.MouseButtonRelease, Qt.RightButton, 1000, 3000)
    centre = view.view.viewRect().center()
    assert abs(centre.x() - 1000) < 40 and abs(centre.y() - 3000) < 40
    assert abs(view.view.viewRect().width() - 4000) < 100     # the zoom kept
    view.close()


def test_a_right_click_on_the_roi_offers_to_delete_it_and_does_not_move_the_view(app):
    from PySide6.QtCore import QEvent, QRectF, Qt
    from smappy.gui.render_view import RenderView
    from smappy.regions import Region
    session = Session(uniform())
    view = RenderView(session)
    view.resize(400, 400)
    view.show()
    app.processEvents()
    view.view.setRange(QRectF(0, 0, 4000, 4000), padding=0)
    session.set_roi(Region.rect(500, 500, 1500, 1500))
    view._show_roi(session.roi)
    app.processEvents()
    menus = []
    view._roi_menu = lambda where: (menus.append(where), view.clear_roi())
    before = view.view.viewRect().center()
    _click_at(view, QEvent.MouseButtonPress, Qt.RightButton, 1000, 1000)
    assert _click_at(view, QEvent.MouseButtonRelease, Qt.RightButton, 1000, 1000)
    assert len(menus) == 1 and session.roi is None and view.roi_item is None
    assert view.view.viewRect().center() == before
    # off the ROI a right click still moves that place to the middle
    session.set_roi(Region.rect(500, 500, 1500, 1500))
    view._show_roi(session.roi)
    _click_at(view, QEvent.MouseButtonPress, Qt.RightButton, 3000, 3000)
    _click_at(view, QEvent.MouseButtonRelease, Qt.RightButton, 3000, 3000)
    assert len(menus) == 1 and session.roi is not None
    centre = view.view.viewRect().center()
    assert abs(centre.x() - 3000) < 40 and abs(centre.y() - 3000) < 40
    view.close()


def test_clearing_the_roi_while_drawing_one_stops_the_drawing(app):
    """*clear ROI* from the button's menu left the cross waiting for a press."""
    from smappy.gui.render_view import RenderView
    from smappy.regions import Region
    session = Session(uniform())
    view = RenderView(session)
    session.set_roi(Region.rect(0, 0, 100, 100))
    view.start_drawing("rect")
    assert view.drawing == "rect" and session.roi is None
    view.clear_roi()
    assert view.drawing is None
    assert all(view.view.state["mouseEnabled"])


def test_a_circle_is_drawn_from_its_centre_and_edited_as_a_circle(app):
    from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
    from smappy.gui.render_view import RenderView
    session = Session(uniform())
    view = RenderView(session)
    view.resize(400, 400)
    view.show()
    app.processEvents()
    view.view.setRange(QRectF(0, 0, 4000, 4000), padding=0)
    app.processEvents()
    view.start_drawing("circle")
    _click_at(view, QEvent.MouseButtonPress, Qt.LeftButton, 2000, 2000)
    _click_at(view, QEvent.MouseButtonRelease, Qt.LeftButton, 2300, 2400)
    roi = session.roi
    assert roi.kind == "circle" and view.drawing is None
    assert np.allclose(roi.points[0], (2000, 2000), atol=20)
    assert abs(roi.radius - 500) < 20
    # dragged on screen, it stays a circle
    view.roi_item.setPos(QPointF(900, 900), finish=False)
    view.roi_item.setSize((200, 200))                # finishes the change
    roi = session.roi
    assert roi.kind == "circle" and roi.radius == 100
    assert np.allclose(roi.points[0], (1000, 1000))
    view.close()


def test_a_double_click_shows_everything(app):
    from PySide6.QtCore import QEvent, QRectF, Qt
    from smappy.gui.render_view import RenderView
    view = RenderView(Session(uniform()))
    view.resize(400, 400)
    view.show()
    app.processEvents()
    view.reset()
    full = view.view.viewRect()
    view.view.setRange(QRectF(100, 100, 200, 200), padding=0)
    app.processEvents()
    assert _click_at(view, QEvent.MouseButtonDblClick, Qt.LeftButton, 200, 200)
    shown = view.view.viewRect()
    assert abs(shown.width() - full.width()) < 1e-6 * full.width() + 1
    view.close()

def test_the_render_toolbar_fits_its_window_with_the_assistant_on_it(app):
    """Eleven items had made it wider than the window, and the assistant's
    button fell into the overflow arrow, where nobody saw it."""
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    render.resize(900, 700)
    bar = render.toolbar
    assert bar.sizeHint().width() <= 860
    widgets = [bar.widgetForAction(a) for a in bar.actions()]
    assert widgets.index(control.assistant_button) == widgets.index(bar.more_button) - 1
    # the line width is there while the ROI kind is a line
    assert bar.line_width_action.isVisible()
    bar._set_kind("rect", "rectangle")
    assert not bar.line_width_action.isVisible()
    bar._set_kind("line", "line")
    assert bar.line_width_action.isVisible()
    # the count is under the picture, and says nothing about an empty table
    assert bar.counts.parent() is render.statusBar() and bar.counts.text() == ""
    more = [a.text() for a in bar.more_button.menu().actions() if a.text()]
    assert more == ["Save image...", "PNG as displayed...", "colour bar", "Mouse and keys..."]
    render.close()
    control.close()


def test_the_layout_choice_is_shown_from_two_layers_on(app):
    from smappy.gui.render_view import RenderToolBar, RenderView
    session = Session(uniform())
    bar = RenderToolBar(RenderView(session))
    assert len(session.layers) == 1 and not bar.layers_action.isVisible()
    session.add_layer()
    assert bar.layers_action.isVisible()


def test_the_assistant_settings_stay_in_the_assistant_menu_on_a_mac(app):
    """Qt moves an item called "Settings..." into a Mac's application menu,
    where it took the place of File > Preferences...."""
    from PySide6.QtGui import QAction
    from smappy.gui.app import ControlWindow, RenderWindow
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    menus = {a.text(): a.menu() for a in control.menuBar().actions() if a.menu()}
    roles = {a.text(): a.menuRole() for m in (menus["Assistant"], menus["File"])
             for a in m.actions()}
    assert roles["Settings..."] == QAction.NoRole
    assert roles["Preferences..."] == QAction.PreferencesRole
    render.close()
    control.close()


def test_a_right_click_offers_no_menu_that_hides_the_toolbar(app):
    """Qt's menu of toolbars had one item, the toolbar, and unticking it left
    the window with no way to bring it back."""
    from smappy.gui.app import RenderWindow
    from smappy.gui.view3d import View3DWindow
    session = Session(uniform())
    for window in (RenderWindow(session), View3DWindow(session)):
        assert window.createPopupMenu() is None
        window.close()


def test_help_opens_the_tutorials_in_the_browser(app, monkeypatch):
    from PySide6.QtGui import QDesktopServices
    from smappy.gui.app import TUTORIALS_URL, ControlWindow, RenderWindow
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    menus = {a.text(): a.menu() for a in control.menuBar().actions() if a.menu()}
    tutorials = next(a for a in menus["Help"].actions() if a.text() == "Tutorials")
    tutorials.trigger()
    assert opened == [TUTORIALS_URL]
    render.close()
    control.close()
