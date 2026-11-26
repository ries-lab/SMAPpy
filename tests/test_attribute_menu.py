"""Rename and remove an attribute from a right-click in the Render tab:
`gui.attribute_menu`, which runs Edit attributes on the session."""
import dataclasses

import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.session import Session


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def session_of(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return Session(Localizations({
        "x_nm": rng.uniform(0, 1e4, n), "y_nm": rng.uniform(0, 1e4, n),
        "frame": np.arange(n), "photons": rng.uniform(100, 900, n),
        "xy_err_nm": rng.uniform(5, 20, n),
        "frame_count": rng.uniform(0, 1, n)}, {"units": "nm"}))


def actions(menu):
    return {a.text(): a for a in menu.actions() if not a.isSeparator()}


def test_the_menu_offers_rename_and_remove_and_greys_the_columns_found_by_name(app):
    from PySide6.QtWidgets import QMenu, QWidget

    from smappy.gui.attribute_menu import add_actions
    session, parent = session_of(), QWidget()
    menu = QMenu()
    add_actions(menu, session, "photons", parent)
    shown = actions(menu)
    assert set(shown) == {"Rename photons...", "Remove photons"}
    assert all(a.isEnabled() for a in shown.values())
    menu = QMenu()
    add_actions(menu, session, "x_nm", parent)
    assert not any(a.isEnabled() for a in actions(menu).values())
    assert "x_nm is" in actions(menu)["Remove x_nm"].toolTip()
    menu = QMenu()
    add_actions(menu, session, "not_a_column", parent)       # a region table's, say
    assert menu.isEmpty()


def test_a_rename_from_the_menu_is_the_plugins_and_an_undo_takes_it_back(app):
    from PySide6.QtWidgets import QWidget

    from smappy.gui.attribute_menu import remove_attribute, rename_attribute
    session = session_of()
    layer = session.layers[0]
    layer.set_bound("photons", 300.0, None)
    layer.state.settings = dataclasses.replace(layer.state.settings, color_field="photons")
    assert rename_attribute(session, "photons", QWidget(), name="N")
    assert "N" in session.locs and "photons" not in session.locs
    assert layer.ranges["N"] == (300.0, None)
    assert layer.state.settings.color_field == "N"
    assert "Edit attributes" in session.undo_labels()[-1]
    session.undo()
    assert "photons" in session.locs and "N" not in session.locs
    assert remove_attribute(session, "frame_count", QWidget())
    assert "frame_count" not in session.locs


def test_a_name_refused_is_said_and_changes_nothing(app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox, QWidget

    from smappy.gui.attribute_menu import rename_attribute
    said = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: said.append(a[2]))
    session = session_of()
    assert not rename_attribute(session, "photons", QWidget(), name="frame_count")
    assert "already has" in said[0]
    assert "photons" in session.locs


def test_the_render_tabs_lists_open_the_menu_of_the_attribute_they_show(app, monkeypatch):
    from PySide6.QtCore import QPoint

    from smappy.gui import attribute_menu
    from smappy.gui.render_tab import RenderTab
    asked = []
    monkeypatch.setattr(attribute_menu, "show_menu",
                        lambda session, column, parent, at: asked.append(column))
    session = session_of()
    tab = RenderTab(session)
    tab.filter._select("photons")
    tab.filter.field.customContextMenuRequested.emit(QPoint(1, 1))
    assert asked == ["photons"]
