"""The File menu: the data and only the data -- open, recent, the saved
session, save, what is open -- with the workspace in View and the microscope
manager in Plugins."""
from __future__ import annotations

import numpy as np
import pytest

from smappy.locs import Localizations


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    from smappy import config
    config.load(reload=True)
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    control = ControlWindow(session, RenderWindow(session))
    yield control
    control.render_window.close()
    control.close()


def menu(window, title):
    return next(a.menu() for a in window.menuBar().actions() if a.text() == title)


def entries(m):
    """The entries' texts, a separator as "--"."""
    return ["--" if a.isSeparator() else a.text() for a in m.actions()]


def saved_table(path):
    from smappy.io.formats import FileInfo
    from smappy.session import Session
    rng = np.random.default_rng(0)
    session = Session()
    session.add_file(Localizations({
        "x_nm": rng.uniform(0, 5000, 200), "y_nm": rng.uniform(0, 5000, 200),
        "frame": np.arange(200), "photons": rng.uniform(100, 1000, 200)}),
        FileInfo(name=path.name, path=str(path), format="smappy", n=200))
    session.save(path)
    return path


def test_file_holds_the_data_in_groups_and_view_the_workspace(window):
    assert entries(menu(window, "File")) == [
        "Open...", "Open recent", "Restore saved session...", "--",
        "Save", "Save as...", "--",
        "Session name...", "File properties...", "Open in a new window", "--",
        "Preferences...", "--", "Quit"]
    assert entries(menu(window, "View"))[-2:] == ["--", "Workspace"]
    window._fill_plugins_menu()
    assert "Microscope manager..." in entries(menu(window, "Plugins"))


def test_recent_files_are_newest_first_once_each_and_only_those_still_there(
        window, tmp_path):
    from smappy import config
    a, b, gone = (tmp_path / n for n in ("a.h5", "b.h5", "gone.h5"))
    for path in (a, b, gone):
        path.write_bytes(b"")
    for path in (a, gone, b, a):
        config.remember_file(path)
    gone.unlink()
    assert config.recent_files() == [a.resolve(), b.resolve()]
    for i in range(config.RECENT_FILES_MAX + 3):
        (tmp_path / f"{i}.h5").write_bytes(b"")
        config.remember_file(tmp_path / f"{i}.h5")
    assert len(config.recent_files()) == config.RECENT_FILES_MAX


def test_an_opened_file_is_in_open_recent_and_opens_from_there(window, tmp_path):
    path = saved_table(tmp_path / "locs.h5")
    window.load_paths([str(path)])
    recent = window.recent_files_menu
    window._fill_recent_files()
    listed = [a for a in recent.actions() if a.toolTip() != a.text()]
    assert [a.toolTip() for a in listed] == [str(path.resolve())]
    opened = []
    window.open_dropped = lambda paths, **kw: opened.append(paths)
    listed[0].trigger()
    assert opened == [[str(path.resolve())]]
    next(a for a in recent.actions() if a.text() == "Clear the list").trigger()
    window._fill_recent_files()
    assert entries(recent) == ["(none yet)"]


def test_restore_saved_session_puts_back_the_open_files_or_opens_another(
        window, tmp_path, monkeypatch):
    here, other = saved_table(tmp_path / "here.h5"), saved_table(tmp_path / "other.h5")
    window.session.path = here
    called = []
    monkeypatch.setattr(window, "restore_open_file", lambda: called.append("open file"))
    monkeypatch.setattr(window, "open_and_restore", lambda p: called.append(p))
    window.restore_session(str(here))
    window.restore_session(str(other))
    assert called == ["open file", str(other)]
