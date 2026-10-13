"""A session's name and colour, in every window's title and in its file."""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.io.formats import FileInfo                                # noqa: E402
from smappy.locs import Localizations                                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    from smappy.gui.identity import uninstall_identity
    app = QApplication.instance() or QApplication([])
    yield app
    uninstall_identity(app)


def table(n=200):
    rng = np.random.default_rng(1)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.arange(n)}, {})


def opened(name="cells.h5"):
    session = Session()
    session.add_file(table(), FileInfo(name=name, path=f"/data/{name}", format="hdf5"))
    return session


def test_the_display_name_is_the_files_until_the_session_is_named():
    session = opened()
    assert session.display_name == "cells.h5"
    session.rename("  drift   corrected ", colour="blue")
    assert session.display_name == "drift corrected"
    session.rename("")
    assert session.display_name == "cells.h5" and session.colour == "blue"
    with pytest.raises(ValueError, match="unknown colour"):
        session.rename("x", colour="teal")


def test_a_figure_window_says_which_session_it_belongs_to_and_follows_a_rename(app):
    from PySide6.QtWidgets import QMainWindow
    from smappy.gui.identity import install_identity
    session = opened()
    install_identity(app, session)
    window = QMainWindow()
    window.setWindowTitle("Statistics")
    assert window.windowTitle() == "Statistics — cells.h5"
    session.rename("NPC", colour="green")
    assert window.windowTitle() == "\U0001F7E2 Statistics — NPC"
    window.setWindowTitle("Statistics: histograms")     # a run retitles it
    assert window.windowTitle() == "\U0001F7E2 Statistics: histograms — NPC"
    window.close()


def test_the_name_and_colour_are_saved_with_the_file_and_belong_to_it(tmp_path):
    session = opened()
    session.rename("NPC", colour="purple")
    path = session.save(tmp_path / "named.h5", gui_state=False)
    again = Session()
    again.load(path)
    assert (again.name, again.colour) == ("NPC", "purple")
    again.add_file(table(), FileInfo(name="other.h5", path="/data/other.h5", format="hdf5"))
    assert (again.name, again.colour) == ("", "")          # a new file, its own


def test_the_control_and_render_windows_carry_the_name_and_a_stripe(app):
    from smappy.gui.app import ControlWindow, RenderWindow
    session = opened()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    session.rename("NPC", colour="red")
    assert control.windowTitle().startswith("\U0001F534 SMAPpy — NPC (workspace:")
    assert "NPC" in render.windowTitle()
    assert "#d62728" in control.statusBar().styleSheet()
    control.close()
    render.close()
