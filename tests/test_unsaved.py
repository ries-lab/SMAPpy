"""A changed table: a `*` in the titles, and, as an option, a question on quitting."""
import numpy as np
import pytest

from smappy.io.formats import FileInfo
from smappy.io.hdf5 import save_localizations
from smappy.locs import Localizations
from smappy.plugins import Plugin, Result
from smappy.session import Session


def table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({"x_nm": rng.uniform(0, 1000, n),
                          "y_nm": rng.uniform(0, 1000, n),
                          "frame": np.arange(n, dtype=np.int64),
                          "photons": np.full(n, 100.0),
                          "xy_err_nm": np.full(n, 10.0)}, {})


class Brighter(Plugin):
    """Changes the table."""
    path = "Test/Brighter"
    grouping = "ungrouped"

    def run(self, ctx, settings=None):
        columns = dict(ctx.locs.columns)
        columns["photons"] = ctx.locs["photons"] * 2
        return Result(locs=Localizations(columns, dict(ctx.locs.metadata)))


class Count(Plugin):
    """Only measures."""
    path = "Test/Count"

    def run(self, ctx, settings=None):
        return Result(text=str(len(ctx.locs)))


def opened(tmp_path, name="cells.hdf5"):
    path = tmp_path / name
    save_localizations(path, table())
    session = Session()
    session.load(path)
    return session


def test_the_table_is_unsaved_after_a_change_an_undo_or_a_fit_and_saved_after_save_or_open(
        tmp_path):
    session = opened(tmp_path)
    said = []
    session.on_change(lambda what: said.append(what) if what == "unsaved" else None)
    assert not session.unsaved                       # as opened
    session.run(Count())
    assert not session.unsaved                       # a measurement changes nothing
    session.run(Brighter())
    assert session.unsaved and said == ["unsaved"]
    session.save()
    assert not session.unsaved
    session.undo()
    assert session.unsaved
    session.save(tmp_path / "other.hdf5")
    session.redo()
    assert session.unsaved
    session.load(tmp_path / "cells.hdf5")
    assert not session.unsaved                       # opening another file
    session.add_file(table(seed=1), FileInfo(name="b.h5", path="/b.h5", format="hdf5"),
                     append=True)
    assert session.unsaved                           # two files in one table
    session.load(tmp_path / "cells.hdf5")
    session.begin_live()                             # a fit, while it runs
    assert session.unsaved


class WritesItsFile(Plugin):
    """As a fitter does: the table it hands back is the file it wrote."""
    path = "Test/Writes"
    grouping = "ungrouped"

    def run(self, ctx, settings=None):
        return Result(locs=table(seed=2), data={"path": "/fit_locs.hdf5",
                                                "saved": True})


def test_a_fit_that_wrote_its_own_file_leaves_nothing_unsaved(tmp_path):
    session = opened(tmp_path)
    session.run(Brighter())
    assert session.unsaved
    session.run(WritesItsFile())
    assert not session.unsaved


@pytest.fixture
def windows(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy import config
    from smappy.gui.app import ControlWindow, RenderWindow
    QApplication.instance() or QApplication([])
    session = opened(tmp_path)
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    control.show()
    asked = []
    control._ask_unsaved = lambda: asked.append(True) or control.answer
    control.answer = "cancel"
    before = config.ask_before_quit()
    yield session, control, asked
    config.set_ask_before_quit(before)
    session.set_unsaved(False)
    control.close()
    render.close()


def test_the_main_windows_titles_carry_a_star_while_the_table_is_unsaved(windows):
    session, control, _ = windows
    render = control.render_window
    assert "*" not in control.windowTitle() and "*" not in render.windowTitle()
    session.run(Brighter())
    assert control.windowTitle().startswith("*SMAPpy")
    assert "*SMAPpy" in render.windowTitle()
    session.save()
    assert "*" not in control.windowTitle() and "*" not in render.windowTitle()


def test_closing_asks_nothing_while_the_preference_is_off(windows):
    from smappy import config
    session, control, asked = windows
    config.set_ask_before_quit(False)
    session.run(Brighter())
    assert control.close()
    assert not asked and not control.isVisible()


def test_closing_with_the_preference_on_asks_save_discard_or_cancel(windows, tmp_path):
    from smappy import config
    from smappy.io.hdf5 import load_localizations
    session, control, asked = windows
    config.set_ask_before_quit(True)
    assert control.close() and not asked            # nothing changed: nothing asked
    control.show()

    session.run(Brighter())
    control.answer = "cancel"
    assert not control.close() and control.isVisible() and len(asked) == 1
    control.answer = "save"
    assert control.close() and not session.unsaved
    assert np.allclose(load_localizations(tmp_path / "cells.hdf5")["photons"], 200)
    control.show()
    session.run(Brighter())
    control.answer = "discard"
    assert control.close() and session.unsaved and len(asked) == 3


def test_the_preference_is_a_tick_in_the_preferences_and_off_by_default(windows):
    from smappy import config
    from smappy.gui.preferences import PreferencesDialog
    _, control, _ = windows
    config.set("ask_before_quit_unsaved", None)
    assert config.ask_before_quit() is False
    dialog = PreferencesDialog(control.workspace, control)
    tick = dialog.general.ask_quit
    assert tick.text() == "ask before quitting with unsaved changes"
    assert not tick.isChecked()
    tick.setChecked(True)
    assert config.ask_before_quit() is True
    dialog.close()
