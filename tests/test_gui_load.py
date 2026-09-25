"""Opening a file does not block the window, and says what it is doing.

Reading and linking a real dataset takes minutes -- 31 s of gzip and 96 s of
grouping on a 57 M localization ``_sml.mat`` -- so both run in a `LoadTask` and
the session is only touched when it comes back.
"""
import numpy as np
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from smappy.group import group                                        # noqa: E402
from smappy.io.formats import FileInfo                                # noqa: E402
from smappy.io.hdf5 import save_localizations                         # noqa: E402
from smappy.locs import Localizations                                 # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def table(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 20_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 20_000, n).astype(np.float32),
        "frame": rng.integers(0, 500, n).astype(np.int64),
        "photons": rng.uniform(100, 900, n).astype(np.float32),
        "xy_err_nm": rng.uniform(5, 25, n).astype(np.float32),
    }, {"units": "nm"})


def test_a_ready_made_grouped_table_gives_what_linking_here_would():
    """What the worker hands back has to be what the main thread would build:
    the same rows, the same columns, and this file's `filenumber` on it."""
    locs = table()
    info = FileInfo("t.hdf5", "/t.hdf5", "smappy")
    here = Session()
    here.add_file(locs, info)
    grouped, _ = group(locs, here.group_settings)        # what LoadTask does
    worker = Session()
    worker.add_file(locs, info, grouped=grouped)

    a = here.layers[0].state.sets["grouped"].locs
    b = worker.layers[0].state.sets["grouped"].locs
    assert len(a) == len(b) and set(a.keys()) == set(b.keys())
    for name in a.keys():
        assert np.allclose(np.asarray(a[name]), np.asarray(b[name]), equal_nan=True), name
    assert np.all(np.asarray(b["filenumber"]) == 0)
    assert worker.layers[0].grouped                      # and it is what is drawn


def test_open_returns_at_once_and_reports_its_stages(app, tmp_path):
    from PySide6.QtCore import QTimer
    from smappy.gui.app import ControlWindow, RenderWindow

    path = tmp_path / "probe.hdf5"
    save_localizations(path, table())

    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)

    control.load_paths([str(path)], reset_view=True)

    assert control._task is not None            # it did not load it inline
    assert not any(a.isEnabled() for a in control._load_locked)   # nothing may change it
    assert "loading" in control.statusBar().currentMessage()

    stages = []                                 # every stage, not a sampled few
    control._task.progress.connect(stages.append)

    QTimer.singleShot(20_000, app.quit)         # a floor under a stuck test
    done = QTimer(control)
    done.timeout.connect(lambda: app.quit() if control._task is None
                         and not control._queue else None)
    done.start(10)
    app.exec()

    assert len(session.locs) == 20_000
    # everything is usable again except the undo and redo entries: a file just
    # opened has no history behind it
    history = {"Undo", "Redo", "Undo steps", "Redo steps"}
    assert all(a.isEnabled() for a in control._load_locked if a.text() not in history)
    assert not control.undo_action.isEnabled() and not control.redo_action.isEnabled()
    said = " ".join(stages)
    assert "Grouper: connect" in said and "Grouper: combine" in said
    # and the status bar is back to what it says the rest of the time
    assert "20000 localizations" in control.statusBar().currentMessage()
    control.stop_loading()                      # the quit path, with nothing running


def test_a_long_file_name_does_not_push_the_clock_off_the_status_bar(app, tmp_path):
    """The control window is 380 px wide.  With the name in front, a long one
    filled the bar on its own and the stage and the clock -- the half that
    moves -- were elided away, so a load that was running looked stuck."""
    from smappy.gui.app import ControlWindow, RenderWindow

    session = Session()
    control = ControlWindow(session, RenderWindow(session))
    bar = control.statusBar()
    bar.resize(380, 22)                            # the width it has on screen
    control._loading_stage = "Grouper: connect"

    control._loading_name = "a" * 300 + "_very_long_name_sml.hdf5"
    control._tick()
    message = bar.currentMessage()
    assert message.startswith("Grouper: connect... (")     # the moving half, first
    assert control._loading_name not in message            # the name is what gives way
    assert bar.fontMetrics().horizontalAdvance(message) <= bar.width()

    control._loading_name = "probe.hdf5"           # one that fits is still shown
    control._tick()
    assert "probe.hdf5" in bar.currentMessage()
