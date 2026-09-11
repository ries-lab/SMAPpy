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
        "loc_precision_nm": rng.uniform(5, 25, n).astype(np.float32),
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
    assert all(a.isEnabled() for a in control._load_locked if a.text() != "Undo")
    said = " ".join(stages)
    assert "Grouper: connect" in said and "Grouper: combine" in said
    # and the status bar is back to what it says the rest of the time
    assert "20000 localizations" in control.statusBar().currentMessage()
    control.stop_loading()                      # the quit path, with nothing running
