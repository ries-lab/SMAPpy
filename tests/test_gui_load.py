"""Opening a file does not block the window, and says what it is doing; nor
does merging its blinks.

Reading a real dataset takes a while -- 31 s of gzip on a 57 M localization
``_sml.mat`` -- so it runs in a `LoadTask` and the session is only touched
when it comes back.  The layer is set to merged blinks, and a large table is
merged after it opens, in a worker (`gui.merging`): the window shows the
localizations until the blinks are there.
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
    worker.show_grouped(0, True)

    a = here.tables.grouped().locs
    b = worker.tables.peek_grouped().locs
    assert worker.tables.is_current                      # taken, not linked again
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
    assert "loading" in said
    assert "connect" not in said                # reading does not merge
    assert session.layers[0].grouped            # the layer is set to blinks
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
    control._loading_stage = "reading the file"

    control._loading_name = "a" * 300 + "_very_long_name_sml.hdf5"
    control._tick()
    message = bar.currentMessage()
    assert message.startswith("reading the file... (")     # the moving half, first
    assert control._loading_name not in message            # the name is what gives way
    assert bar.fontMetrics().horizontalAdvance(message) <= bar.width()

    control._loading_name = "probe.hdf5"           # one that fits is still shown
    control._tick()
    assert "probe.hdf5" in bar.currentMessage()


def _run_until(app, done, limit_s=20.0):
    """Process events until ``done()``; not `app.exec`, which a timer left
    running by an earlier test may quit at once."""
    import time
    start = time.monotonic()
    while not done() and time.monotonic() - start < limit_s:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def _large(monkeypatch):
    """`table()` counts as large: merged in a worker, not on first read."""
    from smappy.gui import merging
    monkeypatch.setattr(merging, "IN_PLACE", 1000)


def test_a_large_file_opens_on_localizations_and_merges_in_the_background(
        app, monkeypatch):
    """Opened, the layer is set to blinks and shows the localizations; the
    status bar says how the merge goes, and the layer shows blinks when they
    are there -- what linking in place would give."""
    from smappy.gui.app import ControlWindow, RenderWindow
    _large(monkeypatch)
    session = Session()
    control = ControlWindow(session, RenderWindow(session))
    said = []
    control.statusBar().messageChanged.connect(said.append)

    session.add_file(table(), FileInfo("t.hdf5", "/t.hdf5", "smappy"))
    layer = session.layers[0]
    assert control.merger.busy                                   # started, not done
    assert layer.grouped and not layer.state.shows_grouped       # localizations meanwhile
    assert len(layer.locs) == 20_000
    _run_until(app, lambda: not control.merger.busy)

    assert layer.state.shows_grouped and session.tables.is_current
    assert "n_in_group" in session.locs                          # written back
    assert any(m.startswith("merging blinks") for m in said)
    assert any("blinks from 20,000 localizations" in m for m in said)
    expected, _ = group(Localizations(dict(table().columns), {"units": "nm"}),
                        session.group_settings)
    assert len(layer.locs) == len(expected)


def test_unticked_while_it_merges_the_layer_keeps_localizations_and_the_merge_is_kept(
        app, monkeypatch):
    from smappy.gui.render_tab import RenderTab
    _large(monkeypatch)
    session = Session()
    tab = RenderTab(session)
    session.add_file(table(), FileInfo("t.hdf5", "/t.hdf5", "smappy"))
    assert tab.grouped.isChecked() and tab.merger.busy      # the setting, from the start
    tab.grouped.click()                                     # unticked meanwhile
    _run_until(app, lambda: not tab.merger.busy)
    layer = session.layers[0]
    assert not layer.grouped and layer.locs is session.locs
    assert session.tables.is_current                         # kept for later
    tab.grouped.click()
    assert layer.state.shows_grouped and not tab.merger.busy  # at once


def test_a_merge_the_table_moved_under_is_done_again_for_the_new_one(app, monkeypatch):
    from smappy.gui.render_tab import RenderTab
    _large(monkeypatch)
    session = Session()
    tab = RenderTab(session)
    session.add_file(table(), FileInfo("t.hdf5", "/t.hdf5", "smappy"))
    assert tab.merger.busy
    half = session.locs[np.arange(len(session.locs)) < 10_000]
    session.set_locs(half, label="halve")              # while the worker links
    _run_until(app, lambda: not tab.merger.busy)
    assert session.layers[0].state.shows_grouped and session.tables.is_current
    assert session.layers[0].locs is session.tables.peek_grouped().locs
    blinks = session.tables.peek_grouped().locs
    assert int(np.asarray(blinks["n_in_group"]).sum()) == 10_000   # of the half


def test_new_merge_settings_merge_again_in_a_worker_and_the_old_blinks_stay_meanwhile(
        app, monkeypatch):
    from smappy.group import GroupSettings
    from smappy.gui.dialogs import GroupingDialog
    from smappy.gui.render_tab import RenderTab
    _large(monkeypatch)
    session = Session()
    tab = RenderTab(session)
    session.add_file(table(), FileInfo("t.hdf5", "/t.hdf5", "smappy"))
    _run_until(app, lambda: not tab.merger.busy)
    old = session.layers[0].locs
    dialog = GroupingDialog(session, tab)
    dialog.form.set(GroupSettings(dx=20.0, dt=1))
    dialog._apply()
    assert session.group_settings.dx == 20.0
    assert tab.merger.busy and session.layers[0].locs is old   # not linked here
    _run_until(app, lambda: not tab.merger.busy)
    assert session.tables.is_current and not session.tables.merging
    assert len(session.layers[0].locs) > len(old)              # a tighter box


def test_a_plugin_run_while_it_merges_measures_what_is_shown(app, monkeypatch):
    """Not a second merge on the GUI thread: the localizations, as drawn."""
    from smappy.gui.render_tab import RenderTab
    _large(monkeypatch)
    session = Session()
    tab = RenderTab(session)
    session.add_file(table(), FileInfo("t.hdf5", "/t.hdf5", "smappy"))
    assert tab.merger.busy
    ctx = session.context()
    assert ctx.locs is session.locs and not session.tables.is_current
    _run_until(app, lambda: not tab.merger.busy)
    assert len(session.context().locs) < len(session.locs)
