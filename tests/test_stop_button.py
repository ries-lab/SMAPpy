"""Stop: while a run goes, Run is Stop; pressing it sets the run's
`Context.stop`, what the plugin kept is applied, and the status says
"stopped" -- or the session is left as it was when the plugin had nothing
worth keeping (`Stopped`)."""
import threading
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.locs import Localizations                                 # noqa: E402
from smappy.plugins import Plugin, Result, Stopped                    # noqa: E402
from smappy.session import Session                                    # noqa: E402

# set by the looping plugin once it is running, so the test presses Stop
# while it is, rather than before the worker has started
STARTED = threading.Event()


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def a_table(n=1000):
    rng = np.random.default_rng(0)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.arange(n, dtype=np.int64)}, {})


class Loops(Plugin):
    """Takes one more row a round until it is told to stop; keeps those."""
    path = "Test/Stop/Loops"
    name = "Loops"
    changes_locs = True

    def run(self, ctx, settings):
        STARTED.set()
        rounds = 0
        deadline = time.monotonic() + 20          # a broken Stop fails, not hangs
        while True:
            rounds += 1
            time.sleep(0.005)
            if ctx.stopped or time.monotonic() > deadline:
                break
        return Result(locs=ctx.locs[:rounds], text=f"{rounds} rounds",
                      data={"stopped": ctx.stopped})


class GivesUp(Plugin):
    """Has nothing worth keeping once stopped."""
    path = "Test/Stop/GivesUp"
    name = "GivesUp"
    changes_locs = True

    def run(self, ctx, settings):
        if ctx.stopped:
            raise Stopped("gave up after 0 of 9 windows")
        return Result(locs=ctx.locs[:1], text="finished")


class Ignores(Plugin):
    """Never looks at Stop."""
    path = "Test/Stop/Ignores"
    name = "Ignores"
    changes_locs = False

    def run(self, ctx, settings):
        STARTED.set()
        time.sleep(0.3)
        return Result(text="all of it")


def wait_for(app, condition, seconds=10.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    assert condition(), "timed out"


def ended_of(panel):
    """What `ended` said, in order."""
    said = []
    panel.ended.connect(lambda kind, payload: said.append((kind, payload)))
    return said


def test_pressing_stop_ends_the_run_keeps_its_part_and_says_stopped(app):
    from smappy.gui.plugin_panel import PluginPanel
    STARTED.clear()
    session = Session(a_table())
    panel = PluginPanel(Loops, session)
    said = ended_of(panel)
    assert panel.run_button.text() == "Run"
    panel.run_button.click()
    wait_for(app, STARTED.is_set)
    assert panel.run_button.text() == "Stop" and panel.run_button.isEnabled()

    panel.run_button.click()                    # Stop
    assert panel.run_button.text() == "stopping…"
    assert not panel.run_button.isEnabled()
    wait_for(app, lambda: said)
    app.processEvents()

    kind, result = said[0]
    assert kind == "done" and result.data["stopped"]
    assert panel.status.text() == "stopped"
    assert 0 < len(session.locs) < 1000          # the rounds it did, applied
    assert len(session.locs) == int(result.text.split()[0])
    assert panel.run_button.text() == "Run" and panel.run_button.isEnabled()
    panel._thread.wait()


def test_a_run_with_nothing_to_keep_leaves_the_table_and_says_stopped_not_failed(app):
    from smappy.gui.plugin_panel import PluginPanel
    session = Session(a_table())
    panel = PluginPanel(GivesUp, session)
    said = ended_of(panel)
    stop = threading.Event()
    stop.set()                                  # pressed before it looked
    assert panel.start_run(ask=False, stop=stop)
    wait_for(app, lambda: said)
    app.processEvents()
    assert said[0] == ("stopped", "gave up after 0 of 9 windows")
    assert panel.status.text() == "stopped"
    assert "gave up after 0 of 9 windows" in panel.output.toPlainText()
    assert len(session.locs) == 1000            # untouched
    assert not session.can_undo
    assert panel.run_button.text() == "Run" and panel.run_button.isEnabled()
    panel._thread.wait()


def test_a_plugin_that_does_not_check_runs_to_the_end_and_says_so(app):
    from smappy.gui.plugin_panel import PluginPanel
    STARTED.clear()
    session = Session(a_table())
    panel = PluginPanel(Ignores, session)
    said = ended_of(panel)
    panel.run_button.click()
    wait_for(app, STARTED.is_set)
    panel.run_button.click()                    # Stop, which it never reads
    wait_for(app, lambda: said)
    app.processEvents()
    assert said[0][0] == "done" and said[0][1].text == "all of it"
    assert panel.status.text() == "done"
    assert "finished before it could stop" in panel.output.toPlainText()
    assert panel.run_button.text() == "Run"
    panel._thread.wait()


def test_a_preview_is_not_offered_a_stop(app):
    from smappy.gui.plugin_panel import PluginPanel

    class Previews(Loops):
        path = "Test/Stop/Previews"

        def preview(self, ctx, settings):
            STARTED.set()
            time.sleep(0.2)
            return Result(text="looked")

    STARTED.clear()
    panel = PluginPanel(Previews, Session(a_table()))
    said = ended_of(panel)
    panel.preview()
    wait_for(app, STARTED.is_set)
    assert panel.run_button.text() == "Run" and not panel.run_button.isEnabled()
    wait_for(app, lambda: said)
    app.processEvents()
    assert panel.run_button.text() == "Run" and panel.run_button.isEnabled()
    panel._thread.wait()


def test_rcc_stopped_from_the_panel_leaves_the_table_uncorrected(app):
    from test_drift import simulate

    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.drift_rcc import RCCDrift
    locs, _, _ = simulate(n_points=300, n_frames=200, per_frame=40)
    session = Session(locs)
    before = np.asarray(session.locs["x_nm"]).copy()
    panel = PluginPanel(RCCDrift, session)
    said = ended_of(panel)
    stop = threading.Event()
    stop.set()
    assert panel.start_run(ask=False, stop=stop)
    wait_for(app, lambda: said)
    app.processEvents()
    kind, text = said[0]
    assert kind == "stopped" and "RCC stopped after 0 of" in text
    assert panel.status.text() == "stopped"
    assert np.array_equal(np.asarray(session.locs["x_nm"]), before)
    panel._thread.wait()
