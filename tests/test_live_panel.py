"""A plugin panel while a fit is still appending: a table change is offered a
new window instead -- before the run when the plugin says it changes the
table, after it when the plugin does not say."""
import numpy as np
import pytest

pytest.importorskip("PySide6")

from smappy.locs import Localizations                                 # noqa: E402
from smappy.plugins import Plugin, Result                             # noqa: E402
from smappy.session import Session                                    # noqa: E402


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def block(frame, n=100):
    rng = np.random.default_rng(frame)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.full(n, frame, np.int64),
                          "photons": rng.uniform(100, 2000, n)}, {})


def live_session():
    session = Session()
    session.begin_live((0, 5000, 0, 5000))
    for frame in range(1, 4):
        session.append(block(frame))
    return session


class Declared(Plugin):
    path = "Test/Declared"
    name = "Declared"
    changes_locs = True

    def run(self, ctx, settings):
        raise AssertionError("must not run during a live fit")


class Undeclared(Plugin):
    path = "Test/Undeclared"
    name = "Undeclared"

    def run(self, ctx, settings):
        return Result(locs=ctx.locs[:10], text="cut")


class Measures(Plugin):
    path = "Test/Measures"
    name = "Measures"
    changes_locs = False

    def run(self, ctx, settings):
        return Result(text=f"{len(ctx.selection.apply(ctx.locs))} seen")


def run_and_wait(panel, app):
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    original = panel._on_done

    def done(result):
        original(result)
        loop.quit()

    panel._on_done = done
    panel.run()
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
    for _ in range(5):
        app.processEvents()


def test_a_declared_table_change_is_offered_a_new_window_before_it_runs(app):
    from smappy.gui.plugin_panel import PluginPanel
    session = live_session()
    asked, opened = [], []
    session.window_openers["new window"] = lambda run=None: opened.append(run) or object()
    panel = PluginPanel(Declared, session)
    panel.ask_new_window = lambda text: asked.append(text) or True
    panel.run()
    assert "still adding localizations" in asked[0]
    assert opened[0][0].path == "Test/Declared"
    assert panel.status.text() == "running in a new window"
    assert session.live and len(session.locs) == 300


def test_cancelling_the_question_runs_nothing(app):
    from smappy.gui.plugin_panel import PluginPanel
    session = live_session()
    opened = []
    session.window_openers["new window"] = lambda run=None: opened.append(run)
    panel = PluginPanel(Declared, session)
    panel.ask_new_window = lambda text: False
    panel.run()
    assert not opened and panel.status.text() == "not run: the fit is still running"


def test_an_undeclared_table_change_is_held_back_and_then_offered(app):
    from smappy.gui.plugin_panel import PluginPanel
    session = live_session()
    asked = []
    panel = PluginPanel(Undeclared, session)
    panel.ask_new_window = lambda text: asked.append(text) or False
    run_and_wait(panel, app)
    assert session.live and len(session.locs) == 300      # not applied
    assert "not applied" in panel.output.toPlainText()
    assert asked and "run again in a new window" in asked[0]


def test_a_measurement_runs_on_the_live_table_as_it_was(app):
    from smappy.gui.plugin_panel import PluginPanel
    session = live_session()
    panel = PluginPanel(Measures, session)
    panel.ask_new_window = lambda text: pytest.fail("a measurement asks nothing")
    run_and_wait(panel, app)
    assert "300 seen" in panel.output.toPlainText()
