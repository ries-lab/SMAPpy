"""A plugin panel while a fit is still appending: a table change is offered a
new window instead, before the run, since every plugin says whether it
changes the table."""
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


class Mistaken(Plugin):
    """Says it measures, and hands back a table."""
    changes_locs = False
    path = "Test/Mistaken"
    name = "Mistaken"

    def run(self, ctx, settings):
        return Result(locs=ctx.locs[:10], text="cut")


class Measures(Plugin):
    path = "Test/Measures"
    name = "Measures"
    changes_locs = False

    def run(self, ctx, settings):
        return Result(text=f"{len(ctx.selection.apply(ctx.locs))} seen")


def run_and_wait(panel, app):
    """Run and wait for the panel to finish -- on its `ended` signal, so that
    `_on_done` stays the slot it is, on the GUI thread (a plain function put
    in its place is called on the worker's)."""
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    panel.ended.connect(lambda *_: loop.quit())
    panel.run()
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
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


def test_a_plugin_that_does_not_say_whether_it_changes_the_table_is_refused():
    with pytest.raises(TypeError, match="changes_locs"):
        class Undeclared(Plugin):
            def run(self, ctx, settings):
                return Result(locs=ctx.locs[:10])


def test_a_plugin_that_wrongly_says_it_measures_is_not_applied_during_a_fit_and_asks_nothing(app):
    from smappy.gui.plugin_panel import PluginPanel
    session = live_session()
    panel = PluginPanel(Mistaken, session)
    panel.ask_new_window = lambda text: pytest.fail("asked after the run")
    run_and_wait(panel, app)
    app.processEvents()
    assert session.live and len(session.locs) == 300      # not applied
    assert "not applied" in panel.output.toPlainText()
    assert panel.status.text() == "not applied: the fit is still running"


def test_a_measurement_runs_on_the_live_table_as_it_was(app):
    from smappy.gui.plugin_panel import PluginPanel
    session = live_session()
    panel = PluginPanel(Measures, session)
    panel.ask_new_window = lambda text: pytest.fail("a measurement asks nothing")
    run_and_wait(panel, app)
    assert "300 seen" in panel.output.toPlainText()


def test_an_expression_runs_during_a_live_fit_and_follows_it(app, tmp_path, monkeypatch):
    """A recipe asks nothing: it goes into the live table and every block."""
    import dataclasses
    from smappy import plugins
    from smappy.gui.plugin_panel import PluginPanel
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    session = live_session()
    cls = plugins.get("Analysis/Process/Math Parser")
    panel = PluginPanel(cls, session)
    panel.form.set(dataclasses.replace(cls.Settings(), field="bright",
                                       expression="photons > 1000"))
    panel.ask_new_window = lambda text: pytest.fail("a recipe asks nothing")
    run_and_wait(panel, app)
    session.append(block(9))
    assert session.live and "bright" in session.locs and len(session.locs) == 400


class Streams(Plugin):
    """A fit in miniature: starts a live table, streams blocks, hands back
    the finished one."""
    path = "Test/Streams"
    name = "Streams"
    changes_locs = True
    grouping = "ungrouped"

    def run(self, ctx, settings):
        ctx.emit("start", {"extent": (0, 5000, 0, 5000)})
        blocks = [block(frame) for frame in range(1, 4)]
        for b in blocks:
            ctx.emit("block", b)
        from smappy.locs import concat
        return Result(locs=concat(blocks), text="fitted")


def test_a_fit_whose_table_cannot_be_put_in_place_fails_and_ends_the_live_table(
        app, monkeypatch):
    """Raised out of the panel's slot, the error left the status at
    'running...' beside a Run button and the session live for good, so every
    table change after it was offered a new window."""
    from smappy.gui.plugin_panel import PluginPanel
    session = Session()
    panel = PluginPanel(Streams, session)
    real = Session.apply

    def broken(self, plugin, result, **kw):
        if kw.get("finishes_live"):
            raise RuntimeError("the table would not go in")
        return real(self, plugin, result, **kw)

    monkeypatch.setattr(Session, "apply", broken)
    run_and_wait(panel, app)
    assert panel.status.text() == "the table would not go in"
    assert panel.run_button.text() == "Run"
    assert not session.live and len(session.locs) == 300   # what came, kept


def test_a_fit_started_with_the_render_tab_on_layer_two_finishes_its_table(app):
    """A live fit started with one new layer; the Render tab, still on
    layer 2, raised inside `begin_live`, and the fit's own table was then
    refused as if the fit were still running.  Now the layers stay."""
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.gui.render_tab import RenderTab
    session = Session(block(1))
    session.add_layer()
    tab = RenderTab(session)
    tab.strip.select(1)
    panel = PluginPanel(Streams, session)
    run_and_wait(panel, app)
    assert panel.status.text() == "done"
    assert not session.live and len(session.locs) == 300
    assert len(tab.strip.buttons) == len(session.layers) == 2
    assert tab.layer is session.layers[1]
