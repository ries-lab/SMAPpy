"""The plugin contract after `run(ctx, settings)` replaced the five arguments.

A loader has no input localizations, a simulator has no input at all and an
evaluator runs once per ROI; none of them fit `run(locs, selection, settings,
progress, stream)`, and all of them fit a context.
"""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Context, Plugin, Result, Selection
from smappy.regions import Region
from smappy.session import Session


def table(n=100):
    rng = np.random.default_rng(0)
    return Localizations({"x_nm": rng.uniform(0, 1000, n), "y_nm": rng.uniform(0, 1000, n),
                          "frame": np.arange(n), "photons": np.full(n, 500.0)})


class Echo(Plugin):
    """Reports what it was given, so a test can look at the context."""
    Settings = None

    def run(self, ctx, settings):
        ctx.report("working")
        ctx.emit("block", ctx.locs)
        return Result(text=f"{len(ctx.locs)} in, {len(ctx.selection)} selected",
                      data={"ctx": ctx}, settings=settings)


def test_an_empty_context_is_usable():
    ctx = Context()
    assert len(ctx.locs) == 0 and len(ctx.selection) == 0
    assert ctx.session is None and ctx.rois is None
    assert ctx.site is None and ctx.site_table is None
    ctx.report("nobody is listening")       # must not raise
    ctx.emit("block", None)


def test_locs_alone_selects_everything():
    locs = table()
    ctx = Context(locs=locs)
    assert ctx.locs is locs and len(ctx.selection) == len(locs)


def test_a_session_context_snapshots_the_table_and_the_selection():
    locs = table()
    session = Session(locs)
    session.set_roi(Region.rect(0, 0, 500, 500))
    ctx = session.context()
    assert ctx.session is session and ctx.locs is session.locs
    assert ctx.selection.roi is session.roi
    inside = ((locs["x_nm"] <= 500) & (locs["y_nm"] <= 500)).sum()
    assert len(ctx.selection) == inside

    # the point of snapshotting: a live fit rebinding the table afterwards must
    # not change what the worker is already running on
    session.set_locs(table(20))
    assert ctx.locs is not session.locs and len(ctx.locs) == 100


def test_report_and_emit_reach_their_callbacks():
    said, streamed = [], []
    ctx = Context(locs=table(), progress=said.append,
                  stream=lambda e, p: streamed.append((e, p)))
    result = Echo()(ctx=ctx)
    assert said == ["working"] and streamed[0][0] == "block"
    assert result.text == "100 in, 100 selected"


def test_for_site_narrows_without_losing_the_rest():
    locs = table()
    said = []
    ctx = Context(session=None, locs=locs, progress=said.append,
                  site_table=[{"roi_id": 1}])
    roi = Region.rect(0, 0, 500, 500)
    mask = roi.mask(locs["x_nm"], locs["y_nm"])
    per_site = ctx.for_site(roi, selection=Selection(mask, name="site 1"))
    assert per_site.site is roi
    assert per_site.locs is locs                       # the table is shared
    assert len(per_site.selection) == mask.sum()
    assert per_site.site_table == ctx.site_table
    per_site.report("in a site")
    assert said == ["in a site"]                       # the callbacks came along


def test_scope_says_what_a_plugin_is():
    assert Plugin.scope == "locs" and Echo.scope == "locs"

    class PerSite(Plugin):
        scope = "site"

    assert PerSite.scope == "site"


def test_the_old_signature_fails_where_it_is_written_not_far_away():
    with pytest.raises(TypeError, match="plugins now take a Context"):
        class Stale(Plugin):
            def run(self, locs, selection, settings, progress=None, stream=None):
                return Result()


def test_calling_a_plugin_needs_no_localizations_at_all():
    class Maker(Plugin):
        def run(self, ctx, settings):
            return Result(locs=table(7), text="made some")

    result = Maker()()
    assert len(result.locs) == 7


def test_session_run_applies_and_logs_through_the_context():
    class Halve(Plugin):
        path = "Test/Halve"
        name = "Halve"

        def run(self, ctx, settings):
            ctx.report(f"halving {len(ctx.locs)}")
            return Result(locs=ctx.locs[:len(ctx.locs) // 2], text="halved")

    session = Session(table())
    said = []
    result = session.run(Halve(), progress=said.append)
    assert said == ["halving 100"] and len(result.locs) == 50
    assert len(session.locs) == 50 and session.can_undo
    assert session.history[-1]["what"] == "Test/Halve"


# ------------------------------------------------------------------- the GUI

def test_the_panel_builds_the_context_on_the_gui_thread_and_streams_back():
    """Progress used to travel on a worker signal; the context is now built
    before the worker exists, so it travels on the panel's own signals."""
    pytest.importorskip("PySide6")
    import threading
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    seen = {}

    class Slow(Plugin):
        path = "Test/Slow"
        name = "Slow"

        def run(self, ctx, settings):
            seen["ran_on"] = threading.get_ident()
            ctx.report("half way")
            return Result(locs=ctx.locs[:10], text="done here")

    from smappy.gui.plugin_panel import PluginPanel
    session = Session(table())
    panel = PluginPanel(Slow, session)
    seen["built_on"] = threading.get_ident()

    loop = QEventLoop()
    panel._on_done_original = panel._on_done

    def done(result):
        panel._on_done_original(result)
        loop.quit()

    panel._on_done = done
    panel.run()
    QTimer.singleShot(10000, loop.quit)          # never hang the suite
    loop.exec()
    app.processEvents()

    assert seen["ran_on"] != seen["built_on"]    # it really was a worker thread
    assert len(session.locs) == 10               # the result was applied
    assert "half way" in panel.output.toPlainText()
    assert panel.status.text() == "done"


def test_a_plugin_with_no_settings_gets_an_empty_form_not_a_crash():
    """`specs or param_specs(...)` used to treat {} as 'not given' and fall
    back to introspecting None.  A loader with sensible defaults has no
    settings, so an empty form has to work."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.params import SettingsForm
    from smappy.plugins import param_specs

    QApplication.instance() or QApplication([])
    assert param_specs(None) == {} and param_specs(int) == {}
    form = SettingsForm(None, {})
    assert form.fields == {} and form.value() is None


def test_a_declined_preflight_question_does_not_run_the_plugin(monkeypatch):
    """The preflight runs before the worker exists, so 'no' means no work at
    all -- not a thread started and abandoned."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication([])
    seen = {"ran": False}

    class Expensive(Plugin):
        path = "Test/Expensive"
        name = "Expensive"

        def preflight(self, ctx, settings):
            ctx.report("this will take 3 hours")
            return "3 hours. Run it?"

        def run(self, ctx, settings):
            seen["ran"] = True
            return Result(locs=ctx.locs[:10])

    from smappy.gui.plugin_panel import PluginPanel
    panel = PluginPanel(Expensive, Session(table()))

    answer = {"button": QMessageBox.StandardButton.No}
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: answer["button"], raising=False)

    panel.run()
    app.processEvents()
    assert seen["ran"] is False
    assert panel.status.text() == "cancelled"
    # what it would have cost is in the log either way, so a 'no' is informed
    assert "this will take 3 hours" in panel.output.toPlainText()

    answer["button"] = QMessageBox.StandardButton.Yes
    assert panel._preflight(None) is True         # and 'yes' lets the run start


def test_a_preflight_that_fails_does_not_block_the_run():
    """An estimate is a courtesy; a table it cannot read must not stop work."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    class Broken(Plugin):
        path = "Test/Broken"
        name = "Broken"

        def preflight(self, ctx, settings):
            raise ValueError("this table is in pixels and has no pixel size")

        def run(self, ctx, settings):
            return Result(locs=ctx.locs[:10])

    from smappy.gui.plugin_panel import PluginPanel
    panel = PluginPanel(Broken, Session(table()))
    assert panel._preflight(None) is True
