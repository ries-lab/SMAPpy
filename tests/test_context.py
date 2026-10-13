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
    changes_locs = False
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
    assert ctx.region is None and ctx.site_table is None
    ctx.report("nobody is listening")       # must not raise
    ctx.emit("block", None)


def test_locs_alone_selects_everything():
    locs = table()
    ctx = Context(locs=locs)
    assert ctx.locs is locs and len(ctx.selection) == len(locs)


def test_a_session_context_snapshots_the_table_and_the_selection():
    locs = table()
    session = Session(locs)
    session.show_grouped(0, False)        # the localizations themselves
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


def test_per_region_says_a_plugin_hands_back_values():
    assert Plugin.per_region is False and Echo.per_region is False

    class PerRegion(Plugin):
        changes_locs = False
        per_region = True

    assert PerRegion.per_region is True


def test_calling_a_plugin_needs_no_localizations_at_all():
    class Maker(Plugin):
        changes_locs = True
        def run(self, ctx, settings):
            return Result(locs=table(7), text="made some")

    result = Maker()()
    assert len(result.locs) == 7


def test_session_run_applies_and_logs_through_the_context():
    class Halve(Plugin):
        changes_locs = True
        path = "Test/Halve"
        name = "Halve"
        grouping = "ungrouped"            # it hands back a new table

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
        changes_locs = True
        path = "Test/Slow"
        name = "Slow"
        grouping = "ungrouped"            # it hands back a new table

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
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    seen = {"ran": False}

    class Expensive(Plugin):
        changes_locs = True
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

    from smappy.plugins import PreflightChoice
    answer = {"run": False}

    def ask(self, title, question):
        return PreflightChoice(question.run_label) if answer["run"] else None

    monkeypatch.setattr(PluginPanel, "ask_choice", ask)
    panel.run()
    app.processEvents()
    assert seen["ran"] is False
    assert panel.status.text() == "cancelled"
    # what it would have cost is in the log either way, so a 'no' is informed
    assert "this will take 3 hours" in panel.output.toPlainText()

    answer["run"] = True
    assert panel._preflight(None)[0] is True      # and 'yes' lets the run start


def test_a_cost_question_defaults_to_the_cheaper_way_or_else_to_cancel():
    """Return never starts the run that was just said to be expensive: it
    takes the first alternative when there is one, Cancel when not, and
    Escape always cancels -- the same for a plain question and a choice."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins import PreflightChoice, PreflightQuestion

    QApplication.instance() or QApplication([])

    class Costly(Plugin):
        changes_locs = False
        path = "Test/Costly"
        name = "Costly"

        def run(self, ctx, settings):
            return Result()

    panel = PluginPanel(Costly, Session(table()))
    plain = PreflightQuestion("3 hours. Run it?", run_label="Run")
    box, answers = panel.choice_box("t", plain)
    assert box.defaultButton() is box.escapeButton()
    assert box.defaultButton() not in answers          # Cancel: no answer
    assert {a.label for a in answers.values()} == {"Run"}

    cheaper = PreflightQuestion("3 hours.", choices=[
        PreflightChoice("RCC prepass", settings="fast")])
    box, answers = panel.choice_box("t", cheaper)
    assert answers[box.defaultButton()].label == "RCC prepass"
    assert box.escapeButton() not in answers


def test_a_preflight_choice_runs_with_the_settings_it_carries(monkeypatch):
    """A question with alternatives: taking one runs what that one says, and
    the form shows it afterwards -- a run must not do something the panel
    does not admit to."""
    pytest.importorskip("PySide6")
    from dataclasses import dataclass
    from PySide6.QtWidgets import QApplication

    from smappy.plugins import PreflightChoice, PreflightQuestion, param

    app = QApplication.instance() or QApplication([])
    used = {}

    @dataclass
    class Cost:
        radius: float = param(500.0, label="radius", min=1)

    class Expensive(Plugin):
        changes_locs = True
        path = "Test/Choosy"
        name = "Choosy"
        Settings = Cost

        def preflight(self, ctx, settings):
            return PreflightQuestion(
                text="this will take 3 hours",
                choices=[PreflightChoice(label="do it cheaply",
                                         settings=Cost(radius=50.0))])

        def run(self, ctx, settings):
            used["radius"] = settings.radius
            return Result(locs=ctx.locs[:10])

    from smappy.gui.plugin_panel import PluginPanel
    panel = PluginPanel(Expensive, Session(table()))
    monkeypatch.setattr(PluginPanel, "ask_choice",
                        lambda self, title, q: q.choices[0])
    panel.run()
    while panel._thread is not None and panel._thread.isRunning():
        app.processEvents()
    app.processEvents()
    assert used["radius"] == 50.0
    assert panel.form.value().radius == 50.0

    # and cancelling runs nothing at all
    used.clear()
    monkeypatch.setattr(PluginPanel, "ask_choice", lambda self, title, q: None)
    panel.run()
    app.processEvents()
    assert not used and panel.status.text() == "cancelled"


def test_choices_of_which_are_asked_until_none_is_left(monkeypatch):
    """A question with no plain run asks *which*; the choice may raise the
    next one (two steps of a chain), and each is put once."""
    pytest.importorskip("PySide6")
    from dataclasses import dataclass, replace
    from PySide6.QtWidgets import QApplication

    from smappy.plugins import PreflightChoice, PreflightQuestion

    QApplication.instance() or QApplication([])

    @dataclass
    class Two:
        first: str = ""
        second: str = ""

    class Choosy(Plugin):
        changes_locs = False
        path = "Test/Which"
        name = "Which"
        Settings = Two

        def preflight(self, ctx, settings):
            for name in ("first", "second"):
                if not getattr(settings, name):
                    return PreflightQuestion(f"which {name}?", run_label="", choices=[
                        PreflightChoice(label=x, settings=replace(settings, **{name: x}))
                        for x in ("a", "b")])
            return None

        def run(self, ctx, settings):
            return Result()

    from smappy.gui.plugin_panel import PluginPanel
    asked = []

    def choose(self, title, question):
        asked.append(question.text)
        return question.choices[-1]

    monkeypatch.setattr(PluginPanel, "ask_choice", choose)
    panel = PluginPanel(Choosy, Session(table()))
    go, settings = panel._preflight(Two())
    assert go and settings == Two(first="b", second="b")
    assert asked == ["which first?", "which second?"]


def test_a_preflight_that_fails_does_not_block_the_run():
    """An estimate is a courtesy; a table it cannot read must not stop work."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    class Broken(Plugin):
        changes_locs = True
        path = "Test/Broken"
        name = "Broken"

        def preflight(self, ctx, settings):
            raise ValueError("this table is in pixels and has no pixel size")

        def run(self, ctx, settings):
            return Result(locs=ctx.locs[:10])

    from smappy.gui.plugin_panel import PluginPanel
    panel = PluginPanel(Broken, Session(table()))
    assert panel._preflight(None)[0] is True


def test_a_report_stays_in_the_panel_unless_the_plugin_is_a_log():
    """A run reports in the panel's output box, however long its text: a
    window popping up with a summary is one more thing to close.  Only a
    plugin whose text is the result -- the history -- opens one."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    class Talkative(Plugin):
        changes_locs = False
        path = "Test/Talkative"
        name = "Talkative"
        Settings = None

        def run(self, ctx, settings):
            return Result(text="\n".join(f"line {i}" for i in range(40)))

    from smappy.gui.plugin_panel import PluginPanel

    def run(panel):
        panel.run()
        while panel._thread is not None and panel._thread.isRunning():
            app.processEvents()
        app.processEvents()

    panel = PluginPanel(Talkative, Session(table()))
    run(panel)
    assert panel._text_window is None
    assert panel.output.toPlainText().endswith("line 39")
    assert panel.text_button.isEnabled()      # but the button offers it
    panel.show_text()
    assert panel._text_window.view.toPlainText().endswith("line 39")

    class Log(Talkative):
        path = "Test/Log"
        name = "Log"
        text_window = True

    panel = PluginPanel(Log, Session(table()))
    run(panel)
    assert panel._text_window is not None
    assert panel._text_window.view.toPlainText().endswith("line 39")


def test_run_opens_the_figures_and_a_run_nobody_asked_for_does_not():
    """A measurement is read off its figures, so Run shows them; a run started
    for nobody (`start_run(ask=False)`, a fit MicroClaw drives) waits for Plot."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    class Drawn(Plugin):
        changes_locs = False
        path = "Test/Drawn"
        name = "Drawn"
        Settings = None

        def run(self, ctx, settings):
            return Result(text="done", plot=lambda ax: ax.plot([0, 1], [0, 1]))

    from smappy.gui.plugin_panel import PluginPanel

    def finish(panel):
        while panel._thread is not None and panel._thread.isRunning():
            app.processEvents()
        app.processEvents()

    asked = PluginPanel(Drawn, Session(table()))
    asked.run()
    finish(asked)
    assert asked._window is not None and asked._window.isVisible()
    asked._window.close()

    unattended = PluginPanel(Drawn, Session(table()))
    assert unattended.start_run(ask=False)
    finish(unattended)
    assert unattended._window is None
    assert unattended.plot_button.isEnabled()          # Plot still offers them


def test_a_plugin_that_follows_the_roi_without_a_preview_measures_with_run_and_applies_nothing():
    pytest.importorskip("PySide6")
    import time

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    runs = []

    class Aimed(Plugin):
        changes_locs = False
        path = "Test/AimedByRun"
        name = "AimedByRun"
        Settings = None
        follow_roi = True

        def run(self, ctx, settings):
            runs.append(1)
            return Result(text="measured", plot=lambda ax: ax.plot([0, 1], [0, 1]))

    from smappy.gui.plugin_panel import PluginPanel
    session = Session(table())
    applied = []
    session.apply = lambda plugin, result: applied.append(result)
    panel = PluginPanel(Aimed, session)
    assert panel.preview_button is None and panel.follow_roi is not None

    panel.follow_roi.setChecked(True)
    session.set_roi(Region.line((0, 0), (500, 500), 100.0))
    deadline = time.time() + 20
    while not runs and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    while panel._thread is not None and panel._thread.isRunning():
        app.processEvents()
    app.processEvents()
    panel.follow_roi.setChecked(False)

    assert runs and not applied
    assert panel.status.text() == "following ROI"
    assert panel._window is not None


def test_a_warning_stays_on_screen_when_progress_follows_it():
    """It was a progress line, and the next progress line replaced it."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    class Warns(Plugin):
        changes_locs = False
        path = "Test/Warns"
        name = "Warns"
        Settings = None

        def run(self, ctx, settings):
            ctx.report("frame 1")
            ctx.warn("the calibration was taken with EM gain")
            for i in range(2, 5):
                ctx.report(f"frame {i}")
            return Result(text="fitted\n" + "\n".join(
                f"warning: {w}" for w in ctx.warnings))

    from smappy.gui.plugin_panel import PluginPanel
    panel = PluginPanel(Warns, Session(table()))
    panel.run()
    while panel._thread is not None and panel._thread.isRunning():
        app.processEvents()
    app.processEvents()
    lines = panel.output.toPlainText().splitlines()
    assert lines[:3] == ["frame 1", "warning: the calibration was taken with EM gain",
                         "frame 4"]
    assert panel.status.text() == "done, 1 warning"


def test_a_plugin_that_follows_the_roi_refits_while_it_moves_and_says_nothing_about_it():
    """Ten times a second: no line in the log, no window taking the focus.
    The figure redrawing where it already is is the whole output."""
    pytest.importorskip("PySide6")
    import time

    from PySide6.QtWidgets import QApplication

    from smappy.regions import Region

    app = QApplication.instance() or QApplication([])
    runs = []

    class Aimed(Plugin):
        changes_locs = False
        path = "Test/Aimed"
        name = "Aimed"
        Settings = None
        follow_roi = True

        def run(self, ctx, settings):
            return self.preview(ctx, settings)

        def preview(self, ctx, settings):
            ctx.report("measuring")
            runs.append(getattr(ctx.session, "roi", None))
            return Result(text="one\ntwo\nthree\nfour\nfive\nsix\nseven\neight",
                          plot=lambda ax: ax.plot([0, 1], [0, 1]))

    from smappy.gui.plugin_panel import PluginPanel
    session = Session(table())
    panel = PluginPanel(Aimed, session)
    assert panel.follow_roi is not None                 # the plugin asked for it

    panel.follow_roi.setChecked(True)
    session.set_roi(Region.line((0, 0), (500, 500), 100.0))
    deadline = time.time() + 20
    while not runs and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    while panel._thread is not None and panel._thread.isRunning():
        app.processEvents()
    app.processEvents()

    assert runs, "the follow ROI tick never refitted"
    assert panel.status.text() == "following ROI"
    assert panel.output.toPlainText().strip() == ""    # not a word about it
    assert panel._text_window is None                  # even though it is long
    assert panel._window is not None                   # but the figure is there
    assert panel.text_button.isEnabled()               # and the text is offered

    panel.follow_roi.setChecked(False)
    while panel._thread is not None and panel._thread.isRunning():
        app.processEvents()


def test_a_plugin_that_still_says_live_is_told_the_new_name():
    """"live" means a fit still appending; the tick's old name would
    otherwise be an attribute nothing reads, and the tick gone silently."""
    with pytest.raises(TypeError, match="follow_roi"):
        class Old(Plugin):
            changes_locs = False
            live = True


def test_a_plugin_that_does_not_follow_the_roi_gets_no_tick():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    class Slow(Plugin):
        changes_locs = False
        path = "Test/Slow"
        name = "Slow"
        Settings = None

        def run(self, ctx, settings):
            return Result(text="done")

    from smappy.gui.plugin_panel import PluginPanel
    assert PluginPanel(Slow, Session(table())).follow_roi is None


def test_a_thin_selection_is_a_warning_that_progress_cannot_overwrite():
    """It went through `report`, and the next progress line replaced it."""
    heard = []
    ctx = Context(locs=Localizations({"x_nm": np.zeros(10), "y_nm": np.zeros(10)}, {}),
                  stream=lambda event, payload: heard.append((event, payload)))
    ctx.selection.require(100, ctx.warn, "a fit")
    assert ctx.warnings == ["only 10 localizations; a fit wants at least ~100"]
    assert ("warning", ctx.warnings[0]) in heard


def test_an_info_notice_is_drawn_quietly_and_a_warning_loudly():
    """Only what will go wrong is orange and bold: a save's note of what it
    leaves out is grey, so the warnings keep being read."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from smappy.gui.plugin_panel import INFO_STYLE, WARNING_STYLE, PluginPanel
    from smappy.plugins import Notice

    QApplication.instance() or QApplication([])

    class Quiet(Plugin):
        changes_locs = False
        path = "Test/Quiet"
        name = "Quiet"

        def run(self, ctx, settings):
            return Result()

    panel = PluginPanel(Quiet, Session(table()))
    panel._show_notice(Notice("worth knowing", kind="info"))
    assert panel.notice.styleSheet() == INFO_STYLE
    panel._show_notice(Notice("this camera is a guess"))
    assert panel.notice.styleSheet() == WARNING_STYLE


def test_an_optional_bool_is_a_choice_of_auto_on_and_off():
    """Not a tri-state tick box, whose third state nobody reads as auto:
    the words, and auto saying what it resolves to."""
    pytest.importorskip("PySide6")
    from dataclasses import dataclass
    from typing import Optional
    from PySide6.QtWidgets import QApplication, QComboBox

    from smappy.gui.params import SettingsForm
    from smappy.plugins import param

    QApplication.instance() or QApplication([])

    @dataclass
    class Cam:
        em_on: Optional[bool] = param(None, label="EM gain on")

    form = SettingsForm(Cam)
    box = form.fields["em_on"].widget
    assert isinstance(box, QComboBox)
    assert [box.itemText(i) for i in range(box.count())] == ["auto", "on", "off"]
    assert form.value().em_on is None
    form.set_hints({"em_on": True})
    assert box.itemText(0) == "auto (on)" and form.value().em_on is None
    for value in (True, False, None):
        form.set_values({"em_on": value})
        assert form.value().em_on is value
