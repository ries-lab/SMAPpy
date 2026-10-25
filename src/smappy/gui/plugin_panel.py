"""A plugin as a section: its settings form, a Run button, and the result.

The plugin runs in a worker thread so the window keeps responding; the result
is applied to the session on the GUI thread.  While a run is going, Run is
Stop: it sets the run's `Context.stop`, and what the plugin hands back then --
the frames fitted, the regions measured -- is applied as any result is, with
the status "stopped".  A plugin that never checks runs to the end.
"""
from __future__ import annotations

import dataclasses
import threading
import traceback
from typing import Optional, Type

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QSizePolicy, QSpinBox, QToolButton, QVBoxLayout,
                               QWidget)

from ..plugins import reads_locs, Plugin, Result, Stopped
from ..session import LiveTableError, Session
from .errors import ERROR_STYLE, details_window, message_of
from .params import SettingsForm
from .widgets import ColumnScroll, note_style

# questions of *which* asked in a row before a run: one per step of a chain
# that has to choose, and a bound so a preflight that never settles cannot
# keep the dialog coming back
MAX_CHOICES = 8

# the status of a run that warned: what was said is in the output box
WARNING_STYLE = "color: #c06000; font-weight: bold;"
RESULT_TIP = ("show what the run made: its figures (the drift curves, say), "
              "or, when it drew none, its text in a window of its own")



class _Worker(QObject):
    done = Signal(object)
    failed = Signal(str, str)       # the message, without its class; the traceback
    stopped = Signal(str)           # `Stopped`: nothing kept, nothing broken

    def __init__(self, plugin, context, settings, job="run", **kwargs):
        super().__init__()
        self.args = (plugin, context, settings)
        self.job = job
        self.kwargs = kwargs

    def run(self) -> None:
        from ..diagnostics import running
        plugin, context, settings = self.args
        try:
            n = len(context.selection)
        except Exception:                   # a File plugin has no table yet
            n = None
        try:
            # a follow-ROI step is a run nobody keeps (`PluginPanel._follow_step`)
            work = getattr(plugin, "run" if self.job == "follow" else self.job)
            # into the diagnostic log: what ran, with what, and the traceback
            # if it failed -- the first thing a bug report is asked for
            with running(plugin, self.job, settings, n):
                result = work(context, settings, **self.kwargs)
            if self.job == "run":
                from ..plugins import refuse_grouped_table
                refuse_grouped_table(context, result, plugin)
            self.done.emit(result)
        except Stopped as e:
            self.stopped.emit(str(e))
        except Exception as error:
            self.failed.emit(message_of(error), traceback.format_exc())


class PluginPanel(QWidget):
    # The context is built before the worker exists, so progress cannot go
    # through a worker signal any more.  These belong to the panel, which lives
    # on the GUI thread: emitting them from the worker delivers queued, which
    # is what keeps `_on_progress` and `_on_stream` off that thread.
    progressed = Signal(str)
    streamed = Signal(str, object)
    # ("done", Result), ("failed", traceback) or ("stopped", message) -- a
    # run that raised `Stopped` -- once a run or preview is over and the
    # panel has dealt with it: what a caller that started the run itself
    # (`gui.live_session`) waits for.  A stopped run that kept a part of its
    # work is "done", its result saying `Result.stopped`.
    ended = Signal(str, object)

    def __init__(self, plugin_cls: Type[Plugin], session: Session, parent=None):
        super().__init__(parent)
        self.plugin = plugin_cls()
        # what the run in flight runs: the plugin, or the one a button inside
        # a part names (`Plugin.part_runs`), which is applied under its own name
        self._runner = self.plugin
        self.session = session
        self.result: Optional[Result] = None
        # the window this plugin's figures live in, made on the first Plot
        # and kept: running again redraws where the user already put it,
        # instead of stacking another copy of it on the screen
        self._window = None
        self._text_window = None
        self._thread: Optional[QThread] = None
        self._progress_lines = 0
        self._job = "run"
        # the running run's `Context.stop`, while Run is Stop; None otherwise
        self._stop: Optional[threading.Event] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.about: Optional[QLabel] = None      # the summary line, if any
        if self.plugin.description:
            about = self.about = QLabel(self.plugin.description)
            about.setWordWrap(True)
            about.setStyleSheet(note_style())
            layout.addWidget(about)
        self.form = SettingsForm(plugin_cls.Settings, plugin_cls.specs())
        layout.addWidget(self.form)
        # a button at the foot of a part, running another plugin on that
        # part's settings: a fitter's *after the fit* on the open table
        self.part_buttons = {}
        for part, label, path in plugin_cls.part_runs:
            target = self.form.fields.get(part)
            if not isinstance(target, SettingsForm):
                continue
            button = QPushButton(label)
            button.setToolTip("run the steps above on the open localizations, "
                              "without fitting again: one undo step")
            button.clicked.connect(lambda _=False, p=part, r=path: self.run_part(p, r))
            target.layout().addWidget(button, 0, Qt.AlignLeft)
            self.part_buttons[part] = button
        # a field that reads the session -- the layers editor offers its
        # columns and copies its layers -- is told which one
        for leaf in self.form.leaves():
            if hasattr(leaf, "take_from_session"):
                leaf.session = session
                leaf.refresh()
        # what the plugin says about the settings as they stand
        # (`Plugin.describe`): a camera the database does not know, with the
        # button that adds it.  Hidden while there is nothing to say.
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet(WARNING_STYLE)
        self.notice_button = QPushButton("")
        self.notice_button.clicked.connect(self._on_notice)
        notice_row = QHBoxLayout()
        notice_row.addWidget(self.notice, 1)
        notice_row.addWidget(self.notice_button, 0, Qt.AlignTop)
        layout.addLayout(notice_row)
        self.notice.hide()
        self.notice_button.hide()
        self._notice_window: Optional[str] = None

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run")
        buttons.addWidget(self.run_button)
        # a preview sits next to Run because it answers the question Run asks:
        # is this set up right?  The work, drawn, and nothing saved.
        self.preview_button: Optional[QPushButton] = None
        self.preview_frame: Optional[QSpinBox] = None
        if plugin_cls.has_preview():
            self.preview_button = QPushButton("Preview")
            self.preview_button.setToolTip(
                plugin_cls.preview_help or
                "do the work and draw it, without saving anything: "
                "the session is not touched.")
            self.preview_button.clicked.connect(self.preview)
            buttons.addWidget(self.preview_button)
            # only a preview *of a frame* asks which one; a plugin that
            # previews the whole selection has nothing to ask
            if plugin_cls.preview_wants_frame():
                self.preview_frame = QSpinBox()
                self.preview_frame.setRange(0, 10**9)
                self.preview_frame.setToolTip("which frame to preview")
                self.preview_frame.setPrefix("frame ")
                self.preview_frame.setMaximumWidth(110)
                buttons.addWidget(self.preview_frame)
        # One button for what the run made: its figures, or -- for a run that
        # drew nothing -- what it reported, in a window that can hold more
        # than the four lines of the output box.  Plot and Text were two
        # buttons for one question, and one of them was always greyed.
        self.result_button = QPushButton("Show result")
        self.result_button.setToolTip(RESULT_TIP)
        self.result_button.setEnabled(False)
        # a measurement one aims with rather than one asked for afterwards:
        # the plugin says it is cheap enough (`Plugin.follow_roi`) and the tick
        # re-measures while the ROI is dragged
        self.follow_roi: Optional[QCheckBox] = None
        if plugin_cls.follow_roi:
            self.follow_roi = QCheckBox("follow ROI")
            self.follow_roi.setToolTip("redraw while the ROI is moved, without "
                                       "touching the session")
            self.follow_roi.toggled.connect(self._on_follow)
        # the status is where a failure is said, in red (`_on_failed`): it
        # wraps rather than widening the row, since a message may be long
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        # ... and the traceback behind it, one click away, only after one
        self.details_button = QToolButton()
        self.details_button.setText("details\u2026")
        self.details_button.setToolTip("the whole error, and a bug report")
        self.details_button.clicked.connect(self.show_details)
        self.details_button.hide()
        self._failure: Optional[tuple] = None
        self._details_window = None
        buttons.addWidget(self.result_button)
        if self.follow_roi is not None:
            buttons.addWidget(self.follow_roi)
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.details_button)
        # Five controls at the style's minimum button width (80 px in Fusion)
        # are wider than the control column, and a fitter has all five: the
        # section then scrolled sideways and took its title bar -- the ? and
        # the detach arrow -- out of view.  Each may shrink to its text.
        for button in (self.run_button, self.preview_button, self.result_button):
            if button is not None:
                button.setMinimumWidth(
                    button.fontMetrics().horizontalAdvance(button.text()) + 20)
        if self.preview_frame is not None:
            self.preview_frame.setMinimumWidth(72)
        layout.addLayout(buttons)
        self.output = QPlainTextEdit(readOnly=True, maximumBlockCount=500)
        self.output.setFixedHeight(90)
        layout.addWidget(self.output)
        # F1 anywhere in the panel opens the plugin's page, wherever the panel
        # sits -- a tab, a detached section or a window of its own
        shortcut = QShortcut(QKeySequence.HelpContents, self, self.show_help)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)

        if self.plugin.window:
            # a launcher (`Plugin.window`): nothing runs here, so nothing to
            # plot or read; the window it opens is the result
            self.run_button.setText("Open")
            self.run_button.setToolTip("open the window, set up as above")
            self.result_button.hide()
            self.output.hide()
            self.run_button.clicked.connect(self.open_window)
            self.window_buttons = []
            for label, window in self.plugin.window_buttons:
                extra = QPushButton(label)
                extra.setMinimumWidth(
                    extra.fontMetrics().horizontalAdvance(label) + 20)
                extra.clicked.connect(lambda _=False, w=window: self.open_window(w))
                buttons.insertWidget(buttons.indexOf(self.run_button) + 1
                                     + len(self.window_buttons), extra)
                self.window_buttons.append(extra)
            if not dataclasses.fields(plugin_cls.Settings):
                # nothing to set: the buttons are the panel, one row, and the
                # summary is their tooltip rather than lines above them
                self.form.hide()
                if self.about is not None:
                    self.about.hide()
                    for button in [self.run_button] + self.window_buttons:
                        button.setToolTip(self.plugin.description)
        else:
            self.run_button.clicked.connect(self._run_or_stop)
        self.result_button.clicked.connect(self.show_result)
        self._following = False
        # this panel's run started the session's live table (the "start" it
        # streamed): its result is that table's finished form, the one
        # table change `Session.apply` takes while a fit is appending
        self._owns_live = False
        self._show_figures = True
        self._warnings = 0
        self._follow_timer = QTimer(self, singleShot=True, interval=120)
        self._follow_timer.timeout.connect(self._follow_step)
        # the token of a run waiting for the blinks to be merged (`gui.merging`)
        self._merge_wait = None
        self.form.field_changed.connect(self._react)
        self.progressed.connect(self._on_progress)
        self.streamed.connect(self._on_stream)
        self._notice_settings = None
        self._describe()
        self._enable_parts()
        session.on_change(self._on_session)
        self._take_saved()

    def release(self) -> None:
        """Stop listening to the session, for a panel about to be replaced
        (a chain rebuilt after its steps were edited)."""
        self.session.off_change(self._on_session)

    def _on_session(self, what: str) -> None:
        if what in ("locs", "rois", "transformations") \
                or (what == "roi" and self.plugin.follow_roi):
            # the lists of evaluations; a save says what it would leave out;
            # a measurement of the ROI says whether there is one to measure;
            # Register layers offers to apply what it measured
            # (only those: a fitter's description opens its file)
            self._describe()
        if what == "locs":
            self._enable_parts()
        if what in ("locs", "results"):
            self._take_saved()
        elif what in ("roi", "roi-edited") and self.follow_roi is not None \
                and self.follow_roi.isChecked():
            # not straight away: a drag is tens of events a second and a
            # refit is tens of milliseconds, so the timer coalesces them and
            # the last position always wins
            self._follow_timer.start()

    def _on_follow(self, on: bool) -> None:
        if on:
            self._follow_timer.start()

    def _follow_step(self) -> None:
        """One follow-ROI refit, skipped while another is still running.

        Skipped rather than queued: the point is to follow the ROI, and the
        next drag event will ask again in a few tens of milliseconds.
        """
        if self.follow_roi is None or not self.follow_roi.isChecked():
            return
        if self._thread is not None and self._thread.isRunning():
            self._follow_timer.start()          # try again once it is free
            return
        self._following = True
        # its preview where it has one; otherwise the run itself, which is
        # then looked at and not applied -- a measurement that only draws
        # needs no Preview button to be aimed
        if self.plugin.has_preview():
            self.preview()
        else:
            self._start("follow", "measuring...")

    def _take_saved(self) -> None:
        """Offer the figure of a run that is over: this file carries its result.

        A drift correction subtracts a curve and the corrected table no longer
        says what the curve was, so the file keeps it (`Plugin.keep`) and the
        panel picks it up here -- on opening the file and after a run, which
        is why the file's own result is never allowed over one made in this
        session.
        """
        if self._thread is not None and self._thread.isRunning():
            return
        saved = self.session.results.get(self.plugin.path)
        if not saved or (self.result is not None and self._job != "preview"):
            return
        result = self.session.restore_result(self.plugin)
        if result is None or not result.figures():
            return
        self.result = result
        self.result_button.setEnabled(True)
        when = saved.get("time", "")
        self.result_button.setToolTip(
            "show the figure of the run kept with this file"
            + (f" ({when})" if when else ""))
        self.status.setText("ran before" + (f", {when}" if when else ""))

    def _react(self, path: str) -> None:
        """Let the plugin answer an edit, e.g. fill the camera from the file."""
        try:
            updates = self.plugin.react(path, self.form.value())
        except (ValueError, TypeError):
            return
        if updates:
            self.form.set_values(updates)
        self._describe()

    def refresh_derived(self) -> None:
        """Read again what the plugin works out from outside the form -- the
        lists, the hints, the notice -- after something it reads changed: a
        camera added to the database, say."""
        self._describe()

    def _describe(self) -> None:
        """Show what the plugin says about the settings as they stand
        (`Plugin.describe`): hints in the fields on *auto*, the fields it does
        not read greyed out or hidden, the lists the session decides, and the notice.

        Best effort on purpose: working this out may mean opening a file that
        is missing, half-written or not an image at all, and none of that is
        worth an error in a field the user has not finished typing.  A part
        that is empty leaves what the form shows as it is.
        """
        from ..plugins import FormState
        try:
            state = self.plugin.describe(
                self.session.context(grouping=self.plugin.grouping), self.form.value())
        except Exception:
            state = FormState()
        # the key parameters first: their copies take the hints too
        self.form.set_keys(state.keys)
        if state.hints:
            self.form.set_hints(state.hints)
        if state.choices:
            self.form.set_choices(state.choices)
        if state.active:
            self.form.set_active(state.active)
        if state.titles:
            self.form.set_titles(state.titles)
        if state.shown:
            self.form.set_shown(state.shown)
        self._show_notice(state.notice)

    def _shapes_drawn(self, field: str, shapes) -> None:
        """Shapes drawn on a preview's figure (`ShapeEdit`), into the form:
        the next run fits with them, and the workspace keeps them."""
        self.form.set_values({field: [dict(s) for s in shapes]})
        self._describe()

    def _show_notice(self, notice) -> None:
        """The plugin's line about the settings (`Notice`), with its button
        when the GUI has the window it names."""
        self.notice.setVisible(notice is not None)
        if notice is None:
            self.notice_button.hide()
            return
        self.notice.setText(notice.text)
        self.notice.setStyleSheet(note_style() if notice.kind == "info"
                                  else WARNING_STYLE)
        self._notice_window = notice.window
        self._notice_settings = notice.settings
        offered = bool(notice.action and notice.window
                       and notice.window in self.session.window_openers)
        self.notice_button.setText(notice.action)
        self.notice_button.setVisible(offered)

    def _on_notice(self) -> None:
        opener = self.session.window_openers.get(self._notice_window)
        if opener is None:
            return
        settings = self._notice_settings       # a chain's step's own
        if settings is None:
            try:
                settings = self.form.value()
            except ValueError as e:
                self.status.setText(str(e))
                return
        opener(settings, **self._reply_to(opener, settings is self._notice_settings))

    def _reply_to(self, opener, foreign: bool = False) -> dict:
        """What a window that answers (an opener with ``replies`` set: the
        bead selector) is opened with besides the settings: this panel's
        plugin, and where to send settings changed in it -- into this form,
        which a run, the workspace and a chain then take.  Not for settings
        that are not the form's own (a chain's step's)."""
        if not getattr(opener, "replies", False) or foreign:
            return {}
        return {"plugin": self.plugin, "reply": self.take_settings}

    def take_settings(self, settings) -> None:
        """Settings sent back by a window this panel opened (`_reply_to`)."""
        self.form.set(settings)
        self._describe()

    def _on_stream(self, event: str, payload) -> None:
        if event == "warning":
            if self._following:
                return            # a follow-ROI step says nothing (`_on_progress`)
            # a line of its own that progress does not overwrite: the next
            # progress line goes below it
            self._warnings += 1
            self.output.appendPlainText(f"warning: {payload}")
            self._progress_lines = 0
            self.status.setText(f"running... ({self._warnings} warning"
                                f"{'s' if self._warnings > 1 else ''})")
            self.status.setStyleSheet(WARNING_STYLE)
        elif event == "start":
            # owned first: a window listening to the session can raise inside
            # `begin_live`, and a live table nobody owns refuses the fit's own
            # result as if another fit were running
            self._owns_live = True
            self.session.begin_live(payload.get("extent"), payload.get("path"))
        elif event == "block":
            self.session.append(payload)

    def preview(self) -> None:
        """The work, shown and thrown away."""
        extra = ({} if self.preview_frame is None
                 else {"frame": self.preview_frame.value()})
        self._start("preview", "previewing...", **extra)

    def run(self) -> None:
        self._start("run", "running...")

    def run_part(self, part: str, path: str) -> bool:
        """Run the plugin at ``path`` with the form's part ``part`` as its
        settings, on the session, as its own panel would."""
        from .. import plugins
        try:
            runner = plugins.get(path)()
        except KeyError:
            self.status.setText(f"no plugin {path} in this program")
            return False
        return self._start("run", "running...", runner=runner, part=part)

    def _enable_parts(self) -> None:
        """A part's button needs localizations to run on, and no run going."""
        has = self.session.locs is not None and len(self.session.locs) > 0
        for button in self.part_buttons.values():
            button.setEnabled(bool(has) and self._thread_idle())

    def _thread_idle(self) -> bool:
        return self._stop is None and self._merge_wait is None

    def _run_or_stop(self) -> None:
        if self._merge_wait is not None:
            self._cancel_wait()
        elif self._stop is not None:
            self.stop()
        else:
            self.run()

    def stop(self) -> None:
        """Ask the running run to end early, keeping what it has done.

        Only asks: the plugin looks at `Context.stop` between steps of its
        own, so the button says "stopping…" until the run has ended.
        """
        if self._stop is None or self._stop.is_set():
            return
        self._stop.set()
        self.run_button.setText("stopping…")
        self.run_button.setEnabled(False)
        self.status.setText("stopping...")

    def open_window(self, window: Optional[str] = None) -> None:
        """Hand the settings to the window the plugin names (`Plugin.window`,
        or one of its `window_buttons`)."""
        window = window or self.plugin.window
        try:
            settings = self.form.value()
        except ValueError as e:
            self.status.setText(str(e))
            return
        opener = self.session.window_openers.get(window)
        if opener is None:
            self.status.setText(f"no {window} window in this program")
            return
        self.status.setText("")
        opener(settings, **self._reply_to(opener))

    def start_run(self, ask: bool = True, **context) -> bool:
        """Run with the form's settings; whether the run started.

        ``context`` goes into the run's `Context` -- `stop` and
        `writer_finished` for a run something else controls -- and ``ask=False``
        skips the preflight question, for a run nobody is there to answer.
        Nor are its figures opened: they wait for Plot.
        """
        return self._start("run", "running...", context=context, ask=ask)

    def _start(self, job: str, message: str, context=None, ask: bool = True,
               runner: Optional[Plugin] = None, part: Optional[str] = None,
               **kwargs) -> bool:
        try:
            settings = self.form.value()
        except ValueError as e:
            self.status.setText(f"bad value: {e}")
            return False
        if part is not None:
            settings = getattr(settings, part)
        self._runner = runner = runner or self.plugin
        if (job == "run" and self.session.live and not self._owns_live
                and runner.changes_locs_for(settings)
                and not runner.column_recipe_for(settings)):
            # (a recipe -- an expression, a colour assignment -- follows the
            # live table instead: `Session._follow`)
            # known before the run: say so now, not after minutes of work
            name = runner.name or runner.path
            self._offer_new_window(
                settings, f"The fit is still adding localizations, and {name} "
                          "changes them, so it cannot run on this session now.\n\n"
                          "It can run in a new window instead, on the "
                          "localizations so far; the fit carries on here.")
            return False
        if job == "run" and ask:
            go, chosen = self._preflight(settings)
            if not go:
                return False
            if chosen is not settings:
                # a preflight choice may hand back other settings than the
                # form's; showing them is what keeps the run from being a
                # surprise afterwards
                settings = chosen
                if part is not None:
                    self.form.fields[part].set(settings)
                else:
                    self.form.set(settings)
        context = dict(context or {})
        stop = None
        if job == "run":
            # what Stop sets: the caller's own when it controls the run
            # (`gui.live_session`), so either can end it
            stop = context.get("stop") or threading.Event()
            context["stop"] = stop
        merger = self._merger()
        if merger is not None and reads_locs(runner):
            if merger.pending():
                # not on localizations standing in for blinks being merged:
                # the run waits for them and starts by itself
                if job == "follow":
                    if self._merge_wait is None:   # measured once they are there
                        self._merge_wait = merger.after(self._follow_after_merge)
                    return False
                self._wait_for_merge(merger, job, message, settings, context,
                                     stop, ask, kwargs)
                return True
            merger.prepare()
        return self._launch(job, message, settings, context, stop, ask, kwargs)

    def _merger(self):
        """The session's background merge (`gui.merging`), if this program
        has one; a panel without a window around it has none."""
        from .merging import _MERGERS
        from shiboken6 import isValid
        found = _MERGERS.get(self.session)
        return found if found is not None and isValid(found) else None

    def _follow_after_merge(self) -> None:
        self._merge_wait = None
        self._follow_timer.start()

    def _wait_for_merge(self, merger, job, message, settings, context, stop,
                        ask, kwargs) -> None:
        """Hold the run until the blinks are merged, saying so, and start
        it then; Stop (the run button) gives up waiting."""
        def go():
            self._merge_wait = None
            merger.status.disconnect(self._on_merge_status)
            self._release_buttons()
            self._launch(job, message, settings, context, stop, ask, kwargs)

        def failed(text):
            self._merge_wait = None
            merger.status.disconnect(self._on_merge_status)
            self._release_buttons()
            self.status.setText(f"could not merge blinks ({text}): not run")
            self.status.setStyleSheet(WARNING_STYLE)

        for button in self._buttons():
            button.setEnabled(False)
        self._stop = stop or threading.Event()
        self.run_button.setText("Stop")
        self.run_button.setToolTip("do not run: stop waiting for the merge")
        self.run_button.setEnabled(True)
        self.status.setStyleSheet("")
        merger.status.connect(self._on_merge_status)
        self._merge_wait = -1                       # waiting, token to come
        self._on_merge_status(merger.text or "merging blinks...")
        self._merge_wait = merger.after(go, failed)

    def _on_merge_status(self, text: str) -> None:
        if text and self._merge_wait is not None:
            self.status.setText(f"{text} -- the run starts when it is done")

    def _cancel_wait(self) -> None:
        merger = self._merger()
        if merger is not None:
            merger.cancel(self._merge_wait)
            try:
                merger.status.disconnect(self._on_merge_status)
            except (RuntimeError, TypeError):
                pass
        self._merge_wait = None
        self._release_buttons()
        self.status.setText("not run")

    def _launch(self, job, message, settings, context, stop, ask, kwargs) -> bool:
        # built here, on the GUI thread: the context reads the table and the
        # selection now, so the worker cannot race a live fit rebinding them
        context = self.session.context(progress=self.progressed.emit,
                                       stream=self.streamed.emit,
                                       grouping=self._runner.grouping,
                                       **context)
        self._job = job
        # a run somebody asked for shows its figures when it is done; one
        # that nobody is there to answer (`start_run(ask=False)`) does not
        self._show_figures = ask
        for button in self._buttons():
            button.setEnabled(False)
        if stop is not None:
            # a preview and a follow-ROI step are short and thrown away: only a run
            # is worth stopping
            self._stop = stop
            self.run_button.setText("Stop")
            self.run_button.setToolTip("end the run early, keeping what it has "
                                       "done so far (a plugin that does not "
                                       "check runs to the end)")
            self.run_button.setEnabled(True)
        self.status.setText(message)
        self.status.setStyleSheet("")
        self.status.setToolTip("")
        self.details_button.hide()
        self._warnings = 0
        self._progress_lines = 0
        self._thread = QThread()
        # Qt's default thread stack (512 kB on macOS) is too small for HDF5 and
        # the fitter's own threads' bookkeeping: a bus error, not an exception
        self._thread.setStackSize(32 * 1024 * 1024)
        self._worker = _Worker(self._runner, context, settings, job, **kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.stopped.connect(self._on_stopped)
        for sig in (self._worker.done, self._worker.failed, self._worker.stopped):
            sig.connect(self._thread.quit)
        self._thread.start()
        return True

    def _preflight(self, settings):
        """Ask the plugin whether this run wants agreeing to first.

        Returns ``(go, settings)``: a plugin may offer a cheaper way of doing
        the same thing (`PreflightQuestion`), and taking it runs with the
        settings that choice carries rather than the ones in the form.

        On the GUI thread and before the worker exists, so the dialog is an
        ordinary modal one.  What the plugin reports here is written straight
        into the log rather than through `progressed`: it is a standing record
        of what was agreed to, and the progress lines that follow overwrite
        each other on the line below it.
        """
        from ..plugins import PreflightQuestion

        # a question that asks *which* (no plain run) is answered by a choice,
        # and the settings it carries may raise the next one -- a chain with
        # two such steps -- so those are asked until none is left
        for _ in range(MAX_CHOICES):
            go, chosen, question = self._ask(settings)
            if not go:
                return go, chosen
            if chosen is settings or not (isinstance(question, PreflightQuestion)
                                          and not question.run_label):
                return go, chosen
            settings = chosen
        return True, settings

    def _ask(self, settings):
        """One preflight question: ``(go, settings, question)``."""
        from ..plugins import PreflightQuestion

        try:
            context = self.session.context(progress=self.output.appendPlainText,
                                           grouping=self._runner.grouping)
            question = self._runner.preflight(context, settings)
        except Exception as e:
            # An estimate is a courtesy.  One that cannot be made -- a table in
            # pixels with no pixel size, a plugin that does not expect this
            # data -- must not be what stops the run the user asked for.  It
            # says so rather than vanishing, so that a broken preflight is
            # visible as one.
            self.output.appendPlainText(f"(no estimate: {type(e).__name__}: {e})")
            return True, settings, None
        if not question:
            return True, settings, None
        title = f"{self._runner.name}: before running"
        if not isinstance(question, PreflightQuestion):
            # the plain yes-or-no is the same dialog with no alternatives, so
            # that every question before a run has the same default (Cancel)
            question = PreflightQuestion(text=str(question), run_label="Run")
        chosen = self.ask_choice(title, question)
        if chosen is None:
            return (*self._not_run(), question)
        if chosen.window:
            # not a way to run but a way to put things right first: the
            # window opens and the run waits for the next Run
            opener = self.session.window_openers.get(chosen.window)
            if opener is not None:
                opener(chosen.settings if chosen.settings is not None else settings)
            self.output.appendPlainText(f"not run: {chosen.label.lower()}")
            self.status.setText("not run")
            return False, None, question
        if chosen.settings is not None:
            self.output.appendPlainText(f"chose: {chosen.label}")
            return True, chosen.settings, question
        return True, settings, question

    def _not_run(self):
        self.output.appendPlainText("not run")
        self.status.setText("cancelled")
        return False, None

    def _offer_new_window(self, settings, text: str) -> bool:
        """Ask whether to run in a new instance; start it if so (`handoff`)."""
        if not self.ask_new_window(text):
            self.status.setText("not run: the fit is still running")
            return False
        opener = self.session.window_openers.get("new window")
        if opener is None:
            self.status.setText("no new window in this program")
            return False
        if opener(run=(self._runner, settings)) is None:
            return False
        self.status.setText("running in a new window")
        return True

    def ask_new_window(self, text: str) -> bool:
        """Run in a new window, or cancel: a dialog, which a test replaces."""
        box = QMessageBox(self)
        box.setWindowTitle(f"{self.plugin.name}: the fit is still running")
        box.setText(text)
        new = box.addButton("Run in a new window", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(new)
        box.exec()
        return box.clickedButton() is new

    def ask_choice(self, title: str, question):
        """Put a question with alternatives; the choice, or None to cancel.

        Its own method because it is the one part of the preflight that opens
        a modal dialog: a test drives the decision by replacing this.
        """
        box, answers = self.choice_box(title, question)
        box.exec()
        return answers.get(box.clickedButton())

    def choice_box(self, title: str, question):
        """The dialog `ask_choice` shows, and what each of its buttons answers.

        The default -- what Return presses -- is the first alternative that
        runs when there is one (the cheaper way: COMET's RCC prepass), and
        otherwise Cancel, never the run that was asked about: the question is
        put because the run costs more than one would guess, and someone who
        has just been told so rarely wants it after all.  An alternative that
        opens a window (`PreflightChoice.window`) is never the default.
        Escape cancels.
        """
        from ..plugins import PreflightChoice

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(question.text)
        answers = {}
        # each alternative is a button of its own rather than a second dialog:
        # the choice is between ways of doing the same thing, and they should
        # be readable side by side
        first = None
        for choice in question.choices:
            button = box.addButton(choice.label, QMessageBox.ButtonRole.AcceptRole)
            if choice.help:
                button.setToolTip(choice.help)
            answers[button] = choice
            if not choice.window:
                first = first or button
        if question.run_label:
            run = box.addButton(question.run_label, QMessageBox.ButtonRole.AcceptRole)
            answers[run] = PreflightChoice(label=question.run_label)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        # explicit, and on every button: a button's own auto-default would
        # otherwise take Return on some styles
        for button in box.buttons():
            button.setAutoDefault(False)
        box.setDefaultButton(first or cancel)
        box.setEscapeButton(cancel)
        detail = "\n\n".join(f"{c.label}\n{c.help}" for c in question.choices if c.help)
        if detail:
            box.setDetailedText(detail)
        return box, answers

    def _on_progress(self, text: str) -> None:
        """Progress replaces the last line while it is a progress line, so a
        long fit does not scroll its own summary away.

        A follow-ROI step reports nothing: it happens ten times a second and would
        fill the box with the same line while saying less than the figure it
        is redrawing.
        """
        if self._following:
            return
        cursor = self.output.textCursor()
        if self._progress_lines:
            # the whole paragraph: `LineUnderCursor` is one *visual* line,
            # and a progress line long enough to wrap left its first part
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock,
                                QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(text)
        else:
            self.output.appendPlainText(text)
        self._progress_lines += 1
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())

    def _buttons(self):
        return [b for b in (self.run_button, self.preview_button) if b is not None] \
            + list(self.part_buttons.values())

    def _release_buttons(self) -> bool:
        """Run is Run again and the buttons take clicks; whether Stop had
        been pressed."""
        stop, self._stop = self._stop, None
        if stop is not None:                 # a run, not a preview
            self.run_button.setText("Run")
            self.run_button.setToolTip("")
        for button in self._buttons():
            button.setEnabled(True)
        self._enable_parts()
        return stop is not None and stop.is_set()

    def _on_done(self, result: Result) -> None:
        following, self._following = self._following, False
        asked = self._release_buttons()
        stopped = result.stopped
        self.result = result
        # a preview is looked at, never applied: it exists so that the session
        # is not changed before the settings are right; nor is a follow-ROI step
        refused = None
        if self._job not in ("preview", "follow"):
            owns, self._owns_live = self._owns_live, False
            try:
                self.session.apply(self._runner, result, finishes_live=owns)
            except LiveTableError as e:
                # a plugin whose `changes_locs` is wrong, or a fit that began
                # while this ran: what was declared is asked about before the
                # run (`_start`), so this is only said, not offered again
                refused = str(e)
            except Exception as e:
                # Putting the result in place failed.  Raised out of this slot,
                # it left the status at "running..." beside a Run button and,
                # for a fit, the session live for good: every table change
                # after it was refused as if the fit were still adding.
                import traceback
                if owns:
                    self._owns_live = True          # `_on_failed` ends it
                self._on_failed(str(e) or type(e).__name__, traceback.format_exc())
                return
        else:
            self.session.show_preview(self.plugin, result)
            # a figure whose panel takes shapes writes them into this form
            for figure in result.figures():
                if figure.edit is not None:
                    figure.edit.on_change = \
                        lambda shapes, field=figure.edit.field: self._shapes_drawn(field, shapes)
        # A follow-ROI step happens ten times a second while the ROI is dragged, so
        # it writes no line, opens no window and takes no focus: the figure
        # redrawing where it already is *is* the output.
        if not following:
            self.output.appendPlainText(result.text)
            if asked and not stopped:
                self.output.appendPlainText("(it finished before it could stop)")
        self._progress_lines = 0
        if refused is not None:
            self.output.appendPlainText(refused + " (its figures are shown)")
            self.status.setText("not applied: the fit is still running")
            self.status.setStyleSheet(WARNING_STYLE)
        elif self._warnings and not following:
            self.status.setText(f"{'stopped' if stopped else 'done'}, "
                                f"{self._warnings} warning"
                                f"{'s' if self._warnings > 1 else ''}")
        else:
            self.status.setText("following ROI" if following
                                else "stopped" if stopped else "done")
        if not following and self.plugin.text_window and result.text:
            self.show_text()          # a log: not something to scroll through
                                      # a four-line slot (`Plugin.text_window`)
        # a run may have added to a list the form offers -- the expressions
        # the math parser has been given, the evaluations a chain made
        self.form.refresh()
        self._describe()
        self.result_button.setEnabled(bool(result.figures() or result.text))
        self.result_button.setToolTip(RESULT_TIP)
        # The figures are what most runs are for: a measurement is read off
        # them, so the window opens without a second click on Show result.
        if result.figures() and (following or self._show_figures):
            self.plot(raise_window=not following)
        self.ended.emit("done", result)

    def _on_failed(self, message: str, text: str) -> None:
        """``message`` what the error says, ``text`` its traceback: the one
        in red beside Run, the other behind *details…* (and in the log)."""
        following, self._following = self._following, False
        if self._owns_live:
            # the fit that was streaming failed: what it streamed stays, as
            # an ordinary table (`Session.end_live`)
            self._owns_live = False
            self.session.end_live()
        self._progress_lines = 0
        print(text)
        self._release_buttons()
        if following:
            # the ROI is being dragged and has passed over a position with too
            # few localizations in it, which is not a failure to report: the
            # next position will ask again
            self.status.setText("follow ROI: " + message[:60])
            self.ended.emit("failed", text)
            return
        self.output.appendPlainText(f"failed: {message}")
        self.status.setText(message)
        self.status.setToolTip(message)
        self.status.setStyleSheet(ERROR_STYLE)
        self._failure = (message, text)
        self.details_button.show()
        self.ended.emit("failed", text)

    def show_details(self) -> None:
        """The last failure's traceback, with *Save a bug report…*."""
        if self._failure is None:
            return
        message, text = self._failure
        if self._details_window is not None:
            self._details_window.close()
        self._details_window = details_window(
            self, f"{self.plugin.name}: failed", message, text, self.session)

    def _on_stopped(self, text: str) -> None:
        """A run that stopped with nothing worth keeping (`Stopped`): the
        session is as it was, and the plugin says why."""
        self._following = False
        if self._owns_live:
            self._owns_live = False
            self.session.end_live()
        self._progress_lines = 0
        self._release_buttons()
        self.output.appendPlainText(text or "stopped")
        self.status.setText("stopped")
        self.status.setStyleSheet("")
        self.ended.emit("stopped", text)

    def show_result(self) -> None:
        """The figures if the run drew any, else what it reported."""
        if self.result is None:
            return
        if self.result.figures():
            self.plot()
        else:
            self.show_text()

    def show_text(self) -> None:
        """What the run reported, in a window that can hold it."""
        if self.result is None or not self.result.text:
            return
        from .figures import TextWindow
        if self._text_window is None:
            self._text_window = TextWindow(self.plugin.name, self)
        self._text_window.show_text(self.result.text)

    def plot(self, raise_window: bool = True) -> None:
        """Show the result's figures: one window, a tab each beyond the first.

        Only the tab being looked at is drawn, here and on every later run,
        which is what keeps a plugin with six figures as quick to plot as one
        with a single figure.  ``raise_window`` is off for a follow-ROI step: taking
        the focus ten times a second would make the ROI impossible to drag.
        """
        if self.result is None:
            return
        from .figures import ResultWindow
        if self._window is None:
            self._window = ResultWindow(self.plugin.name, self)
        self._window.show()
        self._window.show_plots(self.result.figures(), self.result.title)
        if raise_window:
            self._window.raise_()


    def show_help(self) -> None:
        from .help_window import show_help
        show_help(self.plugin.path, self.window())


class PluginWindow(QMainWindow):
    """One plugin in a window of its own, pinned to nothing.

    A tab is a curated list and pinning is a decision; this is the other way
    in -- pick anything from the Plugins menu and use it once, in a window
    that can sit beside the picture rather than in the column of sections.
    The panel is the same `PluginPanel` a tab builds, so a run from here goes
    through the session exactly as one from a tab does.

    Hidden rather than destroyed on close, and kept by the window that opened
    it, so reopening finds the settings that were typed into it.
    """

    def __init__(self, plugin_cls: Type[Plugin], session: Session, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(plugin_cls.name or plugin_cls.path)
        self.panel = PluginPanel(plugin_cls, session)
        area = ColumnScroll(self.panel)
        # the same ? as a section's title bar, in the one place a window has
        self.help_button = QToolButton(text="?", autoRaise=True)
        self.help_button.setToolTip("what this plugin does, how it works, "
                                    "and every setting (F1)")
        self.help_button.clicked.connect(self.panel.show_help)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addStretch(1)
        top.addWidget(self.help_button)
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addLayout(top)
        column.addWidget(area)
        self.setCentralWidget(central)
        self.resize(max(self.panel.sizeHint().width() + 40, 380), 520)
