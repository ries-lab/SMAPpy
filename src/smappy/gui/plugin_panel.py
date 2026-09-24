"""A plugin as a section: its settings form, a Run button, and the result.

The plugin runs in a worker thread so the window keeps responding; the result
is applied to the session on the GUI thread.
"""
from __future__ import annotations

import traceback
from typing import Optional, Type

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..plugins import Plugin, Result
from ..session import Session
from .params import SettingsForm



class _Worker(QObject):
    done = Signal(object)
    failed = Signal(str)

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
            work = getattr(plugin, self.job)
            # into the diagnostic log: what ran, with what, and the traceback
            # if it failed -- the first thing a bug report is asked for
            with running(plugin, self.job, settings, n):
                result = work(context, settings, **self.kwargs)
            self.done.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PluginPanel(QWidget):
    # The context is built before the worker exists, so progress cannot go
    # through a worker signal any more.  These belong to the panel, which lives
    # on the GUI thread: emitting them from the worker delivers queued, which
    # is what keeps `_on_progress` and `_on_stream` off that thread.
    progressed = Signal(str)
    streamed = Signal(str, object)

    def __init__(self, plugin_cls: Type[Plugin], session: Session, parent=None):
        super().__init__(parent)
        self.plugin = plugin_cls()
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self.plugin.description:
            about = QLabel(self.plugin.description)
            about.setWordWrap(True)
            about.setStyleSheet("color: gray")
            layout.addWidget(about)
        self.form = SettingsForm(plugin_cls.Settings, plugin_cls.specs())
        layout.addWidget(self.form)
        # a field that reads the session -- the layers editor offers its
        # columns and copies its layers -- is told which one
        for leaf in self.form.leaves():
            if hasattr(leaf, "take_from_session"):
                leaf.session = session
                leaf.refresh()

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
        self.plot_button = QPushButton("Plot")
        self.plot_button.setToolTip("show the plugin's result figure (the drift curves, say)")
        self.plot_button.setEnabled(False)
        # the output box below is four lines tall; anything longer than a
        # summary is unreadable in it, so it gets a window of its own
        self.text_button = QPushButton("Text")
        self.text_button.setToolTip("show what the run reported, in its own window")
        self.text_button.setEnabled(False)
        # a measurement one aims with rather than one asked for afterwards:
        # the plugin says it is cheap enough (`Plugin.live`) and the tick
        # re-previews while the ROI is dragged
        self.live: Optional[QCheckBox] = None
        if plugin_cls.live and plugin_cls.has_preview():
            self.live = QCheckBox("live")
            self.live.setToolTip("redraw while the ROI is moved, without "
                                 "touching the session")
            self.live.toggled.connect(self._on_live)
        self.status = QLabel("")
        buttons.addWidget(self.plot_button)
        buttons.addWidget(self.text_button)
        if self.live is not None:
            buttons.addWidget(self.live)
        buttons.addWidget(self.status, 1)
        layout.addLayout(buttons)
        self.output = QPlainTextEdit(readOnly=True, maximumBlockCount=500)
        self.output.setFixedHeight(90)
        layout.addWidget(self.output)

        self.run_button.clicked.connect(self.run)
        self.plot_button.clicked.connect(self.plot)
        self.text_button.clicked.connect(self.show_text)
        self._live_running = False
        self._live_timer = QTimer(self, singleShot=True, interval=120)
        self._live_timer.timeout.connect(self._live_step)
        self.form.field_changed.connect(self._react)
        self.progressed.connect(self._on_progress)
        self.streamed.connect(self._on_stream)
        self._active()
        session.on_change(self._on_session)
        self._take_saved()

    def release(self) -> None:
        """Stop listening to the session, for a panel about to be replaced
        (a chain rebuilt after its steps were edited)."""
        self.session.off_change(self._on_session)

    def _on_session(self, what: str) -> None:
        if what in ("locs", "results"):
            self._take_saved()
        elif what in ("roi", "roi-edited") and self.live is not None \
                and self.live.isChecked():
            # not straight away: a drag is tens of events a second and a
            # refit is tens of milliseconds, so the timer coalesces them and
            # the last position always wins
            self._live_timer.start()

    def _on_live(self, on: bool) -> None:
        if on:
            self._live_timer.start()

    def _live_step(self) -> None:
        """One live refit, skipped while another is still running.

        Skipped rather than queued: the point is to follow the ROI, and the
        next drag event will ask again in a few tens of milliseconds.
        """
        if self.live is None or not self.live.isChecked():
            return
        if self._thread is not None and self._thread.isRunning():
            self._live_timer.start()          # try again once it is free
            return
        self._live_running = True
        self.preview()

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
        self.plot_button.setEnabled(True)
        when = saved.get("time", "")
        self.plot_button.setToolTip(
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
        self._hints()
        self._active()

    def _hints(self) -> None:
        """Refresh what the fields left on *auto* say they will resolve to.

        Best effort on purpose: working this out means opening the file, which
        may be missing, half-written or not an image at all, and none of that
        is worth an error in a field the user has not finished typing.
        """
        try:
            hints = self.plugin.hints(self.form.value())
        except Exception:
            return
        if hints:
            self.form.set_hints(hints)

    def _active(self) -> None:
        """Grey out the fields the current settings do not read."""
        try:
            flags = self.plugin.active(self.form.value())
        except (ValueError, TypeError):
            return
        if flags:
            self.form.set_active(flags)

    def _on_stream(self, event: str, payload) -> None:
        if event == "start":
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

    def _start(self, job: str, message: str, **kwargs) -> None:
        try:
            settings = self.form.value()
        except ValueError as e:
            self.status.setText(f"bad value: {e}")
            return
        if job == "run":
            go, chosen = self._preflight(settings)
            if not go:
                return
            if chosen is not settings:
                # a preflight choice may hand back other settings than the
                # form's; showing them is what keeps the run from being a
                # surprise afterwards
                settings = chosen
                self.form.set(settings)
        # built here, on the GUI thread: the context reads the table and the
        # selection now, so the worker cannot race a live fit rebinding them
        context = self.session.context(progress=self.progressed.emit,
                                       stream=self.streamed.emit,
                                       grouping=self.plugin.grouping)
        self._job = job
        for button in self._buttons():
            button.setEnabled(False)
        self.status.setText(message)
        self._progress_lines = 0
        self._thread = QThread()
        # Qt's default thread stack (512 kB on macOS) is too small for HDF5 and
        # the fitter's own threads' bookkeeping: a bus error, not an exception
        self._thread.setStackSize(32 * 1024 * 1024)
        self._worker = _Worker(self.plugin, context, settings, job, **kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        for sig in (self._worker.done, self._worker.failed):
            sig.connect(self._thread.quit)
        self._thread.start()

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

        try:
            context = self.session.context(progress=self.output.appendPlainText,
                                           grouping=self.plugin.grouping)
            question = self.plugin.preflight(context, settings)
        except Exception as e:
            # An estimate is a courtesy.  One that cannot be made -- a table in
            # pixels with no pixel size, a plugin that does not expect this
            # data -- must not be what stops the run the user asked for.  It
            # says so rather than vanishing, so that a broken preflight is
            # visible as one.
            self.output.appendPlainText(f"(no estimate: {type(e).__name__}: {e})")
            return True, settings
        if not question:
            return True, settings
        title = f"{self.plugin.name}: before running"
        if not isinstance(question, PreflightQuestion):
            answer = QMessageBox.question(
                self, title, str(question),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                return True, settings
            return self._not_run()
        chosen = self.ask_choice(title, question)
        if chosen is None:
            return self._not_run()
        if chosen.settings is not None:
            self.output.appendPlainText(f"chose: {chosen.label}")
            return True, chosen.settings
        return True, settings

    def _not_run(self):
        self.output.appendPlainText("not run")
        self.status.setText("cancelled")
        return False, None

    def ask_choice(self, title: str, question):
        """Put a question with alternatives; the choice, or None to cancel.

        Its own method because it is the one part of the preflight that opens
        a modal dialog: a test drives the decision by replacing this, as it
        replaces `QMessageBox.question` for the plain yes-or-no.
        """
        from ..plugins import PreflightChoice

        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(question.text)
        # each alternative is a button of its own rather than a second dialog:
        # the choice is between ways of doing the same thing, and they should
        # be readable side by side
        buttons = []
        for choice in question.choices:
            button = box.addButton(choice.label, QMessageBox.ButtonRole.AcceptRole)
            if choice.help:
                button.setToolTip(choice.help)
            buttons.append((button, choice))
        plain = PreflightChoice(label=question.run_label)
        run = box.addButton(question.run_label, QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(buttons[0][0] if buttons else cancel)
        detail = "\n\n".join(f"{c.label}\n{c.help}" for c in question.choices if c.help)
        if detail:
            box.setDetailedText(detail)
        box.exec()
        clicked = box.clickedButton()
        for button, choice in buttons:
            if clicked is button:
                return choice
        return plain if clicked is run else None

    def _on_progress(self, text: str) -> None:
        """Progress replaces the last line while it is a progress line, so a
        long fit does not scroll its own summary away.

        A live step reports nothing: it happens ten times a second and would
        fill the box with the same line while saying less than the figure it
        is redrawing.
        """
        if self._live_running:
            return
        cursor = self.output.textCursor()
        if self._progress_lines:
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(text)
        else:
            self.output.appendPlainText(text)
        self._progress_lines += 1
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())

    def _buttons(self):
        return [b for b in (self.run_button, self.preview_button) if b is not None]

    def _on_done(self, result: Result) -> None:
        live, self._live_running = self._live_running, False
        self.result = result
        # a preview is looked at, never applied: it exists so that the session
        # is not changed before the settings are right
        if self._job != "preview":
            self.session.apply(self.plugin, result)
        # A live step happens ten times a second while the ROI is dragged, so
        # it writes no line, opens no window and takes no focus: the figure
        # redrawing where it already is *is* the output.
        if not live:
            self.output.appendPlainText(result.text)
        self._progress_lines = 0
        self.status.setText("live" if live else "done")
        self.text_button.setEnabled(bool(result.text))
        if not live and self.plugin.text_window and result.text:
            self.show_text()          # a log: not something to scroll through
                                      # a four-line slot (`Plugin.text_window`)
        # a run may have added to a list the form offers -- the expressions
        # the math parser has been given, say
        self.form.refresh()
        for button in self._buttons():
            button.setEnabled(True)
        self.plot_button.setEnabled(bool(result.figures()))
        self.plot_button.setToolTip(
            "show the plugin's result figure (the drift curves, say)")
        if self._job == "preview" and result.figures():
            self.plot(raise_window=not live)

    def _on_failed(self, text: str) -> None:
        live, self._live_running = self._live_running, False
        self._progress_lines = 0
        print(text)
        for button in self._buttons():
            button.setEnabled(True)
        if live:
            # the ROI is being dragged and has passed over a position with too
            # few localizations in it, which is not a failure to report: the
            # next position will ask again
            self.status.setText("live: " + text.strip().splitlines()[-1][:60])
            return
        self.output.appendPlainText(text.strip().splitlines()[-1])
        self.status.setText("failed")

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
        with a single figure.  ``raise_window`` is off for a live step: taking
        the focus ten times a second would make the ROI impossible to drag.
        """
        if self.result is None:
            return
        from .figures import ResultWindow
        if self._window is None:
            self._window = ResultWindow(self.plugin.name, self)
        self._window.show()
        self._window.show_plots(self.result.figures())
        if raise_window:
            self._window.raise_()


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
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.panel)
        self.setCentralWidget(area)
        self.resize(max(self.panel.sizeHint().width() + 40, 380), 520)
