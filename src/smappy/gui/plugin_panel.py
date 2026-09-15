"""A plugin as a section: its settings form, a Run button, and the result.

The plugin runs in a worker thread so the window keeps responding; the result
is applied to the session on the GUI thread.
"""
from __future__ import annotations

import traceback
from typing import Optional, Type

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

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
        plugin, context, settings = self.args
        try:
            work = getattr(plugin, self.job)
            self.done.emit(work(context, settings, **self.kwargs))
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
        self.status = QLabel("")
        buttons.addWidget(self.plot_button)
        buttons.addWidget(self.status, 1)
        layout.addLayout(buttons)
        self.output = QPlainTextEdit(readOnly=True, maximumBlockCount=500)
        self.output.setFixedHeight(90)
        layout.addWidget(self.output)

        self.run_button.clicked.connect(self.run)
        self.plot_button.clicked.connect(self.plot)
        self.form.field_changed.connect(self._react)
        self.progressed.connect(self._on_progress)
        self.streamed.connect(self._on_stream)
        self._active()

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
        if job == "run" and not self._preflight(settings):
            return
        # built here, on the GUI thread: the context reads the table and the
        # selection now, so the worker cannot race a live fit rebinding them
        context = self.session.context(progress=self.progressed.emit,
                                       stream=self.streamed.emit)
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

    def _preflight(self, settings) -> bool:
        """Ask the plugin whether this run wants agreeing to first.

        On the GUI thread and before the worker exists, so the dialog is an
        ordinary modal one.  What the plugin reports here is written straight
        into the log rather than through `progressed`: it is a standing record
        of what was agreed to, and the progress lines that follow overwrite
        each other on the line below it.
        """
        try:
            context = self.session.context(progress=self.output.appendPlainText)
            question = self.plugin.preflight(context, settings)
        except Exception as e:
            # An estimate is a courtesy.  One that cannot be made -- a table in
            # pixels with no pixel size, a plugin that does not expect this
            # data -- must not be what stops the run the user asked for.  It
            # says so rather than vanishing, so that a broken preflight is
            # visible as one.
            self.output.appendPlainText(f"(no estimate: {type(e).__name__}: {e})")
            return True
        if not question:
            return True
        answer = QMessageBox.question(
            self, f"{self.plugin.name}: before running", question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            return True
        self.output.appendPlainText("not run")
        self.status.setText("cancelled")
        return False

    def _on_progress(self, text: str) -> None:
        """Progress replaces the last line while it is a progress line, so a
        long fit does not scroll its own summary away."""
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
        self.result = result
        # a preview is looked at, never applied: it exists so that the session
        # is not changed before the settings are right
        if self._job != "preview":
            self.session.apply(self.plugin, result)
        self.output.appendPlainText(result.text)
        self._progress_lines = 0
        self.status.setText("done")
        for button in self._buttons():
            button.setEnabled(True)
        self.plot_button.setEnabled(bool(result.figures()))
        if self._job == "preview" and result.figures():
            self.plot()

    def _on_failed(self, text: str) -> None:
        self.output.appendPlainText(text.strip().splitlines()[-1])
        self._progress_lines = 0
        print(text)
        self.status.setText("failed")
        for button in self._buttons():
            button.setEnabled(True)

    def plot(self) -> None:
        """Show every figure the result has, one window each."""
        if self.result is None:
            return
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        for name, draw in self.result.figures():
            fig, ax = plt.subplots()
            draw(ax)
            fig.canvas.manager.set_window_title(
                f"{self.plugin.name}: {name}" if name else self.plugin.name)
            fig.show()
