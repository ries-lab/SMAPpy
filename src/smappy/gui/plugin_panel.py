"""A plugin as a section: its settings form, a Run button, and the result.

The plugin runs in a worker thread so the window keeps responding; the result
is applied to the session on the GUI thread.
"""
from __future__ import annotations

import traceback
from typing import Optional, Type

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QVBoxLayout, QWidget)

from ..plugins import Plugin, Result
from ..session import Session
from .params import SettingsForm


class _Worker(QObject):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, plugin, context, settings):
        super().__init__()
        self.args = (plugin, context, settings)

    def run(self) -> None:
        plugin, context, settings = self.args
        try:
            self.done.emit(plugin.run(context, settings))
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
        self.plot_button = QPushButton("Plot")
        self.plot_button.setToolTip("show the plugin's result figure (the drift curves, say)")
        self.plot_button.setEnabled(False)
        self.status = QLabel("")
        buttons.addWidget(self.run_button)
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

    def _react(self, path: str) -> None:
        """Let the plugin answer an edit, e.g. fill the camera from the file."""
        try:
            updates = self.plugin.react(path, self.form.value())
        except (ValueError, TypeError):
            return
        if updates:
            self.form.set_values(updates)

    def _on_stream(self, event: str, payload) -> None:
        if event == "start":
            self.session.begin_live(payload.get("extent"), payload.get("path"))
        elif event == "block":
            self.session.append(payload)

    def run(self) -> None:
        try:
            settings = self.form.value()
        except ValueError as e:
            self.status.setText(f"bad value: {e}")
            return
        # built here, on the GUI thread: the context reads the table and the
        # selection now, so the worker cannot race a live fit rebinding them
        context = self.session.context(progress=self.progressed.emit,
                                       stream=self.streamed.emit)
        self.run_button.setEnabled(False)
        self.status.setText("running...")
        self._progress_lines = 0
        self._thread = QThread()
        # Qt's default thread stack (512 kB on macOS) is too small for HDF5 and
        # the fitter's own threads' bookkeeping: a bus error, not an exception
        self._thread.setStackSize(32 * 1024 * 1024)
        self._worker = _Worker(self.plugin, context, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        for sig in (self._worker.done, self._worker.failed):
            sig.connect(self._thread.quit)
        self._thread.start()

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

    def _on_done(self, result: Result) -> None:
        self.result = result
        self.session.apply(self.plugin, result)
        self.output.appendPlainText(result.text)
        self._progress_lines = 0
        self.status.setText("done")
        self.run_button.setEnabled(True)
        self.plot_button.setEnabled(result.plot is not None)

    def _on_failed(self, text: str) -> None:
        self.output.appendPlainText(text.strip().splitlines()[-1])
        self._progress_lines = 0
        print(text)
        self.status.setText("failed")
        self.run_button.setEnabled(True)

    def plot(self) -> None:
        if self.result is None or self.result.plot is None:
            return
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        self.result.plot(ax)
        fig.canvas.manager.set_window_title(self.plugin.name)
        fig.show()
