"""A plugin as a section: its settings form, a Run button, and the result.

The plugin runs in a worker thread so the window keeps responding; the result
is applied to the session on the GUI thread.
"""
from __future__ import annotations

import traceback
from typing import Optional, Type

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QVBoxLayout, QWidget)

from ..plugins import Plugin, Result
from ..session import Session
from .params import SettingsForm


class _Worker(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, plugin, locs, selection, settings):
        super().__init__()
        self.args = (plugin, locs, selection, settings)

    def run(self) -> None:
        plugin, locs, selection, settings = self.args
        try:
            self.done.emit(plugin.run(locs, selection, settings, self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


class PluginPanel(QWidget):
    def __init__(self, plugin_cls: Type[Plugin], session: Session, parent=None):
        super().__init__(parent)
        self.plugin = plugin_cls()
        self.session = session
        self.result: Optional[Result] = None
        self._thread: Optional[QThread] = None

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
        self.plot_button.setEnabled(False)
        self.status = QLabel("")
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.plot_button)
        buttons.addWidget(self.status, 1)
        layout.addLayout(buttons)
        self.output = QPlainTextEdit(readOnly=True, maximumBlockCount=200)
        self.output.setFixedHeight(60)
        layout.addWidget(self.output)

        self.run_button.clicked.connect(self.run)
        self.plot_button.clicked.connect(self.plot)

    def run(self) -> None:
        try:
            settings = self.form.value()
        except ValueError as e:
            self.status.setText(f"bad value: {e}")
            return
        selection = self.session.selection()
        self.run_button.setEnabled(False)
        self.status.setText("running...")
        self._thread = QThread()
        self._worker = _Worker(self.plugin, self.session.locs, selection, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        for sig in (self._worker.done, self._worker.failed):
            sig.connect(self._thread.quit)
        self._thread.start()

    def _on_done(self, result: Result) -> None:
        self.result = result
        self.session.apply(self.plugin, result)
        self.output.appendPlainText(result.text)
        self.status.setText("done")
        self.run_button.setEnabled(True)
        self.plot_button.setEnabled(result.plot is not None)

    def _on_failed(self, text: str) -> None:
        self.output.appendPlainText(text.strip().splitlines()[-1])
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
