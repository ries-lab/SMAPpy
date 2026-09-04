"""The windows: a render view, and a compact control window with tabs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QLineEdit,
                               QMainWindow, QScrollArea, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import plugins
from ..session import Session
from .plugin_panel import PluginPanel
from .render_tab import RenderTab
from .render_view import RenderToolBar, RenderView
from .widgets import CollapsibleSection, FloatingWindow

TABS = ("Localize", "Render", "Analysis", "ROI")


class PluginTab(QWidget):
    """Every plugin registered under one tab, as collapsible sections.

    One section is open at a time, and the search box narrows them by name.
    """

    def __init__(self, tab: str, session: Session, parent=None):
        super().__init__(parent)
        self.sections: List[CollapsibleSection] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.search = QLineEdit(placeholderText="find plugin...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        inner = QWidget()
        self.stack = QVBoxLayout(inner)
        self.stack.setContentsMargins(0, 0, 0, 0)
        for path, cls in plugins.available(tab + "/").items():
            title = path[len(tab) + 1:]
            section = CollapsibleSection(title, PluginPanel(cls, session), detachable=True)
            section.toggled.connect(lambda on, s=section: self._one_open(s, on))
            section.detach_requested.connect(lambda s=section: self._detach(s))
            self.sections.append(section)
            self.stack.addWidget(section)
        if not self.sections:
            self.stack.addWidget(QLabel("no plugins yet"))
        self.stack.addStretch(1)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(inner)
        scroll.setFrameShape(QScrollArea.NoFrame)
        layout.addWidget(scroll)
        if self.sections:
            self.sections[0].set_expanded(True)

    def _detach(self, section: CollapsibleSection) -> None:
        """Move a plugin to its own window, so several can be open at once."""
        window = FloatingWindow(section.title, section.detach(), parent=self.window())
        window.closed.connect(section.reattach)
        window.resize(section.content.sizeHint().width() + 20, 400)
        window.show()
        self._windows = getattr(self, "_windows", []) + [window]

    def _one_open(self, opened: CollapsibleSection, on: bool) -> None:
        if on:
            for s in self.sections:
                if s is not opened and s.button.isChecked():
                    s.set_expanded(False)

    def _filter(self, text: str) -> None:
        text = text.lower()
        for s in self.sections:
            s.setVisible(text in s.title.lower())


class ControlWindow(QMainWindow):
    def __init__(self, session: Session, render: "RenderWindow"):
        super().__init__()
        self.session = session
        self.render_window = render
        self.setWindowTitle("smappy")
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(PluginTab("Localize", session), "Localize")
        self.tabs.addTab(RenderTab(session, render.view), "Render")
        self.tabs.addTab(PluginTab("Analysis", session), "Analysis")
        self.tabs.addTab(PluginTab("ROI", session), "ROI")
        self.tabs.setCurrentIndex(1)
        self.setCentralWidget(self.tabs)
        self.resize(360, 640)

        menu = self.menuBar().addMenu("File")
        self._action(menu, "Open...", QKeySequence.Open, self.open)
        self._action(menu, "Save", QKeySequence.Save, self.save)
        self._action(menu, "Save as...", QKeySequence.SaveAs, self.save_as)
        menu.addSeparator()
        self.undo_action = self._action(menu, "Undo", QKeySequence.Undo, session.undo)
        view = self.menuBar().addMenu("View")
        self._action(view, "Reset view", "Ctrl+0", render.view.reset)
        self._action(view, "Show render window", None, render.show)
        session.on_change(self._on_session)
        self._on_session("locs")

    def _action(self, menu, text, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _on_session(self, what: str) -> None:
        self.undo_action.setEnabled(self.session.can_undo)
        name = self.session.path.name if self.session.path else "no file"
        n = len(self.session.locs)
        self.statusBar().showMessage(f"{name}: {n} localizations")
        self.render_window.setWindowTitle(f"smappy - {name}")

    def open(self) -> None:
        start = str(self.session.path.parent) if self.session.path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open localizations", start,
                                              "HDF5 (*.h5 *.hdf5)")
        if path:
            self.session.load(path)

    def save(self) -> None:
        if self.session.path is None:
            return self.save_as()
        self.session.save()
        self.statusBar().showMessage(f"saved {self.session.path}")

    def save_as(self) -> None:
        start = str(self.session.path) if self.session.path else ""
        path, _ = QFileDialog.getSaveFileName(self, "Save localizations", start,
                                              "HDF5 (*.h5 *.hdf5)")
        if path:
            self.session.save(path)
            self._on_session("locs")


class RenderWindow(QMainWindow):
    def __init__(self, session: Session):
        super().__init__()
        self.view = RenderView(session)
        self.setCentralWidget(self.view)
        self.addToolBar(RenderToolBar(self.view))
        self.resize(900, 900)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(argv)
    session = Session()
    if len(argv) > 1:
        session.load(argv[1])
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    screen = app.primaryScreen().availableGeometry()
    control.move(screen.left(), screen.top())
    render.move(screen.left() + control.width() + 20, screen.top())
    render.show()
    control.show()
    if len(session.locs):
        render.view.reset()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
