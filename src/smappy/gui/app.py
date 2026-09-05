"""The windows: a render view, and a compact control window with tabs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout,
                               QMessageBox,
                               QLabel, QLineEdit,
                               QMainWindow, QScrollArea, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import plugins
from ..io.formats import csv_columns, guess_csv_mapping, name_filter, reader_for
from .dialogs import CsvMappingDialog, PixelSizeDialog
from ..session import Session
from .plugin_panel import PluginPanel
from .render_tab import RenderTab
from .render_view import RenderToolBar, RenderView
from .widgets import CollapsibleSection, detach_to_window

TABS = ("Localize", "Render", "Analysis", "ROI")


class PluginTab(QWidget):
    """Every plugin registered under one tab, as collapsible sections.

    Without *all* ticked only the favourites show, flat; with it the full
    tree, grouped by the path between the tab and the plugin.  A star on a
    section toggles favourite (kept in the user's settings), the arrow moves
    the plugin to its own window, and one section is open at a time.
    """

    def __init__(self, tab: str, session: Session, parent=None):
        super().__init__(parent)
        self.tab = tab
        self.settings = QSettings("smappy", "gui")
        self.sections: List[CollapsibleSection] = []
        self.panels: Dict[str, PluginPanel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        self.search = QLineEdit(placeholderText="find plugin...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.all = QCheckBox("all")
        self.all.setToolTip("every plugin, as a tree; otherwise the favourites")
        self.all.setChecked(self.settings.value(f"{tab}/all", False, type=bool))
        self.all.toggled.connect(self.rebuild)
        top.addWidget(self.search, 1)
        top.addWidget(self.all)
        layout.addLayout(top)
        self.inner = QWidget()
        self.stack = QVBoxLayout(self.inner)
        self.stack.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(self.inner)
        scroll.setFrameShape(QScrollArea.NoFrame)
        layout.addWidget(scroll)
        for path, cls in plugins.available(tab + "/").items():
            self.panels[path] = PluginPanel(cls, session)
        self.rebuild()

    # ---------------------------------------------------------- favourites
    def is_favorite(self, path: str) -> bool:
        return self.settings.value(f"favorite/{path}", self.panels[path].plugin.favorite,
                                   type=bool)

    def _set_favorite(self, path: str, on: bool) -> None:
        self.settings.setValue(f"favorite/{path}", on)
        if not self.all.isChecked() and not on:
            self.rebuild()

    # ------------------------------------------------------------- building
    def rebuild(self) -> None:
        self.settings.setValue(f"{self.tab}/all", self.all.isChecked())
        # take every panel out of its section before the sections go
        for section in self.sections:
            if section.content.parent() is section.frame:
                section.detach()
        while self.stack.count():
            item = self.stack.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.sections = []
        prefix = len(self.tab) + 1
        shown = {p: panel for p, panel in self.panels.items()
                 if self.all.isChecked() or self.is_favorite(p)}
        if self.all.isChecked():
            groups: Dict[str, List[str]] = {}
            for path in shown:
                group, _, _ = path[prefix:].rpartition("/")
                groups.setdefault(group, []).append(path)
            for group, paths in groups.items():
                box = QWidget()
                inner = QVBoxLayout(box)
                inner.setContentsMargins(0, 0, 0, 0)
                for path in paths:
                    inner.addWidget(self._section(path, path.rsplit("/", 1)[-1]))
                if group:
                    self.stack.addWidget(CollapsibleSection(group, box, expanded=True))
                else:
                    self.stack.addWidget(box)
        else:
            for path in shown:
                self.stack.addWidget(self._section(path, path[prefix:]))
        if not shown:
            self.stack.addWidget(QLabel("no plugins yet" if not self.panels
                                        else "no favourites: tick 'all'"))
        self.stack.addStretch(1)
        if self.sections:
            self.sections[0].set_expanded(True)
        self._filter(self.search.text())

    def _section(self, path: str, title: str) -> CollapsibleSection:
        panel = self.panels[path]
        if panel.parent() is not None:         # still inside a floating window
            panel.setParent(None)
        section = CollapsibleSection(title, panel, detachable=True,
                                     star=self.is_favorite(path))
        section.toggled.connect(lambda on, s=section: self._one_open(s, on))
        section.detach_requested.connect(lambda s=section: detach_to_window(s, self.window()))
        section.starred.connect(lambda on, p=path: self._set_favorite(p, on))
        self.sections.append(section)
        return section

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
        self._action(menu, "Add file...", "Ctrl+Shift+O", lambda: self.open(append=True))
        self._action(menu, "Open image...", None, self.open_image)
        self._action(menu, "Save", QKeySequence.Save, self.save)
        self._action(menu, "Save as...", QKeySequence.SaveAs, self.save_as)
        menu.addSeparator()
        self.undo_action = self._action(menu, "Undo", QKeySequence.Undo, session.undo)
        view = self.menuBar().addMenu("View")
        self.view3d_window = None
        self._action(view, "3D view", "Ctrl+3", self.show_3d)
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

    def show_3d(self) -> None:
        if self.view3d_window is None:
            from .view3d import View3DWindow
            self.view3d_window = View3DWindow(self.session)
            QApplication.instance().aboutToQuit.connect(self.view3d_window.view.shutdown)
            self.view3d_window.move(self.render_window.x(), self.render_window.y() + 60)
        self.view3d_window.show()
        self.view3d_window.raise_()

    def _on_session(self, what: str) -> None:
        if what in ("layer", "layers", "locs", "roi") and not self.render_window.isVisible():
            self.render_window.show()      # closed by accident: a change wants it back
        self.undo_action.setEnabled(self.session.can_undo)
        names = self.session.file_names()
        name = (names[0] if len(names) == 1 else f"{len(names)} files") if names else "no file"
        n = len(self.session.locs)
        self.statusBar().showMessage(f"{name}: {n} localizations")
        self.render_window.setWindowTitle(f"smappy - {name}")

    def open(self, append: bool = False) -> None:
        """One or more files; the first replaces (unless appending), the rest join."""
        start = str(self.session.path.parent) if self.session.path else ""
        paths, _ = QFileDialog.getOpenFileNames(self, "Add localizations" if append
                                                else "Open localizations", start, name_filter())
        for i, path in enumerate(paths):
            args = {}
            if self._needs_mapping(path):
                dialog = CsvMappingDialog(path, self)
                if dialog.exec() != QDialog.Accepted:
                    continue
                args = dialog.reader_args()
            try:
                self.session.load(path, append=append or i > 0, **args)
            except Exception as e:                      # a bad file is not a crash
                QMessageBox.warning(self, "could not open", f"{Path(path).name}:\n{e}")

    @staticmethod
    def _needs_mapping(path: str) -> bool:
        """A csv whose headers do not say where x and y are."""
        try:
            if "mapping" not in reader_for(path).needs:
                return False
        except ValueError:
            return False
        guessed = set(guess_csv_mapping(csv_columns(Path(path))[0]).values())
        return not {"x_nm", "y_nm"} <= guessed

    def open_image(self) -> None:
        start = str(self.session.path.parent) if self.session.path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start,
                                              "Images (*.tif *.tiff *.png)")
        if not path:
            return
        try:
            self.session.open_image(path)
        except ValueError:                                  # no pixel size in the file
            dialog = PixelSizeDialog(Path(path).name, parent=self)
            if dialog.exec() == QDialog.Accepted:
                px, x0, y0 = dialog.values()
                self.session.open_image(path, px, x0, y0)

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
