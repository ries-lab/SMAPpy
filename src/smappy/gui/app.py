"""The windows: a render view, and a compact control window with tabs."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QElapsedTimer, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout,
                               QMessageBox, QPushButton,
                               QLabel, QLineEdit,
                               QMainWindow, QScrollArea, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import plugins
from ..io.formats import (csv_columns, guess_csv_mapping, load as load_any,
                          name_filter, reader_for)
from .dialogs import CsvMappingDialog, PixelSizeDialog
from ..session import GROUPED_BY_DEFAULT, Session
from .plugin_panel import PluginPanel
from .render_tab import RenderTab
from .roi_tab import ROITab
from .render_view import RenderToolBar, RenderView
from .widgets import CONTROL_WIDTH, CollapsibleSection, detach_to_window

TABS = ("Localize", "Render", "Analysis", "ROI")


class LoadTask(QThread):
    """Read a file, and link it, off the GUI thread.

    Both halves are slow enough to freeze a window on a real dataset -- 31 s of
    gzip and 96 s of linking on a 57 M localization ``_sml.mat`` -- and neither
    touches Qt or the session, so both belong here.  What comes back is plain
    data; the session is only ever changed on the main thread, in `_loaded`.

    An appended file is read but not linked: `Session.add_file` re-links the
    merged table anyway, so linking it here would be work done twice.
    """

    progress = Signal(str)
    loaded = Signal(object, object, object)      # locs, info, grouped or None
    failed = Signal(str)

    def __init__(self, path: str, args: dict, append: bool,
                 group_settings, parent=None):
        super().__init__(parent)
        self.path, self.args, self.append = path, args, append
        self.group_settings = group_settings

    def run(self) -> None:
        try:
            self.progress.emit("loading")
            locs, info = load_any(self.path, **self.args)
            grouped = None
            if not self.append and GROUPED_BY_DEFAULT and len(locs) and "frame" in locs:
                from ..group import group
                grouped, _ = group(
                    locs, self.group_settings,
                    progress=lambda text, _f: self.progress.emit(f"Grouper: {text}"))
            self.loaded.emit(locs, info, grouped)
        except Exception as e:                   # a bad file is not a crash
            self.failed.emit(f"{e}")


class PluginTab(QWidget):
    """Every plugin registered under one tab, as collapsible sections.

    Without *all* ticked only the favourites show, flat; with it the full
    tree, grouped by the path between the tab and the plugin.  A star on a
    section toggles favourite (kept in the user's settings), the arrow moves
    the plugin to its own window, and one section is open at a time.
    """

    def __init__(self, tab: str, session: Session, tools=(), parent=None):
        """``tools`` are (label, tooltip, callback) buttons above the plugins,
        for things that are whole applications rather than plugins."""
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
        if tools:
            row = QHBoxLayout()
            for label, tip, callback in tools:
                button = QPushButton(label)
                button.setToolTip(tip)
                button.clicked.connect(callback)
                row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)
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


def _keys(standard, fallback: str):
    """The platform's standard sequence, plus the fallback if it is different
    (a duplicate would make the shortcut ambiguous and dead)."""
    keys = [QKeySequence(standard), QKeySequence(fallback)]
    unique, seen = [], set()
    for k in keys:
        if k.toString() and k.toString() not in seen:
            unique.append(k)
            seen.add(k.toString())
    return unique


def window_shortcuts(window: QWidget, with_quit: bool = True) -> None:
    """Cmd/Ctrl+W closes this window, Cmd/Ctrl+Q quits: on every window."""
    close = QAction("Close window", window, triggered=window.close)
    close.setShortcuts(_keys(QKeySequence.Close, "Ctrl+W"))
    actions = [close]
    if with_quit:
        quit_ = QAction("Quit", window, triggered=lambda: QApplication.instance().quit())
        quit_.setShortcuts(_keys(QKeySequence.Quit, "Ctrl+Q"))
        actions.append(quit_)
    for action in actions:
        action.setShortcutContext(Qt.WindowShortcut)
        window.addAction(action)


class ControlWindow(QMainWindow):
    def __init__(self, session: Session, render: "RenderWindow"):
        super().__init__()
        self.session = session
        self.render_window = render
        self.setWindowTitle("smappy")
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(PluginTab("Localize", session, tools=[
            ("Bead calibration...", "build an experimental spline PSF calibration from "
             "bead z-stacks; opens its own window", self.open_calibration)]), "Localize")
        self.tabs.addTab(RenderTab(session, render.view), "Render")
        self.tabs.addTab(PluginTab("Analysis", session), "Analysis")
        self.roi_tab = ROITab(session, render.view)
        self.tabs.addTab(self.roi_tab, "ROI")
        self.tabs.setCurrentIndex(1)
        self.setCentralWidget(self.tabs)
        # tall on purpose: with a file open the Render tab's own content wants
        # some 700 px, and scrolling for the display settings every time is
        # what one notices.  Capped so it still fits a laptop screen.
        screen = QApplication.primaryScreen()
        height = screen.availableGeometry().height() - 60 if screen else 900
        self.resize(CONTROL_WIDTH, max(640, min(1000, height)))

        # Loading runs in a LoadTask; these hold its state.  `_load_locked` is
        # everything that would change the table under a running one.
        self._queue: List[tuple] = []
        self._task: Optional[LoadTask] = None
        self._loading_name = ""
        self._loading_stage = ""
        self._elapsed = QElapsedTimer()
        self._elapsed.start()
        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._tick)

        menu = self.menuBar().addMenu("File")
        open_ = self._action(menu, "Open...", QKeySequence.Open, self.open)
        add = self._action(menu, "Add file...", "Ctrl+Shift+O", lambda: self.open(append=True))
        image = self._action(menu, "Open image...", None, self.open_image)
        save = self._action(menu, "Save", QKeySequence.Save, self.save)
        save_as = self._action(menu, "Save as...", QKeySequence.SaveAs, self.save_as)
        menu.addSeparator()
        self.undo_action = self._action(menu, "Undo", QKeySequence.Undo, session.undo)
        self._load_locked = [open_, add, image, save, save_as, self.undo_action]
        menu.addSeparator()
        quit_ = self._action(menu, "Quit", None, lambda: QApplication.instance().quit())
        quit_.setShortcuts(_keys(QKeySequence.Quit, "Ctrl+Q"))
        window_shortcuts(self, with_quit=False)
        view = self.menuBar().addMenu("View")
        self.view3d_window = None
        self.calibration_window = None
        self._action(view, "3D view", "Ctrl+3", self.show_3d)
        tools = self.menuBar().addMenu("Tools")
        self._action(tools, "ROI manager", "Ctrl+R", lambda: self.roi_tab.open_manager())
        tools.addSeparator()
        self._action(tools, "Bead calibration...", None, self.open_calibration)
        self._action(tools, "Dual-colour calibration...", None,
                     lambda: self.open_calibration(dual=True))
        self._action(view, "Reset view", "Ctrl+0", render.view.reset)
        self._action(view, "Show render window", None, render.show)
        QApplication.instance().aboutToQuit.connect(self.stop_loading)
        session.on_change(self._on_session)
        self._on_session("locs")

    def _action(self, menu, text, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def open_calibration(self, dual: bool = False) -> None:
        """The calibration window: bead stacks in, a spline calibration out.

        A window of its own, like the 3D viewer, but in this process, so a
        saved calibration can go straight into the Spline 3D fitter.
        """
        from ..calibrate.qt_gui import CalibrationWindow
        if self.calibration_window is None:
            self.calibration_window = CalibrationWindow(parent=self)
            self.calibration_window.calibrated.connect(self.use_calibration)
        if dual:
            self.calibration_window.mode_box.setCurrentText("Dual colour")
        self.calibration_window.show()
        self.calibration_window.raise_()

    def use_calibration(self, path: str) -> None:
        """Put a fresh calibration into the Spline 3D fitter's settings."""
        from .plugin_panel import PluginPanel
        for panel in self.tabs.widget(0).findChildren(PluginPanel):
            if panel.plugin.path.endswith("Spline 3D"):
                panel.form.set_values({"model.calibration": str(path)})
                self.statusBar().showMessage(
                    f"the Spline 3D fitter now uses {Path(path).name}", 8000)
                return

    def show_3d(self) -> None:
        if self.view3d_window is None:
            from .view3d import View3DWindow
            self.view3d_window = View3DWindow(self.session)
            window_shortcuts(self.view3d_window)
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
        """One or more files; the first replaces (unless appending), the rest join.

        The reading and the linking happen in a `LoadTask`, one file at a time,
        so the window stays alive and says what it is doing.  Anything that
        would change the session underneath a running task is disabled while it
        runs; everything else -- panning, the filters, the other windows --
        keeps working on the file that is already open.
        """
        start = str(self.session.path.parent) if self.session.path else ""
        paths, _ = QFileDialog.getOpenFileNames(self, "Add localizations" if append
                                                else "Open localizations", start, name_filter())
        self.load_paths(paths, append=append)

    def load_paths(self, paths, append: bool = False, reset_view: bool = False) -> None:
        """Queue files for the loader.  ``reset_view`` frames the first one,
        which is what starting with a file on the command line wants."""
        for i, path in enumerate(paths):
            args = {}
            if self._needs_mapping(path):
                dialog = CsvMappingDialog(path, self)
                if dialog.exec() != QDialog.Accepted:
                    continue
                args = dialog.reader_args()
            self._queue.append((str(path), args, append or i > 0,
                                reset_view and i == 0))
        self._start_next()

    def _start_next(self) -> None:
        if self._task is not None:            # one at a time: each one replaces
            return                            # or appends to what the last left
        if not self._queue:
            self._set_loading(False)
            self._on_session("locs")          # back to "name: N localizations"
            return
        path, args, append, reset_view = self._queue.pop(0)
        self._set_loading(True)
        self._loading_name = Path(path).name
        self._task = LoadTask(path, args, append, self.session.group_settings, self)
        self._task.progress.connect(self._loading_progress)
        self._task.loaded.connect(
            lambda l, i, g, a=append, r=reset_view: self._loaded(l, i, g, a, r))
        self._task.failed.connect(self._load_failed)
        self._task.finished.connect(self._task_finished)
        self._elapsed.restart()
        self._loading_progress("loading")
        self._task.start()

    def _set_loading(self, on: bool) -> None:
        """While a file is being read, nothing may change the table under it."""
        for action in self._load_locked:
            action.setEnabled(False if on else True)
        if on:
            self._ticker.start(500)
        else:
            self._ticker.stop()
            self.undo_action.setEnabled(self.session.can_undo)

    def _loading_progress(self, text: str) -> None:
        self._loading_stage = text
        self._tick()

    def _tick(self) -> None:
        """The stage plus a clock -- linking reports once and then runs for
        minutes, and a status bar that never changes reads as a hung window."""
        seconds = self._elapsed.elapsed() // 1000
        clock = f"{seconds // 60}:{seconds % 60:02d}"
        self.statusBar().showMessage(
            f"{self._loading_name}: {self._loading_stage}... ({clock})")

    def _loaded(self, locs, info, grouped, append: bool, reset_view: bool) -> None:
        self.session.add_file(locs, info, append=append, grouped=grouped)
        if reset_view:
            self.render_window.view.reset()

    def _load_failed(self, message: str) -> None:
        QMessageBox.warning(self, "could not open", f"{self._loading_name}:\n{message}")

    def _task_finished(self) -> None:
        self._task = None
        self._start_next()

    def stop_loading(self) -> None:
        """Quitting with a load running.  The queue is dropped and the task is
        given a moment to end on its own; a link that is minutes from finishing
        is cut instead, which is safe here because the process is going away
        and the only file it holds is open for reading."""
        self._queue.clear()
        task, self._task = self._task, None
        if task is not None and task.isRunning():
            if not task.wait(2000):
                task.terminate()
                task.wait(2000)

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
        window_shortcuts(self)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(argv)
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    screen = app.primaryScreen().availableGeometry()
    control.move(screen.left(), screen.top())
    render.move(screen.left() + control.width() + 20, screen.top())
    render.show()
    control.show()
    if len(argv) > 1:            # after the windows: a big file takes minutes,
        control.load_paths(argv[1:], reset_view=True)   # and says so as it goes
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
