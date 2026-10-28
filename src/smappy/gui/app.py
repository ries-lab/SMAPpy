"""The windows: a render view, and a compact control window with tabs."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QElapsedTimer, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QMenuBar,
                               QMessageBox, QPushButton,
                               QLabel, QLineEdit,
                               QMainWindow, QMenu, QScrollArea, QSizePolicy, QTabWidget,
                               QToolButton, QVBoxLayout, QWidget)

from .. import plugins, workspace as workspace_module
from ..io.formats import (csv_columns, guess_csv_mapping, load as load_any,
                          name_filter, reader_for)
from . import drops, folders
from .dialogs import CsvMappingDialog
from ..session import Session, read_localizations
from ..workspace import Workspace
from .plugin_tab import PluginTab
from .preferences import PreferencesDialog
from .render_tab import RenderTab
from .roi_tab import ROIHeader
from .render_view import RenderToolBar, RenderView
from .widgets import CONTROL_WIDTH, place_beside
from .errors import ERROR_STYLE, details_window, message_of
from .merging import merger_for
from . import make_app
from . import identity
from .identity import install_identity


# The main window's tabs where the menu bar is in the window (not a Mac's):
# framed, a gap under the menu bar, and the current one marked by a bar in the
# highlight colour, which keeps its width where bold text would not.
TAB_STYLE = """
QTabBar::tab {
    margin-top: 4px; padding: 3px 8px;
    border: 1px solid palette(mid); border-bottom: none;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
    background: palette(button);
}
QTabBar::tab:selected {
    background: palette(base);
    border-top: 2px solid palette(highlight);
}
QTabBar::tab:!selected:hover { background: palette(light); }
"""


class HandoffTask(QThread):
    """Take a session another instance handed over (`smappy.handoff`) off the
    GUI thread, and merge its blinks there if a layer shows them -- `LoadTask`
    for a table that comes through shared memory instead of a file."""

    progress = Signal(str)
    loaded = Signal(object, object, object)      # locs, state, grouped or None
    failed = Signal(str, str)                    # the message; the traceback

    def __init__(self, path: str, group_settings, parent=None):
        super().__init__(parent)
        self.path, self.group_settings = path, group_settings

    def run(self) -> None:
        from .. import handoff
        try:
            self.loaded.emit(*handoff.take(self.path, self.group_settings,
                                           progress=self.progress.emit))
        except Exception as e:
            self.failed.emit(message_of(e), traceback.format_exc())


def open_filter() -> str:
    """*File > Open*'s filter: the readers' tables and the images it takes."""
    images = "*.tif *.tiff *.png"
    tables, *rest = name_filter().split(";;")
    every = tables[tables.index("(") + 1:-1]
    return ";;".join([f"Localizations and images ({every} {images})", tables,
                      f"Images ({images})"] + rest)


class LoadTask(QThread):
    """Read a file off the GUI thread.

    Reading is slow enough to freeze a window on a real dataset -- 31 s of
    gzip on a 57 M localization ``_sml.mat`` -- and does not touch Qt or the
    session, so it belongs here.  What comes back is plain data; the session
    is only ever changed on the main thread, in `_loaded`.  The blinks are
    not merged here: the file opens on its localizations and is merged in the
    background after (`gui.merging`), so the window has it at once.
    """

    progress = Signal(str)
    loaded = Signal(object, object)              # locs, info
    failed = Signal(str, str)                    # the message; the traceback

    def __init__(self, path: str, args: dict, parent=None):
        super().__init__(parent)
        self.path, self.args = path, args

    def run(self) -> None:
        try:
            self.loaded.emit(*read_localizations(self.path, progress=self.progress.emit,
                                                 **self.args))
        except Exception as e:                   # a bad file is not a crash
            self.failed.emit(message_of(e), traceback.format_exc())


TUTORIALS_URL = "https://ries-lab.github.io/SMAPpy/"


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


def _add_menu(parent, title: str, owner: QWidget) -> QMenu:
    """``parent.addMenu(title)``, but with a menu that Qt owns.

    PySide6 6.10 hands the menu that ``addMenu(str)`` makes to Python, so it
    is deleted with the last reference to it: the File and Tools menus died
    when ``__init__`` returned, and the Plugins submenus with
    ``_fill_plugins_menu``.  A parent keeps it for as long as the window.
    """
    menu = QMenu(title, owner)
    parent.addMenu(menu)
    return menu


def _copy_selected_text(widget) -> bool:
    """Let a text widget with something selected copy it itself."""
    if widget is None or not hasattr(widget, "copy"):
        return False
    if hasattr(widget, "hasSelectedText"):            # line edits, labels
        selected = widget.hasSelectedText()
    elif hasattr(widget, "textCursor"):               # text edits and browsers
        selected = widget.textCursor().hasSelection()
    else:
        return False
    if selected:
        widget.copy()
    return selected


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
        # merges a large table's blinks in the background, after it opens
        # and whenever a layer asks for them again (`gui.merging`)
        self.merger = merger_for(session, self)
        self.merger.message.connect(lambda text, ms: self.statusBar().showMessage(text, ms))
        self.setProperty(identity.OWN_TITLE, True)      # names the session itself
        # the GUI last saved or loaded, or the shipped one: never one written
        # behind anybody's back (see `workspace`)
        self.workspace = workspace_module.load_startup()
        missing = self.workspace.prune(list(plugins.refs()))
        self.session.apply_default_filters(self.workspace.filters)
        if missing:
            print("workspace: dropped pins for plugins that are not installed: "
                  + ", ".join(sorted(set(missing))))
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # more tabs than the column is wide: arrows to scroll through them,
        # names whole.  Without, a Mac's style widened the window to fit them
        # all, and the control column is meant to stay narrow
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideNone)
        if sys.platform != "darwin":
            # Elsewhere the menu bar is in the window, straight above the
            # tabs, and Windows draws both as flat words in a row: File, Edit,
            # View over Render, Analysis, Regions read as one menu of two
            # lines.  So the menu bar gets a rule and the tabs look like tabs.
            # A Mac's menu is at the top of the screen and needs neither.
            self.menuBar().setStyleSheet(
                "QMenuBar { border-bottom: 1px solid palette(mid); }")
            self.tabs.setStyleSheet(TAB_STYLE)
        self.plugin_tabs: List[PluginTab] = []
        self.roi_tab: Optional[ROIHeader] = None
        self.build_tabs()
        self._update_title()
        # under the tabs, shown only after a file failed to open: what went
        # wrong in red, and the traceback behind *details…* -- as a plugin's
        # panel says a failed run, rather than a box to click away
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.tabs, 1)
        self.load_error = QWidget()
        row = QHBoxLayout(self.load_error)
        row.setContentsMargins(6, 2, 6, 2)
        self.load_error_text = QLabel("")
        self.load_error_text.setWordWrap(True)
        self.load_error_text.setStyleSheet(ERROR_STYLE)
        self.load_error_text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.load_error_details = QToolButton()
        self.load_error_details.setText("details\u2026")
        self.load_error_details.setToolTip("the whole error, and a bug report")
        self.load_error_details.clicked.connect(self.show_load_error)
        dismiss = QToolButton()
        dismiss.setText("\u00d7")
        dismiss.setToolTip("hide this message")
        dismiss.clicked.connect(self.load_error.hide)
        row.addWidget(self.load_error_text, 1)
        row.addWidget(self.load_error_details)
        row.addWidget(dismiss)
        column.addWidget(self.load_error)
        self.load_error.hide()
        self._load_failure: Optional[tuple] = None
        self._load_error_window = None
        self.setCentralWidget(central)
        self.setAcceptDrops(True)          # a file dropped here: `open_dropped`
        # tall on purpose: with a file open the Render tab's own content wants
        # some 700 px, and scrolling for the display settings every time is
        # what one notices.  Capped so it still fits a laptop screen.
        screen = QApplication.primaryScreen()
        height = screen.availableGeometry().height() - 60 if screen else 900
        self.resize(self._tabs_width(), max(640, min(1000, height)))

        # Loading runs in a LoadTask; these hold its state.  `_load_locked` is
        # everything that would change the table under a running one.
        self._queue: List[tuple] = []
        self._pending_images: List[Path] = []     # dropped, opened once loading ends
        self._task: Optional[LoadTask] = None
        self._loading_name = ""
        self._loading = False
        self._loading_stage = ""
        self._elapsed = QElapsedTimer()
        self._elapsed.start()
        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._tick)

        menu = _add_menu(self.menuBar(), "File", self)
        # one Open for tables, images and stacks, sorted as a drop is; whether
        # a table replaces the open ones or joins them is asked when there are
        # any -- an Add file... and an Open image... beside it were three
        # entries for one thing, and the wrong one replaced a table
        open_ = self._action(menu, "Open...", QKeySequence.Open, lambda: self.open())
        save = self._action(menu, "Save", QKeySequence.Save, self.save)
        save_as = self._action(menu, "Save as...", QKeySequence.SaveAs, self.save_as)
        self._action(menu, "Session name...", None, self.rename_session)
        self._action(menu, "File properties...", None, self.edit_file_properties).setToolTip(
            "the open files' frame times: what turns frames into seconds")
        # a lambda: `triggered` hands on `checked` (False), which would land
        # in `start` and make the offer without starting anything to take it
        self.new_window_action = self._action(menu, "Open in a new window",
                                              "Ctrl+Shift+N",
                                              lambda: self.open_in_new_window())
        self.new_window_action.setToolTip("the session as it is now -- table, layers, "
                                    "filters, ROIs -- in a new instance of the "
                                    "program; a live fit carries on here")
        menu.addSeparator()
        # the tabs, their plugins and their values (`smappy.workspace`), in a
        # submenu of their own: a "Save GUI" beside Save left one guessing
        # which of the two kept the data
        self.workspace_menu = _add_menu(menu, "Workspace", self)
        self.workspace_menu.setToolTipsVisible(True)
        for text, slot, tip in (
                ("Save", self.save_gui, "overwrite the workspace file that is loaded"),
                ("Save as...", self.save_gui_as,
                 "the tabs, their plugins and their values, to a new file"),
                ("Open...", self.load_gui,
                 "a saved workspace: its tabs and values replace these"),
                ("Recent", None, None),
                ("Reset", self.load_default_gui,
                 "the shipped workspace, now and at the next start"),
                (None, None, None),
                ("From data file...", self.restore_state_from_file,
                 "the workspace a localization file was saved with")):
            if text is None:
                self.workspace_menu.addSeparator()
            elif slot is None:
                # filled as it opens: a file saved in another window, or
                # deleted since, is right without anything to keep in step
                self.recent_gui_menu = _add_menu(self.workspace_menu, text, self)
                self.recent_gui_menu.setToolTipsVisible(True)
                self.recent_gui_menu.aboutToShow.connect(self._fill_recent_guis)
            else:
                self._action(self.workspace_menu, text, None, slot).setToolTip(tip)
        self._action(menu, "Microscope manager...", None,
                     lambda: self.open_microscope_manager()).setToolTip(
            "the microscope entries: which devices set the pixel and the optics; "
            "EMU configurations to start them from")
        self._action(menu, "Preferences...", QKeySequence.Preferences,
                     self.open_preferences).setMenuRole(QAction.PreferencesRole)
        menu.addSeparator()
        quit_ = self._action(menu, "Quit", None, lambda: QApplication.instance().quit())
        quit_.setShortcuts(_keys(QKeySequence.Quit, "Ctrl+Q"))
        window_shortcuts(self, with_quit=False)
        edit = _add_menu(self.menuBar(), "Edit", self)
        self.undo_action = self._action(edit, "Undo", QKeySequence.Undo,
                                        lambda: self.session.undo())
        self.undo_menu = _add_menu(edit, "Undo steps", edit)
        self.redo_action = self._action(edit, "Redo", None, lambda: self.session.redo())
        self.redo_action.setShortcuts(_keys(QKeySequence.Redo, "Ctrl+Shift+Z"))
        self.redo_menu = _add_menu(edit, "Redo steps", edit)
        # Ctrl+Z undoes from the picture too, which is where one sees what to
        # undo; not from every window -- in the batch or calibration window
        # it would take back a step of a table that window is not about
        render.addAction(self.undo_action)
        render.addAction(self.redo_action)
        render.dropped = self.open_dropped      # the picture takes files too
        self._offers: list = []            # sessions handed to new instances
        self._offer_timer = QTimer(self, interval=500)
        self._offer_timer.timeout.connect(self._check_offers)
        self._load_locked = [open_, save, save_as, self.new_window_action,
                             self.undo_action,
                             self.redo_action, self.undo_menu.menuAction(),
                             self.redo_menu.menuAction()]
        edit.addSeparator()
        self.copy_action = self._action(edit, "Copy", QKeySequence.Copy, self.copy)
        self.copy_action.setToolTip("the selected text, else the picture of the "
                                    "window in front: the render, the 3D view, "
                                    "the ROI manager or a figure")
        # application-wide, so Ctrl+C in a figure window copies the figure.  A
        # text field still copies its own text: Qt offers it the key first
        self.copy_action.setShortcutContext(Qt.ApplicationShortcut)
        self._action(edit, "Copy rendered image", None,
                     lambda: self.copy(self.render_window))
        edit.setToolTipsVisible(True)
        # the window Copy means when it is chosen from this menu, which makes
        # the control window the active one: the last picture that was
        self._picture_window: Optional[QWidget] = None
        QApplication.instance().focusWindowChanged.connect(self._on_focus_window)
        view = _add_menu(self.menuBar(), "View", self)
        self.view3d_window = None
        self.calibration_window = None
        self.psf_calibration_window = None
        # the render window, brought to the front; Ctrl+2 beside the 3D
        # view's Ctrl+3.  Reset view is its button, Ctrl+0 its key from either
        # window (`RenderToolBar`); a menu item as well said it twice
        self._action(view, "2D view", "Ctrl+2", self.show_2d)
        self._action(view, "3D view", "Ctrl+3", self.toggle_3d)
        render.toolbar.view3d_button.clicked.connect(self.toggle_3d)
        self.plugins_menu = _add_menu(self.menuBar(), "Plugins", self)
        self._plugin_windows: dict = {}
        # From the scan rather than from imports, so the menu costs nothing at
        # start; again when it is opened, so a plugin dropped into the folder
        # is in it without a restart.  Filled now as well: a Mac leaves an
        # empty menu out of the menu bar, and this one was never seen there.
        self.batch_window = None
        self._fill_plugins_menu()
        self.plugins_menu.aboutToShow.connect(self._fill_plugins_menu)
        # The per-region steps and the bead calibration are not in this menu:
        # each has one way in where it is used (the Regions tab; the Localize
        # tab's section).  The ROI manager is a window one looks at, as the 3D
        # view is, so it is in View beside it; Ctrl+R from this window and
        # the picture.
        manager = QAction("ROI manager", self, triggered=self._open_manager)
        manager.setShortcut("Ctrl+R")
        manager.setShortcutContext(Qt.WindowShortcut)
        view.addAction(manager)
        self.addAction(manager)
        render.addAction(manager)
        # a plugin that would change a table a fit is still appending to
        # offers to run in a new instance instead (`PluginPanel._ask_live`)
        session.window_openers["new window"] = \
            lambda run=None: self.open_in_new_window(run=run)
        # the Localize tab's Bead calibration section (`Plugin.window`): the
        # window asks single or dual itself, and opens as it was left
        session.window_openers["bead calibration"] = \
            lambda settings=None: self.open_calibration()
        # *PSF calibration*: uipsf's pupil model, a window of its own (`calibrate.psf_window`)
        session.window_openers["psf calibration"] = \
            lambda settings=None: self.open_psf_calibration()
        # the File tab's Load line (`Plugin.window`): File > Open, or Add
        # *Load*'s two buttons (`plugins.file.Open`): each says what it does,
        # so neither asks, as File > Open does
        session.window_openers["open files"] = \
            lambda settings=None: self.open(append=False)
        session.window_openers["add files"] = \
            lambda settings=None: self.open(append=True)
        # a fitter whose file's camera is not in the database offers this
        # (`Plugin.describe`), opened on the file that fitter is set to
        session.window_openers["camera editor"] = self.open_camera_editor
        # what a file opener's found microscope offers (`gui.microscope_view`)
        session.window_openers["microscope manager"] = \
            lambda settings=None: self.open_microscope_manager()
        # a written simulation offers this (`Simulate.describe`), with the
        # file's path: into a fitter as a dropped stack goes
        session.window_openers["fit file"] = self.fit_file
        session.window_openers["fit simulation"] = self.fit_simulation
        # Line Profile with no line ROI offers this (`LineProfile.describe`):
        # the render window, drawing a line
        session.window_openers["draw line"] = lambda settings=None: self.draw_line()
        # the 3D window's *Make a movie of this...*: its rock or spin as the
        # Render tab's player's view path (`ScanPlayer.take_path`)
        session.window_openers["view path"] = self.open_view_path
        # Transform with given settings: Register layers' *apply*, with the
        # named transformation it measured (`transform.settings_for`)
        session.window_openers["transform"] = self.open_transform
        # a plugin that needs seconds the files cannot give (`frame_time_notice`)
        session.window_openers["file properties"] = \
            lambda settings=None: self.edit_file_properties()
        # the picture windows' *Save image...*: Export Image, set to the
        # window it was asked from (`plugins.file.image_settings_for`)
        session.window_openers["export image"] = self.open_export_image
        # a plugin that chooses beads (`Plugin.bead_view`): its notice's
        # *Select beads...*; the window answers into the panel that opened it
        def open_bead_selector(settings=None, plugin=None, reply=None):
            self.open_bead_selector(settings, plugin, reply)
        open_bead_selector.replies = True
        session.window_openers["bead selector"] = open_bead_selector
        # an AI assistant in the user's own client, reaching this session
        # through `smappy-mcp --attach` (`assistant_server`, `assistant_pane`):
        # nothing of it is imported until it is asked for, and it listens only
        # when allowed.  Everyday it is the render toolbar's button; the menu
        # is for starting it and its settings
        assistant = _add_menu(self.menuBar(), "Assistant", self)
        assistant.setToolTipsVisible(True)
        self.assistant_controls = None
        self._action(assistant, "Start assistant", "Ctrl+Shift+A",
                     lambda: self._assistant().start()).setToolTip(
            "start the assistant chosen in the settings, connected to this session, "
            "beside the picture")
        self.allow_assistant = self._action(
            assistant, "Allow connections", None,
            lambda on: self._assistant().allow(on))
        self.allow_assistant.setCheckable(True)
        self.allow_assistant.setToolTip("let an assistant you start yourself use this "
                                        "session (smappy-mcp --attach)")
        self._action(assistant, "Proposals", None,
                     lambda: self._assistant().show_proposals()).setToolTip(
            "what the assistant proposed, under its button in the picture's window")
        settings = self._action(assistant, "Settings...", None,
                                lambda: self._assistant().show_settings())
        settings.setToolTip("which assistant to start and where, the folder it may "
                            "write into")
        # left where it is: a Mac's Qt moves an item called "Settings..." into
        # the application menu as its Preferences, where nobody looked for the
        # assistant's -- and where File > Preferences... belongs
        settings.setMenuRole(QAction.NoRole)
        from .assistant_pane import AssistantButton
        self.assistant_button = AssistantButton(self._assistant, host=render.view)
        render.toolbar.add_right(self.assistant_button)
        help_ = _add_menu(self.menuBar(), "Help", self)
        self._action(help_, "Plugin documentation", None, self.show_plugin_docs)
        # the tutorials are videos on the published site, rebuilt from the
        # program on every push (`.github/workflows/tutorials.yml`): in the
        # system's browser, which plays them, not in the Help window
        self._action(help_, "Tutorials", None, self.show_tutorials).setToolTip(
            f"short videos of the program in use, in your browser ({TUTORIALS_URL})")
        help_.addSeparator()
        self._action(help_, "Save a bug report...", None, self.save_bug_report)
        self._action(help_, "Show the log folder", None, self.show_log_folder)
        help_.addSeparator()
        from .updates import Updates
        self.updates = Updates(self)     # started by `main`, so no test asks PyPI
        self._action(help_, "Check for updates...", None, self.updates.check_now)
        help_.addSeparator()
        # AboutRole: on a Mac Qt moves it into the application's own menu
        # (*SMAPpy > About SMAPpy*), where a Mac user looks for it
        about = self._action(help_, "About SMAPpy", None, self.show_about)
        about.setMenuRole(QAction.AboutRole)
        if sys.platform == "darwin":
            # A Mac has one menu bar, the front window's: with the picture in
            # front -- most of the time -- it was empty, and the Plugins menu
            # was nowhere to be found.  A menu bar with no parent is the one
            # shown for a window without its own; the same menus go in it.
            self._shared_menus = QMenuBar(None)
            for action in self.menuBar().actions():
                if action.menu() is not None:
                    self._shared_menus.addMenu(action.menu())
        QApplication.instance().aboutToQuit.connect(self.stop_loading)
        QApplication.instance().aboutToQuit.connect(self.release_offers)
        QApplication.instance().aboutToQuit.connect(self._stop_assistant)
        session.gui_state_provider = self.plugin_state
        session.on_change(self._on_session)
        self._on_session("locs")

    #: tabs the window keeps room for though the GUI may not have them yet
    #: (off in the shipped one, for now), so turning one on does not bring
    #: the scroll arrows back
    ROOM_FOR_TABS = ("Process",)

    def _tabs_width(self) -> int:
        """The control column's width, or wider if its tabs and those of
        `ROOM_FOR_TABS` it lacks would not fit side by side: six names at
        CONTROL_WIDTH on Windows and Linux, wider on a Mac's own tabs."""
        bar = self.tabs.tabBar()
        names = [bar.tabText(i) for i in range(bar.count())]
        width = sum(bar.tabSizeHint(i).width() for i in range(bar.count()))
        metrics = bar.fontMetrics()
        if names:
            # a tab's padding and frame: what its width has beyond its name
            frame = max(bar.tabSizeHint(i).width() - metrics.horizontalAdvance(n)
                        for i, n in enumerate(names))
            width += sum(metrics.horizontalAdvance(n) + frame
                         for n in self.ROOM_FOR_TABS if n not in names)
        return max(CONTROL_WIDTH, width + 8)

    def show_2d(self) -> None:
        """The render window shown and in front, wherever it was."""
        window = self.render_window
        if window.isMinimized():
            window.showNormal()
        window.show()
        window.raise_()
        window.activateWindow()

    def _assistant(self):
        """The assistant's controls, made the first time they are asked for."""
        if self.assistant_controls is None:
            from .assistant_pane import AssistantControls
            from .assistant_server import AssistantServer, RenderViewer
            server = AssistantServer(self.session, RenderViewer(self.render_window.view), self)
            controls = AssistantControls(
                self.session, server, self, open_plugin=self.open_plugin_window,
                load=lambda paths, append: self.load_paths(paths, append=append),
                beside=self.render_window)
            server.assistant.windows = self._assistant_windows()
            controls.find_panel = self._assistant_panel
            controls.saved_bounds = self.saved_terminal
            controls.changed.connect(
                lambda: self.allow_assistant.setChecked(server.listening))
            self.assistant_button.attach(controls)
            self.assistant_controls = controls
        return self.assistant_controls

    def _assistant_panel(self, path: str):
        """The panel of ``path`` where it is pinned -- its tab in front, its
        section open -- for the assistant's run to be seen where the user
        would make it; None if it is pinned nowhere."""
        pinned = self._pinned(path)
        if pinned is None:
            return None
        index, tab, instance = pinned
        self.tabs.setCurrentIndex(index)
        tab.open_section(instance.id)
        slot = tab.slots.get(instance.id)
        return slot.ensure() if slot is not None else None

    def _assistant_windows(self) -> dict:
        """The windows the assistant's `open_window` opens, by the names the
        menus give them; each takes an ROI's id (only the ROI manager uses
        it).  Not *Preferences* or *File properties*: modal dialogs, which
        would hold the call until they were closed."""
        def roi_manager(roi_id=None):
            self._open_manager()
            manager = getattr(self.roi_tab, "manager", None)
            if roi_id is not None and manager is not None:
                manager.select(roi_id)
        return {"ROI manager": roi_manager,
                "3D view": lambda roi_id=None: self.show_3d(),
                "batch window": lambda roi_id=None: self.open_batch(),
                "bead calibration": lambda roi_id=None: self.open_calibration(),
                "PSF calibration": lambda roi_id=None: self.open_psf_calibration(),
                "microscope manager": lambda roi_id=None: self.open_microscope_manager(),
                "camera editor": lambda roi_id=None: self.open_camera_editor(),
                "help": lambda roi_id=None: self.show_plugin_docs()}

    def listen_if_asked(self) -> None:
        """*allow connections when SMAPpy starts*, from the settings."""
        from .assistant_pane import preferences
        if preferences()["allow_at_start"]:
            self._assistant().allow(True)

    def _stop_assistant(self) -> None:
        if self.assistant_controls is not None:
            self.assistant_controls.server.stop()

    def show_about(self) -> None:
        from .about import show_about
        show_about(self)

    def show_tutorials(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(TUTORIALS_URL))

    def show_plugin_docs(self) -> None:
        from .help_window import show_help
        show_help(None, self)

    def open_batch(self) -> None:
        """The batch window: a chain over many files, run as a subprocess.

        Kept once made, so a job being assembled survives closing the window;
        a result it produced can be opened in this session.
        """
        if self.batch_window is None:
            from .batch_window import BatchWindow
            self.batch_window = BatchWindow(
                open_result=lambda path: self.load_paths([path], reset_view=True),
                parent=self)
        self.batch_window.refresh_chains()
        self.batch_window.show()
        self.batch_window.raise_()

    # --------------------------------------------------------------- tabs
    def header_for(self, name: str) -> Optional[QWidget]:
        """A "special" tab is an ordinary one with controls above the list.

        Which is why ROImanager and Localize need no class of their own.
        """
        if name == "roi":
            self.roi_tab = ROIHeader(self.session, self.render_window.view)
            return self.roi_tab
        # the Localize tab has no header: the camera is part of each fitter's microscope
        # control (`gui.microscope_form`), the camera editor a button of the microscope manager
        return None

    def open_camera_parameters(self) -> None:
        """Show the camera behind the fit that is set up in this tab.

        Hidden until asked for: the camera is normally recognised and its
        numbers are its own.  This is for when a conversion looks wrong, or
        nothing was recognised at all -- and a wrong conversion is invisible
        in the result, so the only way to see it is to look at where it came
        from.
        """
        from .camera_view import open_for
        panel = self._fitting_panel()
        if panel is None:
            QMessageBox.information(
                self, "Camera parameters",
                "open a fitter in the Localize tab first: the camera is read "
                "from the acquisition it is set to fit.")
            return
        try:
            settings = panel.form.value()
        except ValueError as error:
            QMessageBox.information(self, "Camera parameters",
                                    f"the fitter's settings do not read: {error}")
            return
        self._camera_window = open_for(panel.plugin, settings, self,
                                       getattr(self, "_camera_window", None))
        self._camera_window.open_editor = self.edit_camera
        self._camera_window.open_manager = self.open_microscope_manager

    def open_microscope_manager(self) -> None:
        """The microscope manager, made once; a saved entry is read again by every fitter, as a
        saved camera is."""
        from .microscope_editor import MicroscopeManager
        if getattr(self, "_microscope_manager", None) is None:
            self._microscope_manager = MicroscopeManager(parent=self)
            self._microscope_manager.saved.connect(lambda name: self._camera_saved(name))
        self._microscope_manager.show()
        self._microscope_manager.raise_()

    def open_camera_editor(self, settings=None) -> None:
        """The camera editor, on the file a fitter is set to.

        `settings` are the fitter's that asked (`Plugin.describe`); without them
        the first open fitter's are used, and without a fitter the editor
        opens on no file at all, for typing an entry in by hand.
        """
        if settings is None:
            panel = self._fitting_panel()
            if panel is not None:
                try:
                    settings = panel.form.value()
                except ValueError:
                    settings = None
        resolved = None
        if settings is not None:
            from ..plugins.fit import camera_resolution
            resolved = camera_resolution(settings)
        if resolved is None:
            self.edit_camera({}, None)
        else:
            self.edit_camera(resolved.metadata, resolved.camera)

    def edit_camera(self, metadata, camera=None) -> None:
        """Show the camera editor on these tags, starting from `camera`."""
        from .camera_editor import open_editor
        window = getattr(self, "_camera_editor", None)
        self._camera_editor = open_editor(metadata, camera, self, window)
        if window is None:
            self._camera_editor.saved.connect(self._camera_saved)

    def _camera_saved(self, name: str) -> None:
        """A camera went into the database: what read the old one reads again."""
        for tab in self.plugin_tabs:
            for panel in tab.panels():
                panel.refresh_derived()
        window = getattr(self, "_camera_window", None)
        if window is not None and window.isVisible():
            window.refresh()
        psf = getattr(self, "psf_calibration_window", None)
        if psf is not None:
            psf._show_optics()                  # a saved microscope changes what its stacks resolve to

    def _fitting_panel(self):
        """The first open panel that knows how to resolve a camera."""
        for tab in self.plugin_tabs:
            for panel in tab.panels():
                if hasattr(panel.plugin, "resolution"):
                    return panel
        return None

    def build_tabs(self) -> None:
        """(Re)build the tab strip from the workspace."""
        current = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else ""
        for tab in self.plugin_tabs:                # keep the values before the
            tab.save_values()                       # widgets go
        while self.tabs.count():
            self.tabs.removeTab(0)
        self.plugin_tabs = []
        for tab in self.workspace.tabs:
            if tab.kind == "render":
                widget = RenderTab(self.session, self.render_window.view)
            else:
                widget = PluginTab(tab, self.session, header=self.header_for(tab.header))
                self.plugin_tabs.append(widget)
            self.tabs.addTab(widget, tab.name)
        # a rebuild keeps the tab one is on; a start is on File, whatever
        # tab the GUI was saved from
        wanted = current or "File"
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == wanted:
                self.tabs.setCurrentIndex(i)
                break
        saved = self.workspace.layout.get("open", {})
        for tab, widget in zip(self.workspace.tabs, self._tab_widgets()):
            if not isinstance(widget, PluginTab):
                continue
            opened = saved.get(tab.name)
            if not opened and tab.name not in saved:
                # a tab nobody has opened a section of opens on *Load*, if it
                # has one: the program starts on File, one click from a file
                opened = next((i.id for i in tab.instances
                               if i.plugin == "File/Load/Load"), None)
            if opened:
                widget.open_section(opened)

    def _tab_widgets(self) -> List[QWidget]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def save_bug_report(self) -> None:
        """The log, the crash stacks and what is open, zipped for an issue."""
        from .errors import save_bug_report
        save_bug_report(self, self.session)

    def show_log_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from .. import diagnostics
        folder = diagnostics.log_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def show_user_plugins(self) -> None:
        """The user's plugin folder, made ready (`user_plugins.ensure`)."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from .. import user_plugins
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_plugins.ensure())))

    def new_user_plugin(self) -> None:
        """A plugin of the user's own, copied from the template, and its folder.

        The folder is shown rather than the file opened: a ``.py`` opened by
        the system is run, on some of them, not edited.
        """
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        from .. import plugins, user_plugins
        title, ok = QInputDialog.getText(
            self, "New plugin", "What the plugin is called (for example: cluster sizes):")
        if not ok or not title.strip():
            return
        try:
            path = user_plugins.new(title)
        except (FileExistsError, ValueError) as error:
            QMessageBox.warning(self, "New plugin", str(error))
            return
        plugins.discover(force=True)
        QMessageBox.information(
            self, "New plugin",
            f"{path.name} is in your plugin folder, a working example to change "
            "into what you need: by hand, or by describing it to an AI assistant "
            "started in that folder.\n\nIt is in the Plugins menu under User, and "
            "a tab's + pins it.  Help > Plugin documentation, \"Your own plugins\", "
            "says more.")
        self.show_user_plugins()

    def open_preferences(self) -> None:
        before = [(t.name, t.kind) for t in self.workspace.tabs]
        PreferencesDialog(self.workspace, self).exec()
        if [(t.name, t.kind) for t in self.workspace.tabs] != before:
            self.build_tabs()

    # ---------------------------------------------------------- workspace
    def plugin_state(self) -> dict:
        """What goes into a localization file's `gui` group: the GUI as a
        file would have it, without where the windows were -- that belongs
        to this screen, not to the data."""
        state = self.collect_workspace().to_dict()
        state.pop("windows", None)
        return state

    def restore_state_from_file(self) -> None:
        """Take the tabs and their parameters from a file that carries them.

        Not done automatically on open: rearranging someone's tabs because they
        looked at a colleague's dataset would be a surprise.  What is restored
        has no file of its own, so *Workspace > Save* asks where to put it.
        """
        from ..io.hdf5 import load_gui_state
        start = str(self.session.path) if self.session.path else folders.start()
        path, _ = QFileDialog.getOpenFileName(self, "Workspace from data file",
                                              start, name_filter())
        if not path:
            return
        state = load_gui_state(path)
        if not state:
            QMessageBox.information(self, "Workspace from data file",
                                    "that file carries no workspace")
            return
        self.workspace = Workspace.from_dict(state)
        missing = self.workspace.prune(list(plugins.refs()))
        self.build_tabs()
        self.session.apply_default_filters(self.workspace.filters)
        self._update_title(f"from {Path(path).name}")
        if missing:
            QMessageBox.information(
                self, "Workspace from data file",
                "restored, without plugins that are not installed here:\n"
                + "\n".join(sorted(set(missing))))

    def collect_workspace(self) -> Workspace:
        """The workspace as it stands: values, order, and which section is open.

        Not which tab is in front, nor where the window is: a GUI starts on
        File, and the geometry belongs to the screen, not to the GUI.
        """
        opened = {}
        for tab, widget in zip(self.workspace.tabs, self._tab_widgets()):
            if isinstance(widget, PluginTab):
                widget.save_values()
                which = widget.open_instance()
                if which:
                    opened[tab.name] = which
        self.workspace.layout = {"open": opened}
        # the filters as they stand (`Session.carried_bounds`), so a GUI
        # opens the next file filtered as this one is
        self.workspace.filters = {f: list(b) for f, b in
                                  self.session.carried_bounds().items()}
        return self.workspace

    def _update_title(self, name: Optional[str] = None) -> None:
        """Say whose workspace this is, so nobody saves over somebody else's --
        and, once named, which session this is (`identity`)."""
        if name is not None:
            self._gui_name = name
        gui = getattr(self, "_gui_name", None) or \
            workspace_module.display_name(self.workspace.path)
        app = f"{identity.unsaved(self.session)}SMAPpy"
        if self.session.name or self.session.colour:
            text = identity.title(self.session, app)
            self.setWindowTitle(f"{text} (workspace: {gui})")
        else:
            self.setWindowTitle(f"{app} \u2014 {gui}")
        colour = identity.colour_hex(self.session)
        # a stripe as well as the circle: seen from across the room, and on a
        # title font that draws the circle in black
        self.statusBar().setStyleSheet(
            f"QStatusBar {{ border-top: 4px solid {colour}; }}" if colour else "")

    # ------------------------------------------------- a new instance
    def open_in_new_window(self, start: bool = True, run=None) -> Optional[object]:
        """Hand the session as it is now to a new instance (`smappy.handoff`).

        The copy is named and coloured as one -- and this session gets a
        colour too if it has none, so the two can be told apart at a glance.
        ``start=False`` makes the offer without starting the program, for a
        test to take.  ``run`` is ``(plugin, settings)`` for the new instance
        to run once it has the session (`PluginPanel`, during a live fit).
        Returns the offer, or None if there was nothing to hand.
        """
        from .. import handoff
        if not len(self.session.locs):
            QMessageBox.information(self, "Open in a new window",
                                    "there are no localizations to hand on")
            return None
        if not self.session.colour:
            self.session.rename(self.session.name, colour="blue")
        name, colour = handoff.snapshot_identity(self.session)
        view = self.render_window.view.view.viewRange()
        try:
            offer = handoff.offer(self.session, name=name, colour=colour,
                                  gui_state=self.plugin_state(), view=view,
                                  run=run)
        except Exception as e:
            QMessageBox.warning(self, "Open in a new window",
                                f"the session could not be handed on:\n{e}")
            return None
        self._offers.append(offer)
        self._offer_timer.start()
        if start:
            try:
                handoff.launch(offer.path)
            except OSError as e:
                offer.release()
                self._offers.remove(offer)
                QMessageBox.warning(self, "Open in a new window",
                                    f"the new window could not be started:\n{e}")
                return None
            self.statusBar().showMessage(f"opening {name} in a new window...", 8000)
        return offer

    def _check_offers(self) -> None:
        """Free each handed-over block once it is taken, or has waited too long."""
        try:
            for offer in list(self._offers):
                if offer.taken or offer.expired:
                    if not offer.taken:
                        self.statusBar().showMessage(
                            "the new window did not start; nothing was handed on", 8000)
                    offer.release()
                    self._offers.remove(offer)
            if not self._offers:
                self._offer_timer.stop()
        except Exception:                    # a timer slot must never raise
            traceback.print_exc()

    def release_offers(self) -> None:
        for offer in self._offers:
            offer.release()
        self._offers.clear()

    def take_handoff(self, path: str) -> None:
        """Open a session another instance handed over: what *Open in a new
        window* there starts this one with."""
        self._set_loading(True)
        self._loading_name = "the handed-over session"
        self._task = HandoffTask(str(path), self.session.group_settings, self)
        self._task.progress.connect(self._loading_progress)
        self._task.loaded.connect(self._took_handoff)
        self._task.failed.connect(self._load_failed)
        self._task.finished.connect(self._task_finished)
        self._elapsed.restart()
        self._loading_progress("taking over")
        self._task.start()

    def _took_handoff(self, locs, state, grouped) -> None:
        from .. import handoff
        handoff.apply_state(self.session, locs, state, grouped)
        gui = state.get("gui_state")
        if gui:
            self.workspace = Workspace.from_dict(gui)
            self.workspace.prune(list(plugins.refs()))
            # for files to come: the layers came over as they were
            self.session.default_filters = self.workspace.filters
            self.build_tabs()
            self._update_title("handed over")
        view = state.get("view")
        if view:
            from PySide6.QtCore import QRectF
            (x0, x1), (y0, y1) = view
            self.render_window.view.view.setRange(QRectF(x0, y0, x1 - x0, y1 - y0),
                                                  padding=0)
            self.render_window.view.schedule()
        else:
            self.render_window.view.reset()
        run = state.get("run")
        if run:
            # after the event loop has shown the session: the run's preflight
            # may ask something, and should ask it over the right picture
            QTimer.singleShot(0, lambda: self._run_handed_over(run))

    def _run_handed_over(self, run: dict) -> None:
        """Run what the other instance could not apply during its live fit."""
        try:
            from .. import plugins as registry
            window = self.open_plugin_window(run["plugin"])
            if window is None:
                return
            cls = registry.get(run["plugin"])
            window.panel.form.set(registry.settings_from(cls.Settings,
                                                         run.get("values") or {}))
            window.panel.start_run()
        except Exception:                    # a singleShot callable must not raise
            traceback.print_exc()

    def edit_file_properties(self) -> None:
        """The open files' frame times (`dialogs.FilePropertiesDialog`)."""
        from .dialogs import FilePropertiesDialog
        if not self.session.files:
            QMessageBox.information(self, "File properties", "no file is open")
            return
        dialog = FilePropertiesDialog(self.session, self)
        if dialog.exec():
            dialog.apply()

    def rename_session(self) -> None:
        """Ask for the session's name and colour (`identity.NameDialog`)."""
        answer = identity.NameDialog.ask(self, self.session)
        if answer is not None:
            self.session.rename(*answer)

    def save_gui_to(self, path) -> Path:
        """Write the GUI to ``path``, make it the loaded one and the next start.

        The windows' places go with it: the one time they are kept.
        """
        workspace = self.collect_workspace()
        workspace.windows = self.window_geometries()
        written = workspace.save(path)
        workspace_module.remember(written)
        self._update_title()
        return written

    def save_gui(self) -> None:
        """Overwrite the loaded GUI file; with none loaded, ask for one.

        The shipped default is never written into: it lives in the package,
        which may be read-only and is the same for everybody.
        """
        if self.workspace.path is None:
            self.save_gui_as()
            return
        self._write_gui(self.workspace.path)

    def save_gui_as(self) -> None:
        start = self.workspace.path or (workspace_module.guis_dir()
                                        / f"my{workspace_module.GUI_SUFFIX}")
        start.parent.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Save workspace", str(start),
                                              "Workspace (*.yaml)")
        if path:
            self._write_gui(path)

    def _write_gui(self, path) -> None:
        try:
            self.save_gui_to(path)
        except OSError as e:
            QMessageBox.warning(self, "Save workspace", f"could not write {path}:\n{e}")

    def load_gui(self) -> None:
        start = self.workspace.path.parent if self.workspace.path else \
            workspace_module.guis_dir()
        path, _ = QFileDialog.getOpenFileName(self, "Open workspace", str(start),
                                              "Workspace (*.yaml)")
        if path:
            self.load_gui_from(path)

    def load_gui_from(self, path) -> None:
        """Replace the tabs and values with those of a GUI file.

        A file that does not read gives the shipped default, and is not
        remembered, so the next start is not a broken one.
        """
        self.workspace = workspace_module.load(path)
        missing = self.workspace.prune(list(plugins.refs()))
        self.tabs.clear()               # start on File, not on the tab one was on
        self.build_tabs()
        self.session.apply_default_filters(self.workspace.filters)
        workspace_module.remember(self.workspace.path)
        self._update_title()
        self.apply_windows()
        if path and self.workspace.path is None:
            QMessageBox.information(self, "Open workspace",
                                    f"{path} is not a workspace file; the default is loaded")
        elif missing:
            QMessageBox.information(
                self, "Open workspace",
                "loaded, without plugins that are not installed here:\n"
                + "\n".join(sorted(set(missing))))

    def _fill_recent_guis(self) -> None:
        """The workspaces last saved or loaded, newest first, one click each."""
        menu = self.recent_gui_menu
        menu.clear()
        paths = workspace_module.recent()
        names = [workspace_module.display_name(p) for p in paths]
        for path, name in zip(paths, names):
            if names.count(name) > 1:          # two "my" in different folders
                name = f"{name} ({path.parent.name})"
            action = menu.addAction(name)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _=False, p=path: self.load_gui_from(p))
        if not paths:
            menu.addAction("(none saved yet)").setEnabled(False)

    def load_default_gui(self) -> None:
        """The shipped GUI, and the shipped GUI at the next start too."""
        self.load_gui_from(None)

    def _windows(self) -> Dict[str, QWidget]:
        """The windows whose places a workspace keeps, by name."""
        windows = {"control": self, "render": self.render_window}
        if self.view3d_window is not None:
            windows["3d"] = self.view3d_window
        return windows

    def window_geometries(self) -> Dict[str, str]:
        """Where the windows are now; the 3D view only once it was opened,
        the assistant's terminal while it is open and can be asked (a Mac's
        Terminal, `launch.terminal_bounds`), as "left,top,right,bottom"."""
        from ..assistant import launch
        geometries = {name: bytes(window.saveGeometry().toBase64()).decode()
                      for name, window in self._windows().items()}
        terminal = launch.terminal_bounds()
        if terminal is not None:
            geometries["terminal"] = ",".join(str(v) for v in terminal)
        return geometries

    def apply_windows(self) -> None:
        """Put the windows where the loaded workspace had them; a workspace
        saved without them leaves them where they are.  The terminal is put
        there when it is started, and moved there if it is open now."""
        for name, window in self._windows().items():
            self._place(name, window)
        terminal = self.saved_terminal()
        if terminal is not None:
            from ..assistant import launch
            launch.set_terminal_bounds(terminal)

    def saved_terminal(self):
        """The loaded workspace's place of the assistant's terminal, (left,
        top, right, bottom), or None."""
        try:
            values = tuple(int(v) for v in self.workspace.windows.get("terminal", "").split(","))
        except ValueError:
            return None
        return values if len(values) == 4 else None

    def _place(self, name: str, window: QWidget) -> None:
        geometry = self.workspace.windows.get(name)
        if geometry:
            from PySide6.QtCore import QByteArray
            window.restoreGeometry(QByteArray.fromBase64(geometry.encode()))

    def _open_manager(self) -> None:
        if self.roi_tab is not None:
            self.roi_tab.open_manager()

    # ------------------------------------------------------------ plugins
    def _fill_plugins_menu(self) -> None:
        """The whole plugin tree as nested menus; picking one opens a window.

        Built from `PluginRef`s, as the chooser is, so opening the menu
        imports no plugin.  Rebuilt on every show rather than cached: the scan
        is cheap and a stale menu is worse than a rebuilt one.
        """
        from .. import plugins as registry

        menu = self.plugins_menu
        # clear() leaves the submenus, which the menu owns: the last show's
        # would pile up, one tree per opening
        for old in menu.findChildren(QMenu, options=Qt.FindDirectChildrenOnly):
            old.deleteLater()
        menu.clear()
        self._action(menu, "Find a plugin...", "Ctrl+Shift+P", self.choose_plugin_window)
        menu.actions()[-1].setToolTip("search every plugin by its name or a word "
                                      "from its documentation, and open it in a window")
        menu.setToolTipsVisible(True)
        # a chain over many files: beside the plugins it runs, not in a Tools
        # menu of its own that held nothing else
        self._action(menu, "Batch...", None, self.open_batch)
        # the user's own plugins (`smappy.user_plugins`): where they are
        # written, by hand or by an assistant, and appear under User
        self._action(menu, "New plugin...", None, self.new_user_plugin)
        menu.actions()[-1].setToolTip("start a plugin of your own from an example, "
                                      "in your plugin folder")
        self._action(menu, "Show my plugin folder", None, self.show_user_plugins)
        menu.actions()[-1].setToolTip("the folder whose plugins appear under User; "
                                      "an AI assistant started there knows how to "
                                      "write one (its AGENTS.md)")
        menu.addSeparator()
        groups = {"": menu}

        def group_for(path: str):
            if path in groups:
                return groups[path]
            head, _, leaf = path.rpartition("/")
            parent = group_for(head)
            groups[path] = _add_menu(parent, leaf, parent)
            return groups[path]

        # forced: a file saved into a plugin folder since the last opening is
        # in the menu without a restart, and the scan parses, never imports
        registry.discover(force=True)
        for path, ref in sorted(registry.refs().items()):
            action = group_for(ref.group).addAction(ref.name)
            action.setToolTip(ref.description or path)
            action.triggered.connect(lambda _=False, p=path: self.open_plugin_window(p))
        if len(groups) == 1:
            menu.addAction("no plugins found").setEnabled(False)

    def choose_plugin_window(self) -> None:
        from .chooser import choose_plugin
        path = choose_plugin(self, title="Open a plugin")
        if path:
            self.open_plugin_window(path)

    def open_transform(self, settings=None) -> None:
        """*Transform* in its window, set to ``settings`` when given."""
        from ..plugins.transform import PATH
        window = self.open_plugin_window(PATH)
        if window is not None and settings is not None:
            window.panel.form.set(settings)

    def open_export_image(self, settings=None) -> None:
        """*Export Image* in its window, for the picture window ``settings``
        names.  Opened the first time it takes them whole; after that it
        keeps what was chosen in it -- what to save, the bars, the pixel --
        and takes only the window, and the file if it had none."""
        from ..plugins.file import IMAGE_PATH
        fresh = IMAGE_PATH not in self._plugin_windows
        window = self.open_plugin_window(IMAGE_PATH)
        if window is None or settings is None:
            return
        if not fresh:
            current = window.panel.form.value()
            current.view = settings.view
            current.path = current.path or settings.path
            settings = current
        window.panel.form.set(settings)

    def open_plugin_window(self, path: str):
        """One plugin in a window of its own, without pinning it to a tab.

        Kept, so that reopening finds the settings that were typed into it.
        """
        from .. import plugins as registry
        from .plugin_panel import PluginWindow

        window = self._plugin_windows.get(path)
        try:
            cls = registry.get(path)
        except Exception as e:
            QMessageBox.warning(self, "plugin", f"{path} could not be opened:\n{e}")
            return None
        if window is not None and type(window.panel.plugin) is not cls:
            # the file changed and was imported again (a user's plugin being
            # written): a window of the old class would run the old code
            window.close()
            window.deleteLater()
            window = None
        if window is None:
            try:
                window = PluginWindow(cls, self.session, self)
            except Exception as e:
                QMessageBox.warning(self, "plugin", f"{path} could not be opened:\n{e}")
                return None
            window_shortcuts(window)
            self._plugin_windows[path] = window
        window.show()
        window.raise_()
        return window

    def _action(self, menu, text, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # ---------------------------------------------------------------- copy
    def _on_focus_window(self, window) -> None:
        """Remember the last window that has a picture to copy."""
        widget = next((w for w in QApplication.topLevelWidgets()
                       if window is not None and w.windowHandle() == window), None)
        if widget is not None and hasattr(widget, "clipboard_image"):
            self._picture_window = widget

    def _copy_window(self) -> QWidget:
        """The window Copy is about: the one in front if it has a picture,
        else the last one that did, else the render window."""
        active = QApplication.activeWindow()
        if active is not None and active is not self:
            return active
        last = self._picture_window
        try:
            if last is not None and last.isVisible():
                return last
        except RuntimeError:                      # closed and deleted since
            self._picture_window = None
        return self.render_window

    def copy(self, window: Optional[QWidget] = None) -> bool:
        """Copy what is in front: selected text, else that window's picture.

        Any window can take part by having a ``clipboard_image()`` that
        returns a QImage, or None when it has nothing to show.  The picture
        is put on the clipboard as an image, to paste into a slide or a
        notebook; saving it at a chosen pixel size is the Save menus' job.
        Returns whether anything was copied.
        """
        if window is None:
            focus = QApplication.focusWidget()
            if _copy_selected_text(focus):
                return True
            window = self._copy_window()
        grab = getattr(window, "clipboard_image", None)
        image = grab() if grab is not None else None
        if image is None or image.isNull():
            self.statusBar().showMessage("Nothing to copy here", 3000)
            return False
        QApplication.clipboard().setImage(image)
        self.statusBar().showMessage(
            f"Copied {window.windowTitle() or 'the picture'} "
            f"({image.width()} x {image.height()} px)", 3000)
        return True

    def _refresh_undo(self) -> None:
        """The menu says what it would undo, and lists the steps behind it.

        Word's arrangement: one entry for the last step, and a submenu where
        picking the n-th undoes n steps at once.  A plain menu cannot
        highlight the run above the cursor the way Word's dropdown does, so
        the labels carry the count instead.
        """
        loading = self._loading
        for action, menu, steps, verb, go in (
                (self.undo_action, self.undo_menu, self.session.undo_entries(),
                 "Undo", self.session.undo),
                (self.redo_action, self.redo_menu, self.session.redo_entries(),
                 "Redo", self.session.redo)):
            ready = bool(steps) and not loading
            action.setText(f"{verb} {steps[0].label}" if steps else verb)
            action.setToolTip(steps[0].text if steps else "")
            action.setEnabled(ready)
            menu.clear()
            menu.setToolTipsVisible(True)     # the plugin's own line, on hover
            menu.setEnabled(ready)
            for n, edit in enumerate(steps, start=1):
                entry = menu.addAction(edit.label if n == 1
                                       else f"{edit.label}  ({n})")
                entry.setToolTip(edit.text)
                # the count is bound here, as a default argument: a lambda
                # closing over the loop variable would undo the whole list
                # whichever entry was picked
                entry.triggered.connect(lambda checked=False, count=n, run=go: run(count))

    def open_view_path(self, path) -> None:
        """The Render tab in front, its player open on ``path``."""
        for index, widget in enumerate(self._tab_widgets()):
            if isinstance(widget, RenderTab):
                self.tabs.setCurrentIndex(index)
                widget.show_view_path(path)
                self.show()
                self.raise_()
                return

    def draw_line(self) -> None:
        """Start drawing a line ROI on the picture, which comes to the front."""
        self.render_window.show()
        self.render_window.raise_()
        self.render_window.activateWindow()
        self.render_window.view.start_drawing("line")

    def open_bead_selector(self, settings=None, plugin=None, reply=None) -> None:
        """The bead window (`gui.bead_window`) for ``plugin`` with ``settings``;
        one, reused, so that it opens where it was left."""
        if plugin is None or settings is None:
            return
        from .bead_window import BeadWindow
        window = getattr(self, "bead_window", None)
        if window is None:
            self.bead_window = window = BeadWindow(
                self.session, plugin, settings, reply,
                centre=self.render_window.view.center_on, parent=self)
            window.setWindowFlag(Qt.Window, True)
        else:
            window.open_for(plugin, settings, reply)
        window.show()
        window.raise_()

    def open_calibration(self, dual: Optional[bool] = None) -> None:
        """The calibration window: bead stacks in, a spline calibration out.

        A window of its own, like the 3D viewer, but in this process, so a
        saved calibration can go straight into the Spline 3D fitter.  ``dual``
        sets the mode; None leaves it as it was.
        """
        from ..calibrate.qt_gui import CalibrationWindow
        if self.calibration_window is None:
            self.calibration_window = CalibrationWindow(parent=self)
            self.calibration_window.calibrated.connect(self.use_calibration)
        if dual is not None:
            self.calibration_window.mode_box.setCurrentText(
                "Dual colour" if dual else "Single channel")
        self.calibration_window.show()
        self.calibration_window.raise_()

    def open_psf_calibration(self) -> None:
        """The PSF calibration window, made once and opened as it was left; what it saves goes to the
        fitter it belongs to, as the bead calibration's does."""
        from ..calibrate.psf_window import PSFCalibrationWindow
        if self.psf_calibration_window is None:
            self.psf_calibration_window = PSFCalibrationWindow(parent=self)
            self.psf_calibration_window.calibrated.connect(self.use_calibration)
            self.psf_calibration_window.open_manager = self.open_microscope_manager
        self.psf_calibration_window.show()
        self.psf_calibration_window.raise_()

    def use_calibration(self, path: str, dual: bool = False) -> None:
        """Put a fresh calibration into the fitter it belongs to.

        A dual-colour calibration is no use to the single-channel fitter, so
        which one gets it follows from the mode it was built in; an
        elliptical Gaussian's curves go to Gaussian 2D, *elliptical* ticked
        and its *after the fit* turning the widths into z.  The panel is
        built and its section opened on the way: a tab imports a plugin only
        when it is first looked at, and a calibration that lands in a plugin
        nobody has opened yet would land nowhere.
        """
        from ..astigmatic_z import is_calibration
        from ..fourpi import is_fourpi_calibration
        if is_fourpi_calibration(path):
            # a 4Pi model: its maps between the channels come with it
            wanted, values, name = ("Localize/Spline 4Pi", {
                "model.calibration": str(path), "model.channels.registration": "model"},
                "Spline 4Pi fitter")
        elif is_calibration(path):
            # the elliptical Gaussian's curves: for the fit that measures the
            # widths, which turns them into z once it is done
            wanted, values, name = (ASTIGMATIC_FITTER, {
                "model.elliptical": True, "finish.astigmatic_z": True,
                "finish.astigmatic.calibration": "file",
                "finish.astigmatic.path": str(path)}, "Gaussian 2D fitter")
        else:
            wanted = "Localize/Spline 3D 2C" if dual else "Localize/Spline 3D"
            values = {"model.calibration": str(path)}
            name = f"{wanted.split('/')[-1]} fitter"
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            slots = getattr(tab, "slots", None)
            if not slots:
                continue
            for instance in tab.tab.instances:
                if instance.plugin != wanted:
                    continue
                panel = slots[instance.id].ensure()
                if panel is None:
                    break
                self.tabs.setCurrentIndex(index)
                tab.open_section(instance.id)
                panel.form.set_values(values)
                self.statusBar().showMessage(f"the {name} now uses {Path(path).name}", 8000)
                return
        self.statusBar().showMessage(
            f"{Path(path).name} was saved, but no {name} is pinned to a tab - "
            "add one with + and set its calibration", 12000)

    def show_3d(self) -> None:
        if self.view3d_window is None:
            from .view3d import View3DWindow
            self.view3d_window = View3DWindow(self.session)
            window_shortcuts(self.view3d_window)
            QApplication.instance().aboutToQuit.connect(self.view3d_window.view.shutdown)
            # the button follows the window, however it was closed
            self.view3d_window.shown.connect(
                self.render_window.toolbar.view3d_button.setChecked)
            self.view3d_window.toolbar.set_border(identity.colour_hex(self.session))
            # beside the 2D picture, not over it: the two are looked at together
            place_beside(self.view3d_window, self.render_window)
            self._place("3d", self.view3d_window)   # where the workspace had it
        self.view3d_window.show()
        self.view3d_window.raise_()

    def toggle_3d(self) -> None:
        """Show the 3D view, or hide it if it is showing.  Hidden, not closed
        for good: the window is kept, so the view comes back as it was left."""
        if self.view3d_window is not None and self.view3d_window.isVisible():
            self.view3d_window.close()
        else:
            self.show_3d()


    def _on_session(self, what: str) -> None:
        if what == "scan":
            # the player's step: nothing here changes, and the counts and
            # the restyled toolbars cost the player a few ms a step
            return
        if what in ("layer", "layers", "locs", "roi") and not self.render_window.isVisible():
            self.render_window.show()      # closed by accident: a change wants it back
        self._refresh_undo()
        names = self.session.file_names()
        name = (names[0] if len(names) == 1 else f"{len(names)} files") if names else "no file"
        n = len(self.session.locs)
        shown = self._shown_text()
        meta = self.session.locs.metadata
        # x_nm on a scale nobody gave: worth seeing whenever the table is
        assumed = (f"; nm on an assumed {meta.get('pixelsize_nm', 0):g} nm pixel"
                   if meta.get("pixelsize_assumed") else "")
        self.statusBar().showMessage(f"{name}: {n} localizations"
                                     + (f", {shown}" if shown else "") + assumed)
        if what in ("name", "locs", "unsaved"):
            self._update_title()
        if self.session.name or self.session.colour:
            name = self.session.display_name or name
        self.render_window.setWindowTitle(
            f"{identity.mark(self.session)} {identity.unsaved(self.session)}"
            f"SMAPpy - {name}".strip()
            + (f" - {shown}" if shown else ""))
        colour = identity.colour_hex(self.session)
        self.render_window.toolbar.set_border(colour)
        if self.view3d_window is not None:
            self.view3d_window.toolbar.set_border(colour)

    def _shown_text(self) -> str:
        """What the render window actually draws: kept by each visible layer.

        The table's length says what was fitted, not what is on screen -- the
        filter and the grouping both stand between them, and a picture that
        looks empty needs the number that explains it.  Per layer when there
        are several, since two layers over one table would otherwise count
        the same localizations twice.
        """
        layers = [l for l in self.session.layers if not l.is_image and l.visible]
        if not layers or not len(self.session.locs):
            return ""

        def one(layer) -> str:
            kept, of = len(layer.filter), len(layer.locs)
            if layer.state.waiting:
                return "merging blinks"
            text = f"{kept:,} of {of:,}" + (" blinks" if layer.grouped else "") + " shown"
            hidden = 1.0 - kept / of if of else 0.0
            # said in so many words: a bound set long ago hides data silently
            return text + (f", filters hide {hidden:.0%}" if hidden >= 0.005 else "")

        if len(layers) == 1:
            return one(layers[0])
        return "; ".join(f"{l.name}: {one(l)}" for l in layers)

    def open(self, append: Optional[bool] = None) -> None:
        """*File > Open*: localization tables, images and stacks, sorted by
        `open_dropped` as a drop is -- a table loads, an image is a layer, a
        long stack goes to a fitter.  ``append`` None asks, when a table is
        open, whether the new ones replace it or join it.

        The reading and the linking happen in a `LoadTask`, one file at a time,
        so the window stays alive and says what it is doing.  Anything that
        would change the session underneath a running task is disabled while it
        runs; everything else -- panning, the filters, the other windows --
        keeps working on the file that is already open.
        """
        paths, _ = QFileDialog.getOpenFileNames(self, "Add" if append else "Open",
                                                folders.start(), open_filter())
        if paths:
            self.open_dropped(paths, append=append)

    def load_paths(self, paths, append: bool = False, reset_view: bool = False) -> None:
        """Queue files for the loader.  ``reset_view`` frames the first one,
        which is what starting with a file on the command line wants."""
        folders.remember(paths)      # however it came: dialog, drop, command line
        if paths:
            self.load_error.hide()   # the last failure is old news
        queued = 0
        for path in paths:
            loads = [{}]
            if self._needs_mapping(path):
                dialog = CsvMappingDialog(path, self)
                if dialog.exec() != QDialog.Accepted:
                    continue
                loads = [dialog.reader_args()]
            elif self._needs_parts(path):
                loads = self._ask_parts(path)
                if loads is None:
                    continue
            for args in loads:
                self._queue.append((str(path), args, append or queued > 0,
                                    reset_view and queued == 0))
                queued += 1
        self._start_next()

    @staticmethod
    def _needs_parts(path: str) -> bool:
        """A file that holds more than one thing to open: an `.msr`."""
        try:
            return "measurements" in reader_for(path).needs
        except ValueError:
            return False

    def _ask_parts(self, path):
        """Which measurements and images of an `.msr` (`MsrImportDialog`):
        the loader's arguments, one per file to open; None if cancelled.
        The images are opened once the files are."""
        from .msr_import import MsrImportDialog
        from ..io import msr
        try:
            dialog = MsrImportDialog(path, self)
        except Exception as e:                   # a damaged file is not a crash
            self._loading_name = Path(path).name
            self._load_failed(message_of(e), traceback.format_exc())
            return None
        if dialog.exec() != QDialog.Accepted:
            return None
        choice = dialog.choice()
        for im in choice["images"]:
            try:
                self._pending_images.append(msr.image(path, im))
            except Exception as e:
                self._loading_name = f"{Path(path).name}: {im.name}"
                self._load_failed(message_of(e), traceback.format_exc())
        if not choice["loads"] and self._task is None and not self._queue:
            self._open_pending_images()
        return choice["loads"]

    def _start_next(self) -> None:
        if self._task is not None:            # one at a time: each one replaces
            return                            # or appends to what the last left
        if not self._queue:
            self._set_loading(False)
            self._on_session("locs")          # back to "name: N localizations"
            self._open_pending_images()
            return
        path, args, append, reset_view = self._queue.pop(0)
        self._set_loading(True)
        self._loading_name = Path(path).name
        self._task = LoadTask(path, args, self)
        self._task.progress.connect(self._loading_progress)
        self._task.loaded.connect(
            lambda l, i, a=append, r=reset_view: self._loaded(l, i, a, r))
        self._task.failed.connect(self._load_failed)
        self._task.finished.connect(self._task_finished)
        self._elapsed.restart()
        self._loading_progress("loading")
        self._task.start()

    def _set_loading(self, on: bool) -> None:
        """While a file is being read, nothing may change the table under it."""
        self._loading = on
        for action in self._load_locked:
            action.setEnabled(False if on else True)
        if on:
            self._ticker.start(500)
        else:
            self._ticker.stop()
            self._refresh_undo()

    def _loading_progress(self, text: str) -> None:
        self._loading_stage = text
        self._tick()

    def _tick(self) -> None:
        """The stage plus a clock -- linking reports once and then runs for
        minutes, and a status bar that never changes reads as a hung window.

        The stage and the clock come first and the file name last, elided to
        whatever room is left: this window is 380 px wide, and with the name
        in front a long one filled the bar on its own, so the half that moves
        was cut off and the load looked stuck.
        """
        seconds = self._elapsed.elapsed() // 1000
        clock = f"{seconds // 60}:{seconds % 60:02d}"
        message = f"{self._loading_stage}... ({clock})"
        bar = self.statusBar()
        if self._loading_name:
            metrics = bar.fontMetrics()
            room = bar.width() - 24 - metrics.horizontalAdvance(message + "  ")
            name = metrics.elidedText(self._loading_name, Qt.ElideMiddle, max(room, 0))
            if name:
                message = f"{message}  {name}"
        bar.showMessage(message)

    def _loaded(self, locs, info, append: bool, reset_view: bool) -> None:
        self.session.add_file(locs, info, append=append)
        if reset_view:
            self.render_window.view.reset()

    def _load_failed(self, message: str, details: str = "") -> None:
        """Say it under the tabs, in red, with the traceback behind
        *details…*; a modal box here held up every file queued after it."""
        print(details or message)
        name = self._loading_name
        self._load_failure = (f"could not open {name}: {message}", details or message)
        self.load_error_text.setText(self._load_failure[0])
        self.load_error_text.setToolTip(self._load_failure[0])
        self.load_error.show()

    def show_load_error(self) -> None:
        if self._load_failure is None:
            return
        message, details = self._load_failure
        if self._load_error_window is not None:
            self._load_error_window.close()
        self._load_error_window = details_window(self, "could not open", message,
                                                 details, self.session)

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
        """A csv whose headers do not say where x and y are, and that no
        saved conversion knows (`io.conversions`)."""
        from ..io.conversions import conversion_for
        try:
            if "mapping" not in reader_for(path).needs:
                return False
        except ValueError:
            return False
        headers = csv_columns(Path(path))[0]
        conv = conversion_for(headers)
        guessed = set(guess_csv_mapping(headers).values()) \
            | set((conv.mapping if conv else {}).values())
        return not {"x_nm", "y_nm"} <= guessed

    # ---------------------------------------------------------------- drops
    def dragEnterEvent(self, event) -> None:
        if drops.accepts(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        _drop(event, self.open_dropped)

    def open_dropped(self, paths, append: Optional[bool] = None) -> None:
        """Files dropped on either main window, sorted by `drops.kind_of`.

        Localization files go to `load_paths` together, which is *File >
        Open* with all of them chosen at once: the first replaces the table,
        the rest are added to it.  A short TIFF (`drops.MAX_IMAGE_FRAMES`
        frames or fewer) or a PNG is an image layer, added to what is open.  A
        long stack goes into a fitter's *file* (`fit_file`); only one can, so
        the first.  Anything else is refused in
        the status bar -- a dialog for a stray file would be out of proportion.
        """
        tables, images, stacks, refused = [], [], [], []
        for path in map(Path, paths):
            kind = drops.kind_of(path)
            (tables if kind == drops.LOCS else images if kind == drops.IMAGE else
             stacks if kind == drops.STACK else refused).append(path)
        if stacks:
            self.fit_file(stacks[0])
        if tables:
            if self._loading:
                # what File > Open says by being disabled: one load at a time
                self.statusBar().showMessage(
                    f"still loading {self._loading_name}; drop "
                    f"{tables[0].name} again when it is done", 8000)
            else:
                if append is None:
                    append = self.replace_or_add(tables)
                if append is not None:
                    self.load_paths([str(p) for p in tables], append=append)
        # after the tables, which load on a thread: a table opened after the
        # image would replace the session the image was added to
        self._pending_images.extend(images)
        if not self._loading:
            self._open_pending_images()
        if refused:
            more = f" (and {len(refused) - 1} more)" if len(refused) > 1 else ""
            self.statusBar().showMessage(
                f"cannot open {refused[0].name}: not a localization file or an "
                f"image stack{more}", 8000)

    def replace_or_add(self, paths) -> Optional[bool]:
        """Whether tables opened now join the open ones (True), replace them
        (False), or are not opened (None).  Asked only when a table is open:
        with none there is nothing to choose between."""
        if not len(self.session.locs):
            return False
        box = QMessageBox(self)
        box.setWindowTitle("Open")
        what = paths[0].name if len(paths) == 1 else f"{len(paths)} files"
        box.setText(f"Replace the open localizations with {what}, or add "
                    f"{'it' if len(paths) == 1 else 'them'} as "
                    f"{'one more file' if len(paths) == 1 else 'more files'}?")
        replace = box.addButton("Replace", QMessageBox.AcceptRole)
        add = box.addButton("Add", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(replace)     # what Open did before it asked
        box.exec()
        clicked = box.clickedButton()
        return True if clicked is add else False if clicked is replace else None

    def _fitter_for_file(self):
        """``(tab index, tab, instance)`` of the fitter a dropped stack goes to.

        The fitter open on the Localize tab, if one is -- which is the one
        last used, since the open section is kept with the GUI -- else the
        Gaussian 2D fitter, on whichever tab it is pinned to.  None if
        neither is there.
        """
        tabs = [(i, w) for i, w in enumerate(self._tab_widgets())
                if isinstance(w, PluginTab)]
        # the Localize tab by its header, so a renamed one is still found
        tabs.sort(key=lambda item: item[1].tab.header != "localize")
        if tabs and tabs[0][1].tab.header == "localize":
            index, tab = tabs[0]
            opened = tab.open_instance()
            slot = tab.slots.get(opened) if opened else None
            if slot is not None and _takes_a_stack(slot.panel):
                instance = tab.tab.instances[tab.tab.index_of(opened)]
                return index, tab, instance
        for index, tab in tabs:
            for instance in tab.tab.instances:
                if instance.plugin == DROP_FITTER:
                    return index, tab, instance
        return None

    def _pinned(self, plugin: str):
        """``(tab index, tab, instance)`` of the first instance of ``plugin``
        pinned to a tab, or None."""
        for index, tab in enumerate(self._tab_widgets()):
            if isinstance(tab, PluginTab):
                for instance in tab.tab.instances:
                    if instance.plugin == plugin:
                        return index, tab, instance
        return None

    def fit_simulation(self, settings) -> bool:
        """*Simulate*'s *Fit this*: the recipe written from ``settings`` as
        they are now -- never a file an earlier run left -- and handed to a
        fitter (`fit_file`)."""
        from ..plugins.simulate import astigmatic_note, astigmatic_values, recipe_of
        from ..simulate.source import write_recipe
        try:
            path = write_recipe(settings, recipe_of(settings))
        except (OSError, ValueError) as e:
            self.statusBar().showMessage(f"the simulation could not be written: {e}", 12000)
            return False
        return self.fit_file(path, elliptical=astigmatic_note(settings),
                             after=astigmatic_values(settings))

    def fit_file(self, path, elliptical: Optional[str] = None,
                 after: Optional[dict] = None) -> bool:
        """Set a fitter's *file* to ``path`` as if it had been picked with the
        field's own button: the fitter hears of it (`react` fills the camera
        and the output, the hints follow), and nothing is fitted.

        ``elliptical`` -- astigmatic frames, simulated: Gaussian 2D with
        *elliptical* ticked, whose widths Astigmatic z turns into z, and this
        line, saying so, in the status bar; ``after``, the values of its
        *after the fit* that do it (`simulate.astigmatic_values`).

        Returns whether a fitter took it.
        """
        path = Path(path)
        found = self._fitter_for_file()
        from ..simulate.source import recipe_calibration, recipe_transform
        if recipe_transform(path) is not None:
            # a simulated split camera: the two-colour fitter, wherever it is
            found = self._pinned(TWO_COLOUR_FITTER) or found
        drawn = recipe_calibration(path)
        if drawn is not None:
            # drawn with a PSF calibration: the spline fitter of its kind,
            # which takes it -- for a split camera that replaces the above
            found = self._pinned({"dual": DUAL_SPLINE_FITTER, "fourpi": FOURPI_FITTER}
                                 .get(drawn[0], SPLINE_FITTER)) or found
        elif elliptical:
            found = self._pinned(DROP_FITTER) or found
        panel = None
        if found is not None:
            index, tab, instance = found
            panel = tab.slots[instance.id].ensure()      # imports the plugin
        if not _takes_a_stack(panel):
            self.statusBar().showMessage(
                f"{path.name} is an image stack, but no fitter is pinned to a "
                "tab - add one to the Localize tab with +", 12000)
            return False
        self.tabs.setCurrentIndex(index)
        tab.open_section(instance.id)
        field = panel.form.fields["source"].fields["path"]
        field.set(str(path))
        folders.remember(path, stack=True)
        field.changed.emit()                  # -> PluginPanel._react
        edit = getattr(field, "widget", None)
        if hasattr(edit, "setCursorPosition"):
            edit.setCursorPosition(len(edit.text()))   # show the file's name
        model = panel.form.fields.get("model")
        if elliptical and "elliptical" in getattr(model, "fields", {}):
            panel.form.set_values({"model.elliptical": True, **(after or {})})
            panel.refresh_derived()               # the after-the-fit steps it now shows
            self.statusBar().showMessage(
                f"{instance.title(panel.plugin.name)}: file set to {path.name}, elliptical; "
                f"Run fits it -- {elliptical}", 15000)
            return True
        self.statusBar().showMessage(
            f"{instance.title(panel.plugin.name)}: file set to {path.name}; "
            "Run fits it", 8000)
        return True

    def _open_pending_images(self) -> None:
        images, self._pending_images = self._pending_images, []
        for path in images:
            self.open_image_path(path)

    def open_image_path(self, path) -> None:
        """An image as a layer of its own -- a layer per channel -- asking its
        pixel size and what its planes are if the file does not say.  An image already read -- an `.msr`'s -- is kept
        as a source, for an image layer to choose, without a layer."""
        from ..images import ImageData
        if isinstance(path, ImageData):
            self.session.add_source(path)
            return
        from .dialogs import read_image_file
        images = read_image_file(str(path), parent=self)
        if images:
            self.session.open_images(images)

    def save(self) -> None:
        from ..io.formats import writer_for
        if self.session.path is None or self.session.is_protected(self.session.path):
            return self.save_as()
        try:
            writer_for(self.session.path)
        except ValueError:
            # opened from another program's format: saving over it would
            # replace that file with HDF5, so ask where instead
            return self.save_as()
        self.session.save()
        self.statusBar().showMessage(f"saved {self.session.path}")

    def save_as(self) -> None:
        start = str(self.session.path) if self.session.path else folders.start()
        if self.session.is_protected(start):
            here = Path(start)          # offer a new name beside the kept file
            start = str(here.with_name(f"{here.stem}_edited{here.suffix}"))
        path, _ = QFileDialog.getSaveFileName(self, "Save localizations", start,
                                              "HDF5 (*.h5 *.hdf5)")
        if path:
            folders.remember(path)
            try:
                self.session.save(path)
            except ValueError as error:
                QMessageBox.warning(self, "Not saved", str(error))
                return
            self._on_session("locs")

    # ------------------------------------------------------------ quitting
    def closeEvent(self, event) -> None:              # noqa: N802 (Qt's name)
        """Closing this window, and quitting -- Qt 6's `quit` closes every
        window first and is called off when one refuses -- asks about a
        changed table, when the preference says to."""
        try:
            leave = self.may_close()
        except Exception:                 # a question that fails must not
            traceback.print_exc()         # keep the window open for good
            leave = True
        if not leave:
            event.ignore()
            return
        super().closeEvent(event)

    def may_close(self) -> bool:
        """Whether to go: always, unless the table has changed since it was
        saved or opened and the preference asks (`config.ask_before_quit`);
        then Save (and go only if it was saved), Discard, or Cancel."""
        from .. import config
        if getattr(self, "leaving", False):       # *Update and restart* asked already
            return True
        if not self.session.unsaved or not config.ask_before_quit():
            return True
        answer = self._ask_unsaved()
        if answer == "save":
            self.save()
            return not self.session.unsaved      # a cancelled Save as stays
        return answer == "discard"

    def _ask_unsaved(self) -> str:
        """"save", "discard" or "cancel"."""
        name = self.session.display_name or "the localizations"
        box = QMessageBox(QMessageBox.Warning, "Unsaved changes",
                          f"{name} changed since it was saved or opened.",
                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                          self)
        box.setInformativeText("Save the table before quitting?")
        box.setDefaultButton(QMessageBox.Save)
        box.exec()
        chosen = box.standardButton(box.clickedButton()) if box.clickedButton() else None
        if chosen == QMessageBox.Save:
            return "save"
        return "discard" if chosen == QMessageBox.Discard else "cancel"


class RenderWindow(QMainWindow):
    def __init__(self, session: Session):
        super().__init__()
        self.view = RenderView(session)
        self.setCentralWidget(self.view)
        self.toolbar = RenderToolBar(self.view)
        self.addToolBar(self.toolbar)
        # the count under the picture, where a readout is looked for
        self.statusBar().addWidget(self.toolbar.counts)
        self.statusBar().setSizeGripEnabled(False)
        self.setProperty(identity.OWN_TITLE, True)      # names the session itself
        self.resize(900, 900)
        window_shortcuts(self)
        # Files dropped on the picture go where those dropped on the control
        # window go (`ControlWindow.open_dropped`), which sets this.  The
        # graphics view accepts drops by default and its scene then refuses
        # them, which would make the picture -- most of this window -- a
        # place the cursor says nothing can be dropped; so it does not take
        # them and they come up to the window.
        self.dropped = None
        self.setAcceptDrops(True)
        for widget in (self.view.graphics, self.view.graphics.viewport()):
            widget.setAcceptDrops(False)

    def createPopupMenu(self):                      # noqa: N802 (Qt's name)
        """No menu of toolbars on a right-click: its one item hid the
        toolbar, and nothing in the window could bring it back."""
        return None

    def dragEnterEvent(self, event) -> None:
        if self.dropped is not None and drops.accepts(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if self.dropped is not None:
            _drop(event, self.dropped)

    def clipboard_image(self):
        """What Edit > Copy puts on the clipboard: the view as displayed."""
        return self.view.qimage()


MIN_RENDER_WIDTH = 400       # narrower than this, the picture overlaps the control window instead
DROP_FITTER = "Localize/Gaussian 2D"      # a dropped stack's fitter, by default
SPLINE_FITTER = "Localize/Spline 3D"            # a simulation drawn with a calibration
DUAL_SPLINE_FITTER = "Localize/Spline 3D 2C"    # ... a split one, with a dual-colour one
FOURPI_FITTER = "Localize/Spline 4Pi"           # ... a 4Pi camera, with its model
TWO_COLOUR_FITTER = "Localize/Gaussian 2D 2C"   # a simulated split camera's
ASTIGMATIC_FITTER = "Localize/Gaussian 2D"     # where an elliptical Gaussian calibration goes


def _takes_a_stack(panel) -> bool:
    """A built panel with a fitter's *file* in it (`plugins.fit.SourceSettings`)."""
    fields = getattr(getattr(panel, "form", None), "fields", {})
    return (hasattr(getattr(panel, "plugin", None), "resolution")
            and "source" in fields and "path" in getattr(fields["source"], "fields", {}))


def _drop(event, handler) -> None:
    """Take a drop, and open what it carried once the drop is over.

    Not inside the event: opening may ask something (a csv's columns), and a
    dialog during a drop holds the program it was dragged from until it is
    answered.  The deferred call must never raise (a `QTimer.singleShot`
    callable that does crashes PySide), so it prints instead.
    """
    paths = drops.local_paths(event.mimeData())
    event.acceptProposedAction()

    def open_them():
        try:
            handler(paths)
        except Exception:
            traceback.print_exc()

    if paths:
        QTimer.singleShot(0, open_them)


def _start_log() -> None:
    """The diagnostic log (`smappy.diagnostics`), and Qt's own warnings in it.

    Never the reason the GUI does not start: a read-only home directory costs
    the log, not the program.
    """
    from .. import diagnostics
    try:
        diagnostics.start("smappy-gui")
    except OSError as e:
        print(f"no diagnostic log: {e}")
        return
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    levels = {QtMsgType.QtDebugMsg: 10, QtMsgType.QtInfoMsg: 20,
              QtMsgType.QtWarningMsg: 30, QtMsgType.QtCriticalMsg: 40,
              QtMsgType.QtFatalMsg: 50}

    def qt_message(kind, context, text):
        diagnostics.logger.log(levels.get(kind, 30), "(qt) %s", text)
        if sys.__stderr__ is not None and kind != QtMsgType.QtDebugMsg:
            sys.__stderr__.write(text + "\n")      # where Qt would have put it

    qInstallMessageHandler(qt_message)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv if argv is None else argv
    app = make_app(argv)
    _start_log()
    session = Session()
    install_identity(app, session)
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    # the control window at the top left, the picture against its right
    # edge and as tall, the assistant's terminal beside that when started
    # (`assistant_pane.terminal_column`); shown first, so its frame -- the
    # title bar included -- is known and the picture goes against that
    screen = app.primaryScreen().availableGeometry()
    control.move(screen.left(), screen.top())
    control.show()
    frame = control.frameGeometry()
    room = screen.right() + 1 - (frame.right() + 1)
    render.resize(max(MIN_RENDER_WIDTH, min(render.width(), room)), control.height())
    render.move(min(frame.right() + 1, screen.right() + 1 - render.width()), frame.top())
    control.apply_windows()      # where the workspace keeps them, if it does
    render.show()
    control.raise_()             # in front, as before, with the focus
    control.activateWindow()
    args = list(argv[1:])
    if "--handoff" in args:      # started by *Open in a new window* elsewhere
        at = args.index("--handoff")
        if at + 1 < len(args):
            control.take_handoff(args[at + 1])
        del args[at:at + 2]
    if args:                     # after the windows: a big file takes minutes,
        control.load_paths(args, reset_view=True)   # and says so as it goes
    control.updates.start()      # once a day at most, and a few seconds in
    control.listen_if_asked()    # the assistant's, if its settings say so
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
