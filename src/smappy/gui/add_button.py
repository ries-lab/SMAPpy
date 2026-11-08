"""*Add...*: files and folders, the one button every list of inputs has.

The batch window, the PSF calibration window and the bead calibration each
had their own pair, *Add files...* and *Add folder...*, which read as if
only the second took folders and differed in what a folder gave.  Here it is
one button.  Its click opens the system's dialog, which on a Mac takes files
and folders together, several at once (`mac_panel`); its menu holds *Select
several folders...*, Qt's own dialog with several folders selectable -- no
native dialog takes more than one folder, and elsewhere the native one takes
no folder at all.  What is chosen goes to the window as plain paths; what a
folder holds is the window's business (`datasets.find`).
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QFileDialog, QMenu, QPushButton, QSizePolicy, QToolButton

from . import folders

ADD = "Add..."
SEVERAL_FOLDERS = "Select several folders..."


class AddButton(QToolButton):
    """Emits `chosen` with the paths picked, files and folders mixed."""

    chosen = Signal(list)

    def __init__(self, title: str = "Add", file_filter: str = "All files (*)", parent=None):
        super().__init__(parent)
        self._push = None               # a push button's size, measured once
        self.title, self.file_filter = title, file_filter
        self.setText(ADD)
        self.setPopupMode(QToolButton.MenuButtonPopup)
        self.setToolTip("files, or on a Mac files and folders together, several at once; "
                        "the arrow: several folders at once.  Files and folders can also "
                        "be dropped onto the window")
        menu = QMenu(self)
        self.several = menu.addAction(SEVERAL_FOLDERS, self.choose_folders)
        self.setMenu(menu)
        self.clicked.connect(self.choose)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        """A push button's size, and room for the arrow: a tool button is
        drawn smaller than the push buttons beside it, and this one is the
        list's most important."""
        own = super().sizeHint()
        if self._push is None:
            self._push = QPushButton(self.text()).sizeHint()
        push = self._push
        return QSize(max(own.width(), push.width() + own.height() // 2),
                     max(own.height(), push.height()))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def choose(self) -> None:
        from . import mac_panel
        picked = mac_panel.choose_files_and_folders(f"{self.title}: files or folders",
                                                    folders.start())
        if picked is None:
            picked, _ = QFileDialog.getOpenFileNames(self, self.title, folders.start(),
                                                     self.file_filter)
        self._emit(picked)

    def choose_folders(self) -> None:
        self._emit(folders.choose_folders(self, f"{self.title}: folders"))

    def _emit(self, paths: List[str]) -> None:
        paths = [str(p) for p in paths or [] if p]
        if paths:
            folders.remember(paths, stack=True)
            self.chosen.emit(paths)
