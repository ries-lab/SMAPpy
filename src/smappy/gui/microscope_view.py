"""What the microscope database found for a file, where the file is opened.

Wherever a stack is opened -- *PSF calibration*, the fitters -- the microscope
and its devices' state are found from the file's own metadata and shown in one
line (`microscope_db.one_line`).  This window is what that line opens: every
device's state and every value, each with where it came from, a microscope
chosen by hand for a file no entry recognises, and the way into the manager.
A microscope chosen here is a setting of the run, as a camera chosen by hand
is: the entries are changed in the manager only.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import microscope_db


def fill_rows(table: QTableWidget, rows, first: int = 0) -> None:
    """``(what, value, source)`` rows into `table` from row `first`; a missing value in grey."""
    table.setRowCount(first + len(rows))
    for i, (what, value, source) in enumerate(rows):
        row = first + i
        table.setItem(row, 0, QTableWidgetItem(str(what)))
        item = QTableWidgetItem("-" if value is None else text(value))
        if value is None:
            item.setForeground(Qt.gray)
        table.setItem(row, 1, item)
        table.setItem(row, 2, QTableWidgetItem(str(source)))


def text(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return " × ".join(text(v) for v in value)
    return str(value)


class FoundMicroscope(QWidget):
    """The microscope found for a file: its line, its rows, a choice by hand.

    `chosen` carries the microscope picked under *microscope* ("" for *auto*);
    the opener resolves the file again with it and calls `show_resolution`.
    `open_manager` is what *Microscope manager...* calls -- the main window's,
    set by whoever opens this; without one the button opens a manager of its own.
    """

    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Microscope")
        layout = QVBoxLayout(self)
        self.summary = QLabel(microscope_db.one_line(None))
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        row = QHBoxLayout()
        row.addWidget(QLabel("microscope"))
        self.choice = QComboBox()
        self.choice.setToolTip("auto: recognised from the file's metadata; or choose one for a "
                               "file no entry recognises -- for this run, the entry is not changed")
        row.addWidget(self.choice, 1)
        layout.addLayout(row)
        self._fill_choices("")
        self.choice.activated.connect(lambda _: self.chosen.emit(self.choice.currentData() or ""))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "value", "from"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        self.note = QLabel("A value typed where the file is opened (the optics, the camera's "
                           "fields) wins over these.")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: gray")
        layout.addWidget(self.note)

        buttons = QHBoxLayout()
        self.manager_button = QPushButton("Microscope manager...")
        self.manager_button.setToolTip("the microscope entries: their devices, states and values, "
                                       "and EMU configurations to start them from")
        self.manager_button.clicked.connect(self._manager)
        buttons.addWidget(self.manager_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.open_manager = None
        self._manager_window = None
        self.resize(560, 420)

    def _fill_choices(self, current: str) -> None:
        self.choice.clear()
        for value, label in microscope_db.choices():
            self.choice.addItem(label, value)
        index = self.choice.findData(current)
        self.choice.setCurrentIndex(max(index, 0))

    def show_resolution(self, resolution, chosen: str = "") -> None:
        """Show what `resolution` (a `MicroscopeResolution`, or None) found."""
        self._fill_choices(chosen)
        found = resolution is not None and resolution.microscope is not None
        self.summary.setText(microscope_db.one_line(resolution.summary() if found else None)
                             + ("" if found or chosen else
                                ": choose one below, or add it in the microscope manager"))
        self.table.setRowCount(0)
        if found:
            fill_rows(self.table, resolution.rows())
            warnings = resolution.warnings()
            if warnings:
                self.note.setText("; ".join(warnings))

    def _manager(self) -> None:
        if self.open_manager is not None:
            self.open_manager()
            return
        from .microscope_editor import MicroscopeManager
        if self._manager_window is None:
            self._manager_window = MicroscopeManager(parent=self)
        self._manager_window.show()
        self._manager_window.raise_()


def open_found(resolution, chosen: str = "", parent=None,
               window: Optional[FoundMicroscope] = None) -> FoundMicroscope:
    """Show the found microscope, reusing the open window."""
    if window is None:
        window = FoundMicroscope(parent)
    window.show_resolution(resolution, chosen)
    window.show()
    window.raise_()
    return window
