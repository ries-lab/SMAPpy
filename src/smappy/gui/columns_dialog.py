"""Which columns of a localization file to read, asked before reading.

Shown by File > Open, a drop and *Load*'s buttons when the File tab's
*columns* says *choose on import*, or says *standard* and the file's
standard columns would not fit (`import_columns.plan`).  Every column the
file holds is listed from its header with the memory it takes once read;
the standard ones start ticked, and the columns a table cannot be without
(`io.formats.REQUIRED`) cannot be unticked.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from ..import_columns import gigabytes
from ..io.formats import REQUIRED
from .widgets import note_style


class ColumnsDialog(QDialog):
    """Tick the columns to read; `columns` is the choice once accepted."""

    def __init__(self, found, why: str = "", available: Optional[int] = None,
                 parent=None):
        super().__init__(parent)
        self.found = list(found)
        self.available = available
        self.setWindowTitle("Columns to read")
        layout = QVBoxLayout(self)
        if why:
            text = QLabel(why)
            text.setWordWrap(True)
            layout.addWidget(text)
        self.tree = QTreeWidget()
        self.tree.setObjectName("columns")
        self.tree.setHeaderLabels(["column", "memory"])
        self.tree.setRootIsDecorated(False)
        self._items: List[QTreeWidgetItem] = []
        for column in self.found:
            item = QTreeWidgetItem([column.name, gigabytes(column.nbytes)])
            item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            required = column.name in REQUIRED
            flags = item.flags() | Qt.ItemIsUserCheckable
            if required:
                flags &= ~Qt.ItemIsEnabled
                item.setToolTip(0, "every table needs it")
            item.setFlags(flags)
            item.setCheckState(0, Qt.Checked if column.standard or required else Qt.Unchecked)
            self.tree.addTopLevelItem(item)
            self._items.append(item)
        self.tree.resizeColumnToContents(0)
        self.tree.itemChanged.connect(lambda *_: self._total())
        layout.addWidget(self.tree, 1)
        self.total = QLabel()
        self.total.setStyleSheet(note_style())
        layout.addWidget(self.total)
        row = QHBoxLayout()
        for label, what in (("Standard", "standard"), ("All", "all"),
                            ("Only required", "none")):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, w=what: self.tick(w))
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Open")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(380, 460)
        self._total()

    def tick(self, which: str) -> None:
        """Tick the standard columns, all of them, or only the required."""
        for column, item in zip(self.found, self._items):
            on = (column.name in REQUIRED or which == "all"
                  or (which == "standard" and column.standard))
            item.setCheckState(0, Qt.Checked if on else Qt.Unchecked)

    def columns(self) -> List[str]:
        return [c.name for c, item in zip(self.found, self._items)
                if item.checkState(0) == Qt.Checked]

    def _total(self) -> None:
        chosen = set(self.columns())
        need = sum(c.nbytes for c in self.found if c.name in chosen)
        text = f"{len(chosen)} of {len(self.found)} columns: {gigabytes(need)}"
        if self.available:
            text += f" of {gigabytes(self.available)} available"
        self.total.setText(text)
