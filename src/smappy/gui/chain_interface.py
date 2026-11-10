"""*interface...* of a chain's step editor: which settings the chain asks.

A chain is made of steps with all their settings, and most of them are how the
chain is made rather than what a user of it decides.  Ticked here, a setting is
shown at the top of the chain's form under the label typed beside it, and the
steps fold away behind *show the steps* (`chain.ChainPlugin.describe`).  What is
chosen is the chain's ``interface:`` (`ChainSpec.interface`), saved with it.
"""
from __future__ import annotations

from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout)

from .. import chain as chains

PATH = Qt.UserRole


class InterfaceDialog(QDialog):
    """Every setting of every step, a tick and a label each."""

    def __init__(self, chain_cls, current: Dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chain interface")
        layout = QVBoxLayout(self)
        note = QLabel("Tick the settings this chain asks; they are shown at the top of its "
                      "form, under the label beside them, and the steps fold away.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["setting", "label in the chain"])
        self.tree.setColumnWidth(0, 260)
        self.items: Dict[str, QTreeWidgetItem] = {}
        for key, title, rows in chains.interface_choices(chain_cls):
            parent_item = QTreeWidgetItem(self.tree, [title])
            for path, label in rows:
                item = QTreeWidgetItem(parent_item, [label, current.get(path, "")])
                item.setData(0, PATH, path)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                item.setCheckState(0, Qt.Checked if path in current else Qt.Unchecked)
                self.items[path] = item
            parent_item.setExpanded(any(p in current for p, _ in rows))
        # only the label column is typed into
        self.tree.itemDoubleClicked.connect(
            lambda item, column: column == 1 and item.data(0, PATH)
            and self.tree.editItem(item, 1))
        self.tree.setEditTriggers(QTreeWidget.NoEditTriggers)
        layout.addWidget(self.tree, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 480)

    def chosen(self) -> Dict[str, str]:
        """The ticked settings, ``{dotted name: label}``, in the order the steps have them; an
        empty label is the field's own."""
        return {path: item.text(1).strip() for path, item in self.items.items()
                if item.checkState(0) == Qt.Checked}

    def choose(self, path: str, on: bool = True, label: str = "") -> None:
        """Tick (or untick) ``path`` with ``label``: what a click and a typed label do."""
        item = self.items[path]
        item.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        item.setText(1, label)
