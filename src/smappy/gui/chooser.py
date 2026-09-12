"""The plugin chooser: the one place the whole tree is shown.

A tab shows only what is pinned to it, so this dialog is how anything gets
pinned.  One tree widget exists instead of every tab rendering its own copy,
and it is built from `PluginRef`s, so opening it imports no plugin at all --
the descriptions come from the scan.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QLineEdit,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from .. import plugins
from ..plugins.discovery import PluginRef

_PATH = Qt.UserRole


class PluginTree(QTreeWidget):
    """The plugin tree, collapsed, filtered by a search box."""

    chosen = Signal(str)            # a leaf was double-clicked or Enter pressed

    def __init__(self, scope: Optional[str] = None, parent=None):
        """``scope`` keeps only plugins of that kind -- "site" for the ROI
        evaluation pipeline.  It is the plugin's own declaration, not its
        folder, which is what makes filtering work with one shared tree."""
        super().__init__(parent)
        self.scope = scope
        self.setHeaderHidden(True)
        self.setUniformRowHeights(True)
        self.itemActivated.connect(self._activated)
        self.itemDoubleClicked.connect(self._activated)
        self.refs: Dict[str, PluginRef] = {}
        self.reload()

    def reload(self) -> None:
        self.clear()
        self.refs = {p: r for p, r in plugins.refs().items()
                     if self.scope is None or r.scope == self.scope}
        groups: Dict[str, QTreeWidgetItem] = {}

        def group_for(path: str) -> Optional[QTreeWidgetItem]:
            """Make the branch, creating each level once."""
            if not path:
                return None
            if path in groups:
                return groups[path]
            head, _, leaf = path.rpartition("/")
            parent = group_for(head)
            item = QTreeWidgetItem(parent or self, [leaf])
            item.setFlags(Qt.ItemIsEnabled)            # a group is not selectable
            groups[path] = item
            return item

        for path, ref in self.refs.items():
            parent = group_for(ref.group)
            item = QTreeWidgetItem(parent or self, [ref.name])
            item.setData(0, _PATH, path)
            item.setToolTip(0, ref.description or path)
        self.collapseAll()

    def _activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        path = item.data(0, _PATH)
        if path:
            self.chosen.emit(path)

    def selected(self) -> Optional[str]:
        items = self.selectedItems()
        return items[0].data(0, _PATH) if items else None

    def filter(self, text: str) -> None:
        """Show what matches, and the branches leading to it.

        With a search in place the matches are expanded; emptying the box puts
        the tree back the way it opens, collapsed.
        """
        text = text.strip().lower()

        def visit(item: QTreeWidgetItem) -> bool:
            path = item.data(0, _PATH)
            hit = not text
            if path and text:
                ref = self.refs[path]
                hit = (text in ref.name.lower() or text in path.lower()
                       or text in ref.description.lower())
            shown = hit
            for i in range(item.childCount()):
                shown = visit(item.child(i)) or shown
            item.setHidden(not shown)
            if text and item.childCount():
                item.setExpanded(shown)
            return shown

        for i in range(self.topLevelItemCount()):
            visit(self.topLevelItem(i))
        if not text:
            self.collapseAll()


class PluginChooser(QDialog):
    """Pick a plugin to pin.  `path` is what was chosen, or None."""

    def __init__(self, parent=None, scope: Optional[str] = None,
                 title: str = "Add a plugin"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.path: Optional[str] = None
        layout = QVBoxLayout(self)
        self.search = QLineEdit(placeholderText="find a plugin...")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.tree = PluginTree(scope=scope)
        layout.addWidget(self.tree, 1)
        self.about = QLabel(" ")
        self.about.setWordWrap(True)
        self.about.setStyleSheet("color: gray")
        self.about.setMinimumHeight(40)
        layout.addWidget(self.about)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(self.buttons)

        trouble = plugins.problems()
        if trouble:
            note = QLabel(f"{len(trouble)} plugin file(s) could not be read")
            note.setToolTip("\n".join(str(d) for d in trouble))
            note.setStyleSheet("color: #a05000")
            layout.addWidget(note)

        self.search.textChanged.connect(self.tree.filter)
        self.tree.itemSelectionChanged.connect(self._selected)
        self.tree.chosen.connect(self._accept_path)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.resize(420, 480)
        self.search.setFocus()

    def _selected(self) -> None:
        self.path = self.tree.selected()
        ref = self.tree.refs.get(self.path) if self.path else None
        self.about.setText(ref.description or ref.path if ref else " ")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.path is not None)

    def _accept_path(self, path: str) -> None:
        self.path = path
        self.accept()


def choose_plugin(parent=None, scope: Optional[str] = None,
                  title: str = "Add a plugin") -> Optional[str]:
    """Show the chooser; the plugin path picked, or None."""
    dialog = PluginChooser(parent, scope=scope, title=title)
    return dialog.path if dialog.exec() == QDialog.Accepted else None
