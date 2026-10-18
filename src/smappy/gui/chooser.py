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
                               QTreeWidget, QTreeWidgetItem,
                               QTreeWidgetItemIterator, QVBoxLayout)

from .. import plugins
from ..plugins.discovery import PluginRef

_PATH = Qt.UserRole
_PAGES: Dict[str, str] = {}        # plugin path -> its page's text


def page_text(ref: PluginRef) -> str:
    """The plugin's documentation page, for the search to look in.

    Read once per plugin and kept: sixty-odd small files are a few
    milliseconds, on the first letter typed, and a plugin is often known by
    what it does -- "drift", "FRC", "Thompson" -- rather than by its name.
    """
    if ref.path not in _PAGES:
        from ..docs import page_file
        try:
            file = page_file(ref.path, ref.origin)
            _PAGES[ref.path] = file.read_text(encoding="utf-8") if file else ""
        except OSError:
            _PAGES[ref.path] = ""
    return _PAGES[ref.path]


def snippet(text: str, word: str, width: int = 60) -> str:
    """A few words either side of the first ``word`` in ``text``."""
    at = text.lower().find(word.lower())
    if at < 0:
        return ""
    start, end = max(at - width, 0), min(at + len(word) + width, len(text))
    piece = " ".join(text[start:end].split())
    return ("..." if start else "") + piece + ("..." if end < len(text) else "")


class PluginTree(QTreeWidget):
    """The plugin tree, collapsed, filtered by a search box."""

    chosen = Signal(str)            # a leaf was double-clicked or Enter pressed

    def __init__(self, per_region: bool = False, parent=None):
        """``per_region`` keeps only the plugins that can run once per
        region (`Plugin.per_region`), for the ROI evaluation pipeline.  It is
        the plugin's own declaration, not its folder, which is what makes
        filtering work with one shared tree."""
        super().__init__(parent)
        self.per_region = per_region
        self.setHeaderHidden(True)
        self.setUniformRowHeights(True)
        self.itemActivated.connect(self._activated)
        self.itemDoubleClicked.connect(self._activated)
        self.refs: Dict[str, PluginRef] = {}
        self.text = ""                 # what is searched for, lower case
        self.reload()

    def reload(self) -> None:
        self.clear()
        self.refs = {p: r for p, r in plugins.refs().items()
                     if not self.per_region or r.per_region}
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
        """Show what matches -- in the name, the path, the summary or the
        documentation page -- and the branches leading to it.

        With a search in place the matches are expanded; emptying the box puts
        the tree back the way it opens, collapsed.
        """
        self.text = text = text.strip().lower()

        def visit(item: QTreeWidgetItem) -> bool:
            path = item.data(0, _PATH)
            hit = not text
            if path and text:
                ref = self.refs[path]
                hit = (text in ref.name.lower() or text in path.lower()
                       or text in ref.description.lower()
                       or text in page_text(ref).lower())
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


# what `choose_plugin` returns for "a new chain...", which is not a plugin
NEW_CHAIN = "<a new chain>"


class PluginChooser(QDialog):
    """Pick a plugin to pin.  `path` is what was chosen, or None -- or
    `NEW_CHAIN`, with ``new_chain``, for an empty chain to build."""

    def __init__(self, parent=None, per_region: bool = False,
                 title: str = "Add a plugin", new_chain: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.path: Optional[str] = None
        layout = QVBoxLayout(self)
        self.search = QLineEdit(placeholderText="find a plugin: its name, or a word from its page")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.tree = PluginTree(per_region=per_region)
        layout.addWidget(self.tree, 1)
        self.about = QLabel(" ")
        self.about.setWordWrap(True)
        self.about.setStyleSheet("color: gray")
        self.about.setMinimumHeight(40)
        layout.addWidget(self.about)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        if new_chain:
            # beside the plugins rather than a button on every tab: a chain
            # is one more thing to pin, made of plugins
            chain = self.buttons.addButton("a new chain...",
                                           QDialogButtonBox.ActionRole)
            chain.setToolTip("several plugins that run as one; add its steps "
                             "with 'edit steps'")
            chain.clicked.connect(lambda: self._accept_path(NEW_CHAIN))
        layout.addWidget(self.buttons)

        trouble = plugins.problems()
        if trouble:
            note = QLabel(f"{len(trouble)} plugin file(s) could not be read")
            note.setToolTip("\n".join(str(d) for d in trouble))
            note.setStyleSheet("color: #a05000")
            layout.addWidget(note)

        self.search.textChanged.connect(self._search)
        self.search.returnPressed.connect(self._first)
        self.tree.itemSelectionChanged.connect(self._selected)
        self.tree.chosen.connect(self._accept_path)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.resize(420, 480)
        self.search.setFocus()

    def _search(self, text: str) -> None:
        """Filter, and select the first match, so Enter opens it."""
        self.tree.filter(text)
        if text.strip():
            self._first(select_only=True)

    def _first(self, select_only: bool = False) -> None:
        """The best plugin left in the tree: selected, or with Enter chosen.

        One whose name has the word comes before one whose page only
        mentions it: "RCC" means RCC, not COMET, whose page compares the two.
        """
        found = []
        it = QTreeWidgetItemIterator(self.tree, QTreeWidgetItemIterator.NotHidden)
        while it.value() is not None:
            if it.value().data(0, _PATH):
                found.append(it.value())
            it += 1
        if not found:
            return
        word = self.tree.text
        item = min(found, key=lambda i: word not in i.text(0).lower())
        self.tree.setCurrentItem(item)
        if not select_only:
            self._accept_path(item.data(0, _PATH))

    def _selected(self) -> None:
        self.path = self.tree.selected()
        ref = self.tree.refs.get(self.path) if self.path else None
        text = ref.description or ref.path if ref else " "
        word = self.tree.text
        if ref and word and word not in (ref.name + ref.path + ref.description).lower():
            # found in the page: say where, or the match looks like a mistake
            text += f"\n\nIn its page: {snippet(page_text(ref), word)}"
        self.about.setText(text)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.path is not None)

    def _accept_path(self, path: str) -> None:
        self.path = path
        self.accept()


def choose_plugin(parent=None, per_region: bool = False,
                  title: str = "Add a plugin", new_chain: bool = False) -> Optional[str]:
    """Show the chooser; the plugin path picked, or None (or `NEW_CHAIN`)."""
    # scanned again, so a plugin saved into a folder since is offered: the
    # scan parses the files and imports none
    plugins.discover(force=True)
    dialog = PluginChooser(parent, per_region=per_region, title=title,
                           new_chain=new_chain)
    return dialog.path if dialog.exec() == QDialog.Accepted else None
