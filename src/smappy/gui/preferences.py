"""Preferences: the folders scanned for plugins, the tabs, light or dark,
whether quitting asks about unsaved changes and whether startup looks for
updates.

Both are things the user names on purpose.  One plugin folder is scanned
without being named, the user's own (`config.user_plugin_dir`), and no tab
appears because a stray directory exists -- but a plugin that no tab shows is
still one click from being pinned, so nothing is unreachable.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog, QHBoxLayout,
                               QInputDialog, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import config, plugins, updates
from ..workspace import Tab, Workspace
from . import theme

BESPOKE = {"render": "the render view"}


def _buttons(*pairs) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    for label, slot in pairs:
        button = QPushButton(label)
        button.clicked.connect(slot)
        row.addWidget(button)
    row.addStretch(1)
    return row


class RootsPage(QWidget):
    """The extra folders to scan.  The shipped ones and the user's own folder
    are always scanned."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        about = QLabel("Folders scanned for plugins. A plugin's place in the tree "
                       "comes from where its file sits: Analysis/Drift/comet.py "
                       "becomes Analysis/Drift/Comet.")
        about.setWordWrap(True)
        about.setStyleSheet("color: gray")
        layout.addWidget(about)
        own = QLabel(f"Always scanned, your own folder (Plugins > Show my plugin "
                     f"folder): {config.user_plugin_dir()}")
        own.setWordWrap(True)
        own.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(own)
        self.list = QListWidget()
        self.list.addItems([str(p) for p in config.plugin_roots()])
        layout.addWidget(self.list, 1)
        layout.addLayout(_buttons(("Add folder...", self.add), ("Remove", self.remove)))
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh()

    def roots(self) -> List[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def add(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Folder with plugins")
        if chosen and chosen not in self.roots():
            self.list.addItem(chosen)
            self.apply()

    def remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.apply()

    def apply(self) -> None:
        """Save and rescan now, so the count below is the truth."""
        config.set_plugin_roots(self.roots())
        plugins.discover(force=True)
        self.refresh()

    def refresh(self) -> None:
        found = plugins.refs()
        trouble = plugins.problems()
        text = f"{len(found)} plugins found"
        if trouble:
            text += f"; {len(trouble)} file(s) could not be read"
            self.status.setToolTip("\n".join(str(d) for d in trouble))
        self.status.setText(text)
        self.status.setStyleSheet("color: #c06000" if trouble else "color: gray")


class GeneralPage(QWidget):
    """Light or dark, and what the program does on its own: whether quitting
    asks, and whether it looks for a newer version."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("appearance"))
        self.appearance = QComboBox()
        for name, text in (("system", "as the system"), ("light", "light"),
                           ("dark", "dark")):
            self.appearance.addItem(text, name)
        self.appearance.setCurrentIndex(max(0, self.appearance.findData(theme.theme())))
        self.appearance.setToolTip("light or dark windows; as the system follows "
                                   "the operating system's setting as it changes")
        # at once, in every open window: the choice is seen, not imagined
        self.appearance.currentIndexChanged.connect(
            lambda _: theme.set_theme(self.appearance.currentData()))
        row.addWidget(self.appearance)
        row.addStretch(1)
        layout.addLayout(row)
        self.ask_quit = QCheckBox("ask before quitting with unsaved changes")
        self.ask_quit.setToolTip("closing the control window or quitting with a "
                                 "table changed since it was saved or opened "
                                 "(* in the title) offers to save it first")
        self.ask_quit.setChecked(config.ask_before_quit())
        self.ask_quit.toggled.connect(config.set_ask_before_quit)
        layout.addWidget(self.ask_quit)
        self.check_updates = QCheckBox("check for a newer SMAPpy once a day")
        self.check_updates.setToolTip("at startup, ask PyPI for the newest version and "
                                      "say so if it is newer; nothing is installed, "
                                      "and Help > Check for updates asks at any time")
        self.check_updates.setChecked(updates.automatic())
        self.check_updates.toggled.connect(updates.set_automatic)
        layout.addWidget(self.check_updates)
        layout.addStretch(1)


class TabsPage(QWidget):
    """The tab strip: add, rename, reorder, remove.  A new tab starts empty."""

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        layout = QVBoxLayout(self)
        about = QLabel("Tabs are lists of plugins you pin, not branches of the "
                       "tree — any plugin can go in any tab. A tab you add "
                       "starts empty; use + in the tab to pin something.")
        about.setWordWrap(True)
        about.setStyleSheet("color: gray")
        layout.addWidget(about)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.list, 1)
        layout.addLayout(_buttons(("Add", self.add), ("Rename...", self.rename),
                                  ("Up", lambda: self.move(-1)),
                                  ("Down", lambda: self.move(1)),
                                  ("Remove", self.remove)))
        self.reset_button = QPushButton("Reset tabs to defaults")
        self.reset_button.clicked.connect(self.reset)
        layout.addWidget(self.reset_button)
        self.refresh()

    def refresh(self, select: int = 0) -> None:
        self.list.clear()
        for tab in self.workspace.tabs:
            if tab.kind in BESPOKE:
                text = f"{tab.name}  ({BESPOKE[tab.kind]})"
            else:
                text = f"{tab.name}  ({len(tab.instances)} plugins)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, tab.name)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(max(0, min(select, self.list.count() - 1)))

    def current(self) -> Optional[int]:
        row = self.list.currentRow()
        return row if 0 <= row < len(self.workspace.tabs) else None

    def add(self) -> None:
        name, ok = QInputDialog.getText(self, "New tab", "Name:")
        name = name.strip()
        if not ok or not name:
            return
        if any(t.name == name for t in self.workspace.tabs):
            QMessageBox.warning(self, "New tab", f"There is already a tab called {name}.")
            return
        self.workspace.tabs.append(Tab(name=name))
        self.refresh(len(self.workspace.tabs) - 1)

    def rename(self) -> None:
        row = self.current()
        if row is None:
            return
        tab = self.workspace.tabs[row]
        name, ok = QInputDialog.getText(self, "Rename tab", "Name:", text=tab.name)
        if ok and name.strip():
            tab.name = name.strip()
            self.refresh(row)

    def move(self, by: int) -> None:
        row = self.current()
        if row is None or not 0 <= row + by < len(self.workspace.tabs):
            return
        tabs = self.workspace.tabs
        tabs[row], tabs[row + by] = tabs[row + by], tabs[row]
        self.refresh(row + by)

    def remove(self) -> None:
        row = self.current()
        if row is None:
            return
        tab = self.workspace.tabs[row]
        pinned = len(tab.instances)
        question = f"Remove the {tab.name} tab?"
        if pinned:
            question += f"\n\n{pinned} pinned plugin(s) go with it."
        if QMessageBox.question(self, "Remove tab", question) != QMessageBox.Yes:
            return
        self.workspace.tabs.pop(row)
        self.refresh(row)

    def reset(self) -> None:
        if QMessageBox.question(self, "Reset tabs",
                                "Replace every tab with the shipped defaults?\n\n"
                                "Anything you pinned is lost.") != QMessageBox.Yes:
            return
        self.workspace.tabs = Workspace.default().tabs
        self.refresh()


class PreferencesDialog(QDialog):
    """Plugin folders and tabs.  Accepting means the window rebuilds its tabs."""

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.workspace = workspace
        layout = QVBoxLayout(self)
        self.pages = QTabWidget()
        self.roots = RootsPage()
        self.tabs = TabsPage(workspace)
        self.general = GeneralPage()
        self.pages.addTab(self.roots, "Plugin folders")
        self.pages.addTab(self.tabs, "Tabs")
        self.pages.addTab(self.general, "General")
        layout.addWidget(self.pages)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.resize(460, 420)
