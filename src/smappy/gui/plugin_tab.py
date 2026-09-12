"""A tab: the plugins pinned to it, in the order the user put them.

There is one plugin tree and a tab is a curated list over it, so this widget
has a single mode -- the simple GUI, always -- and the tree lives in the
chooser dialog behind the `+` button.  Right-clicking a section reorders,
renames, duplicates or removes it, which is SMAP's per-tab listbox context
menu.

A panel is built the first time its section is opened, so a tab of thirty
plugins imports none of them until one is used.  Together with the scanner
that means starting smappy imports no plugin at all.
"""
from __future__ import annotations

import traceback
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QInputDialog, QMenu,
                               QPushButton, QScrollArea, QToolButton, QVBoxLayout,
                               QWidget)

from .. import plugins
from ..session import Session
from ..workspace import Instance, Tab
from .chooser import choose_plugin
from .plugin_panel import PluginPanel
from .widgets import CollapsibleSection, detach_to_window


class _Slot(QWidget):
    """One instance's panel, built the first time it is looked at.

    Holding a `PluginRef` rather than a class is what lets a tab list plugins
    it has never imported; `ensure` is where the import finally happens, and
    where a broken plugin turns into a message instead of a traceback at
    startup.
    """

    def __init__(self, instance: Instance, session: Session, parent=None):
        super().__init__(parent)
        self.instance = instance
        self.session = session
        self.panel: Optional[PluginPanel] = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def ensure(self) -> Optional[PluginPanel]:
        if self.panel is not None:
            return self.panel
        try:
            cls = plugins.get(self.instance.plugin)
            self.panel = PluginPanel(cls, self.session)
        except Exception:
            trouble = QLabel(f"{self.instance.plugin} could not be loaded:\n"
                             + traceback.format_exc().strip().splitlines()[-1])
            trouble.setWordWrap(True)
            trouble.setStyleSheet("color: #a05000")
            self._layout.addWidget(trouble)
            return None
        self._layout.addWidget(self.panel)
        if self.instance.values:
            self.panel.form.restore(self.instance.values)
        return self.panel

    def values(self) -> Dict[str, object]:
        """What to save.  An unopened panel keeps the values it was given."""
        if self.panel is None:
            return dict(self.instance.values)
        return self.panel.form.values()


class PluginTab(QWidget):
    """The instances of one `Tab`, as collapsible sections."""

    changed = Signal()             # the pinned list or its order was edited

    def __init__(self, tab: Tab, session: Session, header: Optional[QWidget] = None,
                 parent=None):
        super().__init__(parent)
        self.tab = tab
        self.session = session
        self.sections: List[CollapsibleSection] = []
        self.slots: Dict[str, _Slot] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        self.search = QLineEdit(placeholderText="find...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        add = QToolButton(text="+", autoRaise=True)
        add.setToolTip("add a plugin to this tab")
        add.clicked.connect(self.add_plugin)
        top.addWidget(self.search, 1)
        top.addWidget(add)
        layout.addLayout(top)
        if header is not None:
            layout.addWidget(header)
        self.inner = QWidget()
        self.stack = QVBoxLayout(self.inner)
        self.stack.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(self.inner)
        scroll.setFrameShape(QScrollArea.NoFrame)
        layout.addWidget(scroll)
        self.rebuild()

    # ------------------------------------------------------------- building
    def rebuild(self) -> None:
        opened = self.open_instance()
        for section in self.sections:            # keep the slots, drop the frames
            if section.content.parent() is section.frame:
                section.detach()
        while self.stack.count():
            item = self.stack.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.sections = []
        known = plugins.refs()
        for instance in self.tab.instances:
            ref = known.get(instance.plugin)
            slot = self.slots.get(instance.id)
            if slot is None:
                slot = self.slots[instance.id] = _Slot(instance, self.session)
            slot.setParent(None)
            title = instance.title(ref.name if ref else "")
            section = CollapsibleSection(title, slot, detachable=True)
            if ref is None:
                section.button.setToolTip(f"{instance.plugin} is not installed")
            section.toggled.connect(lambda on, s=section, i=instance:
                                    self._toggled(s, i, on))
            section.detach_requested.connect(
                lambda s=section: detach_to_window(s, self.window()))
            section.button.setContextMenuPolicy(Qt.CustomContextMenu)
            section.button.customContextMenuRequested.connect(
                lambda point, i=instance, s=section: self._menu(i, s, point))
            self.sections.append(section)
            self.stack.addWidget(section)
        for stale in set(self.slots) - {i.id for i in self.tab.instances}:
            self.slots.pop(stale).deleteLater()
        if not self.tab.instances:
            hint = QLabel("nothing pinned here yet — use + to add a plugin")
            hint.setStyleSheet("color: gray")
            hint.setWordWrap(True)
            self.stack.addWidget(hint)
        self.stack.addStretch(1)
        if opened:
            self.open_section(opened)
        self._filter(self.search.text())

    def _toggled(self, section: CollapsibleSection, instance: Instance, on: bool) -> None:
        if not on:
            return
        self.slots[instance.id].ensure()          # the import happens here
        for other in self.sections:               # one open at a time
            if other is not section and other.button.isChecked():
                other.set_expanded(False)

    def open_section(self, instance_id: str) -> None:
        for section, instance in zip(self.sections, self.tab.instances):
            if instance.id == instance_id:
                section.set_expanded(True)
                return

    def open_instance(self) -> Optional[str]:
        """Which instance is expanded, for saving and for a rebuild."""
        for section, instance in zip(self.sections, self.tab.instances):
            if section.button.isChecked():
                return instance.id
        return None

    def _filter(self, text: str) -> None:
        text = text.lower()
        for section in self.sections:
            section.setVisible(text in section.title.lower())

    # ------------------------------------------------------------- editing
    def add_plugin(self) -> None:
        path = choose_plugin(self.window())
        if not path:
            return
        instance = Instance(plugin=path)
        self.tab.instances.append(instance)
        self.rebuild()
        self.open_section(instance.id)
        self.changed.emit()

    def _menu(self, instance: Instance, section: CollapsibleSection, point) -> None:
        menu = QMenu(self)
        up = menu.addAction("Move up")
        down = menu.addAction("Move down")
        menu.addSeparator()
        rename = menu.addAction("Rename...")
        duplicate = menu.addAction("Duplicate")
        menu.addSeparator()
        detach = menu.addAction("Open in its own window")
        remove = menu.addAction("Remove from this tab")
        index = self.tab.index_of(instance.id)
        up.setEnabled(index > 0)
        down.setEnabled(index < len(self.tab.instances) - 1)
        chosen = menu.exec(section.button.mapToGlobal(point))
        if chosen is None:
            return
        if chosen in (up, down):
            self.save_values()
            self.tab.move(instance.id, -1 if chosen is up else 1)
            self.rebuild()
        elif chosen is rename:
            known = plugins.refs().get(instance.plugin)
            current = instance.title(known.name if known else "")
            name, ok = QInputDialog.getText(self, "Rename", "Name:", text=current)
            if not ok:
                return
            instance.label = name.strip()
            self.rebuild()
        elif chosen is duplicate:
            self.save_values()
            copy = instance.copy()
            copy.label = f"{instance.title()} copy"
            self.tab.instances.insert(index + 1, copy)
            self.rebuild()
        elif chosen is detach:
            detach_to_window(section, self.window())
            return
        elif chosen is remove:
            self.tab.instances.pop(index)
            self.rebuild()
        self.changed.emit()

    # -------------------------------------------------------------- saving
    def save_values(self) -> None:
        """Read every open form back into its instance, ready to be written."""
        for instance in self.tab.instances:
            slot = self.slots.get(instance.id)
            if slot is not None:
                instance.values = slot.values()
