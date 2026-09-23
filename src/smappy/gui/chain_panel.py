"""A chain's panel: the plugin panel it is, and the steps it is made of.

A chain is always one collapsed plugin panel -- its steps are the folded
sections of its form -- and building one happens in the same place as using
it: **edit steps** shows a list of the steps with add, move, remove and
rename, and **Save chain** writes the file.  There is no separate chain
editor, because a chain one builds somewhere else and then finds again in a
tab is two things to keep in step, and this is one.

Editing rebuilds the plugin class (`chain.chain_class`), so the form below is
replaced; the values in it survive, because the edited chain is made from
`current_spec`, which carries the form's values into the steps.  Until it is
saved, the edited chain lives in the workspace instance (`Instance.chain`),
so a restart loses nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QToolButton, QVBoxLayout, QWidget)

from .. import chain as chains, plugins
from ..chain import ChainError, ChainSpec, Step
from ..session import Session
from ..workspace import Instance
from .plugin_panel import PluginPanel


def spec_of(instance: Instance) -> Optional[ChainSpec]:
    """The chain an instance carries inline, with the file it came from."""
    if not instance.chain:
        return None
    raw = dict(instance.chain)
    origin = raw.pop("origin", None)
    return ChainSpec.from_dict(raw, origin=Path(origin) if origin else None)


def is_chain(cls) -> bool:
    return isinstance(cls, type) and issubclass(cls, chains.ChainPlugin)


class ChainPanel(QWidget):
    """A `PluginPanel` for a chain, with its steps editable above it."""

    changed = Signal()              # the steps, or where the chain lives, changed

    def __init__(self, instance: Instance, session: Session, cls: type, parent=None):
        super().__init__(parent)
        self.instance = instance
        self.session = session
        self.cls = cls
        self.panel: Optional[PluginPanel] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.edit = QToolButton(text="edit steps", checkable=True)
        self.edit.setToolTip("add, move, rename or remove the chain's steps")
        self.edit.toggled.connect(self._show_editor)
        self.save = QToolButton(text="save chain")
        self.save.clicked.connect(self.save_chain)
        self.save_as = QToolButton(text="save as...")
        self.save_as.clicked.connect(lambda: self.save_chain(ask=True))
        self.where = QLabel("")
        self.where.setStyleSheet("color: gray")
        for w in (self.edit, self.save, self.save_as):
            bar.addWidget(w)
        bar.addWidget(self.where, 1)
        layout.addLayout(bar)

        self.editor = QWidget()
        column = QVBoxLayout(self.editor)
        column.setContentsMargins(0, 0, 0, 0)
        self.steps = QListWidget()
        self.steps.setMaximumHeight(140)
        self.steps.itemChanged.connect(self._ticked)
        column.addWidget(self.steps)
        row = QHBoxLayout()
        for label, slot in (("add...", self.add_step), ("up", lambda: self.move_step(-1)),
                            ("down", lambda: self.move_step(1)),
                            ("rename...", self.rename_step), ("remove", self.remove_step)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)
        self.editor.setVisible(False)
        layout.addWidget(self.editor)

        self.holder = QVBoxLayout()
        layout.addLayout(self.holder)
        self._build(cls)

    def runnable(self, on: bool) -> None:
        """Whether the chain can be run on the session from here.  Not in the
        batch window, where the session is a stand-in and the files are what
        the chain runs on."""
        self._runnable = on
        self._show_run_controls()

    def _show_run_controls(self) -> None:
        on = getattr(self, "_runnable", True)
        panel = self.panel
        for w in (panel.run_button, panel.preview_button, panel.preview_frame,
                  panel.plot_button, panel.text_button, panel.live, panel.status,
                  panel.output):
            if w is not None:
                w.setVisible(on)

    # a `_Slot` treats this as it treats a `PluginPanel`
    @property
    def form(self):
        return self.panel.form

    def _hints(self) -> None:
        self.panel._hints()

    # ------------------------------------------------------------ building
    def _build(self, cls: type) -> None:
        if self.panel is not None:
            self.panel.release()
            self.holder.removeWidget(self.panel)
            # gone now, not when the event loop gets round to deleting it
            self.panel.hide()
            self.panel.setParent(None)
            self.panel.deleteLater()
        self.cls = cls
        self.panel = PluginPanel(cls, self.session)
        self.holder.addWidget(self.panel)
        self._show_run_controls()
        self._list_steps()
        origin = cls.spec.origin
        self.where.setText(("unsaved changes" if self.instance.chain and origin else
                            "not saved" if self.instance.chain else "")
                           + (f"  {origin.name}" if origin else ""))

    def _list_steps(self) -> None:
        blocked = self.steps.blockSignals(True)
        try:
            self.steps.clear()
            for step in self.cls.spec.steps:
                item = QListWidgetItem(f"{step.title()}   ({step.plugin})")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if step.enabled else Qt.Unchecked)
                self.steps.addItem(item)
        finally:
            self.steps.blockSignals(blocked)

    def _show_editor(self, on: bool) -> None:
        self.editor.setVisible(on)

    def current_spec(self) -> ChainSpec:
        """The chain with what the form holds: what an edit starts from and
        what a save writes."""
        try:
            settings = self.panel.form.value()
        except (ValueError, TypeError):
            settings = None              # a half-typed number: the saved values
        return self.cls.current_spec(settings)

    def _apply(self, spec: ChainSpec) -> bool:
        try:
            cls = chains.chain_class(spec, path=self.instance.plugin)
        except ChainError as error:
            QMessageBox.warning(self, "Chain", str(error))
            return False
        raw = spec.to_dict()
        if spec.origin is not None:
            raw["origin"] = str(spec.origin)
        self.instance.chain = raw
        self.instance.values = {}        # the values are in the steps now
        row = self.steps.currentRow()
        self._build(cls)
        self.steps.setCurrentRow(min(row, self.steps.count() - 1))
        self.changed.emit()
        return True

    # ------------------------------------------------------------- editing
    def add_step(self) -> None:
        from .chooser import choose_plugin
        path = choose_plugin(self.window(), scope="locs")
        if not path:
            return
        spec = self.current_spec()
        try:
            version = plugins.get(path).version
        except Exception:
            version = ""
        at = self.steps.currentRow()
        step = Step(plugin=path, version=version)
        if 0 <= at < len(spec.steps):
            spec.steps.insert(at + 1, step)
        else:
            spec.steps.append(step)
        if self._apply(spec):
            self.steps.setCurrentRow(at + 1 if at >= 0 else self.steps.count() - 1)

    def move_step(self, by: int) -> None:
        at = self.steps.currentRow()
        spec = self.current_spec()
        to = at + by
        if not (0 <= at < len(spec.steps) and 0 <= to < len(spec.steps)):
            return
        spec.steps[at], spec.steps[to] = spec.steps[to], spec.steps[at]
        if self._apply(spec):
            self.steps.setCurrentRow(to)

    def rename_step(self) -> None:
        at = self.steps.currentRow()
        spec = self.current_spec()
        if not 0 <= at < len(spec.steps):
            return
        name, ok = QInputDialog.getText(self, "Rename step", "Name:",
                                        text=spec.steps[at].title())
        if ok and name.strip():
            spec.steps[at].label = name.strip()
            self._apply(spec)

    def remove_step(self) -> None:
        at = self.steps.currentRow()
        spec = self.current_spec()
        if 0 <= at < len(spec.steps):
            spec.steps.pop(at)
            self._apply(spec)

    def _ticked(self, item: QListWidgetItem) -> None:
        at = self.steps.row(item)
        spec = self.current_spec()
        if 0 <= at < len(spec.steps):
            spec.steps[at].enabled = item.checkState() == Qt.Checked
            self._apply(spec)

    # -------------------------------------------------------------- saving
    def save_chain(self, ask: bool = False) -> Optional[Path]:
        """Write the chain; to its own file unless ``ask`` or it has none.

        A chain saved into a folder discovery reads becomes that file's
        plugin: the instance then points at its path and carries nothing
        inline.  Saved anywhere else it keeps its steps inline, and says so.
        """
        spec = self.current_spec()
        target = spec.origin
        if ask or target is None:
            folder = plugins.chains_dir()
            folder.mkdir(parents=True, exist_ok=True)
            name = chains.identifier(spec.name if spec.name != "Unsaved chain"
                                     else "my chain")
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save chain", str(target or folder / f"{name}{chains.SUFFIX}"),
                f"Chains (*{chains.SUFFIX})")
            if not chosen:
                return None
            target = Path(chosen)
            if not target.name.endswith(chains.SUFFIX):
                target = target.with_name(target.name.split(".")[0] + chains.SUFFIX)
        spec.origin = None
        chains.write(spec, target)
        plugins.discover(force=True)
        ref = next((r for r in plugins.refs().values()
                    if r.kind == "chain" and Path(r.origin).resolve() == target.resolve()),
                   None)
        if ref is not None:
            self.instance.plugin = ref.path
            self.instance.chain = None
            self.instance.values = {}
            self._build(plugins.get(ref.path))
        else:
            spec.origin = target
            self._apply(spec)
            self.where.setText(f"saved to {target}, which is not a plugin folder: "
                               "it will not appear in the tree")
        self.changed.emit()
        return target
