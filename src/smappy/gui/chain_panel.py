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

The steps are a tree: a block (`docs/regions.md`) holds steps of its own,
indented under it.  A step is added after the selected one, or into the
selected block; up and down move a step within its own list.  **check**, on
a block, times it in a few of its regions and says what the whole will take
and which columns its steps change without writing them back
(`blocks.BlockPlugin.estimate`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMessageBox, QPushButton, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import chain as chains, plugins
from ..chain import Block, ChainError, ChainSpec, Step

# an item's place in the chain: indices from the top list down into blocks
PATH = Qt.UserRole


def _steps_at(spec, path: Tuple[int, ...]) -> Tuple[list, int]:
    """The list that holds the step at ``path``, and its index in it."""
    holder = spec.steps
    for i in path[:-1]:
        holder = holder[i].steps
    return holder, path[-1]


def _step_at(spec, path):
    holder, i = _steps_at(spec, path)
    return holder[i]
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

    def __init__(self, instance: Instance, session: Session, cls: type, parent=None,
                 root: Tuple[int, ...] = ()):
        """``root`` opens the panel on one block of the chain: its steps are
        the top of the tree and new steps go into it (the region steps
        window, `region_steps.py`)."""
        super().__init__(parent)
        self.root = tuple(root)
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
        self.steps = QTreeWidget()
        self.steps.setHeaderHidden(True)
        self.steps.setIndentation(14)
        self.steps.setMaximumHeight(160)
        self.steps.itemChanged.connect(self._ticked)
        self.steps.currentItemChanged.connect(lambda *_: self._selected())
        column.addWidget(self.steps)
        row = QHBoxLayout()
        for label, slot in (("add...", self.add_step), ("block...", self.add_block),
                            ("up", lambda: self.move_step(-1)),
                            ("down", lambda: self.move_step(1)),
                            ("rename...", self.rename_step), ("remove", self.remove_step)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)
        checking = QHBoxLayout()
        self.check_button = QPushButton("check")
        self.check_button.setToolTip("time the selected block in a few of its "
                                     "regions: what the whole will take, and "
                                     "the columns it changes without writing "
                                     "them back")
        self.check_button.clicked.connect(self.check_block)
        self.check_button.setEnabled(bool(self.root))
        self.checked = QLabel("")
        self.checked.setWordWrap(True)
        self.checked.setStyleSheet("color: gray")
        checking.addWidget(self.check_button)
        checking.addWidget(self.checked, 1)
        column.addLayout(checking)
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
                  panel.plot_button, panel.text_button, panel.follow_roi, panel.status,
                  panel.output):
            if w is not None:
                w.setVisible(on)

    # a `_Slot` treats this as it treats a `PluginPanel`
    @property
    def form(self):
        return self.panel.form

    def refresh_derived(self) -> None:
        self.panel.refresh_derived()

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

            def add(parent, steps, path):
                for i, step in enumerate(steps):
                    if isinstance(step, Block):
                        what = ("each ROI" if step.regions == chains.BLOCK_ROIS
                                else f"each {step.regions}")
                        text = (f"{step.label}   (block: {what})" if step.label
                                else f"{step.title()}   (block)")
                    else:
                        text = f"{step.title()}   ({step.plugin})"
                    item = QTreeWidgetItem(parent, [text])
                    item.setData(0, PATH, path + (i,))
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.Checked if step.enabled else Qt.Unchecked)
                    if isinstance(step, Block):
                        add(item, step.steps, path + (i,))
                        item.setExpanded(True)
            top = (self.cls.spec.steps if not self.root
                   else _step_at(self.cls.spec, self.root).steps)
            add(self.steps, top, self.root)
        finally:
            self.steps.blockSignals(blocked)

    def current_path(self) -> Tuple[int, ...]:
        item = self.steps.currentItem()
        return tuple(item.data(0, PATH)) if item is not None else ()

    def select(self, path: Tuple[int, ...]) -> None:
        """Select the step at ``path`` in the tree."""
        def find(item):
            for k in range(item.childCount()):
                child = item.child(k)
                if tuple(child.data(0, PATH)) == tuple(path):
                    return child
                found = find(child)
                if found is not None:
                    return found
            return None
        item = find(self.steps.invisibleRootItem())
        if item is not None:
            self.steps.setCurrentItem(item)

    def _selected(self) -> None:
        path = self.current_path()
        is_block = False
        if path:
            try:
                is_block = isinstance(_step_at(self.cls.spec, path), Block)
            except (IndexError, AttributeError):
                is_block = False
        self.check_button.setEnabled(bool(self.root) or (is_block and len(path) == 1))
        self.checked.setText("")

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
        path = self.current_path()
        self._build(cls)
        if path:
            self.select(path)
        self.changed.emit()
        return True

    # ------------------------------------------------------------- editing
    def _insert(self, step) -> None:
        """Put ``step`` after the selected one, or into the selected block."""
        spec = self.current_spec()
        path = self.current_path()
        if path and isinstance(_step_at(spec, path), Block):
            holder, at = _step_at(spec, path).steps, len(_step_at(spec, path).steps)
            new = path + (at,)
        elif path:
            holder, i = _steps_at(spec, path)
            at, new = i + 1, path[:-1] + (i + 1,)
        elif self.root:
            holder = _step_at(spec, self.root).steps
            at, new = len(holder), self.root + (len(holder),)
        else:
            holder, at, new = spec.steps, len(spec.steps), (len(spec.steps),)
        holder.insert(at, step)
        if self._apply(spec):
            self.select(new)

    def add_step(self) -> None:
        from .chooser import choose_plugin
        path = choose_plugin(self.window())
        if not path:
            return
        try:
            version = plugins.get(path).version
        except Exception:
            version = ""
        self._insert(Step(plugin=path, version=version))

    def add_block(self) -> None:
        """A block over the ROIs, or over an id column of the table."""
        choices = ["each ROI"] + [f"each {name}" for name in self._id_columns()]
        chosen, ok = QInputDialog.getItem(self, "Add a block", "Repeat its steps for:",
                                          choices, 0, True)
        if not ok or not chosen.strip():
            return
        what = chosen.strip()
        what = what[5:] if what.startswith("each ") else what
        regions = chains.BLOCK_ROIS if what in ("ROI", "ROIs", "rois") else what
        self._insert(Block(regions=regions))

    def _id_columns(self) -> List[str]:
        """The integer columns of the table, the ones a block can repeat over."""
        locs = self.session.locs
        skip = {"frame", "filenumber", "channel", "group_id", "n_in_group"}
        return [name for name in sorted(locs.keys())
                if name not in skip and locs[name].dtype.kind in "iu"]

    def move_step(self, by: int) -> None:
        path = self.current_path()
        if not path:
            return
        spec = self.current_spec()
        holder, at = _steps_at(spec, path)
        to = at + by
        if not 0 <= to < len(holder):
            return
        holder[at], holder[to] = holder[to], holder[at]
        if self._apply(spec):
            self.select(path[:-1] + (to,))

    def rename_step(self) -> None:
        path = self.current_path()
        if not path:
            return
        spec = self.current_spec()
        step = _step_at(spec, path)
        name, ok = QInputDialog.getText(self, "Rename step", "Name:", text=step.title())
        if ok and name.strip():
            step.label = name.strip()
            self._apply(spec)

    def remove_step(self) -> None:
        path = self.current_path()
        if not path:
            return
        spec = self.current_spec()
        holder, at = _steps_at(spec, path)
        holder.pop(at)
        self._apply(spec)

    def _ticked(self, item: QTreeWidgetItem) -> None:
        path = tuple(item.data(0, PATH))
        spec = self.current_spec()
        _step_at(spec, path).enabled = item.checkState(0) == Qt.Checked
        self._apply(spec)

    def check_block(self) -> None:
        """Time the selected block in a few regions and say what it found."""
        path = self.current_path() or self.root
        if not path:
            return
        key = self.cls.spec.keys()[path[0]]
        found = next(((k, cls) for k, step, cls in self.cls.steps if k == key), None)
        if found is None:
            self.checked.setText("the block is switched off")
            return
        try:
            settings = getattr(self.panel.form.value(), key)
            estimate = found[1]().estimate(
                self.session.context(grouping=found[1].grouping), settings)
        except Exception as error:
            self.checked.setText(f"cannot check: {error}")
            return
        self.checked.setText("no regions to time" if estimate is None
                             else estimate.text())

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
