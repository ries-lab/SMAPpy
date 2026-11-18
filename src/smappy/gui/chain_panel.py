"""A chain's panel: the plugin panel it is, and the steps it is made of.

A chain is one collapsed plugin panel, and its steps are the list: each step
is a folded section of the form with a tick in its title -- on, it runs;
off, it stays in the chain with its settings and is skipped -- and unfolding
it shows that step's settings.  A list of the steps above the form, as there
was, said the same steps twice and kept the settings away from the step
they belong to.

What builds the chain is folded away under **build**: *add...*, *block...*,
*interface...* and *check*, with rename, move and remove on a right click on
a step's title.  Opening and saving the chain are in a **⋯** menu with its
file name beside it, which the tab puts in the section's title bar
(`title_widgets`), so they are there without unfolding *build*: using a
chain and building one happen in the same place, and there is no separate
chain editor to keep in step with what a tab shows.

Editing rebuilds the plugin class (`chain.chain_class`), so the form is
replaced; the values in it survive, because the edited chain is made from
`current_spec`, which carries the form's values into the steps.  Until it is
saved, the edited chain lives in the workspace instance (`Instance.chain`),
so a restart loses nothing.

A block (`docs/regions.md`) is a section holding its steps' sections.  A
step is added after the one last opened or right-clicked (`current_path`),
or into it if it is a block; up and down move a step within its own list.
**interface...** chooses the settings the chain asks (`chain_interface.py`).
**check**, on a block, times it in a few of its regions and says what the
whole will take and which columns its steps change without writing them
back (`blocks.BlockPlugin.estimate`).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMenu, QMessageBox, QPushButton, QToolButton, QVBoxLayout,
                               QWidget)

from .. import chain as chains, plugins
from ..chain import Block, ChainError, ChainSpec, Step
from ..session import Session
from ..workspace import Instance
from .plugin_panel import PluginPanel
from .widgets import CollapsibleSection, note_style


def _steps_at(spec, path: Tuple[int, ...]) -> Tuple[list, int]:
    """The list that holds the step at ``path``, and its index in it."""
    holder = spec.steps
    for i in path[:-1]:
        holder = holder[i].steps
    return holder, path[-1]


def _step_at(spec, path):
    holder, i = _steps_at(spec, path)
    return holder[i]


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
    """A `PluginPanel` for a chain: its steps as ticked sections, the tools
    that build it folded above them."""

    changed = Signal()              # the steps, or where the chain lives, changed
    titled = Signal(str)            # what the file line says now (`title_widgets`)

    def __init__(self, instance: Instance, session: Session, cls: type, parent=None,
                 root: Tuple[int, ...] = ()):
        """``root`` opens the panel on one block of the chain: its steps are
        the top of the list and new steps go into it (the per-region steps,
        `region_steps.py`)."""
        super().__init__(parent)
        self.root = tuple(root)
        self.instance = instance
        self.session = session
        self.cls = cls
        self.panel: Optional[PluginPanel] = None
        self._current: Tuple[int, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        # the file and the ⋯ menu: in this row until a section's title bar
        # takes them (`title_widgets`)
        self.top = QHBoxLayout()
        self.top.setContentsMargins(0, 0, 0, 0)
        self.where = QLabel("")
        self.where.setStyleSheet(note_style())
        self.menu_button = QToolButton(text="⋯", autoRaise=True)
        self.menu_button.setToolTip("open, save or save the chain as a file")
        self.menu_button.setPopupMode(QToolButton.InstantPopup)
        self.chain_menu = QMenu(self)
        # through lambdas: an action's triggered and a button's clicked hand
        # over their checked flag, which became the slot's first argument
        # (Open chain took False for a file, interface... for its dialog)
        self.open_action = self.chain_menu.addAction("Open chain...",
                                                     lambda: self.open_chain())
        self.save_action = self.chain_menu.addAction("Save chain",
                                                     lambda: self.save_chain())
        self.save_as_action = self.chain_menu.addAction(
            "Save as...", lambda: self.save_chain(ask=True))
        self.menu_button.setMenu(self.chain_menu)
        self.top.addWidget(self.where, 1)
        self.top.addWidget(self.menu_button)
        layout.addLayout(self.top)

        # ------------------------------------------------------------ build
        self.editor = QWidget()
        column = QVBoxLayout(self.editor)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(4)
        # *add...* stands in build's title line (below), the one tool wanted
        # without unfolding the rest
        self.add_button = QPushButton("add...")
        self.add_button.setToolTip("add a plugin as a step, after the step last "
                                   "opened, or into it if it is a block")
        self.add_button.clicked.connect(lambda: self.add_step())
        self.block_button = QPushButton("block...")
        self.block_button.setToolTip("add a block: steps repeated in each ROI, or "
                                     "each region of an id attribute")
        self.block_button.clicked.connect(lambda: self.add_block())
        row.addWidget(self.block_button)
        self.interface_button = QPushButton("interface...")
        self.interface_button.setToolTip("choose the settings this chain asks: shown at the top "
                                         "of its form, the steps folded away")
        self.interface_button.clicked.connect(lambda: self.edit_interface())
        row.addWidget(self.interface_button)
        self.check_button = QPushButton("check")
        self.check_button.setToolTip("time the block in a few of its regions: what "
                                     "the whole will take, and the attributes it "
                                     "changes without writing them back")
        self.check_button.clicked.connect(lambda: self.check_block())
        row.addWidget(self.check_button)
        row.addStretch(1)
        column.addLayout(row)
        self.hint = QLabel("right-click a step's title to rename, move or remove it")
        self.hint.setStyleSheet(note_style())
        self.hint.setWordWrap(True)
        column.addWidget(self.hint)
        self.checked = QLabel("")
        self.checked.setWordWrap(True)
        self.checked.setStyleSheet(note_style())
        column.addWidget(self.checked)
        self.build = CollapsibleSection("build", self.editor, expanded=False)
        self.build.add_to_title([self.add_button])
        # what a tab or the batch window ticks to open the tools (an empty chain)
        self.edit = self.build.button
        layout.addWidget(self.build)

        self.holder = QVBoxLayout()
        layout.addLayout(self.holder)
        self._build(cls)

    def title_widgets(self) -> List[QWidget]:
        """A section's title bar's own copy of the chain's file and ⋯ menu,
        new each call, for `CollapsibleSection.add_to_title`; this panel's
        own row is then hidden.  The label follows `titled`; the menu asks
        this panel for its entries each time it opens.  Not this panel's
        widgets lent out: moved into a section and back across rebuilds,
        they were freed under their owner now and then."""
        where = QLabel(self.where.text())
        where.setStyleSheet(note_style())
        self.titled.connect(where.setText)       # dropped when the label goes
        button = QToolButton(text="\u22ef", autoRaise=True)
        button.setToolTip(self.menu_button.toolTip())
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)

        def fill():
            menu.clear()
            try:
                self.fill_menu(menu)
            except RuntimeError:                 # the panel is gone
                pass
        menu.aboutToShow.connect(fill)
        button.setMenu(menu)
        self.where.hide()
        self.menu_button.hide()
        return [where, button]

    def fill_menu(self, menu: QMenu) -> None:
        """The chain's file entries, into ``menu``."""
        menu.addAction("Open chain...", lambda: self.open_chain())
        menu.addAction("Save chain", lambda: self.save_chain())
        menu.addAction("Save as...", lambda: self.save_chain(ask=True))

    def _set_where(self, text: str) -> None:
        self.where.setText(text)
        self.titled.emit(text)

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
                  panel.result_button, panel.follow_roi, panel.status,
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
            self.panel.hide()
            # Still this panel's child until Qt deletes it.  An edit now comes
            # from inside the old form -- a step's tick, its title's menu --
            # and unparented, the old panel was Python's: freed the moment
            # `self.panel` let go of it, under the tick whose signal was still
            # running, which crashed a test worker now and then.
            self.panel.deleteLater()
        self.cls = cls
        self.panel = PluginPanel(cls, self.session)
        self.holder.addWidget(self.panel)
        self._show_run_controls()
        self._decorate()
        origin = cls.spec.origin
        self._set_where(("unsaved changes" if self.instance.chain and origin else
                         "not saved" if self.instance.chain else "")
                        + (f"  {origin.name}" if origin else ""))
        self.check_button.setEnabled(bool(self.root) or any(
            isinstance(step, Block) for step in cls.spec.steps))

    def step_sections(self) -> List[tuple]:
        """``(path, key, step, section)`` for every step the form shows, in
        order, a block's steps after it."""
        out = []

        def walk(steps, keys, form, path):
            if form is None or not hasattr(form, "sections"):
                return
            for i, (key, step) in enumerate(zip(keys, steps)):
                section = form.sections.get(key)
                if section is None:
                    continue
                out.append((path + (i,), key, step, section))
                if isinstance(step, Block):
                    walk(step.steps, step.keys(), form.fields.get(key), path + (i,))

        spec = self.cls.spec
        if self.root:
            form, steps, keys = self.panel.form, spec.steps, spec.keys()
            for i in self.root:
                form = form.fields.get(keys[i]) if form is not None else None
                block = steps[i]
                steps, keys = block.steps, block.keys()
            walk(steps, keys, form, self.root)
        else:
            walk(spec.steps, spec.keys(), self.panel.form, ())
        return out

    def _decorate(self) -> None:
        """Each step's section: a tick in its title that switches it on and
        off, its title's right click, and which step was opened last."""
        for path, key, step, section in self.step_sections():
            box = QCheckBox()
            box.setChecked(step.enabled)
            box.setToolTip("run this step; off, it stays in the chain with its "
                           "settings and is skipped")
            box.toggled.connect(lambda on, p=path: self.set_enabled(p, on))
            section.add_before_title(box)
            section.tick = box
            section.content.setEnabled(step.enabled)
            section.button.setToolTip("right-click to rename, move or remove this step")
            section.button.setContextMenuPolicy(Qt.CustomContextMenu)
            section.button.customContextMenuRequested.connect(
                lambda at, p=path, s=section: self._step_menu(p, s, at))
            section.toggled.connect(lambda on, p=path: self._opened(p, on))

    def _opened(self, path, on: bool) -> None:
        if on:
            self._current = tuple(path)

    def current_path(self) -> Tuple[int, ...]:
        """The step last opened or right-clicked, if it is still there."""
        path = self._current
        if not path:
            return ()
        try:
            _step_at(self.cls.spec, path)
        except (IndexError, AttributeError):
            return ()
        return path

    def select(self, path: Tuple[int, ...]) -> None:
        """Make the step at ``path`` the current one, its section open; ()
        for none."""
        self._current = tuple(path)
        for p, _, _, section in self.step_sections():
            if p == tuple(path):
                section.set_expanded(True)

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
    def step_actions(self, path: Optional[Tuple[int, ...]] = None) -> List[tuple]:
        """What a right click on a step's title offers: ``(label, slot,
        enabled)``, for the step at ``path`` (the current one by default)."""
        if path is not None:
            self._current = tuple(path)
        path = self.current_path()
        if not path:
            return []
        holder, at = _steps_at(self.cls.spec, path)
        out = [("Rename...", self.rename_step, True),
               ("Move up", lambda: self.move_step(-1), at > 0),
               ("Move down", lambda: self.move_step(1), at < len(holder) - 1),
               ("Remove", self.remove_step, True)]
        if isinstance(holder[at], Block):
            out.append(("Add a step into it...", self.add_step, True))
            out.append(("Check", self.check_block, True))
        return out

    def _step_menu(self, path, section, at) -> None:
        menu = QMenu(self)
        for label, slot, enabled in self.step_actions(path):
            action = menu.addAction(label)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _=False, f=slot: f())
        menu.exec(section.button.mapToGlobal(at))

    def set_enabled(self, path: Tuple[int, ...], on: bool) -> None:
        """Switch the step at ``path`` on or off: off, it keeps its section
        and its settings, and is skipped."""
        spec = self.current_spec()
        step = _step_at(spec, path)
        if step.enabled == bool(on):
            return
        step.enabled = bool(on)
        self._current = tuple(path)
        self._apply(spec)

    def _insert(self, step) -> None:
        """Put ``step`` after the current one, or into it if it is a block."""
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
        self._current = path[:-1] + (to,)
        self._apply(spec)

    def rename_step(self) -> None:
        path = self.current_path()
        if not path:
            return
        spec = self.current_spec()
        step = _step_at(spec, path)
        name, ok = QInputDialog.getText(self, "Rename step", "Name:", text=step.title())
        if ok and name.strip():
            before = spec.keys()
            step.label = name.strip()
            # the interface names settings by their step's key, which the label makes
            spec.interface = chains.renamed_interface(spec.interface, before, spec.keys())
            self._apply(spec)

    def remove_step(self) -> None:
        path = self.current_path()
        if not path:
            return
        spec = self.current_spec()
        holder, at = _steps_at(spec, path)
        holder.pop(at)
        self._current = ()
        self._apply(spec)

    def check_block(self) -> None:
        """Time a block in a few regions and say what it found: the current
        step's block, the panel's own (the per-region steps), or the first."""
        path = self.current_path()
        spec = self.cls.spec
        if path and isinstance(spec.steps[path[0]], Block):
            top = path[0]
        elif self.root:
            top = self.root[0]
        else:
            top = next((i for i, s in enumerate(spec.steps) if isinstance(s, Block)), None)
            if top is None:
                self.checked.setText("no block to check")
                return
        key = spec.keys()[top]
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
        self.build.set_expanded(True)          # where the answer is written
        self.checked.setText("no regions to time" if estimate is None
                             else estimate.text())

    def edit_interface(self, dialog=None) -> bool:
        """Choose the settings the chain asks (`chain_interface.InterfaceDialog`); ``dialog``,
        already answered, is for a test."""
        from .chain_interface import InterfaceDialog
        spec = self.current_spec()
        if dialog is None:
            dialog = InterfaceDialog(self.cls, dict(spec.interface), self)
            if dialog.exec() != dialog.Accepted:
                return False
        spec.interface = dialog.chosen()
        return self._apply(spec)

    # ------------------------------------------------------ opening, saving
    def open_chain(self, target: Optional[Path] = None) -> bool:
        """Take a saved chain's steps.  For the per-region steps, a chain
        that is one block gives its steps, any other its steps as they are."""
        if target is None:
            folder = plugins.chains_dir()
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Open chain", str(folder), f"Chains (*{chains.SUFFIX})")
            if not chosen:
                return False
            target = Path(chosen)
        try:
            opened = chains.read(target)
        except Exception as error:               # a file that is not a chain
            QMessageBox.warning(self, "Open chain", f"{target.name}: {error}")
            return False
        if self.root:
            spec = self.current_spec()
            block = _step_at(spec, self.root)
            inner = opened.steps
            if len(inner) == 1 and isinstance(inner[0], Block):
                inner = inner[0].steps
            block.steps = list(inner)
            self._current = ()
            return self._apply(spec)
        opened.origin = Path(target)
        self._current = ()
        return self._apply(opened)

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
        if ref is not None and not self.root:
            self.instance.plugin = ref.path
            self.instance.chain = None
            self.instance.values = {}
            self._build(plugins.get(ref.path))
        else:
            spec.origin = target
            self._apply(spec)
            if ref is None:
                self._set_where(f"saved to {target}, which is not a plugin folder: "
                                "it will not appear in the tree")
        self.changed.emit()
        return target
