"""The batch window: a list of files, a chain, where the results go, and Run.

The window is a client of the job file and nothing more.  **Run** writes the
job (`batch.write_job`) and starts ``smappy-batch run`` on it as a separate
process, reading the ``SMAPPY_*`` lines it prints -- the way MPS's GUI runs
its batch -- so a crash or an exhausted memory costs the batch and never
this window or the session behind it, and what the window did can be done
again from a terminal with the file it wrote.  With *watch* ticked it starts
``smappy-batch watch`` instead, which runs on until stopped; its Stop is a
``stop`` line on the runner's stdin, which ends it after the file in hand,
and only a second Stop kills it.

The same class is ``smappy-batch-gui`` on its own and *Tools -> Batch...* in
smappy; the only difference is whether there is a session to open a result
in.  The chain is edited with the `ChainPanel` the plugin tabs use, so a chain
reads and edits the same everywhere; here it cannot be run on the session,
only saved and run over the files.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from .. import batch, chain as chains, plugins
from ..batch import Input, Job, Output, _resolve
from ..chain import ChainSpec
from ..session import Session
from ..workspace import Instance
from . import drops, folders
from .add_button import AddButton
from .chain_panel import ChainPanel
from .widgets import CollapsibleSection, note_style

COLUMNS = ("data", "filter", "status")
ERROR = "#e0453a"          # reads on a light and a dark window


def _count(n: int, what: str) -> str:
    return f"{n} {what}{'' if n == 1 else 's'}"

# The runner's pipe in UTF-8.  It prints paths, and this window finds a
# file's row by the path it prints back (`_item_for`).  On Windows a pipe is
# written in the ANSI code page, so a folder called ``Jürgen`` came back as
# bytes that are not UTF-8 and matched no row, and a name outside cp1252
# (Greek, Chinese) raised UnicodeEncodeError in the runner.  ``cli.batch``
# reconfigures its streams too, for a runner started from anywhere else.
UTF8_ENVIRONMENT = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def stop_process(process, windows: Optional[bool] = None) -> None:
    """End the runner.

    On POSIX ``terminate`` is SIGTERM, which ends it.  On Windows it only
    posts WM_CLOSE, which a console program such as ``python -m
    smappy.cli.batch`` never reads, so Stop did nothing; there it is
    ``kill``.  Either way the file being processed is left unfinished, and
    the next run does it again.
    """
    if sys.platform.startswith("win") if windows is None else windows:
        process.kill()
    else:
        process.terminate()


class BatchWindow(QMainWindow):
    """Files, a chain, output options, and a subprocess that runs them."""

    def __init__(self, open_result: Optional[Callable[[str], None]] = None, parent=None):
        super().__init__(parent)
        from .help_window import attach_help, show_help
        attach_help(self, 'Panels/Batch window')
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("SMAPpy batch")
        self.setAcceptDrops(True)
        # the chain panel needs a session to exist; this one is never run on
        self.session = Session()
        self.open_result = open_result
        self.job_path: Optional[Path] = None
        self.process: Optional[QProcess] = None
        self._buffer = b""
        self._running: Dict[str, int] = {}
        self.report_path: Optional[Path] = None
        self._stopping = False

        central = QWidget()
        root = QVBoxLayout(central)
        # the "?" every plugin's title bar has: F1 alone was never found
        top = QHBoxLayout()
        top.addWidget(QLabel("one chain over many files, each in a fresh session"))
        top.addStretch(1)
        self.help_button = QToolButton(text="?", autoRaise=True)
        self.help_button.setToolTip("how to use the batch window: files, a chain, "
                                    "the output, and what a run writes (F1)")
        self.help_button.clicked.connect(
            lambda: show_help("Panels/Batch window", self))
        top.addWidget(self.help_button)
        root.addLayout(top)
        split = QSplitter(Qt.Vertical)
        root.addWidget(split, 1)

        # ---- inputs: what was added, a folder unfolding into its data sets
        inputs = QGroupBox("files")
        column = QVBoxLayout(inputs)
        self.kind_line = QLabel()
        self.kind_line.setWordWrap(True)
        self.kind_line.setStyleSheet(note_style())
        column.addWidget(self.kind_line)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2):
            header.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setToolTip("a file, or a folder and the data sets found in it; untick one "
                             "to leave it out.  A folder's filter narrows it by name: "
                             "*640*, !*bead*, (*640* & !*647*), *600*")
        self.tree.setMinimumHeight(110)
        self.tree.itemChanged.connect(self._item_changed)
        # only a folder's filter is typed in, on a double-click
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.itemDoubleClicked.connect(
            lambda item, column: self.tree.editItem(item, 1)
            if column == 1 and item.parent() is None and item.data(0, Qt.UserRole).folder
            else None)
        column.addWidget(self.tree)
        row = QHBoxLayout()
        self.add_button = AddButton("Add data")
        self.add_button.chosen.connect(self.add_paths)
        row.addWidget(self.add_button)
        for label, slot in (("remove", self.remove_selected), ("clear", self.clear_inputs)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
        self.subfolders = QCheckBox("search subfolders", checked=True)
        self.subfolders.setToolTip("a folder's data sets in its subfolders too")
        self.subfolders.toggled.connect(self._subfolders_toggled)
        row.addWidget(self.subfolders)
        self.watch = QCheckBox("watch the folders")
        self.watch.setToolTip("run on the data there is, then on each new data set in the "
                              "folders as soon as it is complete, until Stop")
        row.addWidget(self.watch)
        column.addLayout(row)
        split.addWidget(inputs)

        # ---- the chain
        chain_box = QGroupBox("chain")
        column = QVBoxLayout(chain_box)
        pick = QHBoxLayout()
        self.chooser = QComboBox()
        self.chooser.setToolTip("a saved chain, from the plugin folders and the "
                                "chains folder")
        self.chooser.activated.connect(self._chosen)
        open_chain = QPushButton("open...")
        open_chain.clicked.connect(self.open_chain)
        new_chain = QPushButton("new")
        new_chain.clicked.connect(lambda: self.set_chain(ChainSpec()))
        pick.addWidget(self.chooser, 1)
        pick.addWidget(open_chain)
        pick.addWidget(new_chain)
        column.addLayout(pick)
        # a chain of a fit and four analyses, opened, is taller than a screen
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea(widgetResizable=True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        self.chain_holder = QVBoxLayout(inner)
        self.chain_holder.setContentsMargins(0, 0, 0, 0)
        self.chain_holder.addStretch(1)
        scroll.setWidget(inner)
        column.addWidget(scroll, 1)
        self.chain_panel: Optional[ChainPanel] = None
        split.addWidget(chain_box)

        # ---- output: where and what; the rest under "more"
        out_box = QGroupBox("output")
        out_column = QVBoxLayout(out_box)
        form = QFormLayout()
        out_column.addLayout(form)
        self.out_folder = QLineEdit()
        self.out_folder.setPlaceholderText("beside the job file: <job>_output")
        browse = QToolButton(text="...")
        browse.clicked.connect(self._browse_output)
        line = QHBoxLayout()
        line.addWidget(self.out_folder, 1)
        line.addWidget(browse)
        form.addRow("folder", line)
        self.figures = self._combo(batch.FIGURES, "png")
        form.addRow("figures", self.figures)
        self.save_locs = QCheckBox("save the processed localizations", checked=True)
        form.addRow("", self.save_locs)
        more = QWidget()
        form = QFormLayout(more)
        form.setContentsMargins(0, 0, 0, 0)
        self.suffix = QLineEdit(Output.suffix)
        form.addRow("file suffix", self.suffix)
        self.fitted = self._combo(batch.FITTED, "beside_image")
        self.fitted.setToolTip("for a chain that starts with a fit: where the fit "
                               "writes its own file")
        self.delete_fitted = QCheckBox("delete the fitted file afterwards")
        fitted_row = QHBoxLayout()
        fitted_row.addWidget(self.fitted)
        fitted_row.addWidget(self.delete_fitted)
        form.addRow("fitted file", fitted_row)
        self.preflight = self._combo(batch.PREFLIGHT, "skip")
        self.preflight.setToolTip("what to do with a file a step would ask about "
                                  "(\"this will take an afternoon\")")
        form.addRow("when a step asks", self.preflight)
        self.rerun = self._combo(batch.RERUN, "skip_identical")
        form.addRow("files already done", self.rerun)
        self.trends = QLineEdit()
        self.trends.setPlaceholderText("every number that differs between files")
        self.trends.setToolTip("summary.csv columns plotted across the files on the "
                               "summary page, comma-separated; patterns allowed "
                               "(stats.*)")
        form.addRow("trend columns", self.trends)
        self.more = CollapsibleSection("more", more)
        out_column.addWidget(self.more)
        split.addWidget(out_box)

        # ---- running
        run_box = QWidget()
        column = QVBoxLayout(run_box)
        column.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        # push buttons with a menu: as large as Run beside them
        self.job_button = QPushButton("job")
        self.job_button.setToolTip("the files, the chain and the output as one "
                                   ".batch.yaml file, which smappy-batch runs from a terminal")
        menu = QMenu(self.job_button)
        menu.addAction("Open job...", self.open_job)
        menu.addAction("Save job...", self.save_job)
        self.job_button.setMenu(menu)
        buttons.addWidget(self.job_button)
        buttons.addStretch(1)
        self.try_one = QCheckBox("first file only")
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.results_button = QPushButton("results")
        self.results_button.setToolTip("what the batch wrote: the summary page, or the "
                                       "output folder")
        menu = QMenu(self.results_button)
        menu.addAction("Open summary", self.open_page)
        menu.addAction("Open output folder", self.open_output)
        self.results_button.setMenu(menu)
        for w in (self.try_one, self.run_button, self.stop_button, self.results_button):
            buttons.addWidget(w)
        column.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.hide()                # an empty bar before any run read as a field
        column.addWidget(self.progress)
        self.log = QPlainTextEdit(readOnly=True, maximumBlockCount=5000)
        self.log.setMinimumHeight(120)
        column.addWidget(self.log, 1)
        split.addWidget(run_box)
        for index, stretch in enumerate((1, 2, 0, 1)):
            split.setStretchFactor(index, stretch)

        self.setCentralWidget(central)
        self.resize(760, 900)
        self.refresh_chains()
        self.set_chain(ChainSpec())

    @staticmethod
    def _combo(values, current) -> QComboBox:
        box = QComboBox()
        for v in values:
            box.addItem(v, v)
        box.setCurrentIndex(max(box.findData(current), 0))
        return box

    # ---------------------------------------------------------------- inputs
    # A top-level item is an input as the job keeps it (`batch.Input`, in its
    # UserRole); a folder's children are the data sets it gives now, each a
    # path in its UserRole, unticked when the input excludes it.  Everything
    # under a folder is found again whenever what decides it changes: the
    # chain's input kind, the filter, *search subfolders*.

    def _base(self) -> Path:
        return self.job_path.parent if self.job_path is not None else Path.cwd()

    def add_input(self, entry: Input) -> QTreeWidgetItem:
        item = QTreeWidgetItem([entry.file or entry.folder, entry.filter if entry.folder else "", ""])
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        if entry.folder:
            flags |= Qt.ItemIsEditable          # the filter, column 1 (`_edit_filter`)
        item.setFlags(flags)
        item.setData(0, Qt.UserRole, entry)
        item.setToolTip(0, ("folder: " if entry.folder else "file: ") + (entry.file or entry.folder))
        if entry.folder:
            item.setToolTip(1, "narrows the folder by name: *640*, !*bead*, beads/*, "
                               "(*640* & !*647*), *600* -- double-click to edit")
        blocked = self.tree.blockSignals(True)
        item.setCheckState(0, Qt.Checked if entry.include else Qt.Unchecked)
        self.tree.addTopLevelItem(item)
        self.tree.blockSignals(blocked)
        self._fill(item)
        return item

    def add_paths(self, paths) -> None:
        """Files and folders, chosen or dropped: a folder stays a folder (its
        data sets found now, and again on every run and look of a watch); a
        folder that is one data set (NDTiff) and a file are inputs of their
        own.  What is there already is not added twice."""
        from .. import datasets
        have = {(e.file or e.folder) for e in self.inputs()}
        for path in map(Path, paths):
            if str(path) in have:
                continue
            have.add(str(path))
            if path.is_dir() and not datasets.is_ndtiff(path):
                self.add_input(Input(folder=str(path), recursive=self.subfolders.isChecked()))
            else:
                self.add_input(Input(file=str(path)))

    def inputs(self) -> List[Input]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            entry = dataclasses.replace(item.data(0, Qt.UserRole))
            entry.include = item.checkState(0) == Qt.Checked
            out.append(entry)
        return out

    def clear_inputs(self) -> None:
        self.tree.clear()

    def remove_selected(self) -> None:
        """Removes the selected inputs; a data set selected inside a folder is
        unticked instead, since the folder would find it again."""
        for item in self.tree.selectedItems():
            parent = item.parent()
            if parent is None:
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
            else:
                item.setCheckState(0, Qt.Unchecked)

    def refresh_inputs(self) -> None:
        """Every input found again: the chain, and so the kind it reads, changed."""
        kind = self._kind()
        from ..datasets import DESCRIPTIONS
        steps = self.chain_panel.cls.steps if self.chain_panel is not None else ()
        first = (getattr(steps[0][2], "name", "") or steps[0][2].path.rsplit("/", 1)[-1]
                 if steps else "")
        self.kind_line.setText(f"The chain{f' starts with {first}: it' if first else ''} "
                               f"reads {DESCRIPTIONS[kind]}.")
        for i in range(self.tree.topLevelItemCount()):
            self._fill(self.tree.topLevelItem(i))

    def _fill(self, item: QTreeWidgetItem) -> None:
        """An input's data sets as the job would run them now, and its line:
        how many a folder gives, or why a file is skipped."""
        from .. import datasets
        entry: Input = item.data(0, Qt.UserRole)
        kind = self._kind()
        blocked = self.tree.blockSignals(True)
        try:
            item.takeChildren()
            item.setForeground(0, self.palette().text())
            item.setForeground(1, self.palette().text())
            item.setToolTip(2, "")
            try:
                runs, left = batch.entry_datasets(entry, kind, self._base(), self._output())
            except datasets.FilterError as error:
                item.setForeground(1, QColor(ERROR))
                item.setText(2, "filter error")
                item.setToolTip(2, str(error))
                return
            except OSError as error:
                item.setText(2, "unreadable")
                item.setToolTip(2, str(error))
                return
            if entry.file:
                if runs:
                    item.setText(2, "")
                else:
                    item.setForeground(0, self.palette().placeholderText())
                    there = _resolve(entry.file, self._base()).exists()
                    item.setText(2, "skipped" if there else "not found")
                    item.setToolTip(2, f"this chain reads {datasets.DESCRIPTIONS[kind]}"
                                    if there else "")
                return
            folder = _resolve(entry.folder, self._base())
            if not folder.is_dir():
                item.setText(2, "not found")
                return
            for path in sorted(runs + left, key=lambda p: str(p).lower()):
                child = QTreeWidgetItem([datasets.relative(path, folder), "", ""])
                child.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                child.setData(0, Qt.UserRole, str(path))
                child.setToolTip(0, str(path))
                child.setCheckState(0, Qt.Unchecked if path in left else Qt.Checked)
                item.addChild(child)
            item.setText(2, _count(len(runs), "data set"))
            item.setExpanded(item.childCount() <= 12)
        finally:
            self.tree.blockSignals(blocked)

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """A data set ticked or unticked (the folder's exclude), or a folder's
        filter edited."""
        parent = item.parent()
        if parent is not None and column == 0:
            entry: Input = parent.data(0, Qt.UserRole)
            folder = _resolve(entry.folder, self._base())
            from ..datasets import relative
            name = relative(Path(item.data(0, Qt.UserRole)), folder)
            out = [x for x in entry.exclude if x != name]
            if item.checkState(0) != Qt.Checked:
                out.append(name)
            entry.exclude = out
            runs = sum(parent.child(i).checkState(0) == Qt.Checked
                       for i in range(parent.childCount()))
            blocked = self.tree.blockSignals(True)
            parent.setText(2, _count(runs, "data set"))
            self.tree.blockSignals(blocked)
        elif parent is None and column == 1:
            entry = item.data(0, Qt.UserRole)
            if entry.folder and item.text(1).strip() != entry.filter:
                entry.filter = item.text(1).strip()
                self._fill(item)
        elif parent is None and column == 0 and item.text(0) != (
                item.data(0, Qt.UserRole).file or item.data(0, Qt.UserRole).folder):
            blocked = self.tree.blockSignals(True)        # the path is not edited here
            entry = item.data(0, Qt.UserRole)
            item.setText(0, entry.file or entry.folder)
            self.tree.blockSignals(blocked)

    def _subfolders_toggled(self, on: bool) -> None:
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            entry = item.data(0, Qt.UserRole)
            if entry.folder and entry.recursive != on:
                entry.recursive = on
                self._fill(item)

    def _output(self) -> Optional[Path]:
        try:
            return Job(chain=ChainSpec(), output=Output(folder=self.out_folder.text().strip()),
                       origin=self.job_path).output_folder()
        except Exception:
            return None

    def dragEnterEvent(self, event) -> None:
        if drops.accepts(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        data = []
        for path in drops.local_paths(event.mimeData()):
            if path.name.endswith(chains.SUFFIX):
                self.set_chain(chains.read(path))
            elif path.name.endswith((".batch.yaml", ".batch.json")):
                self.load_job(path)
            else:
                data.append(str(path))
        self.add_paths(data)

    # ----------------------------------------------------------------- chain
    def refresh_chains(self) -> None:
        self.chooser.clear()
        self.chooser.addItem("(choose a saved chain)", None)
        for path, ref in plugins.refs().items():
            if ref.kind == "chain":
                self.chooser.addItem(path, str(ref.origin))

    def _chosen(self, index: int) -> None:
        origin = self.chooser.itemData(index)
        if origin:
            self.set_chain(chains.read(origin))

    def open_chain(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open chain", str(plugins.chains_dir()),
                                              f"Chains (*{chains.SUFFIX} *.chain.yml)")
        if path:
            self.set_chain(chains.read(path))

    def set_chain(self, spec: ChainSpec) -> None:
        """Show ``spec`` in the chain panel, editable, not runnable here."""
        raw = spec.to_dict()
        if spec.origin is not None:
            raw["origin"] = str(spec.origin)
        instance = Instance(plugin=f"{chains.DEFAULT_GROUP}/{spec.name}", chain=raw)
        try:
            cls = chains.chain_class(spec, path=instance.plugin)
        except chains.ChainError as error:
            QMessageBox.warning(self, "Chain", str(error))
            return
        if self.chain_panel is not None:
            self.chain_panel.panel.release()
            self.chain_holder.removeWidget(self.chain_panel)
            self.chain_panel.hide()
            self.chain_panel.setParent(None)
            self.chain_panel.deleteLater()
        self.chain_panel = ChainPanel(instance, self.session, cls)
        self.chain_panel.runnable(False)
        self.chain_panel.changed.connect(self.refresh_chains)
        self.chain_panel.changed.connect(self.refresh_inputs)
        self.chain_holder.insertWidget(0, self.chain_panel)
        if not spec.steps:
            self.chain_panel.edit.setChecked(True)
        self.refresh_inputs()

    def current_chain(self) -> ChainSpec:
        return self.chain_panel.current_spec()

    def _kind(self) -> str:
        cls = self.chain_panel.cls if self.chain_panel is not None else None
        return cls.input_kind() if cls is not None else "localizations"

    # ------------------------------------------------------------------- job
    def job(self) -> Job:
        spec = self.current_chain()
        chain_file = ""
        if spec.origin is not None and self.chain_panel.instance.chain is None:
            chain_file = str(spec.origin)      # saved and unchanged: by reference
        return Job(chain=spec, chain_file=chain_file, inputs=self.inputs(),
                   output=Output(folder=self.out_folder.text().strip(),
                                 suffix=self.suffix.text(),
                                 locs=self.save_locs.isChecked(),
                                 figures=self.figures.currentData(),
                                 fitted=self.fitted.currentData(),
                                 delete_fitted=self.delete_fitted.isChecked(),
                                 trends=[t.strip() for t in self.trends.text().split(",")
                                         if t.strip()]),
                   preflight=self.preflight.currentData(),
                   rerun=self.rerun.currentData(), origin=self.job_path)

    def load_job(self, path) -> None:
        job = batch.read_job(path)
        self.job_path = Path(path)
        self.tree.clear()
        self.set_chain(job.chain)          # first: what the inputs give depends on it
        for entry in job.inputs:
            self.add_input(entry)
        out = job.output
        self.out_folder.setText(out.folder)
        self.suffix.setText(out.suffix)
        self.save_locs.setChecked(out.locs)
        for box, value in ((self.figures, out.figures), (self.fitted, out.fitted),
                           (self.preflight, job.preflight), (self.rerun, job.rerun)):
            box.setCurrentIndex(max(box.findData(value), 0))
        self.delete_fitted.setChecked(out.delete_fitted)
        self.trends.setText(", ".join(out.trends))
        self.setWindowTitle(f"SMAPpy batch -- {self.job_path.name}")

    def open_job(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open job", folders.start(),
                                              "Batch jobs (*.batch.yaml *.batch.json)")
        if path:
            folders.remember(path)
            self.load_job(path)

    def save_job(self, ask: bool = True) -> Optional[Path]:
        target = self.job_path
        if ask or target is None:
            start = str(target or folders.start("job.batch.yaml"))
            chosen, _ = QFileDialog.getSaveFileName(self, "Save job", start,
                                                    "Batch jobs (*.batch.yaml)")
            if not chosen:
                return None
            target = Path(chosen)
            if not target.name.endswith(".batch.yaml"):
                target = target.with_name(target.name.split(".")[0] + ".batch.yaml")
        self.job_path = target
        batch.write_job(self.job(), target)
        self.setWindowTitle(f"SMAPpy batch -- {target.name}")
        return target

    def validate(self, need_files: bool = True) -> bool:
        problems = batch.validate(self.job(), need_files=need_files)
        for problem in problems:
            self.log.appendPlainText(str(problem))
        errors = sum(p.level == "error" for p in problems)
        self.log.appendPlainText(f"{errors} error(s), {len(problems) - errors} warning(s)")
        return errors == 0

    # --------------------------------------------------------------- running
    def run(self) -> None:
        if self.process is not None:
            return
        watching = self.watch.isChecked()
        if not self.validate(need_files=not watching):
            return
        path = self.save_job(ask=self.job_path is None)
        if path is None:
            return
        # found again, which also clears the statuses: a second run counts
        # afresh, not on top
        self.refresh_inputs()
        self.progress.setRange(0, 0)
        self.progress.show()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in UTF8_ENVIRONMENT.items():
            environment.insert(key, value)
        self.process.setProcessEnvironment(environment)
        self._stopping = False
        if watching:
            args = ["-m", "smappy.cli.batch", "watch", str(path)]
        else:
            args = ["-m", "smappy.cli.batch", "run", str(path)]
            if self.try_one.isChecked():
                args += ["--limit", "1"]
        self.log.appendPlainText(f"$ smappy-batch {' '.join(args[2:])}")
        self.run_button.setEnabled(False)
        self.watch.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.process.start(sys.executable, args)

    def stop(self) -> None:
        """End the batch.  A watch is asked first, and ends after the file in
        hand with its report written; a second Stop, or a plain run, ends the
        program at once."""
        if self.process is None:
            return
        if self.watch.isChecked() and not self._stopping:
            self._stopping = True
            self.process.write(b"stop\n")
            self.log.appendPlainText("stopping after the file in hand "
                                     "(Stop again to end it now)")
            return
        stop_process(self.process)

    def _read(self) -> None:
        # split as bytes and decode whole lines: a read can end inside a
        # character, and decoding each read on its own replaced both halves
        self._buffer += bytes(self.process.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            self.handle_line(line.decode("utf-8", errors="replace"))

    def handle_line(self, line: str) -> None:
        """One line of the runner's output: shown, and read if it is an event."""
        fields = line.rstrip("\r").split("\t")
        event = fields[0][len("SMAPPY_"):] if fields[0].startswith("SMAPPY_") else ""
        if event == "PROGRESS":
            self.statusBar().showMessage(" ".join(fields[1:]))
            return
        self.log.appendPlainText(f"{event.lower()}: {'  '.join(fields[1:])}"
                                 if event else line)
        if event == "WATCHING":
            self.progress.setRange(0, 0)
        elif event == "JOBS":
            self.progress.setRange(0, max(int(fields[1]), 1))
            self.progress.setValue(0)
        elif event in ("START", "DONE", "SKIP", "FAILED") and len(fields) > 1:
            where = fields[1]
            if event == "START" and len(fields) > 2:
                self._running[where] = fields[2]
            # a file skipped as done before has no START: its SKIP names it
            path = (fields[2] if event == "START" else self._running.get(where)
                    or (fields[2] if event == "SKIP" and len(fields) > 2 else ""))
            if path:
                text = {"START": "running", "DONE": "done", "SKIP": "skipped",
                        "FAILED": "failed"}[event]
                self._mark(path, text, fields[-1] if event in ("FAILED", "SKIP") else "")
            if event != "START" and "/" in where:
                self.progress.setValue(int(where.split("/")[0]))
        elif event == "REPORT":
            self.report_path = Path(fields[1])

    def _item_for(self, path: str) -> Optional[QTreeWidgetItem]:
        """The item a data set the runner names is shown by: a file input's
        own, or its line under its folder -- added there if the folder had
        not found it yet, as a watched folder's new data sets are not."""
        target = Path(path).resolve()
        folder_item = None
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            entry = item.data(0, Qt.UserRole)
            where = _resolve(entry.file or entry.folder, self._base()).resolve()
            if entry.file and where == target:
                return item
            if entry.folder and where in target.parents and folder_item is None:
                folder_item = item
        if folder_item is None:
            return None
        for i in range(folder_item.childCount()):
            child = folder_item.child(i)
            if Path(child.data(0, Qt.UserRole)).resolve() == target:
                return child
        from ..datasets import relative
        folder = _resolve(folder_item.data(0, Qt.UserRole).folder, self._base())
        child = QTreeWidgetItem([relative(target, folder), "", ""])
        child.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        child.setData(0, Qt.UserRole, str(target))
        child.setToolTip(0, str(target))
        blocked = self.tree.blockSignals(True)
        child.setCheckState(0, Qt.Checked)
        folder_item.addChild(child)
        self.tree.blockSignals(blocked)
        return child

    def _mark(self, path: str, status: str, why: str) -> None:
        """A data set's status, and its folder's counts."""
        item = self._item_for(path)
        if item is None:
            return
        blocked = self.tree.blockSignals(True)
        item.setText(2, status)
        item.setToolTip(2, why)
        folder = item.parent()
        if folder is not None:
            counts: Dict[str, int] = {}
            for i in range(folder.childCount()):
                text = folder.child(i).text(2)
                if text in ("done", "skipped", "failed"):
                    counts[text] = counts.get(text, 0) + 1
            folder.setText(2, ", ".join(f"{n} {s}" for s, n in counts.items()) or "running")
        self.tree.blockSignals(blocked)

    def _finished(self, code: int, _status) -> None:
        if self._buffer:
            self.handle_line(self._buffer.decode("utf-8", errors="replace"))
            self._buffer = b""
        self.process = None
        self._running = {}
        self.run_button.setEnabled(True)
        self.watch.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
        if self.watch.isChecked() and code == 0:
            self.log.appendPlainText("watch ended; open summary shows what it did")
            return
        self.log.appendPlainText({0: "finished", 1: "finished, with failures",
                                  2: "did not start: the job has errors"}
                                 .get(code, f"stopped ({code})"))

    def open_output(self) -> None:
        """The output folder in the system's file browser; with a session, a
        processed file can be opened in it instead."""
        folder = self.job().output_folder()
        if self.open_result is not None:
            path, _ = QFileDialog.getOpenFileName(self, "Open a result", str(folder),
                                                  "Localizations (*.h5 *.hdf5)")
            if path:
                self.open_result(path)
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def open_page(self) -> None:
        """The summary page, `index.html`, in the system's browser."""
        page = self.job().output_folder() / "index.html"
        if not page.exists():
            QMessageBox.information(self, "Summary", f"No summary page yet: {page} is "
                                    "written once the first file is done.")
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(page)))

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder",
                                                  self.out_folder.text() or folders.start())
        if folder:
            folders.remember(folder)
            self.out_folder.setText(folder)


def main(argv=None) -> int:
    from . import make_app
    app = make_app(argv if argv is not None else sys.argv)
    window = BatchWindow()
    args = (argv if argv is not None else sys.argv)[1:]
    if args:
        window.load_job(args[0])
    window.show()
    return app.exec()
