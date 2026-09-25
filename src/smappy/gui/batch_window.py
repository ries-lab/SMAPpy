"""The batch window: a list of files, a chain, where the results go, and Run.

The window is a client of the job file and nothing more.  **Run** writes the
job (`batch.write_job`) and starts ``smappy-batch run`` on it as a separate
process, reading the ``SMAPPY_*`` lines it prints -- the way MPS's GUI runs
its batch -- so a crash or an exhausted memory costs the batch and never
this window or the session behind it, and what the window did can be done
again from a terminal with the file it wrote.

The same class is ``smappy-batch-gui`` on its own and *Tools -> Batch...* in
smappy; the only difference is whether there is a session to open a result
in.  The chain is edited with the `ChainPanel` the plugin tabs use, so a chain
reads and edits the same everywhere; here it cannot be run on the session,
only saved and run over the files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout, QWidget)

from .. import batch, chain as chains, plugins
from ..batch import Input, Job, Output
from ..chain import ChainSpec
from ..session import Session
from ..workspace import Instance
from .chain_panel import ChainPanel

COLUMNS = ("use", "input", "pattern", "overrides", "status")


class BatchWindow(QMainWindow):
    """Files, a chain, output options, and a subprocess that runs them."""

    def __init__(self, session: Optional[Session] = None,
                 open_result: Optional[Callable[[str], None]] = None, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("SMAPpy batch")
        self.setAcceptDrops(True)
        # the chain panel needs a session to exist; this one is never run on
        self.session = Session()
        self.open_result = open_result
        self.job_path: Optional[Path] = None
        self.process: Optional[QProcess] = None
        self._buffer = ""
        self._running: Dict[str, int] = {}
        self.report_path: Optional[Path] = None

        central = QWidget()
        root = QVBoxLayout(central)
        split = QSplitter(Qt.Vertical)
        root.addWidget(split, 1)

        # ---- inputs
        inputs = QGroupBox("files")
        column = QVBoxLayout(inputs)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (0, 2, 3, 4):
            header.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip("a file, or a folder and the pattern its files match; "
                              "overrides are `key.field: value`, as in the job file")
        column.addWidget(self.table)
        row = QHBoxLayout()
        for label, slot in (("add files...", self.add_files),
                            ("add folder...", self.add_folder),
                            ("remove", self.remove_selected),
                            ("clear", lambda: self.table.setRowCount(0))):
            button = QPushButton(label)
            button.clicked.connect(slot)
            row.addWidget(button)
        row.addStretch(1)
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

        # ---- output
        out_box = QGroupBox("output")
        form = QFormLayout(out_box)
        self.out_folder = QLineEdit()
        self.out_folder.setPlaceholderText("beside the job file: <job>_output")
        browse = QToolButton(text="...")
        browse.clicked.connect(self._browse_output)
        line = QHBoxLayout()
        line.addWidget(self.out_folder, 1)
        line.addWidget(browse)
        form.addRow("folder", line)
        self.suffix = QLineEdit(Output.suffix)
        form.addRow("file suffix", self.suffix)
        self.save_locs = QCheckBox("save the processed localizations", checked=True)
        form.addRow("", self.save_locs)
        self.figures = self._combo(batch.FIGURES, "png")
        form.addRow("figures", self.figures)
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
        split.addWidget(out_box)

        # ---- running
        run_box = QWidget()
        column = QVBoxLayout(run_box)
        column.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        for label, slot in (("open job...", self.open_job), ("save job...", self.save_job),
                            ("validate", self.validate)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.try_one = QCheckBox("first file only")
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.results_button = QPushButton("open output")
        self.results_button.clicked.connect(self.open_output)
        for w in (self.try_one, self.run_button, self.stop_button, self.results_button):
            buttons.addWidget(w)
        column.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        column.addWidget(self.progress)
        self.log = QPlainTextEdit(readOnly=True, maximumBlockCount=5000)
        self.log.setMinimumHeight(120)
        column.addWidget(self.log, 1)
        split.addWidget(run_box)
        for index, stretch in enumerate((1, 3, 0, 1)):
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
    def add_input(self, entry: Input, status: str = "") -> None:
        n = self.table.rowCount()
        self.table.insertRow(n)
        use = QTableWidgetItem()
        use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        use.setCheckState(Qt.Checked if entry.include else Qt.Unchecked)
        self.table.setItem(n, 0, use)
        where = QTableWidgetItem(entry.file or entry.folder)
        where.setData(Qt.UserRole, "file" if entry.file else "folder")
        where.setToolTip(("folder, recursive: " if entry.folder else "file: ")
                         + (entry.file or entry.folder))
        self.table.setItem(n, 1, where)
        pattern = QTableWidgetItem(", ".join(entry.pattern) if entry.folder else "")
        if entry.file:
            pattern.setFlags(Qt.ItemIsEnabled)
        self.table.setItem(n, 2, pattern)
        overrides = "; ".join(f"{k}: {v}" for k, v in entry.overrides.items())
        self.table.setItem(n, 3, QTableWidgetItem(overrides))
        state = QTableWidgetItem(status)
        state.setFlags(Qt.ItemIsEnabled)
        self.table.setItem(n, 4, state)

    def inputs(self) -> List[Input]:
        import yaml
        found = []
        for r in range(self.table.rowCount()):
            where = self.table.item(r, 1)
            if where is None or not where.text().strip():
                continue
            text = where.text().strip()
            overrides = {}
            written = self.table.item(r, 3).text().strip() if self.table.item(r, 3) else ""
            if written:
                # `key.field: value; key.field: value`, read as YAML so a number
                # is a number and a list is a list
                for part in written.split(";"):
                    if ":" in part:
                        key, value = part.split(":", 1)
                        overrides[key.strip()] = yaml.safe_load(value.strip())
            include = self.table.item(r, 0).checkState() == Qt.Checked
            if where.data(Qt.UserRole) == "folder":
                pattern = self.table.item(r, 2).text() if self.table.item(r, 2) else ""
                found.append(Input(folder=text, include=include, overrides=overrides,
                                   pattern=[p.strip() for p in pattern.split(",")
                                            if p.strip()]))
            else:
                found.append(Input(file=text, include=include, overrides=overrides))
        return found

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add files")
        for path in paths:
            self.add_input(Input(file=path))

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add folder")
        if not folder:
            return
        kind = self._kind()
        pattern, ok = QInputDialog.getText(
            self, "Pattern", "files matching (comma-separated):",
            text=", ".join(batch.PATTERNS[kind]))
        if ok:
            self.add_input(Input(folder=folder, pattern=[p.strip() for p in pattern.split(",")
                                                         if p.strip()]))

    def remove_selected(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self.add_input(Input(folder=str(path)))
            elif path.name.endswith(chains.SUFFIX):
                self.set_chain(chains.read(path))
            elif path.name.endswith((".batch.yaml", ".batch.json")):
                self.load_job(path)
            elif path.exists():
                self.add_input(Input(file=str(path)))

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
        self.chain_holder.insertWidget(0, self.chain_panel)
        if not spec.steps:
            self.chain_panel.edit.setChecked(True)

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
                                 delete_fitted=self.delete_fitted.isChecked()),
                   preflight=self.preflight.currentData(),
                   rerun=self.rerun.currentData(), origin=self.job_path)

    def load_job(self, path) -> None:
        job = batch.read_job(path)
        self.job_path = Path(path)
        self.table.setRowCount(0)
        for entry in job.inputs:
            self.add_input(entry)
        self.set_chain(job.chain)
        out = job.output
        self.out_folder.setText(out.folder)
        self.suffix.setText(out.suffix)
        self.save_locs.setChecked(out.locs)
        for box, value in ((self.figures, out.figures), (self.fitted, out.fitted),
                           (self.preflight, job.preflight), (self.rerun, job.rerun)):
            box.setCurrentIndex(max(box.findData(value), 0))
        self.delete_fitted.setChecked(out.delete_fitted)
        self.setWindowTitle(f"SMAPpy batch -- {self.job_path.name}")

    def open_job(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open job", "",
                                              "Batch jobs (*.batch.yaml *.batch.json)")
        if path:
            self.load_job(path)

    def save_job(self, ask: bool = True) -> Optional[Path]:
        target = self.job_path
        if ask or target is None:
            start = str(target or Path.cwd() / "job.batch.yaml")
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

    def validate(self) -> bool:
        problems = batch.validate(self.job())
        for problem in problems:
            self.log.appendPlainText(str(problem))
        errors = sum(p.level == "error" for p in problems)
        self.log.appendPlainText(f"{errors} error(s), {len(problems) - errors} warning(s)")
        return errors == 0

    # --------------------------------------------------------------- running
    def run(self) -> None:
        if self.process is not None:
            return
        if not self.validate():
            return
        path = self.save_job(ask=self.job_path is None)
        if path is None:
            return
        for r in range(self.table.rowCount()):
            # a folder's counts too: a second run counts afresh, not on top
            self.table.item(r, 4).setText("")
            self.table.item(r, 4).setData(Qt.UserRole, None)
        self.progress.setRange(0, 0)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        args = ["-m", "smappy.cli.batch", "run", str(path)]
        if self.try_one.isChecked():
            args += ["--limit", "1"]
        self.log.appendPlainText(f"$ smappy-batch {' '.join(args[2:])}")
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.process.start(sys.executable, args)

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()

    def _read(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self.handle_line(line)

    def handle_line(self, line: str) -> None:
        """One line of the runner's output: shown, and read if it is an event."""
        fields = line.rstrip("\r").split("\t")
        event = fields[0][len("SMAPPY_"):] if fields[0].startswith("SMAPPY_") else ""
        if event == "PROGRESS":
            self.statusBar().showMessage(" ".join(fields[1:]))
            return
        self.log.appendPlainText(f"{event.lower()}: {'  '.join(fields[1:])}"
                                 if event else line)
        if event == "JOBS":
            self.progress.setRange(0, max(int(fields[1]), 1))
            self.progress.setValue(0)
        elif event in ("START", "DONE", "SKIP", "FAILED") and len(fields) > 1:
            where = fields[1]
            if event == "START" and len(fields) > 2:
                self._running[where] = self._row_for(fields[2])
            row = self._running.get(where, self._row_for(fields[2]) if len(fields) > 2 else -1)
            if row >= 0:
                text = {"START": "running", "DONE": "done", "SKIP": "skipped",
                        "FAILED": "failed"}[event]
                self._mark(row, text, fields[-1] if event in ("FAILED", "SKIP") else "")
            if event != "START":
                self.progress.setValue(int(where.split("/")[0]))
        elif event == "REPORT":
            self.report_path = Path(fields[1])

    def _row_for(self, path: str) -> int:
        """The input row a file came from: its own, or its folder's."""
        target = Path(path).resolve()
        folder_row = -1
        for r in range(self.table.rowCount()):
            text = self.table.item(r, 1).text().strip()
            where = Path(text).expanduser().resolve()
            if self.table.item(r, 1).data(Qt.UserRole) == "file" and where == target:
                return r
            if where in target.parents and folder_row < 0:
                folder_row = r
        return folder_row

    def _mark(self, row: int, status: str, why: str) -> None:
        item = self.table.item(row, 4)
        if self.table.item(row, 1).data(Qt.UserRole) == "folder":
            counts = dict(item.data(Qt.UserRole) or {})
            if status != "running":
                counts[status] = counts.get(status, 0) + 1
            item.setData(Qt.UserRole, counts)
            item.setText(", ".join(f"{n} {s}" for s, n in counts.items()) or "running")
        else:
            item.setText(status)
        if why:
            item.setToolTip(why)

    def _finished(self, code: int, _status) -> None:
        if self._buffer:
            self.handle_line(self._buffer)
            self._buffer = ""
        self.process = None
        self._running = {}
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
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

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder",
                                                  self.out_folder.text())
        if folder:
            self.out_folder.setText(folder)


def main(argv=None) -> int:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    try:
        from .widgets import apply_style
        apply_style(app)
    except Exception:
        pass
    window = BatchWindow()
    args = (argv if argv is not None else sys.argv)[1:]
    if args:
        window.load_job(args[0])
    window.show()
    return app.exec()
