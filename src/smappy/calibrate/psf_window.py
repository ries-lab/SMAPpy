"""The PSF calibration window: one place to make a PSF calibration, by the method chosen.

Only widgets.  The methods are entries in `psf_methods` -- *bead average*
(the bead calibration's) and *pupil model (uiPSF)* -- each saying which of
the sections below it reads, how it runs, its pages and its bead table; the
window shows the chosen one's and knows none of them.  In situ learning from
blinking molecules will add a *source* choice and a method, not a branch
here (`docs/uipsf/plan.md` §7.5).

What every method shares is the window's: the stacks, *mode* (single channel
or dual colour), *save model* -- saved beside the stacks under the bead
calibration's name and handed to the fitter -- *Save as...*, the bead table
with its marks for the next *Recalculate*, and the worker thread.

torch is imported only by the pupil model's run: without it the window opens,
the pupil model says how to install it, and the bead average works as ever.
"""
from __future__ import annotations

import threading
from dataclasses import fields
from pathlib import Path
from typing import List

from PySide6.QtCore import QObject, QProcess, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QListWidget, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSplitter,
                               QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..gui import folders
from ..gui.params import SettingsForm
from ..gui.widgets import CollapsibleSection
from .core import CalibrationSettings
from .dual import DualColorSettings
from .input import discover_acquisitions
from .psf_methods import AVERAGE_ONLY, METHODS
from .pupil import BeadSettings, OpticsSettings, RunSettings, SplineSettings
from .qt_gui import _bead_specs


def torch_missing():
    """What the pupil model lacks, with the command that brings it; None when a usable torch is here."""
    from .. import torch_install
    problem = torch_install.problem()
    if problem is None:
        return None
    return (f"The pupil model needs PyTorch: {problem}.  Install PyTorch puts {torch_install.described()} "
            f"into this Python; or, in a terminal:\n{' '.join(torch_install.command())}")


class _Worker(QObject):
    done = Signal(str, object)
    progress = Signal(object)

    def __init__(self, kind, function):
        super().__init__()
        self.kind, self.function = kind, function

    def run(self):
        try:
            self.done.emit(self.kind, self.function(self.progress.emit))
        except Exception as error:                       # shown, never fatal
            self.done.emit("error", error)


class PSFCalibrationWindow(QMainWindow):
    """Files and settings on the left; the method's pages and the bead table on the right."""

    # the saved calibration's path and whether it is dual colour: which fitter it goes to
    calibrated = Signal(str, bool)

    def __init__(self, paths=(), parent=None):
        super().__init__(parent)
        from ..gui.help_window import attach_help
        attach_help(self, "Panels/PSF calibration")
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("SMAPpy - PSF calibration")
        self.paths: List[str] = [str(p) for p in paths]
        self.method = METHODS[0]
        self.run_result = None
        self.marks: dict = {}
        self.busy = False
        self._thread = None
        self._cancel = threading.Event()
        self.stack_viewer = None
        self.pages: list = []

        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(self._controls())
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(430)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(lambda _: self.redraw())
        self.bead_table = QTableWidget(0, 0)
        self.bead_table.verticalHeader().setVisible(False)
        self.bead_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bead_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.bead_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bead_table.setMinimumHeight(140)
        self.bead_table.itemSelectionChanged.connect(self._selection_changed)
        toggle = QShortcut(QKeySequence(Qt.Key_Space), self)
        toggle.setContext(Qt.WindowShortcut)
        toggle.activated.connect(self.toggle_beads)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.tabs)
        table_side = QWidget()
        tlayout = QVBoxLayout(table_side)
        tlayout.setContentsMargins(2, 2, 2, 2)
        self.table_hint = QLabel()
        tlayout.addWidget(self.table_hint)
        tlayout.addWidget(self.bead_table)
        row = QHBoxLayout()
        self.toggle_button = QPushButton("Exclude / include")
        self.toggle_button.clicked.connect(self.toggle_beads)
        self.rebuild_button = QPushButton("Recalculate")
        self.rebuild_button.setToolTip("calibrate again from the same beads, with your marks")
        self.rebuild_button.clicked.connect(self.rebuild)
        row.addWidget(self.toggle_button)
        row.addWidget(self.rebuild_button)
        row.addStretch(1)
        tlayout.addLayout(row)
        right.addWidget(table_side)
        right.setSizes([620, 220])

        whole = QSplitter(Qt.Horizontal)
        whole.addWidget(scroll)
        whole.addWidget(right)
        whole.setSizes([390, 1000])
        self.setCentralWidget(whole)
        self.status_bar = self.statusBar()
        self.resize(1400, 900)
        self._refresh_files()
        self._show_method()

    # ------------------------------------------------------------ controls
    def _controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        files = QWidget()
        flayout = QVBoxLayout(files)
        flayout.setContentsMargins(0, 0, 0, 0)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMaximumHeight(120)
        flayout.addWidget(self.file_list)
        brow = QHBoxLayout()
        for label, slot in (("Add files...", self.add_files), ("Add folder...", self.add_directory),
                            ("Remove", self.remove_paths)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            brow.addWidget(button)
        flayout.addLayout(brow)
        self.subdirectories = QCheckBox("search subdirectories")
        self.subdirectories.setChecked(True)
        flayout.addWidget(self.subdirectories)
        layout.addWidget(CollapsibleSection("bead stacks", files, expanded=True))

        def choice(label, items, slot):
            line = QHBoxLayout()
            line.addWidget(QLabel(label))
            box = QComboBox()
            box.addItems(items)
            box.currentTextChanged.connect(slot)
            line.addWidget(box, 1)
            layout.addLayout(line)
            return box
        self.method_box = choice("method", [m.label for m in METHODS], self.method_changed)
        self.mode_box = choice("mode", ["Single channel", "Dual colour"], self.mode_changed)

        self.run_form = SettingsForm(RunSettings)
        layout.addWidget(self.run_form)
        self.optics_form = SettingsForm(OpticsSettings)
        self.model_form = SettingsForm(BeadSettings)
        self.spline_form = SettingsForm(SplineSettings)
        holder = QWidget()
        hlayout = QVBoxLayout(holder)
        hlayout.setContentsMargins(0, 0, 0, 0)
        self.detection_forms = {}
        for cls in (CalibrationSettings, DualColorSettings):
            form = SettingsForm(cls, _bead_specs(cls))
            hlayout.addWidget(form)
            self.detection_forms[cls] = form
        self.sections = {"microscope": CollapsibleSection("microscope", self.optics_form, expanded=True),
                         "model": CollapsibleSection("model", self.model_form, expanded=True),
                         "beads": CollapsibleSection("beads", holder, expanded=True),
                         "spline": CollapsibleSection("spline", self.spline_form, expanded=False)}
        for section in self.sections.values():
            layout.addWidget(section)

        self.notice = QLabel(torch_missing() or "")
        self.notice.setWordWrap(True)
        self.notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.notice)
        self.install_button = QPushButton("Install PyTorch")
        self.install_button.setToolTip("install PyTorch into the Python running SMAPpy; pip's progress shows "
                                       "below, and pressing again stops it")
        self.install_button.clicked.connect(self.install_torch)
        layout.addWidget(self.install_button)
        self._installer = None

        self.run_button = QPushButton("Detect + calibrate")
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip("stop the pupil model's learning after the current iteration; nothing is saved")
        self.stop_button.clicked.connect(self._cancel.set)
        self.quality_button = QPushButton("Calculate fit quality")
        self.quality_button.setToolTip("fit the beads back with the fitter the data will meet")
        self.quality_button.clicked.connect(self.fit_quality)
        self.save_as_button = QPushButton("Save as...")
        self.save_as_button.setToolTip("save this calibration somewhere else, or when save model was off")
        self.save_as_button.clicked.connect(self.save_as)
        self.use_button = QPushButton("Use in the Spline 3D fitter")
        self.use_button.setToolTip("put the saved calibration into the fitter's calibration field")
        self.use_button.clicked.connect(self._use_in_fitter)
        self.browse_button = QPushButton("Browse average stack...")
        self.browse_button.setToolTip("look through the averaged bead stack slice by slice, in its own window")
        self.browse_button.clicked.connect(self.browse_stack)
        for button in (self.run_button, self.stop_button, self.quality_button, self.save_as_button,
                       self.use_button, self.browse_button):
            layout.addWidget(button)
        layout.addStretch(1)
        return panel

    @property
    def is_dual(self) -> bool:
        return self.mode_box.currentText() == "Dual colour"

    @property
    def detection_form(self) -> SettingsForm:
        return self.detection_forms[DualColorSettings if self.is_dual else CalibrationSettings]

    def _show_method(self) -> None:
        """Show what the chosen method reads, and only that."""
        reads = self.method.sections
        for name in ("microscope", "model", "spline"):
            self.sections[name].setVisible(reads[name])
        self.run_form.set_shown({"start_from": reads["start from"]})
        wanted = DualColorSettings if self.is_dual else CalibrationSettings
        shared = {f.name for f in fields(CalibrationSettings)}
        for cls, form in self.detection_forms.items():
            form.setVisible(cls is wanted)
            form.set_shown({name: reads["averaging"] for name in AVERAGE_ONLY if name in shared})
        pupil = self.method.key == "pupil"
        missing = torch_missing()
        self.notice.setText(missing or "")
        self.notice.setVisible(pupil and missing is not None)
        # shown only while torch is missing or too old: nothing to press once it is right
        self.install_button.setVisible(pupil and (missing is not None or self._installer is not None))
        self.stop_button.setVisible(pupil)
        self.quality_button.setVisible(self.method.has_quality)
        self.browse_button.setVisible(self.method.has_stack)
        self.table_hint.setText("beads - space marks a used bead to be excluded"
                                + (" or a rejected one to be included" if self.method.marks_include else "")
                                + ", then Recalculate")
        self.use_button.setText("Use in the Spline 3D 2C fitter" if self.is_dual else "Use in the Spline 3D fitter")
        self._update_buttons()

    def _forget(self, message: str) -> None:
        """A change of method or mode makes the result another calibration's: dropped, settings kept."""
        if self.busy:
            return
        self.run_result = None
        self.marks = {}
        self._clear_pages()
        self.bead_table.setRowCount(0)
        self._show_method()
        self.status_bar.showMessage(message)

    def method_changed(self, label: str) -> None:
        self.method = next(m for m in METHODS if m.label == label)
        self._forget("method changed; files and settings kept - calibrate again")

    def mode_changed(self, _text: str) -> None:
        self._forget("mode changed; files and settings kept - calibrate again")

    def _update_buttons(self) -> None:
        ready = not (self.method.key == "pupil" and torch_missing() is not None)
        done = self.run_result is not None
        self.run_button.setEnabled(ready and not self.busy)
        self.stop_button.setEnabled(self.busy)
        self.rebuild_button.setEnabled(ready and done and not self.busy)
        self.toggle_button.setEnabled(done)
        self.quality_button.setEnabled(done and not self.busy)
        self.save_as_button.setEnabled(done and not self.busy)
        self.browse_button.setEnabled(done)
        self.use_button.setEnabled(done and not self.busy and self.run_result.saved is not None)

    # --------------------------------------------------------------- files
    def _refresh_files(self) -> None:
        self.file_list.clear()
        for path in self.paths:
            self.file_list.addItem(path)

    def add_paths(self, paths) -> None:
        for path in paths:
            if path and str(path) not in self.paths:
                self.paths.append(str(path))
        self._refresh_files()

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Bead stacks", folders.start(),
                                                "Image stacks (*.tif *.tiff *.ome.tif);;All files (*)")
        folders.remember(paths, stack=True)
        self.add_paths(paths)

    def add_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder of bead stacks", folders.start())
        if not path:
            return
        folders.remember(path)
        if self.subdirectories.isChecked():
            try:
                found = [str(p) for p in discover_acquisitions([path])]
            except Exception as error:
                self.error(error)
                return
            self.add_paths(found or [path])
        else:
            self.add_paths([path])

    def remove_paths(self) -> None:
        for item in self.file_list.selectedItems():
            if item.text() in self.paths:
                self.paths.remove(item.text())
        self._refresh_files()

    # ---------------------------------------------------------------- work
    def error(self, exc) -> None:
        self.status_bar.showMessage(f"{type(exc).__name__}: {exc}")
        QMessageBox.warning(self, "PSF calibration", f"{type(exc).__name__}: {exc}")

    def _work(self, kind: str, function) -> None:
        if self.busy:
            return
        self.busy = True
        self._cancel.clear()
        self._update_buttons()
        self._thread = QThread()
        self._thread.setStackSize(32 * 1024 * 1024)
        self._worker = _Worker(kind, function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.done.connect(self._finished)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    def _progress(self, entry) -> None:
        if "message" in entry:
            self.status_bar.showMessage(entry["message"])
            return
        stage = {1: "low orders", 2: "all orders"}.get(entry.get("stage"), "learning")
        objective = entry.get("objective")
        self.status_bar.showMessage(f"{stage}: iteration {entry.get('iteration', 0)}"
                                    + ("" if objective is None else f", -log L {float(objective):.6g}"))

    def _finished(self, kind: str, value) -> None:
        self.busy = False
        if kind == "error":
            self._update_buttons()
            self.error(value)
            return
        if kind == "quality":
            self._show_result(focus="Fit quality")
            return
        self.run_result = value
        self.marks = self.method.initial_marks(value)
        self._show_result()
        if value.saved is not None:
            self.calibrated.emit(str(value.saved), value.dual)

    def _inputs(self) -> dict:
        inputs = dict(detection=self.detection_form.value(), run=self.run_form.value())
        if self.method.sections["microscope"]:
            inputs.update(optics=self.optics_form.value(), model=self.model_form.value(),
                          spline=self.spline_form.value())
            if not inputs["optics"].NA or not inputs["optics"].wavelength_nm:
                raise ValueError("give the microscope's NA and emission wavelength: there is no default for them")
        return inputs

    def run(self) -> None:
        if not self.paths:
            QMessageBox.information(self, "no stacks", "add bead stacks first")
            return
        try:
            inputs = self._inputs()
        except ValueError as error:
            self.error(error)
            return
        method, paths, dual = self.method, list(self.paths), self.is_dual
        self.marks = {}
        self.status_bar.showMessage("detecting beads...")
        self._work("result", lambda progress: method.run(paths, dual, inputs, progress, self._cancel))

    def rebuild(self) -> None:
        if self.run_result is None:
            return
        try:
            inputs = self._inputs()
        except ValueError as error:
            self.error(error)
            return
        method, paths, previous, marks = self.method, list(self.paths), self.run_result, dict(self.marks)
        self.status_bar.showMessage("calibrating again with your marks...")
        self._work("result", lambda progress: method.run(paths, previous.dual, inputs, progress, self._cancel,
                                                         previous=previous, marks=marks))

    def fit_quality(self) -> None:
        if self.run_result is None or not self.method.has_quality:
            return
        method, run = self.method, self.run_result
        self.status_bar.showMessage("fitting the beads back with the "
                                    + ("two-channel" if run.dual else "spline") + " fitter...")
        self._work("quality", lambda progress: method.quality(run))

    # -------------------------------------------------------------- results
    def _clear_pages(self) -> None:
        self.tabs.blockSignals(True)
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        self.tabs.blockSignals(False)
        self.pages = []

    def _show_result(self, focus=None) -> None:
        from matplotlib.figure import Figure

        from ..gui.canvas import canvas_class
        current = focus or (self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else None)
        self._clear_pages()
        self.tabs.blockSignals(True)
        for page in self.method.pages(self.run_result):
            figure = Figure(figsize=(9, 6), constrained_layout=True)
            canvas = canvas_class()(figure)
            canvas.mpl_connect("pick_event", self.pick_pair)
            self.tabs.addTab(canvas, page.title)
            self.pages.append((figure, canvas, page))
        titles = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        if current in titles:
            self.tabs.setCurrentIndex(titles.index(current))
        self.tabs.blockSignals(False)
        self.refresh_table()
        self._update_buttons()
        self.redraw()
        saved = self.run_result.saved
        self.status_bar.showMessage(self.method.summary(self.run_result) + (
            f"; saved {saved.name}" if saved else "; not saved (save model is off)"))
        if self.stack_viewer is not None and self.stack_viewer.isVisible():
            self.browse_stack()

    def redraw(self) -> None:
        """Draw the page in front, with the table's selection and marks."""
        index = self.tabs.currentIndex()
        if self.run_result is None or not 0 <= index < len(self.pages):
            return
        figure, canvas, page = self.pages[index]
        excluded = {b for b, wish in self.marks.items() if wish == "exclude"}
        try:
            figure.clear()
            message = page.draw(figure, self.selected_beads(), excluded)
        except Exception as error:
            self.status_bar.showMessage(f"could not draw: {type(error).__name__}: {error}")
            return
        if isinstance(message, str) and message:
            self.status_bar.showMessage(message)
        canvas.draw_idle()

    def _selection_changed(self) -> None:
        index = self.tabs.currentIndex()
        if 0 <= index < len(self.pages) and self.pages[index][2].follows_selection:
            self.redraw()

    def refresh_table(self) -> None:
        if self.run_result is None:
            self.bead_table.setRowCount(0)
            return
        chosen = self.selected_beads()
        rows = self.method.rows(self.run_result, self.marks)
        self.bead_table.blockSignals(True)
        self.bead_table.setColumnCount(len(self.method.columns))
        self.bead_table.setHorizontalHeaderLabels(self.method.columns)
        self.bead_table.setRowCount(len(rows))
        for i, (values, greyed) in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if greyed:
                    item.setForeground(Qt.gray)
                self.bead_table.setItem(i, column, item)
        self.bead_table.resizeColumnsToContents()
        self.bead_table.blockSignals(False)
        for i in chosen:
            if i < self.bead_table.rowCount():
                self.bead_table.selectRow(i)

    def selected_beads(self) -> List[int]:
        return sorted({index.row() for index in self.bead_table.selectionModel().selectedRows()})

    def toggle_beads(self) -> None:
        """Mark the selected beads for the next Recalculate (`Method.mark`); pressed again, unmark them."""
        if self.run_result is None:
            return
        for bead in self.selected_beads():
            if bead in self.marks:
                del self.marks[bead]
            else:
                self.marks[bead] = self.method.mark(self.run_result, bead)
        self.refresh_table()
        self.redraw()

    def pick_pair(self, event) -> None:
        ids = getattr(event.artist, "pair_ids", None)
        if ids is None or not len(ids):
            return
        # a scatter of several beads says which point was hit; a curve is one bead
        hit = getattr(event, "ind", None)
        index = int(ids[hit[0]] if hit is not None and len(hit) and len(ids) > 1 else list(ids)[0])
        if index < self.bead_table.rowCount():
            self.bead_table.selectRow(index)

    def browse_stack(self) -> None:
        if self.run_result is None or not self.method.has_stack:
            return
        from .stack_view import StackViewer
        volumes, titles, mirror = self.method.stacks(self.run_result)
        calibration = self.run_result.result.calibration
        if self.stack_viewer is None:
            self.stack_viewer = StackViewer(volumes, calibration, titles, mirror, self)
        else:
            self.stack_viewer.calibration = calibration
            self.stack_viewer.mirror_axis = mirror
            self.stack_viewer.set_volumes(volumes, titles)
        self.stack_viewer.show()
        self.stack_viewer.raise_()

    # ------------------------------------------------------- installing torch
    def install_torch(self) -> None:
        """pip, in a process of its own: the window stays usable, its output is the progress, and a second
        press stops it.  torch was never imported here, so what pip installs is used without a restart."""
        if self._installer is not None:
            self._installer.kill()
            return
        from .. import torch_install
        program, *arguments = torch_install.command()
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._installer_output)
        process.finished.connect(self._installer_finished)
        self._installer = process
        self.install_button.setText("Stop installing")
        self.status_bar.showMessage("installing " + torch_install.described() + "...")
        process.start(program, arguments)

    def _installer_output(self) -> None:
        lines = bytes(self._installer.readAllStandardOutput()).decode(errors="replace").strip().splitlines()
        if lines:
            self.status_bar.showMessage(lines[-1][:200])

    def _installer_finished(self, code, _status=None) -> None:
        import importlib
        importlib.invalidate_caches()
        self._installer.deleteLater()
        self._installer = None
        self.install_button.setText("Install PyTorch")
        missing = torch_missing()
        self.status_bar.showMessage("PyTorch is installed: the pupil model is ready" if missing is None
                                    else f"pip stopped (exit code {code}); {missing.splitlines()[0]}")
        self._show_method()

    # ------------------------------------------------------------- saving
    def save_as(self) -> None:
        if self.run_result is None:
            return
        from .pupil import default_save_path
        start = self.run_result.saved or default_save_path(self.paths, self.run_result.dual)
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", str(start), "Calibration (*.h5)")
        if not path:
            return
        try:
            saved = self.method.save_as(self.run_result, path)
        except Exception as error:
            self.error(error)
            return
        self._update_buttons()
        self.status_bar.showMessage(f"saved {saved}")
        self.calibrated.emit(str(saved), self.run_result.dual)

    def _use_in_fitter(self) -> None:
        if self.run_result is None or self.run_result.saved is None:
            return
        self.calibrated.emit(str(self.run_result.saved), self.run_result.dual)
        fitter = "Spline 3D 2C" if self.run_result.dual else "Spline 3D"
        self.status_bar.showMessage(f"the {fitter} fitter now uses {Path(self.run_result.saved).name}")


def show_psf_calibration(paths=()):
    """Open the window, making a QApplication if there is none."""
    from PySide6.QtWidgets import QApplication

    from ..gui import make_app
    standalone = QApplication.instance() is None
    app = make_app()
    window = PSFCalibrationWindow(paths)
    window.show()
    if standalone:
        app.exec()
    return window
