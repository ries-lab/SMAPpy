"""The bead PSF calibration window.

Only widgets live here.  The work is `calibrate.core` (single channel) and
`calibrate.dual` (dual colour), run on a worker thread; the pages are
`calibrate.plots`, functions of the result and of which beads are selected
and excluded, so this window only decides which page is in front and hands
them the table's state.  The mode selector picks single channel or dual
colour, and switching keeps the files and the settings both modes share.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog,
                               QHBoxLayout, QLabel, QListWidget,
                               QMainWindow, QMessageBox, QPushButton, QScrollArea,
                               QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..gui import folders
from ..gui.params import SettingsForm
from ..gui.widgets import CollapsibleSection
from ..plugins import ParamInfo, param_specs
from . import plots
from .core import (CalibrationSettings, build_calibration, calibration_save_defaults,
                   calibration_save_path, collect_beads)
from .dual import LAYOUTS, DualColorSettings, collect_dual_beads, build_dual_calibration
from .input import discover_acquisitions

PAGES = ("Overview", "Transformation", "Fit quality", "Field diagnostics", "Bead diagnostics")
COLUMNS = ("bead", "stack", "x, y (px)", "brightness", "dz (nm)", "correlation",
           "residual", "state")

SETTING_INFO = {
    "roi_size": ParamInfo(label="ROI size", unit="px", min=7),
    "padding": ParamInfo(unit="px", min=1, advanced=True),
    "dz_nm": ParamInfo(label="z step", unit="nm", help="auto: from the acquisition"),
    "detection_sigma_px": ParamInfo(label="detection sigma", unit="px", advanced=True),
    "detection_threshold_sigma": ParamInfo(label="threshold", unit="sigma"),
    "min_distance_px": ParamInfo(label="min distance", unit="px"),
    "max_xy_shift_px": ParamInfo(label="max xy shift", unit="px", advanced=True),
    "max_z_shift_nm": ParamInfo(label="max z shift", unit="nm", advanced=True),
    "alignment_range_nm": ParamInfo(label="alignment range", unit="nm", advanced=True),
    "registration_iterations": ParamInfo(label="iterations", min=1, advanced=True),
    "rejection_mad": ParamInfo(label="rejection (MAD)", advanced=True),
    "smooth_z_nm": ParamInfo(label="smoothing z", unit="nm"),
    "smooth_xy_px": ParamInfo(label="smoothing xy", unit="px", advanced=True),
    "min_beads": ParamInfo(label="min beads", min=1, advanced=True),
    "brightness_range": ParamInfo(label="brightness range", advanced=True,
                                  help="largest accepted max/min; auto disables it"),
    "saturation_adu": ParamInfo(label="saturation", unit="ADU", advanced=True,
                                help="auto: the integer type's maximum"),
    "layout": ParamInfo(choices=LAYOUTS),
    "main_channel": ParamInfo(label="main channel",
                              choices=("left", "right", "upper", "lower")),
    "split_position": ParamInfo(label="split position", unit="px",
                                help="auto: the midpoint"),
    "min_pairs": ParamInfo(label="min pairs", min=1),
    "reprojection_threshold_px": ParamInfo(label="RANSAC radius", unit="px"),
    "transform_axis_limit_px": ParamInfo(label="dx/dy limit", unit="px"),
}


class _Worker(QObject):
    done = Signal(str, object)
    progress = Signal(str)

    def __init__(self, kind, function):
        super().__init__()
        self.kind, self.function = kind, function

    def run(self):
        try:
            self.done.emit(self.kind, self.function())
        except Exception as error:                       # shown, never fatal
            self.done.emit("error", error)


class CalibrationWindow(QMainWindow):
    """Files and settings on the left, plots and the bead table on the right."""

    # the path a saved calibration was written to, and whether it is dual
    # colour -- which decides *which* fitter it belongs in
    calibrated = Signal(str, bool)

    def __init__(self, paths=(), settings=None, parent=None):
        super().__init__(parent)
        from ..gui.help_window import attach_help
        attach_help(self, 'Panels/Bead calibration')
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("SMAPpy - bead calibration")
        self.paths: List[str] = [str(p) for p in paths]
        self.result = None
        self.diagnostics = None
        self.profile_data = None
        self.showing_quality = False
        self.excluded: set = set()
        self.busy = False
        self._thread = None
        self.mode = "Single channel"
        self.stack_viewer: Optional[QWidget] = None

        control = self._controls(settings)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(control)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(430)

        self.tabs = QTabWidget()
        self.pages = []
        for title in PAGES:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            figure = Figure(figsize=(9, 6), constrained_layout=True)
            canvas = FigureCanvasQTAgg(figure)
            canvas.mpl_connect("pick_event", self.pick_pair)
            self.tabs.addTab(canvas, title)
            self.pages.append((figure, canvas))
        self.tabs.currentChanged.connect(lambda _: self.redraw())

        self.bead_table = QTableWidget(0, len(COLUMNS))
        self.bead_table.setHorizontalHeaderLabels(COLUMNS)
        self.bead_table.verticalHeader().setVisible(False)
        self.bead_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bead_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.bead_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bead_table.itemSelectionChanged.connect(self.redraw)
        self.bead_table.setMinimumHeight(140)
        exclude = QShortcut(QKeySequence(Qt.Key_Space), self)
        exclude.setContext(Qt.WindowShortcut)
        exclude.activated.connect(self.toggle_excluded)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.tabs)
        table_side = QWidget()
        tlayout = QVBoxLayout(table_side)
        tlayout.setContentsMargins(2, 2, 2, 2)
        tlayout.addWidget(QLabel("beads - select rows to inspect; space excludes or "
                                 "includes them, then Recalculate"))
        tlayout.addWidget(self.bead_table)
        row = QHBoxLayout()
        self.exclude_button = QPushButton("Exclude / include")
        self.exclude_button.clicked.connect(self.toggle_excluded)
        self.rebuild_button = QPushButton("Recalculate")
        self.rebuild_button.setToolTip("build the calibration again without the excluded beads")
        self.rebuild_button.clicked.connect(self.rebuild)
        row.addWidget(self.exclude_button)
        row.addWidget(self.rebuild_button)
        row.addStretch(1)
        tlayout.addLayout(row)
        right.addWidget(table_side)
        right.setSizes([600, 220])

        whole = QSplitter(Qt.Horizontal)
        whole.addWidget(scroll)
        whole.addWidget(right)
        whole.setSizes([390, 1000])
        self.setCentralWidget(whole)
        self.status_bar = self.statusBar()
        self.resize(1400, 900)
        self.update_mode()
        self._refresh_files()

    # ------------------------------------------------------------ controls
    def _controls(self, settings) -> QWidget:
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
        for label, slot in (("Add files...", self.add_files),
                            ("Add folder...", self.add_directory),
                            ("Remove", self.remove_paths)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            brow.addWidget(b)
        flayout.addLayout(brow)
        self.subdirectories = QCheckBox("search subdirectories")
        self.subdirectories.setChecked(True)
        flayout.addWidget(self.subdirectories)
        layout.addWidget(CollapsibleSection("bead stacks", files, expanded=True))

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("mode"))
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Single channel", "Dual colour"])
        self.mode_box.currentTextChanged.connect(self.mode_changed)
        mode_row.addWidget(self.mode_box, 1)
        layout.addLayout(mode_row)

        # both forms are built once and one is hidden, so switching mode keeps
        # what was typed
        self.settings_holder = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_holder)
        self.settings_layout.setContentsMargins(0, 0, 0, 0)
        self.forms = {}
        for cls in (CalibrationSettings, DualColorSettings):
            form = SettingsForm(cls, param_specs(cls, SETTING_INFO))
            self.settings_layout.addWidget(form)
            self.forms[cls] = form
        if settings is not None:
            self._apply_settings(settings)
        self._show_form()
        layout.addWidget(CollapsibleSection("settings", self.settings_holder, expanded=True))

        self.run_button = QPushButton("Detect + calibrate")
        self.run_button.clicked.connect(self.run)
        layout.addWidget(self.run_button)
        self.quality_button = QPushButton("Calculate fit quality")
        self.quality_button.clicked.connect(self.fit_quality)
        layout.addWidget(self.quality_button)
        self.save_button = QPushButton("Save calibration...")
        self.save_button.clicked.connect(self.save)
        layout.addWidget(self.save_button)
        self.use_button = QPushButton("Use in the Spline 3D fitter")
        self.use_button.setToolTip("put the saved calibration into the fitter's "
                                   "calibration field")
        self.use_button.clicked.connect(self._use_in_fitter)
        self.use_button.setEnabled(False)
        layout.addWidget(self.use_button)

        self.browse_button = QPushButton("Browse average stack...")
        self.browse_button.setToolTip("look through the averaged bead stack slice "
                                      "by slice, in its own window")
        self.browse_button.clicked.connect(self.browse_stack)
        self.browse_button.setEnabled(False)
        layout.addWidget(self.browse_button)
        layout.addStretch(1)
        return panel

    def _apply_settings(self, settings) -> None:
        """Put what a caller passed into whichever forms know those fields."""
        given = {f.name: getattr(settings, f.name) for f in fields(settings)}
        for cls, form in self.forms.items():
            known = {f.name for f in fields(cls)}
            form.set(cls(**{k: v for k, v in given.items() if k in known}))

    def _show_form(self) -> None:
        """Show the mode's form, carrying the shared fields over."""
        wanted = DualColorSettings if self.is_dual else CalibrationSettings
        other = CalibrationSettings if self.is_dual else DualColorSettings
        shared = {f.name for f in fields(CalibrationSettings)}
        try:
            values = self.forms[other].value()
        except (ValueError, RuntimeError):
            values = None
        if values is not None:
            for name in shared:
                field_widget = self.forms[wanted].fields.get(name)
                if field_widget is not None:
                    field_widget.set(getattr(values, name))
        for cls, form in self.forms.items():
            form.setVisible(cls is wanted)
        self.form = self.forms[wanted]

    # ---------------------------------------------------------------- mode
    @property
    def is_dual(self) -> bool:
        return self.mode == "Dual colour"

    def mode_changed(self, text: str) -> None:
        if self.busy:
            self.mode_box.setCurrentText(self.mode)
            return
        self.mode = text
        self.result = self.diagnostics = self.profile_data = None
        self.showing_quality = False
        self.excluded.clear()
        self.bead_table.setRowCount(0)
        self.browse_button.setEnabled(False)
        self._show_form()
        self.update_mode()
        for figure, canvas in self.pages:
            figure.clear()
            canvas.draw_idle()
        self.status_bar.showMessage("mode changed; files and settings kept - "
                                "detect and calibrate again")

    def update_mode(self) -> None:
        for index in (1, 3):                       # transformation pages: dual only
            self.tabs.setTabEnabled(index, self.is_dual)
        # the button says where the calibration goes, which the mode decides
        # (`app.use_calibration`)
        self.use_button.setText("Use in the Spline 3D 2C fitter" if self.is_dual
                                else "Use in the Spline 3D fitter")

    # --------------------------------------------------------------- files
    def _refresh_files(self) -> None:
        self.file_list.clear()
        for path in self.paths:
            self.file_list.addItem(path)
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(f"{len(self.paths)} bead stack path(s)")

    def add_paths(self, paths) -> None:
        for path in paths:
            if path and path not in self.paths:
                self.paths.append(str(path))
        self._refresh_files()

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Bead stacks", folders.start(),
                                                "Image stacks (*.tif *.tiff *.ome.tif);;"
                                                "All files (*)")
        folders.remember(paths, stack=True)
        self.add_paths(paths)

    def add_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder of bead stacks",
                                                folders.start())
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
    def settings(self):
        return self.form.value()

    def error(self, exc) -> None:
        self.status_bar.showMessage(f"{type(exc).__name__}: {exc}")
        QMessageBox.warning(self, "calibration", f"{type(exc).__name__}: {exc}")

    def _work(self, kind: str, function) -> None:
        if self.busy:
            return
        self.busy = True
        for button in (self.run_button, self.quality_button, self.rebuild_button):
            button.setEnabled(False)
        self._thread = QThread()
        self._thread.setStackSize(32 * 1024 * 1024)
        self._worker = _Worker(kind, function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._finished)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    def _finished(self, kind: str, value) -> None:
        self.busy = False
        for button in (self.run_button, self.quality_button, self.rebuild_button):
            button.setEnabled(True)
        if kind == "error":
            self.error(value)
            return
        if kind == "result":
            self.result = value
            self.excluded = {i for i, reason in enumerate(value.reasons)
                             if reason == "manual exclusion"}
            self.diagnostics = self.profile_data = None
            self.showing_quality = False
            self.use_button.setEnabled(False)
            self.browse_button.setEnabled(True)
            self.refresh_table()
            self._refresh_stack_viewer()
            self._show_resolved_settings(value)
            self.status_bar.showMessage(f"{int(np.sum(value.accepted))} of "
                                    f"{len(value.beads.records)} beads used")
        elif kind == "quality":
            self.diagnostics, self.profile_data = value
            self.showing_quality = True
            self.tabs.setCurrentIndex(2)
        self.redraw()

    def run(self) -> None:
        if not self.paths:
            QMessageBox.information(self, "no stacks", "add bead stacks first")
            return
        settings = self.settings()
        paths = list(self.paths)
        dual = self.is_dual
        self.excluded.clear()
        self.tabs.setCurrentIndex(0)
        self.status_bar.showMessage("detecting beads...")

        def job():
            if dual:
                beads = collect_dual_beads(paths, settings)
                return build_dual_calibration(beads)
            return build_calibration(collect_beads(paths, settings))

        self._work("result", job)

    def rebuild(self) -> None:
        if self.result is None:
            return
        beads = self.result.beads
        excluded = sorted(self.excluded)
        dual = self.is_dual

        def job():
            if dual:
                return build_dual_calibration(beads, excluded=excluded)
            return build_calibration(beads, excluded=excluded)

        self.status_bar.showMessage(f"rebuilding without {len(excluded)} bead(s)...")
        self._work("result", job)

    def fit_quality(self) -> None:
        """Refit the calibration beads with the fitter the data will meet.

        Single channel that is the spline fitter on one ROI per bead; dual
        colour it is `smappy.dualfit`'s global fit over both channels at once,
        sharing x, y and z -- the two-colour workflow itself, run backwards on
        the beads.  A pair therefore has one z, not one per channel.
        """
        if self.result is None:
            return
        result = self.result
        dual = self.is_dual

        def job():
            from .validation import (aligned_midline_profiles,
                                     fit_bead_diagnostics,
                                     fit_paired_bead_diagnostics, paired_profiles)
            if dual:
                fitted = fit_paired_bead_diagnostics(result)
                result.registration.refits = {"global": fitted}
                return fitted, paired_profiles(result)
            return (fit_bead_diagnostics(result), aligned_midline_profiles(result))

        self.status_bar.showMessage("fitting the beads back with the "
                                    + ("two-channel" if dual else "spline")
                                    + " fitter...")
        self._work("quality", job)

    # -------------------------------------------------------------- table
    def selected_beads(self) -> List[int]:
        return sorted({index.row() for index in self.bead_table.selectionModel().selectedRows()})

    def toggle_excluded(self) -> None:
        for i in self.selected_beads():
            self.excluded.symmetric_difference_update({i})
        self.refresh_table()
        self.redraw()

    def refresh_table(self) -> None:
        result = self.result
        if result is None:
            self.bead_table.setRowCount(0)
            return
        chosen = self.selected_beads()
        self.bead_table.blockSignals(True)
        self.bead_table.setRowCount(len(result.beads.records))
        for i, record in enumerate(result.beads.records):
            state = "manual exclusion" if i in self.excluded else result.reasons[i]
            accepted = getattr(result, "transform_accepted", None)
            if accepted is not None and i not in self.excluded and accepted[i]:
                state = ("PSF + transformation" if result.accepted[i]
                         else "transformation only: " + state)
            values = (i, record["stack"] + 1, f"{record['x_px']}, {record['y_px']}",
                      f"{record['brightness_adu']:.4g}",
                      f"{result.shifts[i, 0] * result.calibration.dz:.1f}",
                      f"{result.correlations[i]:.3f}", f"{result.residuals[i]:.3f}", state)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if i in self.excluded:
                    item.setForeground(Qt.gray)
                self.bead_table.setItem(i, column, item)
        self.bead_table.resizeColumnsToContents()
        self.bead_table.blockSignals(False)
        for i in chosen:
            if i < self.bead_table.rowCount():
                self.bead_table.selectRow(i)

    # ------------------------------------------------------------ drawing
    def redraw(self) -> None:
        """Draw the page in front; a page is drawn when it is looked at."""
        if self.result is None:
            return
        page = self.tabs.currentIndex()
        figure, canvas = self.pages[page]
        result, selected, excluded = self.result, self.selected_beads(), self.excluded
        try:
            if page == 0:
                draw = plots.draw_pairs if self.is_dual else plots.draw_overview
                draw(figure, result, selected, excluded)
            elif page == 1:
                plots.draw_transformation(figure, result, selected, excluded)
            elif page == 3:
                plots.draw_field(figure, result, selected, excluded)
            elif page == 2:
                if self.diagnostics is None:
                    return
                if self.is_dual:
                    plots.draw_paired_fit_quality(figure, result, self.diagnostics,
                                                  self.profile_data, selected)
                else:
                    self.status_bar.showMessage(plots.draw_fit_quality(
                        figure, self.diagnostics, self.profile_data, selected))
            else:
                plots.draw_beads(figure, result, selected, excluded, dual=self.is_dual)
        except Exception as error:
            self.status_bar.showMessage(f"could not draw: {type(error).__name__}: {error}")
            return
        canvas.draw_idle()

    def pick_pair(self, event) -> None:
        ids = getattr(event.artist, "pair_ids", None)
        if ids is None or not len(ids):
            return
        index = int(np.atleast_1d(ids)[0])
        if index < self.bead_table.rowCount():
            self.bead_table.selectRow(index)

    def _show_resolved_settings(self, result) -> None:
        """Put the values a run actually used behind its *auto* fields.

        The settings that say "auto" resolve against the bead stacks -- the z
        step from the acquisition, the saturation from the camera's integer
        type, the split from the image width -- and until a run has happened
        there is nothing to resolve them against.  Afterwards there is, so the
        fields say what they did rather than only that they decided.
        """
        beads = result.beads
        channels = getattr(beads, 'channels', None)     # dual: the two halves
        hints = {'dz_nm': getattr(channels[0] if channels else beads,
                                  'dz_nm', None)}
        records = list(getattr(beads, 'records', ()) or ())
        if records:
            # a pair's record wraps one per channel; both saw the same camera
            first = records[0].get('channel_records', [records[0]])[0]
            hints['saturation_adu'] = first.get('saturation_adu')
        geometry = getattr(result.calibration, 'geometry', None)
        if isinstance(geometry, dict):
            hints['split_position'] = geometry.get('split_position')
        self.form.set_hints({k: v for k, v in hints.items() if v is not None})

    # -------------------------------------------------------- stack viewer
    def average_stacks(self):
        """The averaged bead stack(s) to browse, and what to call them.

        The *raw* average, not the smoothed spline model: it is what the
        beads actually gave, so a bad bead or a registration error still shows
        in it.  Dual colour has one per channel, on a shared z grid.
        """
        result = self.result
        if result is None:
            return [], [], None
        channels = getattr(getattr(result, "registration", None),
                           "channel_raw_psfs", None)
        if channels is not None and len(channels) > 1:
            return (list(channels), ["main", "secondary"],
                    result.beads.geometry.get("mirror_axis_xy"))
        return [np.asarray(result.raw_psf)], ["average bead stack"], None

    def browse_stack(self) -> None:
        volumes, titles, mirror = self.average_stacks()
        if not volumes:
            QMessageBox.information(self, "no stack", "calibrate first")
            return
        from .stack_view import StackViewer
        calibration = self.result.calibration
        if self.stack_viewer is None:
            self.stack_viewer = StackViewer(volumes, calibration, titles, mirror, self)
        else:
            self.stack_viewer.calibration = calibration
            self.stack_viewer.mirror_axis = mirror
            self.stack_viewer.set_volumes(volumes, titles)
        self.stack_viewer.show()
        self.stack_viewer.raise_()

    def _refresh_stack_viewer(self) -> None:
        """A recalculated result belongs in an already open viewer."""
        if self.stack_viewer is not None and self.stack_viewer.isVisible():
            self.browse_stack()

    # ------------------------------------------------------------- saving
    def save(self) -> None:
        if self.result is None:
            return
        folder, name = calibration_save_defaults(self.paths, dual=self.is_dual)
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration",
                                              str(Path(folder) / (name + ".h5")),
                                              "Calibration (*.h5 *.hdf5)")
        if not path:
            return
        if not path.lower().endswith(".hdf5"):
            # one .h5, whether the name was typed without it or a dialog added another
            path = str(calibration_save_path(path))
        try:
            self.result.save(path, overwrite=True)
        except Exception as error:
            self.error(error)
            return
        self.saved_path = path
        self.saved_dual = self.is_dual
        self.use_button.setEnabled(True)
        self.status_bar.showMessage(f"saved {path}")
        self.calibrated.emit(path, self.is_dual)

    def _use_in_fitter(self) -> None:
        path = getattr(self, "saved_path", None)
        if not path:
            return
        dual = getattr(self, "saved_dual", self.is_dual)
        self.calibrated.emit(path, dual)
        fitter = "Spline 3D 2C" if dual else "Spline 3D"
        self.status_bar.showMessage(f"the {fitter} fitter now uses {Path(path).name}")


def show_calibration_qt(paths=(), settings=None):
    """Open the window, making a QApplication if there is none."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    standalone = app is None
    if standalone:
        app = QApplication([])
    from ..gui.widgets import apply_style
    apply_style(app)                 # the same editable fields as the main GUI
    window = CalibrationWindow(paths, settings)
    window.show()
    if standalone:
        app.exec()
    return window
