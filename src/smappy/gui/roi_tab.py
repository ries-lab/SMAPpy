"""The ROI tab: analysis ROIs over the session's files, in Qt.

The model is `smappy.roi_manager`'s project, backed by the session
(`SessionROIs`), so ROIs see the table the image shows, under the layer's
filter and grouping.  ROIs, their review state and their evaluation runs are
part of the session and are saved with the localization file, so closing and
reopening comes back where it was.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGridLayout, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMessageBox, QProgressDialog,
                               QPushButton, QScrollArea, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..roi_manager.plugins import DensityPeaks, Histograms
from ..session import Session
from .widgets import CollapsibleSection, detach_to_window

# outline colours, as in the standalone manager
UNREVIEWED, REVIEWED, EXCLUDED, ACTIVE = "#ffb300", "#2e9e4f", "#8a8a8a", "#2f7fd0"


def roi_colour(roi, active: bool) -> str:
    if active:
        return ACTIVE
    if not roi.use:
        return EXCLUDED
    return REVIEWED if roi.reviewed else UNREVIEWED


class ROITab(QWidget):
    """Files, geometry, candidate finding, the ROI table, evaluation."""

    def __init__(self, session: Session, view=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.view = view
        self.active: Optional[str] = None          # the selected ROI's id
        self._rows: List[str] = []                 # table row -> roi id
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---------------------------------------------------------- files
        files = QWidget()
        frow = QHBoxLayout(files)
        frow.setContentsMargins(0, 0, 0, 0)
        self.file = QComboBox()
        self.file.currentIndexChanged.connect(self._on_file)
        self.file_count = QLabel("")
        frow.addWidget(self.file, 1)
        frow.addWidget(self.file_count)
        layout.addWidget(files)

        # -------------------------------------------------------- geometry
        geometry = QWidget()
        form = QFormLayout(geometry)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(2)
        self.shape = QComboBox()
        self.shape.addItems(["circle", "square"])
        self.size = QDoubleSpinBox(minimum=1, maximum=1e6, decimals=0, suffix=" nm")
        self.size.setToolTip("circle diameter, or the side of a square")
        self.size.setKeyboardTracking(False)
        self.shape.currentTextChanged.connect(self._on_geometry)
        self.size.valueChanged.connect(self._on_geometry)
        form.addRow("shape", self.shape)
        form.addRow("size", self.size)
        layout.addWidget(CollapsibleSection("geometry", geometry, expanded=True))

        # ------------------------------------------------------------ find
        find = QWidget()
        fform = QFormLayout(find)
        fform.setContentsMargins(0, 0, 0, 0)
        fform.setVerticalSpacing(2)
        self.finder: Dict[str, QDoubleSpinBox] = {}
        labels = {"bin_nm": "bin", "sigma_nm": "smoothing", "separation_nm": "separation",
                  "count_radius_nm": "count radius", "min_count": "min localizations"}
        for name, default in DensityPeaks.defaults.items():
            if isinstance(default, int) and not isinstance(default, bool):
                w = QSpinBox(minimum=1, maximum=10 ** 6)
                w.setValue(default)
            else:
                w = QDoubleSpinBox(minimum=0.1, maximum=1e6, decimals=1, suffix=" nm")
                w.setValue(float(default))
            w.setKeyboardTracking(False)
            self.finder[name] = w
            fform.addRow(labels.get(name, name), w)
        buttons = QHBoxLayout()
        self.find_button = QPushButton("Find candidates")
        self.find_button.setToolTip("density peaks in this file's filtered localizations")
        self.find_button.clicked.connect(self._find)
        buttons.addWidget(self.find_button)
        fform.addRow(buttons)
        layout.addWidget(CollapsibleSection("find", find, expanded=True))

        # ------------------------------------------------------------ table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ROI", "reviewed", "use", "comment"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setMinimumHeight(160)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.itemChanged.connect(self._on_item)
        rois = QWidget()
        rlayout = QVBoxLayout(rois)
        rlayout.setContentsMargins(0, 0, 0, 0)
        rlayout.addWidget(self.table)
        grid = QGridLayout()
        for i, (label, tip, slot) in enumerate((
                ("Accept", "mark the selected ROI reviewed", self._accept),
                ("Accept file", "mark every ROI of this file reviewed", self._accept_file),
                ("Toggle use", "include or exclude the selected ROI", self._toggle_use),
                ("Remove", "remove the selected ROI", self._remove))):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            grid.addWidget(b, i // 2, i % 2)
        rlayout.addLayout(grid)
        self.add_mode = QCheckBox("click the image to add an ROI")
        self.add_mode.setToolTip("while this is on, a click in the render window "
                                 "puts a new ROI there; otherwise a click selects one")
        rlayout.addWidget(self.add_mode)
        layout.addWidget(CollapsibleSection("ROIs", rois, expanded=True))

        # -------------------------------------------------------- evaluate
        evaluate = QWidget()
        elayout = QVBoxLayout(evaluate)
        elayout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.evaluate_button = QPushButton("Evaluate all")
        self.evaluate_button.setToolTip("statistics for every reviewed, included ROI")
        self.evaluate_button.clicked.connect(self._evaluate)
        self.histogram_button = QPushButton("Histograms")
        self.histogram_button.setToolTip("histograms of the current results")
        self.histogram_button.clicked.connect(self._histograms)
        row.addWidget(self.evaluate_button)
        row.addWidget(self.histogram_button)
        elayout.addLayout(row)
        self.results = QTableWidget(0, 2)
        self.results.setHorizontalHeaderLabels(["measure", "value"])
        self.results.verticalHeader().setVisible(False)
        self.results.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.results.setFixedHeight(110)
        elayout.addWidget(self.results)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: gray")
        elayout.addWidget(self.summary)
        layout.addWidget(CollapsibleSection("evaluate", evaluate, expanded=True))
        layout.addStretch(1)

        if view is not None:
            view.picked.connect(self._picked)
        session.on_change(self._on_session)
        self.refresh(files_changed=True)

    # ---------------------------------------------------------------- data
    @property
    def project(self):
        return self.session.rois

    def current_file(self) -> Optional[str]:
        return self.file.currentData()

    def _on_session(self, what: str) -> None:
        if what in ("locs", "layers", "rois"):
            self.refresh(files_changed=True)
        elif what in ("layer", "regrouped"):
            self.project.sync()

    # ------------------------------------------------------------ refresh
    def refresh(self, files_changed: bool = False) -> None:
        project = self.project
        self._loading = True
        if files_changed:
            current = self.current_file()
            self.file.clear()
            for source in project.sources.values():
                self.file.addItem(source.name, source.id)
            if current is not None:
                i = self.file.findData(current)
                if i >= 0:
                    self.file.setCurrentIndex(i)
            elif project.navigation.get("file") is not None:
                i = self.file.findData(project.navigation["file"])
                if i >= 0:
                    self.file.setCurrentIndex(i)
            saved = project.navigation.get("roi")     # come back where we were
            if saved in project.rois:
                self.active = saved
            self.shape.setCurrentText(project.shape)
            self.size.setValue(project.size_nm)
        self._loading = False
        self._fill_table()
        self._update_counts()
        self.draw()

    def _fill_table(self) -> None:
        project = self.project
        file_id = self.current_file()
        rois = project.rois_of(file_id) if file_id else []
        self._loading = True
        self.table.setRowCount(len(rois))
        self._rows = []
        for row, roi in enumerate(rois):
            self._rows.append(roi.id)
            centre = QTableWidgetItem(f"{roi.center[0]:.0f}, {roi.center[1]:.0f}")
            centre.setFlags(centre.flags() & ~Qt.ItemIsEditable)
            centre.setForeground(QColor(roi_colour(roi, roi.id == self.active)))
            self.table.setItem(row, 0, centre)
            for column, on in ((1, roi.reviewed), (2, roi.use)):
                item = QTableWidgetItem()
                item.setFlags((item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
                item.setCheckState(Qt.Checked if on else Qt.Unchecked)
                self.table.setItem(row, column, item)
            self.table.setItem(row, 3, QTableWidgetItem(roi.comment))
            if roi.id == self.active:
                self.table.selectRow(row)
        self.table.resizeColumnsToContents()
        self._loading = False

    def _update_counts(self) -> None:
        project = self.project
        file_id = self.current_file()
        rois = project.rois_of(file_id) if file_id else []
        reviewed = sum(1 for r in rois if r.reviewed)
        used = sum(1 for r in rois if r.reviewed and r.use)
        self.file_count.setText(f"{len(rois)} ROIs, {reviewed} reviewed, {used} used")
        rows = project.results()
        self.summary.setText(f"{len(rows)} current results over all files"
                             if rows else "no current results")

    # -------------------------------------------------------------- edits
    def _on_file(self) -> None:
        if self._loading:
            return
        self.active = None
        self.project.navigation["file"] = self.current_file()
        self._fill_table()
        self._update_counts()
        self.draw()

    def _on_geometry(self) -> None:
        if self._loading:
            return
        try:
            self.project.set_geometry(self.size.value(), self.shape.currentText())
        except ValueError as e:
            QMessageBox.warning(self, "geometry", str(e))
            return
        self.draw()

    def _on_select(self) -> None:
        if self._loading:
            return
        rows = self.table.selectionModel().selectedRows()
        self.active = self._rows[rows[0].row()] if rows else None
        self.project.navigation["roi"] = self.active
        self._recolour()
        self.show_active()
        self._show_result()

    def _on_item(self, item) -> None:
        if self._loading or item.row() >= len(self._rows):
            return
        roi = self.project.rois[self._rows[item.row()]]
        if item.column() == 1:
            roi.reviewed = item.checkState() == Qt.Checked
        elif item.column() == 2:
            roi.use = item.checkState() == Qt.Checked
        elif item.column() == 3:
            roi.comment = item.text()
        self._recolour()
        self._update_counts()
        self.draw()

    def _selected(self):
        return self.project.rois.get(self.active) if self.active else None

    def _accept(self) -> None:
        roi = self._selected()
        if roi is not None:
            roi.reviewed = True
            self._fill_table()
            self._update_counts()
            self.draw()

    def _accept_file(self) -> None:
        for roi in self.project.rois_of(self.current_file()):
            roi.reviewed = True
        self._fill_table()
        self._update_counts()
        self.draw()

    def _toggle_use(self) -> None:
        roi = self._selected()
        if roi is not None:
            roi.use = not roi.use
            self._fill_table()
            self._update_counts()
            self.draw()

    def _remove(self) -> None:
        roi = self._selected()
        if roi is not None:
            self.project.rois.pop(roi.id, None)
            self.active = None
            self._fill_table()
            self._update_counts()
            self.draw()

    def add_at(self, x: float, y: float) -> None:
        """A new manual ROI, from a click in the image."""
        file_id = self.current_file()
        if file_id is None:
            return
        roi = self.project.add_roi(file_id, [float(x), float(y)])
        self.active = roi.id
        self._fill_table()
        self._update_counts()
        self.draw()

    # --------------------------------------------------------------- work
    def _find(self) -> None:
        file_id = self.current_file()
        if file_id is None:
            return
        parameters = {name: (w.value() if isinstance(w, QSpinBox) else float(w.value()))
                      for name, w in self.finder.items()}
        try:
            found = self.project.find(file_id, parameters=parameters)
        except Exception as e:
            QMessageBox.warning(self, "find candidates", f"{type(e).__name__}: {e}")
            return
        self._fill_table()
        self._update_counts()
        self.draw()
        self.summary.setText(f"{len(found)} candidates, unreviewed")

    def _evaluate(self) -> None:
        project = self.project
        ids = [r.id for r in project.rois.values() if r.reviewed and r.use]
        if not ids:
            QMessageBox.information(self, "evaluate", "no reviewed, included ROI")
            return
        progress = QProgressDialog("evaluating ROIs...", "stop", 0, len(ids), self)
        progress.setWindowModality(Qt.WindowModal)
        stopped = []

        def report(done, total):
            progress.setValue(done)
            if progress.wasCanceled():
                stopped.append(True)

        run = project.evaluate(progress=report)
        progress.setValue(len(ids))
        failed = sum(1 for r in run["records"].values() if "error" in r)
        self._update_counts()
        self._show_result()
        self.summary.setText(f"{len(run['records']) - failed} evaluated"
                             + (f", {failed} failed" if failed else ""))

    def _show_result(self) -> None:
        roi = self._selected()
        self.results.setRowCount(0)
        if roi is None:
            return
        record, stale = self.project.latest(roi.id)
        if record is None:
            self.results.setRowCount(1)
            self.results.setItem(0, 0, QTableWidgetItem("not evaluated"))
            return
        values = record.get("values") or {"error": record.get("error", "")}
        self.results.setRowCount(len(values))
        for row, (name, value) in enumerate(sorted(values.items())):
            self.results.setItem(row, 0, QTableWidgetItem(name))
            text = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
            self.results.setItem(row, 1, QTableWidgetItem(text + (" (outdated)" if stale else "")))

    def _histograms(self) -> None:
        rows = self.project.results()
        if not rows:
            QMessageBox.information(self, "histograms", "evaluate some ROIs first")
            return
        data = Histograms().analyze(rows, {"bins": 20})
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(data), 1, figsize=(5, 2.2 * len(data)),
                                 constrained_layout=True)
        for ax, (field, d) in zip(np.atleast_1d(axes), data.items()):
            ax.bar(d["edges"][:-1], d["counts"], width=np.diff(d["edges"]), align="edge",
                   color="0.6", edgecolor="0.3")
            ax.set_xlabel(field.replace("_", " "))
            ax.set_ylabel("ROIs")
            if d["missing"]:
                ax.set_title(f"{d['missing']} missing", fontsize=8, color="0.4")
        fig.canvas.manager.set_window_title(f"ROI histograms ({len(rows)} ROIs)")
        fig.show()

    # ------------------------------------------------------------ drawing
    def draw(self) -> None:
        if self.view is not None:
            self.view.show_rois(self.project, self.current_file(), self.active)

    def _recolour(self) -> None:
        for row, roi_id in enumerate(self._rows):
            item = self.table.item(row, 0)
            if item is not None:
                item.setForeground(QColor(roi_colour(self.project.rois[roi_id],
                                                     roi_id == self.active)))

    def show_active(self) -> None:
        """Centre the image on the selected ROI without changing the zoom."""
        roi = self._selected()
        if roi is not None and self.view is not None:
            self.view.center_on(roi.center[0], roi.center[1])
        self.draw()

    def _picked(self, x: float, y: float) -> None:
        """A click in the image: add an ROI, or select the one under it."""
        file_id = self.current_file()
        if file_id is None:
            return
        if self.add_mode.isChecked():
            self.add_at(x, y)
            return
        roi = self.view.roi_at(self.project, file_id, x, y)
        if roi is not None:
            self.select(roi.id)

    def select(self, roi_id: str) -> None:
        """Select an ROI picked in the image."""
        if roi_id in self._rows:
            self.table.selectRow(self._rows.index(roi_id))
