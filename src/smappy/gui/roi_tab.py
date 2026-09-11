"""The ROI tab: what an analysis ROI is, and what to compute on it.

Choosing and judging ROIs happens in the ROI manager window
(`roi_window.py`); this tab holds the geometry, the candidate finder and the
evaluation, which apply to every ROI alike.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QProgressDialog,
                               QPushButton, QScrollArea, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..roi_manager.plugins import DensityPeaks, Histograms
from ..session import Session
from .widgets import CONTROL_WIDTH, CollapsibleSection


class ROITab(QWidget):
    def __init__(self, session: Session, view=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.view = view
        self.manager: Optional[QWidget] = None     # not `window`: that is a QWidget method
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

        self.open_button = QPushButton("Open ROI manager")
        self.open_button.setToolTip("the window for choosing and judging ROIs")
        self.open_button.clicked.connect(self.open_manager)
        layout.addWidget(self.open_button)

        # which file this acts on is chosen in the manager; here it is only named
        self.file_count = QLabel("")
        self.file_count.setWordWrap(True)
        self.file_count.setStyleSheet("color: gray")
        layout.addWidget(self.file_count)

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
        self.preview = QDoubleSpinBox(minimum=0, maximum=1e6, decimals=0, suffix=" nm")
        self.preview.setSpecialValueText("3 x ROI size")
        self.preview.setToolTip("what the ROI image shows around the ROI")
        self.preview.setKeyboardTracking(False)
        self.preview.valueChanged.connect(self._on_geometry)
        form.addRow("shape", self.shape)
        form.addRow("size", self.size)
        form.addRow("ROI view", self.preview)
        self.from_region = QPushButton("Add the drawn region as an ROI")
        self.from_region.setToolTip("turn the rectangle, line or polygon drawn in the "
                                    "2D view into an analysis ROI")
        self.from_region.clicked.connect(self._from_region)
        form.addRow(self.from_region)
        layout.addWidget(CollapsibleSection("geometry", geometry, expanded=True))

        # ------------------------------------------------------------ find
        find = QWidget()
        fform = QFormLayout(find)
        fform.setContentsMargins(0, 0, 0, 0)
        fform.setVerticalSpacing(2)
        self.finder: Dict[str, QWidget] = {}
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
        self.find_button = QPushButton("Find candidates")
        self.find_button.setToolTip("density peaks in this file's filtered localizations")
        self.find_button.clicked.connect(self._find)
        fform.addRow(self.find_button)
        layout.addWidget(CollapsibleSection("find", find, expanded=True))

        # -------------------------------------------------------- evaluate
        evaluate = QWidget()
        elayout = QVBoxLayout(evaluate)
        elayout.setContentsMargins(0, 0, 0, 0)
        brow = QHBoxLayout()
        self.evaluate_button = QPushButton("Evaluate all")
        self.evaluate_button.setToolTip("statistics for every included ROI")
        self.evaluate_button.clicked.connect(self._evaluate)
        self.histogram_button = QPushButton("Histograms")
        self.histogram_button.setToolTip("histograms of the current results")
        self.histogram_button.clicked.connect(self._histograms)
        brow.addWidget(self.evaluate_button)
        brow.addWidget(self.histogram_button)
        elayout.addLayout(brow)
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

        session.on_change(self._on_session)
        self.refresh(files_changed=True)

    def sizeHint(self) -> QSize:
        """Never wider than the control window.  The file and summary lines
        wrap, and a QLabel that wraps still asks for its whole text on one
        line, so a long file name would otherwise stretch the window."""
        hint = super().sizeHint()
        return QSize(min(hint.width(), CONTROL_WIDTH), hint.height())

    # ---------------------------------------------------------------- data
    @property
    def project(self):
        return self.session.rois

    def current_file(self) -> Optional[str]:
        """The file chosen in the manager window."""
        project = self.project
        chosen = project.navigation.get("file")
        if chosen in project.sources:
            return chosen
        return next(iter(project.sources), None)

    def _on_session(self, what: str) -> None:
        if what in ("locs", "layers", "rois"):
            self.refresh(files_changed=True)
        elif what in ("layer", "regrouped"):
            self.project.sync()

    # ------------------------------------------------------------- window
    def open_manager(self) -> None:
        from .roi_window import ROIManagerWindow
        if self.manager is None:
            # a top-level window of its own; parented to the control window so
            # it stays with it, never into the tab's own stack
            self.manager = ROIManagerWindow(self.session, self.window())
            self.manager.changed.connect(self.refresh)
        self.manager.show()
        self.manager.raise_()

    def _notify(self) -> None:
        if self.manager is not None and self.manager.isVisible():
            self.manager.refresh()

    # ------------------------------------------------------------ refresh
    def refresh(self, files_changed: bool = False) -> None:
        project = self.project
        self._loading = True
        self.preview.setValue(project.preview_nm)
        if files_changed:
            self.shape.setCurrentText(project.shape)
            self.size.setValue(project.size_nm)
            self.preview.setValue(project.preview_nm)
        self._loading = False
        self._update_counts()
        self._show_result()

    def _update_counts(self) -> None:
        project = self.project
        file_id = self.current_file()
        source = project.sources.get(file_id)
        rois = project.rois_of(file_id)
        used = sum(1 for r in rois if r.use)
        name = f"{source.number + 1}. {source.name}" if source else "no file"
        self.file_count.setText(f"{name} - {len(rois)} ROIs, {used} used")
        rows = project.results()
        self.summary.setText(f"{len(rows)} current results over all files"
                             if rows else "no current results")

    # -------------------------------------------------------------- edits
    def _on_geometry(self) -> None:
        if self._loading:
            return
        try:
            self.project.set_geometry(self.size.value(), self.shape.currentText())
        except ValueError as e:
            QMessageBox.warning(self, "geometry", str(e))
            return
        self.project.preview_nm = self.preview.value()
        self._notify()

    def _from_region(self) -> None:
        """The region drawn in the 2D view becomes an analysis ROI."""
        region = self.session.roi
        file_id = self.current_file()
        if region is None:
            QMessageBox.information(self, "no region", "draw a region in the 2D view first")
            return
        if file_id is None:
            return
        points = np.asarray(region.points, float)
        if region.kind == "rect":
            x0, y0, x1, y1 = region.bounds
            polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        else:
            polygon = points.tolist()
        center = np.asarray(polygon, float).mean(axis=0)
        self.project.add_roi(file_id, [float(center[0]), float(center[1])], polygon=polygon)
        self._update_counts()
        self._notify()
        self.summary.setText(f"added the drawn {region.kind} as an ROI")

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
        self._update_counts()
        self._notify()
        self.summary.setText(f"{len(found)} candidates found")

    def _evaluate(self) -> None:
        project = self.project
        ids = [r.id for r in project.rois.values() if r.use]
        if not ids:
            QMessageBox.information(self, "evaluate", "no included ROI")
            return
        progress = QProgressDialog("evaluating ROIs...", "stop", 0, len(ids), self)
        progress.setWindowModality(Qt.WindowModal)

        def report(done, total):
            progress.setValue(done)

        run = project.evaluate(progress=report)
        progress.setValue(len(ids))
        failed = sum(1 for r in run["records"].values() if "error" in r)
        self._update_counts()
        self._show_result()
        self.summary.setText(f"{len(run['records']) - failed} evaluated"
                             + (f", {failed} failed" if failed else ""))

    def _show_result(self) -> None:
        project = self.project
        active = project.navigation.get("roi")
        roi = project.rois.get(active) if active else None
        self.results.setRowCount(0)
        if roi is None:
            return
        record, stale = project.latest(roi.id)
        if record is None:
            self.results.setRowCount(1)
            self.results.setItem(0, 0, QTableWidgetItem("not evaluated"))
            return
        values = record.get("values") or {"error": record.get("error", "")}
        self.results.setRowCount(len(values))
        for row, (name, value) in enumerate(sorted(values.items())):
            self.results.setItem(row, 0, QTableWidgetItem(name))
            text = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
            self.results.setItem(row, 1,
                                 QTableWidgetItem(text + (" (outdated)" if stale else "")))

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
