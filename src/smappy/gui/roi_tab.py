"""The ROI tab's header: what an analysis ROI *is*, and the way in.

The tab itself is an ordinary `PluginTab` over whatever the user pinned from
`ROIManager/*` -- the segmenters and the analyses, ordinary run-once plugins.
What cannot be a pinned plugin sits above them, in this header: the geometry
every ROI shares, the way into the ROI manager window where sites are chosen
and judged, and the way into the evaluation window, whose ordered pipeline is a
different object from a tab's list of pins.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout,
                               QHeaderView, QLabel, QMessageBox,
                               QPushButton, QScrollArea, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..roi_manager import pipeline as pipeline_module
from ..session import Session
from .widgets import CONTROL_WIDTH, CollapsibleSection


class ROIHeader(QWidget):
    def __init__(self, session: Session, view=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.view = view
        self.manager: Optional[QWidget] = None     # not `window`: that is a QWidget method
        self.evaluation: Optional[QWidget] = None
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

        # -------------------------------------------------------- evaluate
        evaluate = QWidget()
        elayout = QVBoxLayout(evaluate)
        elayout.setContentsMargins(0, 0, 0, 0)
        self.evaluate_button = QPushButton("Evaluation pipeline...")
        self.evaluate_button.setToolTip("choose the evaluators to run on every ROI, "
                                        "order them and set them up; opens its own window")
        self.evaluate_button.clicked.connect(self.open_evaluation)
        elayout.addWidget(self.evaluate_button)
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
            # one pipeline window per session, wherever it is opened from
            self.manager.evaluation_requested.connect(self.open_evaluation)
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
    def open_evaluation(self) -> None:
        """The pipeline window, one per session, reusing what it was left with.

        The pipeline itself lives on the project, not on the window: closing
        the window must not lose the steps, and the project is what gets saved
        into the localization file beside the results they produce.
        """
        from .evaluation import EvaluationWindow
        project = self.project
        if not project.pipeline:
            project.pipeline = pipeline_module.default_instances()
        if self.evaluation is None:
            self.evaluation = EvaluationWindow(self.session, project.pipeline, self)
            self.evaluation.ran.connect(self._evaluated)
        self.evaluation.show()
        self.evaluation.raise_()

    def _evaluated(self, run) -> None:
        failed = sum(1 for record in run["records"].values()
                     if pipeline_module.errors(record))
        self.project.pipeline = self.evaluation.save_values()
        self._update_counts()
        self._show_result()
        self.summary.setText(f"{len(run['records'])} evaluated"
                             + (f", {failed} with a failing step" if failed else ""))
        self._notify()

    def _show_result(self) -> None:
        project = self.project
        active = project.navigation.get("roi")
        roi = project.rois.get(active) if active else None
        self.results.setRowCount(0)
        if roi is None:
            return
        record, states = project.latest(roi.id)
        if record is None:
            self.results.setRowCount(1)
            self.results.setItem(0, 0, QTableWidgetItem("not evaluated"))
            return
        # a value is as good as the step that produced it, so the mark goes
        # per step and not on the whole row: one edited evaluator does not
        # make the others' numbers wrong
        marks = {label: "" if state == "current" else f" ({state})"
                 for label, state in states.items()}
        values = {}
        for label, entry in (record.get("steps") or {}).items():
            for name, value in ((entry.get("values") or {})).items():
                text = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
                values[name if name not in values else f"{label}.{name}"] = \
                    text + marks.get(label, "")
            if "error" in entry:
                values[f"{label} failed"] = entry["error"]
        self.results.setRowCount(len(values))
        for row, (name, text) in enumerate(sorted(values.items())):
            self.results.setItem(row, 0, QTableWidgetItem(name))
            self.results.setItem(row, 1, QTableWidgetItem(text))
