"""The Regions tab's header: what an analysis ROI *is*, and the way in.

The tab itself (*Regions*) is an ordinary `PluginTab` over whatever the user
pinned from `Analysis/Regions/*` -- the finders and the summaries, ordinary
run-once plugins.
What cannot be a pinned plugin sits above them, in this header: the way into
the ROI manager window where sites are chosen and judged, the steps run once
per ROI (`region_steps.RegionSteps`, a block, which is a different object
from a tab's list of pins), what they measured in the ROI selected there,
and the geometry every ROI shares.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QHeaderView, QLabel, QMessageBox,
                               QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..session import Session
from .widgets import CONTROL_WIDTH, CollapsibleSection
from ..roi_manager.core import annotation_columns
from .roi_window import COUNT_LIMIT


class ROIHeader(QWidget):
    def __init__(self, session: Session, view=None, parent=None):
        super().__init__(parent)
        from .help_window import attach_help
        attach_help(self, 'Panels/Regions tab', within=True)
        self.session = session
        self.view = view
        self.manager: Optional[QWidget] = None     # not `window`: that is a QWidget method
        self._loading = False

        # In the tab's own scroll, above the plugins (`PluginTab`): it had a
        # scroll of its own, which took half the tab whatever it held -- a
        # gap under two folded sections, with the plugins pushed below it.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
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
        self.shape.addItems(["circle", "square", "track"])
        self.shape.setToolTip("what a click in the ROI manager drafts: a circle or "
                              "square there, or the track drawn nearest it.  Until "
                              "one is chosen, a track whenever the layer draws tracks")
        self.size = QDoubleSpinBox(minimum=1, maximum=1e6, decimals=0, suffix=" nm")
        self.size.setToolTip("circle diameter, or the side of a square")
        self.size.setKeyboardTracking(False)
        self.shape.currentTextChanged.connect(self._on_shape)
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
        # first and open: what a click in the manager makes is the first thing
        # a project needs, and it is not a step that runs, so not among them
        layout.addWidget(CollapsibleSection("geometry", geometry, expanded=True, major=True))


        # ------------------------------------------------ per-region steps
        # The steps, how they run on the ROI selected in the manager, and
        # what they measured there, together: they were a window, two ticks
        # in the manager and a folded table here, and which of the three
        # held the answer was not obvious.
        steps = QWidget()
        slayout = QVBoxLayout(steps)
        slayout.setContentsMargins(0, 0, 0, 0)
        from .region_steps import RegionSteps
        self.steps = RegionSteps(session)
        self.steps.ran.connect(self._evaluated)
        slayout.addWidget(self.steps)
        self.measure = QCheckBox("measure the selected ROI")
        self.measure.setToolTip(
            "when an ROI is selected or added in the ROI manager, run the steps "
            "whose result for it is missing or out of date, and keep what they "
            "measure.  Off: a large project can be scrolled without computing "
            "behind you, and the numbers are shown as stored, marked for what "
            "they are.")
        self.measure.toggled.connect(self._on_measure)
        self.plot = QCheckBox("draw its figures")
        self.plot.setToolTip(
            "open the ROI results window: what the steps draw for the selected "
            "ROI, a tab per step, redrawn in place by the next ROI -- so "
            "that stepping down the list compares like with like.  Only the "
            "tab being looked at is drawn.")
        self.plot.toggled.connect(self._on_plot)
        slayout.addWidget(self.measure)
        slayout.addWidget(self.plot)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: gray")
        slayout.addWidget(self.summary)
        self.results = QTableWidget(0, 2)
        self.results.setHorizontalHeaderLabels(["measure", "value"])
        self.results.verticalHeader().setVisible(False)
        self.results.verticalHeader().setDefaultSectionSize(20)
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # the table only once an ROI is selected, and as tall as its rows
        self.results.hide()
        slayout.addWidget(self.results)
        # folded until wanted: the steps' editor is tall, and the tab opens on
        # the geometry and the plugins
        section = CollapsibleSection("per-region steps", steps, expanded=False,
                                     major=True, helpable=True,
                                     help_tip="what the per-region steps are, and "
                                              "how they run")
        section.help_requested.connect(self._steps_help)
        layout.addWidget(section)

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
        elif what == "roi-preview" and self.project.segment_preview is not None:
            # a preview is drawn in the manager and nowhere else
            self.open_manager()
        elif what in ("layer", "regrouped"):
            self.project.sync()
            self._show_shape()         # tracks drawn: a click picks a track

    # ------------------------------------------------------------- window
    def open_manager(self) -> None:
        from .roi_window import ROIManagerWindow
        if self.manager is None:
            # a top-level window of its own; parented to the control window so
            # it stays with it, never into the tab's own stack
            self.manager = ROIManagerWindow(self.session, self.window())
            self.manager.changed.connect(self.refresh)
            # the numbers of whichever ROI is selected, measured now or not:
            # they followed only a measurement, and stood on the last ROI
            self.manager.selected.connect(self._on_selected)
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
            self.size.setValue(project.size_nm)
            self.preview.setValue(project.preview_nm)
        self._show_shape()
        self.measure.setChecked(bool(project.navigation.get("re_evaluate", True)))
        self.plot.setChecked(bool(project.navigation.get("plot_evaluation", False)))
        self._loading = False
        self._update_counts()
        self._show_result()
        self.steps.refresh_counts()

    def _update_counts(self) -> None:
        project = self.project
        file_id = self.current_file()
        source = project.sources.get(file_id)
        rois = project.rois_of(file_id)
        used = sum(1 for r in rois if r.use)
        name = f"{source.number + 1}. {source.name}" if source else "no file"
        self.file_count.setText(f"{name} - {len(rois)} ROIs, {used} used")
        if len(project.rois) > COUNT_LIMIT:
            # whether each result is current means cutting out every ROI's
            # data, a second per 20000 ROIs: count what is stored instead
            stored = sum(1 for roi_id in project.rois
                         if any(str(roi_id) in t for t in project.tables.values()))
            self.summary.setText(f"{stored} ROIs with results over all files"
                                 if stored else "no results")
            return
        rows = project.results()
        self.summary.setText(f"{len(rows)} current results over all files"
                             if rows else "no current results")

    def _show_shape(self) -> None:
        """The shape a click drafts: the project's, or *track*."""
        project = self.project
        loading, self._loading = self._loading, True
        self.shape.setCurrentText("track" if project.drafts_tracks else project.shape)
        self.size.setEnabled(not project.drafts_tracks)
        self._loading = loading

    # -------------------------------------------------------------- edits
    def _on_shape(self, text: str) -> None:
        """A shape chosen: what a click drafts from now on, whatever the
        layer draws.  *track* leaves the shape of the ROIs that have one."""
        if self._loading:
            return
        self.project.navigation["draft"] = "track" if text == "track" else "shape"
        self.size.setEnabled(text != "track")
        if text != "track":
            self._on_geometry()

    def _on_geometry(self) -> None:
        if self._loading:
            return
        try:
            shape = self.shape.currentText()
            self.project.set_geometry(self.size.value(),
                                      None if shape == "track" else shape)
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
        if not self.session.axes().is_default:
            # an ROI is a place in nanometres; this region is in the picture's units
            QMessageBox.information(self, "custom axes",
                                    "the 2D view shows custom axes; switch them off to "
                                    "draw a region that can become an ROI")
            return
        polygon = region.vertices().tolist()
        center = np.asarray(polygon, float).mean(axis=0)
        try:
            with self.project.edit(f"Add the drawn {region.kind}"):
                self.project.add_roi(file_id, [float(center[0]), float(center[1])],
                                     polygon=polygon)
        except ValueError as error:          # a line of zero width, a degenerate outline
            QMessageBox.information(self, "not an ROI", str(error))
            return
        self._update_counts()
        self._notify()
        self.summary.setText(f"added the drawn {region.kind} as an ROI")

    def _steps_help(self) -> None:
        from .help_window import show_help
        show_help("Panels/Regions tab", self.window())

    def _on_selected(self, roi) -> None:
        """An ROI was selected in the manager: its numbers here, and with
        *show in the 2D view* the picture centred on it, on its file."""
        self._show_result()
        if self.view is None or roi is None or \
                not self.project.navigation.get("follow_2d", False):
            return
        number = self.project.file_number(roi.file_id)
        changed = False
        for layer in self.session.layers:
            if not layer.is_image and layer.files is not None and number not in layer.files:
                layer.set_files([number])       # a layer showing another file
                changed = True
        if changed:
            self.session.changed("layer")
        self.view.center_on(*roi.center)

    # --------------------------------------------------------------- work
    def _on_measure(self, on: bool) -> None:
        if self._loading:
            return
        if self.manager is not None:
            self.manager.set_measuring(on)
        else:
            self.project.navigation["re_evaluate"] = bool(on)

    def _on_plot(self, on: bool) -> None:
        """The figures are drawn for the ROI selected in the manager, so
        asking for them opens it."""
        if self._loading:
            return
        if on:
            self.open_manager()
        if self.manager is not None:
            self.manager.set_plotting(on)
        else:
            self.project.navigation["plot_evaluation"] = bool(on)

    def _evaluated(self, result) -> None:
        self._update_counts()
        self._show_result()
        self.summary.setText(result.text.splitlines()[0].split(": ", 1)[-1]
                             if result.text else "measured")
        self._notify()

    def _show_result(self) -> None:
        project = self.project
        active = project.navigation.get("roi")
        roi = project.rois.get(active) if active else None
        self.results.setRowCount(0)
        self.results.setVisible(roi is not None)
        if roi is None:
            return
        self.results.setHorizontalHeaderLabels(
            [f"ROI {project.numbers().get(roi.id, '')}", "value"])
        record, states = project.latest(roi.id)
        if record is None:
            noted = sorted(roi.annotations.items())
            self.results.setRowCount(1 + len(noted))
            self.results.setItem(0, 0, QTableWidgetItem(
                "not measured" if self.measure.isChecked()
                else "not measured: tick measure, or Run"))
            for row, (name, keyword) in enumerate(noted, 1):
                self.results.setItem(row, 0, QTableWidgetItem(name))
                self.results.setItem(row, 1, QTableWidgetItem(keyword))
            self._fit_results()
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
        values.update(annotation_columns(roi, values))     # the keywords given it
        self.results.setRowCount(len(values))
        for row, (name, text) in enumerate(sorted(values.items())):
            self.results.setItem(row, 0, QTableWidgetItem(name))
            self.results.setItem(row, 1, QTableWidgetItem(text))
        self._fit_results()

    def _fit_results(self) -> None:
        """As tall as its rows, up to ten: the tab scrolls, the table need not."""
        rows = min(max(self.results.rowCount(), 1), 10)
        self.results.setFixedHeight(self.results.horizontalHeader().height()
                                    + rows * self.results.verticalHeader().defaultSectionSize()
                                    + 2 * self.results.frameWidth())
