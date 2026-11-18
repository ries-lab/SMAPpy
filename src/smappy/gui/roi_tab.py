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

from typing import Optional

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ..session import Session
from .widgets import CONTROL_WIDTH, CollapsibleSection


def proposal_text(key: str, value) -> str:
    """A proposal in a few words: ``angle 37\u00b0``, ``line axis``,
    ``frames 3-``, ``type: motor``, ``centre``."""
    from ..roi_manager.core import range_description
    kind, _, name = key.partition(":")
    if key == "angle":
        return f"angle {float(value):.0f}\u00b0"
    if kind == "range":
        return range_description(name, value)
    if kind == "line":
        return f"line {name}"
    if kind == "keyword":
        return f"{name}: {value}"
    return key
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
        # what the steps propose the selected ROI be, and a button to take
        # it: here, beside the steps that measured it, not in the ROI manager
        self.proposals = QWidget()
        self.proposals_layout = QVBoxLayout(self.proposals)
        self.proposals_layout.setContentsMargins(0, 2, 0, 0)
        self.proposals_layout.setSpacing(2)
        slayout.addWidget(self.proposals)
        # folded until wanted: the steps' editor is tall, and the tab opens on
        # the geometry and the plugins
        section = CollapsibleSection("per-region steps", steps, expanded=False,
                                     major=True, helpable=True,
                                     help_tip="what the per-region steps are, and "
                                              "how they run")
        section.help_requested.connect(self._steps_help)
        layout.addWidget(section)

        # ------------------------------------------------- region filter
        # what was measured per region, kept or dropped by its values; the
        # flag every layer honours (`region_filter`)
        from .region_filter_panel import RegionFilterPanel
        self.region_filter = RegionFilterPanel(session)
        layout.addWidget(CollapsibleSection("region filter", self.region_filter,
                                            expanded=False, major=True))

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
            if self.manager is not None:   # tracks drawn: a click picks a track
                self.manager.show_geometry()

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
            # what *Apply* takes follows the list's selection
            self.manager.selection_changed.connect(self._show_result)
        self.manager.show()
        self.manager.raise_()

    def _notify(self) -> None:
        if self.manager is not None and self.manager.isVisible():
            self.manager.refresh()

    # ------------------------------------------------------------ refresh
    def refresh(self, files_changed: bool = False) -> None:
        project = self.project
        self._loading = True
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

    def _steps_help(self) -> None:
        from .help_window import show_help
        show_help("Panels/Regions tab", self.window())

    def _on_selected(self, roi) -> None:
        """An ROI was selected in the manager: its numbers here, and as the
        manager's *2D view* says, the picture centred on it, on its file --
        and with *shows the ROI* the ROI made the 2D view's own, so the 3D
        view's box, which follows that, shows it too."""
        from .roi_window import link_mode
        self._show_result()
        mode = link_mode(self.project.navigation)
        if self.view is None or roi is None or mode == "off":
            return
        self._show_file(roi.file_id)
        if mode == "roi":
            line = self.manager.line_name if self.manager is not None else ""
            region = self.project.view_region(roi, line)
            if region is not None:
                self.session.set_roi(region)
        self.view.center_on(*roi.center)

    def _show_file(self, file_id) -> None:
        """Every localization layer filtered to the ROI's file, as the Render
        tab's file list would: a layer that showed every file showed the
        other files' localizations over the ROI, so it is narrowed too."""
        source = self.project.sources.get(file_id)
        if source is None or len(self.project.sources) < 2:
            return
        # the table's `filenumber`, from 0 -- not `file_number`, which is the
        # list's 1-based count, and once put the layer on the next file
        number = int(source.number)
        changed = False
        for layer in self.session.layers:
            if not layer.is_image and layer.files != [number]:
                layer.set_files([number])
                changed = True
        if changed:
            self.session.changed("layer")

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
        self._show_proposals(roi)
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
        values.update(annotation_columns(roi, values, project.uses_angles))     # the keywords given it
        self.results.setRowCount(len(values))
        for row, (name, text) in enumerate(sorted(values.items())):
            self.results.setItem(row, 0, QTableWidgetItem(name))
            self.results.setItem(row, 1, QTableWidgetItem(text))
        self._fit_results()

    def _show_proposals(self, roi) -> None:
        """A line per step that proposes something for the ROI shown, and
        *Apply*: to the ROIs selected in the ROI manager, all when none is,
        each ROI taking its own proposals (`roi_results.apply_proposals`)."""
        while self.proposals_layout.count():
            item = self.proposals_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # out of the panel now, not at the next turn of the event loop
                widget.setParent(None)
                widget.deleteLater()
        if roi is None:
            return
        project = self.project
        chosen = project.chosen()
        where = (("this ROI" if chosen == [roi.id] else f"{len(chosen)} selected")
                 if len(chosen) < len(project.rois) else f"all {len(chosen)}")
        for label, table in project.tables.items():
            entry = table.entry(roi.id) if roi.id in table else None
            proposed = (entry or {}).get("proposals")
            if not proposed:
                continue
            text = QLabel(f"{label} proposes: " + ", ".join(
                proposal_text(k, v) for k, v in proposed.items()))
            text.setWordWrap(True)
            text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            button = QPushButton(f"Apply to {where}")
            button.setToolTip("take these proposals as the ROIs' own -- each ROI its "
                              "own -- leaving what was set by hand; one undo step")
            button.clicked.connect(lambda _=False, step=label: self._apply_proposals(step))
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(text, 1)
            row.addWidget(button)
            line = QWidget()
            line.setLayout(row)
            self.proposals_layout.addWidget(line)

    def _apply_proposals(self, step: str) -> None:
        from ..plugins.roi_results import apply_proposals
        changed, written, counts = apply_proposals(self.project, step,
                                                   self.project.chosen())
        passed = ", ".join(f"{n} {why}" for why, n in counts.items() if n)
        self.summary.setText(f"{step}'s proposals: {written} written, {changed} ROIs "
                             "changed" + (f"; passed over: {passed}" if passed else ""))
        self._notify()
        self._show_result()

    def _fit_results(self) -> None:
        """As tall as its rows, up to ten: the tab scrolls, the table need not."""
        rows = min(max(self.results.rowCount(), 1), 10)
        self.results.setFixedHeight(self.results.horizontalHeader().height()
                                    + rows * self.results.verticalHeader().defaultSectionSize()
                                    + 2 * self.results.frameWidth())
