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

from ..chain import BLOCK_ROIS as ROIS
from ..session import Session
from .widgets import CONTROL_WIDTH, CollapsibleSection


#: what a proposal is, for a person who meets one
PROPOSAL_TIP = ("What this step measured in the ROI, offered as the ROI's own "
                "properties: its angle, a range of an attribute, a line, a keyword, "
                "its centre.  Nothing changes until it is applied; a property set "
                "by hand stays as it is.")


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
    if key == "centre":
        return "its centre"
    return key
from ..roi_manager.core import annotation_columns
from .roi_window import COUNT_LIMIT

# the tab's way in, tinted so that it stands out from the rest a little
OPEN_STYLE = """
QPushButton { font-weight: bold; border-radius: 5px; padding: 5px 8px;
              border: 1px solid rgba(58, 143, 208, 0.75);
              background: rgba(58, 143, 208, 0.28); }
QPushButton:hover { background: rgba(58, 143, 208, 0.40); }
QPushButton:pressed { background: rgba(58, 143, 208, 0.55); }
"""
#: *Results table...*: as large and as bold as *Open ROI manager*, with its
#: border but no tint, so that the way to the ROIs still leads
RESULTS_STYLE = """
QPushButton { font-weight: bold; border-radius: 5px; padding: 5px 8px;
              border: 1px solid rgba(58, 143, 208, 0.75); background: transparent; }
QPushButton:hover { background: rgba(58, 143, 208, 0.12); }
QPushButton:pressed { background: rgba(58, 143, 208, 0.25); }
"""


class ROIHeader(QWidget):
    def __init__(self, session: Session, view=None, parent=None,
                 colour_by=None):
        """``colour_by(field)`` colours the Render tab's layer by a region
        value: what the results window's header offers."""
        super().__init__(parent)
        from .help_window import attach_help
        attach_help(self, 'Panels/Regions tab', within=True)
        self.session = session
        self.view = view
        self.manager: Optional[QWidget] = None     # not `window`: that is a QWidget method
        self.results_window: Optional[QWidget] = None
        self.colour_by = colour_by
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
        # The tab's way in, so it should catch the eye -- but only a little:
        # bold on a light blue tint, which reads on light and dark windows alike.
        self.open_button.setStyleSheet(OPEN_STYLE)
        # every region's numbers in one table, beside the way to the ROIs
        self.results_button = QPushButton("Results table...")
        self.results_button.setToolTip(
            "what was measured per region -- the ROIs, or the clusters or tracks "
            "of an id column -- as one table: sort it, colour the localizations "
            "by a column, copy it or save it as a CSV (Export table)")
        self.results_button.clicked.connect(lambda: self.open_results())
        self.results_button.setStyleSheet(RESULTS_STYLE)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.open_button, 1)
        buttons.addWidget(self.results_button, 1)
        layout.addLayout(buttons)

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
        # the steps' settings changed: what they give on the ROI being looked
        # at, without stepping off it and back or running every ROI
        self.measure_now = QPushButton("Measure this ROI")
        self.measure_now.setToolTip(
            "run the steps on the ROI selected in the ROI manager now: those "
            "whose result for it is missing or out of date, as after a change "
            "of their settings")
        self.measure_now.clicked.connect(self.measure_selected)
        # with Run, at the top of the steps: what applies to all of them
        self.steps.controls.insertWidget(1, self.measure_now)
        # what the steps proposed for the ROIs, set on them in one click: on
        # the ROIs selected in the ROI manager, all when none is
        self.apply_all = QPushButton("Apply proposals")
        self.apply_all.setToolTip(PROPOSAL_TIP + "  On the ROIs selected in the ROI "
                                  "manager, all when none is, each its own; one "
                                  "undo step.  Several steps proposing: choose which.")
        self.apply_all.clicked.connect(lambda: self.apply_proposals())
        self.steps.controls.insertWidget(2, self.apply_all)
        ticks = QHBoxLayout()
        ticks.setContentsMargins(0, 0, 0, 0)
        ticks.addWidget(self.measure)
        ticks.addWidget(self.plot)
        ticks.addStretch(1)
        self.steps.shared.addLayout(ticks)
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
        # the table only once an ROI is selected, and as tall as its rows;
        # in a section of its own, to fold away while stepping (2026-11)
        self.results_section = CollapsibleSection("measured values", self.results,
                                                  expanded=True)
        self.results_section.hide()
        slayout.addWidget(self.results_section)
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
        # the step list's file and ⋯ menu (open, save as) in the title bar
        section.add_to_title(self.steps.panel.title_widgets())
        layout.addWidget(section)

        # ------------------------------------------------- region filter
        # what was measured per region, kept or dropped by its values; the
        # flag every layer honours (`region_filter`)
        from .region_filter_panel import RegionFilterPanel
        self.region_filter = RegionFilterPanel(session)
        self.region_filter_section = CollapsibleSection(
            "region filter", self.region_filter, expanded=False, major=True)
        layout.addWidget(self.region_filter_section)

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
            # another file's project: its own steps in the panel, before
            # anything is measured with them
            self.steps.follow_project()
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
            # its list's header offers the table of every ROI's numbers
            self.manager.open_results = lambda: self.open_results(ROIS)
        self.manager.show()
        self.manager.raise_()

    def open_results(self, source: Optional[str] = None) -> QWidget:
        """The region results window, on ``source``'s regions (ROIS or an id
        column) where given."""
        from .region_results_window import RegionResultsWindow
        if self.results_window is None:
            self.results_window = RegionResultsWindow(
                self.session, self.window(), colour_by=self.colour_by,
                filter_on=self.filter_on, show_roi=self._show_roi,
                center_on=None if self.view is None else self.view.center_on)
        if source is not None:
            self.results_window.show_source(source)
        self.results_window.show()
        self.results_window.raise_()
        return self.results_window

    def filter_on(self, source: str, label: str, value: str) -> None:
        """The region filter unfolded on one value, ready for its bounds."""
        self.region_filter_section.set_expanded(True)
        self.region_filter._show_value((source, label, value))

    def _show_roi(self, roi_id: str) -> None:
        self.open_manager()
        self.manager.select(roi_id)

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
    def measure_selected(self) -> None:
        """The steps on the selected ROI, now: what is out of date runs."""
        roi_id = self.project.navigation.get("roi")
        if roi_id not in self.project.rois:
            return
        self.steps.store()
        self.open_manager()
        if self.manager.active != roi_id:
            self.manager.select(roi_id)
        self.manager.evaluate_active(now=True)
        self._show_result()

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
        self.measure_now.setEnabled(roi is not None)
        self.apply_all.setEnabled(bool(self.proposing_steps()))
        self.steps.show_selection()
        # each step's title says how its result stands for this ROI
        self.steps.show_states(project.latest(roi.id)[1] if roi is not None else None)
        self.results.setRowCount(0)
        self.results_section.setVisible(roi is not None)
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
                else "not measured: Measure this ROI, or Run"))
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
                # -- hidden, never `setParent(None)`: that makes it a window
                # of its own for a moment, and on macOS each one deleted on
                # every ROI stepped through handed the keyboard to the main
                # window (`focus.log`: an untitled window, then none, then
                # the control window, once per line, 2026-11)
                widget.hide()
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
            # every widget in the panel from the start, never a window of its
            # own for a moment (see above)
            line = QWidget(self.proposals)
            text = QLabel(f"{label} proposes for the ROI: " + ", ".join(
                proposal_text(k, v) for k, v in proposed.items()), line)
            text.setToolTip(PROPOSAL_TIP)
            text.setWordWrap(True)
            text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            button = QPushButton(f"Apply to {where}", line)
            button.setToolTip(PROPOSAL_TIP + "  On the ROIs selected in the ROI "
                              "manager, all when none is, each taking its own; "
                              "one undo step.")
            button.clicked.connect(lambda _=False, step=label: self._apply_proposals(step))
            row = QHBoxLayout(line)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(text, 1)
            row.addWidget(button)
            self.proposals_layout.addWidget(line)

    def proposing_steps(self) -> list:
        from ..plugins.roi_results import proposal_steps
        try:
            return list(proposal_steps(self.project))
        except Exception:                        # noqa: BLE001 -- a button's state
            return []

    def apply_proposals(self, step=None) -> None:
        """*Apply proposals*: one step's, or with several a menu to choose
        one or all."""
        steps = self.proposing_steps()
        if not steps:
            self.summary.setText("no step has proposed anything for the ROIs")
            return
        if step is None and len(steps) > 1:
            from PySide6.QtWidgets import QMenu
            menu = QMenu(self)
            for label in steps:
                menu.addAction(label, lambda _=False, s=label: self.apply_proposals(s))
            menu.addSeparator()
            menu.addAction("all of them", lambda: self.apply_proposals("*"))
            menu.exec(self.apply_all.mapToGlobal(self.apply_all.rect().bottomLeft()))
            return
        for label in (steps if step in (None, "*") else [step]):
            self._apply_proposals(label)

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
