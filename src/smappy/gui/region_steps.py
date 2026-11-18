"""The region steps: the ROI project's block, at the top of the Regions tab.

What the evaluation window was -- the evaluators run over every ROI -- is now
a block (`docs/regions.md`): steps run once in each region, the ROIs or the
regions of an id column, with what each step measured kept as that region's
row and the columns named in *write back* merged into the table.  The panel
is the chain panel (`chain_panel.ChainPanel`) opened on that one block, so
building it is the same as building a chain: under *build*, add a step,
time it with *check*; on a step's title, rename, move or remove it; in the
⋯ menu, open a saved list or keep this one as a chain file.

What differs from a chain's panel, and why: every step here is in the one
block, so what it repeats over is a menu at the top (*for each*) and the
block's own section is not shown -- its steps stand as the list; *write
back* and *write values* are part of building it and sit under *build*; and
what applies to all the steps -- Run, *Measure this ROI*, the ticks that
measure and draw the ROI selected in the ROI manager -- is together at the
top (`roi_tab.ROIHeader` adds its own there), with each step's title saying
how its result stands for that ROI (`show_states`).

It was a window of its own, opened from a button, and so a second editor
beside the chain panel every pinned chain already has, for the one block
the Regions tab is about.  It sits in the tab now, above the plugins, with
the selected ROI's results under it (`roi_tab.ROIHeader`).

The block lives on the ROI project, not on the panel (`ROIProject.block`),
so the project saves it into the localization file beside the results it
produced, and the ROI manager measures an ROI on selection with it.  Running
it skips what is still current: an edited step runs over the ROIs, the
others are carried forward.
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton,
                               QVBoxLayout, QWidget)

from ..chain import BLOCK_ROIS, Block, ChainSpec, chain_class
from ..roi_manager.pipeline import default_block
from ..workspace import Instance
from .chain_panel import ChainPanel
from .widgets import note_style

PATH = "Analysis/Regions/Per-region steps"
# the block's own fields, which this panel shows in its own places
BLOCK_FIELDS = ("regions", "write_back", "write_values", "merge")
RUN_TIP = ("run the steps in every ROI (or region) whose result is missing or "
           "out of date; the current ones are kept")


class RegionSteps(QWidget):
    """Edit the project's block and run it over its regions."""

    ran = Signal(object)              # the run's `Result`

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        project = session.rois
        if not project.block.steps:
            project.block.steps = default_block().steps
        spec = ChainSpec(steps=[project.block], path=PATH,
                         description="Each step runs once in every region -- the "
                                     "ROIs, or the regions of an id attribute (for "
                                     "each) -- and what it measures is kept with "
                                     "the region.")
        self.instance = Instance(plugin=PATH, chain=spec.to_dict())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        # what applies to every step: Run (the inner panel's own button,
        # moved here) and what the tab adds beside it; then the tab's ticks
        self.controls = QHBoxLayout()
        self.controls.setContentsMargins(0, 0, 0, 0)
        self.run_button = QPushButton("Measure all")
        self.run_button.setToolTip(RUN_TIP)
        self.run_button.clicked.connect(self._press_run)
        self.controls.addWidget(self.run_button)
        self.controls.addStretch(1)
        layout.addLayout(self.controls)
        self.shared = QVBoxLayout()
        self.shared.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.shared)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(note_style())
        layout.addWidget(self.status)
        each = QHBoxLayout()
        each.setContentsMargins(0, 0, 0, 0)
        each.addWidget(QLabel("for each"))
        self.for_each = QComboBox()
        self.for_each.setToolTip("what the steps are repeated over: every ROI of the "
                                 "ROI manager, or each region of an id attribute")
        self.for_each.activated.connect(self._for_each_chosen)
        each.addWidget(self.for_each, 1)
        layout.addLayout(each)
        self.panel = ChainPanel(self.instance, session, chain_class(spec, path=PATH),
                                root=(0,))
        # The block's write-back is not offered here: what a step measures in
        # a region is kept with the region, a value becomes an ROI's property
        # through *Write results* (to the ROIs) and an attribute of its localizations
        # through *Write results* (to the localizations), and a list of attribute names to
        # type was one more thing to know.  A project saved with one set still
        # writes it, so the rows are shown for it, to be seen and cleared.
        self.writes_box = QWidget()
        writes = QFormLayout(self.writes_box)
        writes.setContentsMargins(0, 2, 0, 0)
        self.writes_note = QLabel("this project's steps write back into the table; "
                                  "empty both to stop")
        self.writes_note.setWordWrap(True)
        self.writes_note.setStyleSheet(note_style())
        writes.addRow(self.writes_note)
        self.write_back = QLineEdit()
        self.write_back.setToolTip("attributes the steps make that go into the real "
                                   "table, comma separated; empty: nothing changes it")
        self.write_values = QLineEdit()
        self.write_values.setToolTip("numbers the steps measure, comma separated, "
                                     "each region's written onto its localizations")
        for name, edit in (("write_back", self.write_back),
                           ("write_values", self.write_values)):
            edit.editingFinished.connect(lambda n=name, e=edit: self._block_field(n, e.text()))
        self.overlaps = QComboBox()
        self.overlaps.setToolTip("a row in two regions takes the last region's value, "
                                 "the first's, their mean, or any (a flag written back)")
        from ..region_runs import WriteBack
        for rule in WriteBack.RULES:
            self.overlaps.addItem(rule, rule)
        self.overlaps.activated.connect(
            lambda i: self._block_field("merge", self.overlaps.itemData(i)))
        writes.addRow("write back", self.write_back)
        writes.addRow("write values", self.write_values)
        writes.addRow("overlaps", self.overlaps)
        self.panel.editor.layout().addWidget(self.writes_box)
        # a block inside the block is for a chain; here the steps are the
        # block's, and *for each* says what they repeat over
        self.panel.block_button.hide()
        self.panel.changed.connect(self._rewire)
        layout.addWidget(self.panel)
        self._project = project
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(0)
        self._rebuild_timer.timeout.connect(self._rebuild_safely)
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(200)
        self._run_timer.timeout.connect(self._sync_run)
        self._states: Optional[Dict[str, str]] = None
        self._rewire()

    def _press_run(self) -> None:
        """Run (or Stop) through the panel's own button, then follow it."""
        self.panel.panel.run_button.click()
        self._sync_run()
        if self.panel.panel.run_button.text() != self.panel.panel.run_text:
            self._run_timer.start()             # a run is going: follow Stop

    def _sync_run(self) -> None:
        """The top button says and allows what the panel's Run does."""
        try:                                    # a timer's callable never raises
            inner = self.panel.panel.run_button
            self.run_button.setText(inner.text())
            self.run_button.setToolTip(inner.toolTip() or RUN_TIP)
            self.run_button.setEnabled(inner.isEnabled())
            if inner.text() == self.panel.panel.run_text:
                self._run_timer.stop()
        except Exception as error:              # noqa: BLE001
            print(f"following Run: {error}")
            self._run_timer.stop()

    def follow_project(self) -> None:
        """The session's ROI project, if it is a new one -- another file
        opened.  The panel kept the last project's steps, and the new
        project's empty list meant an ROI selected in the ROI manager was not
        measured until something stored the panel's steps into it.

        A project with no steps of its own takes the panel's, as they stand:
        the steps one set up go on to the next file, and nothing is rebuilt.
        One that saved its own steps is shown with them -- a rebuild of the
        panel, put off until the event loop is free (`_rebuild_later`)."""
        project = self.session.rois
        if project is self._project:
            return
        self._project = project
        if not project.block.steps:
            self.store()
            return
        self._rebuild_later()

    def _rebuild_later(self) -> None:
        # Rebuilding inside the session's notice of a change -- where a run's
        # own notice may still be on the stack -- crashed a test worker now
        # and then.  Tied to this widget: dropped with it.
        # a timer of this widget's own, not `QTimer.singleShot` with a
        # context: it goes with the widget, and a pending call with it
        self._rebuild_timer.start()

    def _rebuild_safely(self) -> None:
        try:                                    # a singleShot callable never raises
            if self.run_button.text() != self.panel.panel.run_text:
                self._rebuild_later()           # a run is going: after it
                return
            project = self.session.rois
            spec = ChainSpec(steps=[project.block], path=PATH,
                             description=self.panel.cls.spec.description)
            self.panel._current = ()
            self.panel._apply(spec)
        except Exception as error:              # noqa: BLE001
            print(f"showing the ROI project's steps: {error}")

    def block_key(self) -> str:
        return self.panel.cls.spec.keys()[0]

    # the panel builds a new form and plugin panel after every edit of the
    # steps, so what is listened to is listened to again each time
    def _rewire(self) -> None:
        self.store()
        inner = self.panel.panel
        # the run's log: the line under the controls says how it ended, and an
        # empty box under Run read as a gap in the tab
        inner.output.hide()
        # and its status line, empty but for a failure (shown by `_ended`):
        # empty, it was a gap under the steps
        inner.status.setVisible(bool(inner.status.text()))
        if inner.result_button is not None:
            inner.result_button.hide()
        # what the steps are is on the Regions tab's page, behind the
        # section's ?, rather than a paragraph above the form
        if inner.about is not None:
            inner.about.hide()
        # Run, at the top with what else applies to every step
        # The panel's Run stays in the panel, hidden; the button at the top
        # presses it and shows what it says (Stop while a run goes).  Moving
        # the panel's own button up here, across the panels an edit rebuilds,
        # crashed a test worker now and then.
        inner.run_text = "Measure all"
        inner.run_tip = RUN_TIP
        if inner.run_button.text() == "Run":
            inner.run_button.setText(inner.run_text)
            inner.run_button.setToolTip(RUN_TIP)
        inner.run_button.hide()
        self._sync_run()
        # the block's section is not shown: its steps stand as the list, and
        # its own fields are in their places here
        key = self.block_key()
        section = inner.form.sections.get(key)
        if section is not None:
            section.button.hide()
            section.set_expanded(True)
            section.frame.setFrameShape(QFrame.NoFrame)
            section._inner.setContentsMargins(0, 0, 0, 0)
            form = inner.form.fields[key]
            form.set_shown({name: False for name in BLOCK_FIELDS})
        self._fill_block_fields()
        self.show_states(self._states)
        try:
            inner.form.changed.connect(self.store, Qt.UniqueConnection)
            inner.ended.connect(self._ended, Qt.UniqueConnection)
            # the run's progress: over hundreds of ROIs of a slow step the tab
            # looked stuck without it
            inner.progressed.connect(self._progressed, Qt.UniqueConnection)
        except (RuntimeError, TypeError):
            pass

    def _fill_block_fields(self) -> None:
        block = self.session.rois.block
        self.for_each.blockSignals(True)
        self.for_each.clear()
        self.for_each.addItem("ROI", BLOCK_ROIS)
        names = self.panel._id_columns()
        if block.regions != BLOCK_ROIS and block.regions not in names:
            names = [block.regions] + names          # kept, though the table lacks it
        for name in names:
            self.for_each.addItem(name, name)
        self.for_each.setCurrentIndex(max(self.for_each.findData(block.regions), 0))
        self.for_each.blockSignals(False)
        self.overlaps.setCurrentIndex(max(self.overlaps.findData(block.merge), 0))
        for edit, values in ((self.write_back, block.write_back),
                             (self.write_values, block.write_values)):
            if not edit.hasFocus():
                edit.setText(", ".join(values))
        self.writes_box.setVisible(bool(block.write_back or block.write_values)
                                   or self.write_back.hasFocus()
                                   or self.write_values.hasFocus())

    def _block_field(self, name: str, value: str) -> None:
        self.panel.panel.form.set_values({f"{self.block_key()}.{name}": value})
        self.store()

    def _for_each_chosen(self, index: int) -> None:
        self._block_field("regions", self.for_each.itemData(index))

    def show_states(self, states: Optional[Dict[str, str]]) -> None:
        """Each step's title with how its result stands for the ROI selected
        in the ROI manager -- *(stale)*, *(failed)*, *(missing)* -- by its
        label; ``None``: no ROI, the plain titles."""
        self._states = states
        block_cls = next((cls for k, _, cls in self.panel.cls.steps
                          if k == self.block_key()), None)
        labels = (dict(zip([k for k, _, _ in block_cls.steps], block_cls.labels()))
                  if block_cls is not None else {})
        for path, key, step, section in self.panel.step_sections():
            if len(path) != 2 or isinstance(step, Block):
                continue
            state = (states or {}).get(labels.get(key, ""))
            mark = "" if not step.enabled or state in (None, "current") else f"   ({state})"
            section.set_title(step.title() + mark)

    def block(self) -> Optional[Block]:
        spec = self.panel.current_spec()
        first = spec.steps[0] if spec.steps else None
        return first if isinstance(first, Block) else None

    def store(self) -> None:
        """The block as it stands into the project, and how many ROIs it has
        yet to measure."""
        block = self.block()
        if block is None:
            return
        self.session.rois.block = block
        self.refresh_counts()

    def refresh_counts(self) -> None:
        text = self._counts()
        if text is not None:
            self.status.setText(text)

    def _counts(self) -> Optional[str]:
        """What is left to measure, or None when that cannot be said."""
        project = self.session.rois
        if project.block.regions != BLOCK_ROIS:
            return f"for each {project.block.regions}"
        from .roi_window import COUNT_LIMIT
        if len(project.rois) > COUNT_LIMIT:
            # whether each is current means cutting out every ROI's data, and
            # this runs after every ROI the manager measures
            return f"{len(project.rois)} ROIs"
        try:
            steps = project.block_steps()
            waiting = len(project.needs_evaluation(steps=steps)) if steps else 0
        except Exception:
            return None
        return f"{waiting} ROI(s) to measure" if waiting else "every ROI is up to date"

    def _progressed(self, text: str) -> None:
        # the last line only: "37% (37 of 100 ROIs), 2.1 a second, ..."
        self.status.setText(text.splitlines()[-1] if text.strip() else text)

    def run(self):
        """Run the block on the session now, on this thread: what the Run
        button does in a worker, for a script and a test."""
        self.store()
        cls = self.panel.cls
        result = self.session.run(cls(), self.panel.form.value(),
                                  progress=self._progressed)
        self._ended("done", result)
        return result

    def _ended(self, how: str, result) -> None:
        self._sync_run()
        if how != "done" or result is None:
            # not the last percentage, which would read as a run still going:
            # how it ended, and what is left to measure after it
            counts = self._counts()
            self.status.setText(f"{how}; {counts}" if counts else how)
            inner = self.panel.panel
            inner.status.setVisible(bool(inner.status.text()))   # the error, if any
            return
        self.store()
        self.status.setText(result.text.splitlines()[0] if result.text else "done")
        # what is on the ROIs changed: a panel offering their evaluations
        # (*results from*) reads them again
        self.session.changed("rois")
        self.ran.emit(result)

    def showEvent(self, event) -> None:            # noqa: N802 (Qt's name)
        super().showEvent(event)
        self.follow_project()
        self._fill_block_fields()
        self.refresh_counts()
