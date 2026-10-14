"""The region steps: the ROI project's block, at the top of the Regions tab.

What the evaluation window was -- the evaluators run over every ROI -- is now
a block (`docs/regions.md`): steps run once in each region, the ROIs or the
regions of an id column, with what each step measured kept as that region's
row and the columns named in *write back* merged into the table.  The panel
is the chain panel (`chain_panel.ChainPanel`) opened on that one block, so
building it is the same as building a chain: add, move, rename, remove, a
block inside it, *check* to time it, *save as...* to keep it as a chain file.

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

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..chain import BLOCK_ROIS, Block, ChainSpec, chain_class
from ..roi_manager.pipeline import default_block
from ..workspace import Instance
from .chain_panel import ChainPanel

PATH = "Analysis/Regions/Per-region steps"


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
                                     "ROIs, or the regions of an id column (for "
                                     "each) -- and what it measures is kept with "
                                     "the region.")
        self.instance = Instance(plugin=PATH, chain=spec.to_dict())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panel = ChainPanel(self.instance, session, chain_class(spec, path=PATH),
                                root=(0,))
        self.panel.edit.setChecked(True)          # the steps are what it is for
        self.panel.changed.connect(self._rewire)
        layout.addWidget(self.panel)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray")
        layout.addWidget(self.status)
        self._rewire()

    # the panel builds a new form and plugin panel after every edit of the
    # steps, so what is listened to is listened to again each time
    def _rewire(self) -> None:
        self.store()
        inner = self.panel.panel
        # the run's log: the line under the panel says how it ended, and an
        # empty box under Run read as a gap in the tab
        inner.output.hide()
        # what the steps are is on the Regions tab's page, behind the
        # section's ?, rather than a paragraph above the form
        if inner.about is not None:
            inner.about.hide()
        try:
            inner.form.changed.connect(self.store, Qt.UniqueConnection)
            inner.ended.connect(self._ended, Qt.UniqueConnection)
        except (RuntimeError, TypeError):
            pass

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
        project = self.session.rois
        if project.block.regions != BLOCK_ROIS:
            self.status.setText(f"for each {project.block.regions}")
            return
        from .roi_window import COUNT_LIMIT
        if len(project.rois) > COUNT_LIMIT:
            # whether each is current means cutting out every ROI's data, and
            # this runs after every ROI the manager measures
            self.status.setText(f"{len(project.rois)} ROIs")
            return
        try:
            steps = project.block_steps()
            waiting = len(project.needs_evaluation(steps=steps)) if steps else 0
        except Exception:
            return
        self.status.setText(f"{waiting} ROI(s) to measure" if waiting
                            else "every ROI is up to date")

    def run(self):
        """Run the block on the session now, on this thread: what the Run
        button does in a worker, for a script and a test."""
        self.store()
        cls = self.panel.cls
        result = self.session.run(cls(), self.panel.form.value())
        self._ended("done", result)
        return result

    def _ended(self, how: str, result) -> None:
        if how != "done" or result is None:
            return
        self.store()
        self.status.setText(result.text.splitlines()[0] if result.text else "done")
        # what is on the ROIs changed: a panel offering their evaluations
        # (*results from*) reads them again
        self.session.changed("rois")
        self.ran.emit(result)

    def showEvent(self, event) -> None:            # noqa: N802 (Qt's name)
        super().showEvent(event)
        self.refresh_counts()
