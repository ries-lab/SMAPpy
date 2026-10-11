"""The region steps: the ROI project's block, in a window of its own.

What the evaluation window was -- the evaluators run over every ROI -- is now
a block (`docs/regions.md`): steps run once in each region, the ROIs or the
regions of an id column, with what each step measured kept as that region's
row and the columns named in *write back* merged into the table.  The window
is the chain panel (`chain_panel.ChainPanel`) opened on that one block, so
building it is the same as building a chain: add, move, rename, remove, a
block inside it, *check* to time it, *save as...* to keep it as a chain file.

The block lives on the ROI project, not on the window (`ROIProject.pipeline`
for its steps and `ROIProject.block_options` for its own fields), so closing
the window loses nothing, the project saves it into the localization file
beside the results it produced, and the ROI manager re-evaluates an ROI on
selection with it.  Running it skips what is still current: an edited step
runs over the ROIs, the others are carried forward.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..chain import Block, ChainSpec, chain_class
from ..roi_manager import pipeline as pipeline_module
from ..workspace import Instance
from .chain_panel import ChainPanel

PATH = "Analysis/Regions/Region steps"


class RegionStepsWindow(QWidget):
    """Edit the project's block and run it over its regions."""

    ran = Signal(object)              # the run's `Result`

    def __init__(self, session, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Region steps")
        self.session = session
        project = session.rois
        if not project.pipeline:
            project.pipeline = pipeline_module.default_instances()
        block = pipeline_module.block_of(project.pipeline, project.block_options)
        spec = ChainSpec(steps=[block], path=PATH,
                         description="the steps run once in each region")
        self.instance = Instance(plugin=PATH, chain=spec.to_dict())
        layout = QVBoxLayout(self)
        about = QLabel("Each step runs once in every region -- the ROIs, or the "
                       "regions of an id column (<i>for each</i>, in the block's "
                       "section) -- and what it measures is kept with the region.")
        about.setWordWrap(True)
        about.setStyleSheet("color: gray")
        layout.addWidget(about)
        self.panel = ChainPanel(self.instance, session, chain_class(spec, path=PATH),
                                root=(0,))
        self.panel.edit.setChecked(True)
        self.panel.changed.connect(self._rewire)
        layout.addWidget(self.panel, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray")
        layout.addWidget(self.status)
        self.resize(460, 640)
        self._rewire()

    # the panel builds a new form and plugin panel after every edit of the
    # steps, so what is listened to is listened to again each time
    def _rewire(self) -> None:
        self.store()
        inner = self.panel.panel
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
        project = self.session.rois
        project.pipeline, project.block_options = pipeline_module.from_block(block)
        self.refresh_counts()

    def refresh_counts(self) -> None:
        project = self.session.rois
        if project.block_options.get("regions") != "rois":
            self.status.setText(f"for each {project.block_options.get('regions')}")
            return
        try:
            steps = pipeline_module.resolve(project.pipeline)
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
