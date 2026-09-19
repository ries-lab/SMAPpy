"""The evaluation pipeline in a window of its own.

A tab's pins are run one at a time on demand.  An evaluation is an ordered
pipeline run over every site at once, each step contributing columns to that
site's row -- so it is a different object, and thirty evaluators in the ROI tab
would bury the two or three plugins used to find and judge sites.

Laid out like SMAP's `SEEvaluationGui`: the pipeline on the left as a
checkbox-and-name list, the selected step's parameters on the right.  The list
is `list[Instance]`, the same type a tab holds, so ordering, renaming and
duplicate handling are one implementation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout,
                               QInputDialog, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QProgressDialog, QPushButton,
                               QSplitter, QStackedWidget, QToolButton, QVBoxLayout,
                               QWidget)

from .. import plugins
from ..roi_manager import pipeline as pipeline_module
from ..workspace import Instance
from .chooser import choose_plugin
from .params import SettingsForm


class EvaluationWindow(QWidget):
    """Choose evaluators, order them, set them up, run them over every ROI."""

    ran = Signal(object)              # the run record

    def __init__(self, session, instances: Optional[List[Instance]] = None, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("ROI evaluation")
        self.session = session
        self.instances: List[Instance] = list(
            instances if instances is not None else pipeline_module.default_instances())
        self.forms: Dict[str, SettingsForm] = {}

        layout = QVBoxLayout(self)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._show_form)
        self.list.itemChanged.connect(self._toggled)
        left_layout.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        for text, tip, slot in (
                ("+", "add an evaluator", self.add),
                ("−", "remove the selected step", self.remove),
                ("▲", "move up", lambda: self.move(-1)),
                ("▼", "move down", lambda: self.move(1))):
            button = QToolButton(text=text, autoRaise=True)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        left_layout.addLayout(buttons)
        rename = QPushButton("Rename...")
        rename.clicked.connect(self.rename)
        duplicate = QPushButton("Duplicate")
        duplicate.clicked.connect(self.duplicate)
        row = QHBoxLayout()
        row.addWidget(rename)
        row.addWidget(duplicate)
        left_layout.addLayout(row)
        splitter.addWidget(left)

        self.forms_stack = QStackedWidget()
        self.empty = QLabel("select a step to set it up")
        self.empty.setStyleSheet("color: gray")
        self.empty.setAlignment(Qt.AlignCenter)
        self.forms_stack.addWidget(self.empty)
        splitter.addWidget(self.forms_stack)
        splitter.setSizes([180, 320])

        bottom = QHBoxLayout()
        self.run_button = QPushButton("Run on every ROI")
        self.run_button.clicked.connect(self.run)
        # the button that gets pressed after a parameter is edited: one
        # evaluator over the sites, not the whole pipeline over all of them
        self.changed_button = QPushButton("Re-evaluate what changed")
        self.changed_button.setToolTip(
            "run only the steps whose result is out of date -- a parameter "
            "edited here, or an ROI that has moved -- and keep the rest")
        self.changed_button.clicked.connect(lambda: self.run(reuse=True))
        bottom.addWidget(self.changed_button)
        save = QPushButton("Save pipeline...")
        save.clicked.connect(self.save_as)
        load = QPushButton("Load pipeline...")
        load.clicked.connect(self.load_from)
        bottom.addWidget(self.run_button, 1)
        bottom.addWidget(save)
        bottom.addWidget(load)
        layout.addLayout(bottom)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray")
        layout.addWidget(self.status)

        self.resize(560, 420)
        self.rebuild()

    # ------------------------------------------------------------ the list
    def rebuild(self, select: int = 0) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        known = plugins.refs()
        for instance in self.instances:
            ref = known.get(instance.plugin)
            item = QListWidgetItem(instance.title(ref.name if ref else ""))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if instance.enabled else Qt.Unchecked)
            if ref is None:
                item.setToolTip(f"{instance.plugin} is not installed")
                item.setForeground(Qt.gray)
            else:
                item.setToolTip(ref.description or instance.plugin)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self.list.count():
            self.list.setCurrentRow(max(0, min(select, self.list.count() - 1)))
        else:
            self.forms_stack.setCurrentWidget(self.empty)
        self.run_button.setEnabled(any(i.enabled for i in self.instances))

    def current(self) -> Optional[Instance]:
        row = self.list.currentRow()
        return self.instances[row] if 0 <= row < len(self.instances) else None

    def _toggled(self, item: QListWidgetItem) -> None:
        row = self.list.row(item)
        if 0 <= row < len(self.instances):
            self.instances[row].enabled = item.checkState() == Qt.Checked
        self.run_button.setEnabled(any(i.enabled for i in self.instances))
        self.refresh_counts()

    def _show_form(self, row: int) -> None:
        instance = self.instances[row] if 0 <= row < len(self.instances) else None
        if instance is None:
            self.forms_stack.setCurrentWidget(self.empty)
            return
        form = self.forms.get(instance.id)
        if form is None:
            try:
                cls = plugins.get(instance.plugin)
            except Exception as error:
                self.empty.setText(f"{instance.plugin} could not be loaded: {error}")
                self.forms_stack.setCurrentWidget(self.empty)
                return
            form = SettingsForm(cls.Settings, cls.specs())
            if instance.values:
                form.restore(instance.values)
            self.forms[instance.id] = form
            self.forms_stack.addWidget(form)
        self.forms_stack.setCurrentWidget(form)

    # ----------------------------------------------------------- editing
    def add(self) -> None:
        path = choose_plugin(self, scope=pipeline_module.SCOPE,
                             title="Add an evaluator")
        if not path:
            return
        self.instances.append(Instance(plugin=path))
        self.rebuild(len(self.instances) - 1)

    def remove(self) -> None:
        row = self.list.currentRow()
        if not 0 <= row < len(self.instances):
            return
        self.forms.pop(self.instances.pop(row).id, None)
        self.rebuild(row)

    def move(self, by: int) -> None:
        row = self.list.currentRow()
        if not 0 <= row + by < len(self.instances) or row < 0:
            return
        order = self.instances
        order[row], order[row + by] = order[row + by], order[row]
        self.rebuild(row + by)

    def rename(self) -> None:
        instance = self.current()
        if instance is None:
            return
        name, ok = QInputDialog.getText(self, "Rename step", "Name:",
                                        text=instance.title())
        if ok and name.strip():
            instance.label = name.strip()
            self.rebuild(self.list.currentRow())

    def duplicate(self) -> None:
        instance = self.current()
        if instance is None:
            return
        self.save_values()
        row = self.list.currentRow()
        copy = instance.copy()
        copy.label = f"{instance.title()} copy"
        self.instances.insert(row + 1, copy)
        self.rebuild(row + 1)

    # ------------------------------------------------------------ running
    def save_values(self) -> List[Instance]:
        """Read the open forms back into their instances."""
        for instance in self.instances:
            form = self.forms.get(instance.id)
            if form is not None:
                instance.values = form.values()
        return self.instances

    def run(self, reuse: bool = False) -> None:
        """Run the pipeline over the ROIs; with `reuse`, only what is stale.

        A re-evaluation still writes a record for every ROI -- the steps that
        were still current are copied into it -- so what the site table reads
        afterwards is one complete set of numbers and not a patchwork of
        runs.
        """
        project = self.session.rois
        self.save_values()
        ids = [r.id for r in project.rois.values() if r.reviewed and r.use]
        if not ids:
            QMessageBox.information(self, "evaluate", "no reviewed, included ROI")
            return
        steps = pipeline_module.resolve(self.instances)
        if not steps:
            QMessageBox.information(self, "evaluate", "no evaluator is enabled")
            return
        waiting = len(project.needs_evaluation(steps=steps, roi_ids=ids))
        if reuse and not waiting:
            self.status.setText("every ROI is up to date")
            return
        progress = QProgressDialog("evaluating ROIs...", "stop", 0, len(ids), self)
        progress.setWindowModality(Qt.WindowModal)
        run = project.evaluate(steps=steps, reuse=reuse,
                               progress=lambda done, total: progress.setValue(done))
        progress.setValue(len(ids))
        failed = sum(1 for record in run["records"].values()
                     if pipeline_module.errors(record))
        self.status.setText(
            f"{len(run['records'])} ROIs, {len(steps)} step(s)"
            + (f", {waiting} of them out of date" if reuse else "")
            + (f"; {failed} with a failing step" if failed else ""))
        self.refresh_counts()
        self.ran.emit(run)

    def refresh_counts(self) -> None:
        """Say how many ROIs the pipeline as it stands has yet to measure.

        Cosmetic, and deliberately not refreshed on every keystroke: the
        count costs a signature per ROI per step, and the button works
        whatever the label says -- pressing it with nothing out of date says
        so and runs nothing.
        """
        steps = pipeline_module.resolve(self.save_values())
        waiting = len(self.session.rois.needs_evaluation(steps=steps)) if steps else 0
        self.changed_button.setText("Re-evaluate what changed"
                                    + (f" ({waiting})" if waiting else ""))

    def showEvent(self, event) -> None:            # noqa: N802 (Qt's name)
        super().showEvent(event)
        self.refresh_counts()

    # -------------------------------------------------------------- files
    def save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save pipeline", "",
                                              "Pipeline (*.yaml)")
        if path:
            pipeline_module.save(self.save_values(), path)
            self.status.setText(f"saved to {path}")

    def load_from(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load pipeline", "",
                                              "Pipeline (*.yaml)")
        if not path:
            return
        loaded = pipeline_module.load(path)
        if not loaded:
            QMessageBox.warning(self, "Load pipeline", "no steps in that file")
            return
        self.instances = loaded
        self.forms.clear()
        while self.forms_stack.count() > 1:
            self.forms_stack.removeWidget(self.forms_stack.widget(1))
        self.rebuild()
        self.status.setText(f"loaded {len(loaded)} step(s) from {path}")
