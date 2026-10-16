"""What of an Abberior `.msr` to open: its measurements, images and renders.

An `.msr` is an Imspector session, not a file of localizations: ten
measurements of one cell, the confocal images each was aimed on, and what
Imspector drew of them.  Opening it means choosing, so File > Open (and a drop)
asks here before anything is read -- the list costs a tenth of a second
(`io.msr.contents`) -- and hands the choice back to the window, which queues
the measurements as files on its loader and opens the images as layers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit,
                               QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from ..io import minflux, msr

ROLE = Qt.UserRole


class MsrImportDialog(QDialog):
    """Tick the measurements and images to open; the details of the one
    selected are shown beside the list."""

    def __init__(self, path, parent=None, found: "msr.Contents" = None):
        super().__init__(parent)
        self.path = Path(path)
        self.found = found or msr.contents(self.path)
        self.setWindowTitle(f"open {self.path.name}")
        layout = QVBoxLayout(self)
        n = len(self.found.measurements)
        layout.addWidget(QLabel(f"{n} MINFLUX measurement{'s' * (n != 1)}, "
                                f"{len(self.found.images) - len(self.found.renders)} images, "
                                f"{len(self.found.renders)} renders"))
        split = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["open", "details"])
        self.tree.setObjectName("msr contents")
        self._groups = {}
        mf = self._group("MINFLUX measurements", True)
        for m in self.found.measurements:
            dim = f"{m.dimensionality}D" if m.dimensionality else "?"
            beads = f", MBM {m.n_beads} beads" if m.n_beads else ""
            self._item(mf, m.label, f"{dim}, {m.rows} records{beads}", m, True)
        images = self._group("Images", True)
        renders = self._group("Imspector's renders", False)
        for im in self.found.images:
            size = " x ".join(str(k) for k in im.shape[::-1])
            self._item(renders if im.render else images, im.name,
                       f"{size} px, {im.pixelsize_nm:.3g} nm", im, not im.render)
        for top in (mf, images):
            top.setExpanded(True)
        for top in list(self._groups.values()):
            if not top.childCount():
                self.tree.invisibleRootItem().removeChild(top)
        self.tree.resizeColumnToContents(0)
        self.tree.currentItemChanged.connect(lambda item, _old: self._show(item))
        self.tree.itemChanged.connect(lambda *_: self._update())
        split.addWidget(self.tree)
        self.details = QPlainTextEdit(readOnly=True)
        self.details.setObjectName("msr details")
        self.details.setLineWrapMode(QPlainTextEdit.NoWrap)
        split.addWidget(self.details)
        split.setSizes([360, 340])
        layout.addWidget(split, 1)

        options = QFormLayout()
        self.merge = QCheckBox("merge into one file")
        self.merge.setToolTip("one table for the ticked measurements, one after "
                              "another in time, with a measurement column; "
                              "otherwise a file each")
        self.keep_invalid = QCheckBox("keep invalid")
        self.keep_invalid.setToolTip("every final iteration, with vld and sta, "
                                     "not only the valid ones")
        row = QHBoxLayout()
        row.addWidget(self.merge)
        row.addWidget(self.keep_invalid)
        row.addStretch(1)
        options.addRow(row)
        self.z_scale = QDoubleSpinBox(minimum=0.01, maximum=10.0, value=1.0,
                                      decimals=3, singleStep=0.01)
        self.z_scale.setToolTip("multiplies z: the refractive index correction; "
                                "1 keeps Imspector's numbers")
        options.addRow("z scale", self.z_scale)
        layout.addLayout(options)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.resize(760, 460)
        first = mf.child(0) if mf.childCount() else None
        if first is not None:
            self.tree.setCurrentItem(first)
        self._update()

    # ------------------------------------------------------------- the tree
    def _group(self, title: str, checked: bool) -> QTreeWidgetItem:
        top = QTreeWidgetItem(self.tree, [title])
        top.setFlags(top.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
        top.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        self._groups[title] = top
        return top

    def _item(self, parent, name: str, details: str, what, checked: bool) -> None:
        item = QTreeWidgetItem(parent, [name, details])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        item.setData(0, ROLE, what)

    def _ticked(self, kind) -> List[Any]:
        out = []
        for top in self._groups.values():
            for i in range(top.childCount()):
                item = top.child(i)
                what = item.data(0, ROLE)
                if isinstance(what, kind) and item.checkState(0) == Qt.Checked:
                    out.append(what)
        return out

    def _update(self) -> None:
        chosen = self._ticked(msr.Measurement)
        self.merge.setEnabled(len(chosen) > 1)
        self.z_scale.setEnabled(any(m.dimensionality == 3 for m in chosen))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            bool(chosen or self._ticked(msr.Image)))

    def _show(self, item) -> None:
        what = item.data(0, ROLE) if item is not None else None
        if isinstance(what, msr.Measurement):
            text = [what.label, f"acquired   {what.date or 'unknown'}",
                    f"layout     {what.layout}",
                    f"dimension  {what.dimensionality or 'unknown'}",
                    f"records    {what.rows}",
                    f"MBM beads  {what.n_beads or 'none used'}"]
            inst = minflux.instrument(what.attrs)
            for laser in inst.get("lasers", []):
                text.append(f"laser      {laser.get('id')} ({laser.get('calibration')})")
            seq = minflux.sequence(what.attrs)
            if seq["iterations"]:
                text += ["", "sequence", minflux.iteration_table(seq)]
            self.details.setPlainText("\n".join(text))
        elif isinstance(what, msr.Image):
            s = what.stack
            text = [what.summary(),
                    f"offset     {s.offset_m[0] * 1e6:.3f}, {s.offset_m[1] * 1e6:.3f} µm",
                    f"data type  {s.dtype}"]
            if what.render:
                text.append("drawn by Imspector from a measurement")
            self.details.setPlainText("\n".join(text))
        else:
            self.details.setPlainText("")

    # ------------------------------------------------------------ the answer
    def choice(self) -> Dict[str, Any]:
        """What to open: the loader's arguments per file, and the images."""
        chosen = self._ticked(msr.Measurement)
        args = {"valid_only": not self.keep_invalid.isChecked(),
                "z_scale": self.z_scale.value() if self.z_scale.isEnabled() else 1.0}
        numbers = [m.number + 1 for m in chosen]
        if self.merge.isChecked() and len(numbers) > 1:
            loads = [{"measurements": numbers, **args}]
        else:
            loads = [{"measurements": [k], **args} for k in numbers]
        return {"loads": loads, "images": self._ticked(msr.Image)}
