"""Small dialogs: mapping csv columns, asking for a pixel size."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QLabel, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from ..io.formats import CSV_NAMES, csv_columns, guess_csv_mapping

TARGETS = ["-", "x_nm", "y_nm", "z_nm", "frame", "photons", "xy_err_nm",
           "z_err_nm", "sigma_nm", "sigma_y_nm", "background", "logl_rel",
           "channel", "id"]


class CsvMappingDialog(QDialog):
    """Which csv column is which, what the lengths are in, and the number of
    the first frame -- and a saved conversion to do all of it by name
    (`io.conversions`, SMAP's *conversion structure*)."""

    UNITS = (("nm", "nm"), ("um", "µm"), ("mm", "mm"), ("px", "camera pixels"))

    def __init__(self, path, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QHBoxLayout, QInputDialog, QPushButton, QSpinBox
        from ..io.conversions import conversion_for, conversions
        self._input = QInputDialog
        self.setWindowTitle(f"columns of {Path(path).name}")
        headers, sample, has_header = csv_columns(Path(path))
        self.headers = headers
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("conversion"))
        self.conversion = QComboBox()
        self.conversion.addItem("-", "")
        for name in conversions():
            self.conversion.addItem(name, name)
        top.addWidget(self.conversion, 1)
        save = QPushButton("save as...")
        save.setToolTip("keep these columns, units and first frame under a name, "
                        "found again by the headers")
        save.clicked.connect(self._save)
        top.addWidget(save)
        layout.addLayout(top)
        layout.addWidget(QLabel("x and y are needed; the rest is optional.  "
                                "Other columns are kept under their own name."))
        self.table = QTableWidget(len(headers), 3)
        self.table.setHorizontalHeaderLabels(["column", "first value", "means"])
        self.combos = []
        for i, (h, v) in enumerate(zip(headers, sample)):
            self.table.setItem(i, 0, QTableWidgetItem(h))
            self.table.setItem(i, 1, QTableWidgetItem(v))
            combo = QComboBox()
            combo.addItems(TARGETS + [f"keep as '{h}'"])
            self.table.setCellWidget(i, 2, combo)
            self.combos.append((h, combo))
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        form = QFormLayout()
        self.units = QComboBox()
        for value, label in self.UNITS:
            self.units.addItem(label, value)
        self.pixelsize = QDoubleSpinBox(minimum=0.1, maximum=10000, value=100.0, decimals=2)
        self.units.currentIndexChanged.connect(
            lambda _: self.pixelsize.setEnabled(self.units.currentData() == "px"))
        self.first_frame = QSpinBox(minimum=0, maximum=1, value=0)
        self.first_frame.setToolTip("1 for ThunderSTORM and SMAP, 0 for smappy")
        form.addRow("lengths in", self.units)
        form.addRow("camera pixel (nm)", self.pixelsize)
        form.addRow("first frame", self.first_frame)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 420)
        self.conversion.currentIndexChanged.connect(self._apply_conversion)
        found = conversion_for(headers)
        if found is not None:
            self.conversion.setCurrentIndex(self.conversion.findData(found.name))
        self._apply_conversion()

    def _apply_conversion(self, *_) -> None:
        """Fill the table and the units from the conversion chosen, or guess."""
        from ..io.conversions import conversions
        conv = conversions().get(self.conversion.currentData() or "")
        guess = guess_csv_mapping(self.headers)
        mapping = dict(guess, **(conv.mapping if conv else {}))
        for h, combo in self.combos:
            target = mapping.get(h, "-")
            combo.setCurrentText(target if target in TARGETS else f"keep as '{h}'")
        units = conv.units if conv else "nm"
        self.units.setCurrentIndex(max(0, self.units.findData(units)))
        if conv and conv.pixelsize_nm:
            self.pixelsize.setValue(conv.pixelsize_nm)
        self.pixelsize.setEnabled(units == "px")
        self.first_frame.setValue(conv.first_frame if conv else 0)

    def _mapping(self) -> Dict[str, str]:
        mapping = {}
        for h, combo in self.combos:
            t = combo.currentText()
            if t.startswith("keep as"):
                mapping[h] = ""                # read under its own name
            elif t != "-":
                mapping[h] = t
        return mapping

    def _save(self) -> None:
        from ..io.conversions import Conversion, save_conversion
        name, ok = self._input.getText(self, "Save conversion", "name")
        name = name.strip()
        if not ok or not name:
            return
        mapping = {h: t for h, t in self._mapping().items() if t}
        save_conversion(Conversion(
            name=name, headers=list(self.headers), mapping=mapping,
            units=self.units.currentData(),
            pixelsize_nm=self.pixelsize.value() if self.units.currentData() == "px" else None,
            first_frame=self.first_frame.value()))
        if self.conversion.findData(name) < 0:
            self.conversion.addItem(name, name)
        blocked = self.conversion.blockSignals(True)
        self.conversion.setCurrentIndex(self.conversion.findData(name))
        self.conversion.blockSignals(blocked)

    def reader_args(self) -> Dict:
        units = self.units.currentData()
        return {"mapping": {h: t for h, t in self._mapping().items() if t},
                "units": units, "first_frame": self.first_frame.value(),
                "pixelsize_nm": self.pixelsize.value() if units == "px" else None}


class GroupingDialog(QDialog):
    """How localizations are linked into blinks: session-wide, belonging to no
    layer.  OK regroups every layer."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from ..group import GroupSettings
        from ..plugins import ParamInfo, param_specs
        from .params import SettingsForm
        self.session = session
        self.setWindowTitle("Merge blinks")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Merge blinks</b>: the localizations of one blinking "
                                "event, in consecutive frames, become one"))
        note = QLabel("Merging looks at x and y only: emitters this close in consecutive "
                      "frames overlap in the raw data and were fitted as one anyway, so "
                      "a z window would only split blinks whose z scatters.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        layout.addWidget(note)
        self.form = SettingsForm(GroupSettings, param_specs(GroupSettings))
        self.form.set(session.group_settings)
        self.form.field_changed.connect(self._show_mode)
        self._show_mode("link")
        layout.addWidget(self.form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # the fields each mode reads; the other mode's are kept, only hidden
    MODE_FIELDS = {"fixed": ("dx",), "precision": ("k", "dx_min", "dx_max")}

    def _show_mode(self, name: str) -> None:
        if name != "link":
            return
        mode = self.form.fields["link"].value()
        for m, names in self.MODE_FIELDS.items():
            for n in names:
                self.form.fields[n].setVisible(m == mode)
                self.form.labels[n].setVisible(m == mode)

    def _apply(self) -> None:
        try:
            settings = self.form.value()
        except ValueError:
            return
        if settings != self.session.group_settings:
            self.session.set_group_settings(settings)
        self.accept()


class PixelSizeDialog(QDialog):
    """An image without a pixel size in its tags: ask, plus where it sits."""

    def __init__(self, name: str, pixelsize: Optional[float] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"place {name}")
        form = QFormLayout(self)
        self.pixelsize = QDoubleSpinBox(minimum=0.01, maximum=1e6, decimals=2,
                                        value=pixelsize or 100.0)
        self.x0 = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=1)
        self.y0 = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=1)
        form.addRow("pixel size (nm)", self.pixelsize)
        form.addRow("x of the first pixel (nm)", self.x0)
        form.addRow("y of the first pixel (nm)", self.y0)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Tuple[float, float, float]:
        return self.pixelsize.value(), self.x0.value(), self.y0.value()


class CopyLayerDialog(QDialog):
    """Which layer to copy from, and which of its settings to take.

    Everything is the wrong default: what makes two layers two -- the filter,
    or which file each one shows -- is usually exactly what must not be
    carried over, and "make this one look like that one" is the thing people
    actually want.  So the display is ticked and the rest is not.
    """

    PARTS = (("display", "display and rendering",
              "colour, LUT, contrast, mode and the rendering width"),
             ("bounds", "filter bounds",
              "replaces this layer's bounds with the other layer's"),
             ("files", "which files are shown", ""),
             ("grouping", "merge blinks or not", ""))

    def __init__(self, session, target: int, parent=None):
        super().__init__(parent)
        self.session = session
        self.target = target
        self.setWindowTitle(f"copy settings to {session.layers[target].name}")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Take these settings from:"))
        self.source = QComboBox()
        for i, layer in enumerate(session.layers):
            if i != target and not layer.is_image:
                self.source.addItem(f"{i + 1}: {layer.name}", i)
        layout.addWidget(self.source)
        self.parts = {}
        for name, label, help_text in self.PARTS:
            box = QCheckBox(label, checked=(name == "display"))
            if help_text:
                box.setToolTip(help_text)
            layout.addWidget(box)
            self.parts[name] = box
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setEnabled(self.source.count() > 0)

    def _apply(self) -> None:
        source = self.source.currentData()
        if source is not None:
            self.session.copy_layer(source, self.target,
                                    **{n: b.isChecked() for n, b in self.parts.items()})
        self.accept()
