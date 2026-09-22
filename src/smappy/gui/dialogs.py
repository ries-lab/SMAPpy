"""Small dialogs: mapping csv columns, asking for a pixel size."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from ..io.formats import CSV_NAMES, csv_columns, guess_csv_mapping

TARGETS = ["-", "x_nm", "y_nm", "z_nm", "frame", "photons", "loc_precision_nm",
           "loc_precision_z_nm", "sigma_nm", "sigma_y_nm", "background", "logl_rel",
           "channel", "id"]


class CsvMappingDialog(QDialog):
    """Which csv column is which; positions in nm or in pixels."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"columns of {Path(path).name}")
        headers, sample, has_header = csv_columns(Path(path))
        guess = guess_csv_mapping(headers)
        layout = QVBoxLayout(self)
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
            combo.setCurrentText(guess.get(h, "-"))
            self.table.setCellWidget(i, 2, combo)
            self.combos.append((h, combo))
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        form = QFormLayout()
        self.units = QComboBox()
        self.units.addItems(["nm", "px"])
        self.pixelsize = QDoubleSpinBox(minimum=0.1, maximum=10000, value=100.0, decimals=2)
        self.pixelsize.setEnabled(False)
        self.units.currentTextChanged.connect(lambda u: self.pixelsize.setEnabled(u == "px"))
        form.addRow("positions in", self.units)
        form.addRow("pixel size (nm)", self.pixelsize)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 380)

    def reader_args(self) -> Dict:
        mapping = {}
        for h, combo in self.combos:
            t = combo.currentText()
            if t.startswith("keep as"):
                mapping[h] = h
            elif t != "-":
                mapping[h] = t
        units = self.units.currentText()
        return {"mapping": mapping, "units": units,
                "pixelsize_nm": self.pixelsize.value() if units == "px" else None}


class ParametersDialog(QDialog):
    """Session-wide parameters that belong to no layer: how localizations are
    grouped into blinks.  OK regroups every layer."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from ..group import GroupSettings
        from ..plugins import ParamInfo, param_specs
        from .params import SettingsForm
        self.session = session
        self.setWindowTitle("parameters")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>grouping</b>: localizations of one blink are merged"))
        note = QLabel("Linking is lateral: emitters within the link box in consecutive "
                      "frames overlap in the raw data and were fitted as one anyway, so "
                      "a z window only split blinks whose z scatters.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        layout.addWidget(note)
        infos = {"dx": ParamInfo(label="link within", unit="nm", min=0,
                                 help="half-width of the box a localization may move per frame"),
                 "dt": ParamInfo(label="gap", unit="frames", min=0,
                                 help="frames a blink may be dark and still continue"),
                 "block_fields": ParamInfo(hidden=True),
                 "link_chunks": ParamInfo(hidden=True)}
        self.form = SettingsForm(GroupSettings, param_specs(GroupSettings, infos))
        self.form.set(session.group_settings)
        layout.addWidget(self.form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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
