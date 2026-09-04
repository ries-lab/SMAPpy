"""The Render tab: what is drawn (filter) and how (colour, contrast, gamma)."""
from __future__ import annotations

import dataclasses
from typing import Optional

from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QLabel, QLineEdit, QVBoxLayout, QWidget)

from .. import lut as luts
from ..session import Session
from ..viewer import COLOR_FIELDS, FILTER_FIELDS, FIELD_LUT, INTENSITY_LUT
from .widgets import CollapsibleSection


class _Bound(QLineEdit):
    def __init__(self, value: Optional[float]):
        super().__init__("" if value is None else f"{value:g}")
        self.setMaximumWidth(70)
        self.setPlaceholderText("-")

    def value(self) -> Optional[float]:
        text = self.text().strip()
        return None if text in ("", "-") else float(text)


class RenderTab(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.filter_box = QWidget()
        self.filter_grid = QGridLayout(self.filter_box)
        self.filter_grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(CollapsibleSection("filter", self.filter_box, expanded=True))

        display = QWidget()
        form = QFormLayout(display)
        form.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        self.mode.addItems(["precision", "gauss", "hist"])
        self.color = QComboBox()
        self.lut = QComboBox()
        self.lut.addItems(luts.names())
        self.contrast = QDoubleSpinBox(minimum=0, maximum=6, singleStep=0.1, decimals=2)
        self.gamma = QDoubleSpinBox(minimum=0.1, maximum=3, singleStep=0.1, decimals=2)
        form.addRow("render", self.mode)
        form.addRow("colour by", self.color)
        form.addRow("LUT", self.lut)
        form.addRow("contrast", self.contrast)
        form.addRow("gamma", self.gamma)
        layout.addWidget(CollapsibleSection("display", display, expanded=True))
        layout.addStretch(1)

        self.mode.currentTextChanged.connect(self._on_render_settings)
        self.color.currentIndexChanged.connect(self._on_color)
        self.lut.currentTextChanged.connect(self._on_display)
        self.contrast.valueChanged.connect(self._on_display)
        self.gamma.valueChanged.connect(self._on_display)
        session.on_change(self._on_session)
        self.rebuild()

    @property
    def layer(self):
        return self.session.layers[0]

    def _on_session(self, what: str) -> None:
        if what == "locs":
            self.rebuild()

    def rebuild(self) -> None:
        """Rows for the fields this table has, showing the current bounds."""
        grid = self.filter_grid
        while grid.count():
            grid.takeAt(0).widget().deleteLater()
        self.bounds = {}
        locs = self.session.locs
        ranges = self.layer.filter.ranges
        for row, group in enumerate(FILTER_FIELDS):
            field = next((f for f in group if f in locs), None)
            if field is None:
                continue
            lo, hi = ranges.get(field, (None, None))
            lo_box, hi_box = _Bound(lo), _Bound(hi)
            for box in (lo_box, hi_box):
                box.editingFinished.connect(lambda f=field: self._on_bound(f))
            grid.addWidget(QLabel(field), row, 0)
            grid.addWidget(lo_box, row, 1)
            grid.addWidget(hi_box, row, 2)
            self.bounds[field] = (lo_box, hi_box)

        self.color.blockSignals(True)
        self.color.clear()
        self.color.addItem("intensity", None)
        for label, names in COLOR_FIELDS:
            field = next((f for f in names if f in locs), None)
            if field:
                self.color.addItem(label, field)
        self.color.blockSignals(False)
        display = self.layer.state.display
        self.lut.setCurrentText(display.lut if isinstance(display.lut, str) else "hot")
        self.contrast.setValue(display.contrast)
        self.gamma.setValue(display.gamma)
        self.mode.setCurrentText(self.layer.state.settings.mode)

    def _on_bound(self, field: str) -> None:
        lo, hi = (b.value() for b in self.bounds[field])
        self.layer.filter.set(field, lo, hi)
        self.session.changed("layer")

    def _on_render_settings(self) -> None:
        state = self.layer.state
        state.settings = dataclasses.replace(state.settings, mode=self.mode.currentText())
        self.session.changed("layer")

    def _on_color(self) -> None:
        state = self.layer.state
        field = self.color.currentData()
        state.settings = dataclasses.replace(state.settings, color_field=field)
        self.lut.setCurrentText(INTENSITY_LUT if field is None else FIELD_LUT)
        self.session.changed("layer")

    def _on_display(self) -> None:
        state = self.layer.state
        state.display = dataclasses.replace(state.display, lut=self.lut.currentText(),
                                            contrast=self.contrast.value(),
                                            gamma=self.gamma.value())
        self.session.changed("layer")
