"""The Render tab: layers, each with a filter and a way of being drawn.

The filter is one field at a time -- a drop-down, quick buttons for the
usual ones, and a histogram whose shaded region is the range -- plus the
numbers, for typing.  Which layer all of that belongs to is chosen in the
layer strip at the top.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QToolButton, QVBoxLayout, QWidget)

from .. import lut as luts
from ..filter import quantile_range
from ..render import FieldOfView
from ..session import Layer, Session
from ..viewer import COLOR_FIELDS, FIELD_LUT, INTENSITY_LUT
from .widgets import CollapsibleSection

# the fields with a quick button, best-named alternative first
QUICK_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("prec", ("loc_precision_nm", "loc_precision_pix")),
    ("LL", ("logl_rel",)),
    ("frame", ("frame",)),
    ("z", ("z_nm",)),
    ("phot", ("photons",)),
    ("PSF", ("sigma_nm", "sigma_pix")),
)
HIST_BINS = 120


class _Bound(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setMaximumWidth(72)
        self.setPlaceholderText("-")

    def value(self) -> Optional[float]:
        text = self.text().strip()
        return None if text in ("", "-") else float(text)

    def set(self, value: Optional[float]) -> None:
        self.setText("" if value is None else f"{value:g}")


class FilterWidget(QWidget):
    """One field of one layer's filter: histogram, range, numbers."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layer: Optional[Layer] = None
        self._hist_cache: Dict[Tuple[int, str], tuple] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        row = QHBoxLayout()
        self.quick: Dict[str, QToolButton] = {}
        self.quick_row = row
        self.field = QComboBox()
        self.field.setMinimumWidth(110)
        row.addWidget(self.field)
        layout.addLayout(row)

        self.plot = pg.PlotWidget(background=None)
        self.plot.setFixedHeight(110)
        self.plot.setMenuEnabled(False)
        self.plot.hideAxis("left")
        self.plot.setMouseEnabled(x=True, y=False)
        self.bars = pg.PlotCurveItem([0.0, 1.0], [0.0], pen=None,
                                     brush=(120, 120, 120, 160),
                                     fillLevel=0, stepMode="center")
        self.plot.addItem(self.bars)
        self.region = pg.LinearRegionItem(brush=(60, 120, 220, 50))
        self.plot.addItem(self.region)
        layout.addWidget(self.plot)

        numbers = QHBoxLayout()
        self.lo, self.hi = _Bound(), _Bound()
        numbers.addWidget(QLabel("min"))
        numbers.addWidget(self.lo)
        numbers.addWidget(QLabel("max"))
        numbers.addWidget(self.hi)
        self.clear = QPushButton("clear")
        self.clear.setToolTip("no bounds on this field")
        numbers.addWidget(self.clear)
        self.count = QLabel("")
        numbers.addWidget(self.count, 1, Qt.AlignRight)
        layout.addLayout(numbers)

        self.field.currentTextChanged.connect(self._show_field)
        self.region.sigRegionChangeFinished.connect(self._on_region)
        self.lo.editingFinished.connect(self._on_numbers)
        self.hi.editingFinished.connect(self._on_numbers)
        self.clear.clicked.connect(self._on_clear)

    # ------------------------------------------------------------ binding
    def bind(self, layer: Layer) -> None:
        """Show this layer's table and filter.  Keeps the field if it exists."""
        self.layer = layer
        self._hist_cache.clear()
        locs = layer.locs
        current = self.field.currentText()
        numeric = [n for n in locs if np.asarray(locs[n]).dtype.kind in "iuf"
                   and np.asarray(locs[n]).ndim == 1]
        self.field.blockSignals(True)
        self.field.clear()
        self.field.addItems(numeric)
        self.field.blockSignals(False)

        for b in self.quick.values():
            b.deleteLater()
        self.quick.clear()
        for label, names in QUICK_FIELDS:
            name = next((n for n in names if n in locs), None)
            if name is None:
                continue
            b = QToolButton(text=label, checkable=True, autoExclusive=True)
            b.setToolTip(name)
            b.clicked.connect(lambda _=False, n=name: self.field.setCurrentText(n))
            self.quick[name] = b
            self.quick_row.insertWidget(len(self.quick) - 1, b)
        first = current if current in numeric else next(iter(self.quick), None)
        if first is None and numeric:
            first = numeric[0]
        if first:
            self.field.setCurrentText(first)
        self._show_field()

    def refresh(self) -> None:
        """The filter changed elsewhere (undo, new table): redraw."""
        self._show_field()

    # ------------------------------------------------------------- showing
    def _histogram(self, name: str):
        key = (id(self.layer.locs), name)
        if key not in self._hist_cache:
            values = np.asarray(self.layer.locs[name], dtype=np.float64)
            lo, hi = quantile_range(self.layer.locs, name, 0.001, 0.999)
            if hi <= lo:
                hi = lo + 1.0
            counts, edges = np.histogram(values, bins=HIST_BINS, range=(lo, hi))
            self._hist_cache[key] = (counts, edges)
        return self._hist_cache[key]

    def _show_field(self) -> None:
        name = self.field.currentText()
        if self.layer is None or not name or name not in self.layer.locs:
            return
        for n, b in self.quick.items():
            b.setChecked(n == name)
        counts, edges = self._histogram(name)
        self.bars.setData(edges, counts.astype(np.float64))
        lo, hi = self.layer.filter.ranges.get(name, (None, None))
        self.lo.set(lo)
        self.hi.set(hi)
        # the region stands in for an open bound at the histogram's edge
        self.region.blockSignals(True)
        self.region.setRegion((edges[0] if lo is None else lo,
                               edges[-1] if hi is None else hi))
        self.region.blockSignals(False)
        self.plot.setXRange(edges[0], edges[-1], padding=0.02)
        self._update_count()

    def _update_count(self) -> None:
        f = self.layer.filter
        self.count.setText(f"{len(f)} / {len(self.layer.locs)}")
        bounded = set(f.ranges)
        for n, b in self.quick.items():
            b.setStyleSheet("font-weight: bold" if n in bounded and
                            f.ranges[n] != (None, None) else "")

    # ------------------------------------------------------------ editing
    def _apply(self, lo: Optional[float], hi: Optional[float]) -> None:
        name = self.field.currentText()
        if lo is None and hi is None:
            if name in self.layer.filter:
                self.layer.filter.remove(name)
        else:
            self.layer.filter.set(name, lo, hi)
        self._show_field()
        self.changed.emit()

    def _on_region(self) -> None:
        lo, hi = self.region.getRegion()
        _, edges = self._histogram(self.field.currentText())
        # dragged to the edge means "no bound", so the tail is not cut off
        self._apply(None if lo <= edges[0] else float(lo),
                    None if hi >= edges[-1] else float(hi))

    def _on_numbers(self) -> None:
        try:
            self._apply(self.lo.value(), self.hi.value())
        except ValueError:
            self._show_field()

    def _on_clear(self) -> None:
        self._apply(None, None)


class LayerStrip(QWidget):
    """One button per layer (which one the tab edits), a visible box, add/remove."""

    selected = Signal(int)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.current = 0
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.buttons: List[QToolButton] = []
        self.visible = QCheckBox()
        self.visible.setToolTip("visible")
        self.name = QLineEdit()
        self.name.setToolTip("layer name")
        self.name.setMaximumWidth(110)
        self.name.editingFinished.connect(self._on_name)
        self.add = QToolButton(text="+")
        self.remove = QToolButton(text="-")
        self.add.clicked.connect(self._add)
        self.remove.clicked.connect(self._remove)
        self.visible.toggled.connect(self._on_visible)
        self.rebuild()

    def rebuild(self) -> None:
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget() and item.widget() not in (self.visible, self.name,
                                                       self.add, self.remove):
                item.widget().deleteLater()
        self.buttons = []
        for i, layer in enumerate(self.session.layers):
            b = QToolButton(text=str(i + 1), checkable=True, autoExclusive=True)
            b.setToolTip(layer.name)
            b.clicked.connect(lambda _=False, i=i: self.select(i))
            self.buttons.append(b)
            self.layout_.addWidget(b)
        self.layout_.addWidget(self.add)
        self.layout_.addWidget(self.remove)
        self.layout_.addWidget(self.visible)
        self.layout_.addWidget(self.name)
        self.layout_.addStretch(1)
        self.remove.setEnabled(len(self.buttons) > 1)
        self.select(min(self.current, len(self.buttons) - 1))

    def select(self, i: int) -> None:
        self.current = i
        self.buttons[i].setChecked(True)
        self.visible.blockSignals(True)
        self.visible.setChecked(self.session.layers[i].visible)
        self.visible.blockSignals(False)
        self.name.setText(self.session.layers[i].name)
        self.selected.emit(i)

    def _on_name(self) -> None:
        layer = self.session.layers[self.current]
        layer.name = self.name.text().strip() or f"layer {self.current + 1}"
        self.buttons[self.current].setToolTip(layer.name)

    def _add(self) -> None:
        self.session.add_layer()
        self.current = len(self.session.layers) - 1

    def _remove(self) -> None:
        self.session.remove_layer(self.current)

    def _on_visible(self, on: bool) -> None:
        self.session.layers[self.current].visible = on
        self.session.changed("layers")


class Overview(QWidget):
    """The whole field of view, small; click to centre the main image there."""

    def __init__(self, session: Session, view, parent=None):
        super().__init__(parent)
        self.session = session
        self.view = view
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setFixedHeight(150)
        self.box = self.graphics.addViewBox(lockAspect=True, invertY=True,
                                           enableMenu=False, enableMouse=False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.box.addItem(self.image)
        self.frame = pg.QtWidgets.QGraphicsRectItem()
        self.frame.setPen(pg.mkPen((255, 255, 0), width=1))
        self.frame.setBrush(pg.mkBrush(None))
        self.box.addItem(self.frame)
        layout.addWidget(self.graphics)
        row = QHBoxLayout()
        self.update_button = QPushButton("update")
        self.update_button.setToolTip("re-render with the current layers")
        row.addWidget(self.update_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.update_button.clicked.connect(self.update_image)
        self.image.mouseClickEvent = self._on_click
        if view is not None:
            view.view.sigRangeChanged.connect(self._track)
        session.on_change(lambda what: self.update_image() if what == "locs" else None)

    def update_image(self) -> None:
        if self.view is None or not len(self.session.locs):
            self.image.clear()
            return
        (x0, x1), (y0, y1) = self.session.layers[0].state.full_view()
        fov = FieldOfView.fit((x0, x1), (y0, y1), 300, 200)
        rgb, _ = self.view.composite(fov)
        self.image.setImage(np.ascontiguousarray(rgb), levels=[0, 1], autoLevels=False)
        self.image.setRect(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0))
        self.box.setRange(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0),
                          padding=0)
        self._track()

    def _track(self) -> None:
        if self.view is not None:
            self.frame.setRect(self.view.view.viewRect())

    def _on_click(self, event) -> None:
        if self.view is None:
            return
        pos = self.image.mapToView(event.pos())
        self.view.center_on(pos.x(), pos.y())
        event.accept()


class RenderTab(QWidget):
    def __init__(self, session: Session, view=None, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.strip = LayerStrip(session)
        layout.addWidget(self.strip)
        self.filter = FilterWidget()
        layout.addWidget(CollapsibleSection("filter", self.filter, expanded=True))

        display = QWidget()
        form = QFormLayout(display)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(2)
        self.mode = QComboBox()
        self.mode.addItems(["precision", "gauss", "hist"])
        self.color = QComboBox()
        self.lut = QComboBox()
        self.lut.addItems(luts.names())
        self.contrast = QDoubleSpinBox(minimum=0, maximum=6, singleStep=0.1, decimals=2)
        self.grouped = QCheckBox("grouped")
        self.grouped.setToolTip("one entry per blink instead of one per frame; "
                                "links the table on first use")
        form.addRow("render", self.mode)
        form.addRow("colour by", self.color)
        form.addRow("LUT", self.lut)
        form.addRow("contrast", self.contrast)
        form.addRow("", self.grouped)
        more = QWidget()
        more_form = QFormLayout(more)
        more_form.setContentsMargins(0, 0, 0, 0)
        more_form.setVerticalSpacing(2)
        self.sigma = QDoubleSpinBox(minimum=0.1, maximum=1000, singleStep=1, decimals=1)
        self.sigma.setToolTip("rendering sigma for mode 'gauss', in data units")
        self.gamma = QDoubleSpinBox(minimum=0.1, maximum=3, singleStep=0.1, decimals=2)
        more_form.addRow("sigma (gauss)", self.sigma)
        more_form.addRow("gamma", self.gamma)
        form.addRow(CollapsibleSection("more", more, expanded=False))
        layout.addWidget(CollapsibleSection("display", display, expanded=True))
        self.overview = Overview(session, view)
        layout.addWidget(CollapsibleSection("overview", self.overview, expanded=True))
        layout.addStretch(1)

        self.strip.selected.connect(self._bind_layer)
        self.filter.changed.connect(lambda: session.changed("layer"))
        self.mode.currentTextChanged.connect(self._on_render_settings)
        self.sigma.valueChanged.connect(self._on_render_settings)
        self.color.currentIndexChanged.connect(self._on_color)
        self.lut.currentTextChanged.connect(self._on_display)
        self.contrast.valueChanged.connect(self._on_display)
        self.gamma.valueChanged.connect(self._on_display)
        self.grouped.toggled.connect(self._on_grouped)
        self._appended = 0
        session.on_change(self._on_session)
        self._bind_layer(0)

    @property
    def layer(self) -> Layer:
        return self.session.layers[self.strip.current]

    def _on_session(self, what: str) -> None:
        if what == "locs":
            self._bind_layer(self.strip.current)
        elif what == "layers":
            self.strip.rebuild()
        elif what == "append":
            self._appended += 1
            if self._appended % 5 == 1:      # the histogram need not follow every block
                self.filter.bind(self.layer)

    def _bind_layer(self, index: int) -> None:
        """Point every control at one layer, without firing their signals."""
        layer = self.session.layers[index]
        widgets = (self.mode, self.sigma, self.color, self.lut, self.contrast,
                   self.gamma, self.grouped)
        for w in widgets:
            w.blockSignals(True)
        self.filter.bind(layer)
        locs = layer.locs
        self.color.clear()
        self.color.addItem("intensity", None)
        for label, names in COLOR_FIELDS:
            field = next((f for f in names if f in locs), None)
            if field:
                self.color.addItem(label, field)
        settings, display = layer.state.settings, layer.state.display
        self.mode.setCurrentText(settings.mode)
        self.sigma.setValue(settings.sigma)
        i = self.color.findData(settings.color_field)
        self.color.setCurrentIndex(max(i, 0))
        self.lut.setCurrentText(display.lut if isinstance(display.lut, str) else "hot")
        self.contrast.setValue(display.contrast)
        self.gamma.setValue(display.gamma)
        self.grouped.setChecked(layer.grouped)
        for w in widgets:
            w.blockSignals(False)

    def _on_render_settings(self) -> None:
        state = self.layer.state
        state.settings = dataclasses.replace(state.settings, mode=self.mode.currentText(),
                                             sigma=self.sigma.value())
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

    def _on_grouped(self, on: bool) -> None:
        self.layer.show_grouped(on)      # links on first use: a moment
        self.filter.bind(self.layer)     # the grouped table has its own filter
        self.session.changed("layer")
