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
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QPushButton,
                               QScrollArea, QSlider, QToolButton, QVBoxLayout, QWidget)

from .. import lut as luts
from ..filter import quantile_range  # noqa: F401  (kept for callers)
from ..render import FieldOfView, RenderAxes, axis_unit, is_position
from ..session import Layer, Session
from ..viewer import FIELD_LUT, INTENSITY_LUT
from .render_view import nice_below
from .widgets import CollapsibleSection, detach_to_window

# the fields with a quick button, best-named alternative first
QUICK_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("prec", ("loc_precision_nm", "loc_precision_pix")),
    ("LL", ("logl_rel",)),
    ("frame", ("frame",)),
    ("z", ("z_nm",)),
    ("phot", ("photons",)),
    ("PSF", ("sigma_nm", "sigma_pix")),
    ("ch", ("channel",)),
    ("file", ("filenumber",)),
)
HIST_BINS = 120
HIST_SAMPLE = 2_000_000     # beyond this the histogram is of a random sample


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

    def __init__(self, session: Optional[Session] = None, parent=None):
        super().__init__(parent)
        self.session = session
        self.layer: Optional[Layer] = None
        self.layer_index = 0
        self._hist_cache: Dict[Tuple[int, str], tuple] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # The quick buttons have a row to themselves and the drop-down and the
        # count share the next.  All on one line, a table with all eight quick
        # fields made the panel ~440 px wide in a 380 px window, and the scroll
        # area (no horizontal bar) cut off the histogram's right end -- with
        # the handle of the upper bound on it.
        row = QHBoxLayout()
        self.quick: Dict[str, QToolButton] = {}
        self.quick_row = row
        row.addStretch(1)
        layout.addLayout(row)
        field_row = QHBoxLayout()
        self.field = QComboBox()
        self.field.setMinimumWidth(110)
        field_row.addWidget(self.field)
        self.count = QLabel("")
        field_row.addWidget(self.count, 1, Qt.AlignRight)
        layout.addLayout(field_row)

        # for the file column: names to tick, not a histogram
        self.files = QListWidget()
        self.files.setFixedHeight(110)
        self.files.itemChanged.connect(self._on_files)
        self.files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.files.customContextMenuRequested.connect(self._files_menu)
        self.files.setToolTip("tick the files this layer shows; right-click to remove a file")
        self.files.hide()
        layout.addWidget(self.files)
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
        numbers.addStretch(1)
        layout.addLayout(numbers)

        self.field.currentTextChanged.connect(self._show_field)
        self.region.sigRegionChangeFinished.connect(self._on_region)
        self.lo.editingFinished.connect(self._on_numbers)
        self.hi.editingFinished.connect(self._on_numbers)
        self.clear.clicked.connect(self._on_clear)
        self.all_files = QPushButton("all")
        self.all_files.setToolTip("tick every file")
        self.all_files.clicked.connect(self._all_files)
        self.no_files = QPushButton("none")
        self.no_files.setToolTip("untick every file")
        self.no_files.clicked.connect(self._no_files)
        for b in (self.all_files, self.no_files):
            b.hide()
            numbers.insertWidget(numbers.count() - 1, b)

    # ------------------------------------------------------------ binding
    def bind(self, layer: Layer) -> None:
        """Show this layer's table and filter.  Keeps the field if it exists."""
        self.layer = layer
        self._hist_cache.clear()
        locs = layer.locs
        current = self.current_field()
        numeric = [n for n in locs if np.asarray(locs[n]).dtype.kind in "iuf"
                   and np.asarray(locs[n]).ndim == 1]
        self.field.blockSignals(True)
        self.field.clear()
        for name in numeric:
            self.field.addItem(name, name)   # the data stays the plain name
        self.field.blockSignals(False)
        self._mark_bounded()

        for b in self.quick.values():
            b.deleteLater()
        self.quick.clear()
        for label, names in QUICK_FIELDS:
            name = next((n for n in names if n in locs), None)
            if name is None:
                continue
            b = QToolButton(text=label, checkable=True, autoExclusive=True)
            b.setToolTip(name)
            b.clicked.connect(lambda _=False, n=name: self._select(n))
            self.quick[name] = b
            self.quick_row.insertWidget(len(self.quick) - 1, b)
        first = current if current in numeric else next(iter(self.quick), None)
        if first is None and numeric:
            first = numeric[0]
        if first:
            self._select(first)
        self._show_field()

    def refresh(self) -> None:
        """The filter changed elsewhere (undo, new table): redraw."""
        self._show_field()

    def current_field(self) -> str:
        """The column shown, without the marker a bounded field carries."""
        name = self.field.currentData()
        return name if isinstance(name, str) else self.field.currentText()

    def _select(self, name: str) -> None:
        index = self.field.findData(name)
        if index >= 0:
            self.field.setCurrentIndex(index)

    def _mark_bounded(self) -> None:
        """Bold, with a dot, every field that has a bound on it.

        A filter on a field that is not the one on show is otherwise invisible,
        and a table that renders black because of a bound nobody remembers
        setting is the result.  An unbounded range -- (None, None) -- is no
        bound and is not marked.
        """
        if self.layer is None:
            return
        ranges = self.layer.filter.ranges
        bounded = {n for n, r in ranges.items() if r != (None, None)}
        plain, bold = QFont(self.field.font()), QFont(self.field.font())
        bold.setBold(True)
        for i in range(self.field.count()):
            name = self.field.itemData(i)
            on = name in bounded
            self.field.setItemText(i, f"\u25cf {name}" if on else name)
            self.field.setItemData(i, bold if on else plain, Qt.FontRole)
            lo, hi = ranges.get(name, (None, None))
            self.field.setItemData(
                i, (f"filtered: {'' if lo is None else f'{lo:g}'} .. "
                    f"{'' if hi is None else f'{hi:g}'}") if on else None,
                Qt.ToolTipRole)

    # ------------------------------------------------------------- showing
    def _histogram(self, name: str):
        key = (id(self.layer.locs), name)
        if key not in self._hist_cache:
            values = np.asarray(self.layer.locs[name])
            if values.size > HIST_SAMPLE:        # a sample says the same, 25x faster
                pick = np.random.default_rng(0).integers(0, values.size, HIST_SAMPLE)
                values = values[pick]
            values = values.astype(np.float64)
            finite = values[np.isfinite(values)]
            lo, hi = ((float(np.quantile(finite, 0.01)), float(np.quantile(finite, 0.99)))
                      if finite.size else (0.0, 1.0))                  # not the outliers
            if hi <= lo:
                hi = lo + 1.0
            counts, edges = np.histogram(values, bins=HIST_BINS, range=(lo, hi))
            self._hist_cache[key] = (counts, edges)
        return self._hist_cache[key]

    def _show_field(self) -> None:
        name = self.current_field()
        if self.layer is None or not name or name not in self.layer.locs:
            return
        for n, b in self.quick.items():
            b.setChecked(n == name)
        by_file = name == "filenumber" and self.session is not None
        self.files.setVisible(by_file)
        self.plot.setVisible(not by_file)
        for w in (self.lo, self.hi, self.clear):
            w.setVisible(not by_file)
        self.all_files.setVisible(by_file)
        self.no_files.setVisible(by_file)
        if by_file:
            self._show_files()
            self._update_count()
            return
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

    def _show_files(self) -> None:
        """One tickable row per file, plus 'all'."""
        chosen = self.layer.filter._masks.get("files")
        present = {int(n) for n in np.unique(self.layer.locs["filenumber"])}  # float once grouped
        self.files.blockSignals(True)
        self.files.clear()
        names = self.session.file_names()
        for number in sorted(present):
            name = names[number] if number < len(names) else f"file {number}"
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, number)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            on = chosen is None or bool(chosen[self.layer.locs["filenumber"] == number].any())
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.files.addItem(item)
        self.files.blockSignals(False)

    def _files_menu(self, pos) -> None:
        item = self.files.itemAt(pos)
        if item is None:
            return
        number = item.data(Qt.UserRole)
        menu = QMenu(self)
        menu.addAction(f"remove '{item.text()}' from the session",
                       lambda: self.session.remove_file(number))
        menu.exec(self.files.mapToGlobal(pos))

    def _on_files(self) -> None:
        numbers = [self.files.item(i).data(Qt.UserRole) for i in range(self.files.count())
                   if self.files.item(i).checkState() == Qt.Checked]
        every = numbers and len(numbers) == self.files.count()
        self.layer.set_files(None if every else numbers)
        self._update_count()
        self.changed.emit()

    def _update_count(self) -> None:
        f = self.layer.filter
        text = f"{len(f)} / {len(self.layer.locs)}"
        if self.session is not None and self.session.roi is not None:
            text += f", {len(self.session.shown_selection(self.layer_index))} in ROI"
        self.count.setText(text)
        bounded = set(f.ranges)
        for n, b in self.quick.items():
            b.setStyleSheet("font-weight: bold" if n in bounded and
                            f.ranges[n] != (None, None) else "")
        self._mark_bounded()

    # ------------------------------------------------------------ editing
    def _apply(self, lo: Optional[float], hi: Optional[float]) -> None:
        name = self.current_field()
        if lo is None and hi is None:
            self.layer.remove_bound(name)
        else:
            self.layer.set_bound(name, lo, hi)
        self._show_field()
        self.changed.emit()

    def _on_region(self) -> None:
        lo, hi = self.region.getRegion()
        _, edges = self._histogram(self.current_field())
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

    def _all_files(self) -> None:
        self.layer.set_files(None)
        self._show_files()
        self._update_count()
        self.changed.emit()

    def _no_files(self) -> None:
        self.layer.set_files([])
        self._show_files()
        self._update_count()
        self.changed.emit()


class LayerStrip(QWidget):
    """One button per layer (which one the tab edits), a visible box, add/remove."""

    selected = Signal(int)
    add_image_requested = Signal()

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
        self.add = QToolButton(text="+", popupMode=QToolButton.InstantPopup)
        self.add.setToolTip("add a layer")
        menu = QMenu(self.add)
        menu.addAction("localizations", self._add)
        menu.addAction("image...", self.add_image_requested)
        self.add.setMenu(menu)
        self.remove = QToolButton(text="-")
        self.remove.clicked.connect(self._remove)
        # two layers drift apart over a session and one of them is right;
        # setting the other one up again by hand is a dozen controls
        self.copy = QToolButton(text="\u2913")           # downwards arrow to bar
        self.copy.setToolTip("copy settings from another layer")
        self.copy.clicked.connect(self._copy)
        self.visible.toggled.connect(self._on_visible)
        self.rebuild()

    def rebuild(self) -> None:
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget() and item.widget() not in (self.visible, self.name,
                                                       self.add, self.remove,
                                                       self.copy):
                item.widget().deleteLater()
        self.buttons = []
        for i, layer in enumerate(self.session.layers):
            b = QToolButton(text=str(i + 1) + ("i" if layer.is_image else ""),
                            checkable=True, autoExclusive=True)
            b.setToolTip(f"{layer.name}\nright-click: show / hide")
            b.clicked.connect(lambda _=False, i=i: self.select(i))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _, i=i: self._toggle_visible(i))
            self._style(b, layer.visible)
            self.buttons.append(b)
            self.layout_.addWidget(b)
        self.layout_.addWidget(self.add)
        self.layout_.addWidget(self.remove)
        self.layout_.addWidget(self.copy)
        self.layout_.addWidget(self.visible)
        self.layout_.addWidget(self.name)
        self.layout_.addStretch(1)
        self.remove.setEnabled(len(self.buttons) > 1)
        self.copy.setEnabled(sum(not l.is_image for l in self.session.layers) > 1)
        self.select(min(self.current, len(self.buttons) - 1))

    def select(self, i: int) -> None:
        self.current = i
        self.buttons[i].setChecked(True)
        for j, button in enumerate(self.buttons):
            self._style(button, self.session.layers[j].visible)
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
        """The new layer copies this one, and the tab moves onto it: `rebuild`
        has already run from the session's change, so this only re-selects."""
        self.session.add_layer(like=self.current)
        self.select(len(self.session.layers) - 1)

    def _remove(self) -> None:
        self.session.remove_layer(self.current)

    def _copy(self) -> None:
        """Ask which layer to take settings from, and which of them."""
        if self.session.layers[self.current].is_image:
            return
        from .dialogs import CopyLayerDialog
        CopyLayerDialog(self.session, self.current, self).exec()

    def _on_visible(self, on: bool) -> None:
        self.session.layers[self.current].visible = on
        self._style(self.buttons[self.current], on)
        self.session.changed("layers")

    def _toggle_visible(self, i: int) -> None:
        """Right-click on a layer's number: show or hide it, stay where we are."""
        layer = self.session.layers[i]
        layer.visible = not layer.visible
        self._style(self.buttons[i], layer.visible)
        if i == self.current:
            self.visible.blockSignals(True)
            self.visible.setChecked(layer.visible)
            self.visible.blockSignals(False)
        self.session.changed("layers")

    def _style(self, button: QToolButton, visible: bool) -> None:
        """Two cues that must not be confused.  *Visible* is the text: bold
        when the layer is drawn, struck through and grey when it is not.
        *Selected* -- the layer every control below edits -- is the highlighted
        background, so toggling visibility never loses track of it.
        """
        chosen = 0 <= self.current < len(self.buttons) and button is self.buttons[self.current]
        colour = "palette(highlighted-text)" if chosen else ("palette(text)" if visible else "gray")
        text = ("font-weight: bold" if visible else "text-decoration: line-through")
        background = ("background-color: palette(highlight); "
                      "border: 1px solid palette(highlight)"
                      if chosen else "border: 1px solid transparent")
        button.setStyleSheet(f"QToolButton {{ color: {colour}; {text}; {background} }}")


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
        self.parameters_button = QPushButton("parameters...")
        self.parameters_button.setToolTip("settings that are not a layer's: grouping")
        self.parameters_button.clicked.connect(self._parameters)
        row.addWidget(self.parameters_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.update_button.clicked.connect(self.update_image)
        self.image.mouseClickEvent = self._on_click
        if view is not None:
            view.view.sigRangeChanged.connect(self._track)
        # A new table draws itself: this is 300 x 200 pixels, which is 0.4 s
        # for ten million localizations -- next to nothing beside the read and
        # the linking that have just finished -- and a panel headed "overview"
        # that is blank until a button is found is not an overview.  Deferred
        # by a beat so the window paints the new file first; the button stays,
        # for after a filter or a LUT has been changed.
        session.on_change(self._on_session)

    def _on_session(self, what: str) -> None:
        if what == "locs":
            self.image.clear()
            QTimer.singleShot(0, self.update_image)

    def _parameters(self) -> None:
        from .dialogs import ParametersDialog
        dialog = ParametersDialog(self.session, self)
        dialog.exec()

    def update_image(self) -> None:
        if self.view is None or not (len(self.session.locs)
                                     or any(l.is_image for l in self.session.layers)):
            self.image.clear()
            return
        # the same way up as the main view: a plot of two columns is a plot
        self.box.invertY(self.session.axes().is_default)
        (x0, x1), (y0, y1) = self.session.full_view()
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
        self.view.window().show()          # closed by accident: bring it back
        self.view.center_on(pos.x(), pos.y())
        event.accept()


#: how many render units the 1-99 % of an axis is fitted onto.  An arbitrary
#: number -- only the ratio of the two scales is a picture -- but a round one
#: keeps the pixel sizes and sigmas people type in a familiar range.
FIT_SPAN = 1000.0


def numeric_fields(locs) -> List[str]:
    """The 1-D numeric columns of a table, in the order it carries them."""
    return [n for n in locs if np.asarray(locs[n]).dtype.kind in "iuf"
            and np.asarray(locs[n]).ndim == 1]


class AxesSection(QWidget):
    """Which columns the picture's axes are: SMAP's versatile renderer.

    Hidden under "axes" because for every ordinary picture the answer is x and
    y, and opening it is what turns the same renderer -- the same layers,
    LUTs, contrast, filters, ROIs and 3D box -- into a picture of photons
    against frame, or of a fit parameter against another.

    The axes belong to the picture rather than to one layer, since the layers
    are composited onto one grid, so everything here is set on all of them at
    once, as the white background is.
    """

    changed = Signal()          # the widths moved with the axes; re-read them

    def __init__(self, session: Session, view=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.view = view
        self._filling = False
        self._kept = None       # the widths from before the mapping
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(2)
        self.fields: Dict[str, QComboBox] = {}
        self.scales: Dict[str, _Bound] = {}
        for axis, label in (("x", "x"), ("y", "y"), ("z", "z (3D)")):
            combo = QComboBox()
            combo.setToolTip(f"the column the {axis} axis of the render grid is")
            scale = _Bound()
            scale.setToolTip("units of that column per render unit: the two "
                             "scales are what makes the pixels non-square, and "
                             "the grid itself stays square so that the zoom, "
                             "the ROIs and the 3D box are unchanged")
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(combo, 1)
            row.addWidget(QLabel("/"))
            row.addWidget(scale)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)
            self.fields[axis] = combo
            self.scales[axis] = scale
            combo.currentIndexChanged.connect(lambda _, a=axis: self._on_field(a))
            scale.editingFinished.connect(self._apply)
        self.same = QCheckBox("same scale on both axes")
        self.same.setToolTip("one scale for x and y, so distances stay true.  "
                             "Ticked by itself when the two axes are the same "
                             "quantity -- x against z is a picture, and "
                             "stretching it would be a lie")
        form.addRow("", self.same)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self.fit = QPushButton("fit")
        self.fit.setToolTip("scale each axis so its 1-99 % fills the picture, "
                            "and frame the view on it")
        self.reset = QPushButton("reset")
        self.reset.setToolTip("back to x and y: the ordinary picture")
        buttons.addWidget(self.fit)
        buttons.addWidget(self.reset)
        buttons.addStretch(1)
        form.addRow("", buttons)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.hide()
        form.addRow("", self.note)
        self.same.toggled.connect(self._apply)
        self.fit.clicked.connect(self._fit)
        self.reset.clicked.connect(self._reset)

    # ------------------------------------------------------------- binding
    def bind(self) -> None:
        """Show the session's axes, without firing anything."""
        locs = self.session.locs
        axes = self.session.axes()
        names = numeric_fields(locs)
        self._filling = True
        for axis, combo in self.fields.items():
            current = getattr(axes, axis)
            combo.clear()
            combo.addItem("auto", None)
            combo.addItems(names)
            combo.setCurrentIndex(max(0, combo.findText(current)) if current else 0)
            self.scales[axis].set(getattr(axes, f"{axis}_scale"))
        self.same.setChecked(axes.x_scale == axes.y_scale)
        self._filling = False
        self._describe(axes)

    def _chosen(self) -> RenderAxes:
        def field(axis: str) -> Optional[str]:
            combo = self.fields[axis]
            return combo.currentText() if combo.currentIndex() > 0 else None

        def scale(axis: str) -> float:
            value = self.scales[axis].value()
            return float(value) if value else 1.0

        return RenderAxes(x=field("x"), y=field("y"), z=field("z"),
                          x_scale=scale("x"), y_scale=scale("y"), z_scale=scale("z"))

    def _describe(self, axes: RenderAxes) -> None:
        """Say what is no longer in nanometres, where it would mislead."""
        if axes.is_default:
            self.note.hide()
            return
        locs = self.session.locs
        try:
            x_name, y_name = axes.names(locs)
        except KeyError:
            self.note.hide()
            return
        self.note.setText(f"{x_name} across, {y_name} up.  Sites and anything "
                          "that measures nanometres read the columns, not the "
                          "picture; the scale bars are in the axes' own units.")
        self.note.show()

    # -------------------------------------------------------------- editing
    def _on_field(self, axis: str) -> None:
        """A new field: scale it onto the screen rather than leaving it at 1.

        A frame number is in the thousands and a photon count in the hundreds
        of thousands; at scale 1 the first picture would be a single bright
        pixel or an empty field, and the user would have to guess two numbers
        before seeing anything.
        """
        if self._filling:
            return
        locs = self.session.locs
        axes = self._chosen()
        if len(locs):
            try:                       # the same quantity on both axes: one scale
                x_name, y_name = axes.names(locs)
            except KeyError:
                x_name = y_name = None
            if x_name is not None:
                self._filling = True
                self.same.setChecked(axis_unit(x_name) == axis_unit(y_name))
                self._filling = False
        self._fit()

    def _apply(self, *_) -> None:
        if self._filling:
            return
        axes = self._chosen()
        if self.same.isChecked():
            axes = dataclasses.replace(axes, y_scale=axes.x_scale)
        self._set(axes)

    def _reset(self) -> None:
        self._set(RenderAxes())
        self.bind()

    def _set(self, axes: RenderAxes, frame=None) -> None:
        """Put the session on these axes, and show the result.

        ``frame`` is the box to look at, in render units; without one the view
        goes back to showing everything, which is what a reset means.
        """
        locs = self.session.locs
        if len(locs):
            axes = axes.normalised(locs)
        self.session.set_axes(axes)
        self._sigmas(axes)
        self.changed.emit()
        self._describe(axes)
        if self.view is None:
            return
        if frame is None:
            self.view.reset()
        else:
            self.view.frame_on(*frame)

    def _sigmas(self, axes: RenderAxes) -> None:
        """Zero the width of an axis that is not a position, and put it back.

        A leftover 10 nm is not a width in photons, and the honest default on
        an axis with no precision behind it is to bin rather than to blur --
        but the number that was typed for the ordinary picture should still be
        there on the way back to it, rather than a silent histogram.
        """
        locs = self.session.locs
        if not len(locs):
            return
        try:
            x_name, y_name = axes.names(locs)
        except KeyError:
            return
        zero = (not is_position(x_name), not is_position(y_name))
        if any(zero) and self._kept is None:
            first = self.session.layers[self.session.first_locs_layer()].state.settings
            self._kept = (first.sigma, first.sigma_y)
        for layer in self.session.layers:
            if layer.is_image:
                continue
            settings = layer.state.settings
            changes = {}
            for i, name in enumerate(("sigma", "sigma_y")):
                if zero[i]:
                    changes[name] = 0.0
                elif self._kept is not None and getattr(settings, name) == 0.0:
                    changes[name] = self._kept[i]      # theirs, from before
            if changes:
                layer.state.settings = dataclasses.replace(settings, **changes)
        if not any(zero):
            self._kept = None

    def _fit(self, *_) -> None:
        """Scale both axes so their 1-99 % fills the picture."""
        locs = self.session.locs
        if not len(locs):
            return
        axes = self._chosen()
        try:
            names = dict(zip(("x", "y"), axes.names(locs)))
        except KeyError:
            return
        names["z"] = axes.depth_name(locs)
        ranges = {a: _quantile_range(locs, names[a]) for a in ("x", "y", "z")}
        spans = {a: (r[1] - r[0]) if r else None for a, r in ranges.items()}
        lateral = [spans[a] for a in ("x", "y") if spans[a]]
        if not lateral:
            return
        # A position axis is left at one unit per nanometre.  It could be
        # scaled like any other, and then an ROI drawn on the picture would be
        # in tenths of a nanometre and a site somewhere else entirely: a
        # picture of x against z is still a picture of a place, and the rest
        # of the program is entitled to read it as one.
        keep = {a: bool(names[a]) and is_position(names[a]) for a in ("x", "y", "z")}
        # what one render unit is worth, taken from an axis that keeps its own
        # scale so the two spans still land on one screen together
        target = next((spans[a] for a in ("x", "y") if keep[a] and spans[a]), FIT_SPAN)
        scales = {a: 1.0 if keep[a] else nice_below((spans[a] or target) / target)
                  for a in ("x", "y", "z")}
        if self.same.isChecked():
            scales["y"] = scales["x"]
        self._filling = True
        for axis, value in scales.items():
            self.scales[axis].set(value)
        self._filling = False
        frame = tuple(tuple(v / scales[a] for v in ranges[a]) for a in ("x", "y"))
        self._set(dataclasses.replace(axes, x_scale=scales["x"], y_scale=scales["y"],
                                      z_scale=scales["z"]), frame=frame)


def _quantile_range(locs, name: Optional[str]):
    """The 1-99 % of a column, which is what the picture should be filled by.

    The extremes are not: one localization with a photon count a thousand
    times the rest would leave the structure in a corner, and it is the same
    reason the display saturates a quantile rather than the maximum.
    """
    if name is None or name not in locs:
        return None
    values = np.asarray(locs[name], np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    lo, hi = (float(v) for v in np.quantile(values, (0.01, 0.99)))
    return (lo, hi) if hi > lo else None


class RenderTab(QWidget):
    def __init__(self, session: Session, view=None, parent=None):
        super().__init__(parent)
        self.session = session
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        layout = QVBoxLayout(inner)              # the tab's content; scrolls when short
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # the whole picture, every visible layer, above the layer strip
        self.overview = Overview(session, view)
        section = CollapsibleSection("overview", self.overview, expanded=True, detachable=True)
        section.detach_requested.connect(lambda: detach_to_window(section, self.window()))
        layout.addWidget(section)
        self.strip = LayerStrip(session)
        layout.addWidget(self.strip)
        self.filter = FilterWidget(session)
        self.filter_section = CollapsibleSection("filter", self.filter, expanded=True)
        layout.addWidget(self.filter_section)

        image = QWidget()
        image_form = QFormLayout(image)
        image_form.setContentsMargins(0, 0, 0, 0)
        image_form.setVerticalSpacing(2)
        self.image_name = QLabel("")
        self.image_pixelsize = QDoubleSpinBox(minimum=0.01, maximum=1e6, decimals=2)
        self.image_x0 = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=1)
        self.image_y0 = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=1)
        self.image_frame = QSlider(Qt.Horizontal)
        image_form.addRow("file", self.image_name)
        image_form.addRow("pixel size (nm)", self.image_pixelsize)
        image_form.addRow("x0 (nm)", self.image_x0)
        image_form.addRow("y0 (nm)", self.image_y0)
        image_form.addRow("frame", self.image_frame)
        self.image_section = CollapsibleSection("image", image, expanded=True)
        self.image_section.hide()
        layout.addWidget(self.image_section)
        for w in (self.image_pixelsize, self.image_x0, self.image_y0):
            w.valueChanged.connect(self._on_image)
        self.image_frame.valueChanged.connect(self._on_image)

        display = QWidget()
        form = QFormLayout(display)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(2)
        self.mode = QComboBox()
        self.mode.addItems(["precision", "gauss", "hist"])
        self.color = QComboBox()
        self.color.addItem("intensity", None)
        self.color.addItem("field", "field")
        self.color_field = QComboBox()
        self.color_field.setToolTip("the column the colour encodes")
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.addWidget(self.color)
        color_row.addWidget(self.color_field, 1)
        self.color_lo, self.color_hi = _Bound(), _Bound()
        self.color_auto = QPushButton("auto")
        self.color_auto.setToolTip("the field's 0.5-99.5% range")
        range_row = QHBoxLayout()
        range_row.setContentsMargins(0, 0, 0, 0)
        for w in (self.color_lo, QLabel("to"), self.color_hi, self.color_auto):
            range_row.addWidget(w)
        range_row.addStretch(1)
        self.color_range_row = QWidget()
        self.color_range_row.setLayout(range_row)
        self.color_lo.editingFinished.connect(self._on_color_range)
        self.color_hi.editingFinished.connect(self._on_color_range)
        self.color_auto.clicked.connect(self._auto_color_range)
        self.lut = QComboBox()
        self.lut.addItems(luts.names())
        self.invert = QCheckBox("invert")
        self.invert.setToolTip("the opposite colour at the same brightness: red "
                               "becomes cyan.  For black on white pick the "
                               "gray_inverted LUT -- inverting is about hue, not "
                               "about which end is bright.")
        # The two inversions are not the same picture past a ramp of one hue,
        # and which one is wanted depends on what the picture is for, so it is
        # a choice rather than a decision made here.
        self.invert_mode = QComboBox()
        self.invert_mode.addItem("SMAP", "sum")
        self.invert_mode.addItem("grey", "complement")
        self.invert_mode.setToolTip(
            "SMAP: sum(c) - c, as SMAP's lutinvert -- inverted hot runs black "
            "through cyan to white.\n"
            "grey: the hue exactly half a turn away at the same lightness, so "
            "a layer over its inverse goes grey where the two coincide; "
            "inverted hot runs on to blue instead of saturating at white.")
        self.invert_mode.setMaximumWidth(90)
        lut_row = QHBoxLayout()
        lut_row.setContentsMargins(0, 0, 0, 0)
        lut_row.addWidget(self.lut, 1)
        lut_row.addWidget(self.invert)
        lut_row.addWidget(self.invert_mode)
        self.contrast = QDoubleSpinBox(minimum=0, maximum=6, singleStep=0.1, decimals=2)
        self.grouped = QCheckBox("grouped")
        self.grouped.setToolTip("one entry per blink instead of one per frame; "
                                "links the table on first use")
        form.addRow("render", self.mode)
        form.addRow("colour by", color_row)
        form.addRow("colour range", self.color_range_row)
        form.addRow("LUT", lut_row)
        form.addRow("contrast", self.contrast)
        form.addRow("", self.grouped)
        more = QWidget()
        more_form = QFormLayout(more)
        more_form.setContentsMargins(0, 0, 0, 0)
        more_form.setVerticalSpacing(2)
        self.sigma = QDoubleSpinBox(minimum=0.0, maximum=1000, singleStep=1, decimals=1)
        self.sigma.setToolTip("rendering sigma along x for mode 'gauss', in the "
                              "units of that axis; 0 bins instead of blurring")
        self.sigma_y = _Bound()
        self.sigma_y.setToolTip("rendering sigma along y, in the units of that "
                                "axis.  Empty is the same as x where the two "
                                "axes are the same quantity, and 0 -- plain "
                                "binning -- where they are not")
        self.gamma = QDoubleSpinBox(minimum=0.1, maximum=3, singleStep=0.1, decimals=2)
        self.factor = QDoubleSpinBox(minimum=0.05, maximum=5, singleStep=0.1, decimals=2)
        self.factor.setToolTip("rendering sigma = factor x localization precision "
                               "(mode 'precision')")
        self.white = QCheckBox("white background")
        self.white.setToolTip("the picture on white paper, whatever the LUT: the "
                              "brightness is turned over and the hue is kept, so "
                              "red stays red, hot runs white through red and yellow "
                              "to black, and grey is black on white.  A property of "
                              "the picture, so it is set on every layer at once.")
        more_form.addRow("sigma x (gauss)", self.sigma)
        more_form.addRow("sigma y (gauss)", self.sigma_y)
        more_form.addRow("precision factor", self.factor)
        more_form.addRow("gamma", self.gamma)
        more_form.addRow("", self.white)
        form.addRow(CollapsibleSection("more", more, expanded=False))
        layout.addWidget(CollapsibleSection("display", display, expanded=True))
        # last and closed: the ordinary picture is x against y, and this is
        # what turns the renderer into SMAP's versatile one
        self.axes = AxesSection(session, view)
        self.axes.changed.connect(lambda: self._bind_layer(self.strip.current))
        layout.addWidget(CollapsibleSection("axes", self.axes, expanded=False))
        layout.addStretch(1)

        self.strip.selected.connect(self._bind_layer)
        self.strip.add_image_requested.connect(self._add_image)
        self.filter.changed.connect(lambda: session.changed("layer"))
        self.mode.currentTextChanged.connect(self._on_render_settings)
        self.sigma.valueChanged.connect(self._on_render_settings)
        self.sigma_y.editingFinished.connect(self._on_render_settings)
        self.factor.valueChanged.connect(self._on_render_settings)
        self.color.currentIndexChanged.connect(self._on_color)
        self.color_field.currentIndexChanged.connect(self._on_color)
        self.lut.currentTextChanged.connect(self._on_display)
        self.invert.toggled.connect(self._on_display)
        self.invert_mode.currentIndexChanged.connect(self._on_display)
        self.white.toggled.connect(self._on_white)
        self.contrast.valueChanged.connect(self._on_display)
        self.gamma.valueChanged.connect(self._on_display)
        self.grouped.toggled.connect(self._on_grouped)
        self._appended = 0
        session.on_change(self._on_session)
        self._bind_layer(0)
        self.axes.bind()

    @property
    def layer(self) -> Layer:
        return self.session.layers[self.strip.current]

    def _on_session(self, what: str) -> None:
        if what == "locs":
            self._bind_layer(self.strip.current)
            self.axes.bind()
        elif what == "layers":
            self.strip.rebuild()
        elif what == "regrouped":
            self._bind_layer(self.strip.current)
        elif what in ("roi", "roi-edited"):
            self.filter._update_count()
        elif what == "append":
            self._appended += 1
            if self._appended % 5 == 1:      # the histogram need not follow every block
                self.filter.bind(self.layer)
                self._fill_color_fields(self.layer)

    def _fill_color_fields(self, layer: Layer) -> None:
        """The columns this layer could be coloured by.

        Called again as a live fit runs: a fit starts with an empty table and
        the columns -- z_nm among them -- exist only once the first block is
        in, so a list filled once at the start stayed empty for the whole run.
        Rebuilt only when the columns actually change, so the choice the user
        made survives every block after that.
        """
        locs = layer.locs
        numeric = [n for n in locs if np.asarray(locs[n]).dtype.kind in "iuf"
                   and np.asarray(locs[n]).ndim == 1]
        if numeric == [self.color_field.itemText(i)
                       for i in range(self.color_field.count())]:
            return
        settings = layer.state.settings
        chosen = settings.color_field or ("z_nm" if "z_nm" in locs
                                          else (numeric[0] if numeric else ""))
        blocked = self.color_field.signalsBlocked()
        self.color_field.blockSignals(True)
        self.color_field.clear()
        self.color_field.addItems(numeric)
        self.color_field.setCurrentText(chosen)
        self.color_field.blockSignals(blocked)

    def _bind_layer(self, index: int) -> None:
        """Point every control at one layer, without firing their signals."""
        layer = self.session.layers[index]
        self.filter.layer_index = index
        widgets = (self.mode, self.sigma, self.sigma_y, self.factor, self.color, self.color_field,
                   self.lut, self.invert, self.invert_mode, self.white,
                   self.contrast, self.gamma,
                   self.grouped,
                   self.image_pixelsize, self.image_x0, self.image_y0, self.image_frame)
        for w in widgets:
            w.blockSignals(True)
        self.filter_section.setVisible(not layer.is_image)
        self.image_section.setVisible(layer.is_image)
        for w in (self.mode, self.color, self.color_field, self.grouped, self.sigma, self.factor):
            w.setEnabled(not layer.is_image)
        if layer.is_image:
            img = layer.image
            self.image_name.setText(img.name)
            self.image_pixelsize.setValue(img.pixelsize)
            self.image_x0.setValue(img.x0)
            self.image_y0.setValue(img.y0)
            self.image_frame.setRange(0, img.n_frames - 1)
            self.image_frame.setValue(img.frame)
            self.image_frame.setEnabled(img.n_frames > 1)
            display = layer.get_display()
            self.lut.setCurrentText(display.lut if isinstance(display.lut, str) else "gray")
            self._show_invert(display.invert)
            self.white.setChecked(display.white_background)
            self.contrast.setValue(display.contrast)
            self.gamma.setValue(display.gamma)
            for w in widgets:
                w.blockSignals(False)
            return
        self.filter.bind(layer)
        locs = layer.locs
        self._fill_color_fields(layer)
        settings, display = layer.state.settings, layer.state.display
        self.color.setCurrentIndex(1 if settings.color_field else 0)
        self.color_field.setEnabled(settings.color_field is not None)
        self.color_range_row.setEnabled(settings.color_field is not None)
        lo, hi = settings.color_range or (None, None)
        self.color_lo.set(lo)
        self.color_hi.set(hi)
        self.mode.setCurrentText(settings.mode)
        self.sigma.setValue(settings.sigma)
        self.sigma_y.set(settings.sigma_y)
        self.factor.setValue(settings.sigma_settings.factor)
        self.lut.setCurrentText(display.lut if isinstance(display.lut, str) else "hot")
        self._show_invert(display.invert)
        self.white.setChecked(display.white_background)
        self.contrast.setValue(display.contrast)
        self.gamma.setValue(display.gamma)
        self.grouped.setChecked(layer.grouped)
        for w in widgets:
            w.blockSignals(False)

    def _on_render_settings(self) -> None:
        state = self.layer.state
        sigmas = dataclasses.replace(state.settings.sigma_settings, factor=self.factor.value())
        try:
            sigma_y = self.sigma_y.value()
        except ValueError:                       # half-typed; the old one stands
            sigma_y = state.settings.sigma_y
        state.settings = dataclasses.replace(state.settings, mode=self.mode.currentText(),
                                             sigma=self.sigma.value(), sigma_y=sigma_y,
                                             sigma_settings=sigmas)
        self.session.changed("layer")

    def _on_color(self) -> None:
        state = self.layer.state
        by_field = self.color.currentData() == "field"
        field = self.color_field.currentText() if by_field else None
        self.color_field.setEnabled(by_field)
        was = state.settings.color_field
        # a new field starts on its 99% range: one scale for the 2D and 3D views
        color_range = (self._field_range(field) if field and field != was
                       else state.settings.color_range)
        state.settings = dataclasses.replace(state.settings, color_field=field or None,
                                             color_range=color_range if field else None)
        self.color_range_row.setEnabled(field is not None)
        lo, hi = color_range or (None, None)
        self.color_lo.set(lo)
        self.color_hi.set(hi)
        if (was is None) != (field is None):            # switching kind: a fitting LUT
            self.lut.setCurrentText(INTENSITY_LUT if field is None else FIELD_LUT)
        self.session.changed("layer")

    def _field_range(self, field: str):
        locs = self.layer.locs
        if field not in locs:
            return None
        values = np.asarray(locs[field], np.float64)
        finite = values[np.isfinite(values)]
        if not finite.size:
            return None
        return (float(np.quantile(finite, 0.005)), float(np.quantile(finite, 0.995)))

    def _on_color_range(self) -> None:
        state = self.layer.state
        try:
            lo, hi = self.color_lo.value(), self.color_hi.value()
        except ValueError:
            return
        if lo is None or hi is None or hi <= lo:
            return
        state.settings = dataclasses.replace(state.settings, color_range=(lo, hi))
        self.session.changed("layer")

    def _auto_color_range(self) -> None:
        field = self.layer.state.settings.color_field
        rng = self._field_range(field) if field else None
        if rng is None:
            return
        self.color_lo.set(rng[0])
        self.color_hi.set(rng[1])
        self._on_color_range()

    def _on_white(self, on: bool) -> None:
        """White paper is a property of the picture, not of one layer: every
        layer takes it, so the panel and the image cannot disagree."""
        for layer in self.session.layers:
            layer.set_display(dataclasses.replace(layer.get_display(),
                                                  white_background=on))
        self.session.changed("layer")

    def _show_invert(self, invert) -> None:
        """The tick and which inversion, from the one field that carries both."""
        from .. import lut as luts_module
        self.invert.setChecked(bool(invert))
        self.invert_mode.setCurrentIndex(
            max(0, self.invert_mode.findData(luts_module.inversion_name(invert))))
        self.invert_mode.setEnabled(bool(invert))

    def _on_display(self) -> None:
        layer = self.layer
        on = self.invert.isChecked()
        self.invert_mode.setEnabled(on)
        layer.set_display(dataclasses.replace(layer.get_display(), lut=self.lut.currentText(),
                                              invert=self.invert_mode.currentData() if on
                                              else False,
                                              contrast=self.contrast.value(),
                                              gamma=self.gamma.value()))
        self.session.changed("layer")

    def _on_image(self) -> None:
        img = self.layer.image
        if img is None:
            return
        img.pixelsize = self.image_pixelsize.value()
        img.x0, img.y0 = self.image_x0.value(), self.image_y0.value()
        img.frame = self.image_frame.value()
        self.session.changed("layer")

    def _add_image(self) -> None:
        from .dialogs import PixelSizeDialog
        from PySide6.QtWidgets import QDialog
        start = str(self.session.path.parent) if self.session.path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start,
                                              "Images (*.tif *.tiff *.png)")
        if not path:
            return
        try:
            self.session.open_image(path)
        except ValueError:
            dialog = PixelSizeDialog(path.rsplit("/", 1)[-1], parent=self)
            if dialog.exec() == QDialog.Accepted:
                px, x0, y0 = dialog.values()
                self.session.open_image(path, px, x0, y0)
        self.strip.select(len(self.session.layers) - 1)

    def _on_grouped(self, on: bool) -> None:
        self.session.show_grouped(self.strip.current, on)   # links once per table
        self.filter.bind(self.layer)     # the grouped table has its own filter
        self.session.changed("layer")
