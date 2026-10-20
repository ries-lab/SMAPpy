"""The Render tab: layers, each with a filter and a way of being drawn.

The filter is one field at a time -- a drop-down, quick buttons for the
usual ones, and a histogram whose shaded region is the range -- plus the
numbers, for typing.  Which layer all of that belongs to is chosen in the
layer strip at the top.
"""
from __future__ import annotations

import dataclasses
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QPushButton,
                               QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .. import lut as luts
from ..render import FieldOfView, RenderAxes, axis_unit, is_position
from ..filter import histogram_range
from ..session import Layer, Session
from ..lut import FIELD_LUT, INTENSITY_LUT
from . import folders
from .merging import merger_for
from .render_view import nice_below
from .widgets import CollapsibleSection, ColumnScroll, detach_to_window

# an image layer's stack: what its planes are, what it shows of them, and how
# the 3D view draws a z-stack (`session.Layer.set_image_view`)
IMAGE_AXES = (("z", "z-slices"), ("t", "time frames"), ("", "planes"))
IMAGE_SHOW = (("slice", "slice"), ("max", "max"), ("mean", "mean"), ("sum", "sum"))
IMAGE_3D = (("plane", "plane"), ("max", "max (MIP)"), ("volume", "volume"))

# the fields with a quick button, best-named alternative first.  In words,
# not the columns' abbreviations: the tooltip names the column
QUICK_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("precision", ("xy_err_nm",)),
    ("fit quality", ("logl_rel",)),
    ("frame", ("frame",)),
    ("z", ("z_nm",)),
    ("photons", ("photons",)),
    ("PSF", ("sigma_nm",)),
    ("channel", ("channel",)),
    ("file", ("filenumber",)),
)
# `RenderSettings.mode` -> what the *render* list shows.  The key is what is
# stored (layer settings, saved workspaces, chains) and stays short; the
# words are only the list's, kept in its item data
RENDER_MODES: Tuple[Tuple[str, str], ...] = (
    ("precision", "Gaussian (precision)"),
    ("gauss", "Gaussian (fixed)"),
    ("hist", "histogram"),
    ("dl", "diffraction-limited"),
    ("tracks", "tracks"),
    ("voronoi", "Voronoi density"),
)
QUICK_PER_ROW = 4
QUICK_TEXT_WIDTH = 64       # px: a bounded column's button, its name elided to this
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


def bounds_text(bounds: Dict[str, Tuple[Optional[float], Optional[float]]]) -> str:
    """The bounds as one line of links: ``xy_err_nm <= 25 x``, ``-500 <= z_nm
    <= 500 x``; each name edits its bound, each cross removes it."""
    if not bounds:
        return "<span style='color: gray'>no filter</span>"
    parts = []
    for name, (lo, hi) in bounds.items():
        if lo is not None and hi is not None:
            text = f"{lo:g} \u2264 {name} \u2264 {hi:g}"
        elif hi is not None:
            text = f"{name} \u2264 {hi:g}"
        else:
            text = f"{name} \u2265 {lo:g}"
        parts.append(f"<a href='edit:{name}' style='text-decoration: none'>{text}</a>"
                     f"&nbsp;<a href='remove:{name}' style='text-decoration: none; "
                     f"color: gray'>\u00d7</a>")
    return "&nbsp;&nbsp; ".join(parts)


class FilterWidget(QWidget):
    """One field of one layer's filter: histogram, range, numbers."""

    changed = Signal()

    def __init__(self, session: Optional[Session] = None, parent=None):
        super().__init__(parent)
        self.session = session
        self.layer: Optional[Layer] = None
        # the picture's field of view, ``() -> QRectF``, for the count in it;
        # set by the Render tab when there is a picture
        self.view_rect = None
        self.layer_index = 0
        self._hist_cache: Dict[Tuple[int, str], tuple] = {}
        self._shown_region = (None, None)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # The quick buttons have rows to themselves and the drop-down and the
        # count share the next.  All on one line, a table with all eight quick
        # fields made the panel ~440 px wide in a 380 px window, and the scroll
        # area (no horizontal bar) cut off the histogram's right end -- with
        # the handle of the upper bound on it.  Named in words they need two
        # rows of `QUICK_PER_ROW`.
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.setColumnStretch(QUICK_PER_ROW, 1)
        self.quick: Dict[str, QToolButton] = {}
        self.quick_grid = grid
        layout.addLayout(grid)
        # Every bound of the layer, whichever field is on show: a picture
        # that comes out sparse should say why without a hunt through the
        # drop-down.  A bound is a link to its field; the cross removes it.
        self.bounds_line = QLabel("")
        self.bounds_line.setWordWrap(True)
        self.bounds_line.setTextFormat(Qt.RichText)
        self.bounds_line.setToolTip("the bounds on this layer: click one to edit "
                                    "it, x to remove it")
        self.bounds_line.linkActivated.connect(self._on_bound_link)
        layout.addWidget(self.bounds_line)
        field_row = QHBoxLayout()
        self.field = QComboBox()
        self.field.setMinimumWidth(110)
        field_row.addWidget(self.field)
        self.count = QLabel("")
        field_row.addWidget(self.count, 1, Qt.AlignRight)
        # the region filter on this layer or off (`region_filter`): on by
        # default, so seeing everything is a click away without losing the
        # conditions; shown only while there are some
        self.regions = QToolButton(text="regions", checkable=True)
        self.regions.setToolTip("this layer shows only the regions the Regions "
                                "tab's region filter keeps; untick to see all")
        self.regions.toggled.connect(self._on_regions)
        self.regions.hide()
        field_row.addWidget(self.regions)
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

        for b in self.quick.values():
            b.deleteLater()
        self.quick.clear()
        self._fixed_quick = []
        for label, names in QUICK_FIELDS:
            name = next((n for n in names if n in locs), None)
            if name is not None:
                self._fixed_quick.append(name)
                self._add_quick(name, label)
        self._mark_bounded()          # and a button for each other bounded field
        first = current if current in numeric else \
            next((n for n in self._fixed_quick), None)
        if first is None and numeric:
            first = numeric[0]
        if first:
            self._select(first)
        self._show_field()

    def _add_quick(self, name: str, label: str) -> QToolButton:
        b = QToolButton(text=label, checkable=True, autoExclusive=True)
        b.setToolTip(name)
        b.clicked.connect(lambda _=False, n=name: self._select(n))
        self.quick_grid.addWidget(b, *divmod(len(self.quick), QUICK_PER_ROW))
        self.quick[name] = b
        return b

    def _sync_bounded_buttons(self, bounded) -> None:
        """A button for every bounded field beyond the usual ones, gone when
        its bound is: what filters the layer can be seen, and reached, at a
        glance.  Named by the column, shortened to fit the row."""
        from ..region_runs import REGION_FILTER     # has a button of its own
        fixed = getattr(self, "_fixed_quick", [])
        names = [self.field.itemData(i) for i in range(self.field.count())]
        wanted = [n for n in names if n in bounded and n not in fixed
                  and n != REGION_FILTER]
        extra = [n for n in self.quick if n not in fixed]
        if wanted == extra:
            return
        for n in extra:
            b = self.quick.pop(n)
            self.quick_grid.removeWidget(b)
            b.deleteLater()
        shown = self.current_field()
        for n in wanted:
            b = self._add_quick(n, "")
            b.setText(b.fontMetrics().elidedText(n, Qt.ElideMiddle, QUICK_TEXT_WIDTH))
            b.setStyleSheet("font-weight: bold")
            b.setChecked(n == shown)

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
        ranges = self.layer.ranges
        bounded = {n for n, r in ranges.items() if r != (None, None)}
        self._sync_bounded_buttons(bounded)
        self.bounds_line.setText(bounds_text(
            {n: ranges[n] for n in sorted(bounded)}))
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
            counts, edges = np.histogram(values, bins=HIST_BINS,
                                         range=histogram_range(values))  # the bulk
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
        lo, hi = self.layer.ranges.get(name, (None, None))
        self.lo.set(lo)
        self.hi.set(hi)
        # the region stands in for an open bound at the histogram's edge
        self.region.blockSignals(True)
        self._shown_region = (edges[0] if lo is None else lo,
                              edges[-1] if hi is None else hi)
        self.region.setRegion(self._shown_region)
        self._shown_region = tuple(self.region.getRegion())   # as the item keeps it
        self.region.blockSignals(False)
        self.plot.setXRange(edges[0], edges[-1], padding=0.02)
        self._update_count()

    def _show_files(self) -> None:
        """One tickable row per file, plus 'all'."""
        chosen = self.layer.files
        present = {int(n) for n in np.unique(self.layer.locs["filenumber"])}  # float once grouped
        self.files.blockSignals(True)
        self.files.clear()
        names = self.session.file_names()
        for number in sorted(present):
            name = names[number] if number < len(names) else f"file {number}"
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, number)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            on = chosen is None or number in chosen
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
        elif self.session is not None and self.view_rect is not None and len(f):
            # zoomed in, the picture shows a part of what the filter keeps
            r = self.view_rect()
            n = self.session.count_in_view(self.layer_index, r.left(), r.right(),
                                           r.top(), r.bottom())
            if n < len(f):
                text += f", {n} in view"
        self.count.setText(text)
        self._show_regions()
        bounded = set(f.ranges)
        for n, b in self.quick.items():
            b.setStyleSheet("font-weight: bold" if n in bounded and
                            f.ranges[n] != (None, None) else "")
        self._mark_bounded()

    def _show_regions(self) -> None:
        from ..region_runs import REGION_FILTER
        filtered = (self.session is not None and self.layer is not None
                    and bool(self.session.region_filter))
        self.regions.setVisible(filtered)
        if filtered:
            self.regions.blockSignals(True)
            self.regions.setChecked(REGION_FILTER in self.layer.ranges)
            self.regions.blockSignals(False)

    def _on_regions(self, on: bool) -> None:
        from ..region_runs import REGION_FILTER
        if self.layer is None:
            return
        if on:
            self.layer.set_bound(REGION_FILTER, 0.5, None)
        else:
            self.layer.remove_bound(REGION_FILTER)
        self._show_field()
        self.changed.emit()

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
        name = self.current_field()
        _, edges = self._histogram(name)
        # As SMAP: an end dragged to the edge means "no bound", so the tail is
        # not cut off; anywhere else it is the number.  Only the end that was
        # dragged: the other keeps its bound, which may lie beyond the
        # histogram (the 25 nm precision bound over a sharp simulation).
        old_lo, old_hi = self.layer.ranges.get(name, (None, None))
        shown_lo, shown_hi = self._shown_region
        if lo != shown_lo:
            old_lo = None if lo <= edges[0] else float(lo)
        if hi != shown_hi:
            old_hi = None if hi >= edges[-1] else float(hi)
        self._apply(old_lo, old_hi)

    def _on_numbers(self) -> None:
        try:
            self._apply(self.lo.value(), self.hi.value())
        except ValueError:
            self._show_field()

    def _on_clear(self) -> None:
        self._apply(None, None)

    def _on_bound_link(self, link: str) -> None:
        action, _, name = link.partition(":")
        if action == "remove":
            self.layer.remove_bound(name)
            self._show_field()
            self.changed.emit()
        else:
            self._select(name)

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
        self.visible = QCheckBox("visible")
        self.visible.setToolTip("draw this layer")
        self.name_label = QLabel("name")
        self.name = QLineEdit()
        self.name.setToolTip("layer name")
        self.name.setMaximumWidth(110)
        self.name.editingFinished.connect(self._on_name)
        self.add = QToolButton(text="+", popupMode=QToolButton.InstantPopup)
        self.add.setToolTip("add a layer")
        menu = QMenu(self.add)
        menu.addAction("localizations", self._add)
        # emitted from a function, not handed the signal: PySide6 takes the
        # second argument as a slot and *calls* it, a signal instance is not
        # callable, and the TypeError is printed and swallowed -- the menu
        # entry did nothing at all
        menu.addAction("image layer", lambda: self.add_image_requested.emit())
        self.add.setMenu(menu)
        self.remove = QToolButton(text="-")
        self.remove.setToolTip("remove this layer")
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
            if item.widget() and item.widget() not in (self.visible, self.name_label,
                                                       self.name, self.add, self.remove,
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
        self.layout_.addWidget(self.name_label)
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
            # with the overview as context, so Qt drops the call if it is
            # deleted first
            QTimer.singleShot(0, self, self._update_deferred)

    def _update_deferred(self) -> None:
        """`update_image`, for a timer: what it raises is printed, not raised.

        An exception from a singleShot callable is left pending inside
        `processEvents` by PySide, and the next Python override Qt calls --
        a pyqtgraph item's -- segfaults (6.10) or hangs (see gui/canvas.py).
        A table with no precision column raised here when it was opened.
        The main view prints and moves on too (`RenderWorker.render`).
        """
        try:
            self.update_image()
        except Exception:
            traceback.print_exc()
            self.image.clear()

    def update_image(self) -> None:
        if self.view is None or not (len(self.session.locs)
                                     or any(l.is_image for l in self.session.layers)):
            self.image.clear()
            return
        # the same way up as the main view: a plot of two columns is a plot
        self.box.invertY(self.session.axes().is_default)
        (x0, x1), (y0, y1) = self.session.full_view()
        fov = FieldOfView.fit((x0, x1), (y0, y1), 300, 200)
        rgb = self.view.panel_images(fov, ["overlay"])[0]
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
            row.addWidget(QLabel("scale"))
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
        self.note.setText(f"{x_name} across, {y_name} up.  ROIs and anything "
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


PAGE = "Panels/Render tab"
# the tab's own sections: titled as a plugin's are, larger than the folds inside
HELP = dict(helpable=True, major=True,
            help_tip="what the Render tab's controls do (F1)")


class RenderTab(QWidget):
    def __init__(self, session: Session, view=None, parent=None):
        super().__init__(parent)
        self.session = session
        # merges a large table's blinks in the background (`gui.merging`):
        # the control window's, or this tab's own when it stands alone
        self.merger = merger_for(session, self)
        # the layers (their states) already coloured by track (`_colour_by_track`)
        self._tracks_coloured: set = set()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        layout = QVBoxLayout(inner)              # the tab's content; scrolls when short
        outer.addWidget(ColumnScroll(inner))
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # the whole picture, every visible layer, above the layer strip
        self.overview = Overview(session, view)
        section = CollapsibleSection("overview", self.overview, expanded=True,
                                     detachable=True, **HELP)
        section.detach_requested.connect(lambda: detach_to_window(section, self.window()))
        layout.addWidget(section)
        self.strip = LayerStrip(session)
        layout.addWidget(self.strip)
        self.filter = FilterWidget(session)
        if view is not None:
            self.filter.view_rect = view.view.viewRect
            self._count_timer = QTimer(self, singleShot=True, interval=150)
            self._count_timer.timeout.connect(self._count_in_view)
            view.view.sigRangeChanged.connect(lambda *_: self._count_timer.start())
        self.filter_section = CollapsibleSection("filter", self.filter, expanded=True, **HELP)
        layout.addWidget(self.filter_section)

        image = QWidget()
        image_form = QFormLayout(image)
        image_form.setContentsMargins(0, 0, 0, 0)
        image_form.setVerticalSpacing(2)
        # what the layer shows: the frames each open file kept, any image
        # opened, or a new one -- one list, so that the camera's view and a
        # widefield TIFF are chosen the same way
        self.image_source = QComboBox()
        self.image_source.setToolTip("the frames kept with a file, or an image")
        self.image_pixelsize = QDoubleSpinBox(minimum=0.01, maximum=1e6, decimals=2)
        self.image_x0 = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=1)
        self.image_y0 = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=1)
        self.image_frame = QSlider(Qt.Horizontal)
        self.image_frame_label = QLabel("")
        self.image_frame_label.setToolTip("which frame of the stack is shown")
        frame_row = QHBoxLayout()
        frame_row.setContentsMargins(0, 0, 0, 0)
        frame_row.addWidget(self.image_frame, 1)
        frame_row.addWidget(self.image_frame_label)
        # what the planes of a stack are, and how much apart: a file that did
        # not say is asked when it is opened (`gui.image_open`), and this is
        # where a wrong answer, or a wrong step, is put right
        self.image_axis = QComboBox()
        for key, text in IMAGE_AXES:
            self.image_axis.addItem(text, key)
        self.image_axis.setToolTip("what the planes of the stack are")
        self.image_step = QDoubleSpinBox(minimum=0.0, maximum=1e9, decimals=3)
        self.image_step.setToolTip("the z step between planes (nm), or the time "
                                   "between frames (s)")
        axis_row = QHBoxLayout()
        axis_row.setContentsMargins(0, 0, 0, 0)
        axis_row.addWidget(self.image_axis)
        axis_row.addWidget(self.image_step, 1)
        # one plane, or a projection over a range of them
        self.image_show = QComboBox()
        for key, text in IMAGE_SHOW:
            self.image_show.addItem(text, key)
        self.image_show.setToolTip("one plane, or the planes from-to taken together")
        self.image_from = QSpinBox(minimum=1)
        self.image_to = QSpinBox(minimum=1)
        self.image_from.setToolTip("the first plane of the projection")
        self.image_to.setToolTip("the last plane of the projection")
        self.image_follow = QCheckBox("follow z filter")
        self.image_follow.setToolTip("show the plane at the middle of the localization "
                                     "layers' z filter")
        show_row = QHBoxLayout()
        show_row.setContentsMargins(0, 0, 0, 0)
        show_row.addWidget(self.image_show)
        show_row.addWidget(self.image_from)
        show_row.addWidget(self.image_to)
        show_row.addWidget(self.image_follow, 1)
        self.image_3d = QComboBox()
        for key, text in IMAGE_3D:
            self.image_3d.addItem(text, key)
        self.image_3d.setToolTip("how the 3D view draws a z-stack: the plane at the "
                                 "centre of rotation, the brightest value along each "
                                 "line of sight, or the volume added up")
        image_form.addRow("source", self.image_source)
        image_form.addRow("pixel size (nm)", self.image_pixelsize)
        image_form.addRow("x0 (nm)", self.image_x0)
        image_form.addRow("y0 (nm)", self.image_y0)
        image_form.addRow("planes are", axis_row)
        image_form.addRow("show", show_row)
        image_form.addRow("frame", frame_row)
        image_form.addRow("in 3D", self.image_3d)
        self.image_section = CollapsibleSection("image", image, expanded=True, **HELP)
        self.image_section.hide()
        layout.addWidget(self.image_section)
        for w in (self.image_pixelsize, self.image_x0, self.image_y0, self.image_step):
            w.valueChanged.connect(self._on_image)
        self.image_frame.valueChanged.connect(self._on_image)
        for w in (self.image_from, self.image_to):
            w.valueChanged.connect(self._on_image)
        for w in (self.image_axis, self.image_show, self.image_3d):
            w.currentIndexChanged.connect(self._on_image)
        self.image_follow.toggled.connect(self._on_image)
        self.image_source.activated.connect(self._on_image_source)

        display = QWidget()
        form = QFormLayout(display)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(2)
        self.mode = QComboBox()
        for key, text in RENDER_MODES:
            self.mode.addItem(text, key)
        self.mode.setToolTip("how a localization is drawn: a Gaussian as wide as "
                             "its precision, one of a fixed width, a count per "
                             "pixel, the spot the camera saw, the lines of "
                             "its track, or the density of its Voronoi cell")
        # the tracks mode's two settings, shown only while it is chosen
        self.track_column = QComboBox()
        self.track_column.setToolTip("the id column whose localizations are one "
                                     "track; auto: track_id, else tid (MINFLUX), "
                                     "else track_id_true (a simulation)")
        self.line_width = QDoubleSpinBox(minimum=0.2, maximum=20, singleStep=0.5,
                                         decimals=1)
        self.line_width.setToolTip("the tracks' lines, in pixels of the picture")
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
        self.contrast.setToolTip("saturate 10^-p of the pixels: 2 is bright, 4 dark")
        # SMAP's toggle: the quantile, or a set maximum -- here a density, so
        # that the picture keeps its brightness as one zooms (`absolute_imax`)
        self.scale = QComboBox()
        self.scale.addItem("quantile", False)
        self.scale.addItem("absolute", True)
        self.scale.setToolTip(
            "quantile: full scale where 10^-p of the pixels saturate, whatever "
            "the data.\nabsolute: full scale at a set density, the same at every "
            "zoom; switching to it starts from what the view shows now")
        self.maximum = QDoubleSpinBox(minimum=1e-6, maximum=1e12, decimals=3)
        self.maximum.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        self.maximum.setToolTip("full scale, in localizations per µm²; for tracks, "
                                "lines over one pixel; for an image, its own values")
        contrast_row = QHBoxLayout()
        contrast_row.setContentsMargins(0, 0, 0, 0)
        contrast_row.addWidget(self.scale)
        contrast_row.addWidget(self.contrast, 1)
        contrast_row.addWidget(self.maximum, 1)
        self.grouped = QCheckBox("merge blinks")
        self.grouped.setToolTip("one entry per blink instead of one per frame; "
                                "merges the table on first use")
        form.addRow("render", self.mode)
        form.addRow("track id", self.track_column)
        form.addRow("line width", self.line_width)
        self._display_form = form
        form.addRow("colour by", color_row)
        form.addRow("colour range", self.color_range_row)
        form.addRow("LUT", lut_row)
        form.addRow("contrast", contrast_row)
        # the linking settings sit beside the switch that uses them; they are
        # the session's, not the layer's, so OK regroups every grouped layer
        self.grouping_button = QPushButton("merge settings...")
        self.grouping_button.setToolTip("how the localizations of one blink are "
                                        "merged: distance and dark frames, for "
                                        "every layer")
        self.grouping_button.clicked.connect(self._grouping)
        grouped_row = QHBoxLayout()
        grouped_row.setContentsMargins(0, 0, 0, 0)
        grouped_row.addWidget(self.grouped)
        grouped_row.addWidget(self.grouping_button)
        grouped_row.addStretch(1)
        form.addRow("", grouped_row)
        more = QWidget()
        more_form = QFormLayout(more)
        more_form.setContentsMargins(0, 0, 0, 0)
        more_form.setVerticalSpacing(2)
        self.sigma = QDoubleSpinBox(minimum=0.0, maximum=1000, singleStep=1, decimals=1)
        self.sigma.setToolTip("rendering sigma along x for Gaussian (fixed), in the "
                              "units of that axis; 0 bins instead of blurring")
        self.sigma_y = _Bound()
        self.sigma_y.setToolTip("rendering sigma along y, in the units of that "
                                "axis.  Empty is the same as x where the two "
                                "axes are the same quantity, and 0 -- plain "
                                "binning -- where they are not")
        self.gamma = QDoubleSpinBox(minimum=0.1, maximum=3, singleStep=0.1, decimals=2)
        self.factor = QDoubleSpinBox(minimum=0.05, maximum=5, singleStep=0.1, decimals=2)
        self.factor.setToolTip("rendering sigma = factor x localization precision "
                               "(Gaussian (precision))")
        self.white = QCheckBox("white background")
        self.white.setToolTip("the picture on white paper, whatever the LUT: the "
                              "brightness is turned over and the hue is kept, so "
                              "red stays red, hot runs white through red and yellow "
                              "to black, and grey is black on white.  A property of "
                              "the picture, so it is set on every layer at once.")
        more_form.addRow("sigma x (fixed)", self.sigma)
        more_form.addRow("sigma y (fixed)", self.sigma_y)
        more_form.addRow("precision factor", self.factor)
        more_form.addRow("gamma", self.gamma)
        more_form.addRow("", self.white)
        self.lines_add = QCheckBox("lines add up")
        self.lines_add.setToolTip(
            "tracks: where lines overlap -- a track's joints, crossings -- they "
            "add, as localizations do.  Off: a covered pixel is one line, so "
            "the knots do not outshine the lines and set the contrast")
        more_form.addRow("", self.lines_add)
        self._more_form = more_form
        form.addRow(CollapsibleSection("more", more, expanded=False))
        layout.addWidget(CollapsibleSection("display", display, expanded=True, **HELP))
        # last and closed: the ordinary picture is x against y, and this is
        # what turns the renderer into SMAP's versatile one
        self.axes = AxesSection(session, view)
        self.axes.changed.connect(lambda: self._bind_layer(self.strip.current))
        layout.addWidget(CollapsibleSection("axes", self.axes, expanded=False, **HELP))
        layout.addStretch(1)
        # every section's ? and F1 open the tab's page
        from .help_window import attach_help, help_section
        for child in self.findChildren(CollapsibleSection):
            help_section(child, PAGE)
        attach_help(self, PAGE, within=True)

        self.strip.selected.connect(self._bind_layer)
        self.strip.add_image_requested.connect(self._add_image)
        self.filter.changed.connect(lambda: session.changed("layer"))
        self.mode.currentIndexChanged.connect(self._on_render_settings)
        self.track_column.currentIndexChanged.connect(self._on_render_settings)
        self.line_width.valueChanged.connect(self._on_render_settings)
        self.lines_add.toggled.connect(self._on_render_settings)
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
        self.maximum.valueChanged.connect(self._on_display)
        self.scale.currentIndexChanged.connect(self._on_scale)
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
        elif what == "images" and self.layer.is_image:
            self._fill_image_sources(self.layer)
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
        widgets = (self.mode, self.track_column, self.line_width, self.lines_add,
                   self.sigma, self.sigma_y, self.factor, self.color, self.color_field,
                   self.lut, self.invert, self.invert_mode, self.white,
                   self.contrast, self.scale, self.maximum, self.gamma,
                   self.grouped,
                   self.image_source, self.image_pixelsize, self.image_x0,
                   self.image_y0, self.image_frame, self.image_axis, self.image_step,
                   self.image_show, self.image_from, self.image_to, self.image_follow,
                   self.image_3d)
        for w in widgets:
            w.blockSignals(True)
        self.filter_section.setVisible(not layer.is_image)
        self.image_section.setVisible(layer.is_image)
        for w in (self.mode, self.color, self.color_field, self.grouped, self.sigma, self.factor):
            w.setEnabled(not layer.is_image)
        if layer.is_image:
            self._show_track_rows(False)
            img = layer.image
            self._fill_image_sources(layer)
            self.image_pixelsize.setValue(img.pixelsize)
            self.image_x0.setValue(img.x0)
            self.image_y0.setValue(img.y0)
            self.image_frame.setRange(0, img.n_frames - 1)
            self.image_frame.setValue(layer.frame)
            self.image_frame_label.setText(img.label(layer.frame))
            self._set_combo(self.image_axis, img.axis)
            self.image_step.setValue((img.z_step_nm if img.axis == "z" else
                                      img.frame_time_s if img.axis == "t" else 0.0) or 0.0)
            self._set_combo(self.image_show, getattr(layer, "image_show", "slice"))
            planes = getattr(layer, "image_planes", None) or (0, img.n_frames - 1)
            for w, value in ((self.image_from, planes[0]), (self.image_to, planes[1])):
                w.setMaximum(img.n_frames)
                w.setValue(int(value) + 1)
            self.image_follow.setChecked(bool(getattr(layer, "follow_z", False)))
            self._set_combo(self.image_3d, getattr(layer, "image_3d", "plane"))
            self._enable_image_rows(img)
            display = layer.get_display()
            self.lut.setCurrentText(display.lut if isinstance(display.lut, str) else "gray")
            self._show_invert(display.invert)
            self.white.setChecked(display.white_background)
            self._show_scale(display)
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
        self.set_mode(settings.mode)
        self._fill_track_columns(layer)
        self.line_width.setValue(settings.line_width)
        self.lines_add.setChecked(settings.lines_add)
        self._show_track_rows(settings.mode == "tracks" and not layer.is_image)
        self.sigma.setValue(settings.sigma)
        self.sigma_y.set(settings.sigma_y)
        self.factor.setValue(settings.sigma_settings.factor)
        self.lut.setCurrentText(display.lut if isinstance(display.lut, str) else "hot")
        self._show_invert(display.invert)
        self.white.setChecked(display.white_background)
        self._show_scale(display)
        self.gamma.setValue(display.gamma)
        self.grouped.setChecked(layer.grouped)
        for w in widgets:
            w.blockSignals(False)

    def set_mode(self, key: str) -> None:
        """Choose a render mode by its stored key ("precision", "gauss", ...)."""
        index = self.mode.findData(key)
        if index < 0:
            raise ValueError(f"render mode {key!r}: one of "
                             f"{', '.join(k for k, _ in RENDER_MODES)}")
        self.mode.setCurrentIndex(index)

    def _on_render_settings(self) -> None:
        state = self.layer.state
        sigmas = dataclasses.replace(state.settings.sigma_settings, factor=self.factor.value())
        try:
            sigma_y = self.sigma_y.value()
        except ValueError:                       # half-typed; the old one stands
            sigma_y = state.settings.sigma_y
        mode = self.mode.currentData()
        entering_tracks = mode == "tracks" and state.settings.mode != "tracks"
        state.settings = dataclasses.replace(
            state.settings, mode=mode, sigma=self.sigma.value(), sigma_y=sigma_y,
            sigma_settings=sigmas, line_width=self.line_width.value(),
            lines_add=self.lines_add.isChecked(),
            track_column=self._chosen_track_column(state.settings.track_column))
        self._show_track_rows(mode == "tracks")
        if entering_tracks:
            self._colour_by_track()
        if entering_tracks and self.layer.grouped:
            # a track is drawn through its localizations: merged blinks join a
            # slow particle's steps into one point and draw a coarser line.
            # Only on choosing the mode, so ticking it again afterwards stays
            self.grouped.blockSignals(True)
            self.grouped.setChecked(False)
            self.grouped.blockSignals(False)
            self._on_grouped(False)
            return                               # that announced the change
        self.session.changed("layer")

    def _show_track_rows(self, shown: bool) -> None:
        for widget in (self.track_column, self.line_width):
            self._display_form.setRowVisible(widget, shown)
        self._more_form.setRowVisible(self.lines_add, shown)

    def _colour_by_track(self) -> None:
        """The first time a layer draws tracks, and only if it is not coloured
        by a column already: each track its own colour, by its id on *turbo*.
        One intensity LUT made every track the same colour, and which line
        was which track could not be told where they cross.  Once per layer,
        so a colouring chosen afterwards is not undone by a mode switch."""
        state = self.layer.state
        if state.settings.color_field is not None or id(state) in self._tracks_coloured:
            return
        self._tracks_coloured.add(id(state))
        from ..tracking.columns import track_column
        try:
            column = track_column(self.layer.locs, state.settings.track_column)
        except (KeyError, ValueError):
            return                      # no tracks yet: the error says so
        self._fill_color_fields(self.layer)
        widgets = (self.color, self.color_field, self.lut)
        for w in widgets:
            w.blockSignals(True)
        self.color.setCurrentIndex(self.color.findData("field"))
        self.color_field.setCurrentText(column)
        for w in widgets:
            w.blockSignals(False)
        # through the handlers, so the range and the LUT follow as on a click
        self._on_color()
        self.lut.setCurrentText("turbo")

    def _chosen_track_column(self, fallback: str) -> str:
        """The column the list says: "" for *auto*; ``fallback`` while the
        list is empty (being refilled)."""
        if self.track_column.count() == 0:
            return fallback
        return self.track_column.currentData() or ""

    def _fill_track_columns(self, layer: Layer) -> None:
        """*auto* -- saying what it resolves to -- then the integer columns a
        track could be named by; a chosen column the table lacks stays in the
        list, so the error says what is wrong rather than the choice
        silently changing (`tracking.columns`)."""
        from ..tracking.columns import track_column_choices
        locs = layer.locs
        choices = track_column_choices(locs)
        chosen = layer.state.settings.track_column or ""
        if chosen and chosen not in [key for key, _ in choices]:
            choices.append((chosen, chosen))
        blocked = self.track_column.signalsBlocked()
        self.track_column.blockSignals(True)
        self.track_column.clear()
        for key, text in choices:
            self.track_column.addItem(text, key)
        self.track_column.setCurrentIndex(max(0, self.track_column.findData(chosen)))
        self.track_column.blockSignals(blocked)

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
                                              absolute=bool(self.scale.currentData()),
                                              maximum=self.maximum.value(),
                                              gamma=self.gamma.value()))
        self.session.changed("layer")

    def _show_scale(self, display) -> None:
        """The quantile or the absolute maximum, whichever the layer uses."""
        self.contrast.setValue(display.contrast)
        self.maximum.setValue(display.maximum)
        self.scale.setCurrentIndex(1 if display.absolute else 0)
        self.contrast.setVisible(not display.absolute)
        self.maximum.setVisible(display.absolute)

    def _on_scale(self) -> None:
        """Switch between the quantile and an absolute maximum.  As SMAP's
        toggle, the absolute one starts at the brightness the view has now,
        so the picture does not jump: the layer is rendered once on the view."""
        absolute = bool(self.scale.currentData())
        if absolute:
            found = self._current_maximum()
            if found is not None:
                self.maximum.blockSignals(True)
                self.maximum.setValue(found)
                self.maximum.blockSignals(False)
        self.contrast.setVisible(not absolute)
        self.maximum.setVisible(absolute)
        self._on_display()

    def _current_maximum(self) -> Optional[float]:
        """The absolute maximum that shows the layer as the view does now."""
        source = self.session.view_frames.get("2D")
        frame = source() if source is not None else None
        if frame is None or not frame.pixelsize > 0:
            return None
        try:
            fov = FieldOfView.from_range(frame.xrange, frame.yrange, frame.pixelsize)
            display = self.layer.get_display()
            _, rendered = self.layer.render(fov)
            return display.as_absolute(rendered).maximum
        except Exception:                # a picture that cannot be made keeps the value
            traceback.print_exc()
            return None

    def _on_image(self) -> None:
        layer = self.layer
        img = layer.image
        if img is None:
            return
        # the pixel size is the x one; a camera with oblong pixels keeps its
        # ratio, which is the camera's and not something set here
        ratio = img.pixelsize_xy[1] / img.pixelsize
        img.pixelsize = self.image_pixelsize.value()
        if img.pixelsize_y is not None:
            img.pixelsize_y = img.pixelsize * ratio
        img.x0, img.y0 = self.image_x0.value(), self.image_y0.value()
        axis = self.image_axis.currentData() or ""
        if img.n_frames > 1 and axis != img.axis:
            img.axis = axis
            # the step of the old meaning is not one of the new
            blocked = self.image_step.blockSignals(True)
            self.image_step.setValue((img.z_step_nm if axis == "z" else
                                      img.frame_time_s if axis == "t" else 0.0) or 0.0)
            self.image_step.blockSignals(blocked)
        elif axis == "z":
            img.z_step_nm = self.image_step.value() or None
        elif axis == "t":
            img.frame_time_s = self.image_step.value() or None
        img.metadata.pop("axes_unknown", None)
        layer.frame = self.image_frame.value()
        layer.set_image_view(self.image_show.currentData() or "slice",
                             (self.image_from.value() - 1, self.image_to.value() - 1),
                             self.image_follow.isChecked(),
                             self.image_3d.currentData() or "plane")
        self.image_frame_label.setText(img.label(layer.frame))
        self._enable_image_rows(img)
        self.session.changed("layer")

    def _enable_image_rows(self, img) -> None:
        """Only what applies to this image, as it is set, can be changed."""
        stack, zstack = img.n_frames > 1, img.is_zstack
        slice_ = (self.image_show.currentData() or "slice") == "slice"
        self.image_axis.setEnabled(stack)
        self.image_step.setEnabled(stack and img.axis in ("z", "t"))
        self.image_step.setSuffix(" nm" if img.axis == "z" else " s" if img.axis == "t" else "")
        self.image_show.setEnabled(stack)
        for w in (self.image_from, self.image_to):
            w.setVisible(stack and not slice_)
        self.image_follow.setVisible(zstack and slice_)
        self.image_frame.setEnabled(stack and slice_ and not
                                    (zstack and self.image_follow.isChecked()))
        self.image_3d.setEnabled(zstack)

    @staticmethod
    def _set_combo(box: QComboBox, key) -> None:
        i = box.findData(key)
        box.setCurrentIndex(max(i, 0))

    def _fill_image_sources(self, layer: Layer) -> None:
        """Every image the session has, the layer's own chosen, and *open
        file...* last."""
        box = self.image_source
        blocked = box.blockSignals(True)
        box.clear()
        for im in self.session.images:
            box.addItem(im.name or "image", im)
        if layer.image.is_empty:
            # the frames a fit keeps arrive when it ends; the layer takes the
            # first image there is by itself (`Session.changed`)
            box.addItem("none yet: a fit's frames, or a file", layer.image)
        elif all(im is not layer.image for im in self.session.images):
            box.addItem(layer.image.name or "image", layer.image)
        box.addItem("open file...", None)
        box.setCurrentIndex(next(i for i in range(box.count())
                                 if box.itemData(i) is layer.image))
        box.blockSignals(blocked)

    def _on_image_source(self, index: int) -> None:
        layer = self.layer
        image = self.image_source.itemData(index)
        if image is None:
            image = self._open_image_file()
            if image is None:                    # cancelled: nothing changes
                self._fill_image_sources(layer)
                return
        # the first plane: for kept frames the average, which is the one to
        # line up with the localizations before stepping through the rest
        layer.show_image(image, 0)
        self.strip.rebuild()                     # the button's tooltip is the name
        self._bind_layer(self.strip.current)
        self.session.changed("layer")

    def _open_image_file(self):
        """Ask for an image file and open it as a source; None if cancelled."""
        from .dialogs import read_image_file
        path, _ = QFileDialog.getOpenFileName(self, "Open image", folders.start(),
                                              "Images (*.tif *.tiff *.png)")
        if not path:
            return None
        folders.remember(path)
        images = read_image_file(path, parent=self)
        if not images:
            return None
        for image in images:                    # every channel a source
            self.session.add_source(image)
        return images[0]

    def _count_in_view(self) -> None:
        if self.filter.layer is not None and not self.layer.is_image:
            self.filter._update_count()

    def _add_image(self) -> None:
        """*+ > image*: the file's kept frames if it has any, else an image
        already opened, else a layer that waits for one.

        It asked for a TIFF when there were no frames, which during a live
        fit -- whose frames come with its end -- was never what was meant.
        A file is one choice of the layer's *source*, beside the frames.
        """
        from ..images import ImageData
        self.session.add_image(None if self.session.images else ImageData.empty())
        self.strip.select(len(self.session.layers) - 1)

    def _grouping(self) -> None:
        from .dialogs import GroupingDialog
        GroupingDialog(self.session, self).exec()

    def _on_grouped(self, on: bool) -> None:
        # a setting: the blinks are merged when first drawn, or for a large
        # table in the background (`gui.merging`), and the picture follows
        self.session.show_grouped(self.strip.current, on)
        self.filter.bind(self.layer)     # the same bounds, over the other table
        self.session.changed("layer")
