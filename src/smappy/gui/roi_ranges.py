"""An ROI's ranges, set in the ROI manager's *annotate*: the part of it that
is measured, cut on any attribute of the table.

What is measured in an ROI is its shape (or its id) narrowed by these
(`ROI.ranges`): the part of a track before it binds, the slice of a cell
between two heights, a MINFLUX trace's first seconds.  Which attributes are
offered is the project's (`ROIProject.range_columns`, *attributes*), frame
and z unless changed.  Each has two fields, blank for an open end, and a
small plot to drag its edges on (Shift and a drag inside the band moves it
whole) -- so a range is chosen by looking at the data rather than by
guessing numbers; a bound dragged or typed stays as set, and only *Clear*
opens it again.  How the plot looks depends on the
attribute (`range_kind`): a time or an id, where each value has many
localizations, as **bars**, the localizations counted over it; a measured
quantity -- z, sigma -- as **dots**, the attribute against the position
(along the ROI's line when it has one), so a band is set where the points
lie.  The plots show the ROI without its ranges, the range a band over
them.

One drag, one typed field, or one row's *Clear* or *To all used* is one
step of the ROI undo.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout, QWidget)

from ..roi_manager.core import DIRECTION, LINE, range_kind

SIDE_POINTS = 20000          # the dots draw at most this many, a sample
BAND_BRUSH = (46, 158, 79, 70)
PLOT_HEIGHT = 110
BINS = 200                   # bars: at most so many, or one per whole value
EDGE_GRAB_PIXELS = 6         # a band's edge is grabbed so far either side of it


def number_or_none(text: str) -> Optional[float]:
    """A field's value: blank (or not a number) is an open end."""
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def bound_text(value) -> str:
    return "" if value is None else f"{value:g}"


def side_axis(roi, x, y) -> tuple:
    """The dots' horizontal axis: x, or the distance along the ROI's
    *line* (an older project's *direction*) from its start, and what to
    call it."""
    line = roi.lines.get(LINE, roi.lines.get(DIRECTION))
    if line is None:
        return x, "x (nm)"
    (x0, y0), (x1, y1) = line
    d = np.array([x1 - x0, y1 - y0], float)
    d /= max(np.hypot(*d), 1e-12)
    return (x - x0) * d[0] + (y - y0) * d[1], "along the line (nm)"


def whole(values: np.ndarray) -> bool:
    """Are these whole numbers -- frames, ids -- so a bar is one value?"""
    v = values[np.isfinite(values)]
    return v.dtype.kind in "iub" or (len(v) > 0 and bool(np.all(np.mod(v, 1) == 0)))


class Band(pg.LinearRegionItem):
    """The range's band: its edges are what is dragged.

    pyqtgraph's band moves as a whole when dragged inside, and its edges
    are a few pixels wide, so setting a bound moved the window more often
    than not.  Here the edges are wide to grab (`EDGE_GRAB_PIXELS`), a drag
    inside the band moves it only with Shift held, and the band is no part
    of the plot's range: pyqtgraph scales the view to take it in, so an edge
    dragged near the plot's side made the scale run away under the mouse.
    """

    def __init__(self, orientation):
        super().__init__(orientation=orientation, brush=BAND_BRUSH)
        for line in self.lines:
            # the grab zone is the line's bounding box, its pen and the
            # largest marker, in pixels either side (`InfiniteLine`); no
            # marker is drawn, the zone is only made wider
            line._maxMarkerSize = EDGE_GRAB_PIXELS
            line.setPen(pg.mkPen((46, 158, 79), width=2))
            line.setHoverPen(pg.mkPen((46, 158, 79), width=4))

    def dataBounds(self, axis, frac=1.0, orthoRange=None):    # noqa: N802 (pyqtgraph's)
        return None

    def mouseDragEvent(self, ev):                             # noqa: N802
        from PySide6.QtCore import Qt
        if not (ev.modifiers() & Qt.ShiftModifier) and not self.moving:
            ev.ignore()
            return
        super().mouseDragEvent(ev)


class RangePlot(pg.PlotWidget):
    """A small plot with a band to drag: across the bars (the attribute on
    x), or along the dots (the attribute on y)."""

    def __init__(self, vertical: bool, parent=None):
        super().__init__(parent=parent, enableMenu=False)
        self.setFixedHeight(PLOT_HEIGHT)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        orientation = "horizontal" if vertical else "vertical"
        self.band = Band(orientation)
        self.band.setZValue(10)
        self.addItem(self.band)
        self.band.hide()
        self.data = None
        self.whole = False            # bars of whole values: a band snaps to them
        self.vertical = vertical

    def frame(self, lo: float, hi: float, bounds) -> None:
        """Fix the attribute's axis on the data and the range, with a margin,
        so neither a drag nor a bound far outside rescales it on the way."""
        ends = [lo, hi] + [b for b in (bounds or ()) if b is not None]
        a, b = min(ends), max(ends)
        pad = 0.03 * (b - a) if b > a else 0.5
        view = self.getPlotItem().getViewBox()
        if self.vertical:
            view.enableAutoRange(y=False)
            view.setYRange(a - pad, b + pad, padding=0)
        else:
            view.enableAutoRange(x=False)
            view.setXRange(a - pad, b + pad, padding=0)


class RangePanel(QWidget):
    """The ranges of the ROI the ROI image shows: the selected one, or the
    draft while there is one; a row per attribute the project offers."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window             # the `ROIManagerWindow`
        self._filling = False
        self._built: list = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(3)
        top.addWidget(QLabel("attributes"))
        self.columns = QLineEdit()
        self.columns.setToolTip("the attributes ranges are set on, separated by commas: "
                                "frame, z_nm by default; time_s for MINFLUX, sigma_nm, ...")
        self.columns.editingFinished.connect(self._columns_typed)
        top.addWidget(self.columns, 1)
        layout.addLayout(top)
        self.rows_widget = QWidget()
        self.rows = QVBoxLayout(self.rows_widget)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(2)
        layout.addWidget(self.rows_widget)
        self.fields: dict = {}           # (attribute, 0 or 1) -> QLineEdit
        self.buttons: dict = {}          # (attribute, "clear" or "all") -> QPushButton
        self.plots: dict = {}            # attribute -> RangePlot
        self.notes: dict = {}            # attribute -> QLabel

    @property
    def project(self):
        return self.window.project

    # ------------------------------------------------------------ building
    def _build(self, columns) -> None:
        """A row per attribute: its fields, then its plot."""
        while self.rows.count():
            item = self.rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # out of the panel now, not at the next turn of the event loop
                widget.setParent(None)
                widget.deleteLater()
        self.fields, self.plots, self.notes, self.buttons = {}, {}, {}, {}
        for column in columns:
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 2, 0, 0)
            v.setSpacing(1)
            grid = QGridLayout()
            grid.setHorizontalSpacing(3)
            grid.addWidget(QLabel(column), 0, 0)
            for k in (0, 1):
                edit = QLineEdit()
                edit.setPlaceholderText("all" if k == 0 else "")
                edit.setMaximumWidth(70)
                edit.setToolTip(f"the {column} range's {'low' if k == 0 else 'high'} end; "
                                "blank: open")
                edit.editingFinished.connect(lambda c=column: self._typed(c))
                grid.addWidget(edit, 0, 1 + 2 * k)
                if k == 0:
                    grid.addWidget(QLabel("to"), 0, 2)
                self.fields[(column, k)] = edit
            # on the range's own row: what they act on is in line with them
            clear = QPushButton("Clear")
            clear.setToolTip(f"clear this ROI's {column} range: all of it")
            clear.clicked.connect(lambda _=False, c=column: self._clear(c))
            to_all = QPushButton("To all used")
            to_all.setToolTip(f"give every used ROI this ROI's {column} range")
            to_all.clicked.connect(lambda _=False, c=column: self._to_all(c))
            grid.addWidget(clear, 0, 4)
            grid.addWidget(to_all, 0, 5)
            self.buttons[(column, "clear")], self.buttons[(column, "all")] = clear, to_all
            grid.setColumnStretch(6, 1)
            v.addLayout(grid)
            bars = range_kind(column) == "bars"
            plot = RangePlot(vertical=not bars)
            plot.column = column
            if bars:
                plot.setLabel("bottom", column)
                plot.setToolTip(f"the ROI's localizations over {column}; drag the band "
                                "or its edges to set the range")
            else:
                plot.setLabel("left", column)
                plot.setToolTip(f"{column} against the position; drag the band or its "
                                "edges to set the range")
            plot.band.sigRegionChangeFinished.connect(lambda _, c=column: self._dragged(c))
            v.addWidget(plot)
            note = QLabel("")
            note.setStyleSheet("color: gray")
            note.setVisible(False)
            v.addWidget(note)
            self.plots[column], self.notes[column] = plot, note
            self.rows.addWidget(box)
        self._built = list(columns)

    # ------------------------------------------------------------ showing
    def refresh(self) -> None:
        """The attributes, fields and plots for the ROI shown."""
        columns = list(self.project.range_columns)
        if columns != self._built:
            self._build(columns)
        if not self.columns.hasFocus():
            self.columns.setText(", ".join(columns))
        roi = self.window.shown_roi()
        self._filling = True
        try:
            for (column, k), edit in self.fields.items():
                bounds = roi.ranges.get(column) if roi is not None else None
                if not edit.hasFocus():
                    edit.setText(bound_text(bounds[k]) if bounds else "")
                edit.setEnabled(roi is not None)
            for b in self.buttons.values():
                b.setEnabled(roi is not None)
            self._plot(roi)
        finally:
            self._filling = False

    def _plot(self, roi) -> None:
        for column, plot in self.plots.items():
            for item in list(plot.listDataItems()):
                plot.removeItem(item)
            plot.getPlotItem().getViewBox().enableAutoRange()
            plot.band.hide()
            plot.band.data = None
            self.notes[column].setVisible(False)
        if roi is None or roi.file_id not in self.project.sources:
            return
        project = self.project
        locs = project.state(roi.file_id).locs
        # the ROI as it would be without its ranges: what they are cut from
        rows = project.indices(replace(roi, ranges={}))
        pick = rows if len(rows) <= SIDE_POINTS else np.sort(
            np.random.default_rng(0).choice(rows, SIDE_POINTS, replace=False))
        for column, plot in self.plots.items():
            note = self.notes[column]
            if column not in locs:
                plot.setVisible(False)
                note.setText(f"no attribute {column}: its range cuts nothing")
                note.setVisible(True)
                for k in (0, 1):
                    self.fields[(column, k)].setEnabled(False)
                continue
            plot.setVisible(True)
            if not len(rows):
                continue
            bounds = roi.ranges.get(column)
            if range_kind(column) == "bars":
                v = np.asarray(locs[column], float)[rows]
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                lo, hi = float(v.min()), float(v.max())
                plot.whole = whole(v)
                if plot.whole:
                    bins = int(min(max(hi - lo + 1, 1), BINS))
                    counts, edges = np.histogram(v, bins=bins, range=(lo, hi + 1))
                    top = hi + 1
                else:
                    counts, edges = np.histogram(v, bins=BINS, range=(lo, hi or lo + 1))
                    top = hi
                plot.plot(edges, counts, stepMode="center", fillLevel=0,
                          brush=(200, 200, 200, 160), pen=None)
                self._band(plot, bounds, lo, top)
            else:
                x = np.asarray(locs["x_nm"], float)[pick]
                y = np.asarray(locs["y_nm"], float)[pick]
                v = np.asarray(locs[column], float)[pick]
                h, label = side_axis(roi, x, y)
                plot.setLabel("bottom", label)
                plot.plot(h, v, pen=None, symbol="o", symbolSize=2,
                          symbolPen=None, symbolBrush=(230, 230, 230, 200))
                good = np.isfinite(v)
                if good.any():
                    self._band(plot, bounds, float(v[good].min()), float(v[good].max()))

    def _band(self, plot, bounds, lo, hi) -> None:
        """The band at the range, or over all of it when there is none."""
        a = lo if not bounds or bounds[0] is None else bounds[0]
        b = hi if not bounds or bounds[1] is None else bounds[1]
        if plot.whole and bounds and bounds[1] is not None and range_kind(plot.column) == "bars":
            # the last value kept is a bar, whose right edge is one past it:
            # drawn at the value itself, every drag let go took one off
            b = bounds[1] + 1
        plot.frame(lo, hi, bounds)
        blocked = plot.band.blockSignals(True)
        plot.band.setRegion((a, b))
        plot.band.blockSignals(blocked)
        plot.band.data = (lo, hi)
        plot.band.show()

    # ------------------------------------------------------------ editing
    def _set(self, roi_ids, ranges: dict) -> None:
        """Set ranges of these stored ROIs, one undo step, or of the draft
        (its id is in none of them), which is no edit of the project."""
        stored = [i for i in roi_ids if i in self.project.rois]
        if not stored:
            self.window.set_draft_ranges(ranges)
            return
        self.project.set_ranges(stored, ranges=ranges)
        self.window.ranges_changed()

    def _columns_typed(self) -> None:
        names = [n.strip() for n in self.columns.text().split(",")]
        before = list(self.project.range_columns)
        self.project.set_range_columns(names)
        if self.project.range_columns != before:
            self.refresh()

    def _typed(self, column: str) -> None:
        roi = self.window.shown_roi()
        if self._filling or roi is None:
            return
        bounds = [number_or_none(self.fields[(column, k)].text()) for k in (0, 1)]
        if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
            bounds.reverse()
        wanted = None if bounds == [None, None] else bounds
        if wanted != roi.ranges.get(column):
            self._set([roi.id], {column: wanted})

    def _dragged(self, column: str) -> None:
        """A band let go: its ends, as dragged.

        Both ends are kept as numbers, wherever they were let go.  An end
        at the data's edge was left open, so a band dragged out to it
        cleared its field and the range jumped back to *all*; a bound set by
        hand stays set, and only *Clear* opens it."""
        roi = self.window.shown_roi()
        plot = self.plots.get(column)
        if self._filling or roi is None or plot is None or plot.band.data is None:
            return
        a, b = plot.band.getRegion()
        if range_kind(column) == "bars" and plot.whole:
            # whole values; the last bar's right edge is one past the last
            a, b = float(np.round(a)), float(np.round(b)) - 1
        else:
            a, b = float(f"{a:.4g}"), float(f"{b:.4g}")
        if b < a:
            a, b = b, a
        self._set([roi.id], {column: [a, b]})

    def _clear(self, column: str) -> None:
        roi = self.window.shown_roi()
        if roi is not None and column in roi.ranges:
            self._set([roi.id], {column: None})

    def _to_all(self, column: str) -> None:
        """Every used ROI gets this ROI's range of ``column`` -- none, if it
        has none."""
        roi = self.window.shown_roi()
        if roi is None:
            return
        used = [i for i, r in self.project.rois.items() if r.use]
        if used:
            self._set(used, {column: roi.ranges.get(column)})

