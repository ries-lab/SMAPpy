"""An ROI's frame range and z range, set in the ROI manager's *annotate*.

What is measured in an ROI is its shape (or its id) narrowed by these
(`ROI.frames`, `ROI.z`): the part of a track before it binds, the slice of a
cell between two heights.  Each has two fields, blank for an open end, and a
small plot to drag it on -- the ROI's localizations per frame, and a side
view, x against z (along the ROI's *direction* line when it has one) -- so
a range is chosen by looking at the data rather than by guessing numbers.
The plots show the ROI without its ranges, the range a band over them.

One drag, one typed field or one *to all used* is one step of the ROI undo.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGridLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)

from ..roi_manager.core import DIRECTION

SIDE_POINTS = 20000          # the side view draws at most this many, a sample
BAND_BRUSH = (46, 158, 79, 70)
PLOT_HEIGHT = 110


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
    """The side view's horizontal axis: x, or the distance along the ROI's
    *direction* line from its start, and what to call it."""
    line = roi.lines.get(DIRECTION)
    if line is None:
        return x, "x (nm)"
    (x0, y0), (x1, y1) = line
    d = np.array([x1 - x0, y1 - y0], float)
    d /= max(np.hypot(*d), 1e-12)
    return (x - x0) * d[0] + (y - y0) * d[1], "along direction (nm)"


class RangePlot(pg.PlotWidget):
    """A small plot with a band to drag: horizontal for frames, vertical
    (a z range) for the side view.  ``moved`` fires when a drag ends."""

    def __init__(self, vertical: bool, parent=None):
        super().__init__(parent=parent, enableMenu=False)
        self.setFixedHeight(PLOT_HEIGHT)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.getPlotItem().setContentsMargins(0, 0, 0, 0)
        orientation = "horizontal" if vertical else "vertical"
        self.band = pg.LinearRegionItem(orientation=orientation, brush=BAND_BRUSH)
        self.band.setZValue(10)
        self.addItem(self.band)
        self.band.hide()
        self.data = None


class RangePanel(QWidget):
    """The selected ROI's frame range and z range."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window             # the `ROIManagerWindow`
        self._filling = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(3)
        self.fields = {}
        for row, (name, label, unit) in enumerate((("frames", "frames", ""),
                                                   ("z", "z", "nm"))):
            grid.addWidget(QLabel(label), row, 0)
            for k, end in enumerate(("from", "to")):
                edit = QLineEdit()
                edit.setPlaceholderText("all" if k == 0 else "")
                edit.setMaximumWidth(70)
                edit.setToolTip(f"the {label} range's {'low' if k == 0 else 'high'} end"
                                + (f", {unit}" if unit else "") + "; blank: open")
                edit.editingFinished.connect(lambda n=name: self._typed(n))
                grid.addWidget(edit, row, 1 + 2 * k)
                if k == 0:
                    grid.addWidget(QLabel("to"), row, 2)
                self.fields[(name, k)] = edit
            if unit:
                grid.addWidget(QLabel(unit), row, 4)
        self.clear_button = QPushButton("All frames and z")
        self.clear_button.setToolTip("clear the selected ROI's ranges: every frame, "
                                     "every z")
        self.clear_button.clicked.connect(self._clear)
        self.all_button = QPushButton("To all used")
        self.all_button.setToolTip("give every used ROI the selected ROI's frame "
                                   "and z range")
        self.all_button.clicked.connect(self._to_all)
        grid.addWidget(self.clear_button, 0, 5)
        grid.addWidget(self.all_button, 1, 5)
        grid.setColumnStretch(6, 1)
        layout.addLayout(grid)
        self.frame_plot = RangePlot(vertical=False)
        self.frame_plot.setLabel("bottom", "frame")
        self.frame_plot.setToolTip("the ROI's localizations per frame; drag the "
                                   "band or its edges to set the frame range")
        self.side_plot = RangePlot(vertical=True)
        self.side_plot.setLabel("left", "z (nm)")
        self.side_plot.setToolTip("the ROI from the side; drag the band or its "
                                  "edges to set the z range")
        for name, plot in (("frames", self.frame_plot), ("z", self.side_plot)):
            plot.band.sigRegionChangeFinished.connect(lambda _, n=name: self._dragged(n))
            layout.addWidget(plot)
        self.note = QLabel("")
        self.note.setStyleSheet("color: gray")
        layout.addWidget(self.note)

    @property
    def project(self):
        return self.window.project

    # ------------------------------------------------------------ showing
    def refresh(self) -> None:
        """The fields and plots for the selected ROI."""
        roi = self.window._selected()
        self._filling = True
        try:
            for (name, k), edit in self.fields.items():
                bounds = getattr(roi, name) if roi is not None else None
                if not edit.hasFocus():
                    edit.setText(bound_text(bounds[k]) if bounds else "")
                edit.setEnabled(roi is not None)
            for b in (self.clear_button, self.all_button):
                b.setEnabled(roi is not None)
            self._plot(roi)
        finally:
            self._filling = False

    def _plot(self, roi) -> None:
        for plot in (self.frame_plot, self.side_plot):
            for item in list(plot.listDataItems()):
                plot.removeItem(item)
            plot.band.hide()
        if roi is None or roi.file_id not in self.project.sources:
            self.note.setText("")
            return
        project = self.project
        locs = project.state(roi.file_id).locs
        # the ROI as it would be without its ranges: what they are cut from
        rows = project.indices(replace(roi, frames=None, z=None))
        if "frame" in locs and len(rows):
            frames = np.asarray(locs["frame"])[rows]
            lo, hi = float(frames.min()), float(frames.max())
            bins = int(min(max(hi - lo + 1, 1), 200))
            counts, edges = np.histogram(frames, bins=bins, range=(lo, hi + 1))
            self.frame_plot.plot(edges, counts, stepMode="center", fillLevel=0,
                                 brush=(200, 200, 200, 160), pen=None)
            self._band(self.frame_plot, roi.frames, lo, hi + 1)
        has_z = "z_nm" in locs
        self.side_plot.setVisible(has_z)
        self.fields[("z", 0)].setEnabled(has_z and roi is not None)
        self.fields[("z", 1)].setEnabled(has_z and roi is not None)
        if has_z and len(rows):
            pick = rows if len(rows) <= SIDE_POINTS else np.sort(
                np.random.default_rng(0).choice(rows, SIDE_POINTS, replace=False))
            x = np.asarray(locs["x_nm"], float)[pick]
            y = np.asarray(locs["y_nm"], float)[pick]
            z = np.asarray(locs["z_nm"], float)[pick]
            h, label = side_axis(roi, x, y)
            self.side_plot.setLabel("bottom", label)
            self.side_plot.plot(h, z, pen=None, symbol="o", symbolSize=2,
                                symbolPen=None, symbolBrush=(230, 230, 230, 200))
            good = np.isfinite(z)
            if good.any():
                self._band(self.side_plot, roi.z, float(z[good].min()),
                           float(z[good].max()))
        self.note.setText("" if has_z else "no z column: the z range cuts nothing")

    def _band(self, plot, bounds, lo, hi) -> None:
        """The band at the range, or over all of it when there is none."""
        a = lo if not bounds or bounds[0] is None else bounds[0]
        b = hi if not bounds or bounds[1] is None else bounds[1]
        blocked = plot.band.blockSignals(True)
        plot.band.setRegion((a, b))
        plot.band.blockSignals(blocked)
        plot.band.data = (lo, hi)
        plot.band.show()

    # ------------------------------------------------------------ editing
    def _set(self, roi_ids, **ranges) -> None:
        self.project.set_ranges(roi_ids, **ranges)
        self.window.ranges_changed()

    def _typed(self, name: str) -> None:
        roi = self.window._selected()
        if self._filling or roi is None:
            return
        bounds = [number_or_none(self.fields[(name, k)].text()) for k in (0, 1)]
        if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
            bounds.reverse()
        current = getattr(roi, name)
        wanted = None if bounds == [None, None] else bounds
        if wanted != current:
            self._set([roi.id], **{name: wanted})

    def _dragged(self, name: str) -> None:
        """A band let go: its ends, an end at the data's edge left open."""
        roi = self.window._selected()
        plot = self.frame_plot if name == "frames" else self.side_plot
        if self._filling or roi is None or plot.band.data is None:
            return
        lo, hi = plot.band.data
        a, b = plot.band.getRegion()
        if name == "frames":
            # frames are whole; the histogram's last edge is one past the last
            a, b = float(np.round(a)), float(np.round(b)) - 1
            hi = hi - 1
        else:
            a, b = round(float(a), 1), round(float(b), 1)
        bounds = [None if a <= lo else a, None if b >= hi else b]
        self._set([roi.id], **{name: None if bounds == [None, None] else bounds})

    def _clear(self) -> None:
        roi = self.window._selected()
        if roi is not None:
            self._set([roi.id], frames=None, z=None)

    def _to_all(self) -> None:
        roi = self.window._selected()
        if roi is None:
            return
        used = [i for i, r in self.project.rois.items() if r.use]
        self._set(used, frames=roi.frames, z=roi.z)
