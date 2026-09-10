"""The ROI manager window: four quadrants over one file's localizations.

Upper left the whole file, upper right the zoom, lower right the ROI, and at
the lower left the files and the ROIs.  A click in the file picks where to
zoom, a click in the zoom drafts an ROI, and *Add* stores it.  The controls
that decide what an ROI *is* -- its geometry, the finder, the evaluation --
live in the control window's ROI tab; this window is for choosing and
judging.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ..render import FieldOfView
from ..session import Session

DRAFT_PEN = pg.mkPen("#2f7fd0", width=2, style=Qt.DashLine)
DRAWING_PEN = pg.mkPen("#00c4ff", width=2, style=Qt.DashLine)
DIRECTION_PEN = pg.mkPen("#e05fd0", width=2)
ROI_PEN = pg.mkPen("#2e9e4f", width=2)          # the selected ROI
EXCLUDED_PEN = pg.mkPen("#8a8a8a", width=1)     # not used
EXCLUDED_ACTIVE_PEN = pg.mkPen("#8a8a8a", width=2)
OTHER_PEN = pg.mkPen("#ffb300", width=1)
FRAME_PEN = pg.mkPen("#ffd54a", width=1)
DETAIL_NM = 3000.0            # the zoom's width to start with
RIM_PIXELS = 6.0              # how near the outline a click must be to select


def arrow(direction):
    """A line with a small head at its end, for a direction."""
    (x0, y0), (x1, y1) = np.asarray(direction, float)
    d = np.array([x1 - x0, y1 - y0])
    length = float(np.hypot(*d)) or 1.0
    unit = d / length
    side = np.array([-unit[1], unit[0]])
    head = 0.18 * length
    left = np.array([x1, y1]) - head * unit + 0.5 * head * side
    right = np.array([x1, y1]) - head * unit - 0.5 * head * side
    xs = [x0, x1, np.nan, left[0], x1, right[0]]
    ys = [y0, y1, np.nan, left[1], y1, right[1]]
    return xs, ys


def rim_distance(x: float, y: float, center, shape: str, size: float, polygon=None) -> float:
    """Distance from a point to an ROI's outline, in data units.

    Selection asks for this rather than for "inside", so that a click in the
    middle of an ROI is free to start a new, overlapping one.
    """
    p = np.array([x, y], float)
    if polygon is not None:
        v = np.asarray(polygon, float)
        a, b = v, np.roll(v, -1, axis=0)
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
        return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))
    d = p - np.asarray(center, float)
    if shape == "circle":
        return abs(float(np.hypot(*d)) - size / 2)
    q = np.abs(d) - size / 2                     # signed distance to the square
    inside = min(max(q[0], q[1]), 0.0)
    return abs(float(np.linalg.norm(np.maximum(q, 0.0)) + inside))


def outline(center, shape: str, size: float, polygon=None):
    """The closed outline of an ROI, in data coordinates."""
    if polygon is not None:
        v = np.asarray(polygon, float)
        return np.r_[v[:, 0], v[0, 0]], np.r_[v[:, 1], v[0, 1]]
    cx, cy = center
    if shape == "circle":
        t = np.linspace(0, 2 * np.pi, 65)
        return cx + size / 2 * np.cos(t), cy + size / 2 * np.sin(t)
    h = size / 2
    return (np.array([cx - h, cx + h, cx + h, cx - h, cx - h]),
            np.array([cy - h, cy - h, cy + h, cy + h, cy - h]))


class ImagePane(QWidget):
    """One rendered quadrant: an image, outlines over it, and clicks."""

    clicked = Signal(float, float)
    moved = Signal(float, float)           # only while `tracking` is on
    zoomed = Signal(float)
    panned = Signal(float, float)          # how far to move the view, in nm

    def __init__(self, title: str, zoomable: bool = False, pannable: bool = False,
                 parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        self.label = QLabel(title)
        self.label.setStyleSheet("color: gray; font-weight: bold")
        layout.addWidget(self.label)
        self.graphics = pg.GraphicsLayoutWidget()
        self.view = self.graphics.addViewBox(lockAspect=True, invertY=True, enableMenu=False,
                                             enableMouse=False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image)
        self.shapes = {name: pg.PlotDataItem(pen=pen, connect="finite")
                       for name, pen in (("others", OTHER_PEN), ("excluded", EXCLUDED_PEN),
                                         ("excluded_active", EXCLUDED_ACTIVE_PEN),
                                         ("roi", ROI_PEN), ("draft", DRAFT_PEN),
                                         ("direction", DIRECTION_PEN),
                                         ("drawing", DRAWING_PEN), ("frame", FRAME_PEN))}
        for item in self.shapes.values():
            self.view.addItem(item)
        layout.addWidget(self.graphics)
        self.width_nm = DETAIL_NM
        self.zoomable = zoomable
        self.pannable = pannable
        self.fov: Optional[FieldOfView] = None
        self._press = None
        self._last = None
        self._dragged = False
        self.tracking = False
        self.graphics.viewport().installEventFilter(self)

    def set_tracking(self, on: bool) -> None:
        """Report the mouse without a button held, for a rubber line."""
        self.tracking = bool(on)
        self.graphics.viewport().setMouseTracking(bool(on))

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind.name == "MouseButtonPress" if hasattr(kind, "name") else False:
            pass
        from PySide6.QtCore import QEvent
        if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._press = self._last = event.position()
            self._dragged = False
        elif kind == QEvent.MouseMove and self.tracking and self.fov is not None:
            p = self.view.mapSceneToView(self.graphics.mapToScene(event.position().toPoint()))
            self.moved.emit(p.x(), p.y())
            return False
        elif kind == QEvent.MouseMove and self._press is not None and self.pannable:
            # drag to move the view; a click that never moves still counts
            delta = event.position() - self._last
            if self._dragged or (event.position() - self._press).manhattanLength() > 3:
                self._dragged = True
                self._last = event.position()
                if self.fov is not None:
                    self.panned.emit(-delta.x() * self.fov.pixelsize,
                                     -delta.y() * self.fov.pixelsize)
                return True
        elif kind == QEvent.MouseButtonRelease and self._press is not None:
            moved = (event.position() - self._press).manhattanLength()
            dragged, self._press, self._dragged = self._dragged, None, False
            if not dragged and moved <= 3 and self.fov is not None:
                p = self.view.mapSceneToView(self.graphics.mapToScene(event.position().toPoint()))
                self.clicked.emit(p.x(), p.y())
                return True
        elif kind == QEvent.Wheel and self.zoomable:
            steps = event.angleDelta().y() / 120.0
            self.width_nm = float(np.clip(self.width_nm * 1.2 ** -steps, 50.0, 1e7))
            self.zoomed.emit(self.width_nm)
            return True
        return super().eventFilter(obj, event)

    def render(self, state, x0: float, x1: float, y0: float, y1: float) -> None:
        size = self.graphics.size()
        nx, ny = max(size.width(), 32), max(size.height(), 32)
        fov = FieldOfView.fit((x0, x1), (y0, y1), nx, ny)
        rgb, _ = state.image(fov)
        self.image.setImage(np.ascontiguousarray(rgb), levels=[0, 1], autoLevels=False)
        self.image.setRect(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0))
        self.view.setRange(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0), padding=0)
        self.fov = fov

    def clear(self) -> None:
        self.image.clear()
        self.fov = None
        for item in self.shapes.values():
            item.setData([], [])

    def draw(self, name: str, xs=None, ys=None) -> None:
        item = self.shapes[name]
        if xs is None or len(xs) == 0:
            item.setData([], [])
        else:
            item.setData(np.asarray(xs, float), np.asarray(ys, float), connect="finite")


class ROIManagerWindow(QMainWindow):
    changed = Signal()          # ROIs added, removed or toggled

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)      # a window, not a child widget
        self.setWindowTitle("smappy ROI manager")
        self.session = session
        self.active: Optional[str] = None
        self.draft: Optional[np.ndarray] = None
        self.draft_polygon: Optional[list] = None
        self.draft_direction: Optional[list] = None
        self.drawing: Optional[str] = None       # "polygon" or "direction"
        self._points: list = []
        self._hover: Optional[np.ndarray] = None
        self.zoom_center: Optional[np.ndarray] = None
        self._loading = False

        self.file_pane = ImagePane("file")
        self.zoom_pane = ImagePane("zoom", zoomable=True, pannable=True)
        self.roi_pane = ImagePane("ROI", zoomable=True)

        lists = QWidget()
        llayout = QVBoxLayout(lists)
        llayout.setContentsMargins(2, 2, 2, 2)
        llayout.setSpacing(2)
        side = QHBoxLayout()
        side.setSpacing(6)
        file_side = QVBoxLayout()
        file_side.setSpacing(1)
        file_side.addWidget(QLabel("<b>files</b>"))
        self.files = QListWidget()
        self.files.currentRowChanged.connect(self._on_file)
        file_side.addWidget(self.files, 1)
        side.addLayout(file_side, 1)
        roi_side_list = QVBoxLayout()
        roi_side_list.setSpacing(1)
        roi_side_list.addWidget(QLabel("<b>ROIs</b>"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ROI", "file", "use"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.itemChanged.connect(self._on_item)
        self.table.setMaximumWidth(190)          # the numbers need no more
        roi_side_list.addWidget(self.table, 1)
        side.addLayout(roi_side_list, 0)
        llayout.addLayout(side, 1)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("remove the selected ROI")
        self.remove_button.clicked.connect(self._remove)
        llayout.addWidget(self.remove_button)
        toggle = QShortcut(QKeySequence(Qt.Key_Space), self)
        toggle.setContext(Qt.WindowShortcut)
        toggle.activated.connect(self._toggle)
        cancel = QShortcut(QKeySequence(Qt.Key_Escape), self)
        cancel.setContext(Qt.WindowShortcut)
        cancel.activated.connect(self._cancel_drawing)

        roi_side = QWidget()
        rlayout = QVBoxLayout(roi_side)
        rlayout.setContentsMargins(0, 0, 0, 0)
        rlayout.addWidget(self.roi_pane, 1)
        tools = QHBoxLayout()
        tools.setSpacing(3)
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setCheckable(True)
        self.polygon_button.setToolTip("draw an outline in this image: a click per "
                                       "vertex, the first vertex or a double-click "
                                       "closes it, Escape cancels")
        self.polygon_button.clicked.connect(lambda: self._start_drawing("polygon"))
        self.direction_button = QPushButton("Direction")
        self.direction_button.setCheckable(True)
        self.direction_button.setToolTip("two clicks, start then end: an arrow the "
                                         "analysis plugins can read")
        self.direction_button.clicked.connect(lambda: self._start_drawing("direction"))
        self.clear_shape_button = QPushButton("Clear shape")
        self.clear_shape_button.setToolTip("back to the global circle or square")
        self.clear_shape_button.clicked.connect(self._clear_shape)
        self.clear_line_button = QPushButton("Clear line")
        self.clear_line_button.clicked.connect(self._clear_line)
        for b in (self.polygon_button, self.direction_button,
                  self.clear_shape_button, self.clear_line_button):
            tools.addWidget(b)
        rlayout.addLayout(tools)
        self.add_button = QPushButton("Add")
        self.add_button.setToolTip("store the drafted ROI (Enter)")
        self.add_button.setShortcut(Qt.Key_Return)
        self.add_button.clicked.connect(self._add)
        rlayout.addWidget(self.add_button)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.file_pane)
        top.addWidget(self.zoom_pane)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(lists)
        bottom.addWidget(roi_side)
        whole = QSplitter(Qt.Vertical)
        whole.addWidget(top)
        whole.addWidget(bottom)
        for s in (top, bottom, whole):
            s.setSizes([500, 500])
        self.setCentralWidget(whole)
        self.status = self.statusBar()
        self.resize(1100, 800)

        self.file_pane.clicked.connect(self._file_clicked)
        self.zoom_pane.clicked.connect(self._zoom_clicked)
        self.roi_pane.clicked.connect(self._roi_clicked)
        self.roi_pane.moved.connect(self._roi_hover)
        self.zoom_pane.zoomed.connect(lambda _: self.redraw())
        self.zoom_pane.panned.connect(self._zoom_panned)
        self.roi_pane.zoomed.connect(self._preview_zoomed)
        session.on_change(self._on_session)
        self.refresh(files=True)

    # ---------------------------------------------------------------- data
    @property
    def project(self):
        return self.session.rois

    def current_file(self) -> Optional[str]:
        row = self.files.currentRow()
        ids = list(self.project.sources)
        return ids[row] if 0 <= row < len(ids) else None

    def _on_session(self, what: str) -> None:
        if what in ("locs", "rois", "layers"):
            self.refresh(files=True)
        elif what in ("layer", "regrouped"):
            self.project.sync()
            self.redraw()

    # ------------------------------------------------------------ refresh
    def refresh(self, files: bool = False) -> None:
        project = self.project
        self._loading = True
        if files:
            row = self.files.currentRow()
            self.files.clear()
            for source in project.sources.values():
                self.files.addItem(QListWidgetItem(f"{source.number + 1}. {source.name}"))
            saved = project.navigation.get("file")
            ids = list(project.sources)
            if saved in ids:
                row = ids.index(saved)
            self.files.setCurrentRow(max(0, min(row, self.files.count() - 1)))
        self._fill_table()
        self._loading = False
        self.redraw()

    def _fill_table(self) -> None:
        project = self.project
        was_loading, self._loading = self._loading, True
        numbers = project.numbers()
        self._rows = list(project.rois)
        self.table.setRowCount(len(self._rows))
        for row, roi_id in enumerate(self._rows):
            roi = project.rois[roi_id]
            for column, text in ((0, str(numbers[roi_id])),
                                 (1, str(project.file_number(roi.file_id)))):
                item = QTableWidgetItem(text)
                if not roi.use:
                    item.setForeground(pg.mkColor("#8a8a8a"))
                self.table.setItem(row, column, item)
            use = QTableWidgetItem()
            use.setFlags((use.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            use.setCheckState(Qt.Checked if roi.use else Qt.Unchecked)
            use.setToolTip("include this ROI in the evaluation (space toggles it)")
            self.table.setItem(row, 2, use)
            if roi_id == self.active:
                self.table.selectRow(row)
        self._loading = was_loading
        self.status.showMessage(f"{len(self._rows)} ROIs, "
                                f"{sum(1 for r in project.rois.values() if r.use)} used")

    # ------------------------------------------------------------ drawing
    def redraw(self) -> None:
        project = self.project
        file_id = self.current_file()
        if file_id is None:
            for pane in (self.file_pane, self.zoom_pane, self.roi_pane):
                pane.clear()
            return
        state = project.state(file_id)
        x0, y0, x1, y1 = state.index.bounds
        self.file_pane.render(state, x0, x1, y0, y1)
        rois = project.rois_of(file_id)

        # the file: every ROI, small
        self._draw_outlines(self.file_pane, rois)

        centre = self.zoom_center
        if centre is None:
            centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            self.zoom_center = centre
        half = self.zoom_pane.width_nm / 2
        self.zoom_pane.render(state, centre[0] - half, centre[0] + half,
                              centre[1] - half, centre[1] + half)
        self._draw_outlines(self.zoom_pane, rois)
        self.zoom_pane.draw("draft", *([[], []] if self.draft is None else
                                       outline(self.draft, project.shape, project.size_nm)))
        # the frame the zoom covers, on the file image
        self.file_pane.draw("frame",
                            [centre[0] - half, centre[0] + half, centre[0] + half,
                             centre[0] - half, centre[0] - half],
                            [centre[1] - half, centre[1] - half, centre[1] + half,
                             centre[1] + half, centre[1] - half])
        self._draw_roi_pane(state)

    def _draw_outlines(self, pane, rois) -> None:
        """Every ROI on one pane: grey when it is not used, green when selected."""
        project = self.project
        groups = {"others": [], "excluded": [], "excluded_active": [], "roi": []}
        for roi in rois:
            active = roi.id == self.active
            name = ("excluded_active" if active else "excluded") if not roi.use else (
                "roi" if active else "others")
            groups[name].append(roi)
        for name, chosen in groups.items():
            xs, ys = [], []
            for roi in chosen:
                ox, oy = outline(roi.center, project.shape, project.size_nm, roi.polygon)
                xs.extend(list(ox) + [np.nan])
                ys.extend(list(oy) + [np.nan])
            pane.draw(name, xs, ys)
        dx, dy = [], []
        for roi in rois:
            if roi.direction:
                ax, ay = arrow(roi.direction)
                dx.extend(list(ax) + [np.nan])
                dy.extend(list(ay) + [np.nan])
        pane.draw("direction", dx, dy)

    def _draw_roi_pane(self, state) -> None:
        project = self.project
        roi = project.rois.get(self.active) if self.active else None
        centre = self.draft if self.draft is not None else (
            np.asarray(roi.center, float) if roi is not None else None)
        if centre is None:
            self.roi_pane.clear()
            return
        half = project.preview_width / 2
        self.roi_pane.width_nm = project.preview_width
        self.roi_pane.render(state, centre[0] - half, centre[0] + half,
                             centre[1] - half, centre[1] + half)
        drafted = self.draft is not None
        polygon = self.draft_polygon if drafted else (roi.polygon if roi is not None else None)
        direction = self.draft_direction if drafted else (roi.direction if roi is not None else None)
        xs, ys = outline(centre, project.shape, project.size_nm, polygon)
        used = drafted or roi is None or roi.use
        for name in ("draft", "roi", "excluded_active"):
            self.roi_pane.draw(name)
        self.roi_pane.draw("draft" if drafted else ("roi" if used else "excluded_active"), xs, ys)
        self.roi_pane.draw("direction", *(arrow(direction) if direction else ([], [])))
        self.roi_pane.label.setText("ROI (draft)" if drafted else "ROI")
        self.add_button.setEnabled(drafted)

    # --------------------------------------------------------- interaction
    def _on_file(self, row: int) -> None:
        if self._loading:
            return
        self.project.navigation["file"] = self.current_file()
        self.zoom_center = None
        self.draft = None
        self.redraw()
        self.changed.emit()          # the tab names the file the finder acts on

    def _zoom_panned(self, dx: float, dy: float) -> None:
        if self.zoom_center is not None:
            self.zoom_center = self.zoom_center + np.array([dx, dy])
            self.redraw()

    def _preview_zoomed(self, width: float) -> None:
        """The wheel over the ROI image sets the overview width everywhere."""
        self.project.preview_nm = float(width)
        self.changed.emit()
        self.redraw()

    def _file_clicked(self, x: float, y: float) -> None:
        if np.isnan(x):
            self.redraw()
            return
        self.zoom_center = np.array([x, y])
        self.redraw()

    def _zoom_clicked(self, x: float, y: float) -> None:
        """A click drafts an ROI there, unless it lands on a stored one."""
        if np.isnan(x):
            self.redraw()
            return
        hit = self._roi_at(x, y)
        if hit is not None:
            self.draft = None
            self.select(hit.id)
            return
        self.draft = np.array([x, y])
        self.draft_polygon = self.draft_direction = None
        self.redraw()

    def _roi_clicked(self, x: float, y: float) -> None:
        """A click in the ROI image: place a vertex, or recentre the draft."""
        if np.isnan(x):
            self.redraw()
            return
        if self.drawing:
            self._place(x, y)
            return
        hit = self._roi_at(x, y, self.roi_pane)
        if hit is not None and self.draft is None:
            self.select(hit.id)
        elif self.draft is not None:
            self.draft = np.array([x, y])
            self.redraw()

    def _roi_hover(self, x: float, y: float) -> None:
        if self.drawing and self._points:
            self._hover = np.array([x, y])
            self._draw_in_progress()

    # ----------------------------------------------- polygon and direction
    def _start_drawing(self, kind: str) -> None:
        """Draw an outline or a direction in the ROI image."""
        if self.drawing == kind:
            self._cancel_drawing()
            return
        if self.roi_pane.fov is None:
            self._cancel_drawing()
            self.status.showMessage("select or draft an ROI first")
            return
        self.drawing = kind
        self._points = []
        self._hover = None
        self.roi_pane.set_tracking(True)
        self.polygon_button.setChecked(kind == "polygon")
        self.direction_button.setChecked(kind == "direction")
        self.status.showMessage("a click per vertex; the first vertex or a "
                                "double-click closes it, Escape cancels" if kind == "polygon"
                                else "click the start, then the end")

    def _cancel_drawing(self) -> None:
        self.drawing = None
        self._points = []
        self._hover = None
        self.roi_pane.set_tracking(False)
        self.polygon_button.setChecked(False)
        self.direction_button.setChecked(False)
        self.roi_pane.draw("drawing")
        self.redraw()

    def _place(self, x: float, y: float) -> None:
        point = np.array([x, y], float)
        if self.drawing == "direction":
            self._points.append(point)
            if len(self._points) == 2:
                self._finish_direction()
            else:
                self._draw_in_progress()
            return
        near = self.roi_pane.fov.pixelsize * 8 if self.roi_pane.fov else 1.0
        if len(self._points) >= 3 and np.hypot(*(point - self._points[0])) <= near:
            self._finish_polygon()                 # clicked the first vertex
            return
        self._points.append(point)
        self._draw_in_progress()

    def _draw_in_progress(self) -> None:
        points = list(self._points)
        if self._hover is not None:
            points = points + [self._hover]
        if not points:
            self.roi_pane.draw("drawing")
            return
        v = np.array(points, float)
        if self.drawing == "polygon" and len(v) > 2:
            v = np.vstack([v, v[0]])               # show how it would close
        self.roi_pane.draw("drawing", v[:, 0], v[:, 1])

    def _finish_polygon(self) -> None:
        polygon = [[float(p[0]), float(p[1])] for p in self._points]
        self._cancel_drawing()
        if len(polygon) < 3:
            return
        centre = np.asarray(polygon, float).mean(axis=0)
        roi = self._selected()
        if self.draft is not None or roi is None:
            self.draft = centre                    # a drafted ROI with this outline
            self.draft_polygon = polygon
        else:
            roi.polygon = polygon
            roi.center = [float(centre[0]), float(centre[1])]
            self.changed.emit()
        self.redraw()

    def _finish_direction(self) -> None:
        line = [[float(p[0]), float(p[1])] for p in self._points]
        self._cancel_drawing()
        roi = self._selected()
        if self.draft is not None or roi is None:
            if self.draft is None:
                self.draft = np.asarray(line, float).mean(axis=0)
            self.draft_direction = line
        else:
            roi.direction = line
            self.changed.emit()
        self.redraw()

    def _clear_shape(self) -> None:
        """Back to the global circle or square."""
        self._cancel_drawing()
        if self.draft is not None:
            self.draft_polygon = None
        roi = self._selected()
        if roi is not None and self.draft is None:
            roi.polygon = None
            self.changed.emit()
        self.redraw()

    def _clear_line(self) -> None:
        self._cancel_drawing()
        if self.draft is not None:
            self.draft_direction = None
        roi = self._selected()
        if roi is not None and self.draft is None:
            roi.direction = None
            self.changed.emit()
        self.redraw()

    def _roi_at(self, x: float, y: float, pane=None):
        """The ROI whose outline the click landed on, if any.

        Only the rim counts: clicking inside an ROI starts a new one, which is
        what makes overlapping ROIs drawable.
        """
        project = self.project
        pane = pane or self.zoom_pane
        tolerance = RIM_PIXELS * (pane.fov.pixelsize if pane.fov is not None else 1.0)
        best, best_d = None, np.inf
        for roi in project.rois_of(self.current_file()):
            d = rim_distance(x, y, roi.center, project.shape, project.size_nm, roi.polygon)
            if d <= tolerance and d < best_d:
                best, best_d = roi, d
        return best

    def _on_item(self, item) -> None:
        """The use box in the table."""
        if self._loading or item.column() != 2 or item.row() >= len(self._rows):
            return
        roi = self.project.rois[self._rows[item.row()]]
        roi.use = item.checkState() == Qt.Checked
        self._fill_table()
        self.redraw()
        self.changed.emit()

    def _on_select(self) -> None:
        if self._loading:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self.select(self._rows[rows[0].row()], from_table=True)

    def select(self, roi_id: str, from_table: bool = False) -> None:
        """Show an ROI: its file, the zoom around it, and the ROI image."""
        project = self.project
        roi = project.rois.get(roi_id)
        if roi is None:
            return
        self.active = roi_id
        self.draft = self.draft_polygon = self.draft_direction = None
        project.navigation["roi"] = roi_id
        if roi.file_id != self.current_file():
            ids = list(project.sources)
            if roi.file_id in ids:
                self._loading = True
                self.files.setCurrentRow(ids.index(roi.file_id))
                project.navigation["file"] = roi.file_id
                self._loading = False
        self.zoom_center = np.asarray(roi.center, float)
        if not from_table:
            self._fill_table()
        self.redraw()

    # --------------------------------------------------------------- edits
    def _add(self) -> None:
        file_id = self.current_file()
        if file_id is None or self.draft is None:
            return
        roi = self.project.add_roi(file_id, [float(self.draft[0]), float(self.draft[1])],
                                   polygon=self.draft_polygon)
        roi.direction = self.draft_direction
        self.draft = self.draft_polygon = self.draft_direction = None
        self.active = roi.id
        self._fill_table()
        self.redraw()
        self.changed.emit()

    def _selected(self):
        return self.project.rois.get(self.active) if self.active else None

    def _toggle(self) -> None:
        """Space: include or exclude the selected ROI."""
        roi = self._selected()
        if roi is not None:
            roi.use = not roi.use
            self._fill_table()
            self.redraw()
            self.changed.emit()

    def _remove(self) -> None:
        roi = self._selected()
        if roi is not None:
            self.project.rois.pop(roi.id, None)
            self.active = None
            self._fill_table()
            self.redraw()
            self.changed.emit()
