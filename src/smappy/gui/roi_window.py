"""The ROI manager window: four quadrants over one file's localizations.

Upper left the whole file, upper right the zoom, lower right the ROI, and at
the lower left the files and the ROIs.  A click in the file picks where to
zoom, a click in the zoom drafts an ROI, and *Add* stores it.  The controls
that decide what an ROI *is* -- its geometry, the finder, the evaluation --
live in the control window's Regions tab; this window is for choosing and
judging.

**Many ROIs** -- the clusters of a whole cell made into ROIs, tens of
thousands -- cost what is on screen, not what is in the project: the list
makes a row (and checks whether its ROI is out of date) only when it is
shown, a pane draws the outlines of the ROIs in its view only, and past
`OUTLINE_LIMIT` of them a dot at each centre instead, with the selected ROI
still outlined.  Measured before: 7 s for every refresh of 20000 ROIs, the
list two thirds of it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRectF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDoubleSpinBox,
                               QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QPushButton,
                               QSplitter, QTableView, QVBoxLayout, QWidget)

from ..regions import Region
from ..render import FieldOfView
from ..session import Session

DRAFT_PEN = pg.mkPen("#2f7fd0", width=2, style=Qt.DashLine)
DRAWING_PEN = pg.mkPen("#00c4ff", width=2, style=Qt.DashLine)
DIRECTION_PEN = pg.mkPen("#e05fd0", width=2)
ROI_PEN = pg.mkPen("#2e9e4f", width=2)          # the selected ROI
EXCLUDED_PEN = pg.mkPen("#8a8a8a", width=1)     # not used
EXCLUDED_ACTIVE_PEN = pg.mkPen("#8a8a8a", width=2)
OTHER_PEN = pg.mkPen("#ffb300", width=1)
# a segmenter's preview: proposed, not added -- grey and dashed, so it reads
# as provisional over any LUT; the ones the finder rejected only dotted
PREVIEW_PEN = pg.mkPen("#b0b0b0", width=1.5, style=Qt.DashLine)
PREVIEW_REJECTED_PEN = pg.mkPen("#8a8a8a", width=1, style=Qt.DotLine)
FRAME_PEN = pg.mkPen("#ffd54a", width=1)
TILE_PEN = pg.mkPen("#5a6a7a", width=1)         # the grid
TILE_HERE_PEN = pg.mkPen("#00c4ff", width=2)    # the tile being looked at
STALE_COLOUR = "#c07000"        # an ROI whose evaluation is out of date
DETAIL_NM = 3000.0            # the zoom's width to start with
DEFAULT_TILE_NM = 5000.0      # a tile to start from
RIM_PIXELS = 6.0              # how near the outline a click must be to select
# a pane with more ROIs than this in view draws a dot at each centre instead
# of its outline: past it the outlines are a tangle and cost a second a redraw
OUTLINE_LIMIT = 1000
# the status line counts the ROIs waiting on the evaluation only up to this
# many: counting means checking every one, the cost the list no longer pays
COUNT_LIMIT = 2000


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


UNDO_TEXT_WIDTH = 130      # px of an undo button's text, past which it is elided


def _lowered(label: str) -> str:
    """"Remove ROI 3" -> "remove ROI 3", to follow "Undo"; an acronym
    ("NPC") keeps its capitals."""
    if len(label) > 1 and label[0].isupper() and label[1].islower():
        return label[0].lower() + label[1:]
    return label


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
                       for name, pen in (("preview", PREVIEW_PEN),
                                         ("preview_rejected", PREVIEW_REJECTED_PEN),
                                         ("others", OTHER_PEN), ("excluded", EXCLUDED_PEN),
                                         ("excluded_active", EXCLUDED_ACTIVE_PEN),
                                         ("roi", ROI_PEN), ("draft", DRAFT_PEN),
                                         ("direction", DIRECTION_PEN),
                                         ("drawing", DRAWING_PEN), ("frame", FRAME_PEN),
                                         ("tiles", TILE_PEN), ("tile", TILE_HERE_PEN))}
        for name, colour in (("centres", OTHER_PEN.color()),
                             ("centres_excluded", EXCLUDED_PEN.color())):
            self.shapes[name] = pg.PlotDataItem(pen=None, symbol="o", symbolSize=3,
                                                symbolPen=None, symbolBrush=colour)
        for item in self.shapes.values():
            self.view.addItem(item)
        layout.addWidget(self.graphics)
        self.title = title
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

    def clipboard_image(self):
        """The pane as it is on screen, outlines and all: the outlines are
        what this window is for, and the plain picture is the render's."""
        return self.graphics.grab().toImage() if self.fov is not None else None

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

    def note(self, text: str = "") -> None:
        """A word after the pane's title: why it shows dots, say."""
        self.label.setText(f"{self.title} - {text}" if text else self.title)


class ROIList(QAbstractTableModel):
    """The ROIs as the list shows them, a row made only when it is shown.

    Whether an ROI waits on the evaluation is a signature of its data and
    the steps, about 0.15 ms an ROI: checked as a row is drawn and kept
    until the next `reset`, rather than for every ROI on every refresh.
    """

    HEADERS = ("ROI", "file", "use")

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.ids: list = []
        self.row_of: dict = {}
        self.numbers: dict = {}
        self._steps = None
        self._waiting: dict = {}

    def reset(self, project) -> None:
        self.beginResetModel()
        self.ids = list(project.rois)
        self.row_of = {roi_id: i for i, roi_id in enumerate(self.ids)}
        self.numbers = project.numbers()
        self._waiting = {}
        try:
            self._steps = project.resolved()
        except Exception:
            self._steps = None
        self.endResetModel()

    def waiting(self, roi_id) -> bool:
        """Whether this ROI's numbers are older than its data or the steps:
        a list that does not say so invites reading last week's answer."""
        found = self._waiting.get(roi_id)
        if found is None:
            project = self.window.project
            roi = project.rois.get(roi_id)
            found = False
            if roi is not None and roi.use and self._steps:
                try:
                    found = bool(project.stale_steps(roi_id, self._steps))
                except Exception:
                    found = False
            self._waiting[roi_id] = found
        return found

    def rowCount(self, parent=QModelIndex()) -> int:          # noqa: N802 (Qt's)
        return 0 if parent.isValid() else len(self.ids)

    def columnCount(self, parent=QModelIndex()) -> int:       # noqa: N802 (Qt's)
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.HEADERS[section]
        return None

    def flags(self, index):
        out = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return out | Qt.ItemIsUserCheckable if index.column() == 2 else out

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self.ids):
            return None
        roi_id = self.ids[index.row()]
        project = self.window.project
        roi = project.rois.get(roi_id)
        if roi is None:
            return None
        column = index.column()
        if column == 2:
            if role == Qt.CheckStateRole:
                return Qt.Checked if roi.use else Qt.Unchecked
            if role == Qt.ToolTipRole:
                return "include this ROI when the per-region steps run (space toggles it)"
            return None
        if role == Qt.DisplayRole:
            return str(self.numbers.get(roi_id, "")) if column == 0 else str(
                project.file_number(roi.file_id))
        if role == Qt.ForegroundRole:
            if not roi.use:
                return pg.mkColor("#8a8a8a")
            if self.waiting(roi_id):
                return pg.mkColor(STALE_COLOUR)
        if role == Qt.ToolTipRole and roi.use and self.waiting(roi_id):
            return ("the data or the steps' parameters have changed since "
                    "this ROI was measured")
        return None

    def setData(self, index, value, role=Qt.EditRole) -> bool:  # noqa: N802
        if role != Qt.CheckStateRole or index.column() != 2:
            return False
        self.window._set_use(self.ids[index.row()], Qt.CheckState(value) == Qt.Checked)
        return True


class ROIManagerWindow(QMainWindow):
    changed = Signal()          # ROIs added, removed or toggled
    evaluation_requested = Signal()   # the pipeline window, opened by the Regions tab

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        from .help_window import attach_help
        attach_help(self, 'Panels/ROI manager')
        self.setWindowFlag(Qt.Window, True)      # a window, not a child widget
        self.setWindowTitle("SMAPpy ROI manager")
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
        # the window the evaluators' figures are drawn in, kept so that
        # walking down the list redraws them in place instead of burying the
        # screen; made on first use
        self._site = None
        self._evaluating = False
        self._reaches: dict = {}     # ROI id -> (its polygon, how far it reaches)

        self.file_pane = ImagePane("file")
        self.zoom_pane = ImagePane("zoom", zoomable=True, pannable=True)
        self.roi_pane = ImagePane("ROI", zoomable=True)

        # tiles: a grid over the whole file, walked one square at a time, so a
        # dataset can be gone through systematically instead of by eye
        file_column = QWidget()
        flayout = QVBoxLayout(file_column)
        flayout.setContentsMargins(0, 0, 0, 0)
        flayout.addWidget(self.file_pane, 1)
        trow = QHBoxLayout()
        trow.setSpacing(3)
        self.tiles_button = QPushButton("Tiles")
        self.tiles_button.setCheckable(True)
        self.tiles_button.setToolTip("cover the file with a grid; click a tile's edge "
                                     "to look at it, the arrow keys walk through them")
        self.tiles_button.clicked.connect(self._on_tiles)
        self.tile_size = QDoubleSpinBox(minimum=100, maximum=1e7, decimals=0, suffix=" nm")
        self.tile_size.setSingleStep(1000)
        self.tile_size.setKeyboardTracking(False)
        self.tile_size.setToolTip("the side of one tile")
        self.tile_size.valueChanged.connect(self._on_tile_size)
        self.tile_label = QLabel("")
        self.tile_label.setStyleSheet("color: gray")
        self.previous_tile = QPushButton("<")
        self.previous_tile.setToolTip("the tile before this one (left arrow)")
        self.previous_tile.clicked.connect(lambda: self._step_tile(-1))
        self.next_tile = QPushButton(">")
        self.next_tile.setToolTip("the tile after this one (right arrow)")
        self.next_tile.clicked.connect(lambda: self._step_tile(1))
        for w in (self.tiles_button, self.tile_size, self.previous_tile,
                  self.next_tile, self.tile_label):
            trow.addWidget(w)
        trow.addStretch(1)
        flayout.addLayout(trow)

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
        self.table = QTableView()
        self.rows = ROIList(self)
        self.table.setModel(self.rows)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        # sized once, not to the contents: that would read every row
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setStretchLastSection(False)
        for column, width in enumerate((60, 40, 40)):
            header.resizeSection(column, width)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._on_select())
        self.table.setMaximumWidth(190)          # the numbers need no more
        roi_side_list.addWidget(self.table, 1)
        side.addLayout(roi_side_list, 0)
        llayout.addLayout(side, 1)
        edits = QHBoxLayout()
        edits.setSpacing(3)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("remove the selected ROI")
        self.remove_button.clicked.connect(self._remove)
        # the ROIs' own undo (`ROIProject.history`), not the main window's,
        # which is for tables: Ctrl+Z here never undoes a drift correction.
        # The buttons name the edit ("Undo remove ROI 3"), as Edit > Undo
        # names the table's step ("Undo RCC"), so the two read differently
        self.undo_button = QPushButton("Undo ROI edit")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QPushButton("Redo ROI edit")
        self.redo_button.clicked.connect(self.redo)
        edits.addWidget(self.remove_button, 1)
        edits.addWidget(self.undo_button)
        edits.addWidget(self.redo_button)
        llayout.addLayout(edits)
        # shown only while a segmenter's preview is drawn: Run adds it, a
        # new preview replaces it, and this is the way to have neither
        self.clear_preview_button = QPushButton("Discard preview")
        self.clear_preview_button.setToolTip("remove the grey ROIs a segmenter's "
                                             "Preview proposed, adding none of them")
        self.clear_preview_button.clicked.connect(self.session.clear_preview)
        self.clear_preview_button.setVisible(False)
        llayout.addWidget(self.clear_preview_button)
        # the pipeline is set up in the control window's Regions tab, but it is
        # here that its results are looked at, one site after another; the
        # same window, opened from where the eye already is
        self.evaluation_button = QPushButton("Per-region steps...")
        self.evaluation_button.setToolTip("the steps run once in every ROI: choose "
                                          "them, order them and set them up")
        self.evaluation_button.clicked.connect(self.evaluation_requested.emit)
        llayout.addWidget(self.evaluation_button)
        self.evaluate_live = QCheckBox("plot the ROI's figures")
        self.evaluate_live.setToolTip(
            "open the ROI results window: what the steps draw for the selected "
            "ROI, a tab per step, redrawn in place by the next ROI -- so "
            "that stepping down the list compares like with like.  Only the "
            "tab being looked at is drawn, and only that step runs.")
        self.evaluate_live.toggled.connect(self._on_evaluate_live)
        llayout.addWidget(self.evaluate_live)
        self.re_evaluate = QCheckBox("measure again when out of date")
        self.re_evaluate.setChecked(True)
        self.re_evaluate.setToolTip(
            "a selected ROI whose stored numbers were measured with data or "
            "parameters that have since changed is measured again, and the "
            "result kept.  Turn it off to scroll a large project without it "
            "computing behind you: the numbers are then shown as they were "
            "stored, marked for what they are.")
        self.re_evaluate.toggled.connect(self._on_re_evaluate)
        llayout.addWidget(self.re_evaluate)
        # the platform's keys, and Ctrl+Shift+Z everywhere, as the main window
        # has them; once each, since two shortcuts on one key fire neither
        keys = [(QKeySequence(QKeySequence.Undo), self.undo),
                (QKeySequence(QKeySequence.Redo), self.redo)]
        if QKeySequence("Ctrl+Shift+Z") not in (k for k, _ in keys):
            keys.append((QKeySequence("Ctrl+Shift+Z"), self.redo))
        for sequence, slot in keys:
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)
        toggle = QShortcut(QKeySequence(Qt.Key_Space), self)
        toggle.setContext(Qt.WindowShortcut)
        toggle.activated.connect(self._toggle)
        cancel = QShortcut(QKeySequence(Qt.Key_Escape), self)
        cancel.setContext(Qt.WindowShortcut)
        cancel.activated.connect(self._cancel_drawing)
        for key, step in ((Qt.Key_Right, 1), (Qt.Key_Left, -1)):
            walk = QShortcut(QKeySequence(key), self)
            walk.setContext(Qt.WindowShortcut)
            walk.activated.connect(lambda step=step: self._step_tile(step))

        roi_side = QWidget()
        rlayout = QVBoxLayout(roi_side)
        rlayout.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        tools.setSpacing(3)
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setCheckable(True)
        self.polygon_button.setToolTip("draw an outline in this image: a click per "
                                       "vertex, a click on the first vertex closes it, "
                                       "Escape cancels")
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
        # the tools that say what to draw sit above the image they draw into;
        # *Add*, the one button that is pressed over and over, sits directly
        # under it, where the eye already is after judging the ROI
        rlayout.addLayout(tools)
        rlayout.addWidget(self.roi_pane, 1)
        self.add_button = QPushButton("Add")
        self.add_button.setToolTip("store the drafted ROI (Enter)")
        self.add_button.setShortcut(Qt.Key_Return)
        self.add_button.setDefault(True)
        self.add_button.setStyleSheet(
            "QPushButton { background: #2e9e4f; color: white; font-weight: bold;"
            " border: 1px solid #24783c; border-radius: 3px; padding: 5px; }"
            "QPushButton:pressed { background: #24783c; }"
            "QPushButton:disabled { background: #8a8a8a; border-color: #6e6e6e; }")
        self.add_button.clicked.connect(self._add)
        rlayout.addWidget(self.add_button)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(file_column)
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
        elif what == "roi-preview":
            self._fill_table()           # the status line counts the preview
            self.redraw()
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
        self.tiles_button.setChecked(bool(self.project.tile_nm))
        self.tile_size.setValue(self.project.tile_nm or DEFAULT_TILE_NM)
        self.evaluate_live.setChecked(bool(project.navigation.get("plot_evaluation")))
        self.re_evaluate.setChecked(bool(project.navigation.get("re_evaluate", True)))
        self._loading = False
        self.redraw()

    def _fill_table(self) -> None:
        project = self.project
        was_loading, self._loading = self._loading, True
        self.rows.reset(project)
        row = self.rows.row_of.get(self.active)
        if row is not None:
            self.table.selectRow(row)
            self.table.scrollTo(self.rows.index(row, 0))
        self._loading = was_loading
        used = sum(1 for r in project.rois.values() if r.use)
        waiting = []
        if len(self.rows.ids) <= COUNT_LIMIT:
            waiting = [i for i in self.rows.ids if self.rows.waiting(i)]
        preview = project.segment_preview
        self.clear_preview_button.setVisible(preview is not None)
        proposed = (f"; previewing {len(preview['sites'])} (grey): Run adds them"
                    if preview is not None else "")
        self.status.showMessage(f"{len(self.rows.ids)} ROIs, {used} used"
                                + (f", {len(waiting)} to measure" if waiting else "")
                                + proposed)

    # ------------------------------------------------------------ drawing
    def redraw(self) -> None:
        project = self.project
        self._update_undo()
        file_id = self.current_file()
        if file_id is None:
            for pane in (self.file_pane, self.zoom_pane, self.roi_pane):
                pane.clear()
            return
        state = project.state(file_id)
        x0, y0, x1, y1 = state.index.bounds
        self.file_pane.render(state, x0, x1, y0, y1)
        rois = project.rois_of(file_id)
        preview = project.segment_preview
        if preview is not None and preview.get("file_id") == file_id:
            # a replacing run removes these first, so the preview hides them
            replaced = set(preview.get("replaces") or ())
            rois = [roi for roi in rois if roi.id not in replaced]
        else:
            preview = None

        # the file: every ROI, small, over the tile grid
        self._draw_outlines(self.file_pane, rois, preview)
        self._draw_tiles(file_id)

        centre = self.zoom_center
        if centre is None:
            centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            self.zoom_center = centre
        half = self.zoom_pane.width_nm / 2
        self.zoom_pane.render(state, centre[0] - half, centre[0] + half,
                              centre[1] - half, centre[1] + half)
        self._draw_outlines(self.zoom_pane, rois, preview)
        self.zoom_pane.draw("draft", *([[], []] if self.draft is None else
                                       Region.site(self.draft, project.shape, project.size_nm).outline()))
        # the frame the zoom covers, on the file image
        self.file_pane.draw("frame",
                            [centre[0] - half, centre[0] + half, centre[0] + half,
                             centre[0] - half, centre[0] - half],
                            [centre[1] - half, centre[1] - half, centre[1] + half,
                             centre[1] + half, centre[1] - half])
        self._draw_roi_pane(state)

    # ----------------------------------------------------------------- tiles
    @property
    def tile(self) -> Optional[int]:
        """Which tile the zoom is on, or None."""
        index = self.project.navigation.get("tile")
        return int(index) if isinstance(index, (int, float)) else None

    def _draw_tiles(self, file_id) -> None:
        """The grid, and the one square being looked at, on the file image."""
        tiles = self.project.tiles(file_id)
        xs, ys = [], []
        for x0, y0, x1, y1 in tiles:
            xs.extend([x0, x1, x1, x0, x0, np.nan])
            ys.extend([y0, y0, y1, y1, y0, np.nan])
        self.file_pane.draw("tiles", xs, ys)
        here = self.tile
        if tiles and here is not None and 0 <= here < len(tiles):
            x0, y0, x1, y1 = tiles[here]
            box = ([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
            self.file_pane.draw("tile", *box)
            # and in the zoom, which is wider than the tile where the pane is:
            # the outline says where the square being walked actually ends
            self.zoom_pane.draw("tile", *box)
            self.tile_label.setText(f"tile {here + 1} / {len(tiles)}")
        else:
            self.file_pane.draw("tile")
            self.zoom_pane.draw("tile")
            self.tile_label.setText(f"{len(tiles)} tiles" if tiles else "")
        for w in (self.previous_tile, self.next_tile):
            w.setEnabled(bool(tiles))

    def _on_tiles(self, on: bool) -> None:
        """The button: lay a grid over the file, or take it away."""
        self.project.set_tiles(self.tile_size.value() if on else 0.0)
        self.project.navigation.pop("tile", None)
        if on:
            self._go_to_tile(0)
        else:
            self.redraw()

    def _on_tile_size(self) -> None:
        if self._loading or not self.tiles_button.isChecked():
            return
        self.project.set_tiles(self.tile_size.value())
        self._go_to_tile(min(self.tile or 0, max(0, len(self.project.tiles(
            self.current_file())) - 1)))

    def _step_tile(self, step: int) -> None:
        """The arrow keys: the next tile, wrapping at the end of the file."""
        tiles = self.project.tiles(self.current_file())
        if not tiles:
            return
        here = self.tile
        self._go_to_tile(0 if here is None else (here + step) % len(tiles))

    def _go_to_tile(self, index: int) -> None:
        """Put the zoom on one tile: it fills the pane, and the draft is dropped."""
        tiles = self.project.tiles(self.current_file())
        if not tiles:
            self.redraw()
            return
        index = int(np.clip(index, 0, len(tiles) - 1))
        x0, y0, x1, y1 = tiles[index]
        self.project.navigation["tile"] = index
        self.zoom_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.zoom_pane.width_nm = max(x1 - x0, y1 - y0)
        self.draft = self.draft_polygon = self.draft_direction = None
        self.redraw()
        self.status.showMessage(f"tile {index + 1} of {len(tiles)}")

    def _draw_outlines(self, pane, rois, preview=None) -> None:
        """Every ROI on one pane: grey when it is not used, green when
        selected; and a segmenter's preview, dashed."""
        project = self.project
        for name, use in (("preview", True), ("preview_rejected", False)):
            xs, ys = [], []
            for site in (preview or {}).get("sites", ()):
                if bool(site.get("use", True)) == use:
                    ox, oy = Region.site(site["center"], project.shape, project.size_nm).outline()
                    xs.extend(list(ox) + [np.nan])
                    ys.extend(list(oy) + [np.nan])
            pane.draw(name, xs, ys)
        rois = self._in_view(pane, rois)
        if len(rois) > OUTLINE_LIMIT:
            # a dot at each centre; the selected ROI keeps its outline
            pane.note(f"{len(rois)} ROIs in view: their centres; zoom in for outlines")
            for name, use in (("centres", True), ("centres_excluded", False)):
                centres = np.array([roi.center for roi in rois if bool(roi.use) == use
                                    and roi.id != self.active], float).reshape(-1, 2)
                pane.draw(name, centres[:, 0], centres[:, 1])
            rois = [roi for roi in rois if roi.id == self.active]
        else:
            pane.note()
            pane.draw("centres")
            pane.draw("centres_excluded")
        groups = {"others": [], "excluded": [], "excluded_active": [], "roi": []}
        for roi in rois:
            active = roi.id == self.active
            name = ("excluded_active" if active else "excluded") if not roi.use else (
                "roi" if active else "others")
            groups[name].append(roi)
        for name, chosen in groups.items():
            xs, ys = [], []
            for roi in chosen:
                ox, oy = project.region(roi).outline()
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

    def _reach(self, roi) -> float:
        """How far an ROI's outline reaches from its centre: the project's
        size, or its polygon's -- kept until the polygon is replaced."""
        if roi.polygon is None:
            return float(self.project.size_nm)
        cache = self._reaches
        found = cache.get(roi.id)
        if found is None or found[0] is not roi.polygon:
            v = np.asarray(roi.polygon, float)
            found = (roi.polygon, float(np.hypot(*(np.abs(v - roi.center).max(axis=0)))))
            cache[roi.id] = found
        return found[1]

    def _near(self, rois, x0: float, x1: float, y0: float, y1: float) -> list:
        """The ROIs whose outline may reach into the box."""
        if not rois:
            return []
        centres = np.array([roi.center for roi in rois], float).reshape(-1, 2)
        reach = np.array([self._reach(roi) for roi in rois])
        hit = ((centres[:, 0] + reach >= x0) & (centres[:, 0] - reach <= x1)
               & (centres[:, 1] + reach >= y0) & (centres[:, 1] - reach <= y1))
        return [roi for roi, h in zip(rois, hit) if h]

    def _in_view(self, pane, rois) -> list:
        fov = pane.fov
        if fov is None:
            return list(rois)
        return self._near(list(rois), fov.x0, fov.x1, fov.y0, fov.y1)

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
        xs, ys = Region.site(centre, project.shape, project.size_nm, polygon).outline()
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
        self.project.navigation.pop("tile", None)
        if self.project.tile_nm:     # a new file: start its walk at the first tile
            self._go_to_tile(0)
        else:
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
        """Centre the zoom here -- or, on a tile's edge, go to that tile.

        Only the edge counts, so a click inside a tile still centres freely,
        the way it does without a grid.
        """
        if np.isnan(x):
            self.redraw()
            return
        index = self._tile_edge_at(x, y)
        if index is not None:
            self._go_to_tile(index)
            return
        self.zoom_center = np.array([x, y])
        self.redraw()

    def _tile_edge_at(self, x: float, y: float) -> Optional[int]:
        """The tile whose outline the click landed on, within a few pixels."""
        file_id = self.current_file()
        index = self.project.tile_at(file_id, x, y)
        if index is None:
            return None
        x0, y0, x1, y1 = self.project.tiles(file_id)[index]
        fov = self.file_pane.fov
        tolerance = RIM_PIXELS * (fov.pixelsize if fov is not None else 1.0)
        near = (min(abs(x - x0), abs(x - x1)) <= tolerance
                or min(abs(y - y0), abs(y - y1)) <= tolerance)
        return index if near else None

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
        self.status.showMessage("a click per vertex; a click on the first vertex "
                                "closes it, Escape cancels" if kind == "polygon"
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

    # ---------------------------------------------------------------- copy
    def clipboard_image(self):
        """What Edit > Copy takes from here: the pane under the mouse, else
        the ROI's own -- from the menu, the mouse is in another window."""
        from PySide6.QtGui import QCursor
        panes = (self.roi_pane, self.zoom_pane, self.file_pane)
        under = QApplication.widgetAt(QCursor.pos())
        pane = next((p for p in panes if under is not None
                     and (under is p or p.isAncestorOf(under))), self.roi_pane)
        return pane.clipboard_image()

    # ---------------------------------------------------------------- undo
    def _update_undo(self) -> None:
        history = self.project.history
        for button, can, label, what in (
                (self.undo_button, history.can_undo, history.undo_label, "undo"),
                (self.redo_button, history.can_redo, history.redo_label, "redo")):
            button.setEnabled(can)
            button.setToolTip(f"{what}: {label}" if can else f"nothing to {what}")
            text = f"{what.capitalize()} {_lowered(label) if can else 'ROI edit'}"
            # elided: a long label must not widen the column of the ROI list
            button.setText(button.fontMetrics().elidedText(
                text, Qt.ElideRight, UNDO_TEXT_WIDTH))

    def _edit(self, what: str):
        """An undoable edit, named after the selected ROI's number."""
        roi = self._selected()
        number = self.project.numbers().get(roi.id) if roi is not None else None
        return self.project.edit(f"{what} ROI {number}" if number else what)

    def undo(self) -> None:
        self._travel(self.project.undo, "undid")

    def redo(self) -> None:
        self._travel(self.project.redo, "redid")

    def _travel(self, step, verb: str) -> None:
        label = step()
        if label is None:
            return
        if self.active not in self.project.rois:
            self.active = None
        # the Regions tab counts them too, and its listener is the session's
        self.session.changed("rois")
        self.status.showMessage(f"{verb} {label}")

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
            with self._edit("Outline"):
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
            with self._edit("Direction of"):
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
            with self._edit("Clear shape of"):
                roi.polygon = None
            self.changed.emit()
        self.redraw()

    def _clear_line(self) -> None:
        self._cancel_drawing()
        if self.draft is not None:
            self.draft_direction = None
        roi = self._selected()
        if roi is not None and self.draft is None:
            with self._edit("Clear line of"):
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
        near = self._near(project.rois_of(self.current_file()), x - tolerance,
                          x + tolerance, y - tolerance, y + tolerance)
        for roi in near:
            d = project.region(roi).rim_distance(x, y)
            if d <= tolerance and d < best_d:
                best, best_d = roi, d
        return best

    def _set_use(self, roi_id, use: bool) -> None:
        """The use box in the table."""
        roi = self.project.rois.get(roi_id)
        if self._loading or roi is None:
            return
        number = self.project.numbers().get(roi.id)
        with self.project.edit(f"{'Include' if use else 'Exclude'} ROI {number}"):
            roi.use = use
        self._fill_table()
        self.redraw()
        self.changed.emit()

    def _on_select(self) -> None:
        if self._loading:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.rows.ids):
            return
        self.select(self.rows.ids[rows[0].row()], from_table=True)

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
        if self.evaluate_live.isChecked():
            self.evaluate_active()

    # ---------------------------------------------------- the evaluation
    def _on_evaluate_live(self, on: bool) -> None:
        if self._loading:            # restored with the rest of the window
            return
        self.project.navigation["plot_evaluation"] = bool(on)
        if on:
            self.site_window().show()
            self.evaluate_active()
        elif self._site is not None:
            self._site.hide()

    def _on_re_evaluate(self, on: bool) -> None:
        if self._loading:
            return
        self.project.navigation["re_evaluate"] = bool(on)
        if on and self.evaluate_live.isChecked():
            self.evaluate_active()

    def site_window(self):
        """The window the selected ROI's figures are drawn in, made once."""
        from .roi_site import SiteWindow
        if self._site is None:
            self._site = SiteWindow(self)
            # switching evaluator asks for that one: until it is looked at,
            # it has not been run and there is nothing to draw
            self._site.shown.connect(lambda label: self.evaluate_active(label))
            # closed by hand: measuring for a window nobody is looking at is
            # the cost this whole arrangement exists to avoid
            self._site.closed.connect(lambda: self.evaluate_live.setChecked(False))
        return self._site

    def evaluate_active(self, label: Optional[str] = None) -> None:
        """Measure the ROI being looked at, and draw what its evaluator draws.

        On this thread and on one step of one site: an evaluator sees a few
        hundred localizations, and a figure that arrives after the next ROI
        has been selected would be worse than a moment's wait.

        What runs is decided here and nowhere else.  The evaluator whose tab
        is open runs, because a plot is a closure over the data it drew and
        nothing stored brings it back; the others are run only when their
        stored result is out of date and *re-evaluate* is on, and what runs
        is kept, so the same work is not done again on the next visit.
        """
        from ..roi_manager import pipeline as pipeline_module
        project = self.project
        roi = self._selected()
        if roi is None or self._evaluating or not self.evaluate_live.isChecked():
            return
        steps = project.block_steps()
        if not steps:
            self.status.showMessage("no per-region step is enabled")
            return
        window = self.site_window()
        window.set_steps([step.label for step in steps])
        label = label or window.current
        number = project.numbers().get(roi.id, "")
        window.title_for(f"ROI {number}")
        self._evaluating = True
        try:
            record, results, updated = project.evaluate_one(
                roi.id, steps=steps, reuse=True, force=[label] if label else [],
                store=True, run="stale" if self.re_evaluate.isChecked() else "forced")
        except Exception as error:
            self.status.showMessage(f"the measurement failed: {error}")
            return
        finally:
            self._evaluating = False
        failed = pipeline_module.errors(record)
        for step in steps:
            result = results.get(step.label)
            if step.label in failed:
                window.show_message(step.label, failed[step.label], failed=True)
            elif result is not None and result.figures():
                window.show_plots(step.label, result.figures(), f"ROI {number}")
            elif result is not None:
                window.show_message(step.label, "this step draws nothing; "
                                                "its numbers are in the Regions tab")
        # the table first: it writes the ROI counts into the status bar, and
        # what just happened to this site is the more useful of the two
        self._fill_table()
        drew = [label for label, result in results.items() if result is not None]
        if updated:
            self.status.showMessage(f"ROI {number}: {', '.join(updated)} measured")
        elif drew:
            self.status.showMessage(f"ROI {number}: {', '.join(drew)} drawn from "
                                    "the result already stored")
        else:
            self.status.showMessage(f"ROI {number}: nothing to draw")

    # --------------------------------------------------------------- edits
    def _add(self) -> None:
        file_id = self.current_file()
        if file_id is None or self.draft is None:
            return
        with self.project.edit("Add ROI") as step:
            roi = self.project.add_roi(file_id, [float(self.draft[0]), float(self.draft[1])],
                                       polygon=self.draft_polygon)
            roi.direction = self.draft_direction
            step.label = f"Add ROI {self.project.numbers().get(roi.id)}"
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
            with self._edit("Include" if not roi.use else "Exclude"):
                roi.use = not roi.use
            self._fill_table()
            self.redraw()
            self.changed.emit()

    def _remove(self) -> None:
        roi = self._selected()
        if roi is not None:
            with self._edit("Remove"):
                self.project.rois.pop(roi.id, None)
            self.active = None
            self._fill_table()
            self.redraw()
            self.changed.emit()
