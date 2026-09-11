"""The image: a pyqtgraph view that re-renders whenever the view moves.

The render grid follows the widget, so a render costs the same at any zoom
(`FieldOfView.fit`), and the image item is placed in data coordinates so that
pyqtgraph's pan/zoom acts on the localization coordinates directly.
"""
from __future__ import annotations

import atexit
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QImage
from PySide6.QtWidgets import (QFileDialog, QGraphicsPathItem, QInputDialog, QLabel,
                               QMenu, QToolBar, QToolButton, QVBoxLayout, QWidget)

from ..regions import Region
from ..render import FieldOfView
from ..session import Session

TILE = 1.5          # render this many view widths, so a pan needs no render
_LIVE_THREADS = []  # every render thread, so exit can stop them all


def stop_render_threads() -> None:
    """Stop every render thread.  A QThread destroyed while it runs aborts the
    process, so this runs on aboutToQuit *and* at interpreter exit."""
    while _LIVE_THREADS:
        thread = _LIVE_THREADS.pop()
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(2000)
        except RuntimeError:                 # already gone
            pass


atexit.register(stop_render_threads)
# A thin yellow line disappears over a bright render, so every ROI outline is
# drawn twice: a broad dark line first, the yellow one on top of it.  The pens
# are cosmetic (pyqtgraph's default), so the widths are screen pixels and the
# outline stays equally readable at any zoom.
ROI_PEN = pg.mkPen((255, 255, 0), width=2)
ROI_HALO_PEN = pg.mkPen((0, 0, 0, 200), width=5)
DRAW_PEN = pg.mkPen((255, 255, 0), width=2, style=Qt.DashLine)
DRAW_HALO_PEN = pg.mkPen((0, 0, 0, 200), width=5)


class _Renderer(QObject):
    """Renders on its own thread; the view keeps answering to the mouse."""

    done = Signal(object, object, int)      # rgb, fov, generation

    def __init__(self):
        super().__init__()
        self.thread = QThread()
        self.thread.setStackSize(32 * 1024 * 1024)
        self.moveToThread(self.thread)
        self.thread.start()
        _LIVE_THREADS.append(self.thread)

    def render(self, view: "RenderView", fov: FieldOfView, generation: int) -> None:
        try:
            rgb, _ = view.composite(fov)
        except Exception as e:           # a table swapped mid-render: try again
            print(f"render failed: {e}")
            return
        self.done.emit(rgb, fov, generation)


class RenderView(QWidget):
    render_requested = Signal(object, object, int)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.graphics = pg.GraphicsLayoutWidget()
        self.view = self.graphics.addViewBox(lockAspect=True, invertY=True,
                                            enableMenu=False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image)
        self.scalebar = pg.ScaleBar(size=1000, suffix="nm")
        self.scalebar.setParentItem(self.view)
        self.scalebar.anchor((1, 1), (1, 1), offset=(-20, -20))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

        # pan/zoom fires many range changes; render once they settle
        self._timer = QTimer(singleShot=True, interval=40)
        self._timer.timeout.connect(self.render)
        self.view.sigRangeChanged.connect(self.schedule)
        self.graphics.viewport().installEventFilter(self)     # the trackpad's pinch
        session.on_change(self._on_session)
        self.fov = None                  # the field of view of the shown tile
        self.tile_valid = False
        self._generation = 0             # the newest request; older results are dropped
        self._busy = False
        self._pending = False
        self._renderer = _Renderer()
        self.render_requested.connect(self._renderer.render)
        self._renderer.done.connect(self._on_rendered)
        app = pg.QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

        # ROI drawing: click to start, click to finish (polygon: click per
        # vertex, double-click to close), Escape to cancel
        self.roi_item = None
        self.drawing: Optional[str] = None
        self.line_width = 100.0
        self._points: List[QPointF] = []
        self._preview_halo = pg.PlotDataItem(pen=DRAW_HALO_PEN)
        self._preview = pg.PlotDataItem(pen=DRAW_PEN)
        self.roi_halo: Optional[QGraphicsPathItem] = None
        for item in (self._preview_halo, self._preview):
            self.view.addItem(item)
        self.graphics.scene().sigMouseMoved.connect(self._on_move)
        self._pressed = False
        self.graphics.setFocusPolicy(Qt.StrongFocus)
        self.graphics.keyPressEvent = self._on_key
        self.reset()

    def eventFilter(self, obj, event) -> bool:
        if self.drawing and event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                                             QEvent.MouseButtonDblClick):
            return self._draw_event(event)
        if event.type() == QEvent.NativeGesture and \
                event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            factor = 1.0 / (1.0 + event.value())
            centre = self.view.mapSceneToView(self.graphics.mapToScene(event.position().toPoint()))
            self.view.scaleBy((factor, factor), centre)
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def shutdown(self) -> None:
        """Stop the render thread; called on the GUI thread at exit."""
        stop_render_threads()

    # ---------------------------------------------------------------- flow
    def _on_session(self, what: str) -> None:
        if what == "locs":
            self.tile_valid = False
            self.reset()
        elif what in ("layer", "layers", "append", "regrouped"):
            self.tile_valid = False      # the picture changed, not just the view
            self.schedule()
        elif what == "roi":
            self._show_roi(self.session.roi)

    def schedule(self) -> None:
        self._timer.start()

    def reset(self) -> None:
        """Show everything."""
        if not self._has_content():
            self.image.clear()
            return
        (x0, x1), (y0, y1) = self.session.full_view()
        self.view.setRange(QRectF(x0, y0, x1 - x0, y1 - y0), padding=0)
        self.schedule()

    def _has_content(self) -> bool:
        if any(l.is_image for l in self.session.layers) or len(self.session.locs):
            return True
        state = self.session.layers[0].state
        return state is not None and bool(getattr(state.index, "extent", None))

    def composite(self, fov: FieldOfView) -> Tuple[np.ndarray, np.ndarray]:
        """Every visible layer rendered on ``fov`` and added up, as SMAP does.

        Returns the RGB image in [0, 1] and the summed intensity planes.
        """
        rgb = np.zeros((fov.ny, fov.nx, 3), np.float32)
        weight = np.zeros((fov.ny, fov.nx), np.float32)
        for layer in self.session.layers:
            if layer.visible:
                image, rendered = layer.render(fov)
                rgb += image
                weight += rendered.weight
        return np.clip(rgb, 0, 1), weight

    def current_fov(self, nx: int = None, ny: int = None) -> FieldOfView:
        rect = self.view.viewRect()
        size = self.graphics.size()
        return FieldOfView.fit((rect.left(), rect.right()), (rect.top(), rect.bottom()),
                               nx or max(size.width(), 16), ny or max(size.height(), 16))

    def center_on(self, x: float, y: float) -> None:
        """Move the view to (x, y) at the current zoom."""
        rect = self.view.viewRect()
        rect.moveCenter(rect.center().__class__(x, y))
        self.view.setRange(rect, padding=0)

    def _covered(self, view: FieldOfView) -> bool:
        """Does the shown tile cover this view at (nearly) this pixel size?"""
        t = self.fov
        if t is None or not self.tile_valid:
            return False
        return (abs(t.pixelsize / view.pixelsize - 1) < 0.02
                and t.x0 <= view.x0 and t.y0 <= view.y0
                and t.x1 >= view.x1 and t.y1 >= view.y1)

    def render(self) -> None:
        """Ask for a tile around the current view; nothing here blocks.

        The tile is TILE view widths across at the view's pixel size, so a
        pan inside it needs no render, and the last tile stays on screen --
        scaled by the view -- until the new one arrives.
        """
        if not self._has_content():
            self.image.clear()
            self.fov = None
            return
        view = self.current_fov()
        self._update_scalebar(view)
        if self._covered(view):
            return
        if self._busy:                   # one at a time; the newest view wins
            self._pending = True
            return
        w, h = view.x1 - view.x0, view.y1 - view.y0
        tile = FieldOfView.fit((view.x0 - (TILE - 1) / 2 * w, view.x1 + (TILE - 1) / 2 * w),
                               (view.y0 - (TILE - 1) / 2 * h, view.y1 + (TILE - 1) / 2 * h),
                               int(view.nx * TILE), int(view.ny * TILE))
        self._generation += 1
        self._busy = True
        self.render_requested.emit(self, tile, self._generation)

    def _on_rendered(self, rgb, fov: FieldOfView, generation: int) -> None:
        self._busy = False
        if generation != self._generation:      # superseded while it ran
            self._pending = True
        else:
            rgb = np.ascontiguousarray(rgb)
            self.image.setImage(rgb, levels=[0, 1] if rgb.dtype.kind == "f" else None,
                                autoLevels=False)
            self.image.setRect(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0))
            self.fov = fov
            self.tile_valid = True
        if self._pending:
            self._pending = False
            self.render()

    def render_now(self) -> np.ndarray:
        """Synchronous: the exact current view, for saving and tests."""
        fov = self.current_fov()
        rgb, _ = self.composite(fov)
        return np.ascontiguousarray(rgb)

    def _update_scalebar(self, fov: FieldOfView) -> None:
        self.scalebar.size = _nice(0.2 * (fov.x1 - fov.x0))
        size = self.scalebar.size
        self.scalebar.text.setText(f"{size / 1000:g} µm" if size >= 1000 else f"{size:g} nm")
        self.scalebar.updateBar()

    # ---------------------------------------------------------------- roi
    def start_drawing(self, kind: str) -> None:
        """Begin a new ROI of ``kind``; the old one goes."""
        self.clear_roi()
        self.drawing = kind
        self._points = []
        self.view.setMouseEnabled(False, False)
        self.graphics.setCursor(Qt.CrossCursor)
        self.graphics.setFocus()

    def clear_roi(self) -> None:
        if self.roi_item is not None:
            self.view.removeItem(self.roi_item)
            self.roi_item = None
        if self.session.roi is not None:
            self.session.set_roi(None)

    def _stop_drawing(self) -> None:
        self.drawing = None
        self._points = []
        self._preview.setData([], [])
        self._preview_halo.setData([], [])
        self.view.setMouseEnabled(True, True)
        self.graphics.unsetCursor()

    def _on_key(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.drawing:
            self._stop_drawing()
        else:
            pg.GraphicsLayoutWidget.keyPressEvent(self.graphics, event)

    def _draw_event(self, event) -> bool:
        """Rectangle and line: press, drag, release.  Polygon: a press per
        vertex, a double-click (or right button) closes it."""
        pos = self.view.mapSceneToView(self.graphics.mapToScene(event.position().toPoint()))
        kind = event.type()
        if kind == QEvent.MouseButtonDblClick or event.button() == Qt.RightButton:
            if self.drawing == "polygon" and len(self._points) >= 3:
                self._finish(self._points)
            return True
        if kind == QEvent.MouseButtonPress:
            self._pressed = True
            if self.drawing == "polygon":
                self._points.append(pos)
            else:
                self._points = [pos]
            return True
        if kind == QEvent.MouseButtonRelease and self._pressed:
            self._pressed = False
            if self.drawing in ("rect", "line") and self._points:
                start = self._points[0]
                if (pos - start).manhattanLength() > 0:        # a drag, not a tap
                    self._finish([start, pos])
                else:
                    self._points = []
            return True
        return True

    def _on_move(self, scene_pos) -> None:
        if not self.drawing or not self._points:
            return
        pos = self.view.mapSceneToView(scene_pos)
        pts = self._points + [pos]
        if self.drawing == "rect":
            (x0, y0), (x1, y1) = (pts[0].x(), pts[0].y()), (pos.x(), pos.y())
            xs, ys = [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]
        else:
            xs, ys = [p.x() for p in pts], [p.y() for p in pts]
        self._preview_halo.setData(xs, ys)
        self._preview.setData(xs, ys)

    def _finish(self, pts: List[QPointF]) -> None:
        kind = self.drawing
        self._stop_drawing()
        if kind == "rect":
            region = Region.rect(pts[0].x(), pts[0].y(), pts[1].x(), pts[1].y())
        elif kind == "line":
            region = Region.line((pts[0].x(), pts[0].y()), (pts[1].x(), pts[1].y()),
                                 self.line_width)
        else:
            region = Region("polygon", [(p.x(), p.y()) for p in pts])
        self._show_roi(region)
        self.session.set_roi(region)

    def _show_roi(self, region: Optional[Region]) -> None:
        """An editable pyqtgraph ROI for the region; edits go back to the session."""
        if self.roi_item is not None:
            self.view.removeItem(self.roi_item)
            self.roi_item = None
        if self.roi_halo is not None:
            self.view.removeItem(self.roi_halo)
            self.roi_halo = None
        if region is None:
            return
        if region.kind == "rect":
            x0, y0, x1, y1 = region.bounds
            item = pg.RectROI([x0, y0], [x1 - x0, y1 - y0], pen=ROI_PEN)
            item.addScaleHandle([0, 0], [1, 1])
        elif region.kind == "line":
            p = region.points                      # corners: p0+n, p1+n, p1-n, p0-n
            start, end = (p[0] + p[3]) / 2, (p[1] + p[2]) / 2
            item = pg.LineROI(start, end, region.width, pen=ROI_PEN)
        else:
            item = pg.PolyLineROI(region.points.tolist(), closed=True, pen=ROI_PEN)
        item.sigRegionChangeFinished.connect(self._roi_edited)
        self.roi_halo = QGraphicsPathItem()
        self.roi_halo.setPen(ROI_HALO_PEN)
        self.roi_halo.setBrush(Qt.NoBrush)
        self.roi_halo.setZValue(item.zValue() - 1)
        self.view.addItem(self.roi_halo)
        self.view.addItem(item)
        self.roi_item = item
        item.sigRegionChanged.connect(self._track_halo)
        self._track_halo()

    def _track_halo(self) -> None:
        """The dark line under the outline, in the ROI's own shape."""
        if self.roi_halo is None or self.roi_item is None:
            return
        self.roi_halo.setPath(self.roi_item.mapToParent(self.roi_item.shape()))

    def _roi_edited(self) -> None:
        item, old = self.roi_item, self.session.roi
        if item is None or old is None:
            return
        if old.kind == "rect":
            pos, size = item.pos(), item.size()
            region = Region.rect(pos.x(), pos.y(), pos.x() + size.x(), pos.y() + size.y())
        elif old.kind == "line":
            size = item.size()
            corners = [item.mapToParent(QPointF(x, y)) for x, y in
                       ((0, 0), (size.x(), 0), (size.x(), size.y()), (0, size.y()))]
            start = ((corners[0].x() + corners[3].x()) / 2, (corners[0].y() + corners[3].y()) / 2)
            end = ((corners[1].x() + corners[2].x()) / 2, (corners[1].y() + corners[2].y()) / 2)
            region = Region.line(start, end, size.y())
        else:
            pts = [item.mapToParent(pg.Point(p)) for p in item.getState()["points"]]
            region = Region("polygon", [(p.x(), p.y()) for p in pts])
        self.session.roi = region                  # no redraw: the item is the truth
        self.session.changed("roi-edited")

    # ------------------------------------------------------------- saving
    def save_png(self, path) -> None:
        """The view as displayed, 8-bit RGB."""
        rgb = (self.render_now() * 255).astype(np.uint8)
        h, w, _ = rgb.shape
        QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).save(str(path))

    def save_tiff(self, path, pixelsize: float, what: str = "rgb") -> None:
        """Re-render the current field of view at ``pixelsize`` (data units).

        ``what`` is "rgb" (8-bit colour, as displayed) or "intensity" (32-bit
        float, the summed weights, for quantitative use).  Written as an ImageJ
        TIFF with the pixel size in its resolution tags.
        """
        import tifffile
        rect = self.view.viewRect()
        fov = FieldOfView.from_range((rect.left(), rect.right()),
                                     (rect.top(), rect.bottom()), pixelsize)
        rgb, weight = self.composite(fov)
        data = ((rgb * 255).astype(np.uint8) if what == "rgb"
                else weight.astype(np.float32))
        px_um = pixelsize / 1000.0
        tifffile.imwrite(str(path), data, imagej=True,
                         resolution=(1 / px_um, 1 / px_um),
                         metadata={"unit": "um", "pixelsize_nm": pixelsize,
                                   "x0": fov.x0, "y0": fov.y0})


class RenderToolBar(QToolBar):
    """Save (with a menu of what to save) and, later, ROI tools."""

    def __init__(self, view: RenderView, parent=None):
        super().__init__("render", parent)
        self.view = view
        self.setMovable(False)

        save = QToolButton(text="Save", popupMode=QToolButton.InstantPopup)
        menu = QMenu(save)
        menu.addAction("PNG as displayed...", self._png)
        menu.addAction("TIFF, colour at pixel size...", lambda: self._tiff("rgb"))
        menu.addAction("TIFF, intensity (float) at pixel size...",
                       lambda: self._tiff("intensity"))
        save.setMenu(menu)
        self.addWidget(save)

        self.roi_button = QToolButton(text="ROI: line")
        self.roi_button.setToolTip("left: draw a new ROI (press, drag, release; polygon: "
                                   "click per vertex, double-click closes); right: the kind")
        self.roi_menu = QMenu(self.roi_button)
        kinds = QActionGroup(self.roi_menu)
        self.kind = "line"
        for kind, name in (("line", "line"), ("rect", "rectangle"), ("polygon", "polygon")):
            action = self.roi_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(kind == "line")
            action.triggered.connect(lambda _=False, k=kind, n=name: self._set_kind(k, n))
            kinds.addAction(action)
        self.roi_menu.addSeparator()
        self.roi_menu.addAction("line width...", self._line_width)
        self.roi_menu.addAction("clear ROI", view.clear_roi)
        self.roi_button.clicked.connect(lambda: view.start_drawing(self.kind))
        self.roi_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_button.customContextMenuRequested.connect(
            lambda pos: self.roi_menu.exec(self.roi_button.mapToGlobal(pos)))
        self.addWidget(self.roi_button)

        self.addAction(QAction("Reset view", self, triggered=view.reset))
        self.counts = QLabel("")
        self.counts.setStyleSheet("padding-left: 12px")
        self.addWidget(self.counts)
        view.session.on_change(self._update_counts)
        self._update_counts("locs")

    def _update_counts(self, what: str) -> None:
        """Localizations per layer, and inside the ROI when there is one."""
        if what not in ("locs", "layer", "layers", "append", "roi", "roi-edited"):
            return
        session = self.view.session
        parts = []
        for i, layer in enumerate(session.layers):
            if layer.is_image:
                parts.append(f"{layer.name}: image")
                continue
            n = len(layer.filter)
            if session.roi is not None:
                n = f"{len(session.selection(i))} in {session.roi}"
            parts.append(f"{layer.name}: {n}")
        self.counts.setText("   ".join(parts))

    def _set_kind(self, kind: str, name: str) -> None:
        self.kind = kind
        self.roi_button.setText(f"ROI: {name}")

    def _line_width(self) -> None:
        width, ok = QInputDialog.getDouble(self, "Line ROI", "width (data units):",
                                           self.view.line_width, 0.1, 1e6, 1)
        if ok:
            self.view.line_width = width
            if self.view.session.roi is not None and self.view.session.roi.kind == "line":
                p = self.view.session.roi.points
                self.view.session.set_roi(Region.line((p[0] + p[3]) / 2, (p[1] + p[2]) / 2, width))

    def _default(self, suffix: str) -> str:
        path = self.view.session.path
        return str(path.with_name(path.stem + "_image" + suffix)) if path else ""

    def _png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save image", self._default(".png"),
                                              "PNG (*.png)")
        if path:
            self.view.save_png(path)

    def _tiff(self, what: str) -> None:
        current = self.view.fov.pixelsize if self.view.fov else 10.0
        pixelsize, ok = QInputDialog.getDouble(self, "Pixel size", "nm per pixel:",
                                               round(current, 2), 0.1, 10000, 2)
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save TIFF", self._default(".tif"),
                                              "TIFF (*.tif *.tiff)")
        if path:
            self.view.save_tiff(path, pixelsize, what)


def _nice(span: float) -> float:
    decade = 10 ** np.floor(np.log10(span))
    for m in (1, 2, 5, 10):
        if m * decade >= span:
            return float(m * decade)
    return float(decade)
