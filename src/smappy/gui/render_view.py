"""The image: a pyqtgraph view that re-renders whenever the view moves.

The render grid follows the widget, so a render costs the same at any zoom
(`FieldOfView.fit`), and the image item is placed in data coordinates so that
pyqtgraph's pan/zoom acts on the localization coordinates directly.
"""
from __future__ import annotations

import traceback

import atexit
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QImage
from PySide6.QtWidgets import (QDoubleSpinBox, QFileDialog, QGraphicsPathItem,
                               QInputDialog, QLabel, QMenu, QToolBar, QToolButton,
                               QVBoxLayout, QWidget)

from .. import lut as luts
from ..regions import Region
from ..render import FieldOfView, axis_unit
from ..session import Session
from . import folders

TILE = 1.5          # render this many view widths, so a pan needs no render
DEFAULT_PIXEL = 10.0  # nm per screen pixel that a click on *pixel* zooms to

# How the layers are laid out: added up into one picture, as SMAP does by
# default, or a picture each side by side (SMAP's *split*), with or without
# the added-up one after them (*comp*).
LAYOUTS = (("overlay", "overlay"), ("side", "side by side"),
           ("side+overlay", "side by side + overlay"))
PANEL_LABEL = (255, 255, 255)
PANEL_BORDER = pg.mkPen((128, 128, 128), width=1)
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
    failed = Signal(str, int)               # what went wrong, generation

    def __init__(self):
        super().__init__()
        self.thread = QThread()
        self.thread.setStackSize(32 * 1024 * 1024)
        self.moveToThread(self.thread)
        self.thread.start()
        _LIVE_THREADS.append(self.thread)

    def render(self, view: "RenderView", fov: FieldOfView, generation: int) -> None:
        try:
            rgb = view.panel_images(fov)
        except Exception as e:
            # Always answer.  Returning in silence left the view waiting for
            # this render forever, so it never asked for another: one failure
            # -- a table swapped mid-render, a layer that cannot be drawn --
            # and the image stayed black for good.
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}", generation)
            return
        self.done.emit(rgb, fov, generation)


class RenderView(QWidget):
    render_requested = Signal(object, object, int)
    render_failed = Signal(str)             # shown by the window, not swallowed

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
        self.axis_bars = AxisBars(self.view)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

        # pan/zoom fires many range changes; render once they settle
        self._timer = QTimer(singleShot=True, interval=40)
        self._timer.timeout.connect(self.render)
        self.view.sigRangeChanged.connect(self.schedule)
        self.graphics.viewport().installEventFilter(self)     # the trackpad's pinch
        session.on_change(self._on_session)
        # side by side: the first panel is this view, which keeps the ROI
        # tools; the others are `_Mirror`s linked to it, which pan and zoom
        # with it and show the ROI's outline
        self.layout_mode = "overlay"
        self.panels: List = ["overlay"]  # what each panel shows, this view first
        self.mirrors: List[_Mirror] = []
        self.stacked = False             # panels one above another
        self.label = _panel_label(self.view)
        self.fov = None                  # the field of view of the shown tile
        self.tile_valid = False
        self._generation = 0             # the newest request; older results are dropped
        self._busy = False
        self._pending = False
        self.last_error = ""
        self._renderer = _Renderer()
        self.render_requested.connect(self._renderer.render)
        self._renderer.done.connect(self._on_rendered)
        self._renderer.failed.connect(self._on_render_failed)
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
            self._arrange()
            self.reset()
        elif what in ("layer", "layers", "append", "regrouped"):
            self.tile_valid = False      # the picture changed, not just the view
            self._arrange()
            self.schedule()
        elif what == "roi":
            self._show_roi(self.session.roi)

    # ------------------------------------------------------------ panels
    def set_layout_mode(self, mode: str) -> None:
        """"overlay", "side" or "side+overlay": see `LAYOUTS`."""
        if mode not in dict(LAYOUTS):
            raise ValueError(f"layout {mode!r}: one of {', '.join(dict(LAYOUTS))}")
        self.layout_mode = mode
        self.tile_valid = False
        shown = self.view.viewRect()
        self._arrange(force=True)
        # the same place, in panels of another size: lockAspect widens the
        # range to the new shape around it
        self.view.setRange(shown, padding=0)
        self.schedule()

    def panel_specs(self) -> List:
        """What each panel shows: a layer's index, or "overlay"."""
        if self.layout_mode == "overlay":
            return ["overlay"]
        shown = [i for i, layer in enumerate(self.session.layers) if layer.visible]
        if len(shown) < 2:
            return ["overlay"]           # one layer is its own overlay
        return shown + (["overlay"] if self.layout_mode == "side+overlay" else [])

    def _stack(self, n: int) -> bool:
        """Whether ``n`` panels go one above another: whichever way shows the
        data larger in this window.  SMAP stacks data 1.3 times wider than
        tall whatever the window; this also turns with the window."""
        try:
            (x0, x1), (y0, y1) = self.session.full_view()
        except Exception:
            return False
        w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
        size = self.graphics.size()
        W, H = max(size.width(), 1), max(size.height(), 1)
        across = min(W / n / w, H / h)
        above = min(W / w, H / n / h)
        return above > across

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if len(self.panels) > 1:
            self._arrange()

    def _arrange(self, force: bool = False) -> None:
        """Make the panels the layers and the layout ask for."""
        specs = self.panel_specs()
        stacked = self._stack(len(specs)) if len(specs) > 1 else False
        if not force and specs == self.panels and stacked == self.stacked:
            self._name_panels()
            return
        for mirror in self.mirrors:
            mirror.remove(self.graphics)
        self.mirrors = []
        self.panels, self.stacked = specs, stacked
        for n in range(1, len(specs)):
            row, col = (n, 0) if stacked else (0, n)
            self.mirrors.append(_Mirror(self.graphics, self.view, row, col))
        # a grey frame tells the panels apart, as SMAP's grey line does
        self.view.setBorder(PANEL_BORDER if self.mirrors else None)
        # the panels' sizes now, not at the next event: a reset straight after
        # would fit the data to the old size
        self.graphics.ci.layout.activate()
        self._name_panels()
        self._mirror_roi()

    def _name_panels(self) -> None:
        names = [self._panel_name(spec) for spec in self.panels]
        many = len(self.panels) > 1
        self.label.setText(names[0] if many else "")
        for mirror, name in zip(self.mirrors, names[1:]):
            mirror.label.setText(name)

    def _panel_name(self, spec) -> str:
        if spec == "overlay":
            return "overlay"
        try:
            return self.session.layers[spec].name
        except IndexError:
            return ""

    def panel_images(self, fov: FieldOfView) -> List[np.ndarray]:
        """An RGB image per panel on ``fov``; one for the overlay alone.

        Each layer is rendered once: the overlay of the side-by-side panels
        is their sum, as `composite` makes it.
        """
        specs = list(self.panels)
        if specs == ["overlay"]:
            return [self.composite(fov)[0]]
        images, white_any = {}, False
        total = np.zeros((fov.ny, fov.nx, 3), np.float32)
        for i, layer in enumerate(self.session.layers):
            if not layer.visible:
                continue
            white = layer.get_display().white_background
            white_any = white_any or white
            image, _ = layer.render(fov, white_background=False)
            total += image
            image = np.clip(image, 0, 1)
            images[i] = luts.on_white(image) if white else image
        total = np.clip(total, 0, 1)
        images["overlay"] = luts.on_white(total) if white_any else total
        blank = np.zeros((fov.ny, fov.nx, 3), np.float32)
        return [images.get(spec, blank) for spec in specs]

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

    def frame_on(self, xrange, yrange) -> None:
        """Show this box, whatever else the data reaches beyond it.

        `reset` frames everything, which is right for a picture of a cell and
        wrong for one of a photon count: a handful of localizations ten times
        brighter than the rest would leave the structure in a corner.  What
        chose these axes chooses the box too.
        """
        (x0, x1), (y0, y1) = xrange, yrange
        # a little beyond the box: a hollow structure -- a ring in x against z
        # -- has its brightest arcs just outside its own 99 %, and cutting
        # them off is the one thing the fit must not do
        self.view.setRange(QRectF(x0, y0, x1 - x0, y1 - y0), padding=0.05)
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
        white = False
        for layer in self.session.layers:
            if layer.visible:
                white = white or layer.get_display().white_background
                image, rendered = layer.render(fov, white_background=False)
                rgb += image
                weight += rendered.weight
        rgb = np.clip(rgb, 0, 1)
        # once, to the sum: a white ground added to a white ground is still
        # white and would swallow everything drawn on either of them
        return (luts.on_white(rgb) if white else rgb), weight

    def _screen_size(self) -> Tuple[int, int]:
        """The main panel's size in screen pixels: the whole widget unless
        the layers are side by side and it is one of several."""
        if self.mirrors:
            size = self.view.size()
            if size.width() >= 16 and size.height() >= 16:
                return int(size.width()), int(size.height())
        size = self.graphics.size()
        return max(size.width(), 16), max(size.height(), 16)

    def current_fov(self, nx: int = None, ny: int = None) -> FieldOfView:
        rect = self.view.viewRect()
        w, h = self._screen_size()
        return FieldOfView.fit((rect.left(), rect.right()), (rect.top(), rect.bottom()),
                               nx or w, ny or h)

    def pixel_size(self) -> float:
        """What one screen pixel is in render units: nm, in the ordinary picture."""
        return self.current_fov().pixelsize

    def set_pixel_size(self, pixelsize: float) -> None:
        """Zoom about the centre until one screen pixel is ``pixelsize``.

        SMAP's *pixrec*: the pixel size is the zoom.  It is not a setting that
        stays -- the next scroll changes it again, as any zoom does.
        """
        if not pixelsize > 0:
            return
        w, h = self._screen_size()
        w, h = w * pixelsize, h * pixelsize
        centre = self.view.viewRect().center()
        self.view.setRange(QRectF(centre.x() - w / 2, centre.y() - h / 2, w, h),
                           padding=0)

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
        self._update_guides(view)
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
        elif len(rgb) != 1 + len(self.mirrors):  # the panels changed meanwhile
            self._pending = True
        else:
            rect = QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0)
            for item, image in zip([self.image] + [m.image for m in self.mirrors], rgb):
                image = np.ascontiguousarray(image)
                item.setImage(image, levels=[0, 1] if image.dtype.kind == "f" else None,
                              autoLevels=False)
                item.setRect(rect)
            self.fov = fov
            self.tile_valid = True
        if self._pending:
            self._pending = False
            self.render()

    def _on_render_failed(self, message: str, generation: int) -> None:
        """Free the view for the next render, and say why this one is missing.

        The last good tile stays on screen.  Only a newer request is retried:
        re-running the one that just failed would fail the same way, in a loop.
        """
        self._busy = False
        self.last_error = message
        self.render_failed.emit(message)
        if self._pending:
            self._pending = False
            self.render()

    def render_now(self) -> np.ndarray:
        """Synchronous: the exact current view, for saving and tests."""
        fov = self.current_fov()
        rgb, _ = self.composite(fov)
        return np.ascontiguousarray(rgb)

    def _update_guides(self, fov: FieldOfView) -> None:
        """Which way up the picture hangs, and what its bars say.

        A picture of a place hangs the way the camera saw it, y downwards,
        which is the image convention the rest of the program uses.  A plot of
        one column against another is a plot: y goes up, as it does in SMAP's
        versatile renderer, because a photon count that grows downwards reads
        as a lie.  The bars follow: one in nanometres, or one per axis in
        whatever that axis counts.
        """
        axes = self.session.axes()
        versatile = not axes.is_default and len(self.session.locs)
        if self.view.yInverted() == bool(versatile):
            self.view.invertY(not versatile)
        for mirror in self.mirrors:
            if mirror.view.yInverted() != self.view.yInverted():
                mirror.view.invertY(self.view.yInverted())
        self.scalebar.setVisible(not versatile)
        self.axis_bars.setVisible(versatile)
        if versatile:
            self.axis_bars.update(fov, axes, self.session.locs)
            return
        self.scalebar.size = nice_step(0.2 * (fov.x1 - fov.x0))
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
            self._mirror_roi()
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
        self._mirror_roi()

    def _mirror_roi(self) -> None:
        """The ROI's outline on every other panel, where it is drawn, not edited."""
        path = (self.roi_item.mapToParent(self.roi_item.shape())
                if self.roi_item is not None else None)
        for mirror in self.mirrors:
            mirror.show_outline(path)

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
        self.session.roi_edited(region)            # no redraw: the item is the truth

    # ------------------------------------------------------------- saving
    def save_png(self, path) -> None:
        """The view as displayed, 8-bit RGB: the panels side by side, if they are."""
        panels = self.panel_images(self.current_fov())
        if len(panels) > 1:
            # a grey line between panels, as SMAP draws it
            h, w, _ = panels[0].shape
            line = np.full((h, 2, 3) if not self.stacked else (2, w, 3), 0.5,
                           np.float32)
            joined = []
            for n, panel in enumerate(panels):
                joined += ([line] if n else []) + [panel]
            rgb = np.concatenate(joined, axis=0 if self.stacked else 1)
        else:
            rgb = panels[0]
        rgb = (np.ascontiguousarray(rgb) * 255).astype(np.uint8)
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


def _panel_label(view) -> pg.TextItem:
    """A panel's name in its top left corner, fixed on the screen."""
    label = pg.TextItem("", color=PANEL_LABEL, anchor=(0, 0))
    label.setParentItem(view)
    label.setPos(4, 2)
    label.setZValue(100)
    return label


class _Mirror:
    """A further panel: its own image of the same tile, panned and zoomed
    with the main view, with the ROI's outline drawn on it."""

    def __init__(self, graphics, main, row: int, col: int):
        self.view = graphics.addViewBox(row=row, col=col, lockAspect=True,
                                        invertY=main.yInverted(), enableMenu=False,
                                        border=PANEL_BORDER)
        # it follows the main view and never fits itself to its image: the
        # link would carry that fit back to the main view and every panel
        self.view.disableAutoRange()
        self.view.setXLink(main)
        self.view.setYLink(main)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image)
        self.outline = []
        for pen in (ROI_HALO_PEN, ROI_PEN):
            item = QGraphicsPathItem()
            item.setPen(pen)
            item.setBrush(Qt.NoBrush)
            item.setZValue(50)
            item.setVisible(False)
            self.view.addItem(item)
            self.outline.append(item)
        self.label = _panel_label(self.view)

    def show_outline(self, path) -> None:
        for item in self.outline:
            item.setVisible(path is not None)
            if path is not None:
                item.setPath(path)

    def remove(self, graphics) -> None:
        self.view.setXLink(None)
        self.view.setYLink(None)
        graphics.removeItem(self.view)


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
        self.roi_menu.addAction("clear ROI", view.clear_roi)
        self.roi_button.clicked.connect(lambda: view.start_drawing(self.kind))
        self.roi_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_button.customContextMenuRequested.connect(
            lambda pos: self.roi_menu.exec(self.roi_button.mapToGlobal(pos)))
        self.addWidget(self.roi_button)

        # The width of a line ROI is the one number a line is drawn *for* --
        # a profile is taken across it -- and it lived behind a right-click on
        # the button and a modal dialog, which is where nobody found it.  On
        # the toolbar, and it resizes the line that is already drawn.
        self.width_label = QLabel(" width ")
        self.addWidget(self.width_label)
        self.line_width = QDoubleSpinBox(minimum=0.1, maximum=1e6, decimals=1)
        self.line_width.setKeyboardTracking(False)
        self.line_width.setValue(view.line_width)
        self.line_width.setMaximumWidth(110)
        self.line_width.setToolTip("how wide a line ROI is: the band a profile "
                                   "is taken over, and the slab's second axis "
                                   "in the 3D window")
        self.line_width.valueChanged.connect(self._on_width)
        self.addWidget(self.line_width)

        self.addAction(QAction("Reset view", self, triggered=view.reset))

        # SMAP's *split* and *comp* ticks, as one choice of three
        self.layers_button = QToolButton(popupMode=QToolButton.InstantPopup)
        self.layers_button.setToolTip("the layers added up into one picture, or a "
                                      "picture each side by side (stacked when "
                                      "the data is wide), with or without the "
                                      "added-up one after them")
        layers_menu = QMenu(self.layers_button)
        layouts = QActionGroup(layers_menu)
        self.layout_actions = {}
        for mode, name in LAYOUTS:
            action = layers_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(mode == view.layout_mode)
            action.triggered.connect(lambda _=False, m=mode: self.set_layout(m))
            layouts.addAction(action)
            self.layout_actions[mode] = action
        self.layers_button.setMenu(layers_menu)
        self.addWidget(self.layers_button)
        self._layout_text()

        # The size of a screen pixel, as SMAP shows it: the zoom, written as
        # a number.  Typing one zooms to it; the button goes to DEFAULT_PIXEL.
        self.pixel_button = QToolButton(text="pixel")
        self.pixel_button.setToolTip(f"zoom to {DEFAULT_PIXEL:g} nm per screen pixel")
        self.pixel_button.clicked.connect(lambda: self.pixel.setValue(DEFAULT_PIXEL))
        self.addWidget(self.pixel_button)
        self.pixel = QDoubleSpinBox(minimum=0.01, maximum=1e6, decimals=2)
        self.pixel.setKeyboardTracking(False)
        self.pixel.setMaximumWidth(110)
        self.pixel.setToolTip("what one screen pixel is: type a size to zoom to "
                              "it, scroll to zoom as usual")
        self.pixel.valueChanged.connect(view.set_pixel_size)
        self.addWidget(self.pixel)
        view.view.sigRangeChanged.connect(self._show_pixel)
        self._show_pixel()
        self._width_unit()

        self.counts = QLabel("")
        self.counts.setStyleSheet("padding-left: 12px")
        self.addWidget(self.counts)
        view.session.on_change(self._update_counts)
        self._update_counts("locs")
        self._style_buttons()

    def _style_buttons(self) -> None:
        """Buttons in the label's font, bold and framed, so they read as buttons.

        A toolbar's buttons are frameless, and on macOS Qt gives QToolButton
        the small system font, so *Save*, *ROI* and *Reset view* came out
        smaller and fainter than the *width* and layer labels beside them --
        and looked like labels too.  The size is the label's, measured rather
        than written down, so it follows the platform and the user's font.
        """
        font = self.width_label.font()
        size = (f"{font.pointSizeF():g}pt" if font.pointSizeF() > 0
                else f"{font.pixelSize()}px")
        self.setStyleSheet(
            f"QToolButton {{ font-size: {size}; font-weight: bold;"
            " border: 1px solid palette(mid); border-radius: 4px;"
            " padding: 2px 8px; margin: 1px 2px; background: palette(button); }"
            " QToolButton:hover { background: palette(light); }"
            " QToolButton:pressed { background: palette(midlight); }"
            " QToolButton[popupMode=\"2\"] { padding-right: 16px; }"
            " QToolButton::menu-indicator { subcontrol-origin: padding;"
            " subcontrol-position: right center; right: 4px; }")

    def _update_counts(self, what: str) -> None:
        """Localizations per layer, and inside the ROI when there is one."""
        if what not in ("locs", "layer", "layers", "append", "roi", "roi-edited"):
            return
        roi = self.view.session.roi
        if roi is not None and roi.kind == "line":
            blocked = self.line_width.blockSignals(True)
            self.line_width.setValue(float(roi.width))    # dragged on screen
            self.line_width.blockSignals(blocked)
        if what in ("locs", "layer"):
            self._width_unit()
        session = self.view.session
        parts = []
        for i, layer in enumerate(session.layers):
            if layer.is_image:
                parts.append(f"{layer.name}: image")
                continue
            n = len(layer.filter)
            if session.roi is not None:
                n = f"{len(session.shown_selection(i))} in {session.roi}"
            parts.append(f"{layer.name}: {n}")
        self.counts.setText("   ".join(parts))

    def set_layout(self, mode: str) -> None:
        """Overlay, side by side, or both: `LAYOUTS`."""
        self.view.set_layout_mode(mode)
        self.layout_actions[mode].setChecked(True)
        self._layout_text()

    def _layout_text(self) -> None:
        self.layers_button.setText(f"Layers: {dict(LAYOUTS)[self.view.layout_mode]}")

    def _show_pixel(self, *_) -> None:
        """The pixel size after a zoom, written without zooming again."""
        blocked = self.pixel.blockSignals(True)
        self.pixel.setValue(self.view.pixel_size())
        self.pixel.blockSignals(blocked)

    def _set_kind(self, kind: str, name: str) -> None:
        self.kind = kind
        self.roi_button.setText(f"ROI: {name}")

    def _on_width(self, width: float) -> None:
        """The width the next line gets, and the one on screen now."""
        self.view.line_width = width
        roi = self.view.session.roi
        if roi is not None and roi.kind == "line" and abs(roi.width - width) > 1e-9:
            p = roi.points
            self.view.session.set_roi(
                Region.line((p[0] + p[3]) / 2, (p[1] + p[2]) / 2, width))

    def _width_unit(self) -> None:
        """The suffix: the picture's own x unit, which need not be nanometres."""
        axes = self.view.session.axes()
        locs = self.view.session.locs
        try:
            name = axes.names(locs)[0] if len(locs) else "x_nm"
        except KeyError:
            name = "x_nm"
        unit = axis_unit(name)
        suffix = f" {unit}" if unit in ("nm", "pixels") else ""
        self.line_width.setSuffix(suffix)
        self.pixel.setSuffix(suffix)

    def _default(self, suffix: str) -> str:
        path = self.view.session.path
        return (str(path.with_name(path.stem + "_image" + suffix)) if path
                else folders.start("image" + suffix))

    def _png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save image", self._default(".png"),
                                              "PNG (*.png)")
        if path:
            folders.remember(path)
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
            folders.remember(path)
            self.view.save_tiff(path, pixelsize, what)


class AxisBars:
    """A scale bar per axis, each in that axis's own quantity.

    One nanometre bar says everything about the ordinary picture, and nothing
    about a picture of photons against frame: there the horizontal bar has to
    say how many frames it spans and the vertical one how many photons, and
    neither of them is a distance.

    They are drawn in data coordinates -- a scale bar *should* grow with the
    zoom -- and re-anchored to the corner of the view on every render, which
    is where the render already knows what the view is.
    """

    MARGIN = 0.06           # of the view, from the bottom right corner
    FRACTION = 0.2          # of the view, at most: the label rounds down

    def __init__(self, view):
        self.view = view
        self.items = []
        for _ in range(2):
            line = pg.PlotDataItem(pen=pg.mkPen((255, 255, 255), width=3))
            text = pg.TextItem("", color=(255, 255, 255))
            line.setZValue(100)
            text.setZValue(100)
            view.addItem(line)
            view.addItem(text)
            self.items.append((line, text))
        self.items[0][1].setAnchor(pg.Point(0.5, 1.0))     # above its bar
        self.items[1][1].setAnchor(pg.Point(1.0, 0.5))     # left of its bar
        self.setVisible(False)

    def setVisible(self, on: bool) -> None:
        for line, text in self.items:
            line.setVisible(bool(on))
            text.setVisible(bool(on))

    def update(self, fov: FieldOfView, axes, locs) -> None:
        try:
            names = axes.names(locs)
        except KeyError:
            self.setVisible(False)
            return
        w, h = fov.x1 - fov.x0, fov.y1 - fov.y0
        # the corner the two bars meet in, bottom right.  These axes are drawn
        # the way up a plot is, y increasing upwards, so the bottom is y0
        cx, cy = fov.x1 - self.MARGIN * w, fov.y0 + self.MARGIN * h
        for i, (span, scale) in enumerate(((w, axes.x_scale), (h, axes.y_scale))):
            line, text = self.items[i]
            # round in the quantity that is written on the label, not in
            # render units: "2000 photons" rather than "1873.4 photons"
            native = nice_below(self.FRACTION * abs(span) * scale)
            length = native / scale if scale else 0.0
            if i == 0:
                line.setData([cx - length, cx], [cy, cy])
                text.setPos(cx - length / 2, cy)
            else:
                line.setData([cx, cx], [cy, cy + length])
                text.setPos(cx, cy + length / 2)
            text.setText(bar_label(native, axis_unit(names[i])))


def bar_label(value: float, unit: str) -> str:
    """"500 nm", "2 µm", "2000 photons"."""
    if unit == "nm" and value >= 1000:
        return f"{value / 1000:g} µm"
    return f"{value:g} {unit}"


def nice_below(span: float) -> float:
    """A round 1 / 2 / 5 x 10^k at or below ``span``.

    `nice_step` rounds up, which is what a bar of "at least this much" wants;
    a bar that must not eat a third of the picture wants the other direction.
    """
    if not span or not np.isfinite(span):
        return 1.0
    decade = 10.0 ** np.floor(np.log10(abs(span)))
    return float(max((m * decade for m in (1, 2, 5) if m * decade <= abs(span)),
                     default=decade))


def nice_step(span: float) -> float:
    decade = 10 ** np.floor(np.log10(span))
    for m in (1, 2, 5, 10):
        if m * decade >= span:
            return float(m * decade)
    return float(decade)
