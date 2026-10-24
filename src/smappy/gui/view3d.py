"""The 3D window: engine A on the render worker, a mouse, a title bar.

Layers, filters and display come from the Render tab; this window owns the
projection (rotation, zoom, pan) and edits the session's slab.

It is the picture and one row of buttons, like the 2D window.  The settings
used to be a second window as tall as the screen beside it -- the slab's
ranges, two view ticks, eleven rendering fields, a depth histogram -- which
was more to look at than the picture.  Most are touched once or never, so
they drop from the title bar (*options*) and fold away inside it, and the
engine is chosen by the size of the slab rather than asked for.
"""
from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QMainWindow, QMenu,
                               QPushButton, QRadioButton, QToolBar, QToolButton,
                               QVBoxLayout, QWidget)

from ..layer_state import LayerSnapshot
from ..picture import FrameSource, ViewFrame
from ..render import FieldOfView, axis_unit
from ..session import Session
from . import folders
from .render_view import (RenderWorker, bar_label, frame_canvas, nice_step,
                          toolbar_border, toolbar_sheet)
from .widgets import CollapsibleSection, Popup
from ..view3d import (PRESETS, PREVIEW_SCALE, PreviewBudget, Projection, Slab, engine_for,
                      render_3d, snapshot_3d, upscale)

BOX_PEN = pg.mkPen((255, 255, 0, 160), width=1)
DEGREES_PER_PIXEL = 0.4
# the slab's own axes, which are the box's edges: x red, y green, z blue
AXIS_COLORS = ((255, 90, 90), (110, 220, 110), (120, 160, 255))
AXIS_FRACTION = 0.09        # arm length, as a fraction of the view's width
AXIS_CORNER = 0.12          # where the tripod sits, from the bottom left corner


@dataclass(frozen=True)
class Job3D:
    """The 3D view's render: the layers frozen on the GUI thread
    (`snapshot_3d`), a copy of the projection and the slab, the canvas size.
    Run by a `RenderWorker`, whose thread holds the GPU engine and the
    preview budgets."""

    layers: Tuple[LayerSnapshot, ...]
    projection: Projection
    slab: Optional[Slab]
    nx: int
    ny: int
    preview: bool

    def run(self, worker):
        """``(rgb, fov)``."""
        projection, preview = self.projection, self.preview
        scale = PREVIEW_SCALE if preview else 1
        fov = projection.fov(self.nx, self.ny, scale)
        # "auto" is settled here, before the GPU is asked for: a small slab
        # never starts it at all
        wanted = engine_for(projection, self.layers, self.slab)
        engine = worker.gpu() if wanted != "cpu" else None
        if engine is None:
            wanted = "cpu"
        if wanted != projection.engine:
            projection = copy.copy(projection)
            projection.engine = wanted
        if projection.engine == "spheres" and preview:
            preview = False                      # the g-buffer look must not change mid-drag
        budget = worker.budgets.setdefault(projection.engine, PreviewBudget())
        rgb = render_3d(self.layers, projection, self.slab, fov, preview, engine,
                        budget=budget)
        return upscale(rgb, scale, self.ny, self.nx), fov


class View3D(QWidget):
    """The canvas: an image in view coordinates, the slab's box over it."""

    render_requested = Signal(object, int)       # a `Job3D`, generation
    changed = Signal()          # projection or slab edited here

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.projection = copy.deepcopy(session.projection)
        self.graphics = pg.GraphicsLayoutWidget()
        self.view = self.graphics.addViewBox(lockAspect=True, invertY=True,
                                            enableMenu=False, enableMouse=False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image)
        self.box = pg.PlotDataItem(pen=BOX_PEN, connect="pairs")
        self.view.addItem(self.box)
        self.handles = pg.ScatterPlotItem(size=10, pen=BOX_PEN, brush=(255, 255, 0, 60))
        self.view.addItem(self.handles)
        # the same scale bar as the 2D view, and a tripod of the slab's own
        # axes: a rotated view says nothing about its own orientation, and a
        # rendered image says nothing about its size
        self.scalebar = pg.ScaleBar(size=1000, suffix="nm")
        self.scalebar.setParentItem(self.view)
        self.scalebar.anchor((1, 1), (1, 1), offset=(-20, -20))
        self.axes = []
        for i, color in enumerate(AXIS_COLORS):
            arm = pg.PlotDataItem(pen=pg.mkPen(color, width=2))
            label = pg.TextItem("xyz"[i], color=color, anchor=(0.5, 0.5))
            self.view.addItem(arm)
            self.view.addItem(label)
            self.axes.append((arm, label))
        self.show_box = True
        self.show_guides = True
        self._fov: Optional[FieldOfView] = None
        self._face: Optional[tuple] = None       # (axis, sign) while a face drags
        # the slab's angle when the view last followed it: a line ROI turned
        # since turns the view by the difference
        self._slab_angle: Optional[float] = None if session.slab is None else session.slab.angle
        # off: the box follows the ROI and the view stays where it is -- a
        # movie made by moving a line along a structure wants the camera
        # still and the box travelling, not the picture recentred every frame
        self.centre_on_roi = True
        # what the render grid's axes were last time: a change of quantity is
        # a change of units, and the zoom and the pivot mean nothing across it
        self._render_axes = session.axes()
        frame_canvas(self.graphics)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

        self._generation = 0
        self._busy = False
        self._pending: Optional[bool] = None       # queued render: preview flag
        self._dragging = False
        self._last: Optional[QPointF] = None
        self._renderer = RenderWorker()
        self.render_requested.connect(self._renderer.render)
        self._renderer.done.connect(self._on_rendered)
        self._renderer.failed.connect(self._on_render_failed)
        self._timer = QTimer(singleShot=True, interval=30)
        self._timer.timeout.connect(lambda: self.render(preview=False))
        self.graphics.mousePressEvent = self._press
        self.graphics.mouseMoveEvent = self._move
        self.graphics.mouseReleaseEvent = self._release
        self.graphics.wheelEvent = self._wheel
        self.graphics.keyPressEvent = self._key
        self.graphics.setFocusPolicy(Qt.StrongFocus)
        self.graphics.viewport().installEventFilter(self)
        session.on_change(self._on_session)
        # what *Export Image* frames its picture on (`picture.FrameSource`)
        self.frame_source = FrameSource()
        session.view_frames["3D"] = self.frame_source

    # ------------------------------------------------------------ session
    def _on_session(self, what: str) -> None:
        if what == "scan":
            # at once and flagged, as the 2D view does (`RenderView._on_session`)
            self.frame_source.drawing = self.isVisible()
            self._timer.stop()
            self.render()
            return
        if what == "projection":
            # the player's view path (`view_path`): its view, the rest of the
            # projection -- the engine, the opacity -- left as it is here
            new = self.session.projection
            proj = self.projection
            proj.azimuth, proj.elevation, proj.roll = new.azimuth, new.elevation, new.roll
            proj.pivot, proj.offset = new.pivot.copy(), new.offset.copy()
            proj.zoom, proj.focal = new.zoom, new.focal
            self._draw_box()
            self.changed.emit()
            self.frame_source.drawing = self.isVisible()
            self._timer.stop()
            self.render()
            return
        if what in ("locs", "layer", "layers", "append", "slab", "roi", "roi-edited"):
            axes = self.session.axes()
            if axes != self._render_axes:
                # a picture of another quantity: the session has already built
                # the slab again, and the view has to be framed on it afresh
                self._render_axes = axes
                self.fit()
                return
            session, slab = self.session, self.session.slab
            following = session.slab_follows_roi and session.roi is not None
            if what == "locs":
                self.fit()
            elif what == "slab" and slab is not None:
                # the slab's centre is the centre of rotation; follow it in place
                self.projection.move_pivot(slab.center)
                if not following:
                    self._slab_angle = slab.angle
                self._draw_box()
                self.changed.emit()
            elif what in ("roi", "roi-edited") and following and slab is not None:
                # the box follows the 2D ROI, and the view follows the box:
                # centred on it and turned with it, so a line moved or turned
                # is looked at as it was rather than sliding off the screen.
                # On the ROI's edit, not the slab's: a slab moved in depth
                # here (Ctrl+wheel) keeps the view where it was panned to
                turned = 0.0 if self._slab_angle is None else slab.angle - self._slab_angle
                if self.centre_on_roi:
                    self.projection.follow(slab, (turned + 180.0) % 360.0 - 180.0)
                self._slab_angle = slab.angle
                self._draw_box()
                self.changed.emit()
            self.schedule()

    def shutdown(self) -> None:
        from .render_view import stop_render_threads
        stop_render_threads()

    # ---------------------------------------------------------- rendering
    def fit(self) -> None:
        # While "follow 2D ROI" is on, the ROI is the slab -- including a line
        # ROI's width, which is the box's second axis.  Opening the window on a
        # drawn ROI used to show whatever slab was there and wait for `from ROI`.
        session = self.session
        if session.slab is None or (session.slab_follows_roi and session.roi is not None):
            slab = session.slab_from_roi()
        else:
            slab = session.slab
        size = self.graphics.size()
        self.projection.fit(slab, max(size.width(), 16), max(size.height(), 16))
        self.changed.emit()
        self.schedule()

    def schedule(self, preview: bool = False) -> None:
        if preview:
            self.render(preview=True)
        else:
            self._timer.start()

    def render(self, preview: bool = False) -> None:
        self.refresh_frame()
        if not len(self.session.locs):
            self.image.clear()
            self.frame_source.drawing = False
            return
        if self._busy:
            self._pending = preview if self._pending is None else (self._pending and preview)
            return
        size = self.graphics.size()
        self._generation += 1
        self._busy = True
        # frozen here, on the GUI thread: the worker sees nothing else
        job = Job3D(self.session.scanned(snapshot_3d(self.session.layers)),
                    copy.deepcopy(self.projection),
                    copy.deepcopy(self.session.slab),
                    max(size.width(), 16), max(size.height(), 16), preview)
        self.render_requested.emit(job, self._generation)

    def _on_render_failed(self, message: str, generation: int) -> None:
        """Free the view for the next render; the last picture stays.  It
        used to return in silence, and the view then waited for good."""
        self._busy = False
        if self._pending is not None:
            preview, self._pending = self._pending, None
            self.render(preview)
        else:
            self.frame_source.drawing = False

    def _on_rendered(self, result, generation: int) -> None:
        rgb, fov = result
        self._busy = False
        if generation == self._generation:
            self.image.setImage(np.ascontiguousarray(rgb), levels=[0, 1], autoLevels=False)
            self.image.setRect(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0))
            self.view.setRange(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0),
                               padding=0)
            self._fov = fov
            self._draw_box()
        if self._pending is not None:
            preview, self._pending = self._pending, None
            self.render(preview)
        else:
            self.frame_source.drawing = False

    def _face_centers(self, slab: Slab) -> np.ndarray:
        """(6, 3) data points at the face centres, ordered (axis, sign)."""
        c = slab.corners().reshape(2, 2, 2, 3)      # [sx, sy, sz]
        return np.array([c[0].mean(axis=(0, 1)), c[1].mean(axis=(0, 1)),
                         c[:, 0].mean(axis=(0, 1)), c[:, 1].mean(axis=(0, 1)),
                         c[:, :, 0].mean(axis=(0, 1)), c[:, :, 1].mean(axis=(0, 1))])

    def _draw_box(self) -> None:
        self._draw_guides()
        slab = self.session.slab
        if slab is None or not self.show_box:
            self.box.setData([], [])
            self.handles.setData([], [])
            return
        c = slab.corners()
        xv, yv, _ = self.projection.apply(c[:, 0], c[:, 1], c[:, 2])
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        xs = [xv[i] for e in edges for i in e]
        ys = [yv[i] for e in edges for i in e]
        self.box.setData(xs, ys)
        f = self._face_centers(slab)
        fx, fy, _ = self.projection.apply(f[:, 0], f[:, 1], f[:, 2])
        self.handles.setData(fx, fy)

    def _draw_guides(self) -> None:
        """The scale bar, and a tripod of the slab's axes in the corner.

        The arms are the *box's* own axes -- the slab's x, y and z, which are
        the edges the box is drawn with -- projected exactly as the data is,
        so an axis pointing at the viewer is short and one across the screen
        is full length.  The scale bar reads the same as the 2D view's.

        On custom render axes it is the tripod that carries the scale, and the
        bar goes: a rotated view mixes the axes, so a bar across the screen is
        not a length in any one of them, while an arm *is* its own axis.  Each
        arm is then a round number of its own units and says which.
        """
        fov = self._fov
        on = self.show_guides and fov is not None
        axes = self.session.axes()
        versatile = not axes.is_default and len(self.session.locs)
        self.scalebar.setVisible(bool(on) and not versatile)
        for arm, label in self.axes:
            arm.setVisible(bool(on))
            label.setVisible(bool(on))
        if not on:
            return
        span = fov.x1 - fov.x0
        if not versatile:
            size = nice_step(0.2 * span)
            self.scalebar.size = size
            self.scalebar.text.setText(f"{size / 1000:g} µm" if size >= 1000 else f"{size:g} nm")
            self.scalebar.updateBar()

        slab = self.session.slab
        length = AXIS_FRACTION * span
        ox = fov.x0 + AXIS_CORNER * span
        oy = fov.y1 - AXIS_CORNER * (fov.y1 - fov.y0)     # y grows downwards here
        m = self.projection.matrix
        names = self._axis_names(axes) if versatile else None
        scales = (axes.x_scale, axes.y_scale, axes.z_scale)
        for i, (arm, label) in enumerate(self.axes):
            unit = slab.axis_unit(i) if slab is not None else np.eye(3)[i]
            arm_length = length
            if versatile and names[i] is not None:
                native = nice_step(length * scales[i])
                arm_length = native / scales[i]
                label.setText(f"{names[i]} {bar_label(native, axis_unit(names[i]))}",
                              color=AXIS_COLORS[i])
            d = (m @ unit)[:2] * arm_length
            arm.setData([ox, ox + d[0]], [oy, oy + d[1]])
            label.setPos(ox + d[0] * 1.25, oy + d[1] * 1.25)

    def _axis_names(self, axes) -> List[Optional[str]]:
        """The three columns the box's axes are, for labelling the tripod."""
        locs = self.session.locs
        try:
            x_name, y_name = axes.names(locs)
        except KeyError:
            return [None, None, None]
        return [x_name, y_name, axes.depth_name(locs)]

    def _face_at(self, pos: QPointF) -> Optional[tuple]:
        """Which face handle is under a widget position, if any."""
        slab = self.session.slab
        if slab is None or not self.show_box:
            return None
        f = self._face_centers(slab)
        fx, fy, _ = self.projection.apply(f[:, 0], f[:, 1], f[:, 2])
        scene = self.graphics.mapToScene(pos.toPoint())
        view = self.view.mapSceneToView(scene)
        d = np.hypot(fx - view.x(), fy - view.y()) / self.projection.zoom    # pixels
        i = int(np.argmin(d))
        return (i // 2, 1 if i % 2 else -1) if d[i] < 12 else None

    def _drag_face(self, dx: float, dy: float) -> None:
        """Move the dragged face along its axis by the mouse's component on it."""
        slab = self.session.slab
        axis, sign = self._face
        unit = np.zeros(3)
        unit[axis] = 1.0
        if axis < 2:                                   # the slab's in-plane rotation
            a = np.radians(slab.angle)
            unit = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                             [0, 0, 1]]) @ unit
        screen = (self.projection.matrix @ unit)[:2]  # nm on screen per nm along the axis
        norm = float(screen @ screen)
        if norm < 1e-9:
            return
        step = float(np.array([dx, dy]) * self.projection.zoom @ screen) / norm
        lo, hi = slab.axis_range(axis)
        if sign > 0:
            hi = max(hi + step, lo + 1.0)
        else:
            lo = min(lo + step, hi - 1.0)
        slab.set_axis_range(axis, lo, hi)

    def image_rgb(self) -> Optional[np.ndarray]:
        return self.image.image

    # -------------------------------------------------------------- mouse
    def _key(self, event) -> None:
        """Arrows: left / right turn about z, up / down tilt the z axis."""
        step = 5.0
        turns = {Qt.Key_Left: (-step, 0.0), Qt.Key_Right: (step, 0.0),
                 Qt.Key_Up: (0.0, -step), Qt.Key_Down: (0.0, step)}
        if event.key() in (Qt.Key_PageUp, Qt.Key_PageDown):
            self.move_along_sight(1.0 if event.key() == Qt.Key_PageUp else -1.0)
            event.accept()
        elif event.key() in turns:
            az, el = turns[event.key()]
            proj = self.projection
            if proj.rotate_at_centre:
                proj.pivot_at_centre()
            proj.roll = 0.0
            proj.azimuth = (proj.azimuth + az) % 360
            proj.elevation = (proj.elevation + el) % 360
            self._draw_box()
            self.changed.emit()
            self.schedule()
            event.accept()
        else:
            pg.GraphicsLayoutWidget.keyPressEvent(self.graphics, event)

    def _press(self, event) -> None:
        self.graphics.setFocus()
        self._last = event.position()
        self._dragging = True
        self._face = self._face_at(event.position()) if event.button() == Qt.LeftButton else None
        self._mode = ("face" if self._face else "pan" if event.button() == Qt.MiddleButton
                      or event.modifiers() & Qt.ShiftModifier else "rotate")
        if self._mode == "rotate" and self.projection.rotate_at_centre:
            self.projection.pivot_at_centre()

    def _move(self, event) -> None:
        if not self._dragging or self._last is None:
            return
        pos = event.position()
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        if self._mode == "rotate":
            # screen y grows downward: a drag down tips the top towards the viewer
            self.projection.rotate_view(dx * DEGREES_PER_PIXEL, -dy * DEGREES_PER_PIXEL)
        elif self._mode == "face":
            self._drag_face(dx, dy)
        else:
            self.projection.offset -= np.array([dx, dy]) * self.projection.zoom
        self._draw_box()
        self.changed.emit()
        self.schedule(preview=True)

    def _release(self, event) -> None:
        self._dragging = False
        if self._mode == "face" and self.session.slab is not None:
            # while it follows a line or rectangle ROI the box is that ROI,
            # and the drag edits it (`Session.slab_resized`)
            self.session.slab_resized(self.session.slab)
        self._face = None
        self.schedule(preview=False)

    def preset(self, name: str) -> None:
        """Look at the slab from the top, the front or the side (`PRESETS`)."""
        slab = self.session.slab
        self.projection.preset(name, slab.angle if slab is not None else 0.0)
        self._draw_box()
        self.changed.emit()
        self.schedule()

    def refresh_frame(self) -> None:
        """Hand the session what the view shows now (`frame_source`)."""
        try:
            self.frame_source.frame = self.view_frame()
        except Exception:                 # half built, or being torn down
            pass

    def view_frame(self) -> ViewFrame:
        """What this window shows, for *Export Image*: the projection, the
        slab, the canvas at the zoom.  The picture is drawn on the CPU, the
        summed look of the *auto*, *cpu* and *gpu* engines."""
        proj = copy.deepcopy(self.projection)
        size = self.graphics.size()
        fov = proj.fov(max(size.width(), 16), max(size.height(), 16))
        return ViewFrame((fov.x0, fov.x1), (fov.y0, fov.y1), fov.pixelsize,
                         projection=proj, slab=self.session.slab)

    def move_along_sight(self, steps: float) -> None:
        """Fly in (positive) or out: a tenth of the eye's distance a step in
        perspective, where that is the scale of what changes, and a tenth of
        the slab's depth through the screen otherwise."""
        proj, slab = self.projection, self.session.slab
        if proj.focal:
            step = 0.1 * proj.focal
        elif slab is not None:
            step = 0.1 * float(np.abs(slab.size @ np.abs(proj.view_axis(2))))
        else:
            step = 100.0 * proj.zoom
        proj.move_along_sight(step * steps)
        self.changed.emit()
        self._draw_box()
        self.schedule(preview=True)
        self._timer.start()

    def _wheel(self, event) -> None:
        # Qt hands an alt-wheel to the horizontal delta on some platforms
        delta = event.angleDelta()
        steps = (delta.y() or delta.x()) / 120.0
        mods = event.modifiers()
        slab = self.session.slab
        if mods & Qt.AltModifier:                               # the eye, along the sight
            self.move_along_sight(steps)
            return
        if mods & Qt.ControlModifier and slab is not None:      # move along the depth axis
            slab.center -= self.projection.view_axis(2) * slab.size.min() * 0.1 * steps
            self.session.changed("slab")
        elif mods & Qt.ShiftModifier and slab is not None:      # thickness along it
            # size is in the slab's own axes, turned by its angle about z, so
            # the line of sight is turned into them before choosing one
            sight = self.projection.view_axis(2)
            turn = np.radians(slab.angle)
            c, s = np.cos(turn), np.sin(turn)
            local = (c*sight[0] + s*sight[1], -s*sight[0] + c*sight[1], sight[2])
            axis = int(np.argmax(np.abs(local)))
            slab.size[axis] = max(slab.size[axis] * (1.1 ** -steps), 1.0)
            self.session.changed("slab")
        else:
            self.zoom_by(1.15 ** -steps)
        self._draw_box()

    def zoom_by(self, factor: float) -> None:
        """Zoom continuously: a preview now, the exact image once it settles."""
        self.projection.zoom *= factor
        self.changed.emit()
        self._draw_box()
        self.schedule(preview=True)
        self._timer.start()

    def eventFilter(self, obj, event) -> bool:
        # the trackpad's pinch arrives as a native gesture on macOS
        if event.type() == QEvent.NativeGesture and \
                event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom_by(1.0 / (1.0 + event.value()))
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.schedule()




# what the image does with the mouse and the keys, behind *keys*: a line of it
# across the title bar was the widest thing in the window and read once
KEYS = """<table cellspacing="0" cellpadding="3">
<tr><td><b>drag</b></td><td>turn</td></tr>
<tr><td><b>shift-drag</b>, middle button</td><td>pan</td></tr>
<tr><td><b>wheel</b>, pinch</td><td>zoom</td></tr>
<tr><td><b>arrow keys</b></td><td>turn by 5°</td></tr>
<tr><td><b>alt-wheel</b>, page up / down</td><td>move along the line of sight</td></tr>
<tr><td><b>ctrl-wheel</b></td><td>move the slab in depth</td></tr>
<tr><td><b>shift-wheel</b></td><td>slab thicker or thinner in depth</td></tr>
<tr><td><b>drag a dot</b> on the box</td><td>move that face of the slab</td></tr>
</table>"""

# the looks, and the `Projection.engine` each one is: the plain sum is "auto",
# drawn on the GPU or the CPU by the slab's size (`view3d.engine_for`)
LOOKS = (("sum", "auto"), ("points", "points"), ("spheres", "spheres"))


class OptionsPanel(QWidget):
    """What *options* drops down: the look and its settings, then the slab and
    the view folded away.  Each look shows only the fields it reads."""

    def __init__(self, session: Session, view: View3D, parent=None):
        super().__init__(parent)
        self.session = session
        self.view3d = view
        proj = view.projection
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("look"))
        self.looks = QButtonGroup(self)
        tips = ("every localization as its blur, added up, as in 2D",
                "every localization as a small transparent dot (GPU)",
                "every localization as a shaded ball (GPU)")
        engines = [engine for _, engine in LOOKS]
        current = engines.index(proj.engine) if proj.engine in engines else 0
        for i, ((name, _), tip) in enumerate(zip(LOOKS, tips)):
            button = QRadioButton(name)
            button.setToolTip(tip)
            button.setChecked(i == current)
            self.looks.addButton(button, i)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self.looks.idToggled.connect(lambda _, on: on and self._on_projection_settings())
        self.gpu_note = QLabel("no GPU here: drawn as a sum")
        self.gpu_note.setStyleSheet("color: gray")
        self.gpu_note.setVisible(False)
        self._gpu = None                         # the GPU's name once asked; "" for none
        view._renderer.engine_ready.connect(self._on_gpu)
        layout.addWidget(self.gpu_note)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(2)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.attenuation = QDoubleSpinBox(minimum=0, maximum=1e6, decimals=0, suffix=" nm")
        self.attenuation.setSpecialValueText("off")
        self.attenuation.setToolTip("the far side fades over this depth; off at 0")
        self.opacity = QDoubleSpinBox(minimum=0, maximum=1, singleStep=0.1, decimals=2)
        self.opacity.setToolTip("how much the front hides the back; 0 = a plain sum "
                                "(fastest), 1 = opaque")
        self.point_size = QDoubleSpinBox(minimum=0, maximum=10000, decimals=1, suffix=" nm")
        self.point_size.setSpecialValueText("precision")
        self.point_size.setToolTip("radius in nm; at 0 the median localization precision")
        self.point_size.setValue(proj.point_size)
        self.point_alpha = QDoubleSpinBox(minimum=0.01, maximum=1, singleStep=0.01, decimals=2)
        self.point_alpha.setToolTip("how opaque one dot is: thousands pile up on a "
                                    "pixel, so a useful value is small")
        self.point_alpha.setValue(Projection.point_alpha)
        self.ssao = QDoubleSpinBox(minimum=0, maximum=1, singleStep=0.1, decimals=2)
        self.ssao.setValue(proj.ssao_strength)
        self.ssao.setToolTip("how much the balls shade their neighbours; 0 = off")
        form.addRow("dim with depth", self.attenuation)
        form.addRow("opacity", self.opacity)
        form.addRow("size", self.point_size)
        form.addRow("alpha", self.point_alpha)
        form.addRow("occlusion", self.ssao)
        # the fields each look reads; the others are not shown
        self._rows = {
            "auto": {self.attenuation, self.opacity},
            "points": {self.point_size, self.point_alpha},
            "spheres": {self.attenuation, self.point_size, self.ssao},
        }
        # focal distance in the unit the box is thought of in, and a tick rather
        # than a magic 0: off is orthographic, on starts at ten box lengths
        # away, which is a lens that shows depth without distorting it.
        self.perspective_on = QCheckBox("perspective")
        self.perspective_on.setToolTip("off: parallel edges stay parallel")
        self.perspective_on.toggled.connect(self._on_projection_settings)
        self.perspective = QDoubleSpinBox(minimum=0.1, maximum=1e6, decimals=1, suffix=" µm")
        self.perspective.setToolTip("how far the eye is from the box")
        self.perspective.setEnabled(False)
        self._focal_typed = False
        self.perspective.valueChanged.connect(lambda _=0.0: setattr(self, "_focal_typed", True))
        self.depth_color = QCheckBox("colour by depth")
        self.depth_color.setToolTip("colour by the distance from the viewer, in each "
                                    "layer's colour LUT, in place of its colour field")
        self.depth_color.toggled.connect(self._on_projection_settings)
        for w in (self.attenuation, self.opacity, self.perspective, self.point_size,
                  self.point_alpha, self.ssao):
            w.setKeyboardTracking(False)
            w.setMaximumWidth(110)
            w.valueChanged.connect(self._on_projection_settings)
        self._form = form
        layout.addLayout(form)
        perspective = QHBoxLayout()
        perspective.addWidget(self.perspective_on)
        perspective.addWidget(self.perspective)
        perspective.addStretch(1)
        layout.addLayout(perspective)
        layout.addWidget(self.depth_color)

        self.box = QCheckBox("show box")
        self.box.setToolTip("the slab's edges, and the dots that move its faces")
        self.box.setChecked(True)
        self.box.toggled.connect(self._on_box)
        self.guides = QCheckBox("scale bar and axes")
        self.guides.setToolTip("the bar is a length on the screen; the arms are the "
                               "box's own x, y and z")
        self.guides.setChecked(True)
        self.guides.toggled.connect(self._on_guides)
        shows = QHBoxLayout()
        shows.setSpacing(12)
        shows.addWidget(self.box)
        shows.addWidget(self.guides)
        shows.addStretch(1)
        layout.addLayout(shows)
        self.in_slab = QCheckBox("plugins use the slab")
        self.in_slab.setToolTip("while this window is open, every plugin measures only "
                                "what is inside the slab")
        self.in_slab.setChecked(session.select_in_slab)
        self.in_slab.toggled.connect(self._on_in_slab)
        layout.addWidget(self.in_slab)

        self.slab_section = CollapsibleSection("slab", self._slab_part(), expanded=False)
        self.view_section = CollapsibleSection("view", self._view_part(), expanded=False)
        for section in (self.slab_section, self.view_section):
            # a popup does not grow by itself when a part opens inside it
            section.toggled.connect(lambda _: QTimer.singleShot(0, self._fit_popup))
            layout.addWidget(section)

        self._show_rows()
        session.on_change(lambda what: self.refresh() if what in ("slab", "roi", "locs") else None)
        view.changed.connect(self.refresh)
        self.refresh()

    def _slab_part(self) -> QWidget:
        part = QWidget()
        grid = QGridLayout(part)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        self.follow = QCheckBox("follow 2D ROI")
        self.follow.setToolTip("the slab's footprint is the ROI in the 2D view, "
                               "and its depth the z filter")
        self.follow.setChecked(self.session.slab_follows_roi)
        self.follow.toggled.connect(self._on_follow)
        from_roi = QPushButton("from ROI")
        from_roi.setToolTip("take the footprint from the ROI once, now")
        # from the ROI afresh: a z range dragged since goes back to the filter's
        from_roi.clicked.connect(lambda: self.session.slab_from_roi(keep_z=False))
        grid.addWidget(self.follow, 0, 0, 1, 3)
        grid.addWidget(from_roi, 0, 3)
        self.ranges: List[tuple] = []
        for i, name in enumerate(("x", "y", "z")):
            lo, hi = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=0), \
                     QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=0)
            for w in (lo, hi):
                w.setKeyboardTracking(False)
                w.valueChanged.connect(lambda _=0, i=i: self._on_range(i))
            grid.addWidget(QLabel(name), i + 1, 0)
            grid.addWidget(lo, i + 1, 1)
            grid.addWidget(QLabel("to"), i + 1, 2)
            grid.addWidget(hi, i + 1, 3)
            self.ranges.append((lo, hi))
        self.angle = QDoubleSpinBox(minimum=-180, maximum=180, decimals=1, suffix="°")
        self.angle.setKeyboardTracking(False)
        self.angle.setToolTip("the slab's turn in the plane; a line ROI sets it")
        self.angle.valueChanged.connect(lambda _=0: self._on_range(None))
        grid.addWidget(QLabel("angle"), 4, 0)
        grid.addWidget(self.angle, 4, 1)
        return part

    def _view_part(self) -> QWidget:
        part = QWidget()
        box = QVBoxLayout(part)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        proj = self.view3d.projection
        self.fix_roll = QCheckBox("keep horizon level")
        self.fix_roll.setToolTip("a drag turns about z and tilts z forward or back; "
                                 "off: a trackball, turning about the screen's axes")
        self.fix_roll.setChecked(proj.fix_roll)
        self.fix_roll.toggled.connect(self._on_fix_roll)
        self.at_centre = QCheckBox("turn about screen centre")
        self.at_centre.setToolTip("on: a turn is about what is in the middle of the "
                                  "screen, so after a pan it stays where one looks.  "
                                  "off: about the slab's centre")
        self.at_centre.setChecked(proj.rotate_at_centre)
        self.at_centre.toggled.connect(self._on_at_centre)
        self.centre_roi = QCheckBox("centre on the ROI")
        self.centre_roi.setToolTip("while the slab follows the 2D ROI, a moved or "
                                   "turned ROI is centred and turned with it.  off: "
                                   "the view stays put and the box moves in it, "
                                   "as a movie along a line wants")
        self.centre_roi.setChecked(self.view3d.centre_on_roi)
        self.centre_roi.toggled.connect(self._on_centre_roi)
        box.addWidget(self.fix_roll)
        box.addWidget(self.at_centre)
        box.addWidget(self.centre_roi)
        return part

    def _on_centre_roi(self, on: bool) -> None:
        self.view3d.centre_on_roi = on

    def _fit_popup(self) -> None:
        try:
            window = self.window()
            if window is not self and window.isVisible():
                window.adjustSize()
        except Exception:                   # a singleShot must never raise
            traceback.print_exc()

    @property
    def look(self) -> str:
        """The chosen look as a `Projection.engine`: "auto", "points", "spheres"."""
        return LOOKS[max(self.looks.checkedId(), 0)][1]

    def refresh(self) -> None:
        slab = self.session.slab
        widgets = [w for pair in self.ranges for w in pair] + [self.angle, self.follow]
        for w in widgets:
            w.blockSignals(True)
        if slab is not None:
            for i, (lo, hi) in enumerate(self.ranges):
                a, b = slab.axis_range(i)
                lo.setValue(a)
                hi.setValue(b)
            self.angle.setValue(slab.angle)
        self.follow.setChecked(self.session.slab_follows_roi)
        for w in widgets:
            w.blockSignals(False)
        if not self._focal_typed:        # follows the box until it is set by hand
            self.perspective.blockSignals(True)
            self.perspective.setValue(self._default_focal_um())
            self.perspective.blockSignals(False)

    def _on_range(self, axis) -> None:
        slab = self.session.slab
        if slab is None:
            return
        if axis is not None:
            lo, hi = self.ranges[axis]
            slab.set_axis_range(axis, lo.value(), hi.value())
        slab.angle = self.angle.value()
        self.session.set_slab(slab, follow_roi=False)

    def _on_follow(self, on: bool) -> None:
        self.session.slab_follows_roi = on
        if on:
            self.session.slab_from_roi()

    def _on_fix_roll(self, on: bool) -> None:
        proj = self.view3d.projection
        proj.fix_roll = on
        if on:                                   # level the horizon now
            proj.set_matrix(proj.matrix)
            proj.roll = 0.0
        self.view3d._draw_box()
        self.view3d.schedule()

    def _on_at_centre(self, on: bool) -> None:
        proj = self.view3d.projection
        proj.rotate_at_centre = on
        if not on and self.session.slab is not None:      # back to the slab's centre
            proj.move_pivot(self.session.slab.center)

    def _show_rows(self) -> None:
        used = self._rows[self.look]
        for widgets in self._rows.values():
            for w in widgets:
                self._form.setRowVisible(w, w in used)
        self.perspective.setEnabled(self.perspective_on.isChecked())
        self.gpu_note.setVisible(self.look != "auto" and self._gpu == "")
        self._fit_popup()

    def _on_gpu(self, name: str) -> None:
        self._gpu = name
        self._show_rows()

    def _default_focal_um(self) -> float:
        """Ten box lengths away: far enough that the box is not distorted, near
        enough that the far face is visibly smaller than the near one."""
        slab = self.session.slab
        box = float(np.max(slab.size)) if slab is not None else 1000.0
        return max(round(10.0 * box / 1000.0, 1), 0.1)

    def _on_projection_settings(self) -> None:
        self._show_rows()
        proj = self.view3d.projection
        proj.depth_lambda = self.attenuation.value() or None
        proj.opacity = self.opacity.value()
        proj.focal = (self.perspective.value() * 1000.0     # µm -> nm
                      if self.perspective_on.isChecked() else None)
        proj.color_by_depth = self.depth_color.isChecked()
        proj.engine = self.look
        proj.point_size = self.point_size.value()
        proj.point_alpha = self.point_alpha.value()
        proj.ssao_strength = self.ssao.value()
        self.view3d._draw_box()
        self.view3d.schedule()

    def _on_in_slab(self, on: bool) -> None:
        self.session.select_in_slab = on
        self.session.changed("slab")

    def _on_box(self, on: bool) -> None:
        self.view3d.show_box = on
        self.view3d._draw_box()

    def _on_guides(self, on: bool) -> None:
        self.view3d.show_guides = on
        self.view3d._draw_guides()


class View3DToolBar(QToolBar):
    """The 3D window's title bar, in the 2D one's style (`toolbar_sheet`):
    what is pressed while looking -- the views, fit -- and two buttons that
    drop down the rest."""

    def __init__(self, window: "View3DWindow"):
        super().__init__("3D", window)
        self.setMovable(False)
        self.border = ""
        save = QToolButton(text="Save", popupMode=QToolButton.InstantPopup)
        menu = QMenu(save)
        menu.addAction("PNG as displayed...", window._save)
        menu.addAction("Save image...", window._save_image)
        save.setMenu(menu)
        self.addWidget(save)
        tips = {"top": "look down z", "front": "along the slab's width: z up",
                "side": "along the slab's length: z up"}
        self.presets = {}
        for name in PRESETS:
            button = QToolButton(text=name)
            button.setToolTip(tips.get(name, name))
            button.clicked.connect(lambda _=False, n=name: window._preset(n))
            self.addWidget(button)
            self.presets[name] = button
        fit = QToolButton(text="fit")
        fit.setToolTip("centre the slab and fill the window with it")
        fit.clicked.connect(window.view.fit)
        self.addWidget(fit)
        self.options_button = QToolButton(text="options")
        self.options_button.setToolTip("how it is drawn, the slab, the view")
        self.options_button.clicked.connect(
            lambda: window.options.show_below(self.options_button))
        self.addWidget(self.options_button)
        self.keys_button = QToolButton(text="keys")
        self.keys_button.setToolTip("what the mouse and the keys do in the picture")
        self.keys_button.clicked.connect(lambda: window.keys.show_below(self.keys_button))
        self.addWidget(self.keys_button)
        self._style()

    def _style(self) -> None:
        sheet, _ = toolbar_sheet(QApplication.font("QLabel"), self.border)
        self.setStyleSheet(sheet)

    def set_border(self, colour: str) -> None:
        """The session's colour along the bottom, as on the 2D title bar."""
        border = toolbar_border(colour)
        if border != self.border:       # a style sheet set again restyles everything
            self.border = border
            self._style()


class View3DWindow(QMainWindow):
    shown = Signal(bool)            # shown or hidden: the 2D window's *3D* follows it

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        from .help_window import attach_help
        attach_help(self, 'Panels/3D view')
        self.setWindowTitle("SMAPpy 3D")
        self._fitted = False
        self.view = View3D(session)
        self.setCentralWidget(self.view)
        self.panel = OptionsPanel(session, self.view)
        self.options = Popup(self.panel, self)
        keys = QLabel(KEYS)
        keys.setTextFormat(Qt.RichText)
        self.keys = Popup(keys, self)
        self.toolbar = View3DToolBar(self)
        self.addToolBar(self.toolbar)
        self.resize(900, 760)

    def _preset(self, name: str) -> None:
        self.view.preset(name)

    def clipboard_image(self) -> Optional[QImage]:
        """What Edit > Copy puts on the clipboard: the projection as displayed."""
        rgb = self.view.image_rgb()
        if rgb is None:
            return None
        data = (np.ascontiguousarray(rgb) * 255).astype(np.uint8)
        h, w, _ = data.shape
        return QImage(data.data, w, h, 3 * w, QImage.Format_RGB888).copy()

    def _save(self) -> None:
        image = self.clipboard_image()
        if image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D image", folders.start(),
                                              "PNG (*.png)")
        if path:
            folders.remember(path)
            image.save(path)

    def _save_image(self) -> None:
        """*File/Export/Image* in its window, set to this view's projection."""
        from ..plugins.file import image_settings_for
        self.view.refresh_frame()
        opener = self.view.session.window_openers.get("export image")
        if opener is not None:
            opener(image_settings_for(self.view.session, "3D"))

    def _slab_shown(self, shown: bool) -> None:
        """The slab restricts plugins only while it can be seen (`Session.slab_shown`)."""
        session = self.view.session
        if session.slab_shown != shown:
            session.slab_shown = shown
            if session.select_in_slab:
                session.changed("slab")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._slab_shown(True)
        self.shown.emit(True)
        roi = self.view.session.roi
        if roi is not None and roi.kind == "line":
            # a line is drawn to be looked at from the side: along it, z up
            # ("front", the slab's long axis across the screen), framed --
            # also when the window comes back, which otherwise keeps its view
            self._fitted = True
            self.view.fit()
            self.view.preset("front")
            self.view.fit()
        elif not self._fitted:
            self._fitted = True
            self.view.fit()
        else:
            self.view.schedule()

    def closeEvent(self, event) -> None:
        # hidden, not deleted (the control window keeps it): shown again, it
        # is the same projection and slab
        self.options.hide()
        self.keys.hide()
        self._slab_shown(False)
        self.shown.emit(False)
        super().closeEvent(event)
