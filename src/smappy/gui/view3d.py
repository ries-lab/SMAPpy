"""The 3D window: engine A on the render worker, a mouse, a side panel.

Layers, filters and display come from the Render tab; this window owns the
projection (rotation, zoom, pan) and edits the session's slab.
"""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDial, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QHBoxLayout,
                               QInputDialog, QLabel, QMainWindow, QMenu, QPushButton,
                               QToolBar, QToolButton, QVBoxLayout, QWidget)

from ..render import FieldOfView
from ..session import Session
from ..view3d import PRESETS, PREVIEW_SCALE, Projection, Slab, render_3d, upscale

BOX_PEN = pg.mkPen((255, 255, 0, 160), width=1)
DEGREES_PER_PIXEL = 0.4


class _Renderer3D(QObject):
    done = Signal(object, object, object, int)      # rgb, fov, depth histogram, generation

    engine_ready = Signal(str)            # the GPU's name, or "" if there is none

    def __init__(self):
        super().__init__()
        self.thread = QThread()
        self.thread.setStackSize(32 * 1024 * 1024)
        self.moveToThread(self.thread)
        self.thread.start()
        self._gpu = None
        self._gpu_tried = False

    def gpu(self):
        """The GPU engine, made on this thread on first use; None without one."""
        if not self._gpu_tried:
            self._gpu_tried = True
            try:
                from ..gpu import GPUEngine
                self._gpu = GPUEngine()
                self.engine_ready.emit(self._gpu.name)
            except Exception as e:
                print(f"no GPU engine: {e}")
                self.engine_ready.emit("")
        return self._gpu

    def render(self, session: Session, projection: Projection, slab, nx: int, ny: int,
               preview: bool, generation: int) -> None:
        scale = PREVIEW_SCALE if preview else 1
        fov = projection.fov(nx, ny, scale)
        engine = self.gpu() if projection.engine != "cpu" else None
        if projection.engine != "cpu" and engine is None:
            projection = copy.copy(projection)
            projection.engine = "cpu"
        if projection.engine == "spheres" and preview:
            preview = False                      # the g-buffer look must not change mid-drag
        try:
            rgb, hist = render_3d(session.layers, projection, slab, fov, preview, engine)
        except Exception as e:                      # the table changed under us
            print(f"3D render failed: {e}")
            return
        self.done.emit(upscale(rgb, scale, ny, nx), fov, hist, generation)


class View3D(QWidget):
    """The canvas: an image in view coordinates, the slab's box over it."""

    render_requested = Signal(object, object, object, int, int, bool, int)
    changed = Signal()          # projection or slab edited here
    depth_histogram = Signal(object)

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
        self.show_box = True
        self._face: Optional[tuple] = None       # (axis, sign) while a face drags
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.graphics)

        self._generation = 0
        self._busy = False
        self._pending: Optional[bool] = None       # queued render: preview flag
        self._dragging = False
        self._last: Optional[QPointF] = None
        self._renderer = _Renderer3D()
        self.render_requested.connect(self._renderer.render)
        self._renderer.done.connect(self._on_rendered)
        self._timer = QTimer(singleShot=True, interval=30)
        self._timer.timeout.connect(lambda: self.render(preview=False))
        self.graphics.mousePressEvent = self._press
        self.graphics.mouseMoveEvent = self._move
        self.graphics.mouseReleaseEvent = self._release
        self.graphics.wheelEvent = self._wheel
        self.graphics.viewport().installEventFilter(self)
        session.on_change(self._on_session)

    # ------------------------------------------------------------ session
    def _on_session(self, what: str) -> None:
        if what in ("locs", "layer", "layers", "append", "slab", "roi"):
            if what == "locs":
                self.fit()
            elif what == "slab" and self.session.slab is not None:
                # the slab's centre is the centre of rotation; follow it in place
                self.projection.move_pivot(self.session.slab.center)
                self._draw_box()
                self.changed.emit()
            self.schedule()

    def shutdown(self) -> None:
        self._renderer.thread.quit()
        self._renderer.thread.wait(2000)

    # ---------------------------------------------------------- rendering
    def fit(self) -> None:
        slab = self.session.slab or self.session.slab_from_roi()
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
        if not len(self.session.locs):
            self.image.clear()
            return
        if self._busy:
            self._pending = preview if self._pending is None else (self._pending and preview)
            return
        size = self.graphics.size()
        self._generation += 1
        self._busy = True
        self.render_requested.emit(self.session, copy.deepcopy(self.projection),
                                   copy.deepcopy(self.session.slab),
                                   max(size.width(), 16), max(size.height(), 16),
                                   preview, self._generation)

    def _on_rendered(self, rgb, fov: FieldOfView, hist, generation: int) -> None:
        self._busy = False
        if generation == self._generation:
            self.depth_histogram.emit(hist)
            self.image.setImage(np.ascontiguousarray(rgb), levels=[0, 1], autoLevels=False)
            self.image.setRect(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0))
            self.view.setRange(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0),
                               padding=0)
            self._draw_box()
        if self._pending is not None:
            preview, self._pending = self._pending, None
            self.render(preview)

    def _face_centers(self, slab: Slab) -> np.ndarray:
        """(6, 3) data points at the face centres, ordered (axis, sign)."""
        c = slab.corners().reshape(2, 2, 2, 3)      # [sx, sy, sz]
        return np.array([c[0].mean(axis=(0, 1)), c[1].mean(axis=(0, 1)),
                         c[:, 0].mean(axis=(0, 1)), c[:, 1].mean(axis=(0, 1)),
                         c[:, :, 0].mean(axis=(0, 1)), c[:, :, 1].mean(axis=(0, 1))])

    def _draw_box(self) -> None:
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
    def _press(self, event) -> None:
        self._last = event.position()
        self._dragging = True
        self._face = self._face_at(event.position()) if event.button() == Qt.LeftButton else None
        self._mode = ("face" if self._face else "pan" if event.button() == Qt.MiddleButton
                      or event.modifiers() & Qt.ShiftModifier else "rotate")

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
            self.session.set_slab(self.session.slab, follow_roi=False)
        self._face = None
        self.schedule(preview=False)

    def render_at(self, pixelsize: float, what: str = "rgb") -> np.ndarray:
        """Synchronous: the slab at ``pixelsize`` nm per pixel, for saving."""
        proj = copy.deepcopy(self.projection)
        slab = self.session.slab
        c = slab.corners() if slab is not None else None
        if c is not None:
            xv, yv, _ = proj.apply(c[:, 0], c[:, 1], c[:, 2])
            proj.offset = np.array([(xv.min() + xv.max()) / 2, (yv.min() + yv.max()) / 2])
            nx, ny = int(np.ptp(xv) / pixelsize) + 2, int(np.ptp(yv) / pixelsize) + 2
        else:
            size = self.graphics.size()
            nx, ny = size.width(), size.height()
        proj.zoom = pixelsize
        fov = proj.fov(nx, ny)
        if what == "rgb":
            return render_3d(self.session.layers, proj, slab, fov)[0]
        weight = np.zeros((fov.ny, fov.nx), np.float32)
        from ..view3d import render_layer_3d
        for layer in self.session.layers:
            if layer.visible and not layer.is_image:
                st = layer.state
                weight += render_layer_3d(st.locs, st.filter.mask, proj, slab, fov,
                                          st.settings, st.display)[1].weight
        return weight

    def _wheel(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        mods = event.modifiers()
        slab = self.session.slab
        if mods & Qt.ControlModifier and slab is not None:      # move along the depth axis
            slab.center -= self.projection.view_axis(2) * slab.size.min() * 0.1 * steps
            self.session.changed("slab")
        elif mods & Qt.ShiftModifier and slab is not None:      # thickness along it
            axis = int(np.argmax(np.abs(self.projection.view_axis(2))))
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


class SlabPanel(QWidget):
    """Ranges of the slab per axis, the view angles, presets."""

    def __init__(self, session: Session, view: View3D, parent=None):
        super().__init__(parent)
        self.session = session
        self.view3d = view
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("<b>slab</b> (nm)"))
        grid = QGridLayout()
        self.ranges: List[tuple] = []
        for i, name in enumerate(("x", "y", "z")):
            lo, hi = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=0), \
                     QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=0)
            for w in (lo, hi):
                w.setKeyboardTracking(False)
                w.valueChanged.connect(lambda _=0, i=i: self._on_range(i))
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(lo, i, 1)
            grid.addWidget(QLabel("to"), i, 2)
            grid.addWidget(hi, i, 3)
            self.ranges.append((lo, hi))
        self.angle = QDoubleSpinBox(minimum=-180, maximum=180, decimals=1, suffix="°")
        self.angle.setKeyboardTracking(False)
        self.angle.valueChanged.connect(lambda _=0: self._on_range(None))
        grid.addWidget(QLabel("angle"), 3, 0)
        grid.addWidget(self.angle, 3, 1)
        layout.addLayout(grid)
        row = QHBoxLayout()
        self.follow = QCheckBox("follow 2D ROI")
        self.follow.setChecked(session.slab_follows_roi)
        self.follow.toggled.connect(self._on_follow)
        from_roi = QPushButton("from ROI")
        from_roi.clicked.connect(lambda: session.slab_from_roi())
        row.addWidget(self.follow)
        row.addWidget(from_roi)
        layout.addLayout(row)

        layout.addWidget(QLabel("<b>view</b>"))
        dials = QGridLayout()
        self.dials: List[QDial] = []
        for i, name in enumerate(("azimuth", "elevation", "roll")):
            dial = QDial(minimum=-180 if i != 1 else 0, maximum=180, wrapping=i != 1,
                         notchesVisible=True)
            dial.setFixedSize(56, 56)
            dial.valueChanged.connect(lambda v, i=i: self._on_dial(i, v))
            dials.addWidget(dial, 0, i, Qt.AlignCenter)
            dials.addWidget(QLabel(name), 1, i, Qt.AlignCenter)
            self.dials.append(dial)
        layout.addLayout(dials)
        presets = QHBoxLayout()
        for name in PRESETS:
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self._preset(n))
            presets.addWidget(b)
        reset = QPushButton("fit")
        reset.clicked.connect(view.fit)
        presets.addWidget(reset)
        layout.addLayout(presets)

        form = QFormLayout()
        self.attenuation = QDoubleSpinBox(minimum=0, maximum=1e6, decimals=0, suffix=" nm")
        self.attenuation.setToolTip("depth attenuation length; 0 = off")
        self.opacity = QDoubleSpinBox(minimum=0, maximum=1, singleStep=0.1, decimals=2)
        self.opacity.setToolTip("how much the front hides the back; 0 = plain sum "
                                "(fast), 1 = opaque.  Rendered in depth slices.")
        self.slices = QDoubleSpinBox(minimum=2, maximum=256, decimals=0)
        self.slices.setValue(32)
        self.perspective = QDoubleSpinBox(minimum=0, maximum=1e7, decimals=0, suffix=" nm")
        self.perspective.setToolTip("focal distance; 0 = orthographic")
        for w in (self.attenuation, self.opacity, self.slices, self.perspective):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(self._on_projection_settings)
        self.engine = QComboBox()
        self.engine.addItem("CPU", "cpu")
        self.engine.addItem("GPU", "gpu")
        self.engine.addItem("GPU points", "points")
        self.engine.addItem("GPU spheres", "spheres")
        self.engine.setToolTip("CPU and GPU give the same image; GPU points draws "
                               "sprites with alpha, back to front")
        self.engine.currentIndexChanged.connect(self._on_projection_settings)
        self.point_size = QDoubleSpinBox(minimum=0, maximum=10000, decimals=1, suffix=" nm")
        self.point_size.setSpecialValueText("median precision")
        self.point_size.setToolTip("sprite radius in nm, scales with the zoom; "
                                   "0 = the median precision of the shown localizations")
        self.point_size.setValue(0.0)
        self.point_alpha = QDoubleSpinBox(minimum=0.01, maximum=1, singleStep=0.1, decimals=2)
        self.point_alpha.setValue(0.5)
        for w in (self.point_size, self.point_alpha):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(self._on_projection_settings)
        self.ssao = QDoubleSpinBox(minimum=0, maximum=1, singleStep=0.1, decimals=2)
        self.ssao.setValue(0.7)
        self.ssao.setToolTip("spheres: ambient occlusion strength, 0 = off")
        self.ssao_radius = QDoubleSpinBox(minimum=0, maximum=10000, decimals=0, suffix=" nm")
        self.ssao_radius.setSpecialValueText("3 x radius")
        self.ssao_radius.setToolTip("spheres: how far the occlusion looks; 0 = 3 x the sphere radius")
        for w in (self.ssao, self.ssao_radius):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(self._on_projection_settings)
        self.gpu_name = QLabel("")
        self.gpu_name.setStyleSheet("color: gray")
        view._renderer.engine_ready.connect(
            lambda name: self.gpu_name.setText(name or "no GPU: CPU used"))
        self.depth_color = QCheckBox("colour by depth")
        self.depth_color.setToolTip("overrides the layers' colour field with the view depth")
        self.depth_color.toggled.connect(self._on_projection_settings)
        form.addRow("dim with depth", self.attenuation)
        form.addRow("opacity", self.opacity)
        form.addRow("slices", self.slices)
        form.addRow("perspective", self.perspective)
        form.addRow("", self.depth_color)
        form.addRow("engine", self.engine)
        form.addRow("", self.gpu_name)
        form.addRow("point size", self.point_size)
        form.addRow("point alpha", self.point_alpha)
        form.addRow("occlusion", self.ssao)
        form.addRow("occlusion radius", self.ssao_radius)
        self.box = QCheckBox("show box")
        self.box.setChecked(True)
        self.box.toggled.connect(self._on_box)
        form.addRow("", self.box)
        self.in_slab = QCheckBox("plugins use the slab")
        self.in_slab.setToolTip("a plugin's selection is restricted to the slab")
        self.in_slab.setChecked(session.select_in_slab)
        self.in_slab.toggled.connect(self._on_in_slab)
        form.addRow("", self.in_slab)
        layout.addLayout(form)

        layout.addWidget(QLabel("<b>depth</b> of the slab's localizations"))
        self.hist = pg.PlotWidget(background=None)
        self.hist.setFixedHeight(90)
        self.hist.hideAxis("left")
        self.hist.setMenuEnabled(False)
        self.hist.setMouseEnabled(x=False, y=False)
        self.hist_bars = pg.PlotCurveItem([0.0, 1.0], [0.0], pen=None,
                                          brush=(120, 120, 120, 160), fillLevel=0,
                                          stepMode="center")
        self.hist.addItem(self.hist_bars)
        layout.addWidget(self.hist)
        view.depth_histogram.connect(self._on_histogram)
        layout.addStretch(1)

        session.on_change(lambda what: self.refresh() if what in ("slab", "roi", "locs") else None)
        view.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        slab, proj = self.session.slab, self.view3d.projection
        widgets = [w for pair in self.ranges for w in pair] + [self.angle] + self.dials
        for w in widgets:
            w.blockSignals(True)
        if slab is not None:
            for i, (lo, hi) in enumerate(self.ranges):
                a, b = slab.axis_range(i)
                lo.setValue(a)
                hi.setValue(b)
            self.angle.setValue(slab.angle)
        az = ((proj.azimuth + 180) % 360) - 180
        roll = ((proj.roll + 180) % 360) - 180
        el = proj.elevation % 360
        el = el if el <= 180 else 360 - el          # the dial shows the ZXZ range
        for dial, v in zip(self.dials, (az, el, roll)):
            dial.setValue(int(round(v)))
        self.follow.setChecked(self.session.slab_follows_roi)
        for w in widgets:
            w.blockSignals(False)

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

    def _on_dial(self, i: int, value: int) -> None:
        proj = self.view3d.projection
        setattr(proj, ("azimuth", "elevation", "roll")[i], float(value))
        self.view3d._draw_box()
        self.view3d.schedule()

    def _preset(self, name: str) -> None:
        slab = self.session.slab
        self.view3d.projection.preset(name, slab.angle if slab is not None else 0.0)
        self.refresh()
        self.view3d._draw_box()
        self.view3d.schedule()

    def _on_projection_settings(self) -> None:
        proj = self.view3d.projection
        proj.depth_lambda = self.attenuation.value() or None
        proj.opacity = self.opacity.value()
        proj.slices = int(self.slices.value())
        proj.focal = self.perspective.value() or None
        proj.color_by_depth = self.depth_color.isChecked()
        proj.engine = self.engine.currentData()
        proj.point_size = self.point_size.value()
        proj.point_alpha = self.point_alpha.value()
        proj.ssao_strength = self.ssao.value()
        proj.ssao_radius = self.ssao_radius.value()
        self.view3d._draw_box()
        self.view3d.schedule()

    def _on_in_slab(self, on: bool) -> None:
        self.session.select_in_slab = on
        self.session.changed("slab")

    def _on_histogram(self, hist) -> None:
        centres, counts = hist[:, 0], hist[:, 1]
        if counts.sum() <= 0:
            self.hist_bars.setData([0.0, 1.0], [0.0])
            return
        step = centres[1] - centres[0] if len(centres) > 1 else 1.0
        edges = np.r_[centres - step / 2, centres[-1] + step / 2]
        self.hist_bars.setData(edges, counts)

    def _on_box(self, on: bool) -> None:
        self.view3d.show_box = on
        self.view3d._draw_box()


class View3DWindow(QMainWindow):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("smappy 3D")
        self._fitted = False
        self.view = View3D(session)
        self.setCentralWidget(self.view)
        bar = QToolBar("3d")
        bar.setMovable(False)
        save = QToolButton(text="Save", popupMode=QToolButton.InstantPopup)
        menu = QMenu(save)
        menu.addAction("PNG as displayed...", self._save)
        menu.addAction("TIFF, colour at pixel size...", lambda: self._tiff("rgb"))
        menu.addAction("TIFF, intensity (float) at pixel size...", lambda: self._tiff("intensity"))
        save.setMenu(menu)
        bar.addWidget(save)
        for name in PRESETS:
            bar.addAction(QAction(name, self, triggered=lambda _=False, n=name: self._preset(n)))
        bar.addAction(QAction("fit", self, triggered=self.view.fit))
        self.hint = QLabel("  drag: rotate about the slab centre   shift-drag: pan   "
                           "wheel: zoom   ctrl-wheel: slab depth   shift-wheel: thickness")
        bar.addWidget(self.hint)
        self.addToolBar(bar)
        self.panel = SlabPanel(session, self.view)
        dock = QDockWidget("slab and view", self)
        dock.setWidget(self.panel)
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.resize(1000, 700)

    def _preset(self, name: str) -> None:
        self.panel._preset(name)

    def _save(self) -> None:
        rgb = self.view.image_rgb()
        if rgb is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D image", "", "PNG (*.png)")
        if path:
            data = (np.ascontiguousarray(rgb) * 255).astype(np.uint8)
            h, w, _ = data.shape
            QImage(data.data, w, h, 3 * w, QImage.Format_RGB888).save(path)

    def _tiff(self, what: str) -> None:
        pixelsize, ok = QInputDialog.getDouble(self, "Pixel size", "nm per pixel:",
                                               round(self.view.projection.zoom, 2), 0.1, 10000, 2)
        if not ok:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D TIFF", "", "TIFF (*.tif *.tiff)")
        if not path:
            return
        import tifffile
        data = self.view.render_at(pixelsize, what)
        data = (data * 255).astype(np.uint8) if what == "rgb" else data.astype(np.float32)
        px_um = pixelsize / 1000.0
        tifffile.imwrite(path, data, imagej=True, resolution=(1 / px_um, 1 / px_um),
                         metadata={"unit": "um", "pixelsize_nm": pixelsize,
                                   "projection": str(self.view.projection.to_dict()),
                                   "slab": str(self.view.session.slab.to_dict()
                                               if self.view.session.slab else None)})

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            self.view.fit()
        else:
            self.view.schedule()

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
