"""The 3D window: engine A on the render worker, a mouse, a side panel.

Layers, filters and display come from the Render tab; this window owns the
projection (rotation, zoom, pan) and edits the session's slab.
"""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (QCheckBox, QDial, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QToolBar,
                               QVBoxLayout, QWidget)

from ..render import FieldOfView
from ..session import Session
from ..view3d import (PRESETS, PREVIEW_SCALE, Projection, Slab, render_layer_3d,
                      upscale)

BOX_PEN = pg.mkPen((255, 255, 0, 160), width=1)
DEGREES_PER_PIXEL = 0.4


class _Renderer3D(QObject):
    done = Signal(object, object, int)

    def __init__(self):
        super().__init__()
        self.thread = QThread()
        self.thread.setStackSize(32 * 1024 * 1024)
        self.moveToThread(self.thread)
        self.thread.start()

    def render(self, session: Session, projection: Projection, slab, nx: int, ny: int,
               preview: bool, generation: int) -> None:
        scale = PREVIEW_SCALE if preview else 1
        fov = projection.fov(nx, ny, scale)
        rgb = np.zeros((fov.ny, fov.nx, 3), np.float32)
        try:
            for layer in session.layers:
                if not layer.visible or layer.is_image:
                    continue
                state = layer.state
                image, _ = render_layer_3d(state.locs, state.filter.mask, projection, slab,
                                           fov, state.settings, state.display, preview,
                                           n_threads=state.n_threads)
                rgb += image
        except Exception as e:                      # the table changed under us
            print(f"3D render failed: {e}")
            return
        self.done.emit(upscale(np.clip(rgb, 0, 1), scale, ny, nx), fov, generation)


class View3D(QWidget):
    """The canvas: an image in view coordinates, the slab's box over it."""

    render_requested = Signal(object, object, object, int, int, bool, int)
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
        self.show_box = True
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
        session.on_change(self._on_session)

    # ------------------------------------------------------------ session
    def _on_session(self, what: str) -> None:
        if what in ("locs", "layer", "layers", "append", "slab", "roi"):
            if what in ("locs",):
                self.fit()
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

    def _on_rendered(self, rgb, fov: FieldOfView, generation: int) -> None:
        self._busy = False
        if generation == self._generation:
            self.image.setImage(np.ascontiguousarray(rgb), levels=[0, 1], autoLevels=False)
            self.image.setRect(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0))
            self.view.setRange(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0),
                               padding=0)
            self._draw_box()
        if self._pending is not None:
            preview, self._pending = self._pending, None
            self.render(preview)

    def _draw_box(self) -> None:
        slab = self.session.slab
        if slab is None or not self.show_box:
            self.box.setData([], [])
            return
        c = slab.corners()
        xv, yv, _ = self.projection.apply(c[:, 0], c[:, 1], c[:, 2])
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        xs = [xv[i] for e in edges for i in e]
        ys = [yv[i] for e in edges for i in e]
        self.box.setData(xs, ys)

    def image_rgb(self) -> Optional[np.ndarray]:
        return self.image.image

    # -------------------------------------------------------------- mouse
    def _press(self, event) -> None:
        self._last = event.position()
        self._dragging = True
        self._mode = ("pan" if event.button() == Qt.MiddleButton
                      or event.modifiers() & Qt.ShiftModifier else "rotate")

    def _move(self, event) -> None:
        if not self._dragging or self._last is None:
            return
        pos = event.position()
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        if self._mode == "rotate":
            self.projection.rotate_by(dx * DEGREES_PER_PIXEL, dy * DEGREES_PER_PIXEL)
        else:
            self.projection.offset -= np.array([dx, dy]) * self.projection.zoom
        self._draw_box()
        self.changed.emit()
        self.schedule(preview=True)

    def _release(self, event) -> None:
        self._dragging = False
        self.schedule(preview=False)

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
            self.projection.zoom *= 1.15 ** -steps
            self.changed.emit()
            self.schedule()
        self._draw_box()

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
            dial = QDial(minimum=-180 if i != 1 else -90, maximum=180 if i != 1 else 90,
                         wrapping=i != 1, notchesVisible=True)
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
        self.attenuation.setKeyboardTracking(False)
        self.attenuation.valueChanged.connect(self._on_attenuation)
        form.addRow("dim with depth", self.attenuation)
        self.box = QCheckBox("show box")
        self.box.setChecked(True)
        self.box.toggled.connect(self._on_box)
        form.addRow("", self.box)
        layout.addLayout(form)
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
        for dial, v in zip(self.dials, (az, proj.elevation, roll)):
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
        self.view3d.projection.preset(name)
        self.refresh()
        self.view3d._draw_box()
        self.view3d.schedule()

    def _on_attenuation(self, value: float) -> None:
        self.view3d.projection.depth_lambda = value or None
        self.view3d.schedule()

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
        bar.addAction(QAction("Save PNG...", self, triggered=self._save))
        for name in PRESETS:
            bar.addAction(QAction(name, self, triggered=lambda _=False, n=name: self._preset(n)))
        bar.addAction(QAction("fit", self, triggered=self.view.fit))
        self.hint = QLabel("  drag: rotate   shift-drag: pan   wheel: zoom   "
                           "ctrl-wheel: slab depth   shift-wheel: thickness")
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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            self.view.fit()
        else:
            self.view.schedule()

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
