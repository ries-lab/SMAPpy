"""The image: a pyqtgraph view that re-renders whenever the view moves.

The render grid follows the widget, so a render costs the same at any zoom
(`FieldOfView.fit`), and the image item is placed in data coordinates so that
pyqtgraph's pan/zoom acts on the localization coordinates directly.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..render import FieldOfView
from ..session import Session


class RenderView(QWidget):
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
        session.on_change(self._on_session)
        self.fov = None
        self.reset()

    # ---------------------------------------------------------------- flow
    def _on_session(self, what: str) -> None:
        if what == "locs":
            self.reset()
        elif what in ("layer", "layers", "append"):
            self.schedule()

    def schedule(self) -> None:
        self._timer.start()

    def reset(self) -> None:
        """Show everything."""
        layer = self.session.layers[0]
        if not len(self.session.locs) and layer.state.index.n_localizations == 0 \
                and not getattr(layer.state.index, "extent", None):
            self.image.clear()
            return
        (x0, x1), (y0, y1) = layer.state.full_view()
        self.view.setRange(QRectF(x0, y0, x1 - x0, y1 - y0), padding=0)
        self.schedule()

    def render(self) -> None:
        """Render every visible layer and add them up, as SMAP does."""
        if not len(self.session.locs):
            self.image.clear()
            return
        rect = self.view.viewRect()
        size = self.graphics.size()
        nx, ny = max(size.width(), 16), max(size.height(), 16)
        fov = FieldOfView.fit((rect.left(), rect.right()),
                              (rect.top(), rect.bottom()), nx, ny)
        rgb = np.zeros((fov.ny, fov.nx, 3), np.float32)
        for layer in self.session.layers:
            if layer.visible:
                rgb += layer.state.image(fov)[0]
        rgb = np.clip(rgb, 0, 1)
        self.image.setImage(rgb, levels=[0, 1] if rgb.dtype.kind == "f" else None,
                            autoLevels=False)
        self.image.setRect(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0))
        self.fov = fov
        self.scalebar.size = _nice(0.2 * (fov.x1 - fov.x0))
        size = self.scalebar.size
        self.scalebar.text.setText(f"{size / 1000:g} µm" if size >= 1000 else f"{size:g} nm")
        self.scalebar.updateBar()


def _nice(span: float) -> float:
    decade = 10 ** np.floor(np.log10(span))
    for m in (1, 2, 5, 10):
        if m * decade >= span:
            return float(m * decade)
    return float(decade)
