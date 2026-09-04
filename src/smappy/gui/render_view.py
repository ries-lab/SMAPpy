"""The image: a pyqtgraph view that re-renders whenever the view moves.

The render grid follows the widget, so a render costs the same at any zoom
(`FieldOfView.fit`), and the image item is placed in data coordinates so that
pyqtgraph's pan/zoom acts on the localization coordinates directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (QFileDialog, QInputDialog, QMenu, QToolBar,
                               QToolButton, QVBoxLayout, QWidget)

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

    def composite(self, fov: FieldOfView) -> Tuple[np.ndarray, np.ndarray]:
        """Every visible layer rendered on ``fov`` and added up, as SMAP does.

        Returns the RGB image in [0, 1] and the summed intensity planes.
        """
        rgb = np.zeros((fov.ny, fov.nx, 3), np.float32)
        weight = np.zeros((fov.ny, fov.nx), np.float32)
        for layer in self.session.layers:
            if layer.visible:
                image, rendered = layer.state.image(fov)
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

    def render(self) -> None:
        if not len(self.session.locs):
            self.image.clear()
            return
        fov = self.current_fov()
        rgb, _ = self.composite(fov)
        self.image.setImage(rgb, levels=[0, 1] if rgb.dtype.kind == "f" else None,
                            autoLevels=False)
        self.image.setRect(QRectF(fov.x0, fov.y0, fov.x1 - fov.x0, fov.y1 - fov.y0))
        self.fov = fov
        self.scalebar.size = _nice(0.2 * (fov.x1 - fov.x0))
        size = self.scalebar.size
        self.scalebar.text.setText(f"{size / 1000:g} µm" if size >= 1000 else f"{size:g} nm")
        self.scalebar.updateBar()


    # ------------------------------------------------------------- saving
    def save_png(self, path) -> None:
        """The image as displayed, 8-bit RGB."""
        rgb = (np.ascontiguousarray(self.image.image) * 255).astype(np.uint8)
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

        roi = QToolButton(text="ROI", popupMode=QToolButton.InstantPopup)
        roi_menu = QMenu(roi)
        for name in ("rectangle", "polygon", "line (width...)"):
            action = roi_menu.addAction(name)
            action.setEnabled(False)
            action.setToolTip("not there yet")
        roi.setMenu(roi_menu)
        self.addWidget(roi)

        self.addAction(QAction("Reset view", self, triggered=view.reset))

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
