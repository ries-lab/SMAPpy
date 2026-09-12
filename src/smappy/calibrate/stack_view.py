"""A window for looking through an averaged bead stack, slice by slice.

The controls that decide *how* a stack is shown -- which slice plane, how the
contrast is scaled, whether the secondary channel is put back into its native
camera orientation -- belong next to the image they change, not in the
calibration window's settings column, which is about how the calibration is
*computed*.  So they live here, and the calibration window only has a button
that opens this.

Single channel shows one volume, dual colour the two averaged channels side by
side on synchronized slices, which is what makes a registration error visible.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QMainWindow, QSlider, QVBoxLayout, QWidget)

ORIENTATIONS = ("XY", "XZ", "YZ")


def stack_slice(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    """One displayed plane out of a z,y,x stack."""
    if orientation == "XY":
        return volume[index]
    if orientation == "XZ":
        return volume[:, index, :]
    if orientation == "YZ":
        return volume[:, :, index]
    raise ValueError(f"unknown orientation {orientation!r}")


class StackViewer(QMainWindow):
    """One or two averaged bead stacks, browsed along any axis.

    ``volumes`` are z,y,x arrays that must share a shape; ``calibration`` is
    anything with ``z_index_to_nm``, so the axial axis reads in nanometres
    rather than in plane numbers.
    """

    def __init__(self, volumes: Sequence[np.ndarray], calibration,
                 titles: Optional[Sequence[str]] = None,
                 mirror_axis: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("smappy average bead stack")
        self.calibration = calibration
        self.mirror_axis = mirror_axis
        self.set_volumes(volumes, titles)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)

        whole = QWidget()
        layout = QVBoxLayout(whole)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(QLabel("slice"))
        self.orientation_box = QComboBox()
        self.orientation_box.addItems(ORIENTATIONS)
        self.orientation_box.setToolTip("which plane to cut the stack along")
        self.orientation_box.currentTextChanged.connect(self._on_orientation)
        top.addWidget(self.orientation_box)
        self.auto_box = QCheckBox("auto contrast to the slice")
        self.auto_box.setChecked(True)
        self.auto_box.setToolTip("scale to this slice's maximum rather than the "
                                 "whole stack's, so a dim plane is still visible")
        self.auto_box.toggled.connect(self.draw)
        top.addWidget(self.auto_box)
        top.addWidget(QLabel("contrast"))
        self.contrast = QDoubleSpinBox(minimum=0.05, maximum=2.0, decimals=2)
        self.contrast.setSingleStep(0.05)
        self.contrast.setValue(1.0)
        self.contrast.setToolTip("upper-limit factor; below 1 reveals dim structure")
        self.contrast.valueChanged.connect(self.draw)
        top.addWidget(self.contrast)
        self.linked_box = QCheckBox("linked channel contrast")
        self.linked_box.setChecked(True)
        self.linked_box.setToolTip("one scale for both channels, so their "
                                   "brightness can be compared")
        self.linked_box.toggled.connect(self.draw)
        self.native_box = QCheckBox("native camera orientation")
        self.native_box.setToolTip("undo the mirror the registration applied, "
                                   "showing the secondary channel as the camera saw it")
        self.native_box.toggled.connect(self.draw)
        for box in (self.linked_box, self.native_box):
            box.setVisible(len(self.volumes) > 1)
            top.addWidget(box)
        top.addStretch(1)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)

        bottom = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slice)
        self.slice_label = QLabel("")
        self.slice_label.setStyleSheet("color: gray")
        self.slice_label.setMinimumWidth(170)
        bottom.addWidget(self.slider, 1)
        bottom.addWidget(self.slice_label)
        layout.addLayout(bottom)

        self.setCentralWidget(whole)
        self.resize(1000 if len(self.volumes) > 1 else 760, 760)
        self._on_orientation(self.orientation_box.currentText())

    # ----------------------------------------------------------------- data
    def set_volumes(self, volumes: Sequence[np.ndarray],
                    titles: Optional[Sequence[str]] = None) -> None:
        """Show a different stack; the controls and the slice are kept."""
        self.volumes: List[np.ndarray] = [np.asarray(v, float) for v in volumes]
        if not self.volumes:
            raise ValueError("a stack viewer needs at least one volume")
        shapes = {v.shape for v in self.volumes}
        if len(shapes) != 1 or self.volumes[0].ndim != 3:
            raise ValueError("every channel must be one z,y,x stack of the same shape")
        self.titles = list(titles) if titles else (
            ["main", "secondary"][:len(self.volumes)] if len(self.volumes) > 1 else [""])
        if hasattr(self, "canvas"):
            self._on_orientation(self.orientation_box.currentText())

    @property
    def orientation(self) -> str:
        return self.orientation_box.currentText()

    def slice_count(self) -> int:
        z, y, x = self.volumes[0].shape
        return {"XY": z, "XZ": y, "YZ": x}[self.orientation]

    def shown_volumes(self) -> List[np.ndarray]:
        """The volumes as asked for: mirrored back if the native box is on."""
        volumes = list(self.volumes)
        if (self.native_box.isChecked() and self.mirror_axis is not None
                and len(volumes) > 1):
            volumes[1] = np.flip(volumes[1], 2 - self.mirror_axis)
        return volumes

    # -------------------------------------------------------------- events
    def _on_orientation(self, _text: str) -> None:
        count = self.slice_count()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(count - 1, 0))
        self.slider.setValue(count // 2)
        self.slider.blockSignals(False)
        self.draw()

    def _on_slice(self, _value: int) -> None:
        self.draw()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self.slider.setValue(int(np.clip(self.slider.value() - steps, 0,
                                         self.slice_count() - 1)))

    def keyPressEvent(self, event) -> None:
        step = {Qt.Key_Left: -1, Qt.Key_Down: -1,
                Qt.Key_Right: 1, Qt.Key_Up: 1}.get(event.key())
        if step is None:
            super().keyPressEvent(event)
            return
        self.slider.setValue(int(np.clip(self.slider.value() + step, 0,
                                         self.slice_count() - 1)))

    # ------------------------------------------------------------- drawing
    def draw(self) -> None:
        volumes = self.shown_volumes()
        orientation = self.orientation
        index = int(np.clip(self.slider.value(), 0, self.slice_count() - 1))
        images = [stack_slice(v, orientation, index) for v in volumes]
        maxima = [float(np.max(image if self.auto_box.isChecked() else volume))
                  for image, volume in zip(images, volumes)]
        if self.linked_box.isChecked() and len(maxima) > 1:
            maxima = [max(maxima)] * len(maxima)
        self.figure.clear()
        axes = np.atleast_1d(self.figure.subplots(1, len(volumes),
                                                  sharex=True, sharey=True))
        depth = self.volumes[0].shape[0]
        for ax, image, maximum, title in zip(axes, images, maxima, self.titles):
            options = dict(cmap="inferno", vmin=0,
                           vmax=max(maximum * self.contrast.value(), 1e-15))
            if orientation != "XY":
                options.update(aspect="auto",
                               extent=(0, image.shape[1] - 1,
                                       self.calibration.z_index_to_nm(depth - 1),
                                       self.calibration.z_index_to_nm(0)))
            ax.imshow(image, **options)
            ax.set(xlabel="y (pixels)" if orientation == "YZ" else "x (pixels)",
                   ylabel="y (pixels)" if orientation == "XY" else "emitter z (nm)",
                   title=" - ".join(p for p in (title, orientation) if p))
        self.canvas.draw_idle()
        where = (f"emitter z = {self.calibration.z_index_to_nm(index):.1f} nm"
                 if orientation == "XY" else
                 f"{'y' if orientation == 'XZ' else 'x'} = {index} px")
        self.slice_label.setText(f"slice {index + 1}/{self.slice_count()} - {where}")
