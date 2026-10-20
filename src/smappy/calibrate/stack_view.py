"""A window for looking through an averaged bead stack, slice by slice.

The controls that decide *how* a stack is shown -- which slice plane, how the
contrast is scaled, whether the secondary channel is put back into its native
camera orientation -- belong next to the image they change, not in the
calibration window's settings column, which is about how the calibration is
*computed*.  So they live here, and the calibration window only has a button
that opens this.

Single channel shows one volume, dual colour the two averaged channels side by
side on synchronized slices, which is what makes a registration error visible.

Given a calibration's `model_views.ModelViews` (`set_views`), each channel's
data is shown beside its model on one contrast, or beside the residual, and a
second slider steps from the average to each bead alone, aligned onto the
model as the average is: the PSF calibration window's *Browse stacks...*.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QMainWindow, QSlider, QVBoxLayout, QWidget)

from .plots import stack_slice

ORIENTATIONS = ("XY", "XZ", "YZ")
#: beside the data: nothing, the model, or the residual data - model
BESIDE = ("data", "data | model", "data | residual")


class StackViewer(QMainWindow):
    """One or two averaged bead stacks, browsed along any axis.

    ``volumes`` are z,y,x arrays that must share a shape; ``calibration`` is
    anything with ``z_index_to_nm``, so the axial axis reads in nanometres
    rather than in plane numbers.
    """

    def __init__(self, volumes: Sequence[np.ndarray], calibration,
                 titles: Optional[Sequence[str]] = None,
                 mirror_axis: Optional[int] = None, parent=None, views=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("smappy average bead stack")
        self.calibration = calibration
        self.mirror_axis = mirror_axis
        self.views = None
        self.set_volumes(volumes, titles)

        from matplotlib.figure import Figure

        from ..gui.canvas import canvas_class
        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = canvas_class()(self.figure)

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
        self.beside_box = QComboBox()
        self.beside_box.addItems(BESIDE)
        self.beside_box.setToolTip("the model beside the data, on the same contrast, or the residual "
                                   "data - model on its own symmetric scale")
        self.beside_box.currentTextChanged.connect(self.draw)
        top.addWidget(self.beside_box)
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

        beads = QHBoxLayout()
        self.bead_slider = QSlider(Qt.Horizontal)
        self.bead_slider.setToolTip("0: the average of the used beads; then each bead alone, aligned onto "
                                    "the model as the average is")
        self.bead_slider.valueChanged.connect(self.draw)
        self.bead_label = QLabel("")
        self.bead_label.setStyleSheet("color: gray")
        self.bead_label.setMinimumWidth(170)
        self.bead_caption = QLabel("bead")
        beads.addWidget(self.bead_caption)
        beads.addWidget(self.bead_slider, 1)
        beads.addWidget(self.bead_label)
        layout.addLayout(beads)

        self.setCentralWidget(whole)
        self.resize(1000 if len(self.volumes) > 1 else 760, 760)
        if views is not None:
            self.set_views(views)
        else:
            self._show_view_controls()
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

    def set_views(self, views) -> None:
        """Show a calibration's data and model (`model_views.ModelViews`); the planes are its own."""
        self.views = views
        self.mirror_axis = views.mirror_axis
        self._bead_ids = sorted(views.beads)
        self.bead_slider.blockSignals(True)
        self.bead_slider.setRange(0, len(self._bead_ids))
        self.bead_slider.setValue(min(self.bead_slider.value(), len(self._bead_ids)))
        self.bead_slider.blockSignals(False)
        self.set_volumes(views.data, views.titles)
        self._show_view_controls()

    def _show_view_controls(self) -> None:
        has = self.views is not None
        self.beside_box.setVisible(has)
        for widget in (self.bead_slider, self.bead_label, self.bead_caption):
            widget.setVisible(has and bool(self.views.beads))
        for box in (self.linked_box, self.native_box):
            box.setVisible(len(self.volumes) > 1)

    @property
    def bead(self):
        """The bead shown alone, or None for the average."""
        if self.views is None or self.bead_slider.value() == 0:
            return None
        return self._bead_ids[self.bead_slider.value() - 1]

    def z_nm(self, index):
        if self.views is not None:
            return float(np.asarray(self.views.z_nm)[index])
        return float(self.calibration.z_index_to_nm(index))

    @property
    def orientation(self) -> str:
        return self.orientation_box.currentText()

    def slice_count(self) -> int:
        z, y, x = self.volumes[0].shape
        return {"XY": z, "XZ": y, "YZ": x}[self.orientation]

    def _mirrored(self, volumes) -> List[np.ndarray]:
        volumes = list(volumes)
        if (self.native_box.isChecked() and self.mirror_axis is not None
                and len(volumes) > 1):
            volumes[1] = np.flip(volumes[1], 2 - self.mirror_axis)
        return volumes

    def shown_volumes(self) -> List[np.ndarray]:
        """The volumes as asked for: the bead chosen, mirrored back if the native box is on."""
        if self.views is None:
            return self._mirrored(self.volumes)
        return self._mirrored(self.views.of(self.bead)[0])

    def panels(self) -> list:
        """(title, volume, kind) of each panel, in order: each channel's data, and its model or
        residual beside it; kind "data" or "residual" decides the colour scale."""
        data = self.shown_volumes()
        beside = self.beside_box.currentText() if self.views is not None else "data"
        if beside == "data":
            return [(title, v, "data") for title, v in zip(self.titles, data)]
        model = self._mirrored(self.views.of(self.bead)[1])
        out = []
        for title, d, m in zip(self.titles, data, model):
            out.append((" ".join(p for p in (title, "data") if p), d, "data"))
            out.append(((" ".join(p for p in (title, "model") if p), m, "data") if beside == "data | model"
                        else (" ".join(p for p in (title, "residual") if p), d - m, "residual")))
        return out

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
        if not hasattr(self, "canvas"):
            return
        panels = self.panels()
        orientation = self.orientation
        index = int(np.clip(self.slider.value(), 0, self.slice_count() - 1))
        images = [stack_slice(v, orientation, index) for _, v, _ in panels]

        def peak(image, volume):
            return float(np.nanmax(np.abs(image if self.auto_box.isChecked() else volume)))
        maxima = [peak(image, v) for image, (_, v, _) in zip(images, panels)]
        # one scale for a channel's data and model; linked, one for every channel's
        groups = [i // 2 if self.views is not None and len(panels) > len(self.titles) else i
                  for i in range(len(panels))]
        if self.linked_box.isChecked() and len(self.titles) > 1:
            groups = [0] * len(panels)
        scale = {}
        for g, m, (_, _, kind) in zip(groups, maxima, panels):
            if kind == "data":
                scale[g] = max(scale.get(g, 0.0), m)
        self.figure.clear()
        axes = np.atleast_1d(self.figure.subplots(1, len(panels), sharex=True, sharey=True))
        depth = self.volumes[0].shape[0]
        for ax, image, maximum, g, (title, _, kind) in zip(axes, images, maxima, groups, panels):
            if kind == "residual":
                limit = max(maximum, 1e-15)
                options = dict(cmap="RdBu_r", vmin=-limit, vmax=limit)
                title = f"{title} ±{100 * limit / max(scale.get(g, limit), 1e-15):.2g} %"
            else:
                options = dict(cmap="inferno", vmin=0, vmax=max(scale.get(g, maximum) * self.contrast.value(), 1e-15))
            if orientation != "XY":
                options.update(aspect="auto",
                               extent=(0, image.shape[1] - 1, self.z_nm(depth - 1), self.z_nm(0)))
            ax.imshow(image, **options)
            ax.set(xlabel="y (px)" if orientation == "YZ" else "x (px)",
                   ylabel="y (px)" if orientation == "XY" else "emitter z (nm)",
                   title=" - ".join(p for p in (title, orientation) if p))
        self.canvas.draw_idle()
        where = (f"emitter z = {self.z_nm(index):.1f} nm"
                 if orientation == "XY" else
                 f"{'y' if orientation == 'XZ' else 'x'} = {index} px")
        self.slice_label.setText(f"slice {index + 1}/{self.slice_count()} - {where}")
        if self.views is not None:
            bead = self.bead
            self.bead_label.setText("average of the used beads" if bead is None else
                                    f"bead {bead} ({self.views.states.get(bead, '')})")
