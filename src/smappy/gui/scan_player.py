"""The Render tab's player: a window moved through the data, live.

What is scanned and how is `smappy.scan`; this is the transport -- play,
pause, stop, a slider to go to a value, the speed -- and *Record*, which
writes the scan as the 2D or the 3D window shows it.

The player narrows the pictures through `Session.set_scan`, never the
layers' filters, so stopping gives back the view as it was and no plugin
run meanwhile sees a scanned table.  Its timer is a `QTimer`'s ``timeout``
rather than a ``singleShot`` (a single shot that raises leaves PySide in a
state the next override segfaults in), and each tick catches what it
raises, stops and says so.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QLabel, QMessageBox,
                               QProgressDialog, QPushButton, QSlider, QSpinBox,
                               QToolButton, QVBoxLayout, QWidget)

from .. import scan as sc
from ..session import Session

#: the coordinates offered before the table's other columns
FIRST = ("frame", "time_s", "z_nm")
ENDS = (("loop", "loop"), ("bounce", "back and forth"), ("stop", "stop"))


class ScanPlayer(QWidget):
    """Play a window of one coordinate through the data, in every window."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.steps: List = []
        self._dirty = True
        self._direction = 1
        self._range = (0.0, 0.0)
        self._integer = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.coordinate = QComboBox()
        self.coordinate.setToolTip("what the window moves through: a column, or the "
                                   "position along or across the line ROI")
        self.coordinate.setMinimumContentsLength(10)
        self.coordinate.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.window_ = self._spin("the width of what is shown; auto: a twentieth "
                                  "of the range")
        self.step = self._spin("how far the window moves each step; smaller than "
                               "the window overlaps them, for smooth transitions; "
                               "auto: a quarter of the window")
        # a row each: "auto (12.5)" in two fields side by side is wider than
        # the control column
        for row, (label, widget) in enumerate((("scan", self.coordinate),
                                               ("window", self.window_),
                                               ("step", self.step))):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        transport = QHBoxLayout()
        self.play = QToolButton(text="▶")
        self.play.setToolTip("play; again to pause, where the window stays")
        self.play.setCheckable(True)
        self.stop_button = QToolButton(text="■")
        self.stop_button.setToolTip("stop: show everything again")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setToolTip("go to a step")
        self.slider.setMinimum(0)
        transport.addWidget(self.play)
        transport.addWidget(self.stop_button)
        transport.addWidget(self.slider, 1)
        layout.addLayout(transport)
        self.where = QLabel("")
        self.where.setStyleSheet("color: grey")
        layout.addWidget(self.where)

        row = QHBoxLayout()
        self.speed = QSpinBox(minimum=1, maximum=60, value=5)
        self.speed.setSuffix(" /s")
        self.speed.setToolTip("steps per second while playing, and the frame rate "
                              "of a recorded movie")
        self.ends = QComboBox()
        for value, text in ENDS:
            self.ends.addItem(text, value)
        self.ends.setToolTip("what playing does at the last step")
        row.addWidget(QLabel("speed"))
        row.addWidget(self.speed)
        row.addWidget(QLabel("at the end"))
        row.addWidget(self.ends, 1)
        layout.addLayout(row)

        record = QHBoxLayout()
        self.record_from = QComboBox()
        self.record_from.addItem("2D picture", "2D")
        self.record_from.addItem("3D view", "3D")
        self.record_from.setToolTip("which window's picture a recording is of")
        self.record_button = QPushButton("Record...")
        self.record_button.setToolTip("every step, as the window shows it, to a TIFF "
                                      "stack, an MP4 or a GIF")
        record.addWidget(QLabel("record"))
        record.addWidget(self.record_from, 1)
        record.addWidget(self.record_button)
        layout.addLayout(record)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.coordinate.currentIndexChanged.connect(self._changed)
        self.window_.valueChanged.connect(self._changed)
        self.step.valueChanged.connect(self._changed)
        self.speed.valueChanged.connect(self._set_speed)
        self.play.toggled.connect(self._on_play)
        self.stop_button.clicked.connect(self.stop)
        self.slider.valueChanged.connect(self._on_slider)
        self.record_button.clicked.connect(self.record)
        session.on_change(self._on_session)
        self._fill_coordinates()

    @staticmethod
    def _spin(tip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(minimum=0.0, maximum=1e9, decimals=2)
        spin.setSpecialValueText("auto")
        spin.setKeyboardTracking(False)
        spin.setToolTip(tip)
        return spin

    # -------------------------------------------------------------- choices
    def _fill_coordinates(self) -> None:
        locs = self.session.locs
        names = [c for c in locs.columns
                 if np.asarray(locs[c]).ndim == 1 and np.asarray(locs[c]).dtype.kind in "iuf"]
        wanted = [c for c in FIRST if c in names] + list(sc.LINE_COORDINATES) + \
            [c for c in names if c not in FIRST]
        current = self.coordinate.currentText()
        items = [self.coordinate.itemText(i) for i in range(self.coordinate.count())]
        if items == wanted:
            return
        blocked = self.coordinate.blockSignals(True)
        self.coordinate.clear()
        self.coordinate.addItems(wanted)
        if current in wanted:
            self.coordinate.setCurrentText(current)
        self.coordinate.blockSignals(blocked)
        self._dirty = True

    def _on_session(self, what: str) -> None:
        try:
            if what == "locs":
                self._fill_coordinates()
                self._dirty = True
            elif what in ("layer", "layers", "append"):
                self._dirty = True        # what is shown, and so the range, moved
            elif what in ("roi", "roi-edited") and \
                    self.coordinate.currentText() in sc.LINE_COORDINATES:
                self._dirty = True
                if self.session.scan is not None and not self.timer.isActive():
                    self.show_step(self.slider.value())
        except Exception:                 # a slot must not raise into Qt
            traceback.print_exc()

    def _changed(self, *_) -> None:
        self._dirty = True
        if self.session.scan is not None:
            self.show_step(self.slider.value())

    # ---------------------------------------------------------------- steps
    def prepare(self) -> List:
        """The steps over what the layers show now, worked out again only
        when something they depend on changed."""
        if not self._dirty and self.steps:
            return self.steps
        from ..layer_state import snapshot_layers
        coordinate = self.coordinate.currentText()
        line = self.session.roi
        if coordinate in sc.LINE_COORDINATES:
            if line is None or line.kind != "line":
                raise ValueError(f"{coordinate}: draw a line ROI first")
            p0, p1 = line.ends()
            half = (float(np.hypot(*(p1 - p0))) if coordinate == sc.ALONG
                    else line.width) / 2
            lo, hi = -half, half
            self._integer = False
        else:
            layers = snapshot_layers(self.session.layers)
            lo, hi = sc.shown_range(layers, coordinate)
            self._integer = sc.is_integer(layers, coordinate)
        window, step = sc.auto_window(lo, hi, self._integer)
        window = self.window_.value() or window
        step = self.step.value() or (window / sc.AUTO_OVERLAP if self.window_.value()
                                     else step)
        if self._integer:
            step = max(1.0, round(step))
        self._range = (lo, hi)
        self._hints(window, step)
        self.steps = sc.scan_steps(lo, hi, window, step, self._integer)
        blocked = self.slider.blockSignals(True)
        self.slider.setMaximum(len(self.steps) - 1)
        self.slider.blockSignals(blocked)
        self._dirty = False
        return self.steps

    def _hints(self, window: float, step: float) -> None:
        """What *auto* is now, greyed in the fields, as a plugin's form says."""
        for spin, value in ((self.window_, window), (self.step, step)):
            spin.setSpecialValueText(f"auto ({value:.4g})")

    def show_step(self, index: int) -> None:
        """Show step ``index``: the scan in every window, the slider and the
        line under it."""
        try:
            steps = self.prepare()
        except (ValueError, KeyError) as error:
            self.where.setText(str(error).strip("'\""))
            self.pause()
            return
        index = int(np.clip(index, 0, len(steps) - 1))
        step = steps[index]
        coordinate = self.coordinate.currentText()
        line = self.session.roi if coordinate in sc.LINE_COORDINATES else None
        self.session.set_scan(sc.Scan(coordinate, step.lo, step.hi, line))
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(blocked)
        self.where.setText(f"{sc.step_text(coordinate, step, self._frame_time())}"
                           f"   ({index + 1} of {len(steps)})")

    def _frame_time(self) -> Optional[float]:
        from ..metadata import table_frame_time
        try:
            return table_frame_time(self.session.locs)
        except Exception:
            return None

    # ------------------------------------------------------------ transport
    def _on_play(self, on: bool) -> None:
        self.play.setText("❚❚" if on else "▶")
        if on:
            if self.session.scan is None:
                self.show_step(self.slider.value())
            self._set_speed()
            self.timer.start()
        else:
            self.timer.stop()

    def pause(self) -> None:
        if self.play.isChecked():
            self.play.setChecked(False)

    def stop(self) -> None:
        """Stop and show everything again."""
        self.pause()
        self.session.set_scan(None)
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(blocked)
        self._direction = 1
        self.where.setText("")

    def _set_speed(self, *_) -> None:
        self.timer.setInterval(int(round(1000.0 / self.speed.value())))

    def _on_slider(self, index: int) -> None:
        self.show_step(index)

    def _tick(self) -> None:
        try:
            steps = self.prepare()
            index = self.slider.value() + self._direction
            if not 0 <= index < len(steps):
                end = self.ends.currentData()
                if end == "stop":
                    self.pause()
                    return
                if end == "bounce":
                    self._direction = -self._direction
                    index = self.slider.value() + self._direction
                else:
                    index = 0
            self.show_step(int(np.clip(index, 0, len(steps) - 1)))
        except Exception as error:        # never into Qt: stop and say why
            traceback.print_exc()
            self.pause()
            self.where.setText(f"stopped: {error}")

    # --------------------------------------------------------------- record
    def record(self) -> None:
        """Every step, as the chosen window shows it, written to a file."""
        try:
            self._record()
        except Exception as error:
            traceback.print_exc()
            QMessageBox.warning(self, "Record", str(error))

    def _record(self) -> None:
        from ..layer_state import snapshot_layers
        from ..movie import write_stack
        from ..view3d import snapshot_3d
        target = self.record_from.currentData()
        maker = self.session.view_frames.get(target)
        frame = maker() if maker is not None else None
        if frame is None or (target == "3D" and frame.projection is None):
            raise ValueError(f"the {self.record_from.currentText()} is not open")
        steps = self.prepare()
        path, _ = QFileDialog.getSaveFileName(
            self, "Record the scan", str(Path(self.session.path or "scan").with_suffix("")),
            "MP4 (*.mp4);;TIFF stack (*.tif);;GIF (*.gif)")
        if not path:
            return
        self.pause()
        layers = (snapshot_3d(self.session.layers) if target == "3D"
                  else snapshot_layers(self.session.layers))
        coordinate = self.coordinate.currentText()
        line = self.session.roi if coordinate in sc.LINE_COORDINATES else None
        ft = self._frame_time()
        pictures = sc.scan_pictures(layers, frame, coordinate, steps, line,
                                    text=lambda s: sc.step_text(coordinate, s, ft))
        progress = QProgressDialog("Recording...", "Cancel", 0, len(steps), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)

        def counted():
            for k, image in enumerate(pictures):
                progress.setValue(k)
                QApplication.processEvents()
                if progress.wasCanceled():
                    raise InterruptedError("cancelled")
                yield image

        try:
            written = write_stack(path, counted(), len(steps), fps=self.speed.value(),
                                  pixel_nm=frame.pixelsize, description=str(self.session.scan
                                                                             or coordinate))
        except InterruptedError:
            Path(path).unlink(missing_ok=True)
            self.where.setText("recording cancelled")
            return
        finally:
            progress.close()
        self.where.setText(f"recorded {len(steps)} steps to {Path(written).name}")
        self.session.log("record scan", f"{coordinate}, {len(steps)} steps to {written}")
