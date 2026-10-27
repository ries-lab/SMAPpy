"""The Render tab's player: a window moved through the data, live.

What is scanned and how is `smappy.scan`; this is the transport -- play,
pause, stop, a slider to go to a value, the speed -- and *Record*, which
writes the scan as the 2D or the 3D window shows it.

The *view path* moves the 3D view along a path of key views
(`smappy.view_path`) on the same time line: over the scan's steps while
a coordinate is scanned, or by itself, at its keys' times, with *scan* on
*none*.  It sets the session's projection (`Session.set_projection`), which
the 3D window takes the view from and keeps its own engine and look.

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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QLabel, QListWidget, QMenu, QMessageBox,
                               QProgressDialog, QPushButton, QSlider, QSpinBox,
                               QToolButton, QVBoxLayout, QWidget)

from .. import view_path as vp
from .. import scan as sc
from ..session import Session
from .widgets import CollapsibleSection, CompactSpinBox

#: the coordinates offered before the table's other columns
FIRST = ("frame", "time_s", "z_nm")
ENDS = (("loop", "loop"), ("bounce", "back and forth"), ("stop", "stop"))
#: the coordinate that narrows nothing: the view path alone moves
NONE = "none"


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
        # the positions of the layers left out of the scan, shown whole; reset
        # when the number of layers changes, since positions then mean others
        self._unscanned: set = set()
        self._n_layers = len(session.layers)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)            # the rows belong together; the default parted them
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
        self.layer_choice = QToolButton()
        self.layer_choice.setToolTip("which layers the window narrows; the others "
                                     "are shown whole -- tracks played over the "
                                     "localizations, say")
        self.layer_choice.setPopupMode(QToolButton.InstantPopup)
        self.layer_choice.setMenu(QMenu(self.layer_choice))
        self.layer_choice.menu().aboutToShow.connect(self._fill_layers)
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
        self.where.hide()               # a blank line under the slider while stopped
        layout.addWidget(self.where)

        row = QHBoxLayout()
        self.speed = QSpinBox(minimum=1, maximum=60, value=5)
        self.speed.setSuffix(" /s")
        self.speed.setToolTip("steps per second while playing -- at most what the "
                              "windows can draw -- and the frame rate of a "
                              "recorded movie")
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
        # the simple player is the above; which layers, and the 3D view's
        # path, are folded away until wanted
        self.path = vp.ViewPath()
        more = QWidget()
        more_layout = QVBoxLayout(more)
        more_layout.setContentsMargins(0, 0, 0, 0)
        more_layout.setSpacing(2)
        layers_row = QHBoxLayout()
        layers_row.addWidget(QLabel("layers"))
        layers_row.addWidget(self.layer_choice, 1)
        more_layout.addLayout(layers_row)
        self.path_section = CollapsibleSection("view path", self._path_part(), expanded=True)
        more_layout.addWidget(self.path_section)
        self.more_section = CollapsibleSection("more", more, expanded=False)
        layout.addWidget(self.more_section)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.coordinate.currentIndexChanged.connect(self._changed)
        self.window_.valueChanged.connect(self._changed)
        self.step.valueChanged.connect(self._changed)
        self.speed.valueChanged.connect(self._set_speed)
        self.ends.currentIndexChanged.connect(self._changed)
        self.play.toggled.connect(self._on_play)
        self.stop_button.clicked.connect(self.stop)
        self.slider.valueChanged.connect(self._on_slider)
        self.record_button.clicked.connect(self.record)
        session.on_change(self._on_session)
        self._fill_coordinates()
        self._layer_text()

    def _say(self, text: str) -> None:
        """The line under the slider, shown only when it says something."""
        self.where.setText(text)
        self.where.setVisible(bool(text))

    @staticmethod
    def _spin(tip: str) -> QDoubleSpinBox:
        spin = CompactSpinBox(minimum=0.0, maximum=1e9, decimals=2)
        spin.setSpecialValueText("auto")
        spin.setKeyboardTracking(False)
        spin.setToolTip(tip)
        return spin

    # ------------------------------------------------------------ view path
    def _path_part(self) -> QWidget:
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        self.keys = QListWidget()
        self.keys.setMaximumHeight(90)
        self.keys.setToolTip("the key views; a click shows one in the 3D view")
        self.add_key = QToolButton(text="+")
        self.add_key.setToolTip("add the 3D view as it is now, after the selected key")
        self.set_key = QToolButton(text="set")
        self.set_key.setToolTip("make the selected key the 3D view as it is now")
        self.remove_key = QToolButton(text="−")
        self.remove_key.setToolTip("remove the selected key")
        self.save_path = QToolButton(text="save...")
        self.save_path.setToolTip("write the path to a file")
        self.load_path = QToolButton(text="load...")
        self.load_path.setToolTip("read a path written before")
        buttons = QHBoxLayout()
        for button in (self.add_key, self.set_key, self.remove_key):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.save_path)
        buttons.addWidget(self.load_path)
        self.key_seconds = CompactSpinBox(minimum=0.0, maximum=3600.0, decimals=1, value=2.0)
        self.key_seconds.setSuffix(" s")
        self.key_seconds.setKeyboardTracking(False)
        self.key_seconds.setToolTip("the time from the key before to the selected one")
        self.turn_axis = QComboBox()
        self.turn_axis.addItems(vp.AXES)
        self.turn_axis.setToolTip("what the turn is about: the data's z, x or y through "
                                  "the middle of the screen, or the screen's own axes")
        self.turn_degrees = CompactSpinBox(minimum=-3600.0, maximum=3600.0, decimals=0,
                                           value=360.0)
        self.turn_degrees.setSuffix("°")
        self.turn_degrees.setToolTip("how far it turns; 0 for a zoom alone")
        self.turn_zoom = CompactSpinBox(minimum=0.01, maximum=1000.0, decimals=2, value=1.0)
        self.turn_zoom.setPrefix("×")
        self.turn_zoom.setToolTip("how much larger the picture is at the end; below 1 "
                                  "zooms out")
        self.turn_seconds = CompactSpinBox(minimum=0.1, maximum=3600.0, decimals=1, value=12.0)
        self.turn_seconds.setSuffix(" s")
        self.turn_seconds.setToolTip("how long the turn takes")
        self.add_turn = QPushButton("add turn")
        self.add_turn.setToolTip("add the turn and the zoom to the path, from its last "
                                 "key (or the 3D view as it is now)")
        self.ease = QCheckBox("ease in and out")
        self.ease.setChecked(True)
        self.ease.setToolTip("start and end the path slowly; off for a turn played in "
                             "a loop")
        # what moves is what the keys differ in: the view, the box, or both.
        # No ticks to switch either off -- a path that should leave the view
        # or the box alone is made of keys that do not change it
        grid.addWidget(self.keys, 1, 0, 1, 2)
        grid.addLayout(buttons, 2, 0, 1, 2)
        grid.addWidget(QLabel("seconds"), 3, 0)
        grid.addWidget(self.key_seconds, 3, 1)
        turn = QHBoxLayout()
        turn.addWidget(self.turn_axis, 1)
        turn.addWidget(self.turn_degrees)
        grid.addWidget(QLabel("turn"), 4, 0)
        grid.addLayout(turn, 4, 1)
        zoom = QHBoxLayout()
        zoom.addWidget(self.turn_zoom)
        zoom.addWidget(QLabel("over"))
        zoom.addWidget(self.turn_seconds)
        grid.addWidget(QLabel("zoom"), 5, 0)
        grid.addLayout(zoom, 5, 1)
        last = QHBoxLayout()
        last.addWidget(self.ease, 1)
        last.addWidget(self.add_turn)
        grid.addLayout(last, 6, 0, 1, 2)
        grid.setColumnStretch(1, 1)

        self.ease.toggled.connect(self._set_ease)
        self.keys.currentRowChanged.connect(self._key_selected)
        self.key_seconds.valueChanged.connect(self._set_seconds)
        for button, slot in ((self.add_key, self.add_view), (self.set_key, self.set_view),
                             (self.remove_key, self.remove_view),
                             (self.add_turn, self.add_turn_keys),
                             (self.save_path, self._save_path),
                             (self.load_path, self._load_path)):
            button.clicked.connect(lambda *_, slot=slot: self._guarded(slot))
        self._fill_keys()
        return box

    def _guarded(self, slot) -> None:
        """A button's work, its error said under the slider rather than
        raised into Qt."""
        try:
            slot()
        except (ValueError, OSError) as error:
            self._say(str(error))
        except Exception as error:
            traceback.print_exc()
            self._say(str(error))

    def path_moves(self) -> bool:
        return self.path.moves

    def view_now(self):
        """The 3D window's projection as it is, or an error if it is closed."""
        source = self.session.view_frames.get("3D")
        frame = source() if source is not None else None
        if frame is None or frame.projection is None:
            raise ValueError("open the 3D view first")
        return frame.projection

    def box_moves(self) -> bool:
        return self.path_moves() and self.path.moves_box

    def show_view(self, view) -> None:
        """``view`` in the 3D window: its box first, when the box moves, then
        the view, which renders at once."""
        if view.slab is not None and self.box_moves():
            self.session.set_slab(view.slab, follow_roi=False)
        self.session.set_projection(view.apply(self._base()))

    def _base(self):
        """What the path's viewpoints are put on: the 3D window's projection,
        or the session's while it is closed."""
        try:
            return self.view_now()
        except ValueError:
            return self.session.projection

    def add_view(self) -> None:
        """The 3D view as it is now, a key after the selected one."""
        view = vp.Viewpoint.of(self.view_now(), self.session.slab)
        row = self.keys.currentRow()
        at = len(self.path.keys) if row < 0 else row + 1
        self.path.keys.insert(at, vp.Key(view, self.key_seconds.value() or 2.0))
        self._fill_keys(at)

    def set_view(self) -> None:
        row = self.keys.currentRow()
        if row < 0:
            raise ValueError("select a key first")
        self.path.keys[row].view = vp.Viewpoint.of(self.view_now(), self.session.slab)
        self._fill_keys(row)

    def remove_view(self) -> None:
        row = self.keys.currentRow()
        if row < 0:
            raise ValueError("select a key first")
        del self.path.keys[row]
        self._fill_keys(min(row, len(self.path.keys) - 1))

    def add_turn_keys(self) -> None:
        """The turn and the zoom, from the last key on."""
        if not self.path.keys:
            self.path.keys.append(vp.Key(vp.Viewpoint.of(self.view_now(), self.session.slab)))
        start = self.path.keys[-1].view
        self.path.keys += vp.turn_keys(start, self.turn_axis.currentText(),
                                       self.turn_degrees.value(), self.turn_zoom.value(),
                                       self.turn_seconds.value())
        self._fill_keys(len(self.path.keys) - 1)

    def take_path(self, path: vp.ViewPath) -> None:
        """Play ``path`` alone, looped, recorded from the 3D view: the 3D
        window's ⟳ handed over by *Make a movie of this...*.  What was in
        the path before is replaced."""
        self.stop()
        self.path = path
        blocked = self.ease.blockSignals(True)
        self.ease.setChecked(path.ease)
        self.ease.blockSignals(blocked)
        self.coordinate.setCurrentText(NONE)
        self.ends.setCurrentIndex(self.ends.findData("loop"))
        self.record_from.setCurrentIndex(self.record_from.findData("3D"))
        self.more_section.set_expanded(True)
        self.path_section.set_expanded(True)
        self._fill_keys()
        self._dirty = True
        self._say("the 3D view's motion: ▶ plays it, Record... writes it")

    def _save_path(self) -> None:
        if not self.path.keys:
            raise ValueError("the path has no keys")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the view path", str(Path(self.session.path or "path")
                                              .with_suffix(".view.yaml")),
            "View path (*.view.yaml *.yaml)")
        if path:
            vp.save(self.path, path)
            self._say(f"view path saved to {Path(path).name}")

    def _load_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load a view path", "",
                                              "View path (*.view.yaml *.yaml)")
        if path:
            self.path = vp.load(path)
            blocked = self.ease.blockSignals(True)
            self.ease.setChecked(self.path.ease)
            self.ease.blockSignals(blocked)
            self._fill_keys(0)

    def _fill_keys(self, select: Optional[int] = None) -> None:
        """The list from the path, ``select`` selected; the path changed."""
        blocked = self.keys.blockSignals(True)
        self.keys.clear()
        for i, key in enumerate(self.path.keys):
            self.keys.addItem(self._key_text(i))
        if select is not None and 0 <= select < len(self.path.keys):
            self.keys.setCurrentRow(select)
        self.keys.blockSignals(blocked)
        self._show_seconds(self.keys.currentRow())
        self.set_key.setEnabled(self.keys.currentRow() >= 0)
        self.remove_key.setEnabled(self.keys.currentRow() >= 0)
        self.save_path.setEnabled(bool(self.path.keys))
        self._path_changed()

    def _key_text(self, i: int) -> str:
        """A key's line: its time and its box, in nm."""
        key = self.path.keys[i]
        text = f"{i + 1}   " + ("start" if i == 0 else f"+{key.seconds:g} s")
        if key.view.slab is not None:
            text += "   box " + " x ".join(f"{v:.0f}" for v in key.view.slab.size)
        return text

    def _show_seconds(self, row: int) -> None:
        blocked = self.key_seconds.blockSignals(True)
        if row > 0:
            self.key_seconds.setValue(self.path.keys[row].seconds)
        self.key_seconds.blockSignals(blocked)
        self.key_seconds.setEnabled(row > 0)

    def _key_selected(self, row: int) -> None:
        """Show the key in the 3D view, unless playing."""
        try:
            self._show_seconds(row)
            self.set_key.setEnabled(row >= 0)
            self.remove_key.setEnabled(row >= 0)
            if 0 <= row < len(self.path.keys) and not self.timer.isActive():
                self.show_view(self.path.keys[row].view)
        except Exception:                 # a slot must not raise into Qt
            traceback.print_exc()

    def _set_seconds(self, value: float) -> None:
        row = self.keys.currentRow()
        if row > 0:
            self.path.keys[row].seconds = float(value)
            self.keys.item(row).setText(self._key_text(row))
            self._path_changed()

    def _set_ease(self, on: bool) -> None:
        self.path.ease = on
        self._path_changed()

    def _path_changed(self, *_) -> None:
        """Viewpoint-only, the number of steps is the path's: work them out again."""
        if self.coordinate.currentText() == NONE:
            self._dirty = True

    def step_time(self, index: int, n: int) -> float:
        """Where step ``index`` of ``n`` is on the view path, 0 to 1:
        spread over the scan, or at the path's own times."""
        if self.coordinate.currentText() == NONE:
            return float(self.path.samples(n, loop=self.ends.currentData() == "loop")[index])
        return index / (n - 1) if n > 1 else 0.0

    # -------------------------------------------------------------- choices
    def scanned_layers(self) -> Optional[tuple]:
        """`Scan.layers`: the positions scanned, None for all of them."""
        if not self._unscanned:
            return None
        return tuple(i for i in range(len(self.session.layers)) if i not in self._unscanned)

    def set_scanned(self, index: int, on: bool) -> None:
        """Scan layer ``index``, or show it whole."""
        if on:
            self._unscanned.discard(index)
        else:
            self._unscanned.add(index)
        self._layer_text()
        self._changed()

    def _fill_layers(self) -> None:
        """The menu of layers, built as it opens: the names are the layers'
        as they are now."""
        menu = self.layer_choice.menu()
        menu.clear()
        for i, layer in enumerate(self.session.layers):
            action = menu.addAction(layer.name or f"layer {i + 1}")
            action.setCheckable(True)
            action.setChecked(i not in self._unscanned)
            action.setEnabled(not layer.is_image)
            action.toggled.connect(lambda on, i=i: self.set_scanned(i, on))

    def _layer_text(self) -> None:
        which = self.scanned_layers()
        if which is None:
            text = "all"
        elif not which:
            text = "none"
        else:
            text = ", ".join(self.session.layers[i].name or f"layer {i + 1}" for i in which)
        self.layer_choice.setText(text if len(text) <= 24 else text[:23] + "…")

    def _fill_coordinates(self) -> None:
        locs = self.session.locs
        names = [c for c in locs.columns
                 if np.asarray(locs[c]).ndim == 1 and np.asarray(locs[c]).dtype.kind in "iuf"]
        wanted = [NONE] + [c for c in FIRST if c in names] + list(sc.LINE_COORDINATES) + \
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
        else:
            self.coordinate.setCurrentIndex(1)      # a coordinate, not the view path alone
        self.coordinate.blockSignals(blocked)
        self._dirty = True

    def _on_session(self, what: str) -> None:
        try:
            if what == "locs":
                self._fill_coordinates()
                self._dirty = True
            elif what in ("layer", "layers", "append"):
                self._dirty = True        # what is shown, and so the range, moved
                if len(self.session.layers) != self._n_layers:
                    self._n_layers = len(self.session.layers)
                    self._unscanned.clear()
                self._layer_text()
            elif what in ("roi", "roi-edited") and \
                    self.coordinate.currentText() in sc.LINE_COORDINATES:
                self._dirty = True
                if self.session.scan is not None and not self.timer.isActive():
                    self.show_step(self.slider.value())
        except Exception:                 # a slot must not raise into Qt
            traceback.print_exc()

    def _changed(self, *_) -> None:
        self._dirty = True
        path_only = self.coordinate.currentText() == NONE
        for widget in (self.window_, self.step, self.layer_choice):
            widget.setEnabled(not path_only)
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
        if coordinate == NONE:
            if not self.path_moves():
                raise ValueError("nothing moves: choose what to scan, or give the view path "
                                 "two keys")
            loop = self.ends.currentData() == "loop" and self.path.closed
            n = max(2, int(round(self.path.seconds * self.speed.value())) + (0 if loop else 1))
            self.steps = [None] * n
            blocked = self.slider.blockSignals(True)
            self.slider.setMaximum(n - 1)
            self.slider.blockSignals(blocked)
            self._dirty = False
            return self.steps
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
            lo, hi = sc.shown_range(layers, coordinate, which=self.scanned_layers())
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
            self._say(str(error).strip("'\""))
            self.pause()
            return
        index = int(np.clip(index, 0, len(steps) - 1))
        step = steps[index]
        coordinate = self.coordinate.currentText()
        if self.path_moves():
            t = self.step_time(index, len(steps))
            self.show_view(self.path.at(t))
        if coordinate == NONE:
            if self.session.scan is not None:
                self.session.set_scan(None)
            text = f"{t * self.path.seconds:.1f} of {self.path.seconds:g} s"
        else:
            line = self.session.roi if coordinate in sc.LINE_COORDINATES else None
            self.session.set_scan(sc.Scan(coordinate, step.lo, step.hi, line,
                                          self.scanned_layers()))
            text = sc.step_text(coordinate, step, self._frame_time())
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(blocked)
        self._say(f"{text}   ({index + 1} of {len(steps)})")

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
            if self.session.scan is None or self.path_moves():
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
        self.timer.stop()                 # whatever the play button says
        self.session.set_scan(None)
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(blocked)
        self._direction = 1
        self._say("")

    def _set_speed(self, *_) -> None:
        self.timer.setInterval(int(round(1000.0 / self.speed.value())))
        self._path_changed()            # path alone, the steps are seconds x speed

    def _on_slider(self, index: int) -> None:
        self.show_step(index)

    def drawing(self) -> bool:
        """A picture window has not drawn the last step yet
        (`picture.FrameSource.drawing`)."""
        return any(getattr(source, "drawing", False)
                   for source in self.session.view_frames.values())

    def _tick(self) -> None:
        try:
            if self.drawing():
                # faster than the windows draw: wait for them, so the speed
                # is capped at what can be shown.  Stepping on regardless
                # superseded every picture before it was drawn, and above
                # a few steps a second nothing was shown at all
                return
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
            self._say(f"stopped: {error}")

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
        coordinate = self.coordinate.currentText()
        if coordinate == NONE and target != "3D":
            raise ValueError("the view path moves the 3D view: record the 3D view")
        path, _ = QFileDialog.getSaveFileName(
            self, "Record the scan", str(Path(self.session.path or "scan").with_suffix("")),
            "MP4 (*.mp4);;TIFF stack (*.tif);;GIF (*.gif)")
        if not path:
            return
        self.pause()
        layers = (snapshot_3d(self.session.layers) if target == "3D"
                  else snapshot_layers(self.session.layers))
        line = self.session.roi if coordinate in sc.LINE_COORDINATES else None
        ft = self._frame_time()
        projections = slabs = None
        if target == "3D" and self.path_moves():
            views = [self.path.at(self.step_time(k, len(steps))) for k in range(len(steps))]
            projections = [v.apply(frame.projection) for v in views]
            if self.box_moves():
                slabs = [v.slab for v in views]
        pictures = sc.scan_pictures(layers, frame, None if coordinate == NONE else coordinate,
                                    steps, line,
                                    text=lambda s: sc.step_text(coordinate, s, ft),
                                    which=self.scanned_layers(), projections=projections,
                                    slabs=slabs)
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
            self._say("recording cancelled")
            return
        finally:
            progress.close()
        self._say(f"recorded {len(steps)} steps to {Path(written).name}")
        what = "view path" if coordinate == NONE else coordinate + (
            ", view path" if projections is not None else "")
        self.session.log("record scan", f"{what}, {len(steps)} steps to {written}")
