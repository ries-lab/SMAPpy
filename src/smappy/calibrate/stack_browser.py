"""The bead stacks as they were recorded, one plane at a time: the PSF calibration window's *Stacks* page.

Looking at the stacks before calibrating answers what no setting can: are
these beads, are they in focus somewhere in the stack, are there clumps, is
the chip split where it was thought to be.  So the page is there as soon as
files are added, and *Detect beads* rings what it found on it -- before an
afternoon's learning is started on the wrong beads.

One file at a time, with a slider over the files and one over the planes,
rather than a tab per file: a folder of thirty acquisitions is thirty tabs.
A file is read when it is shown (two are kept), not when it is added, so
adding a large folder costs nothing until one looks.  Only widgets: the rings
come from the methods (`psf_methods.Detected.positions`) and their colours
from the bead table's states.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpinBox, QVBoxLayout,
                               QWidget)

#: a ring's colour by the bead's state; a bead not yet calibrated is "found"
COLOURS = {"found": "#00e5ff", "used": "#1f77b4", "accepted": "#1f77b4", "held out": "#2ca02c",
           "excluded": "#7f7f7f", "manual exclusion": "#7f7f7f", "candidate": "#00e5ff", "not drawn": "#bbbbbb"}
REJECTED = "#d62728"
#: files kept read
CACHED = 2


class StackBrowser(QWidget):
    """A file slider, a plane slider and the plane, with the detected beads ringed and numbered."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.figure import Figure

        from ..gui.canvas import canvas_class
        self.acquisitions: list = []
        self._cache: dict = {}
        self.positions: list = []            # (stack, x, y, bead), stacks as detection read them
        self.sources: list = []              # detection's per-stack sources, to place its stack indices
        self.states: dict = {}
        self.selected: set = set()
        self.dz_nm = None                    # the detection's z step; None: the files' own
        self.box_px = 15                     # a ring's diameter: the bead box
        self.movie = False                   # movies (in situ): frames read one at a time, rings on their frame
        self._sources: dict = {}             # movie mode: the open acquisitions
        self.on_pick = None                  # called with a bead when its ring is clicked
        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = canvas_class()(self.figure)
        self.canvas.mpl_connect("pick_event", self._picked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel("add bead stacks: they are shown here")
        self.title.setStyleSheet("color: gray")
        # a long path or error wraps or is cut, never widens the window
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.title)
        layout.addWidget(self.canvas, 1)

        def row(caption, tooltip):
            line = QHBoxLayout()
            line.addWidget(QLabel(caption))
            slider = QSlider(Qt.Horizontal)
            slider.setToolTip(tooltip)
            slider.valueChanged.connect(self.draw)
            label = QLabel("")
            label.setStyleSheet("color: gray")
            label.setMinimumWidth(170)
            line.addWidget(slider, 1)
            line.addWidget(label)
            layout.addLayout(line)
            return line, slider, label
        line, self.file_slider, self.file_label = row("file", "which of the added files")
        self.stack_box = QSpinBox(minimum=1, maximum=1)
        self.stack_box.setToolTip("an acquisition of several stacks (channels, positions): which one")
        self.stack_box.valueChanged.connect(self.draw)
        line.insertWidget(2, self.stack_box)
        line, self.plane_slider, self.plane_label = row("plane", "which plane of the stack, or frame of the movie; "
                                                                 "the mouse wheel too")
        self.plane_caption = line.itemAt(0).widget()
        self.log_box = QCheckBox("log contrast")
        self.log_box.setToolTip("a logarithmic scale, so that faint beads and the rings of bright ones show")
        self.log_box.toggled.connect(self.draw)
        layout.addWidget(self.log_box)

    # ----------------------------------------------------------------- data
    def set_paths(self, paths, dz_nm=None) -> None:
        """The files to browse, as the calibration will read them (`input.discover_acquisitions`), with
        the detection's z step (None: the files' own)."""
        if dz_nm != self.dz_nm:
            self.dz_nm, self._cache = dz_nm, {}
        from .input import discover_acquisitions
        try:
            found = [str(p) for p in discover_acquisitions(paths)] if paths else []
        except Exception as error:                        # a path gone since: said, not fatal
            found = []
            self.title.setText(f"{type(error).__name__}: {error}")
        if found != self.acquisitions:
            self.acquisitions = found
            self._cache = {k: v for k, v in self._cache.items() if k in found}
            self.file_slider.blockSignals(True)
            self.file_slider.setRange(0, max(len(found) - 1, 0))
            self.file_slider.blockSignals(False)
        self.draw()

    def set_movie(self, movie: bool) -> None:
        """Movies rather than bead stacks: a frame is read when it is shown, never the whole file, and a
        ring is drawn on its own frame, (stack, x, y, unit, frame) in `set_marks`."""
        if movie != self.movie:
            self.movie, self._cache, self._sources = movie, {}, {}
            self.plane_caption.setText("frame" if movie else "plane")
            self.draw()

    def set_marks(self, positions, sources=(), states: Optional[dict] = None) -> None:
        """Ring the beads at ``positions`` ((stack, x, y, bead), the stack indexing ``sources``), coloured
        by ``states`` ({bead: state}); nothing to clear them."""
        self.positions, self.sources, self.states = list(positions or []), list(sources or []), dict(states or {})
        self.draw()

    def select(self, beads) -> None:
        self.selected = set(beads)
        self.draw()

    def stacks(self, index: int) -> list:
        """The stacks of acquisition ``index``, read once and kept for a while."""
        from .input import acquisition_kind, read_bead_stacks
        path = self.acquisitions[index]
        if path not in self._cache:
            # never read a whole movie as one stack: every page of it in memory, then a long refusal
            kind, n = acquisition_kind(path)
            if kind == "movie":
                raise ValueError(f"a movie of {n} frames, not a bead z-stack: in situ learns from movies")
            while len(self._cache) >= CACHED:
                self._cache.pop(next(iter(self._cache)))
            try:
                self._cache[path] = read_bead_stacks(path, self.dz_nm)
            except ValueError as error:
                if "z spacing" not in str(error):
                    raise
                # a file that does not say its z step is still worth looking at: its planes, unnamed
                self._cache[path] = read_bead_stacks(path, 1.0)
                for st in self._cache[path]:
                    st.axes = dict(st.axes, unknown_dz=True)
        return self._cache[path]

    def _stack_index(self, acquisition: int, k: int) -> Optional[int]:
        """The detection's stack index of the ``k``-th stack of ``acquisition``: its sources come in the
        acquisitions' order, a run of entries per acquisition."""
        group, previous = -1, object()
        for g, source in enumerate(self.sources):
            path = source.get("path") if isinstance(source, dict) else None
            if path != previous:
                group, previous, first = group + 1, path, g
            if group == acquisition:
                return first + k
        return None if self.sources else (acquisition if k == 0 else None)

    # ------------------------------------------------------------- drawing
    def draw(self) -> None:
        self.figure.clear()
        if not self.acquisitions:
            self.title.setText("add bead stacks: they are shown here")
            self.canvas.draw_idle()
            return
        a = int(np.clip(self.file_slider.value(), 0, len(self.acquisitions) - 1))
        if self.movie:
            self._draw_frame(a)
            return
        try:
            stacks = self.stacks(a)
        except Exception as error:
            self.title.setText(f"could not read {self.acquisitions[a]}: {type(error).__name__}: {error}")
            self.canvas.draw_idle()
            return
        self.plane_slider.setEnabled(True)
        self.stack_box.blockSignals(True)
        self.stack_box.setMaximum(max(len(stacks), 1))
        self.stack_box.setVisible(len(stacks) > 1)
        self.stack_box.blockSignals(False)
        k = min(self.stack_box.value() - 1, len(stacks) - 1)
        st = stacks[k]
        images = np.asarray(st.images)
        self.plane_slider.blockSignals(True)
        if self.plane_slider.maximum() != len(images) - 1:
            self.plane_slider.setRange(0, len(images) - 1)
            self.plane_slider.setValue(len(images) // 2)
        self.plane_slider.blockSignals(False)
        z = int(np.clip(self.plane_slider.value(), 0, len(images) - 1))
        plane = images[z].astype(float)
        ax = self.figure.subplots()
        low, high = np.percentile(plane, (1, 99.95))
        if self.log_box.isChecked():
            shown = np.log1p(np.maximum(plane - low, 0))
            ax.imshow(shown, cmap="gray", vmin=0, vmax=max(float(shown.max()), 1e-9))
        else:
            ax.imshow(plane, cmap="gray", vmin=low, vmax=max(high, low + 1))
        stack = self._stack_index(a, k)
        here = [p for p in self.positions if p[0] == stack]
        from matplotlib.patches import Circle
        radius = self.box_px / 2
        for s, x, y, bead in here:
            state = self.states.get(bead, "found")
            colour = COLOURS.get(state, REJECTED)
            ring = Circle((x, y), radius, fill=False, edgecolor=colour, picker=True,
                          linewidth=2.5 if bead in self.selected else 1.0)
            ring.pair_ids = np.array([bead])
            ax.add_patch(ring)
            ax.text(x + radius, y - radius, str(bead), color=colour, fontsize="x-small", ha="left", va="bottom")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        dz = (f", {float(st.z_nm[z] - st.z_nm[len(st.z_nm) // 2]):+.0f} nm from the middle"
              if len(st.z_nm) > 1 and not st.axes.get("unknown_dz") else "")
        self.plane_label.setText(f"plane {z + 1}/{len(images)}{dz}")
        self.file_label.setText(f"file {a + 1}/{len(self.acquisitions)}")
        found = len({p[3] for p in here})
        self.title.setText(f"{self.acquisitions[a]}" + (f" - stack {k + 1}/{len(stacks)}" if len(stacks) > 1 else "")
                           + (f" - {found} beads ringed" if self.positions else ""))
        self.canvas.draw_idle()

    def _draw_frame(self, a: int) -> None:
        """Movie mode: only the frames molecules were found on, once they are, each read when shown; before
        detection the first frame alone, to see that the movie is the right one.  A slider over every frame
        of a movie of 300 000 was a slider nobody could aim, over frames mostly never read."""
        from matplotlib.patches import Circle

        from ..io.tiff import open_stack
        path = self.acquisitions[a]
        found = sorted({p[4] for p in self.positions if p[0] == a and len(p) > 4})
        try:
            if path not in self._sources:
                while len(self._sources) >= CACHED:
                    self._sources.pop(next(iter(self._sources)))
                self._sources[path] = open_stack(path)
            src = self._sources[path]
            shown = found or [0]
            self.plane_slider.blockSignals(True)
            if self.plane_slider.maximum() != len(shown) - 1:
                self.plane_slider.setRange(0, len(shown) - 1)
                self.plane_slider.setValue(0)
            self.plane_slider.setEnabled(len(shown) > 1)
            self.plane_slider.blockSignals(False)
            k = int(np.clip(self.plane_slider.value(), 0, len(shown) - 1))
            f = shown[k]
            plane = np.asarray(src.frame(f), float)
        except Exception as error:
            self.title.setText(f"could not read {path}: {type(error).__name__}: {error}")
            self.canvas.draw_idle()
            return
        self.stack_box.setVisible(False)
        ax = self.figure.subplots()
        low, high = np.percentile(plane, (1, 99.95))
        if self.log_box.isChecked():
            shown = np.log1p(np.maximum(plane - low, 0))
            ax.imshow(shown, cmap="gray", vmin=0, vmax=max(float(shown.max()), 1e-9))
        else:
            ax.imshow(plane, cmap="gray", vmin=low, vmax=max(high, low + 1))
        here = [p for p in self.positions if p[0] == a and len(p) > 4 and p[4] == f]
        radius = self.box_px / 2
        for _, x, y, unit, _ in here:
            colour = COLOURS.get(self.states.get(unit, "found"), REJECTED)
            ring = Circle((x, y), radius, fill=False, edgecolor=colour, picker=True,
                          linewidth=2.5 if unit in self.selected else 1.0)
            ring.pair_ids = np.array([unit])
            ax.add_patch(ring)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        self.plane_label.setText(f"{k + 1}/{len(found)}: frame {f + 1} of {src.n_frames}" if found
                                 else f"frame 1 of {src.n_frames}")
        self.file_label.setText(f"file {a + 1}/{len(self.acquisitions)}")
        self.title.setText(f"{path} - {len(here)} molecules ringed on this frame; "
                           f"{len(found)} frames with molecules" if found
                           else f"{path} - its first frame: Detect molecules shows the frames they are on")
        self.canvas.draw_idle()

    def _picked(self, event) -> None:
        ids = getattr(event.artist, "pair_ids", None)
        if ids is not None and len(ids) and self.on_pick is not None:
            self.on_pick(int(ids[0]))

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self.plane_slider.setValue(int(np.clip(self.plane_slider.value() - steps, 0, self.plane_slider.maximum())))
