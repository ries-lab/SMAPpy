"""A window moved through the data while one looks: the Render tab's player.

*File/Export/Stack* writes a movie; this shows one, live, in the 2D and the
3D window at once, through any coordinate: the frame or the time (bleaching,
a structure building up, something moving), z (slices through a cell), any
column of the table (photons, a fit's quality, an id), or the position along
or across a line ROI -- which with the 3D window following the line moves a
thin slab through a structure seen from the side.

**One narrowing.**  What is shown is each layer's own filter and, on top of
it, ``lo <= value < hi`` of the scanned coordinate (`scanned`): the
snapshots the windows render are narrowed, not the layers, so stopping
gives back exactly what was there, no filter is touched, and nothing a
plugin reads changes.  It is `movie.restrict` with the value worked out per
layer, which the line's coordinates need: they are not a column.

**Steps.**  A *window* of the coordinate is shown, moved by a *step*
(`scan_steps`); a step smaller than the window overlaps them, which is what
makes a scan smooth rather than a slide show -- SMAP's movie's running
window.

**Recorded** frames are drawn by `picture.render_picture`, the renderer of
*Export Image* and *Export Stack*, through the window's own frame (its view,
its 3D projection), so a recorded movie is what the window shows, at the
window's pixel, and is written by `movie.write_stack`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .layer_state import LayerSnapshot
from .movie import Step

#: the coordinates that are not columns: along and across a line ROI, in nm
#: from its middle
ALONG = "along the line"
ACROSS = "across the line"
LINE_COORDINATES = (ALONG, ACROSS)
#: the steps of a scan when none is set: the window a twentieth of the
#: range, moved by a quarter of itself
AUTO_WINDOWS = 20
AUTO_OVERLAP = 4


@dataclass(frozen=True)
class Scan:
    """What is shown now: ``lo <= coordinate < hi``.  ``line`` is the line
    ROI the line's coordinates are measured on."""
    coordinate: str
    lo: float
    hi: float
    line: Optional[object] = None        # a `regions.Region` of kind "line"

    def __str__(self) -> str:
        return f"{self.coordinate} {self.lo:g} to {self.hi:g}"


def line_frame(line) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A line ROI's middle and its unit vectors along and across it."""
    p0, p1 = line.ends()
    d = p1 - p0
    along = d / (np.hypot(*d) or 1.0)
    return (p0 + p1) / 2, along, np.array([-along[1], along[0]])


def values(locs, coordinate: str, line=None) -> np.ndarray:
    """The scanned coordinate of every row of ``locs``: a column, or the
    position along or across ``line`` from its middle, in nm."""
    if coordinate in LINE_COORDINATES:
        if line is None or getattr(line, "kind", None) != "line":
            raise ValueError(f"{coordinate}: draw a line ROI first")
        mid, along, across = line_frame(line)
        unit = along if coordinate == ALONG else across
        x = np.asarray(locs["x_nm"], np.float64)
        y = np.asarray(locs["y_nm"], np.float64)
        return (x - mid[0]) * unit[0] + (y - mid[1]) * unit[1]
    if coordinate not in locs:
        raise KeyError(f"no column {coordinate!r}; the table has "
                       + ", ".join(locs.columns))
    return np.asarray(locs[coordinate], np.float64)


class ValueCache:
    """Each layer table's scanned values, worked out once per table, scan
    coordinate and line: a scan steps through the same values hundreds of
    times, and a line's coordinates are two multiplications of the whole
    table each."""

    def __init__(self, size: int = 8):
        self._found: Dict[tuple, np.ndarray] = {}
        self._size = size

    def __call__(self, locs, coordinate: str, line=None) -> np.ndarray:
        key = (id(locs), len(locs), coordinate,
               None if line is None else tuple(np.round(line.points, 6).ravel()))
        found = self._found.get(key)
        if found is None:
            if len(self._found) >= self._size:
                self._found.clear()
            found = self._found[key] = values(locs, coordinate, line)
        return found


def scanned(layers: Sequence[LayerSnapshot], scan: Optional[Scan],
            cache: Optional[ValueCache] = None) -> Tuple[LayerSnapshot, ...]:
    """The layers showing only what ``scan`` lets through, on top of their
    own filters; image layers and a layer without the column as they are
    -- a second file without z is not emptied by a scan of z."""
    if scan is None:
        return tuple(layers)
    get = cache or values
    out = []
    for layer in layers:
        if not layer.visible or layer.is_image or layer.locs is None:
            out.append(layer)
            continue
        try:
            v = get(layer.locs, scan.coordinate, scan.line)
        except KeyError:
            out.append(layer)
            continue
        keep = (v >= scan.lo) & (v < scan.hi)
        mask = layer.mask
        mask = keep if mask is None else np.asarray(mask, bool) & keep[:len(mask)]
        out.append(replace(layer, mask=mask, known_indices=None, memo={}))
    return tuple(out)


def shown_range(layers: Sequence[LayerSnapshot], coordinate: str, line=None,
                cache: Optional[ValueCache] = None) -> Tuple[float, float]:
    """The lowest and highest value of the coordinate among what the visible
    layers show (their filters), over which a scan runs."""
    get = cache or values
    lo, hi = np.inf, -np.inf
    for layer in layers:
        if not layer.visible or layer.is_image or layer.locs is None:
            continue
        try:
            v = get(layer.locs, coordinate, line)
        except KeyError:
            continue
        if layer.mask is not None:
            v = v[np.asarray(layer.mask, bool)[:len(v)]]
        v = v[np.isfinite(v)]
        if len(v):
            lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
    if not np.isfinite(lo):
        raise ValueError(f"nothing shown has {coordinate}")
    return lo, hi


def auto_window(lo: float, hi: float, integer: bool = False) -> Tuple[float, float]:
    """``(window, step)`` when none is set: a twentieth of the range, moved
    by a quarter of itself; whole numbers for frames."""
    window = (hi - lo) / AUTO_WINDOWS or 1.0
    step = window / AUTO_OVERLAP
    if integer:
        window, step = max(1.0, round(window)), max(1.0, round(step))
    return window, step


def scan_steps(lo: float, hi: float, window: float, step: float,
               integer: bool = False) -> List[Step]:
    """Windows of ``window``, one every ``step``, from ``lo`` until a window
    reaches ``hi``.  For a whole-number coordinate (frames) the windows end
    one past ``hi``, so the last frame is shown."""
    if not window > 0 or not step > 0:
        raise ValueError("the window and the step must be above 0")
    top = hi + 1.0 if integer else hi
    n = max(1, int(np.ceil((top - lo - window) / step - 1e-9)) + 1)
    return [Step(lo + k * step, lo + k * step + window) for k in range(n)]


def is_integer(layers: Sequence[LayerSnapshot], coordinate: str) -> bool:
    """Whether the coordinate is whole numbers in the tables (frames, ids)."""
    if coordinate in LINE_COORDINATES:
        return False
    for layer in layers:
        if layer.visible and layer.locs is not None and coordinate in layer.locs:
            return np.asarray(layer.locs[coordinate]).dtype.kind in "iub"
    return False


def step_text(coordinate: str, step: Step, frame_time_ms: Optional[float] = None) -> str:
    """What a recorded frame says in its corner."""
    if coordinate == "frame" and frame_time_ms:
        return f"{step.lo * frame_time_ms / 1000:.1f}-{step.hi * frame_time_ms / 1000:.1f} s"
    if coordinate == "frame":
        return f"frames {step.lo:.0f}-{step.hi - 1:.0f}"
    unit = " nm" if coordinate in LINE_COORDINATES or coordinate.endswith("_nm") else ""
    name = coordinate[:-3] if coordinate.endswith("_nm") else coordinate
    return f"{name} {step.lo:.4g} to {step.hi:.4g}{unit}"


def scan_pictures(layers: Sequence[LayerSnapshot], frame, coordinate: str,
                  steps: Sequence[Step], line=None, text: Optional[Callable] = None,
                  pixelsize: Optional[float] = None) -> Iterator[np.ndarray]:
    """Each step's picture, as the window ``frame`` (a `picture.ViewFrame`)
    shows it: the 2D window's view and panels, or the 3D window's
    projection.  8-bit RGB, in order, for `movie.write_stack`."""
    from .picture import frame_fov, render_picture, stamp
    fov = frame_fov(frame, "view", pixelsize)
    cache = ValueCache()
    white = any(l.display.white_background for l in layers if l.visible)
    for step in steps:
        shown = scanned(layers, Scan(coordinate, step.lo, step.hi, line), cache)
        if frame.projection is not None:
            picture, = render_picture(shown, fov, what="composite", projection=frame.projection,
                                      slab=frame.slab)
        else:
            picture, = render_picture(shown, fov, what="view", panels=frame.panels,
                                      stacked=frame.stacked)
        image = picture.image
        if text is not None:
            image = stamp(image, text(step), white)
        yield image
