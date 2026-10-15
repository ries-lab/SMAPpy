"""Analysis ROIs: their geometry, their history and what was measured in them.

All geometry is in nm.  The data an ROI sees is the session's: one source per
loaded file, the filters and grouping of a layer, so what an ROI sees is what
the image shows, and selection always uses the same filter as rendering,
never the visible field of view.  The project's state travels in the
localization file's metadata, so ROIs and their results are saved with
*File > Save* and come back with the file.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from ..group import GROUP_COLUMNS, GroupSettings
from ..layer_state import Tables, ViewState
from ..locs import Localizations
from ..region_table import RegionTable
from ..regions import Region
from ..render import RenderAxes


def json_text(value):
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Cannot save {type(obj).__name__}")
    return json.dumps(value, default=default, sort_keys=True)


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def point(value):
    p = np.asarray(value, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError("A position must contain two finite coordinates in nm")
    return p.tolist()


def polygon_vertices(value):
    v = np.asarray(value, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3 or not np.isfinite(v).all():
        raise ValueError("A polygon needs at least three finite xy vertices")
    # Use local coordinates to avoid cancellation far from the file origin.
    w = v - v[0]
    area = np.sum(w[:, 0] * np.roll(w[:, 1], -1) -
                  w[:, 1] * np.roll(w[:, 0], -1))
    if abs(area) < 1e-10:
        raise ValueError("A polygon must enclose nonzero area")
    return v.tolist()


def rows_in(locs, index, region, keep=None):
    """The rows of ``locs`` inside ``region``, sorted, through its spatial
    index: only the rows in the region's bounding box are tested.  ``keep``,
    over the whole table, narrows them first (a file of several)."""
    x0, y0, x1, y1 = region.bounds
    candidates = index.query(x0, x1, y0, y1)
    if keep is not None:
        candidates = candidates[keep[candidates]]
    x = np.asarray(locs["x_nm"][candidates], dtype=float)
    y = np.asarray(locs["y_nm"][candidates], dtype=float)
    return np.sort(candidates[region.mask(x, y)])


def changes_table(step) -> bool:
    """Does this step hand back a new table (`Plugin.changes_locs_for`)?  In
    a region's run its table is what the steps after it measure."""
    try:
        return bool(step.plugin.changes_locs_for(step.settings))
    except Exception:
        return bool(getattr(step.plugin, "changes_locs", False))


def region_entry(result, plugin=None, keep: bool = False) -> Dict:
    """What one run in one region stores: its `Result.values`, a number
    each, and its `Result.arrays`.

    A value that is not a number (or a string, a flag, None) is refused
    rather than stored: the values are the region table's columns, and a
    list in a column is a table nobody can histogram.  An array belongs in
    `arrays`, kept here as JSON lists beside the values; `RegionTable`
    writes them to the file end to end.

    With ``keep``, what ``plugin.keep(result)`` returns -- the data its
    figure is drawn from -- goes with the row as ``kept``, so the figure can
    come back (`Plugin.restore`) without measuring again.
    """
    values = {}
    for name, value in (result.values or {}).items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise TypeError(f"value {name!r} is a {type(value).__name__}, not a "
                            "number: put it in Result.arrays")
        values[name] = value
    entry = {"values": values}
    if result.arrays:
        entry["arrays"] = {name: np.asarray(a).tolist()
                           for name, a in result.arrays.items()}
    if keep and plugin is not None:
        try:
            kept = plugin.keep(result)
        except Exception:
            kept = None             # the numbers stand; only the replot is lost
        if kept:
            entry["kept"] = kept
    return json.loads(json_text(entry))


@dataclass
class ROI:
    file_id: str
    center: list
    id: str = field(default_factory=lambda: uuid4().hex)
    # what the user sees and a column carries (`roi_id`): given once, from the
    # project's counter, and never reused, so removing ROI 3 leaves ROI 5 as
    # 5 -- in the list, on the localizations and in the results alike.  The
    # uuid stays the key the results are stored under.
    number: int = 0
    polygon: Optional[list] = None
    # named lines, ``{name: [[x0, y0], [x1, y1]]}`` in absolute nm, start
    # then end: an orientation or a distance drawn by hand (SMAP's ``line1``
    # and ``line2``, with names).  Replaced, never edited in place, as
    # `annotations` is
    lines: dict = field(default_factory=dict)
    use: bool = True
    comment: str = ""
    origin: dict = field(default_factory=lambda: {"method": "manual"})
    # what the user said of it, ``{list: keyword}`` from the project's
    # `ROIProject.annotation_lists`; replaced, never edited in place (see
    # `copy_rois`), and no part of what is measured, so not of a signature
    annotations: dict = field(default_factory=dict)

    def region(self, shape: str, size_nm: float) -> Region:
        """This ROI's shape: its own polygon, or the project's ``shape`` of
        ``size_nm`` around its centre -- which the ROI does not know, since
        changing it is one edit for every ROI of the project.  So the shape
        is worked out when asked for rather than held."""
        return Region.site(self.center, shape, size_nm, self.polygon)

    @classmethod
    def from_dict(cls, saved: dict) -> "ROI":
        """An ROI as a file kept it.  Files from before the review flag went
        carry ``reviewed``: it was never shown in the GUI, where every ROI
        counted as reviewed, so it is dropped and ``use`` alone decides.
        Files from before the named lines carry one ``direction``: it is the
        line called ``direction``."""
        names = {f.name for f in fields(cls)}
        roi = cls(**{k: v for k, v in saved.items() if k in names})
        point(roi.center)
        if roi.polygon is not None:
            polygon_vertices(roi.polygon)
        lines = {str(k): line_points(v) for k, v in (roi.lines or {}).items()}
        if saved.get("direction") is not None and DIRECTION not in lines:
            lines[DIRECTION] = line_points(saved["direction"])
        roi.lines = lines
        return roi


# Past this nobody is walking back through a list of edits.
MAX_ROI_STEPS = 20


@dataclass
class ROIStep:
    """One edit of the ROIs, and the ROIs as they were before it."""
    label: str
    rois: Dict[str, "ROI"]


# The line a new project offers, and the name an old file's one line had.
DIRECTION = "direction"


def line_points(line) -> list:
    """``[[x0, y0], [x1, y1]]`` as floats, or a ValueError."""
    a = np.asarray(line, dtype=float)
    if a.shape != (2, 2) or not np.all(np.isfinite(a)):
        raise ValueError(f"a line is two points, start and end: {line!r}")
    return a.tolist()


def line_angle_length(line) -> tuple:
    """A line's angle in degrees, start to end, and its length in nm.

    The angle is SMAP's (``pos2angle``): ``atan2(dy, dx)``, -180 to 180,
    counted from +x towards +y.  The images show y downwards, so a positive
    angle turns clockwise on the screen."""
    (x0, y0), (x1, y1) = np.asarray(line, dtype=float)
    return (float(np.degrees(np.arctan2(y1 - y0, x1 - x0))),
            float(np.hypot(x1 - x0, y1 - y0)))


def line_columns(roi: "ROI") -> Dict[str, float]:
    """An ROI's lines as numbers of its row: ``<name>_angle_deg`` and
    ``<name>_length_nm``."""
    out = {}
    for name, line in roi.lines.items():
        angle, length = line_angle_length(line)
        out[f"{name}_angle_deg"] = angle
        out[f"{name}_length_nm"] = length
    return out


def parse_line_names(text: str) -> List[str]:
    """``"direction, axis"`` -> ``["direction", "axis"]``: commas or new
    lines, repeats and blanks dropped.  A name is a column's prefix, so it
    keeps to letters, digits and ``_`` (a space becomes ``_``)."""
    names: List[str] = []
    for word in text.replace("\n", ",").split(","):
        word = "_".join(word.split())
        word = "".join(c for c in word if c.isalnum() or c == "_")
        if word and word not in names:
            names.append(word)
    return names


# The lists a new project offers: one, so that the keys do something before
# anyone has written a list of their own.
DEFAULT_ANNOTATION_LISTS = {"quality": ["good", "bad", "unsure"]}


def parse_annotation_lists(text: str) -> Dict[str, List[str]]:
    """``"quality: good, bad\nshape: ring, arc"`` -> ``{"quality": [...], ...}``.

    One list a line, its name before the colon and its keywords after,
    separated by commas -- SMAP's list file, with names.  A line without a
    colon is a list of that name with no keywords yet; blank lines and
    repeated keywords are dropped.
    """
    lists: Dict[str, List[str]] = {}
    for line in text.splitlines():
        name, _, words = line.partition(":")
        name = name.strip()
        if not name:
            continue
        keywords = lists.setdefault(name, [])
        for word in words.split(","):
            word = word.strip()
            if word and word not in keywords:
                keywords.append(word)
    return lists


def annotation_lists_text(lists: Dict[str, List[str]]) -> str:
    """The text `parse_annotation_lists` reads back."""
    return "\n".join(f"{name}: {', '.join(words)}" for name, words in lists.items())


def annotation_columns(roi: ROI, values: Dict[str, Any]) -> Dict[str, Any]:
    """What was said of an ROI by hand, as columns of its row in the results
    table: its keywords, named after their lists -- ``annotation.<list>``
    where a measurement already has the name -- as text, so the numeric
    summaries pass over them; and its lines' angles and lengths
    (`line_columns`), numbers, so Filter, Sort and the summaries read them."""
    out: Dict[str, Any] = {(name if name not in values else f"annotation.{name}"): keyword
                           for name, keyword in roi.annotations.items()}
    out.update({(name if name not in values else f"annotation.{name}"): value
                for name, value in line_columns(roi).items()})
    return out


def copy_rois(rois: Dict[str, "ROI"]) -> Dict[str, "ROI"]:
    """A copy that later edits cannot reach.  Shallow per ROI, which is
    enough because an ROI is only ever edited by assigning its fields --
    ``roi.use = ...``, ``roi.polygon = [...]``, never ``roi.center[0] = ...``
    -- so the lists and the origin dict can be shared."""
    return {roi_id: replace(roi) for roi_id, roi in rois.items()}


class ROIHistory:
    """Undo and redo for the set of ROIs, apart from the session's undo.

    The session's undo is for tables (`smappy.undo` says why only tables);
    this one is for the ROI manager's edits: a segmenter's run, an ROI
    added, removed, excluded or reshaped.  Not for evaluations, which can be
    run again, and whose results stay with the ROI's id -- so an ROI brought
    back by an undo brings its numbers back with it.  Whole copies of the
    ROIs rather than diffs: an ROI is a few small fields, and a step of a
    few thousand costs less than a millisecond's worth of thinking about
    what a diff would have to cover.
    """

    def __init__(self, max_steps: int = MAX_ROI_STEPS):
        self.max_steps = max_steps
        self._undo: List[ROIStep] = []
        self._redo: List[ROIStep] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> Optional[str]:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._redo[-1].label if self._redo else None

    def push(self, step: ROIStep) -> None:
        self._undo.append(step)
        del self._undo[:-self.max_steps]
        self._redo.clear()

    def undo(self, rois: Dict[str, "ROI"]) -> Optional[ROIStep]:
        """Put the ROIs before the last edit into `rois`; the edit undone."""
        return self._travel(rois, self._undo, self._redo)

    def redo(self, rois: Dict[str, "ROI"]) -> Optional[ROIStep]:
        return self._travel(rois, self._redo, self._undo)

    @staticmethod
    def _travel(rois, source, target) -> Optional[ROIStep]:
        if not source:
            return None
        step = source.pop()
        target.append(ROIStep(step.label, copy_rois(rois)))
        # in place: whoever holds the project's dict keeps holding the ROIs
        rois.clear()
        rois.update(copy_rois(step.rois))
        return step

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


FORMAT_VERSION = 1

# Columns the session writes onto the table itself, rather than the data
# bringing them: `Session.load` adds `filenumber`, and grouping adds the group
# columns in place.  Whether they are there yet says nothing about the
# localizations, so the fingerprint counts them as always present -- present
# rather than absent so that the fingerprints already saved, which were taken
# from loaded, usually grouped tables, still match.
BOOKKEEPING = ("filenumber",) + GROUP_COLUMNS


class Source:
    """What the project needs of a source; the data stays in the session."""

    def __init__(self, source_id: str, name: str, path: Optional[str], number: int):
        self.id = source_id
        self.name = name
        self.path = path
        self.number = number          # the table's `filenumber`
        self.state: Optional[ViewState] = None
        self.fingerprint = ""


class ROIProject:
    """The ROIs, their results and the steps that made them, over the
    session's files (`Session.rois`).

    The data stays in the session: a plugin may replace the table (a drift
    correction, a fit), and the project follows (`sync`).  ``layer`` says
    which layer's filter and grouping the ROIs follow; the layer's *file*
    choice is deliberately ignored, since an ROI already knows its file.  A
    script makes ROIs through a headless session, ``Session(locs).rois``, so
    it sees the filters and grouping the GUI would.
    """

    def __init__(self, session):
        self.session = session
        self.layer = 0
        self.preview_nm = 0.0        # the ROI image's width; 0 = three ROI widths
        self._states: Dict[str, ViewState] = {}
        self._built = None            # what the sources were built from
        # (id and length of a table, file number) -> that file's rows of it:
        # computed once, not once per ROI measured
        self._file_masks: Dict[tuple, np.ndarray] = {}
        self.sources: Dict[str, Source] = {}
        self.rois: Dict[str, ROI] = {}
        self.size_nm = 300.0  # circle diameter or square side length
        self.shape = "circle"
        self.tile_nm = 0.0    # a grid over each file for a systematic walk; 0 = none
        # the layer's, which `sync` puts here
        self.filters: Dict[str, tuple] = {}
        self.grouped = False
        # The results, one table per evaluation by its name, a row per ROI
        # (`RegionTable`): the newest result of each evaluation on each ROI,
        # with the step that made it.  Kept in the order they were last
        # written, so the newest is last.  Rows of ROIs that were removed
        # stay until the project is saved (`tables_dict`), so an undo brings
        # the ROI back with its numbers.
        self.tables: Dict[str, RegionTable] = {}
        # What was run, newest last: the pipeline, when, and whether a chain
        # or one site's re-evaluation did it -- a log, without the results,
        # which are in `tables`
        self.runs = []
        self.navigation = {}  # GUI state only; never used for extraction
        # ROI ids in the order they are listed and their results reported
        # (`sort`); empty: by number.  The numbers themselves never change.
        self.order: List[str] = []
        # the keyword lists an ROI is annotated from (`ROI.annotations`)
        self.annotation_lists: Dict[str, List[str]] = {
            k: list(v) for k, v in DEFAULT_ANNOTATION_LISTS.items()}
        # the names a line is drawn under (`ROI.lines`)
        self.line_names: List[str] = [DIRECTION]
        # The per-region steps, a `chain.Block`: what the steps window edits
        # and runs, and what the ROI manager measures an ROI with.  It
        # travels with the project because the columns of a results table
        # mean nothing without knowing which steps produced them, with what
        # parameters.  No steps: the newest run's, or the default ones
        # (`resolved`).
        from ..chain import Block
        self.block = Block()
        # What a segmenter's Preview proposed and nobody has added yet:
        # ``{"file_id", "method", "sites": [{"center", "use"}], "replaces":
        # [roi ids]}``, or None.  Drawn by the manager and never saved -- it
        # exists so that a finder's settings can be tried without each try
        # leaving ROIs behind to be removed by hand.
        self.segment_preview = None
        self.history = ROIHistory()
        # the number the next ROI is given (`ROI.number`): it only counts up
        # and is saved with the ROIs, so an undone ROI's number is not handed
        # to the next one, and its redo brings it back unchanged
        self.next_number = 1

    def set_geometry(self, size_nm, shape=None):
        size = positive(size_nm, "ROI size")
        shape = shape or self.shape
        if shape not in ("circle", "square"):
            raise ValueError("Shape must be circle or square")
        self.size_nm, self.shape = size, shape

    def set_tiles(self, size_nm):
        """Cover each file with a grid of squares this wide; 0 turns it off."""
        self.tile_nm = 0.0 if not size_nm else positive(size_nm, "Tile size")

    def tiles(self, file_id):
        """The grid over one file, row by row: ``(x0, y0, x1, y1)`` in nm.

        Tiles are not stored: they follow the file's extent and the one size,
        so they never go stale.  What they are for is a systematic walk --
        every part of the data looked at once, in a fixed order, rather than
        whichever bright spot the eye landed on.
        """
        if not self.tile_nm or file_id not in self.sources:
            return []
        x0, y0, x1, y1 = self.state(file_id).index.bounds
        size = self.tile_nm
        nx = max(1, int(np.ceil((x1 - x0) / size)))
        ny = max(1, int(np.ceil((y1 - y0) / size)))
        return [(x0 + i * size, y0 + j * size, x0 + (i + 1) * size, y0 + (j + 1) * size)
                for j in range(ny) for i in range(nx)]

    def tile_at(self, file_id, x, y):
        """The index of the tile a point falls in, or None outside the grid.

        Tiles share their edges, so the ranges are half open: a point on a
        boundary belongs to the tile it opens, and only the far edge of the
        last row and column falls back to the tile it closes.
        """
        tiles = self.tiles(file_id)
        for i, (tx0, ty0, tx1, ty1) in enumerate(tiles):
            if tx0 <= x < tx1 and ty0 <= y < ty1:
                return i
        for i, (tx0, ty0, tx1, ty1) in enumerate(tiles):
            if tx0 <= x <= tx1 and ty0 <= y <= ty1:
                return i
        return None

    @contextmanager
    def edit(self, label: str):
        """An undoable edit of the ROIs: ``with project.edit("Remove ROI 3"):``.

        Yields the step, whose label may be finished once the edit knows
        what it did; nothing is recorded when the ROIs come out unchanged,
        so a run that found nothing leaves no step to undo.
        """
        step = ROIStep(label, copy_rois(self.rois))
        yield step
        if step.rois != self.rois:
            self.history.push(step)

    def undo(self) -> Optional[str]:
        """Undo the last edit of the ROIs; what it was, or None."""
        step = self.history.undo(self.rois)
        return step.label if step else None

    def redo(self) -> Optional[str]:
        step = self.history.redo(self.rois)
        return step.label if step else None

    def annotate(self, roi_id, name: str, keyword: Optional[str]) -> None:
        """Give an ROI ``keyword`` from list ``name``; None or "" clears it.
        Undoable, as any edit of the ROIs."""
        roi = self.rois[roi_id]
        if keyword and keyword not in self.annotation_lists.get(name, ()):
            raise ValueError(f"{keyword!r} is not in the list {name!r}")
        if (roi.annotations.get(name) or None) == (keyword or None):
            return
        with self.edit(f"Annotate ROI {roi.number}"):
            annotations = {k: v for k, v in roi.annotations.items() if k != name}
            if keyword:
                annotations[name] = keyword
            roi.annotations = annotations        # replaced: see `copy_rois`

    def set_annotation_lists(self, lists: Dict[str, List[str]]) -> None:
        """New keyword lists.  An ROI keeps its keywords even where a list no
        longer has them -- a list rewritten by mistake must not wipe a day's
        annotation -- and they show until it is given another."""
        self.annotation_lists = {str(k): [str(w) for w in v] for k, v in lists.items()}

    def set_line_names(self, names: List[str]) -> None:
        """New line names.  An ROI keeps a line whose name went, as it keeps
        a keyword whose list went."""
        self.line_names = parse_line_names(",".join(str(n) for n in names))

    def set_line(self, roi_id, name: str, line) -> None:
        """Draw the line ``name`` of an ROI; None removes it.  Undoable."""
        roi = self.rois[roi_id]
        line = None if line is None else line_points(line)
        if roi.lines.get(name) == line:
            return
        verb = "Clear" if line is None else "Draw"
        with self.edit(f"{verb} {name} of ROI {roi.number}"):
            lines = {k: v for k, v in roi.lines.items() if k != name}
            if line is not None:
                lines[name] = line
            roi.lines = lines                     # replaced: see `copy_rois`

    def reverse_line(self, roi_id, name: str) -> bool:
        """Swap a line's start and end (its angle turns by 180 degrees);
        False when the ROI has no such line."""
        line = self.rois[roi_id].lines.get(name)
        if line is None:
            return False
        roi = self.rois[roi_id]
        with self.edit(f"Reverse {name} of ROI {roi.number}"):
            roi.lines = {**roi.lines, name: [line[1], line[0]]}
        return True

    def add_roi(self, file_id, center, polygon=None, origin=None):
        if file_id not in self.sources:
            raise KeyError(file_id)
        roi = ROI(file_id, point(center), polygon=(None if polygon is None else
                  polygon_vertices(polygon)))
        if origin is not None:
            roi.origin = json.loads(json_text(origin))
        roi.number = self.next_number
        self.next_number += 1
        self.rois[roi.id] = roi
        return roi

    def numbers(self) -> Dict[str, int]:
        """Each ROI's number (`ROI.number`), by its id."""
        return {roi_id: roi.number for roi_id, roi in self.rois.items()}

    def roi_numbered(self, number) -> Optional[ROI]:
        """The ROI with this number -- a result row's ``roi_id`` -- or None."""
        try:
            number = int(number)
        except (TypeError, ValueError):
            return None
        return next((r for r in self.rois.values() if r.number == number), None)

    def number_rois(self, next_number=None) -> None:
        """Numbers for the ROIs that have none (a file from before they were
        kept), in the order the ROIs were made and after the highest in use,
        and a counter past all of them."""
        start = max([int(next_number or 1)]
                    + [r.number + 1 for r in self.rois.values() if r.number > 0])
        seen = set()
        for roi in self.rois.values():
            if roi.number <= 0 or roi.number in seen:
                roi.number = start
                start += 1
            seen.add(roi.number)
        self.next_number = start

    def move_roi(self, roi_id, center):
        """Translate the entire ROI, including its polygon and lines."""
        roi = self.rois[roi_id]
        center = point(center)
        delta = np.asarray(center) - roi.center
        if roi.polygon is not None:
            roi.polygon = (np.asarray(roi.polygon) + delta).tolist()
        roi.lines = {name: (np.asarray(line) + delta).tolist()
                     for name, line in roi.lines.items()}
        roi.center = center

    def geometry(self, roi):
        """The ROI's shape and lines, as its measurements' signature takes
        them.  The line called ``direction`` keeps the key the one direction
        had, and the others go under ``lines`` only when there are any, so
        the signatures saved before named lines still match."""
        if roi.polygon is not None:
            out = {"center": roi.center, "polygon": roi.polygon}
        else:
            out = {"center": roi.center, "shape": self.shape, "size_nm": self.size_nm}
        out["direction"] = roi.lines.get(DIRECTION)
        others = {k: v for k, v in roi.lines.items() if k != DIRECTION}
        if others:
            out["lines"] = others
        return out

    def region(self, roi) -> Region:
        """The ROI's shape, with this project's shared shape and size."""
        return roi.region(self.shape, self.size_nm)

    def indices(self, roi):
        state = self.state(roi.file_id)
        return rows_in(state.locs, state.index, self.region(roi),
                       keep=np.asarray(state.filter.mask))

    def extract(self, roi):
        """What the ROI shows: the shown table's filtered rows inside it."""
        return self.state(roi.file_id).locs[self.indices(roi)]

    # ------------------------------------------------ measuring in an ROI
    def region_context(self, roi, **extra):
        """The context a plugin measuring this ROI is given.

        The ordinary contract, confined to the ROI (`docs/regions.md`):
        ``ctx.locs`` is the ROI's rows of its file, unfiltered, of the table
        the layer shows (the grouped one cut to the ROI where there is one);
        ``ctx.ungrouped()`` the localizations among them, and
        ``ctx.layer_selection(i)`` each layer's filter over those;
        ``ctx.table(i)`` hands out each layer's table cut the same way;
        ``ctx.region`` is the ROI's shape.  Which rows belong is the ROI's shape alone -- a
        filter is applied inside it, never to decide membership -- and no
        drawn ROI or 3D slab narrows it further: the region replaces them.
        """
        region = self.region(roi)
        return self._context(region, self._region_tables(roi, region),
                             region_id=roi.id, **extra)

    def _context(self, region, tables, **extra):
        """A context over what `_region_tables` cut: one per run, since a
        context keeps what it was told, over tables the steps share."""
        from ..plugins import Context
        pinned, rows, layer, session = tables
        locs, selection = pinned[layer]["ungrouped"]
        return Context(session=session, locs=locs, selection=selection, layer=layer,
                       pinned=pinned, rows=rows, region=region, rois=self, **extra)

    def inputs(self, roi):
        """What a measurement of this ROI depends on: the ROI's own inputs and
        every layer's filters and grouping -- a plugin in an ROI may read any
        layer (`_region_tables`), so a filter changed on any of them makes
        its numbers out of date."""
        state = self.state(roi.file_id)
        out = {"source": self.sources[roi.file_id].fingerprint,
               "geometry": self.geometry(roi), "filters": state.filter.ranges,
               "grouped": self.grouped,
               "group_settings": asdict(self.group_settings) if self.grouped else None}
        from ..region_runs import _layer_inputs
        out["layers"] = [_layer_inputs(lay) for lay in self.session.layers
                         if not lay.is_image]
        return out


    # ------------------------------------------------- what is still current
    def resolved(self, steps=None, block=None):
        """The steps that would run: resolved once, for callers that share them.

        ``block`` -- a `chain.Block`, or a list of its steps -- or the
        project's own, or -- when that has none -- the steps of the newest
        run, which is what a script that handed its steps straight to
        `evaluate` leaves behind.  Falling back to the installed plugins
        instead would report steps nobody ran.
        """
        from . import pipeline as pipeline_module
        from ..chain import Block
        if steps is not None:
            return list(steps)
        if block is None:
            block = self.block
        elif not isinstance(block, Block):
            block = Block(steps=list(block))
        if not block.steps:
            for run in reversed(self.runs):
                if run.get("chain"):
                    continue        # a chain's steps are its own, not the window's
                block = pipeline_module.block_from_run(run.get("pipeline") or [])
                if block.steps:
                    break
        if not block.steps:
            block = pipeline_module.default_block()
        return pipeline_module.block_steps(block)

    def block_steps(self):
        """The project's own block's steps, with no fallback: what the ROI
        manager measures the ROI being looked at with."""
        from . import pipeline as pipeline_module
        return pipeline_module.block_steps(self.block)

    def step_signature(self, step, inputs, before=()):
        """What this step's result depends on: the data *and* the parameters.

        The record used to carry one signature over the ROI's inputs alone,
        so editing an evaluator's parameter left every stored number looking
        perfectly current -- the one case where "out of date" matters most.
        ``before`` are the identities of the steps before it that change the
        region's table (`changes_table`): in a block what a step measures is
        what they left, so editing one of them is editing it.
        """
        if not before:
            return digest({"inputs": inputs, "step": step.identity()})
        return digest({"inputs": inputs, "step": step.identity(),
                       "after": list(before)})

    def signatures(self, steps, inputs):
        """`step_signature` for each step, by label, in order."""
        out, before = {}, []
        for step in steps:
            out[step.label] = self.step_signature(step, inputs, before)
            if changes_table(step):
                before.append(step.identity())
        return out

    def entries(self, roi_id, steps=None):
        """The newest stored entry for each step of this ROI, and its state.

        ``{label: (entry or None, state)}`` where state is "current" (same
        data and same parameters), "stale" (something changed) or
        "missing".  Only the steps that would run are reported: a step taken
        out of the pipeline stops contributing columns.
        """
        steps = self.resolved(steps)
        inputs = json.loads(json_text(self.inputs(self.rois[roi_id])))
        wanted = self.signatures(steps, inputs)
        # by signature as well as by label: a step that was renamed, or moved
        # up the pipeline, measured the same thing and its numbers stand.  The
        # label is what the columns are called, and calling them something
        # else is not a reason to run anything again.
        by_signature: Dict[str, Dict] = {}
        by_label: Dict[str, Dict] = {}
        for label, table in self._newest_first():
            entry = table.entry(roi_id)
            if entry is None:
                continue
            by_label[label] = entry
            if entry.get("signature"):
                by_signature.setdefault(entry["signature"], entry)
        found = {}
        for label, signature in wanted.items():
            if signature in by_signature:
                found[label] = (by_signature[signature], "current")
            elif label in by_label:
                found[label] = (by_label[label], "stale")
            else:
                found[label] = (None, "missing")
        return found

    def evaluations(self, roi_id):
        """Every evaluation this ROI has had that still describes it.

        ``{label: (entry, step)}``: under each name, the newest stored result
        that was computed from the ROI's data as it is now (place, file,
        filter, grouping), and the step that made it -- plugin, version,
        parameters -- as its run recorded it.  The name is what makes two
        evaluations different: the same evaluator renamed and run with other
        settings is a second evaluation beside the first, so the two can be
        compared, and running it again under its old name replaces the old
        numbers.  Whether the GUI's pipeline, a chain or a script ran it does
        not matter: a result belongs to the ROI, and says itself how it was
        made.
        """
        stored = str(roi_id)
        if not any(stored in table for table in self.tables.values()):
            return {}          # its data is not cut out for nothing: 60 us an ROI
        inputs = json.loads(json_text(self.inputs(self.rois[roi_id])))
        data = digest(inputs)
        found = {}
        for label, table in self._newest_first():
            entry = table.entry(roi_id)
            if entry is None:
                continue
            step = table.step(roi_id)
            if entry.get("signature") and step:
                identity = {k: v for k, v in step.items() if k != "label"}
                current = entry["signature"] == digest({"inputs": inputs,
                                                        "step": identity})
            else:                         # stored without a step to check against
                current = table.data_signature(roi_id) == data
            if current:
                found[label] = (entry, step)
        return found

    def evaluation_steps(self, label, roi_ids=None):
        """The distinct steps that made the evaluation called ``label`` on these
        ROIs (all included ones by default): one, unless some ROIs were
        evaluated again with other settings and some not."""
        ids = self._included() if roi_ids is None else list(roi_ids)
        out = []
        for roi_id in ids:
            entry, step = self.evaluations(roi_id).get(label, (None, None))
            if step and step not in out:
                out.append(step)
        return out

    def labels(self, plugin=None, roi_ids=None):
        """The names of the current evaluations on these ROIs, newest
        evaluator first, optionally only those made by ``plugin``."""
        ids = self._included() if roi_ids is None else list(roi_ids)
        out = []
        for roi_id in ids:
            for label, (entry, step) in self.evaluations(roi_id).items():
                if label not in out and (plugin is None or step.get("plugin") == plugin):
                    out.append(label)
        return out

    def _included(self):
        return [i for i in self.ordered_ids() if self.rois[i].use]

    # ------------------------------------------------------------- order
    def ordered_ids(self) -> List[str]:
        """Every ROI id, in the order the list and the results show them:
        as sorted (`sort`), with any ROI added since after the rest; by
        number when unsorted.  An ROI keeps its number whatever the order."""
        if not self.order:
            return list(self.rois)
        known = [i for i in self.order if i in self.rois]
        placed = set(known)
        return known + [i for i in self.rois if i not in placed]

    def sort_rows(self) -> Dict[str, Dict[str, Any]]:
        """What a sort or a filter reads of each ROI: where it is (its
        centre, ``roi_x_nm`` and ``roi_y_nm``, and ``in_drawn_roi``), the newest stored values of every evaluation,
        current or not, named as in the results table, and its keywords."""
        from . import pipeline as pipeline_module
        inside = self.in_drawn_roi()
        out = {}
        for roi_id, roi in self.rois.items():
            steps = {}
            for label, table in self.tables.items():
                entry = table.entry(roi_id) if roi_id in table else None
                if entry and "values" in entry:
                    steps[label] = entry
            values = pipeline_module.merged_values({"steps": steps})
            place = {"roi_x_nm": float(roi.center[0]), "roi_y_nm": float(roi.center[1]),
                     "in_drawn_roi": None if inside is None else inside[roi_id]}
            out[roi_id] = {**place, **values, **annotation_columns(roi, values)}
        return out

    def drawn_roi(self) -> Optional[Region]:
        """The ROI drawn in the render window, where an ROI's centre can be
        tested against it: None when nothing is drawn, or when the picture
        is of other columns than the positions (photons against frame), on
        which the ROIs' centres mean nothing."""
        session = self.session
        drawn = getattr(session, "roi", None)
        if not isinstance(drawn, Region):
            return None
        axes = getattr(session, "axes", None)
        if axes is not None and not getattr(axes(self.layer), "is_default", True):
            return None
        return drawn

    def in_drawn_roi(self) -> Optional[Dict[str, bool]]:
        """``{roi id: centre inside the drawn ROI}`` -- SMAP's UseSitesInROI
        -- or None when there is no drawn ROI to test (`drawn_roi`).  An ROI
        of a file the layer does not show is outside, whatever its centre:
        files share coordinates, and the drawn ROI was drawn on the files
        the picture shows."""
        drawn = self.drawn_roi()
        if drawn is None:
            return None
        shown = None
        layers = getattr(self.session, "layers", None) or []
        if layers:
            lay = layers[min(self.layer, len(layers) - 1)]
            shown = getattr(getattr(lay, "state", None), "files", None)
        ids = list(self.rois)
        if not ids:
            return {}
        centers = np.array([self.rois[i].center for i in ids], dtype=float)
        inside = np.asarray(drawn.mask(centers[:, 0], centers[:, 1]), dtype=bool)
        out = {}
        for roi_id, ok in zip(ids, inside):
            source = self.sources.get(self.rois[roi_id].file_id)
            if shown is not None and (source is None or source.number not in shown):
                ok = False
            out[roi_id] = bool(ok)
        return out

    def rows_of(self, roi_ids) -> np.ndarray:
        """Which rows of the session's table lie in any of these ROIs, as a
        mask: each ROI's shape in its own file, through that file's index,
        no filter deciding membership (`docs/regions.md`)."""
        locs = self.session.locs
        inside = np.zeros(len(locs), dtype=bool)
        for roi_id in roi_ids:
            roi = self.rois[roi_id]
            source = self.sources[roi.file_id]
            ungrouped = self.state(roi.file_id).tables.ungrouped
            found = rows_in(ungrouped.locs, ungrouped.index, self.region(roi))
            in_file = self._file_mask(locs, source.number)
            inside[found if in_file is None else np.flatnonzero(in_file)[found]] = True
        return inside

    def sort_columns(self) -> List[str]:
        """The names a sort or a filter can use: every measured column, then
        the keyword lists."""
        names: List[str] = []
        for row in self.sort_rows().values():
            for name in row:
                if name not in names and name not in self.annotation_lists:
                    names.append(name)
        return sorted(names) + [n for n in self.annotation_lists if n not in names]

    def sort(self, keys) -> List[str]:
        """Order the ROIs by ``keys``, ``[(column, descending)]``
        (`order.sort_ids`); the order, which is also kept."""
        from .order import sort_ids
        numbers = {i: r.number for i, r in self.rois.items()}
        self.order = sort_ids(self.sort_rows(), keys, self.annotation_lists, numbers)
        return list(self.order)

    def unsort(self) -> None:
        self.order = []

    def filter_rois(self, text: str, roi_ids=None) -> Dict[str, bool]:
        """Which ROIs meet ``text`` (`order.condition`), ``{id: bool}``,
        without changing anything; a name no ROI has is refused, since it is
        nearly always a typing error that would exclude every ROI."""
        from .order import condition
        test, names = condition(text)
        if "in_drawn_roi" in names and self.drawn_roi() is None:
            raise ValueError("in_drawn_roi needs an ROI drawn in the render window, "
                             "on the picture of positions")
        rows = self.sort_rows()
        known = set(self.sort_columns())
        unknown = [n for n in names if n not in known]
        if unknown:
            raise ValueError(f"no ROI has {', '.join(map(repr, unknown))}: there is "
                             + (", ".join(sorted(known)) or "nothing measured yet"))
        ids = self.ordered_ids() if roi_ids is None else list(roi_ids)
        return {i: test(rows[i]) for i in ids}

    def stale_steps(self, roi_id, steps=None):
        """The steps of this ROI that would have to run to be sure."""
        steps = self.resolved(steps)
        states = self.entries(roi_id, steps)
        return [step for step in steps
                if states[step.label][1] != "current"]

    def needs_evaluation(self, steps=None, roi_ids=None):
        """Which included ROIs have a step that is not current."""
        steps = self.resolved(steps)
        ids = list(self.rois) if roi_ids is None else list(roi_ids)
        return [i for i in ids if self.rois[i].use
                and self.stale_steps(i, steps)]

    # -------------------------------------------------------------- running
    def evaluate(self, block=None, roi_ids=None, progress=None, steps=None,
                 reuse=False, chain="", report=None, stop=None, after=None,
                 tag=False):
        """Run a block's steps over every included ROI.

        ``block`` is a `chain.Block`, or a list of `chain.Step`s, resolved
        here into steps; by default the project's own (`resolved`).  Each step contributes columns to a
        site's row, and a step that raises costs its own columns and not the
        ROI, nor the ROIs after it.

        With `reuse`, a step whose stored result is still current is copied
        forward instead of run again -- which is what "re-evaluate what
        changed" means: editing one evaluator's parameter costs that
        evaluator over the sites, not the whole pipeline over all of them.
        ``chain`` marks the run as a chain's (`chain.evaluate_sites`).
        ``progress(done, total)`` is told after every ROI; ``report(text)``
        gets the status line -- percentage, rate, time left
        (`region_runs.Progress`); ``stop`` (an Event, or a callable such as a
        dialog's ``wasCanceled``) ends the run between ROIs, keeping the ones
        measured.  ``after`` and ``tag`` are a block's (`_evaluate_record`).
        """
        steps = self.resolved(steps, block)
        ids = list(self.rois) if roi_ids is None else list(roi_ids)
        ids = [i for i in ids if self.rois[i].use]
        from ..region_runs import Progress, stopped
        records = {}
        status = Progress(len(ids), report, progress, what="ROIs")
        for n, roi_id in enumerate(ids):
            if stopped(stop):
                break
            record, _, _ = self._evaluate_record(roi_id, steps, reuse=reuse,
                                                 after=after, tag=tag)
            records[roi_id] = record
            status(n + 1)
        run = self._log(steps, len(records), chain=chain,
                        stopped=len(records) < len(ids))
        self._store(records, steps)
        # the records go back to the caller, who counts what failed; what
        # is kept of them is in `tables`
        return {**run, "records": records}

    def _log(self, steps, regions, **marks):
        """A run in the log: its pipeline, when, how many ROIs, and the
        marks given (``chain``, ``scope``) when they say something."""
        run = {"id": uuid4().hex, "time": datetime.now(timezone.utc).isoformat(),
               "pipeline": [step.as_record() for step in steps], "regions": regions}
        run.update({k: v for k, v in marks.items() if v})
        self.runs.append(run)
        return run

    def _store(self, records, steps):
        """Records into the tables: each step's entry the ROI's row of the
        table named after the step, with the step's record beside it."""
        made = {step.label: step.as_record() for step in steps}
        for roi_id, record in records.items():
            for label, entry in (record.get("steps") or {}).items():
                table = self.tables.pop(label, None) or RegionTable(label)
                table.set(roi_id, entry, made.get(label), record.get("signature", ""))
                self.tables[label] = table          # newest last

    def _newest_first(self):
        return reversed(list(self.tables.items()))

    def _evaluate_record(self, roi_id, steps, reuse=False, force=(), run="stale",
                         after=None, tag=False):
        """One ROI: a complete record, and the results of what actually ran.

        `force` names steps to run even when their stored result is current,
        which is how a figure is got back: the numbers are in the record but
        a plot is a closure over the data it drew, and nothing in a file can
        bring that back -- unless the row kept what its figure is drawn from
        (`Plugin.keep`), which is then restored instead of measuring again.

        `run="forced"` runs nothing else -- a stored result that is out of
        date is carried forward as it is, still signed with what made it, so
        it goes on reading as out of date.  That is the ROI manager with
        re-evaluation turned off: scrolling a list stays free, and what is
        shown is what was measured, labelled for what it is.

        The steps run in one `RegionScratch`: a step that hands back a new
        table hands it to the steps after it.  So once a step has to run,
        every step before it that changes the table runs as well, whatever
        is stored.  ``after(roi_id, scratch, original)`` is told the region's
        tables at the end (a block's write-back); ``tag`` gives them the
        rows they came from.
        """
        from ..region_runs import RegionScratch
        roi = self.rois[roi_id]
        inputs = json.loads(json_text(self.inputs(roi)))
        wanted = self.signatures(steps, inputs)
        known = self.entries(roi_id, steps) if reuse else {}
        plan = []
        for step in steps:
            entry, state = known.get(step.label, (None, "missing"))
            keep = entry is not None and (state == "current" or run == "forced")
            plan.append([step, entry, state, keep])
        last = max((k for k, (step, _, _, keep) in enumerate(plan)
                    if not keep or step.label in force), default=-1)
        for k in range(last):
            if changes_table(plan[k][0]):
                plan[k][3] = False                   # its table is needed
        scratch = None
        record = {"inputs": inputs, "signature": digest(inputs),
                  "file_id": roi.file_id, "steps": {}}
        results = {}
        updated = []
        for step, entry, state, keep in plan:
            if keep and step.label not in force:
                record["steps"][step.label] = dict(entry)
                results[step.label] = None          # kept, so not drawn
                continue
            if keep and entry.get("kept") and not changes_table(step):
                # the figure from what the row kept: no measuring again
                try:
                    restored = step.plugin.restore(entry["kept"])
                except Exception:
                    restored = None
                if restored is not None:
                    record["steps"][step.label] = dict(entry)
                    results[step.label] = restored
                    continue
            if scratch is None:                     # only if something runs
                region = self.region(roi)
                scratch = RegionScratch(self._region_tables(roi, region), region,
                                        roi_id, tag=tag)
                original = scratch.locs
            if state != "current":
                # a step re-run only for its figure tells nobody anything
                # new, and storing it again would grow the file per click
                updated.append(step.label)
            entry, result = self._one_step(step, scratch)
            entry["signature"] = wanted[step.label]
            record["steps"][step.label] = entry
            results[step.label] = result
        if after is not None and scratch is not None:
            after(roi_id, scratch, original)
        return record, results, updated

    def _one_step(self, step, scratch):
        """One evaluator on one ROI, as (what is recorded, what it returned).

        The record is data and outlives the session; the result is the live
        object, with the figures the evaluator drew, and is kept by nobody
        unless the caller wants it.  A failure is recorded, not raised.
        A new table the step hands back becomes the ROI's for the steps after
        it (`RegionScratch.apply`).
        """
        try:
            result = step.plugin.run(scratch.context(
                rois=self, grouping=step.plugin.grouping), step.settings)
            scratch.apply(result, step.label)
            # an ROI is looked at one by one: it keeps what its figures are
            # drawn from (an id column's regions do not, `evaluate_regions`)
            return region_entry(result, step.plugin, keep=True), result
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}, None

    def evaluate_one(self, roi_id, block=None, steps=None, reuse=False,
                     force=(), store=False, run="stale"):
        """The pipeline on a single ROI, figures and all.

        What `evaluate` does per site, for the one site being looked at, and
        handing back each step's whole `Result` rather than only the numbers
        it recorded, and the labels of the steps whose stored result it
        replaced -- which is what lets the ROI manager draw what the
        evaluators drew while the list is walked through.

        `reuse` takes the stored numbers for the steps that are still current
        and runs only the rest, so that a list can be scrolled through
        without recomputing what has not changed; `force` runs a step anyway,
        for the figure of the tab being looked at.  With `store` the record
        joins the runs as a one-site run, which is what makes a re-evaluation
        on selection worth anything: without it the same stale step would be
        re-run on every visit and the site table would never catch up.  A
        step re-run only for its figure has nothing new to say and is not
        stored again.
        """
        steps = self.resolved(steps, block)
        record, results, updated = self._evaluate_record(roi_id, steps, reuse=reuse,
                                                         force=force, run=run)
        if store and updated:
            self._log(steps, 1, scope="site")
            self._store({roi_id: record}, steps)
        return record, results, updated

    def latest(self, roi_id, steps=None):
        """The newest result for this ROI, step by step, and what it is worth.

        ``(record, states)``: a record assembled from the newest entry of
        each step the pipeline would run, and ``{label: state}`` saying which
        of them is still current.  `None` when the ROI has nothing stored for
        any of those steps.

        Per step rather than per record, because the two things that go out
        of date do not go out of date together: moving an ROI invalidates all
        of its steps, editing one evaluator's parameter invalidates that
        evaluator over every ROI.
        """
        steps = self.resolved(steps)
        if not steps:
            # nothing resolves -- the evaluators that made these numbers are
            # not installed here.  Show what was stored rather than nothing,
            # and say that only the data behind it could be checked.
            data = digest(json.loads(json_text(self.inputs(self.rois[roi_id]))))
            record = {"file_id": self.rois[roi_id].file_id, "steps": {}}
            states = {}
            for label, table in self._newest_first():
                entry = table.entry(roi_id)
                if entry is not None:
                    record["steps"][label] = entry
                    states[label] = ("unverified" if table.data_signature(roi_id) == data
                                     else "stale")
            return (record, states) if states else (None, {})
        found = self.entries(roi_id, steps)
        # and what else has measured this ROI: a chain's evaluators, or a
        # renamed one that is no longer in the pipeline
        for label, (entry, _) in self.evaluations(roi_id).items():
            found.setdefault(label, (entry, "current"))
        if all(entry is None for entry, _ in found.values()):
            return None, {label: state for label, (_, state) in found.items()}
        record = {"file_id": self.rois[roi_id].file_id,
                  "steps": {label: entry for label, (entry, _) in found.items()
                            if entry is not None}}
        return record, {label: state for label, (_, state) in found.items()}

    def results(self, steps=None, labels=None):
        """The site table: a row per included ROI that has results.

        A row holds every evaluation the ROI has that still describes it
        (`evaluations`), whoever ran it -- the evaluation window's pipeline,
        a chain, a script -- the pipeline's first, then the rest by name.  A
        column keeps its plain name while one evaluation writes it, and is
        qualified with the evaluation's name when two do
        (`pipeline.merged_values`), so two settings of one evaluator run
        under two names read side by side.

        With ``labels`` only those evaluations are in it, so a column is
        called as its evaluator calls it: what an analysis of one evaluation
        reads (`plugins.roi.SiteAnalysisPlugin`).

        With ``steps`` the row is instead those steps only, and is left out
        while any of them is out of date or missing: what a pipeline would
        report, for a caller that wants exactly that.
        """
        from . import pipeline as pipeline_module
        if steps is not None:
            return self._pipeline_results(steps)
        order = {}
        try:
            order = {step.label: n for n, step in enumerate(self.resolved())}
        except Exception:
            pass
        rows = []
        for roi_id in self._included():
            found = self.evaluations(roi_id)
            names = sorted((l for l in found if labels is None or l in labels),
                           key=lambda l: (order.get(l, len(order)), l))
            values = pipeline_module.merged_values(
                {"steps": {label: found[label][0] for label in names}})
            if values:
                roi = self.rois[roi_id]
                rows.append({"roi_id": roi.number, "file_id": roi.file_id, **values,
                             **annotation_columns(roi, values)})
        return rows

    def _pipeline_results(self, steps):
        from . import pipeline as pipeline_module
        steps = self.resolved(steps)
        rows = []
        for roi in (self.rois[i] for i in self.ordered_ids()):
            if not roi.use:
                continue
            record, states = self.latest(roi.id, steps)
            if record is None or any(states.get(step.label) in ("stale", "missing", None)
                                     for step in steps):
                continue
            record = {**record, "steps": {step.label: record["steps"][step.label]
                                          for step in steps}}
            values = pipeline_module.merged_values(record)
            if values:
                rows.append({"roi_id": roi.number, "file_id": roi.file_id, **values,
                             **annotation_columns(roi, values)})
        return rows

    def find(self, file_id, plugin=None, settings=None, parameters=None):
        """Add the ROIs a segmentation plugin finds on one file.

        Each is included as the finder judged it (``use``); `evaluate` and
        `results` take every included ROI, as the GUI does.
        """
        origin, proposals = self.propose(file_id, plugin, settings, parameters)
        found = []
        for proposed in proposals:
            roi = self.add_roi(file_id, proposed["center"],
                               origin={**origin, **proposed["origin"]})
            roi.use = proposed["use"]
            found.append(roi)
        return found

    def propose(self, file_id, plugin=None, settings=None, parameters=None,
                exclude=()):
        """``(origin, proposals)``: what `find` would add, adding nothing.

        Each proposal is ``{"center", "use", "origin"}``, the last what the
        finder measured, to go into the ROI's origin beside the provenance.
        The ROIs in `exclude` are left out of the ones the finder keeps its
        distance from -- the ones a run would remove first, so that a
        preview of a replacing run proposes what the run will add.
        """
        from ..plugins import settings_from, settings_values
        from ..plugins.segment import SegmentRegions
        if plugin is None:
            plugin = SegmentRegions()
        if settings is None:
            settings = settings_from(plugin.Settings, parameters or {})
        state = self.state(file_id)
        origin = {"method": getattr(plugin, "path", "") or getattr(plugin, "name", ""),
                  "version": str(getattr(plugin, "version", "1")),
                  "parameters": settings_values(settings),
                  "filters": state.filter.ranges, "grouped": self.grouped,
                  "group_settings": asdict(self.group_settings) if self.grouped else None}
        exclude = set(exclude)
        existing = [roi.center for roi in self.rois.values()
                    if roi.file_id == file_id and roi.id not in exclude]
        proposals = []
        for proposed in plugin.propose(state.locs[state.filter.indices], settings,
                                       existing):
            # a centre, or a dict for a finder that also judges what it found:
            # its "use", and what it measured, kept with the ROI's origin
            if isinstance(proposed, dict):
                proposals.append({"center": point(proposed["center"]),
                                  "use": bool(proposed.get("use", True)),
                                  "origin": dict(proposed.get("origin", {}))})
            else:
                proposals.append({"center": point(proposed), "use": True, "origin": {}})
        return origin, proposals

    def absorb_run(self, run):
        """A run as an older file kept it -- its records inside -- into the
        log and the tables; a run without records is only logged."""
        records = run.get("records") or {}
        header = {k: v for k, v in run.items() if k != "records"}
        header.setdefault("regions", len(records))
        self.runs.append(header)
        made = {s.get("label"): s for s in run.get("pipeline") or []}
        for roi_id, record in records.items():
            for label, entry in (record.get("steps") or {}).items():
                table = self.tables.pop(label, None) or RegionTable(label)
                table.set(roi_id, entry, made.get(label), record.get("signature", ""))
                self.tables[label] = table

    def tables_dict(self, roi_ids=None):
        """The tables as plain data, keeping only the rows of ``roi_ids``
        (all when None): what the project's JSON carries."""
        out = []
        for table in self.tables.values():
            doc = table.to_dict(roi_ids)
            if doc["ids"]:
                out.append(doc)
        return out

    # ------------------------------------------------------------- sources
    def sync(self) -> None:
        """Follow the session: files, filters, grouping.  Cheap when nothing moved."""
        locs = self.session.locs
        names = self.session.file_names() or (["(unsaved)"] if len(locs) else [])
        signature = (id(locs), len(locs), tuple(names))
        if self._built != signature:
            self._rebuild(names)
            self._built = signature
        layer = self.session.layers[min(self.layer, len(self.session.layers) - 1)]
        if layer.is_image:
            layer = self.session.layers[self.session.first_locs_layer()]
        self.filters = layer.ranges
        self.grouped = layer.grouped
        # so the manager's images look like the main view -- except for the
        # axes, which stay the positions: a site is a place, and a versatile
        # render is a picture of two columns rather than of anywhere
        self.render = replace(layer.state.settings, axes=RenderAxes())
        self.display = layer.state.display
        for source in self.sources.values():         # the states follow the layer
            if source.state is not None:
                self._apply(source)

    def _rebuild(self, names: List[str]) -> None:
        """Sources for the session's files, keeping the ids ROIs refer to."""
        by_number = {s.number: s for s in self.sources.values()}
        self.sources = {}
        self._states = {}
        for number, name in enumerate(names):
            info = self.session.files[number] if number < len(self.session.files) else None
            old = by_number.get(number)
            source = Source(old.id if old else f"file{number}", name,
                                   info.path if info else None, number)
            source.fingerprint = self._fingerprint(source)
            self.sources[source.id] = source
        # ROIs of files that are gone stay in the project but cannot be used
        self._states.clear()

    def _fingerprint(self, source: Source) -> str:
        """What an evaluation was run on: the data, not the file's name.

        Row count, column names and a sample of the positions -- enough to
        notice a corrected or refitted table, cheap enough to recompute on
        every staleness check, and unchanged by a rename or a "save as" --
        or by a save and reopen, which adds `filenumber` to a table that
        never came from a file (see `BOOKKEEPING`).  The columns written from
        regions' results (`region_runs.REGION_COLUMNS`) are not counted either:
        an ROI's number or its measured value written onto its localizations
        changes nothing it was measured on, and used to make the very
        evaluation it came from out of date.
        """
        from ..region_runs import REGION_COLUMNS
        locs = self.session.locs
        written = set(locs.metadata.get(REGION_COLUMNS) or {})
        names = sorted((set(locs.columns) - written) | set(BOOKKEEPING))
        if not len(locs):
            return digest([0, names])
        rows = np.arange(len(locs))
        if "filenumber" in locs and len(self.sources) > 1:
            rows = rows[np.asarray(locs["filenumber"]) == source.number]
        sample = rows[:: max(1, len(rows) // 512)][:512]
        values = [np.asarray(locs[c])[sample] for c in ("x_nm", "y_nm") if c in locs]
        return digest([len(rows), names,
                       [np.round(v.astype(float), 3).tolist() for v in values]])

    def _slice(self, source: Source) -> Localizations:
        locs = self.session.locs
        if "filenumber" not in locs or len(self.sources) <= 1:
            return locs
        return locs[np.asarray(locs["filenumber"]) == source.number]

    @property
    def group_settings(self) -> GroupSettings:
        """The session's: there is one grouped table, and the ROIs see it."""
        return self.session.group_settings

    def _apply(self, source: Source) -> None:
        """Put the layer's bounds, grouping and look on this source's state."""
        state = source.state
        if state is None:
            return
        settings, display = getattr(self, "render", None), getattr(self, "display", None)
        if settings is not None:
            state.settings, state.display = settings, display
        state.tables.settings = self.session.group_settings
        state.use_grouped = self.grouped
        state.set_ranges(self.filters)

    def _tables(self, source: Source) -> Tables:
        """The tables an ROI of this file looks at.  One file: the session's
        own, grouped table and all.  One of several: its rows of the session's
        table, and of the session's grouped table -- linking never joins two
        files (`GroupSettings.block_fields`), so those rows are what linking
        the file alone would give, without linking anything again."""
        session = self.session
        if "filenumber" not in session.locs or len(self.sources) <= 1:
            return session.tables

        def link(settings, attach_columns, number=source.number):
            grouped = session.tables.grouped().locs
            if "filenumber" not in grouped:
                return grouped
            return grouped[np.asarray(grouped["filenumber"]) == number]
        return Tables(self._slice(source), session.group_settings, link=link)

    def state(self, file_id) -> ViewState:
        """The data an ROI of this file sees: the session's rows for that file,
        under the layer's filter and grouping."""
        source = self.sources[file_id]
        if source.state is None:
            source.state = ViewState(tables=self._tables(source))
            self._apply(source)
        return source.state

    # ------------------------------------------------ measuring in an ROI
    def _file_mask(self, locs: Localizations, number: int) -> Optional[np.ndarray]:
        """Which rows of ``locs`` are file ``number``'s; None when all are."""
        if "filenumber" not in locs or len(self.sources) <= 1:
            return None
        key = (id(locs), len(locs), number)
        if key not in self._file_masks:
            if len(self._file_masks) > 16:
                self._file_masks.clear()
            self._file_masks[key] = np.asarray(locs["filenumber"]) == number
        return self._file_masks[key]

    def _region_tables(self, roi, region):
        """Every localization layer of the session, cut to the ROI.

        The rows are found through the file's own index (the ROI belongs to
        one file) and are rows of the session's table; each layer's
        selection is its own filter over them -- not the session's
        selection, which is cut to the drawn ROI and the slab -- and its
        grouped table is offered, cut the same way, where it has a current
        one.  The ROI manager's own state for the file decides nothing here
        but where to look.
        """
        from ..plugins import Selection
        session = self.session
        source = self.sources[roi.file_id]
        ungrouped = self.state(roi.file_id).tables.ungrouped
        found = rows_in(ungrouped.locs, ungrouped.index, region)
        in_file = self._file_mask(session.locs, source.number)
        rows = found if in_file is None else np.flatnonzero(in_file)[found]
        # linked here if a layer shows blinks and nothing defers it;
        # otherwise offered only where it is current.  Before the rows are
        # cut: linking writes `group_id` onto the localizations, and a cut
        # taken first lacked it, so the blinks could not be traced back
        locsets = {i: (session.tables.available() if lay.grouped else
                       session.tables.peek_grouped() if session.tables.is_current
                       else None)
                   for i, lay in enumerate(session.layers) if not lay.is_image}
        locs = session.locs[rows]
        tables = {}
        for i, lay in enumerate(session.layers):
            if lay.is_image:
                continue
            mask = np.asarray(lay.selection(i).mask, dtype=bool)
            entry = {"ungrouped": (locs, Selection(mask[rows], layer=i, roi=region,
                                                   name=f"{lay.name}, {region}"))}
            locset = locsets[i]
            if locset is not None:
                keep = self._file_mask(locset.locs, source.number)
                g = rows_in(locset.locs, locset.index, region, keep)
                entry["grouped"] = (locset.locs[g], Selection(
                    np.asarray(lay.state.filter_for(locset).mask, dtype=bool)[g],
                    layer=i, roi=region, name=f"{lay.name}, blinks merged, {region}"))
            tables[i] = entry
        layer = min(self.layer, len(session.layers) - 1)
        if session.layers[layer].is_image:
            layer = session.first_locs_layer()
        return tables, rows, layer, session

    # --------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        """The project as JSON-compatible data, for the localization file."""
        return {"format_version": FORMAT_VERSION,
                "size_nm": self.size_nm, "shape": self.shape,
                "preview_nm": self.preview_nm, "tile_nm": self.tile_nm,
                "navigation": self.navigation,
                "annotation_lists": self.annotation_lists,
                "line_names": self.line_names,
                "order": [i for i in self.order if i in self.rois],
                "sources": [{"id": s.id, "number": s.number, "name": s.name}
                            for s in self.sources.values()],
                "rois": [asdict(r) for r in self.rois.values()],
                "next_number": self.next_number,
                "runs": self.runs,
                # the results, a columnar table per evaluation (`RegionTable`)
                "tables": self.tables_dict(self.rois),
                # the block is provenance for the columns in `runs`, so it
                # is saved with them rather than only in the workspace
                "block": self.block.to_dict()}

    def from_dict(self, doc: Optional[dict]) -> None:
        """Restore ROIs, runs and navigation saved with a file."""
        self.rois = {}
        self.next_number = 1
        self.runs = []
        self.tables = {}
        self.navigation = {}
        self.set_annotation_lists(DEFAULT_ANNOTATION_LISTS)
        self.line_names = [DIRECTION]
        self.order = []
        from ..chain import Block
        self.block = Block()
        self.segment_preview = None
        self.history.clear()        # another file's edits
        if not doc:
            return
        doc = json.loads(json_text(doc)) if not isinstance(doc, dict) else doc
        if doc.get("format_version", FORMAT_VERSION) > FORMAT_VERSION:
            raise ValueError("this file's ROIs were written by a newer smappy")
        self.set_geometry(doc.get("size_nm", self.size_nm), doc.get("shape", self.shape))
        self.preview_nm = float(doc.get("preview_nm", 0.0) or 0.0)
        self.set_tiles(doc.get("tile_nm", 0.0) or 0.0)
        self.navigation = dict(doc.get("navigation", {}))
        self.set_annotation_lists(doc.get("annotation_lists", DEFAULT_ANNOTATION_LISTS))
        self.set_line_names(doc.get("line_names", [DIRECTION]))
        self.order = [str(i) for i in doc.get("order", [])]
        # ids the file used, mapped onto the files this session has
        by_number = {s.number: s for s in self.sources.values()}
        remap = {}
        for saved in doc.get("sources", []):
            source = by_number.get(saved.get("number"))
            if source is not None:
                remap[saved["id"]] = source.id     # the name stays the session's
        if self.navigation.get("file") in remap:
            self.navigation["file"] = remap[self.navigation["file"]]
        for saved in doc.get("rois", []):
            roi = ROI.from_dict(saved)
            roi.file_id = remap.get(roi.file_id, roi.file_id)
            if roi.file_id in self.sources:
                self.rois[roi.id] = roi
        self.number_rois(doc.get("next_number"))
        for run in doc.get("runs", []):
            records = {i: r for i, r in (run.get("records") or {}).items()
                       if i in self.rois}
            self.absorb_run({**run, "records": records})
        for saved in doc.get("tables", []):
            table = RegionTable.from_dict(saved)
            table.keep_only(self.rois)
            if len(table):
                self.tables[table.label] = table
        self.block = block_from(doc)

    # ---------------------------------------------------------------- misc
    @property
    def preview_width(self) -> float:
        """What the ROI image covers, in nm."""
        return self.preview_nm or 3.0 * self.size_nm

    def rois_of(self, file_id) -> List[ROI]:
        return [r for r in self.rois.values() if r.file_id == file_id]

    def file_number(self, file_id) -> int:
        source = self.sources.get(file_id)
        return source.number + 1 if source is not None else 0

    def source_of(self, number: int) -> Optional[Source]:
        return next((s for s in self.sources.values() if s.number == number), None)


def block_from(doc: dict):
    """The project's block as a saved project holds it: ``block``, a
    `chain.Block`'s own form, or -- in a file from before the project held
    one -- its steps as ``pipeline`` (workspace instances, a block among them
    as ``plugin: Chain/Block`` with its ``chain``) and its own fields as
    ``block_options``.  Read tolerantly: a file outlives the code that wrote
    it, and a step that cannot be read costs that step."""
    from ..chain import Block, ChainError, Step
    if isinstance(doc.get("block"), dict):
        try:
            return Block.from_dict(doc["block"])
        except ChainError:
            return Block()
    options = doc.get("block_options") or {}
    steps = []
    for raw in doc.get("pipeline") or []:
        if not isinstance(raw, dict) or not raw.get("plugin"):
            continue
        enabled = bool(raw.get("enabled", True))
        try:
            if raw["plugin"] == "Chain/Block" and isinstance(raw.get("chain"), dict):
                inner = Block.from_dict(raw["chain"])
                inner.label = str(raw.get("label") or "") or inner.label
                inner.enabled = enabled
                steps.append(inner)
                continue
        except ChainError:
            continue
        values = raw.get("values")
        steps.append(Step(plugin=str(raw["plugin"]), label=str(raw.get("label") or ""),
                          values=dict(values) if isinstance(values, dict) else {},
                          enabled=enabled))
    own = {k: options[k] for k in ("regions", "write_back", "write_values", "merge")
           if k in options}
    try:
        block = Block.from_dict({**own, "steps": []})
    except ChainError:
        block = Block()
    block.steps = steps
    return block
