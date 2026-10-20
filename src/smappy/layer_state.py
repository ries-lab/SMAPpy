"""What a layer draws: one table, its grouped form, and a filter over either.

Three objects, from the data outwards:

* `LocSet` -- a table and what is derived from the table alone: its spatial
  index and its median precision.  Building the index is what a new layer
  over a large table would otherwise spend its time on, so there is one per
  table, however many layers look at it.
* `Tables` -- one ungrouped table and its grouped form.  A session has one
  (`Session.tables`), and every layer of it shares it.  The grouped table is a
  cache keyed by the rows it was linked from and the link settings
  (`GroupSettings`): asked for when the key no longer matches, it is linked
  again.  So there is no "stale" flag to keep, and nothing to copy when the
  settings change -- the key says whether what is there is still the answer.
* `ViewState` -- one layer's view: its **bounds** (one dict, the user's
  setting), its file choice, whether it shows the grouped table, and its
  render and display settings.  Its filter is *derived*: a `LocFilter` built
  from those bounds over whichever table is shown, kept as long as that table
  is, and rebuilt from the bounds when the table changes.

Why one bounds dict rather than one filter per table: a layer used to carry a
filter for each of its two tables, and the bounds were copied from one to the
other in five places (a new layer, a relink, new link settings, a corrected
table, a chain's copy of the session).  Every one of them was a place to
forget, and forgetting showed a grouped table unfiltered.  With one dict there
is nothing to copy.  The same bound then keeps different localizations on the
two tables -- ``xy_err_nm <= 25`` is each localization's precision ungrouped
and the combined precision of a blink grouped -- and that is intended: it is
the user's statement about what is good enough, applied to what is shown.

Moved out of `viewer.py`, whose matplotlib window is the oldest user of these
and the one that goes first.
"""
from __future__ import annotations

import copy
import weakref
from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import lut as luts
from .filter import LocFilter
from .group import GroupSettings, group
from .locs import Localizations
from .render import (DisplaySettings, FieldOfView, RenderSettings, RenderedImage,
                     camera_pixelsize_nm, positions, render_locs)
from .spatial import GrowingIndex, SpatialIndex


class LocSet:
    """One table with the things that are derived from it and nothing else.

    Grouped and ungrouped localizations are different tables, so each has its
    own spatial index.  The filters are not here: they belong to the layers,
    which share this.
    """

    def __init__(self, locs: Localizations,
                 extent: Optional[Sequence[float]] = None,
                 growable: bool = False):
        self.locs = locs
        self._precision_field: Optional[str] = \
            "xy_err_nm" if "xy_err_nm" in locs else None
        # the extent on custom render axes, cached
        self._bounds_key = None
        self._bounds = (0.0, 0.0, 1.0, 1.0)
        # a live view is opened before the first block has been fitted, so an
        # empty table is a normal starting point, not an error
        empty = np.empty(0, np.float32)
        x, y = positions(locs) if locs.columns else (empty, empty)
        self.index = (GrowingIndex(x, y, extent=extent) if growable
                      else SpatialIndex(x, y, extent=extent))
        # for the query margin: how far the widest blob reaches beyond the view
        self.median_precision = 0.0
        self._precision_n = 0
        self._update_precision()
        self._files = None              # (files, rows), found on first use

    def _file_blocks(self):
        if self._files is None:
            self._files = file_blocks(self.locs["filenumber"] if "filenumber" in self.locs
                                      else np.zeros(0, np.int32), len(self.locs))
        return self._files

    @property
    def files(self) -> Optional[List[int]]:
        """The file numbers in the table, in order; None for a table that
        is still growing, whose files are not settled."""
        if self.growing:
            return None
        return self._file_blocks()[0]

    @property
    def file_rows(self) -> Optional[Dict[int, Tuple[int, int]]]:
        """Each file's rows as one ``(start, stop)``, or None when a file's
        rows are not one block (or the table still grows).  `Session.add_file`
        appends, so they are, unless a plugin reordered the rows."""
        if self.growing:
            return None
        return self._file_blocks()[1]

    @property
    def growing(self) -> bool:
        """Whether this table takes `append` (a live fit's)."""
        return isinstance(self.index, GrowingIndex)

    def _update_precision(self, force: bool = True) -> None:
        """Recompute the median localization precision.

        It only sets the cap on the rendering sigma and it hardly moves when a
        few percent more localizations arrive, so while data is streaming in it
        is recomputed once the table has grown by a fifth rather than on every
        block -- a median over millions of values is not free.
        """
        if self._precision_field is None and "xy_err_nm" in self.locs:
            self._precision_field = "xy_err_nm"     # the first block brought it
        if self._precision_field is None:
            return
        if not force and len(self.locs) < self._precision_n * 1.2:
            return
        values = np.asarray(self.locs[self._precision_field], np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            self.median_precision = float(np.median(values))
        self._precision_n = len(self.locs)

    def append(self, new: Localizations) -> int:
        """Add localizations to the end of the table.  Returns how many.

        The table and the index are extended rather than rebuilt; the layers'
        filters catch up with the new rows on the same thread (`ViewState.catch_up`),
        testing only those, against the bounds already set.
        """
        n = len(new)
        if n == 0:
            return 0
        if not self.growing:
            raise TypeError("this table was not opened for appending; "
                            "build it with live=True")
        x, y = positions(new)
        self.locs.extend(new)
        self.index.append(x, y)
        self._update_precision(force=False)
        return n

    def axis_bounds(self, axes) -> Tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of the table on non-default render axes.

        The spatial index stays in the table's own coordinates -- that is what
        an ROI and a neighbour query mean -- so a picture of photons against
        frame has nowhere to read its extent from but the columns.  Cached
        against the axes and the length, because the overview asks for it on
        every redraw.
        """
        key = (axes, len(self.locs))
        if self._bounds_key != key:
            if not len(self.locs):
                self._bounds = (0.0, 0.0, 1.0, 1.0)
            else:
                x, y = axes.coordinates(self.locs)
                self._bounds = (float(np.nanmin(x)), float(np.nanmin(y)),
                                float(np.nanmax(x)), float(np.nanmax(y)))
            self._bounds_key = key
        return self._bounds

    def __len__(self) -> int:
        return len(self.locs)


def file_blocks(numbers, n: int) -> Tuple[List[int], Optional[Dict[int, Tuple[int, int]]]]:
    """The file numbers of a table's ``filenumber`` column and, when each
    file's rows are one block, those blocks as ``{file: (start, stop)}``.

    One pass over the column, for the places where a file of several is all
    that is wanted -- a layer's filter, linking -- and a pass over every row
    is what they would otherwise cost: the file choice was an `np.isin` over
    the whole table, 0.44 s at 10^8 rows.  A table without the column is one
    file, 0, of all its rows.
    """
    numbers = np.asarray(numbers)
    if not len(numbers):
        return ([0], {0: (0, n)}) if n else ([], {})
    change = np.flatnonzero(numbers[1:] != numbers[:-1]) + 1
    if len(change) > 100_000:             # interleaved: no blocks to speak of
        return [int(v) for v in np.unique(numbers)], None
    starts = np.r_[0, change]
    stops = np.r_[change, len(numbers)]
    values = [int(v) for v in numbers[starts]]
    files = sorted(set(values))
    if len(files) != len(values):
        return files, None
    return files, {v: (int(a), int(b)) for v, a, b in zip(values, starts, stops)}


def merge_spans(spans) -> List[Tuple[int, int]]:
    """``(start, stop)`` spans sorted, with the touching ones joined."""
    out: List[Tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        elif b > a:
            out.append((a, b))
    return out


# How a table is linked: ``link(settings, attach_columns) -> grouped table``.
Linker = Callable[[GroupSettings, bool], Localizations]


class Tables:
    """An ungrouped table and its grouped form, linked once and shared.

    The grouped table is cached under ``(rows, settings)``: the number of rows
    it was linked from (a live table only ever grows, and any other change of
    the table is a new `Tables`) and the link settings.  `grouped` links again
    when the key no longer matches; `peek_grouped` hands back whatever was
    linked last, current or not, for the one reader that must not wait for a
    relink -- a picture of a live fit, which shows the blinks as they were
    linked until somebody asks again.

    ``link`` replaces the linking, for a table whose grouped form is cut from
    another one rather than linked afresh (one file of a session, in the ROI
    manager); ``ready()`` then says whether that is cheap now.

    **Deferred linking.**  A layer reads the grouped table when it is drawn,
    and headless that links it there and then, on first read.  In the GUI
    that read is on the GUI thread, so a large table is not linked by
    reading it: with ``defer_from`` set (the GUI sets it, `gui.merging`), a
    table of at least that many rows is `deferred` -- a layer that shows
    blinks draws nothing meanwhile (`pending`), and the GUI links it in a
    worker (`link_job`, `take_grouped`).  A copy made with `sharing` (a
    chain's) links on read, into columns of its own.

    **File by file.**  A table of several files is linked one file at a
    time, and only the files wanted: those the layers showing blinks show
    (`ViewState` registers itself here), or the ones a caller names
    (``files=``).  Linking never joins two files (`GroupSettings.block_fields`),
    so a file linked alone gives the blinks linking the whole table gives
    it; what changes is that ten files of 10 M are not all linked before a
    layer showing one of them draws -- 86 s and a 13 GB peak at 10^8 rows
    (`docs/performance.md`, "Large tables") -- and that a file once linked
    stays linked while the layers move between files.  The grouped table is
    the files linked so far, a block each in the order they were linked, and
    their `group_id` numbers run on from one file to the next, so the ids
    written onto the localizations stay those of the grouped table's rows as
    files are added.  The localizations of files not linked yet have
    `group_id` 0.  A table of one file, one that still grows and one cut from
    another (``link``) are linked whole, as before.
    """

    def __init__(self, locs: Localizations, settings: Optional[GroupSettings] = None,
                 *, grouped: Optional[Localizations] = None, live: bool = False,
                 extent: Optional[Sequence[float]] = None,
                 link: Optional[Linker] = None,
                 ready: Optional[Callable[[], bool]] = None,
                 defer_from: Optional[int] = None):
        """``grouped`` is ``locs`` already linked with ``settings`` -- by a
        worker thread, off the GUI's."""
        self.ungrouped = LocSet(locs, extent=extent, growable=live)
        self.settings = settings if settings is not None else GroupSettings()
        self._link_with = link
        self._ready = ready
        self.defer_from = defer_from
        self._grouped: Optional[LocSet] = None
        self._key = None
        self._key_use = None
        # set while a merge of these rows runs elsewhere (the GUI's worker,
        # `gui.merging`): a layer showing blinks keeps the ones it has until
        # the new ones arrive, rather than linking them again, here, at once
        self.merging = False
        self._pending = None            # the empty table shown meanwhile
        self._shared = False
        # the GUI's merge of these rows failed: the layers show the
        # localizations rather than nothing, and it is not tried again
        self.merge_failed = False
        # file by file: the files linked under `_key`, and the next group id
        self._linked: set = set()
        self._next_id = 1
        # the layers looking at these tables, for the files they want linked
        self._views: "weakref.WeakSet" = weakref.WeakSet()
        if grouped is not None:
            self._grouped = LocSet(grouped)
            self._keep_key(self._wanted())
            self._linked = set(self.ungrouped.files or ())

    @classmethod
    def sharing(cls, other: "Tables") -> "Tables":
        """A second `Tables` over the same table and the same grouped table,
        without building anything: a chain's copy of the session.  What the
        copy links afterwards is its own; this one keeps what it had."""
        made = cls.__new__(cls)
        made.ungrouped = other.ungrouped
        made.settings = other.settings
        made._link_with = other._link_with
        made._ready = other._ready
        made._grouped = other._grouped
        made._key = other._key
        made._key_use = other._key_use
        made._linked = set(other._linked)
        made._next_id = other._next_id
        made._views = weakref.WeakSet()
        made.merging = False
        made.defer_from = None          # a copy is headless: it links on read
        made._pending = None
        made.merge_failed = False
        # what the copy links is its own, and so are the columns linking
        # writes back (`group_id`, `n_in_group`): it runs on a worker, and
        # the table is the session's (`grouped` copies it first)
        made._shared = True
        return made

    @property
    def locs(self) -> Localizations:
        return self.ungrouped.locs

    @property
    def growing(self) -> bool:
        return self.ungrouped.growing

    def _wanted(self):
        from .group import use_token
        # which rows take part (`group.USE`) is part of what linking reads
        return (len(self.locs), dict(asdict(self.settings), use=use_token(self.locs)))

    def _keep_key(self, key) -> None:
        """Take ``key`` as the grouped table's, holding the `group.USE` array
        it names: alive, its identity cannot be another array's."""
        from .group import USE
        self._key = key
        self._key_use = self.locs.columns.get(USE)

    def _per_file(self) -> bool:
        """Whether these rows are linked file by file (see the class)."""
        if self.growing or self._link_with is not None or "filenumber" not in self.locs:
            return False
        files = self.ungrouped.files
        return files is not None and len(files) > 1

    def wanted_files(self, files=None) -> set:
        """The files a grouped table must hold: ``files``, or those the
        layers showing blinks show -- every file when one of them shows every
        file, or when no layer shows blinks and a caller asks anyway."""
        present = set(self.ungrouped.files or ())
        if files is not None:
            return set(int(f) for f in files) & present
        shown = [v._files for v in list(self._views) if v.use_grouped]
        if not shown or any(f is None for f in shown):
            return present
        wanted = set().union(*map(set, shown)) & present
        # a layer showing no file that is there still draws a grouped table
        return wanted or {min(present)}

    def covers(self, files=None) -> bool:
        """Whether the grouped table there is current and holds ``files``
        (as `wanted_files` reads them)."""
        if self._grouped is None or self._key != self._wanted():
            return False
        return not self._per_file() or self.wanted_files(files) <= self._linked

    @property
    def is_current(self) -> bool:
        """Whether the grouped table there is the one `grouped` would hand out."""
        return self.covers()

    def peek_grouped(self) -> Optional[LocSet]:
        """The grouped table linked last, current or not; None if never."""
        return self._grouped

    def _unshare(self) -> None:
        """Take a table of one's own over the shared one's arrays, so that
        linking writes its columns there and not into the session's."""
        shared = self.ungrouped
        own = copy.copy(shared)
        own.locs = Localizations(dict(shared.locs.columns), dict(shared.locs.metadata))
        self.ungrouped = own
        self._shared = False

    def pending(self) -> LocSet:
        """What a layer showing blinks draws while they are merged elsewhere:
        nothing -- an empty table with the localizations' columns, so the
        picture shows no single localizations in place of blinks."""
        if self._pending is None or set(self._pending.locs.keys()) != set(self.locs.keys()):
            self._pending = LocSet(self.locs[np.zeros(0, np.int64)])
        return self._pending

    def available(self, files=None) -> Optional[LocSet]:
        """The current grouped table for a reader that can do without it:
        linked now if it is not current and linking here is allowed, None
        while it is deferred to the GUI's worker or the table still grows.
        ``files`` as `grouped` takes them."""
        if self.covers(files):
            return self._grouped
        if self.growing or self.deferred(files):
            return None
        return self.grouped(files=files)

    def deferred(self, files=None) -> bool:
        """Whether reading the grouped table must not link it now: a merge
        runs elsewhere, or the table is large enough for the GUI to link it
        in a worker (``defer_from``).  A table cut from another one is
        never deferred once that one is current -- cutting is cheap."""
        if self.covers(files):
            return False
        if self._link_with is not None and self._ready is not None and self._ready():
            return False
        return self.merging or (self.defer_from is not None
                                and len(self.locs) >= self.defer_from)

    def grouped(self, progress=None, files=None) -> LocSet:
        """The grouped table for the rows and settings as they are now.

        Links -- the slow part -- when there is none or the key has moved.
        Linking writes `group_id` and `n_in_group` back onto the ungrouped
        table (`group.attach`), but not while the table is still growing: a
        block arriving from the fitter would not carry those columns, and
        `extend` refuses a table whose columns do not match.  A table of
        several files links the ``files`` wanted (`wanted_files`) that are
        not linked yet, one at a time.
        """
        if self.covers(files):
            return self._grouped
        if self._shared:
            self._unshare()
        key = self._wanted()
        if self._per_file():
            if self._key != key:
                self._forget_files()
            missing = sorted(self.wanted_files(files) - self._linked)
            pieces = []
            for k, number in enumerate(missing):
                if progress is not None:
                    progress(f"file {number + 1} ({k + 1} of {len(missing)})", 0.0)
                piece, rows = self._file_table(number)
                table, index = group(piece, self.settings, attach_columns=False,
                                     progress=progress)
                pieces.append((number, rows, table, index))
            self._add_files(pieces, key)
            return self._grouped
        if self._link_with is not None:
            table = self._link_with(self.settings, not self.growing)
        else:
            table, _ = group(self.locs, self.settings,
                             attach_columns=not self.growing, progress=progress)
        self._grouped = LocSet(table)
        self._keep_key(key)
        return self._grouped

    def _file_table(self, number: int):
        """File ``number``'s rows: a view of its block, or a copy of its rows
        where they are not one; and where they are, for writing back."""
        blocks = self.ungrouped.file_rows
        if blocks is not None:
            a, b = blocks[number]
            rows = slice(a, b)
            return Localizations({k: v[a:b] for k, v in self.locs.columns.items()},
                                 dict(self.locs.metadata)), rows
        rows = np.flatnonzero(np.asarray(self.locs["filenumber"]) == number)
        return self.locs[rows], rows

    def _forget_files(self) -> None:
        """The settings moved: nothing linked before is current, and the ids
        it wrote onto the localizations go with it."""
        from .group import GROUP_COLUMNS
        self._grouped = None
        self._linked = set()
        self._next_id = 1
        for name in GROUP_COLUMNS:
            self.locs.columns.pop(name, None)

    def _add_files(self, pieces, key) -> None:
        """Keep files linked on their own, as ``(number, rows, grouped,
        group_index)``: their blinks after those already there, their ids
        numbered on, and the ids written onto their localizations."""
        from .mathparse import apply_recipes
        if not pieces:
            if self._grouped is None:       # nothing wanted is there: no blinks
                self._grouped = LocSet(self.locs[np.zeros(0, np.int64)])
            self._keep_key(key)
            return
        n = len(self.locs)
        # new arrays, never written into: a render thread may be reading the
        # ones there are (`ViewState.snapshot`)
        gid = (np.array(self.locs["group_id"], np.int32) if "group_id" in self.locs
               else np.zeros(n, np.int32))
        tables = [] if self._grouped is None else [self._grouped.locs]
        on = None
        for number, rows, table, index in pieces:
            base = self._next_id - 1
            table.columns["group_id"] = (np.asarray(table["group_id"], np.int64)
                                         + base).astype(np.int32)
            index = np.asarray(index, np.int64)
            grouped_row = index > 0         # 0: left out of grouping (`group.USE`)
            gid[rows] = np.where(grouped_row, index + base, 0)
            counts = np.asarray(table["n_in_group"])
            if on is None:
                on = (np.array(self.locs["n_in_group"], counts.dtype)
                      if "n_in_group" in self.locs else np.zeros(n, counts.dtype))
            on[rows] = np.where(grouped_row, counts[np.maximum(index, 1) - 1]
                                if len(counts) else 0, 0)
            self._next_id += len(table)
            self._linked.add(number)
            tables.append(table)
        self.locs.columns["group_id"] = gid
        self.locs.columns["n_in_group"] = on
        apply_recipes(self.locs)
        from .locs import concat
        merged = concat(tables) if len(tables) > 1 else tables[0]
        merged.metadata = dict(tables[0].metadata)
        self._grouped = LocSet(merged)
        self._keep_key(key)

    def link_job(self) -> Optional[Tuple[Localizations, GroupSettings, tuple]]:
        """What linking these rows takes, for a worker thread:
        ``(locs, settings, key)``, to run `group.group` on with
        ``attach_columns=False`` and hand back to `take_grouped`.  None when
        there is nothing to do, or when the linking is not `group` (a
        `Tables` cut from another one).  A growing table is handed over as a
        snapshot, so the worker reads rows the fit is not adding to.  A table
        of several files hands over one file it wants linked, the first; the
        merge asks again for the next."""
        if self.is_current or self._link_with is not None:
            return None
        if self._per_file():
            if self._key != self._wanted():
                self._forget_files()
            missing = sorted(self.wanted_files() - self._linked)
            if not missing:
                return None
            piece, _rows = self._file_table(missing[0])
            return piece, self.settings, self._wanted() + (missing[0],)
        locs = self.locs.snapshot() if self.growing else self.locs
        return locs, self.settings, (len(locs), self._wanted()[1])

    def take_grouped(self, grouped: Localizations, group_index: np.ndarray,
                     key: tuple) -> bool:
        """Keep a grouped table linked elsewhere from `link_job`'s rows, on
        the thread that owns the tables.  False, and nothing kept, when the
        rows or the settings moved since the job was handed out -- except
        that a growing table keeps what was linked, as `peek_grouped` does.
        Writes `group_id` and `n_in_group` back, as `grouped` does."""
        from .group import attach
        wanted = self._wanted()
        if len(key) > 2:                    # one file of several
            if key[:2] != wanted or not self._per_file():
                return False
            if self._key != wanted:
                self._forget_files()
            number = key[2]
            if number not in self._linked:
                _piece, rows = self._file_table(number)
                self._add_files([(number, rows, grouped, group_index)], wanted)
            return True
        if self.is_current:
            return True
        if key[1] != wanted[1] or (key[0] != wanted[0] and not self.growing):
            return False
        if not self.growing:
            attach(self.locs, grouped, group_index)
        self._grouped = LocSet(grouped)
        self._keep_key(key)
        return True

    def append(self, new: Localizations) -> int:
        return self.ungrouped.append(new)


class ViewState:
    """One layer's view: its bounds, files and grouping over shared `Tables`,
    and everything needed to render it, without a window around it.

    Standing alone (``ViewState(locs)``) it makes `Tables` of its own; a
    session's layers are given the session's (``tables=``).
    """

    def __init__(self, locs: Optional[Localizations] = None,
                 settings: Optional[RenderSettings] = None,
                 display: Optional[DisplaySettings] = None,
                 filter: Optional[LocFilter] = None,
                 grouped: Optional[Localizations] = None,
                 live: bool = False,
                 extent: Optional[Sequence[float]] = None,
                 tables: Optional[Tables] = None,
                 group_settings: Optional[GroupSettings] = None):
        """``live`` prepares the table for `append`; ``extent`` fixes the area
        the index covers, so the full view does not move as data arrives.
        ``filter`` gives the bounds to start with (its ranges)."""
        self.settings = settings or RenderSettings()
        self.display = display or DisplaySettings()
        self.n_threads = 0
        if tables is None:
            tables = Tables(locs if locs is not None else Localizations({}, {}),
                            group_settings, grouped=grouped, live=live, extent=extent)
        self.tables = tables
        tables._views.add(self)
        self.use_grouped = False
        # the user's setting: field -> (lo, hi), one dict for both tables
        self._ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._files: Optional[List[int]] = None     # file numbers shown; None = all
        # "ungrouped"/"grouped" -> (the LocSet it was built over, its filter):
        # derived from the bounds, never edited on its own
        self._filters: Dict[str, Tuple[LocSet, LocFilter]] = {}
        if filter is not None:
            self._ranges = dict(filter.ranges)

    # ------------------------------------------------------------- the bounds
    @property
    def ranges(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """The bounds this layer keeps, as ``{field: (lo, hi)}`` (a copy)."""
        return dict(self._ranges)

    def _has(self, field: str) -> bool:
        grouped = self.tables.peek_grouped()
        return field in self.tables.locs or (grouped is not None and field in grouped.locs)

    def set_bound(self, field: str, lo: Optional[float], hi: Optional[float]) -> None:
        """Keep ``lo <= field <= hi`` on whichever table is shown.  A field
        neither table has is ignored, as a filter on it would raise."""
        if not self._has(field):
            return
        self._ranges[field] = (lo, hi)
        self.catch_up()            # a mask over every row, not over the old ones
        for locset, f in self._filters.values():
            if field in locset.locs:
                f.set(field, lo, hi)

    def remove_bound(self, field: str) -> None:
        self._ranges.pop(field, None)
        for _locset, f in self._filters.values():
            f.remove(field)

    def set_ranges(self, ranges: Dict[str, Tuple]) -> None:
        """Replace every bound -- a config, a copy of another layer's.  Nothing
        is rebuilt when they are the ones already set."""
        wanted = {f: (lo, hi) for f, (lo, hi) in ranges.items() if self._has(f)}
        if wanted == self._ranges:
            return
        self._ranges = wanted
        self._filters.clear()

    @property
    def files(self) -> Optional[List[int]]:
        return None if self._files is None else list(self._files)

    def set_files(self, files: Optional[Sequence[int]]) -> None:
        """Restrict to these file numbers; None means every file."""
        self._files = None if files is None else sorted(int(f) for f in files)
        self.catch_up()
        for locset, f in self._filters.values():
            self._files_mask(locset, f)

    def _spans(self, locset: LocSet) -> Optional[List[Tuple[int, int]]]:
        """The rows of the files shown, as spans, when they are blocks."""
        if self._files is None or "filenumber" not in locset.locs:
            return None
        rows = locset.file_rows
        if rows is None:
            return None
        return merge_spans(rows[k] for k in self._files if k in rows)

    def _files_mask(self, locset: LocSet, f: LocFilter) -> None:
        """The file choice: spans of rows where each file is one block -- the
        ranges are then tested on the files shown and no other row is read --
        else a mask over every row."""
        spans = self._spans(locset)
        if spans is not None:
            f.remove("files")
            f.restrict(spans)
            return
        f.restrict(None)
        if self._files is None:
            f.remove("files")
        elif "filenumber" in locset.locs:
            f.set_mask("files", np.isin(locset.locs["filenumber"], self._files))

    def filter_for(self, locset: LocSet) -> LocFilter:
        """This layer's filter over ``locset`` -- the ungrouped or the grouped
        table -- built from the bounds the first time it is asked for, and
        kept while that table is the one shown.

        Never extended here: this is read by render threads while a live
        fit appends, and a mask extended there could be longer than a
        column the appending thread has not replaced yet.  `catch_up`, on
        the appending thread, extends it.
        """
        kind = "ungrouped" if locset is self.tables.ungrouped else "grouped"
        cached = self._filters.get(kind)
        if cached is None or cached[0] is not locset:
            # the spans first, so that each range is tested once, on them
            f = LocFilter(locset.locs, spans=self._spans(locset))
            for field, (lo, hi) in self._ranges.items():
                if field in locset.locs:
                    f.set(field, lo, hi)
            self._files_mask(locset, f)
            self._filters[kind] = (locset, f)
            return f
        return cached[1]

    def catch_up(self) -> None:
        """Extend the filters already built to the rows a live table has
        gained -- on the thread that appends, so that a render thread finds
        them current rather than extending them itself.  Only the new rows
        are tested; a hand-set mask (the file choice) has no rule to extend
        it, so its new rows are excluded until it is set again, as
        `LocFilter.append` says."""
        for locset, f in list(self._filters.values()):
            if f.n_rows < len(locset.locs):
                f.append(len(locset.locs) - f.n_rows)

    def columns_changed(self, names: Sequence[str]) -> None:
        """Columns of the shown tables were replaced in place (a live recipe):
        the masks of their bounds are of the old values."""
        self.catch_up()
        for locset, f in self._filters.values():
            for name in names:
                if name in self._ranges and name in locset.locs:
                    f.set(name, *self._ranges[name])

    def rebind(self, tables: Tables) -> None:
        """Look at another table, keeping the bounds the user set -- those its
        columns can take -- and the file choice."""
        self.tables = tables
        tables._views.add(self)
        self._filters.clear()
        self._ranges = {f: b for f, b in self._ranges.items() if f in tables.locs}

    # ------------------------------------------------------- the active table
    @property
    def current(self) -> LocSet:
        """The table shown.  Grouped, it is linked if it is not current --
        except on a table that is still growing, whose grouped form stays as it
        was linked until somebody asks for it again (`group`), and on a
        table whose linking is deferred to a worker (`Tables.deferred`),
        where it is an empty table until the blinks are there."""
        if not self.use_grouped:
            return self.tables.ungrouped
        tables = self.tables
        if tables.growing:
            # a live fit's: as linked last, linked here the first time
            return tables.peek_grouped() or tables.grouped()
        if tables.deferred():
            # being merged elsewhere: nothing until the blinks arrive -- not
            # the localizations, which would be taken for blinks, nor blinks
            # merged with other settings
            return tables.ungrouped if tables.merge_failed else tables.pending()
        return tables.grouped()

    @property
    def locs(self) -> Localizations:
        return self.current.locs

    @property
    def filter(self) -> LocFilter:
        """The filter over the table shown, derived from the bounds."""
        return self.filter_for(self.current)

    @property
    def index(self) -> SpatialIndex:
        return self.current.index

    @property
    def has_grouped(self) -> bool:
        return self.tables.peek_grouped() is not None

    def group(self, settings: Optional[GroupSettings] = None) -> LocSet:
        """The grouped table, linked now if the one there is not current."""
        if settings is not None:
            self.tables.settings = settings
        return self.tables.grouped()

    def append(self, new: Localizations) -> int:
        """Add localizations to the ungrouped table.  Returns how many.

        Called from whichever thread owns the view -- the GUI one -- so that
        nothing here has to be thread-safe; a fitter running in the background
        hands blocks over through a queue.
        """
        n = self.tables.append(new)
        self.catch_up()
        return n

    def show_grouped(self, on: bool, settings: Optional[GroupSettings] = None) -> None:
        """Show blinks, or localizations.  A setting: nothing is linked here,
        but when the table is first read (`current`)."""
        if settings is not None:
            self.tables.settings = settings
        self.use_grouped = bool(on)

    @property
    def waiting(self) -> bool:
        """Whether this layer is set to blinks that are being merged elsewhere,
        and so draws nothing for now (`Tables.pending`)."""
        return self.use_grouped and self.current is self.tables._pending

    @property
    def cull_index(self):
        """The index to narrow a view with, or None when it cannot be used.

        On non-default render axes the index answers questions about the
        table's positions, which is not where the picture is looking, so the
        whole filtered table is rendered instead.  A versatile render is one
        picture of everything rather than a pan over a field, so the pass it
        costs is what SMAP pays too.
        """
        return None if not self.settings.axes.is_default else self.index

    def bounds(self) -> Tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of the whole table, in render coordinates."""
        if self.settings.axes.is_default:
            # the localizations' extent, which the blinks' lies within: a
            # full view does not need them merged
            return self.tables.ungrouped.index.bounds
        return self.current.axis_bounds(self.settings.axes)

    def full_view(self, margin_fraction: float = 0.01) -> Tuple[tuple, tuple]:
        """The coordinate ranges covering every localization."""
        x0, y0, x1, y1 = self.bounds()
        mx, my = (x1 - x0) * margin_fraction, (y1 - y0) * margin_fraction
        return (x0 - mx, x1 + mx), (y0 - my, y1 + my)

    def max_sigma(self, pixelsize: float) -> float:
        """The largest rendering sigma at this zoom, in data units."""
        return max_sigma(self.settings, self.locs, self.current.median_precision, pixelsize)

    def select(self, fov: FieldOfView) -> np.ndarray:
        """The localizations that can contribute to this view, after filtering."""
        return self.snapshot().select(fov)

    def snapshot(self, name: str = "", visible: bool = True) -> "LayerSnapshot":
        """Everything a render of this view needs, frozen, for another thread.

        Taken on the thread that owns the view (the GUI's), and cheap: the
        columns, the mask and the index are references, never copies -- each
        is only ever replaced, never written -- and only the settings are
        copied, because the panels edit those in place.  What a live fit
        appends afterwards goes into the view's own table, filter and index,
        and the snapshot keeps the rows it was taken with
        (`Localizations.snapshot`, `GrowingIndex.snapshot`).
        """
        current = self.current        # once: the table and its selection agree
        f = self.filter_for(current)
        index = self.cull_index
        return LayerSnapshot(
            name=name, visible=visible, locs=current.locs.snapshot(), mask=f.mask,
            known_indices=f.cached_indices,
            index=None if index is None else index.snapshot(),
            median_precision=current.median_precision,
            settings=copy.deepcopy(self.settings), display=copy.deepcopy(self.display),
            n_threads=self.n_threads)

    def keep_indices(self, snapshot: "LayerSnapshot") -> None:
        """Keep the rows a render thread compacted from a snapshot of this
        view, if its mask is still the filter's (`LocFilter.keep_indices`).
        On the GUI thread, once the render is back."""
        found = snapshot.memo.get("indices")
        if found is not None:
            for _locset, f in self._filters.values():
                f.keep_indices(snapshot.mask, found)

    def render(self, fov: FieldOfView) -> RenderedImage:
        return self.snapshot().render(fov)

    def image(self, fov: FieldOfView, white_background=None
              ) -> Tuple[np.ndarray, RenderedImage]:
        return self.snapshot().image(fov, white_background)


def max_sigma(settings: RenderSettings, locs: Localizations, median_precision: float,
              pixelsize: float) -> float:
    """The largest rendering sigma at this zoom, in data units: how far
    beyond the view a localization can still reach into it."""
    if settings.mode in ("hist", "voronoi"):
        return 0.0
    if settings.mode == "tracks":
        return 0.5 * settings.line_width * pixelsize
    if settings.mode == "gauss":
        return settings.sigma
    if settings.mode == "dl":
        # a PSF on whole camera pixels: about one pixel wide in focus,
        # up to three out of focus on an astigmatic one
        return 3 * camera_pixelsize_nm(locs)
    s = settings.sigma_settings
    floor = max(s.min_sigma, s.min_sigma_pixels * pixelsize)
    return max(floor, s.max_factor * median_precision * s.factor)


@dataclass(frozen=True)
class LayerSnapshot:
    """One layer as a render needs it, frozen on the GUI thread.

    The render threads (`gui.render_view`, `gui.view3d`) are handed these and
    nothing else: no layer, no session, no widget.  They used to be handed the
    view widget or the session and walk the live layers from their own
    thread, so a filter changed, a table swapped or a block appended on the
    GUI thread meanwhile was read half-way -- and a widget referenced from a
    worker is how a Qt object's last reference ends up dropped off the GUI
    thread, the deadlock behind `gui.collector`.

    A localization layer carries its table (a `Localizations.snapshot`), the
    filter's mask, the spatial index to cull with (None on custom axes),
    the median precision and copies of its settings; an image layer carries
    a copy of its `ImageData` (the pixels shared) and its plane.  An
    invisible layer carries only its name.
    """

    name: str = ""
    visible: bool = True
    locs: Optional[Localizations] = None
    mask: Optional[np.ndarray] = None
    known_indices: Optional[np.ndarray] = None   # the mask's rows, if the filter had them
    index: object = None
    median_precision: float = 0.0
    settings: Optional[RenderSettings] = None
    display: DisplaySettings = dataclass_field(default_factory=DisplaySettings)
    n_threads: int = 0
    image_data: object = None                    # an `ImageData`, for an image layer
    frame: int = 0
    # an image layer's view of a stack (`session.Layer.set_image_view`): the
    # plane, or a projection over ``image_planes``; and the 3D view's way
    image_show: str = "slice"
    image_planes: Optional[Tuple[int, int]] = None
    image_3d: str = "plane"
    # what the render thread computed and the GUI thread may keep (`indices`)
    memo: dict = dataclass_field(default_factory=dict, compare=False, repr=False)

    @property
    def is_image(self) -> bool:
        return self.image_data is not None

    @property
    def indices(self) -> np.ndarray:
        """The rows the filter keeps.  Compacted here, on the render thread,
        when the filter had not done it yet; `memo` hands them back to it."""
        if self.known_indices is not None:
            return self.known_indices
        found = self.memo.get("indices")
        if found is None:
            found = self.memo["indices"] = np.flatnonzero(self.mask)
        return found

    def max_sigma(self, pixelsize: float) -> float:
        return max_sigma(self.settings, self.locs, self.median_precision, pixelsize)

    def select(self, fov: FieldOfView) -> np.ndarray:
        """The localizations that can contribute to this view, after filtering."""
        if self.index is None:
            return self.indices
        if self.settings.mode == "tracks":
            # a segment across the view can have both ends outside it: the
            # tracks are culled whole, by their boxes (`render.render_tracks`)
            return self.indices
        if self.settings.mode == "voronoi":
            # the cells are of everything the filter keeps, or they would
            # change along the view's edge with every pan (`render.render_voronoi`)
            return self.indices
        margin = self.settings.roi_sigma * self.max_sigma(fov.pixelsize)
        candidates = self.index.query(fov.x0, fov.x1, fov.y0, fov.y1, margin)
        if candidates.size == self.index.n_localizations:
            return self.indices       # nothing to narrow: skip the gather
        # while a fit appends, the index takes a block before the filters
        # have caught up with it (`catch_up`): those rows wait for the next
        mask = self.mask
        candidates = candidates[candidates < len(mask)]
        return candidates[mask[candidates]]

    def render(self, fov: FieldOfView) -> RenderedImage:
        """The planes on ``fov``: accumulated, or an image resampled."""
        if self.is_image:
            return self.image_data.resample(fov, plane=self.shown_plane())
        return render_locs(self.locs, fov, self.settings, self.display,
                           select=self.select(fov), n_threads=self.n_threads)

    def shown_plane(self) -> np.ndarray:
        """An image layer's pixels in 2D: its plane, or its projection."""
        return self.image_data.shown_plane(self.image_show, self.frame, self.image_planes)

    def image(self, fov: FieldOfView, white_background=None
              ) -> Tuple[np.ndarray, RenderedImage]:
        """RGB in [0, 1] and the planes; ``white_background`` overrides the
        display's own, as `Layer.render` does."""
        rendered = self.render(fov)
        return self.display.apply(rendered, white_background), rendered


def snapshot_layer(layer, focus_z: Optional[float] = None) -> LayerSnapshot:
    """A session `Layer`'s snapshot, of either kind.  ``focus_z`` is the
    height an image layer that follows the z filter shows (`focus_z`)."""
    if not layer.visible:
        return LayerSnapshot(name=layer.name, visible=False)
    if layer.is_image:
        # a copy of the record, so that a plane chosen or a registration
        # moved meanwhile is not seen half-way; the pixels are shared, and
        # are replaced rather than written when they change
        image, frame = layer.image, int(layer.frame)
        if getattr(layer, "follow_z", False) and focus_z is not None:
            found = image.index_at_z(focus_z)
            frame = frame if found is None else found
        return LayerSnapshot(name=layer.name, image_data=copy.copy(image),
                             frame=frame, display=copy.deepcopy(layer.display),
                             image_show=getattr(layer, "image_show", "slice"),
                             image_planes=getattr(layer, "image_planes", None),
                             image_3d=getattr(layer, "image_3d", "plane"))
    return layer.state.snapshot(name=layer.name)


def focus_z(layers) -> Optional[float]:
    """The height the localizations are looked at in: the middle of the
    first visible localization layer's z filter, if it has both ends.

    What an image layer that follows z shows the plane at -- a z-stack under
    a slab of 3D localizations then shows the plane they are in.  None when
    no layer limits z on both sides, and the layer keeps its own plane.
    """
    for layer in layers:
        if layer.visible and not layer.is_image:
            lo, hi = layer.ranges.get("z_nm", (None, None))
            if lo is not None and hi is not None:
                return 0.5 * (float(lo) + float(hi))
    return None


def snapshot_layers(layers) -> Tuple[LayerSnapshot, ...]:
    """Every layer of a session, in order, as `snapshot_layer` takes it."""
    z = focus_z(layers) if any(layer.is_image and getattr(layer, "follow_z", False)
                               for layer in layers) else None
    return tuple(snapshot_layer(layer, z) for layer in layers)


def render_panels(layers: Sequence[LayerSnapshot], fov: FieldOfView,
                  panels: Sequence = ("overlay",), planes: Optional[dict] = None,
                  draw: Optional[Callable] = None
                  ) -> Tuple[List[np.ndarray], np.ndarray]:
    """An RGB image per panel on ``fov``, and the summed intensity planes.

    A panel is a layer's position in ``layers``, or ``"overlay"``: every
    visible layer added up, as SMAP does.  Each layer is rendered once
    whatever the panels, so the overlay of side-by-side panels is exactly
    their sum.  A panel of a layer that is not there, or not visible, is
    black.  This is the whole render, and runs on any thread.

    ``planes``, a dict, is given each layer's `RenderedImage` by position:
    what an export of grey intensities per layer reads (`smappy.picture`).
    ``draw(layer, fov)`` renders one layer as ``(rgb, planes)`` instead of
    `LayerSnapshot.image` -- the 3D view's projection -- or returns None to
    leave it out; the panels are put together the same way either way.
    """
    images = {}
    total = np.zeros((fov.ny, fov.nx, 3), np.float32)
    weight = np.zeros((fov.ny, fov.nx), np.float32)
    white_any = False
    wanted = set(panels)
    for i, layer in enumerate(layers):
        if not layer.visible:
            continue
        if draw is None:
            image, rendered = layer.image(fov, white_background=False)
        else:
            drawn = draw(layer, fov)
            if drawn is None:
                continue
            image, rendered = drawn
        white = layer.display.white_background
        white_any = white_any or white
        total += image
        weight += rendered.weight
        if planes is not None:
            planes[i] = rendered
        if i in wanted:
            image = np.clip(image, 0, 1)
            images[i] = luts.on_white(image) if white else image
    total = np.clip(total, 0, 1)
    # once, to the sum: a white ground added to a white ground is still
    # white and would swallow everything drawn on either of them
    images["overlay"] = luts.on_white(total) if white_any else total
    return [images[spec] if spec in images else np.zeros((fov.ny, fov.nx, 3), np.float32)
            for spec in panels], weight
