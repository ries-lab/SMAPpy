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
    manager).
    """

    def __init__(self, locs: Localizations, settings: Optional[GroupSettings] = None,
                 *, grouped: Optional[Localizations] = None, live: bool = False,
                 extent: Optional[Sequence[float]] = None,
                 link: Optional[Linker] = None):
        """``grouped`` is ``locs`` already linked with ``settings`` -- by a
        worker thread, off the GUI's, which is the only way opening a large
        file does not freeze the window."""
        self.ungrouped = LocSet(locs, extent=extent, growable=live)
        self.settings = settings if settings is not None else GroupSettings()
        self._link_with = link
        self._grouped: Optional[LocSet] = None
        self._key = None
        if grouped is not None:
            self._grouped = LocSet(grouped)
            self._key = self._wanted()

    @classmethod
    def sharing(cls, other: "Tables") -> "Tables":
        """A second `Tables` over the same table and the same grouped table,
        without building anything: a chain's copy of the session.  What the
        copy links afterwards is its own; this one keeps what it had."""
        made = cls.__new__(cls)
        made.ungrouped = other.ungrouped
        made.settings = other.settings
        made._link_with = other._link_with
        made._grouped = other._grouped
        made._key = other._key
        return made

    @property
    def locs(self) -> Localizations:
        return self.ungrouped.locs

    @property
    def growing(self) -> bool:
        return self.ungrouped.growing

    def _wanted(self):
        return (len(self.locs), asdict(self.settings))

    @property
    def is_current(self) -> bool:
        """Whether the grouped table there is the one `grouped` would hand out."""
        return self._grouped is not None and self._key == self._wanted()

    def peek_grouped(self) -> Optional[LocSet]:
        """The grouped table linked last, current or not; None if never."""
        return self._grouped

    def grouped(self, progress=None) -> LocSet:
        """The grouped table for the rows and settings as they are now.

        Links -- the slow part -- when there is none or the key has moved.
        Linking writes `group_id` and `n_in_group` back onto the ungrouped
        table (`group.attach`), but not while the table is still growing: a
        block arriving from the fitter would not carry those columns, and
        `extend` refuses a table whose columns do not match.
        """
        if not self.is_current:
            key = self._wanted()
            if self._link_with is not None:
                table = self._link_with(self.settings, not self.growing)
            else:
                table, _ = group(self.locs, self.settings,
                                 attach_columns=not self.growing, progress=progress)
            self._grouped = LocSet(table)
            self._key = key
        return self._grouped

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

    def _files_mask(self, locset: LocSet, f: LocFilter) -> None:
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
            f = LocFilter(locset.locs)
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
        self._filters.clear()
        self._ranges = {f: b for f, b in self._ranges.items() if f in tables.locs}

    # ------------------------------------------------------- the active table
    @property
    def current(self) -> LocSet:
        """The table shown.  Grouped, it is linked if it is not current --
        except on a table that is still growing, whose grouped form stays as it
        was linked until somebody asks for it again (`group`)."""
        if not self.use_grouped:
            return self.tables.ungrouped
        if self.tables.growing and self.tables.peek_grouped() is not None:
            return self.tables.peek_grouped()
        return self.tables.grouped()

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
        if on:
            self.group(settings)
        self.use_grouped = bool(on)

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
            return self.index.bounds
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
    if settings.mode == "hist":
        return 0.0
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
            return self.image_data.resample(fov, self.frame)
        return render_locs(self.locs, fov, self.settings, self.display,
                           select=self.select(fov), n_threads=self.n_threads)

    def image(self, fov: FieldOfView, white_background=None
              ) -> Tuple[np.ndarray, RenderedImage]:
        """RGB in [0, 1] and the planes; ``white_background`` overrides the
        display's own, as `Layer.render` does."""
        rendered = self.render(fov)
        return self.display.apply(rendered, white_background), rendered


def snapshot_layer(layer) -> LayerSnapshot:
    """A session `Layer`'s snapshot, of either kind."""
    if not layer.visible:
        return LayerSnapshot(name=layer.name, visible=False)
    if layer.is_image:
        # a copy of the record, so that a plane chosen or a registration
        # moved meanwhile is not seen half-way; the pixels are shared, and
        # are replaced rather than written when they change
        return LayerSnapshot(name=layer.name, image_data=copy.copy(layer.image),
                             frame=int(layer.frame),
                             display=copy.deepcopy(layer.display))
    return layer.state.snapshot(name=layer.name)


def snapshot_layers(layers) -> Tuple[LayerSnapshot, ...]:
    """Every layer of a session, in order, as `snapshot_layer` takes it."""
    return tuple(snapshot_layer(layer) for layer in layers)


def render_panels(layers: Sequence[LayerSnapshot], fov: FieldOfView,
                  panels: Sequence = ("overlay",)
                  ) -> Tuple[List[np.ndarray], np.ndarray]:
    """An RGB image per panel on ``fov``, and the summed intensity planes.

    A panel is a layer's position in ``layers``, or ``"overlay"``: every
    visible layer added up, as SMAP does.  Each layer is rendered once
    whatever the panels, so the overlay of side-by-side panels is exactly
    their sum.  A panel of a layer that is not there, or not visible, is
    black.  This is the whole render, and runs on any thread.
    """
    images = {}
    total = np.zeros((fov.ny, fov.nx, 3), np.float32)
    weight = np.zeros((fov.ny, fov.nx), np.float32)
    white_any = False
    wanted = set(panels)
    for i, layer in enumerate(layers):
        if not layer.visible:
            continue
        white = layer.display.white_background
        white_any = white_any or white
        image, rendered = layer.image(fov, white_background=False)
        total += image
        weight += rendered.weight
        if i in wanted:
            image = np.clip(image, 0, 1)
            images[i] = luts.on_white(image) if white else image
    total = np.clip(total, 0, 1)
    # once, to the sum: a white ground added to a white ground is still
    # white and would swallow everything drawn on either of them
    images["overlay"] = luts.on_white(total) if white_any else total
    return [images[spec] if spec in images else np.zeros((fov.ny, fov.nx, 3), np.float32)
            for spec in panels], weight
