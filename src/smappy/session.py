"""The state a GUI or a script works on: one table, its layers, undo, history.

No Qt in here.  The GUI subscribes to `on_change` and redraws; a script uses
it the same way without subscribing to anything.
"""
from __future__ import annotations

import dataclasses
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .filter import LocFilter
from .group import GROUP_COLUMNS, GroupSettings
from .images import ImageData, load_images
from .locs import KEPT_BOUNDS, Localizations, concat
from .region_runs import REGION_FILTER as REGION_FILTER_COLUMN
from .metadata import frame_time_ms
from .plugins import Context, Plugin, Result, Selection, refuse_grouped_table
from .io.formats import FileInfo, load as load_any, own_metadata
from .regions import Region
from .view3d import Projection, Slab
from .render import DisplaySettings, RenderAxes, RenderSettings, SigmaSettings
from .undo import Edit, configured as undo_stack
from .layer_state import Tables, ViewState

# the LUTs of a file's channels opened as image layers, in order
CHANNEL_LUTS = ("green", "magenta", "cyan", "yellow", "red", "blue")

# The bounds a layer opens with: what is thrown away is on screen, not hidden.
DEFAULT_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "xy_err_nm": (None, 25.0),
    "logl_rel": (-2.0, None),
    "z_nm": (-500.0, 500.0),
}
# only for a 2D table: a 3D fit's PSF size varies with z by design
DEFAULT_BOUNDS_2D = {"sigma_nm": (None, 180.0)}
PRECISION_FACTOR = 0.5           # rendering sigma = factor * localization precision
GROUPED_BY_DEFAULT = True
# grouping in the GUI: 50 nm / 1 frame (SMAP's File tab opens at 35 nm), xy only.  Two emitters within
# that box in consecutive frames have overlapping PSFs and could not have been
# fitted apart anyway, so a z window would only split real blinks; it stays an
# option in the parameters dialog, with that caveat.
DEFAULT_GROUP_SETTINGS = GroupSettings(dx=50.0, dt=1)


def default_bounds(locs: Localizations,
                   base: Optional[Dict[str, Tuple]] = None) -> Dict[str, Tuple]:
    """What a new layer starts with on this table.

    The defaults -- ``base`` when given (the filters a GUI file or the last
    file's layer carry, `Session.carried_bounds`), else the shipped ones --
    then the bounds the table carries with it (`KEPT_BOUNDS`: Remove
    Localizations' hide flag), for the columns the table has.  One function
    for a layer and for Analysis/Process/Layers' *start from defaults*, which
    kept its own copy without the carried bounds and so showed what the file
    had hidden.
    """
    if base is not None:
        bounds = {f: tuple(b) for f, b in base.items()}
        # the 2D default on a PSF size that varies with z by design: a filter
        # carried from a 2D table onto a 3D one is the default, not a choice
        if "z_nm" in locs and bounds.get("sigma_nm") == DEFAULT_BOUNDS_2D["sigma_nm"]:
            bounds.pop("sigma_nm")
    else:
        bounds = dict(DEFAULT_BOUNDS)
        if "z_nm" not in locs:
            bounds.update(DEFAULT_BOUNDS_2D)
    bounds.update({f: tuple(b) for f, b in
                   (locs.metadata.get(KEPT_BOUNDS) or {}).items()})
    return {f: b for f, b in bounds.items() if f in locs}


class Layer:
    """A filter and how its localizations are drawn -- or a pixel image.

    A ``"locs"`` layer wraps a `ViewState` over the session's `Tables`: its
    bounds (the user's setting, one dict for the grouped and the ungrouped
    table alike), its file choice, whether it shows the grouped table, and its
    render and display settings.  The tables themselves -- the ungrouped one,
    the grouped one, their spatial indices -- are the session's, shared by
    every layer.  An ``"image"`` layer holds an `ImageData` and only the
    display settings; it is resampled onto the view instead of rendered.
    """

    def __init__(self, tables: Tables, name: str = "layer 1",
                 defaults: bool = True,
                 settings: Optional[RenderSettings] = None,
                 display: Optional[DisplaySettings] = None,
                 grouped: Optional[bool] = None,
                 base: Optional[Dict[str, Tuple]] = None):
        """``grouped`` None opens the layer grouped where that is the default
        (`GROUPED_BY_DEFAULT`, a table with frames that is not still being
        fitted).  Nothing is linked here: the grouped table is linked when it
        is first read, or in the GUI's worker (`gui.merging`).  ``base`` is
        what `default_bounds` starts from in place of the shipped defaults."""
        self.kind = "locs"
        self.image: Optional[ImageData] = None
        self.name = name
        self.visible = True
        if settings is None:
            settings = RenderSettings(sigma_settings=SigmaSettings(factor=PRECISION_FACTOR))
        self.state = ViewState(settings=settings, display=display, tables=tables)
        if defaults:
            self.apply_defaults(base)
        locs = tables.locs
        if grouped is None:
            grouped = (GROUPED_BY_DEFAULT and not tables.growing and bool(len(locs))
                       and "frame" in locs)
        if grouped:
            self.show_grouped(True)

    @classmethod
    def from_image(cls, image: ImageData, name: Optional[str] = None,
                   display: Optional[DisplaySettings] = None) -> "Layer":
        layer = cls.__new__(cls)
        layer.kind = "image"
        layer.image = image
        layer.name = name or image.name or "image"
        layer.visible = True
        layer.state = None
        layer.display = display or DisplaySettings(lut="gray")
        # the layer's own plane, not the image's: two layers may show the
        # average and one frame of the same kept stack
        layer.frame = int(image.frame)
        layer.set_image_view()
        return layer

    def set_image_view(self, show: str = "slice", planes=None, follow_z: bool = False,
                       in_3d: str = "plane") -> None:
        """How an image layer shows a stack: ``show`` one plane ("slice") or a
        projection of the planes ``planes`` (first, last; None: all) --
        "max", "mean" or "sum"; with ``follow_z`` the slice is the plane at
        the height the localization layers' z filter is centred on; and
        ``in_3d`` how the 3D view draws a z-stack -- "plane", "max" (a
        maximum-intensity projection along each ray) or "volume"."""
        self.image_show, self.image_planes = show, planes
        self.follow_z, self.image_3d = bool(follow_z), in_3d

    def shown_plane(self, frame: Optional[int] = None) -> np.ndarray:
        """The pixels an image layer shows in 2D (`ImageData.shown_plane`)."""
        return self.image.shown_plane(self.image_show,
                                      self.frame if frame is None else frame,
                                      self.image_planes)

    def show_image(self, image: ImageData, frame: int = 0) -> None:
        """Point an image layer at another image: the source it shows.  A
        layer still called after its old image is renamed after the new."""
        if self.image is not None and self.name == (self.image.name or "image"):
            self.name = image.name or "image"
        self.image, self.frame = image, int(frame)
        self.image_planes = None             # a range of the old stack's planes

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    def get_display(self) -> DisplaySettings:
        return self.display if self.is_image else self.state.display

    def set_display(self, display: DisplaySettings) -> None:
        if self.is_image:
            self.display = display
        else:
            self.state.display = display

    def render(self, fov, white_background=None):
        """RGB in [0, 1] and the intensity plane on ``fov``, whatever the kind.

        ``white_background`` overrides the display's own: what adds several
        layers up turns the sum over once rather than every layer.
        """
        if self.is_image:
            rendered = self.image.resample(fov, plane=self.shown_plane())
            return self.display.apply(rendered, white_background), rendered
        return self.state.image(fov, white_background)

    def bounds(self):
        """(x0, y0, x1, y1) this layer covers, in render coordinates."""
        if self.is_image:
            return None if self.image.is_empty else self.image.bounds
        return self.state.bounds()

    def apply_defaults(self, base: Optional[Dict[str, Tuple]] = None) -> None:
        for field, (lo, hi) in default_bounds(self.state.tables.locs, base).items():
            self.set_bound(field, lo, hi)

    @property
    def ranges(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """The bounds the user set, ``{field: (lo, hi)}``: one set for the
        layer, applied to whichever table it shows."""
        return self.state.ranges

    def set_bound(self, field: str, lo: Optional[float], hi: Optional[float]) -> None:
        """One bound, applied to the table the layer shows -- the grouped
        one's combined values when it is grouped (`layer_state`)."""
        self.state.set_bound(field, lo, hi)

    def remove_bound(self, field: str) -> None:
        self.state.remove_bound(field)

    def set_ranges(self, ranges) -> None:
        """Replace every bound at once."""
        self.state.set_ranges(ranges)

    @property
    def locs(self) -> Localizations:
        """The table this layer draws: grouped or not."""
        return self.state.locs

    @property
    def filter(self) -> LocFilter:
        """The filter over the table this layer draws (derived from `ranges`)."""
        return self.state.filter

    @property
    def files(self) -> Optional[List[int]]:
        """File numbers shown; None = all."""
        return None if self.is_image else self.state.files

    @files.setter
    def files(self, files: Optional[Sequence[int]]) -> None:
        self.set_files(files)

    def set_files(self, files: Optional[Sequence[int]]) -> None:
        """Restrict to these file numbers; None means every file."""
        self.state.set_files(files)

    def rebind(self, tables: Tables) -> None:
        """Look at the session's new tables, keeping the bounds and display
        the user set."""
        if self.is_image:
            return
        self.state.rebind(tables)

    @property
    def grouped(self) -> bool:
        return self.state.use_grouped

    def show_grouped(self, on: bool) -> None:
        """Draw one entry per blink instead of one per frame.  Linked when
        first read -- once for the session, since the grouped table is
        shared."""
        self.state.show_grouped(on)

    def selection(self, index: int = 0, without: Tuple[str, ...] = ()) -> Selection:
        """The *ungrouped* localizations this layer's bounds keep.

        ``without`` names filters to leave out (`LocFilter.mask_without`).
        """
        f = self.state.filter_for(self.state.tables.ungrouped)
        mask = f.mask_without(*without) if without else f.mask
        return Selection(mask, layer=index, name=self.name)

    def grouped_selection(self, index: int = 0, without: Tuple[str, ...] = (),
                          progress=None, locset=None) -> Tuple[Localizations, Selection]:
        """The grouped table and what this layer's bounds keep of it,
        linking it if it is not current (``progress(text, fraction)``) --
        or ``locset``, a grouped table already in hand."""
        if locset is None:
            # the files this layer shows, linked if they are not yet
            locset = self.state.tables.grouped(progress=progress, files=self.state.files)
        f = self.state.filter_for(locset)
        mask = f.mask_without(*without) if without else f.mask
        return locset.locs, Selection(mask, layer=index, name=f"{self.name}, blinks merged")


# The colours a session can be marked with, and the mark each puts in a
# window title.  Coloured circles because a title bar takes no colour of its
# own, and these are drawn in colour by every platform's title font.
COLOURS: Dict[str, Tuple[str, str]] = {
    "red": ("#d62728", "\U0001F534"), "orange": ("#ff7f0e", "\U0001F7E0"),
    "yellow": ("#e8c000", "\U0001F7E1"), "green": ("#2ca02c", "\U0001F7E2"),
    "blue": ("#1f77b4", "\U0001F535"), "purple": ("#9467bd", "\U0001F7E3"),
    "brown": ("#8c564b", "\U0001F7E4"),
}


class LiveTableError(ValueError):
    """A result that changes the localizations, while a fit still adds to them."""


# How many log entries a file carries.  One entry is a plugin path, a line of
# text and its settings -- a few hundred bytes -- and the whole metadata block
# is written as a single HDF5 attribute, so the oldest are dropped rather than
# risking a file that cannot be written.
MAX_HISTORY = 500


def file_history(locs: Localizations) -> List[Dict]:
    """What the file says was done to these localizations, read tolerantly.

    The log is provenance: which plugins changed the table since it was
    loaded, and with which settings.  It is written by another version of the
    program than the one reading it, so an entry that is not a dictionary is
    dropped rather than being allowed to stop a file from opening.
    """
    found = locs.metadata.get("history") or []
    if not isinstance(found, list):
        return []
    return [dict(e) for e in found if isinstance(e, dict)][-MAX_HISTORY:]


def read_localizations(path, progress: Optional[Callable[[str], None]] = None,
                       reader=None, **reader_args) -> Tuple[Localizations, FileInfo]:
    """Read a localization file: ``(locs, info)``.

    The one implementation, used by the GUI's `LoadTask`, by `Session.load` and
    by the loader plugin -- all of which need the slow part off whatever
    thread owns the session, and none of which should own a copy of it.

    Nothing is merged here, though a layer shows merged blinks by default:
    they are merged when first read, and in the GUI in a worker after the
    file has opened (`gui.merging`), so that a file opens at once.
    """
    if progress:
        progress("loading")
    return load_any(path, reader=reader, **reader_args)


def _saved_files(locs: Localizations, info: FileInfo) -> Optional[List[FileInfo]]:
    """The files of a session saved with several, as they were saved; None
    for a table that is one file.

    A smappy HDF5 file written from a session of three files holds their
    entries (``metadata["files"]``, each with its own metadata) and the
    `filenumber` of every row.  Opened again, they are three files again --
    the rows numbered as they were, each with its name, frame time and
    metadata -- and no longer one file whose second and third had lost what
    they were.
    """
    if info.format != "smappy" or "filenumber" not in locs:
        return None
    entries = locs.metadata.get("files")
    if not isinstance(entries, list) or len(entries) < 2 \
            or not all(isinstance(e, dict) for e in entries):
        return None
    numbers = np.asarray(locs["filenumber"])
    if len(numbers) and (numbers.min() < 0 or numbers.max() >= len(entries)):
        return None
    return [FileInfo.from_dict(e) for e in entries]


class Session:
    def __init__(self, locs: Optional[Localizations] = None, path=None):
        # how localizations are linked into blinks: the session's, for every
        # layer -- there is one grouped table (`tables`)
        self._group_settings = DEFAULT_GROUP_SETTINGS
        # the table and its grouped form, shared by every layer
        # the size from which the GUI merges blinks in a worker rather than
        # on first read (`Tables.defer_from`, `gui.merging`); None headless
        self.defer_merge_from: Optional[int] = None
        self.tables = self._new_tables(locs if locs is not None else Localizations({}, {}),
                                       self._group_settings)
        self.path: Optional[Path] = Path(path) if path else None
        # Files `save` refuses to write over.  A fit run for MicroClaw hands
        # its file on as the job's result, with a digest taken there and then,
        # and the window stays open on it: a Ctrl+S afterwards would change
        # what the record vouches for.  The session's work goes to a new file.
        self.protected: set = set()
        # the filters a new file's layer starts from where the last table had
        # no such column: a GUI file's (`Workspace.filters`), else None, the
        # shipped `DEFAULT_BOUNDS` (`carried_bounds`)
        self.default_filters: Optional[Dict[str, Tuple]] = None
        self._live_bounds: Optional[Dict[str, Tuple]] = None
        self.layers: List[Layer] = [Layer(self.tables)]
        self.roi: Optional[Region] = None          # the drawn 2D ROI (regions.py)
        self._rois = None                          # the ROI manager's project
        # what was measured per region of an id column (a cluster's
        # `cluster_id`): column -> evaluation name -> `RegionTable`.  Read
        # from the file on first use, as the ROI project is.
        self._id_regions = None
        # the region filter's last answer and what it was of (`refresh_region_filter`)
        from .region_filter import Outcome
        self._region_outcome = (None, Outcome())
        self._refreshing = False
        self.slab: Optional[Slab] = None          # the 3D view's volume
        self.slab_follows_roi = True              # the 2D ROI sets its footprint
        self.select_in_slab = False               # plugins see only the slab
        # ... and only while the 3D window shows it: a restriction nobody can
        # see is one nobody remembers, and a plugin run an hour after the
        # window was closed would quietly measure a box.  The window clears
        # this when it closes; a script, with no window, keeps it True.
        self.slab_shown = True
        self.projection = Projection()
        self.files: List[FileInfo] = []
        # every pixel image the session knows: the TIFFs opened as images and
        # the camera frames each file kept (`kind == "raw"`, with the file's
        # number in `metadata["filenumber"]`).  An image layer shows one of
        # them; the Render tab offers them all as its sources.
        self.images: List[ImageData] = []
        self.history: List[Dict] = []
        # what the tools that ran on this file left behind, by plugin path:
        # enough to draw their figures again, saved with the file and read
        # back when it is opened.  See `Plugin.keep`.
        self.results: Dict[str, Dict] = {}
        # named transformations (`smappy.transformation`), by name: what
        # Transform applies by name and Register layers measures.  They are
        # the data's, not the GUI's: saved with the localization file
        # (`file_metadata`), never in a workspace.
        self.transformations: Dict[str, Any] = {}
        # every table this session replaced, newest last; see `smappy.undo`
        self.undo_stack = undo_stack()
        self._live = False                 # the table is being appended to
        # what follows the live table: `ColumnRecipe`s applied while it grows
        self._recipes: List = []
        self._listeners: List[Callable[[str], None]] = []
        # set by the GUI: what to write into the file's `gui` group.  A session
        # in a script has none, and saves only data.
        self.gui_state_provider: Optional[Callable[[], Dict]] = None
        # set by the GUI: the windows a plugin may open (`Plugin.window`), by
        # name, each called with the plugin's settings
        self.window_openers: Dict[str, Callable[[Any], None]] = {}
        # set by the GUI: what each picture window shows, by name ("2D",
        # "3D"), as a `picture.ViewFrame` -- where it looks, how fine, its
        # panels -- for *Export Image* to frame its picture on
        self.view_frames: Dict[str, Callable[[], Any]] = {}
        # False in a chain's scratch copy: the chain keeps one record and one
        # undo step for all its steps, so the copy keeps neither (`scratch`)
        self.recording = True
        # What this session is called and its colour, for telling several
        # apart -- two instances side by side, or a snapshot of a live fit
        # beside the fit -- in every window title (`gui.identity`).  Empty
        # name: the file's.  The colour is a name from `COLOURS`, "" for none.
        self.name = ""
        self.colour = ""
        # Whether the table changed since it was last saved or opened: a
        # change that went into the table (`apply`, undo, redo, a fit), not
        # a filter, a layer or an ROI, which are views of it.  The windows put
        # a `*` in their titles, and quitting can ask (`set_unsaved`).
        self.unsaved = False

    @property
    def locs(self) -> Localizations:
        """The table: always the ungrouped one (`tables` has the grouped)."""
        return self.tables.locs

    @locs.setter
    def locs(self, locs: Localizations) -> None:
        """Put another table under the layers, without an undo step or a log
        line -- for a script; a correction goes through `set_locs`."""
        self._take_tables(self._new_tables(locs, self._group_settings))

    def _new_tables(self, locs: Localizations, settings: GroupSettings,
                    **kwargs) -> Tables:
        """`Tables` over ``locs``, deferring its linking as the session does
        (`defer_merge_from`)."""
        return Tables(locs, settings, defer_from=self.defer_merge_from, **kwargs)

    def _take_tables(self, tables: Tables) -> None:
        """Make ``tables`` the session's and every layer's.  Its grouped table
        is linked when a layer showing blinks reads it (`ViewState.current`;
        render threads are handed snapshots taken on the GUI thread, so none
        meets one being linked)."""
        self.tables = tables
        for layer in self.layers:
            layer.rebind(tables)

    # ----------------------------------------------------------- observers
    def on_change(self, callback: Callable[[str], None]) -> None:
        """``callback(what)``: "locs" (new table), "layer" (a filter or display
        setting changed), "layers" (added/removed/visibility), "history" or
        "results" (a tool kept something worth drawing again), "rois" (the
        ROI manager's project), "roi-preview" (a segmenter's proposals,
        `Session.show_preview`), "name" (`rename`, or a file opened with
        a name of its own) and "unsaved" (`set_unsaved`)."""
        self._listeners.append(callback)

    def off_change(self, callback: Callable[[str], None]) -> None:
        """Stop telling ``callback``: a panel that is being replaced."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def changed(self, what: str) -> None:
        # the region filter's flag follows the table, the ROIs and their
        # results before anyone draws them (`region_filter`)
        if what in ("locs", "rois", "append"):
            # and before it, the maths on region values that it may read
            from .region_maths import refresh
            if getattr(self, "_refreshing", True) is False:
                refresh(self)
            self.refresh_region_filter()
        # before anyone hears of the images, so a list of sources shows the
        # one a waiting layer has just taken
        filled = what == "images" and self._fill_empty_image_layers()
        for cb in list(self._listeners):
            cb(what)
        if filled:
            self.changed("layer")

    def _fill_empty_image_layers(self) -> bool:
        """Point every image layer still waiting (`ImageData.empty`) at the
        first image there is: the kept frames, else any opened image.  True
        if one was."""
        waiting = [l for l in self.layers if l.is_image and l.image.is_empty]
        sources = self.raw_images() or [im for im in self.images if not im.is_empty]
        if not waiting or not sources:
            return False
        for layer in waiting:
            layer.show_image(sources[0], 0)
        return True

    def set_unsaved(self, on: bool) -> None:
        """Say whether the table differs from the file; "unsaved" when that
        changes, for the titles."""
        on = bool(on)
        if on != self.unsaved:
            self.unsaved = on
            self.changed("unsaved")

    # ---------------------------------------------------------------- data
    def load(self, path, append: bool = False, progress=None, **reader_args) -> FileInfo:
        """Open a localization file of any known format.

        With ``append`` it joins the table as one more file (a ``filenumber``
        column tells them apart, and the layers can pick); otherwise it
        replaces everything.
        """
        locs, info = read_localizations(path, progress=progress, **reader_args)
        return self.add_file(locs, info, append=append)

    @staticmethod
    def _file_results(info: FileInfo) -> Dict[str, Dict]:
        """The tool results saved in the file ``info`` names; {} otherwise.

        Only a smappy HDF5 file has them, and `load_results` answers {} for
        anything it cannot read, so a table built in memory or read from
        another format opens with none, as before.
        """
        path = getattr(info, "path", "") or ""
        if not str(path).lower().endswith((".h5", ".hdf5")):
            return {}
        from .io.hdf5 import load_results
        return load_results(path)

    @staticmethod
    def _file_images(info: FileInfo) -> List[ImageData]:
        """The images kept in the file ``info`` names, as `_file_results`."""
        path = getattr(info, "path", "") or ""
        if not str(path).lower().endswith((".h5", ".hdf5")):
            return []
        from .io.hdf5 import load_images
        return load_images(path)

    def _take_images(self, images: Sequence[ImageData], number: int) -> None:
        """Add a file's images; its raw frames are file ``number``'s.

        A file saved from a session of several keeps each one's frames under
        that file's number; opened again they keep their order, from
        ``number`` on.  A TIFF saved with a file is simply an image again, and
        one the session has already (the same file) is not added twice.
        """
        raws = [im for im in images if im.kind == "raw"]
        saved = sorted({int(im.metadata.get("filenumber", 0)) for im in raws})
        for im in images:
            if im.kind == "raw":
                im.metadata["filenumber"] = number + saved.index(
                    int(im.metadata.get("filenumber", 0)))
            elif im.path and any(o.kind != "raw" and o.path == im.path
                                 and o.name == im.name and o.shape == im.shape
                                 for o in self.images):
                # (by name too: an `.msr` holds many images of one size)
                continue
            self.images.append(im)
        if images:
            self.changed("images")

    def raw_images(self, number: Optional[int] = None) -> List[ImageData]:
        """The camera frames the files kept -- file ``number``'s, or all."""
        return [im for im in self.images if im.kind == "raw" and
                (number is None or im.metadata.get("filenumber") == number)]

    def add_file(self, locs: Localizations, info: FileInfo, append: bool = False,
                 grouped: Optional[Localizations] = None) -> FileInfo:
        """``grouped`` is ``locs`` already linked, for a caller that did the
        slow part in a worker thread.  It is only used for the first file: an
        append re-links the merged table anyway, since linking cannot be
        extended, and a second file's ids would not follow the first's.
        """
        if not append or not len(self.locs):
            self.files = []
            self.path = Path(info.path)
            # the frames belong to the files they were kept with; an image
            # opened as one stays, as its layer does
            self.images = [im for im in self.images if im.kind != "raw"]
        number = len(self.files)
        saved = _saved_files(locs, info)
        if saved:
            # a session of several files, saved: each comes back as the file
            # it was, its rows keeping their number, rather than as one file
            infos = saved
            numbers = np.asarray(locs["filenumber"], np.int32) + number
        else:
            # the file's own metadata, kept with it: the table's becomes the
            # first file's once another joins it
            info.metadata = own_metadata({**locs.metadata, **info.metadata})
            info.columns = [c for c in locs.columns if c != "filenumber"]
            infos = [info]
            numbers = np.full(len(locs), number, np.int32)
        for each in infos:
            if each.frame_time_ms is None:
                each.frame_time_ms = frame_time_ms(
                    (each.metadata or {}).get("camera") or locs.metadata.get("camera"))
        self.files.extend(infos)
        self._take_images(self._file_images(info), number)
        columns = dict(locs.columns)
        columns["filenumber"] = numbers
        locs = Localizations(columns, dict(locs.metadata))
        if number == 0:
            if grouped is not None:      # the same column, on the linked table
                gc = dict(grouped.columns)
                gc["filenumber"] = (np.full(len(grouped), number, np.int32)
                                    if not saved or "filenumber" not in grouped
                                    else np.asarray(grouped["filenumber"], np.int32))
                grouped = Localizations(gc, dict(grouped.metadata))
            self.set_locs(locs, undoable=False, grouped=grouped)
            # the file's own log continues rather than starting again: what
            # was done to these localizations before they were saved is the
            # part of the record that cannot be reconstructed from anything
            # else, and clearing it here is what used to lose it -- `save`
            # writes this list back over the file's
            self.history = file_history(locs)
            # and what the tools kept with the file comes back with it, by
            # whichever way it was opened -- the GUI and the File/Load
            # plugins come here without passing `load`, and the next save,
            # writing only what the session has, used to drop it
            self.results = self._file_results(info)
            self.changed("results")
            self.take_transformations(self.locs.metadata.get("named_transformations"))
            saved = self.locs.metadata.get("roi")
            self.set_roi(Region.from_dict(saved) if saved else None)
            # a name belongs to the dataset it was given to: a new file takes
            # its own, or none, rather than the last one's
            self._take_identity(self.locs.metadata.get("session"))
            self._rois = None            # this file's own ROIs, read on first use
            self._id_regions = None      # and its own per-cluster results
            self.changed("rois")
            if len(infos) > 1:
                # a session of several files opens on the first (below)
                self._on_files(range(1))
        else:
            merged = concat([self.locs, locs])
            merged.metadata = dict(self.locs.metadata)
            # the linking of the first file says nothing about the merged
            # table, so its group columns go rather than being carried over
            # as a plausible-looking wrong answer; the next grouping writes
            # them again (`group.attach`)
            for column in GROUP_COLUMNS:
                merged.columns.pop(column, None)
            self.set_locs(merged, undoable=True, keep_layers=True,
                          label=f"add {info.name}", text=str(info.path))
            self._on_files(range(number))
        self.locs.metadata["files"] = [f.to_dict() for f in self.files]
        self.log("load", str(info.path), append=append)
        # a file opened is the table as saved; one added to it makes a table
        # no file holds yet
        self.set_unsaved(number > 0)
        return info

    def _follow_files(self) -> None:
        """Take the file list of a table change that changed how many files
        there are -- *Combine files* -- from its ``metadata["files"]``, as the
        undo does (`_restore`).

        A plugin cannot touch the session, and the table's entries are the
        record of its files anyway.  Only a change of count is taken: every
        other plugin hands back the entries it was given, and a list that
        merely differs in a detail is not worth trusting over the session's.
        The layers' file choices and the files' camera frames were numbers
        into the old list, so a layer shows every file again, and the frames
        of files that are gone go -- the first file's stay, its frames being
        where they were.
        """
        entries = self.locs.metadata.get("files")
        if not isinstance(entries, list) or len(entries) == len(self.files) \
                or not all(isinstance(e, dict) for e in entries):
            return
        self.files = [FileInfo.from_dict(e) for e in entries]
        for layer in self.layers:
            if not layer.is_image and layer.files is not None:
                layer.set_files(None)
        kept = [im for im in self.images if im.kind != "raw"
                or int(im.metadata.get("filenumber", 0)) < len(self.files)]
        if len(kept) != len(self.images):
            self.images = kept
            self.changed("images")

    def _on_files(self, numbers) -> None:
        """Layers that showed every file show ``numbers`` -- the files there
        were before one was added, or the first of a session reopened.

        A layer of a session of several files starts on one file, as in SMAP,
        and ticks the others in the Render tab: shown together, every file is
        filtered and linked into blinks before anything is drawn -- 86 s and
        13 GB for ten files of 10 M, where one takes 6 s (`docs/performance.md`,
        "Large tables") -- and files opened together are as often two cells
        as one.  A layer that had chosen its files keeps them.
        """
        for layer in self.layers:
            if not layer.is_image and layer.files is None:
                layer.set_files(list(numbers))

    def file_names(self) -> List[str]:
        return [f.name for f in self.files]

    def files_without(self, column: str) -> List[int]:
        """The files read without ``column``: in the one table their rows
        hold padding there (NaN, or 0 in an integer column) unless a plugin
        has written it since.  Files whose columns are not known (a session
        saved before they were kept) are never listed."""
        return [k for k, f in enumerate(self.files)
                if f.columns is not None and column not in f.columns]

    @property
    def display_name(self) -> str:
        """What the windows call this session: its name, else its file's."""
        if self.name:
            return self.name
        names = self.file_names()
        if len(names) == 1:
            return names[0]
        return f"{len(names)} files" if names else ""

    def rename(self, name: str, colour: Optional[str] = None) -> None:
        """Name the session (and colour it); "" goes back to the file's name."""
        if colour is not None and colour and colour not in COLOURS:
            raise ValueError(f"unknown colour {colour!r}: one of {', '.join(COLOURS)}")
        self.name = " ".join(str(name).split())
        if colour is not None:
            self.colour = colour
        self.changed("name")

    def remove_file(self, number: int) -> None:
        """Drop one file's localizations; the others are renumbered densely."""
        if not 0 <= number < len(self.files):
            return
        numbers = np.asarray(self.locs["filenumber"])
        keep = numbers != number
        columns = {k: np.asarray(v)[keep] for k, v in self.locs.columns.items()}
        renumbered = columns["filenumber"].copy()
        renumbered[renumbered > number] -= 1
        columns["filenumber"] = renumbered
        removed = self.files.pop(number)
        kept = []
        for im in self.images:
            owner = im.metadata.get("filenumber") if im.kind == "raw" else None
            if owner == number:
                continue
            if owner is not None and owner > number:
                im.metadata["filenumber"] = owner - 1
            kept.append(im)
        self.images = kept
        locs = Localizations(columns, dict(self.locs.metadata))
        locs.metadata["files"] = [f.to_dict() for f in self.files]
        for layer in self.layers:               # a layer's file choice follows the numbering
            if not layer.is_image and layer.files is not None:
                layer.files = [n - 1 if n > number else n for n in layer.files if n != number]
        self.set_locs(locs, undoable=True, keep_layers=True,
                      label=f"remove {removed.name}")
        self.log("remove file", removed.name, changed=True)

    # ----------------------------------------------------- transformations
    def add_transformation(self, name: str, transformation) -> None:
        """Keep ``transformation`` under ``name``, replacing one of that name."""
        name = str(name).strip()
        if not name:
            raise ValueError("a named transformation needs a name")
        self.transformations[name] = transformation
        self.changed("transformations")

    def transformation_dicts(self) -> Dict[str, Dict]:
        return {name: t.to_dict() for name, t in self.transformations.items()}

    def take_transformations(self, saved) -> None:
        """Add the named transformations a file kept, as
        `transformation_dicts` wrote them; one that does not read is left out."""
        if not isinstance(saved, dict) or not saved:
            return
        from .transformation import Transformation
        for name, value in saved.items():
            try:
                self.transformations[str(name)] = Transformation.from_dict(value)
            except (KeyError, TypeError, ValueError):
                continue
        self.changed("transformations")

    # -------------------------------------------------------------- images
    def add_image(self, image: Optional[ImageData] = None,
                  name: Optional[str] = None, frame: int = 0) -> Layer:
        """A layer showing ``image``, which joins the session's sources.

        Without one, the layer shows the camera frames kept with the file --
        their average -- when there are any: it is what a new image layer is
        nearly always for, the camera's view under the localizations.  Else
        the first image opened; with neither, pass `ImageData.empty()` for a
        layer that waits for one.
        """
        if image is None:
            sources = self.raw_images() or self.images
            if not sources:
                raise ValueError("no image: open one, or fit with raw frames kept")
            image = sources[0]
        if not image.is_empty and all(im is not image for im in self.images):
            self.images.append(image)
            self.changed("images")
        layer = Layer.from_image(image, name)
        layer.frame = int(frame)
        self.layers.append(layer)
        self.changed("layers")
        return layer

    def add_source(self, image: ImageData) -> ImageData:
        """An image the image layers can show, without a layer of its own:
        an `.msr` holds a dozen, of which one or two are wanted on screen.
        Like any image, it is saved with the session only once a layer
        shows it (`saved_images`)."""
        if all(im is not image for im in self.images):
            self.images.append(image)
            self.changed("images")
        return image

    def open_image(self, path, pixelsize: Optional[float] = None,
                   x0: float = 0.0, y0: float = 0.0) -> Layer:
        """A file's images, a layer each (`open_images`); the first layer."""
        return self.open_images(load_images(path, pixelsize, x0, y0))[0]

    def open_images(self, images: Sequence[ImageData]) -> List[Layer]:
        """A layer for each image -- a file's channels, `images.load_images`
        -- the channels in the colours two localization channels would get,
        a single image in grey."""
        luts_ = CHANNEL_LUTS if len(images) > 1 else ("gray",)
        layers = []
        for i, image in enumerate(images):
            layer = self.add_image(image)
            layer.display = DisplaySettings(lut=luts_[i % len(luts_)])
            layers.append(layer)
        return layers

    def load_image(self, path, pixelsize: Optional[float] = None,
                   x0: float = 0.0, y0: float = 0.0) -> ImageData:
        """Open a file's images as sources, without layers; the first."""
        images = load_images(path, pixelsize, x0, y0)
        for image in images:
            self.add_source(image)
        return images[0]

    def saved_images(self) -> List[ImageData]:
        """What `save` keeps: every file's raw frames, and each image a
        layer shows.  An image opened and then left is not kept: the file
        records what was looked at, not everything that was tried."""
        shown = [l.image for l in self.layers if l.is_image and not l.image.is_empty]
        out = self.raw_images()
        for im in shown:
            if all(im is not o for o in out):
                out.append(im)
        return out

    def full_view(self, margin_fraction: float = 0.01):
        """The ranges covering every visible layer -- or all, if none is."""
        boxes = [l.bounds() for l in self.layers if l.visible] or \
                [l.bounds() for l in self.layers]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return (0.0, 1.0), (0.0, 1.0)
        b = np.array(boxes)
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        mx, my = (x1 - x0) * margin_fraction, (y1 - y0) * margin_fraction
        return (x0 - mx, x1 + mx), (y0 - my, y1 + my)

    def first_locs_layer(self) -> int:
        return next((i for i, l in enumerate(self.layers) if not l.is_image), 0)

    def is_protected(self, path) -> bool:
        """Whether ``path`` is one of the files `save` will not write over."""
        if not path:
            return False
        target = Path(path).resolve()
        return any(target == Path(p).resolve() for p in self.protected)

    def file_metadata(self) -> Dict[str, Any]:
        """The table's metadata as a file carries it: with the session's log,
        its ROI, its name and the ROI manager's project.  What `save` writes,
        and what a new instance is handed (`smappy.handoff`)."""
        metadata = dict(self.locs.metadata)
        # all of it goes in one JSON attribute, so the log is capped rather
        # than allowed to grow without limit over a file's life
        metadata["history"] = self.history[-MAX_HISTORY:]
        if self.roi is not None:
            metadata["roi"] = self.roi.to_dict()
        else:
            metadata.pop("roi", None)
        if self.transformations:
            metadata["named_transformations"] = self.transformation_dicts()
        else:
            metadata.pop("named_transformations", None)
        if self.name or self.colour:
            metadata["session"] = {"name": self.name, "colour": self.colour}
        else:
            metadata.pop("session", None)
        rois = self.roi_state()
        # a tile grid alone is worth keeping: it is where a systematic walk
        # through the file was left off
        if rois and (rois["rois"] or rois["runs"] or rois.get("tile_nm")):
            metadata["roi_project"] = rois
        if self._id_regions is not None:
            kept = {column: [t.to_dict() for t in tables.values() if len(t)]
                    for column, tables in self._id_regions.items()}
            kept = {column: tables for column, tables in kept.items() if tables}
            if kept:
                metadata["id_regions"] = kept
            else:
                metadata.pop("id_regions", None)
        return metadata

    def save(self, path=None, gui_state: Optional[bool] = None,
             region_results: bool = True, region_figures: bool = True) -> Path:
        """Write the table, its history, its ROIs and -- optionally -- the GUI.

        `gui_state` defaults to the `save_gui_state_in_files` preference.  What
        gets written is whatever `gui_state_provider` returns, so the session
        stays ignorant of tabs and windows; the GUI sets it.  Without
        `region_results` the ROIs are saved and the numbers measured in them
        are not -- for a project whose results are large and can be measured
        again.  Without `region_figures` the numbers are saved and what each
        region kept to draw its figures again (`Plugin.keep`) is not.
        """
        from .io.formats import writer_for
        from .io.hdf5 import save_gui_state, save_localizations, save_results
        path = Path(path or self.path)
        if self.is_protected(path):
            raise ValueError(f"{path.name} is kept as the fit wrote it: "
                             f"save under another name")
        # before anything is written: the path is the open file's by default,
        # and a SMAP _sml.mat, a csv or a MINFLUX export is not a file this
        # writes -- it used to be overwritten with HDF5 under its own name
        writer_for(path)
        from .region_results import without_results
        if any(l.grouped for l in self.layers if not l.is_image):
            # a layer shows blinks: the file gets each localization's
            # `group_id` and `n_in_group`, merged now where that is allowed
            # (not while the GUI merges a large table in its worker) -- every
            # file, not only those a layer shows: the file keeps all of them
            self.tables.available(files=self.tables.ungrouped.files)
        metadata = without_results(self.file_metadata(), region_results, region_figures)
        # read before the file is started again: frames opened from it are
        # read from it a plane at a time (`StoredStack`)
        from .io.hdf5 import save_images
        images = [_in_memory(im, path) for im in self.saved_images()]
        for im in images:
            if im.kind == "raw":
                im.metadata["filenumber"] = int(im.metadata.get("filenumber", 0))
        save_localizations(path, self.locs, metadata)
        # after the table: all of these write into the same file, and they are
        # a convenience where the localizations are the point
        if images:
            save_images(path, images)
        if self.results:
            save_results(path, self.results)
        if gui_state is None:
            from . import config
            gui_state = bool(config.get("save_gui_state_in_files", True))
        if gui_state and self.gui_state_provider is not None:
            try:
                save_gui_state(path, self.gui_state_provider())
            except Exception as e:
                # deliberately everything: the localizations are already on
                # disk by now, and the GUI state is a convenience.  Whatever a
                # provider manages to raise must not turn a good save into a
                # reported failure.
                self.log("save", f"the GUI state was not saved: {e}")
        self.path = path
        self.set_unsaved(False)
        return path

    def set_locs(self, locs: Localizations, undoable: bool = True,
                 keep_layers: bool = False,
                 grouped: Optional[Localizations] = None,
                 label: str = "", text: str = "") -> None:
        """Replace the table.  ``label`` is what the undo menu will call this.

        A step is pushed only when ``undoable``; everything else -- a new file,
        the finished form of a live run -- starts the history again, because
        there is nothing left to go back to.
        """
        if self._live:                     # the finished form of the live table
            # the filters from before the run, and any set while it ran
            carried = self.carried_bounds(self._live_bounds)
            self.tables = self._new_tables(locs, self._group_settings)
            self.layers = [Layer(self.tables, base=carried)]    # (the step before the run is already pushed)
            self._live = False
        elif undoable or keep_layers:      # the same data, corrected: keep the layers
            if undoable:
                self._push_undo(label or "change", text)
            self._take_tables(self._new_tables(locs, self._group_settings))
        else:                              # a new file: start over with one layer
            self.undo_stack.clear()
            # with the filters the last file's first layer had: one sets them
            # once for a series of files, and a reset to the defaults on every
            # open undid that each time
            carried = self.carried_bounds()
            self.tables = self._new_tables(locs, self._group_settings, grouped=grouped)
            self.layers = [l for l in self.layers if l.is_image]
            self.layers.insert(0, Layer(self.tables, base=carried))
        self.changed("locs")

    def carried_bounds(self, base: Optional[Dict[str, Tuple]] = None
                       ) -> Dict[str, Tuple]:
        """The filters to keep: what a GUI file saves and a new file starts with.

        The first localization layer's bounds on the columns its table has --
        one removed there stays removed -- and, for the columns it does not
        have, the defaults (`default_filters`, else the shipped ones): a 2D
        table says nothing about the z filter a 3D file should get.  ``base``
        in place of those defaults.  Without
        the bounds the table carries itself (`KEPT_BOUNDS`: a hide flag, the
        region filter), which belong to that table.
        """
        locs = self.locs
        if base is None:
            base = self.default_filters
        if base is None:
            base = dict(DEFAULT_BOUNDS)
            base.update(DEFAULT_BOUNDS_2D)
        bounds = {f: tuple(b) for f, b in base.items() if f not in locs}
        layer = self._locs_layer()
        if layer is not None:
            kept = locs.metadata.get(KEPT_BOUNDS) or {}
            bounds.update({f: tuple(b) for f, b in layer.ranges.items()
                           if f not in kept})
        return bounds

    def apply_default_filters(self, filters: Optional[Dict[str, Tuple]]) -> None:
        """Take a GUI file's filters: for new files, and on the first layer now.

        None is the shipped defaults -- for files to come; the layer keeps
        what it has, as a GUI file without filters says nothing about them.
        """
        self.default_filters = None if filters is None else \
            {f: tuple(b) for f, b in filters.items()}
        layer = self._locs_layer()
        if filters is None or layer is None or not len(self.locs):
            return
        layer.set_ranges(default_bounds(self.locs, self.default_filters))
        self.changed("layer")

    @property
    def group_settings(self) -> GroupSettings:
        """How localizations are linked into blinks: one setting for the
        session, since there is one grouped table."""
        return self._group_settings

    def set_group_settings(self, settings: GroupSettings) -> None:
        """New linking parameters: the grouped table is linked again, once
        for every layer, when it is next read -- in the GUI in a worker for a
        large table, the layers drawing nothing until then.

        Logged like a plugin run: the grouped table is data the user reads
        numbers off, and which `dx` and `dt` produced it is not recoverable
        from the file afterwards.
        """
        self._group_settings = settings
        self.tables.settings = settings
        self.log("merge settings", f"within {settings.dx:g}, dark frames {settings.dt}",
                 settings=asdict(settings), changed=True)
        self.changed("regrouped")

    def _locs_layer(self) -> Optional[Layer]:
        return next((l for l in self.layers if not l.is_image), None)

    def show_grouped(self, index: int, on: bool) -> None:
        """Grouped display for one layer; the grouped table is the session's."""
        self.layers[index].show_grouped(on)

    def add_layer(self, like: Optional[int] = None) -> Layer:
        """A new layer on the same table.

        ``like`` is the index of a localization layer to copy: bounds, files,
        render and display settings, grouping.  A second layer is nearly
        always the first one with one thing changed, so copying is the useful
        start; without ``like`` (or when it names an image) the layer gets the
        defaults and shows no file until one is ticked.
        """
        template = None
        if like is not None and 0 <= like < len(self.layers) and not self.layers[like].is_image:
            template = self.layers[like]
        # on the session's tables: a live table's new rows reach it too, and
        # it is not linked while growing (the default grouping says so)
        layer = Layer(self.tables, name=f"layer {len(self.layers) + 1}",
                      defaults=template is None, base=self.default_filters,
                      settings=(dataclasses.replace(template.state.settings)
                                if template is not None else None),
                      display=(dataclasses.replace(template.get_display())
                               if template is not None else None),
                      grouped=template.grouped if template is not None else None)
        if template is None:
            if "filenumber" in self.locs:
                layer.set_files([])        # starts empty: pick the file(s) it shows
        else:
            layer.set_ranges(template.ranges)
            layer.set_files(template.files)
        self.layers.append(layer)
        self.changed("layers")
        return layer

    def copy_layer(self, source: int, target: int, display: bool = True,
                   bounds: bool = False, files: bool = False,
                   grouping: bool = False) -> None:
        """Make one layer like another, in the parts that are asked for.

        `add_layer` copies everything, because a new layer starts as the old
        one with one thing changed.  This is the other half of that: two
        layers that have drifted apart over a session, and one of them is
        right.  What to copy is a choice rather than everything, since the
        thing that makes two layers two -- the filter, or which file each
        shows -- is usually exactly what must *not* be carried over.
        """
        layers = self.layers
        if not (0 <= source < len(layers) and 0 <= target < len(layers)) \
                or source == target:
            return
        src, dst = layers[source], layers[target]
        if src.is_image or dst.is_image:
            raise ValueError("an image layer has no localization settings to copy")
        if display:
            dst.state.settings = dataclasses.replace(src.state.settings)
            dst.set_display(dataclasses.replace(src.get_display()))
        if bounds:
            dst.set_ranges(src.ranges)
        if files:
            dst.set_files(src.files)
        if grouping and dst.grouped != src.grouped:
            dst.show_grouped(src.grouped)
        self.changed("layer")

    def layer_configs(self) -> List[Dict]:
        """What each localization layer keeps and shows: bounds, grouping,
        files.  The form a chain hands its layers back in, and `Analysis/Process/Layers`
        reads the current bounds from."""
        out = []
        for layer in self.layers:
            if layer.is_image:
                continue
            ranges = layer.ranges
            out.append({"grouped": bool(layer.grouped),
                        "bounds": {f: [lo, hi] for f, (lo, hi) in ranges.items()},
                        "files": None if layer.files is None else list(layer.files)})
        return out

    def set_layer_configs(self, configs: Sequence[Dict]) -> None:
        """Make the localization layers say what ``configs`` say.

        A layer that is named and missing is made as a copy of the first (a
        new layer without a template shows no file until one is ticked, which
        is not what a chain that asked for a second layer means).  Layers
        beyond the list are left as they are: the user may have more than the
        chain knows about.  Each layer's bounds are replaced, not merged --
        the config is the whole filter.
        """
        indices = [i for i, l in enumerate(self.layers) if not l.is_image]
        for n, config in enumerate(configs):
            if n >= len(indices):
                self.add_layer(like=indices[0] if indices else None)
                indices = [i for i, l in enumerate(self.layers) if not l.is_image]
            index = indices[n]
            layer = self.layers[index]
            if "bounds" in config:
                layer.set_ranges({name: (lo, hi) for name, (lo, hi)
                                  in (config.get("bounds") or {}).items()
                                  if name in self.locs})
            if "files" in config:
                layer.set_files(config["files"])
            if "grouped" in config and bool(config["grouped"]) != layer.grouped:
                self.show_grouped(index, bool(config["grouped"]))
        self.changed("layer")

    def scratch(self, locs: Optional[Localizations] = None) -> "Session":
        """A headless copy to run a chain on: same table, layers, filters,
        grouping, ROI and slab; no undo, no log, no listeners.

        The arrays are shared, not copied -- a plugin hands back a new table
        and never edits the one it was given -- and so are the spatial index
        and the grouped table (`Tables.sharing`), so a copy costs neither
        memory nor a relink.  What the copy links afterwards is its own.
        What the chain does to the copy reaches this session only as the
        chain's result.
        """
        copy = Session()
        copy.recording = False
        # A live table is one object that keeps growing, and building the
        # copy's layers links it, which writes `group_id` onto the table it
        # is given: shared, that broke the live view's next append on a
        # column mismatch.  A snapshot shares the arrays and nothing else.
        # ``locs`` is that snapshot when the caller already holds one -- a
        # pinned context's, taken on the GUI thread, which a chain running on
        # a worker passes rather than snapshotting a table that may be
        # growing as it reads it.
        if locs is None:
            locs = self.locs.snapshot() if self._live else self.locs
        if locs is self.locs and not self._live:
            # the same table: its index and its grouped table come along
            tables = Tables.sharing(self.tables)
        else:
            # a snapshot: the grouped table comes along only if it was
            # linked from exactly these rows
            grouped = self.tables.peek_grouped()
            take = (grouped is not None and self.tables.is_current
                    and len(locs) == len(self.locs))
            tables = Tables(locs, self._group_settings,
                            grouped=grouped.locs if take else None)
        copy._group_settings = self._group_settings
        copy.tables = tables
        copy.path = self.path
        copy.files = list(self.files)
        copy.images = list(self.images)
        copy.history = list(self.history)
        copy.roi, copy.slab = self.roi, self.slab
        copy.select_in_slab = self.select_in_slab
        copy.slab_shown = self.slab_shown
        copy.projection = self.projection
        copy.view_frames = self.view_frames     # the windows', framing an export
        copy.layers = []
        for layer in self.layers:
            if layer.is_image:
                copy.layers.append(layer)
                continue
            made = Layer(tables, name=layer.name, defaults=False,
                         settings=dataclasses.replace(layer.state.settings),
                         display=dataclasses.replace(layer.get_display()),
                         grouped=layer.grouped)
            made.set_ranges(layer.ranges)
            made.set_files(layer.files)
            copy.layers.append(made)
        if not any(not l.is_image for l in copy.layers):
            copy.layers.insert(0, Layer(copy.tables))
        if self._rois is not None:
            # the ROI manager's project as it is now, not as the file was
            # saved: a chain's segmenter adds to it and its evaluators run on
            # it, and the chain hands it back as `Result.roi_project`
            from .roi_manager.core import ROIProject
            copy._rois = ROIProject(copy)
            copy._rois.layer = self._rois.layer
            copy._rois.sync()
            copy._rois.from_dict(self._rois.to_dict())
        # what was measured per cluster, so a block finds what is current
        copy._id_regions = {column: dict(tables)
                            for column, tables in self.id_regions.items()}
        return copy

    def remove_layer(self, index: int) -> None:
        if len(self.layers) > 1:
            del self.layers[index]
            self.changed("layers")

    # ------------------------------------------------------------ streaming
    @property
    def live(self) -> bool:
        """Whether a fit is still appending to the table (`begin_live`)."""
        return self._live

    def begin_live(self, extent=None, path=None) -> None:
        """Start an empty table that `append` grows, for a running fit.

        ``extent`` (x0, x1, y0, y1) frames the view before any data arrives.
        The old table is kept for undo.
        """
        if len(self.locs):
            self._push_undo("live acquisition")
        self._live = True
        self._recipes = []
        # the filters the finished table's layer starts with: an empty table
        # has no column to hold them meanwhile
        self._live_bounds = self.carried_bounds()
        self.tables = self._new_tables(Localizations({}, {}), self._group_settings,
                             live=True, extent=extent)
        self.layers = [Layer(self.tables)]
        if path is not None:
            self.path = Path(path)
        self.changed("locs")
        self.set_unsaved(True)

    def _take_identity(self, saved) -> None:
        name = colour = ""
        if isinstance(saved, dict):
            name = str(saved.get("name") or "")
            colour = str(saved.get("colour") or "")
            colour = colour if colour in COLOURS else ""
        if (name, colour) != (self.name, self.colour):
            self.name, self.colour = name, colour
        self.changed("name")       # the file's name is the default either way

    # ------------------------------------------------------- live recipes
    def _follow(self, recipe, plugin: Plugin, result: Result) -> None:
        """Apply a `ColumnRecipe` to the live table now, and to every block.

        The table is the one the fit is still appending to, so the columns go
        into it in place -- on the table so far, which has grown since the
        plugin's snapshot -- rather than replacing it.  No undo step: there is
        no table before it to go back to that the fit could go on filling.
        """
        recipe.name = recipe.name or plugin.path.rsplit("/", 1)[-1]
        self._set_columns(recipe.columns(self.locs))
        derived = result.locs.metadata.get("derived") if result.locs is not None else None
        if derived is not None:          # the math parser's recipe, for grouping
            self.locs.metadata["derived"] = derived
        self._recipes.append(recipe)
        self.changed("append")

    def _set_columns(self, columns: Dict[str, np.ndarray]) -> None:
        """Put columns into the live table, every layer's filter following."""
        for name, values in columns.items():
            values = np.asarray(values)
            self.locs.columns[name] = values
            self.locs._buffers[name] = values    # `extend` grows it from here
        for layer in self.layers:
            if not layer.is_image:               # their masks are of the old values
                layer.state.columns_changed(list(columns))

    def _recipe_block(self, block: Localizations) -> Localizations:
        """A block with the recipes' columns, so it matches the live table."""
        if not self._recipes:
            return block
        columns = dict(block.columns)
        for recipe in self._recipes:
            columns.update(recipe.columns(Localizations(columns, block.metadata)))
        return Localizations(columns, block.metadata)

    def _finish_recipes(self, locs: Localizations) -> Localizations:
        """The fit's finished table with the live recipes applied again --
        over all of it, once, which for a reduction or a colour histogram is
        the answer the partial table only approximated."""
        names = []
        for recipe in self._recipes:
            derived = self.locs.metadata.get("derived")
            locs = recipe.final(locs)
            if derived is not None:
                locs.metadata["derived"] = derived
            names.append(recipe.name)
        self._recipes = []
        self.log("live recipes", "applied to the finished table: " + ", ".join(names))
        return locs

    def end_live(self) -> None:
        """The fit ended without a finished table -- it failed: keep what came.

        What was appended stays, as an ordinary table, so the session is not
        left refusing every table change for a fit that will never finish.
        """
        if self._live:
            self.set_locs(self.locs.snapshot(), undoable=False)

    def append(self, block: Localizations) -> int:
        """Add a block to the live table; every layer takes it in -- with the
        columns of the recipes followed so far (`_follow`)."""
        block = self._recipe_block(block)
        first = not len(self.locs)
        n = self.tables.append(block)
        for layer in self.layers:
            if layer.is_image:
                continue
            layer.state.catch_up()
            if first:
                layer.apply_defaults()          # the columns exist only now
        for recipe in self._recipes:
            if recipe.whole and n:
                self._set_columns(recipe.columns(self.locs))
        if n:
            self.changed("append")
        return n

    # ------------------------------------------------------------- undo
    def _push_undo(self, label: str, text: str = "") -> None:
        """Remember the table as it is now, under the name of what replaces it."""
        if not self.recording:
            return
        self.undo_stack.push(self._here(label, text))

    def _here(self, label: str, text: str = "") -> Edit:
        """The current state as a step, for the other side of the stack."""
        return Edit(label=label, locs=self.locs, text=text,
                    placements=self._placements())

    def _placements(self) -> tuple:
        """Every image there is with where it is: a few numbers each."""
        images = list(self.images)
        for layer in self.layers:
            image = layer.image if layer.is_image else None
            if image is not None and all(image is not im for im in images):
                images.append(image)
        return tuple((im, None if im.placement is None else np.array(im.placement))
                     for im in images)

    def _restore(self, edit: Edit) -> None:
        """Put a remembered table back, rebinding every layer onto it.

        The layers keep their bounds and their display, and a grouped layer is
        linked again on the way (`Layer.rebind`): coming back from a drift
        correction has to show what it showed before, filters and all.
        """
        self._take_tables(self._new_tables(edit.locs, self._group_settings))
        self.files = [FileInfo.from_dict(f)
                      for f in self.locs.metadata.get("files", [])] or self.files
        for image, placement in edit.placements:
            image.placement = None if placement is None else np.array(placement)
        if edit.placements:
            self.changed("layer")

    @property
    def can_undo(self) -> bool:
        return self.undo_stack.can_undo

    @property
    def can_redo(self) -> bool:
        return self.undo_stack.can_redo

    def undo_entries(self) -> List[Edit]:
        """The steps that can be undone, most recent first."""
        return self.undo_stack.entries()

    def redo_entries(self) -> List[Edit]:
        return self.undo_stack.redo_entries()

    def undo_labels(self) -> List[str]:
        return self.undo_stack.labels()

    def redo_labels(self) -> List[str]:
        return self.undo_stack.redo_labels()

    def undo(self, steps: int = 1) -> None:
        """Go back ``steps`` table changes, the most recent first.

        More than one at a time because the menu lists them: picking the third
        entry means the last three, as it does in every other program.  The log
        keeps its entries and gains one saying what was undone -- it is the
        provenance of the localizations, not a second copy of this stack.
        """
        self._travel(steps, "undo")

    def redo(self, steps: int = 1) -> None:
        """Take back an undo, one step or several."""
        self._travel(steps, "redo")

    def _travel(self, steps: int, direction: str) -> None:
        """Walk the stack, restoring each step on the way.

        One method for both directions: they differ in which side of the stack
        is read and which is handed the state being left behind, and nothing
        else.  The log gets a single entry naming every step that was passed,
        rather than one per step, because it is one thing the user did.
        """
        back = direction == "undo"
        done = []
        for _ in range(max(1, int(steps))):
            entries = self.undo_entries() if back else self.redo_entries()
            if not entries:
                break
            here = self._here(entries[0].label, entries[0].text)
            edit = (self.undo_stack.undo(redo_of=here) if back
                    else self.undo_stack.redo(undo_of=here))
            if edit is None:
                break
            self._restore(edit)
            done.append(entries[0].label)
        if done:
            self.log(direction, ", ".join(done), changed=True)
            self.changed("locs")
            self.set_unsaved(True)

    # -------------------------------------------------------- ROI manager
    @property
    def rois(self):
        """The ROI manager's project over this session's files.

        Separate from `roi`, the rectangle or line drawn in the 2D view: this
        is the collection of analysis ROIs, their *use* and their
        evaluation runs, and it is saved with the localization file.
        """
        if self._rois is None:
            from .roi_manager.core import ROIProject
            self._rois = ROIProject(self)
            saved = self.locs.metadata.get("roi_project")
            self._rois.sync()
            if saved:
                self._rois.from_dict(saved)
        else:
            self._rois.sync()
        return self._rois

    # ------------------------------------------------------ region filter
    @property
    def region_filter(self) -> list:
        """The region filter's conditions (`region_filter.Condition`), as the
        table carries them."""
        from .region_filter import conditions
        return conditions(self.locs)

    def set_region_filter(self, found) -> None:
        """Filter on ``found``, a list of conditions -- empty to filter on
        nothing.  The flag is written and, the first time, a bound on it put
        on every layer and kept with the table (`KEPT_BOUNDS`), so a saved
        file opens filtered; with nothing left, both go.  Not a table change:
        no undo step, as an ROI's edit makes none."""
        from .region_filter import store
        from .region_runs import REGION_FILTER
        if not self.recording and not self._live:
            # a chain's copy shares the table with the session it came from
            # (`scratch`): the filter goes into a table of its own, and back
            # to that session as the chain's result
            self.set_locs(Localizations(dict(self.locs.columns), dict(self.locs.metadata)),
                          undoable=False, keep_layers=True)
        had = bool(self.region_filter)
        store(self.locs, list(found))
        self.refresh_region_filter(force=True)
        kept = dict(self.locs.metadata.get(KEPT_BOUNDS) or {})
        if found and not had:
            kept[REGION_FILTER] = [0.5, None]
            for layer in self.layers:
                if not layer.is_image:
                    layer.set_bound(REGION_FILTER, 0.5, None)
        elif not found and had:
            kept.pop(REGION_FILTER, None)
            for layer in self.layers:
                if not layer.is_image:
                    layer.remove_bound(REGION_FILTER)
        if kept:
            self.locs.metadata[KEPT_BOUNDS] = kept
        else:
            self.locs.metadata.pop(KEPT_BOUNDS, None)
        self.set_unsaved(True)
        self.changed("region_filter")
        self.changed("locs")

    def region_filter_outcome(self):
        """What the conditions give now (`region_filter.Outcome`): the
        regions each source keeps, the counts, the warnings."""
        self.refresh_region_filter()
        return self._region_outcome[1]

    def refresh_region_filter(self, force: bool = False) -> None:
        """Write the flag again if anything it depends on moved -- the table,
        the conditions, a region table, an ROI -- and only then: this runs on
        every change of the table or the ROIs."""
        from .region_filter import Outcome, conditions, evaluate, remember_ids, write
        if getattr(self, "_refreshing", True):     # also: not yet made
            return
        found = conditions(self.locs)
        if not found:
            if self._region_outcome[1].flag is not None or REGION_FILTER_COLUMN in self.locs:
                write(self, Outcome())
            self._region_outcome = (None, Outcome())
            return
        key = self._region_filter_key(found)
        if not force and self._region_outcome[0] == key:
            return
        self._refreshing = True
        try:
            outcome = evaluate(self, found)
            write(self, outcome)
            remember_ids(self, outcome)
        finally:
            self._refreshing = False
        self._region_outcome = (self._region_filter_key(found), outcome)

    def _region_filter_key(self, found):
        """What the flag depends on, cheaply: the table, the conditions, the
        tables they read and, for the ROIs, each one's place and results."""
        import json
        from .chain import BLOCK_ROIS
        parts = [id(self.locs), len(self.locs), json.dumps([c.to_dict() for c in found])]
        for c in found:
            if c.source == BLOCK_ROIS:
                project = self.rois
                table = project.tables.get(c.label)
                parts.append(("rois", c.label, id(table),
                              None if table is None else tuple(table.signatures),
                              tuple((r.id, r.use, tuple(r.center), id(r.polygon), r.file_id)
                                    for r in project.rois.values())))
            else:
                table = (self.id_regions.get(c.source) or {}).get(c.label)
                parts.append((c.source, c.label, id(table),
                              None if table is None else (len(table), tuple(table.data[:1])),
                              id(self.locs[c.source]) if c.source in self.locs else None))
        return tuple(parts)

    @property
    def id_regions(self) -> Dict[str, Dict[str, Any]]:
        """What was measured per region of an id column: ``{column: {name:
        RegionTable}}`` (`evaluate_regions`), as the file left it at first."""
        if self._id_regions is None:
            from .region_table import RegionTable
            self._id_regions = {}
            for column, tables in (self.locs.metadata.get("id_regions") or {}).items():
                self._id_regions[column] = {}
                for doc in tables:
                    table = RegionTable.from_dict(doc)
                    self._id_regions[column][table.label] = table
        return self._id_regions

    def evaluate_regions(self, column: str, steps, layer: int = 0,
                         report=None, stop=None, reuse: bool = True,
                         after=None, tag: bool = False):
        """Run ``steps`` once in each region of the id ``column`` and keep
        their tables under it (`region_runs.ColumnRegions.evaluate`): the
        per-cluster counterpart of `ROIProject.evaluate`."""
        from .region_runs import ColumnRegions
        tables, _ = ColumnRegions(self, column, layer).evaluate(
            steps, report=report, stop=stop, reuse=reuse, after=after, tag=tag)
        return tables

    def roi_state(self) -> Optional[dict]:
        """The ROI manager's project as data, or None if it was never opened."""
        return None if self._rois is None else self._rois.to_dict()

    # ----------------------------------------------------------------- roi
    def set_roi(self, roi: Optional[Region]) -> None:
        self.roi = roi
        if self.slab_follows_roi:
            self.slab_from_roi()
        self.changed("roi")

    # ---------------------------------------------------------------- 3D
    def z_range(self) -> Tuple[float, float]:
        """The depth the first locs layer's filter keeps, else the data's.

        In render units, like the slab it goes into: the third axis is ``z_nm``
        unless the render axes say otherwise, and a filter bound is in the
        column's own units and has to be scaled with it.
        """
        layer = self._locs_layer()
        if layer is None or not len(self.locs):
            return (-1.0, 1.0)
        axes = self.axes()
        name = axes.depth_name(self.locs)
        if name is None:
            return (-1.0, 1.0)
        lo, hi = layer.ranges.get(name, (None, None))
        z = np.asarray(axes.depth(self.locs))
        finite = z[np.isfinite(z)]
        dlo, dhi = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
        return (dlo if lo is None else lo / axes.z_scale,
                dhi if hi is None else hi / axes.z_scale)

    def slab_from_roi(self, use_roi: bool = True) -> Slab:
        """The slab from the ROI (or the whole field), z from the filter."""
        z0, z1 = self.z_range()
        if use_roi and self.roi is not None:
            self.slab = Slab.from_region(self.roi, (z0, z1))
        else:
            (x0, x1), (y0, y1) = self.full_view(0.0)
            self.slab = Slab.from_bounds(x0, x1, y0, y1, z0, z1)
        self.changed("slab")
        return self.slab

    def roi_edited(self, roi: Region) -> None:
        """The same ROI, dragged or resized on screen.

        Not `set_roi`: the item under the mouse is the truth and redrawing it
        from here would fight the drag.  The slab still follows it, which is
        what made a 3D view of a line ROI stale until `from ROI` was pressed --
        including the width, which is the one thing a line ROI is dragged for.
        """
        self.roi = roi
        if self.slab_follows_roi:
            self.slab_from_roi()
        self.changed("roi-edited")

    def set_slab(self, slab: Slab, follow_roi: bool = False) -> None:
        self.slab = slab
        self.slab_follows_roi = follow_roi
        self.changed("slab")

    def axes(self, layer: int = 0) -> RenderAxes:
        """Which columns the picture's axes are.

        One set for the whole session in practice -- the layers are composited
        into one grid, so two layers on different axes would mean nothing --
        but it is kept on each layer's `RenderSettings`, as
        `DisplaySettings.white_background` is, so that a render outside a
        session needs nothing else.
        """
        if not self.layers or self.layers[layer].is_image:
            layer = self.first_locs_layer()
        state = self.layers[layer].state if self.layers else None
        return state.settings.axes if state is not None else RenderAxes()

    def set_axes(self, axes: RenderAxes) -> None:
        """Put every localization layer on these axes, and say so.

        The slab is a box in *render* units, so it belongs to the axes it was
        built on: kept across a change it is a box in the wrong quantity, and
        a 3D window showing a ROI-sized box of nanometres against a picture of
        photons against frame is empty -- and stays empty on the way back,
        which is what made it look as though nothing could be selected.  So a
        real change rebuilds it, from the ROI when there is one.
        """
        before = self.axes()
        for layer in self.layers:
            if not layer.is_image:
                layer.state.settings = dataclasses.replace(layer.state.settings, axes=axes)
        if axes != before and len(self.locs):
            # not from the ROI: that was drawn on the old picture and is in its
            # coordinates too, so the whole field on the new axes is the only
            # box that means anything until the user draws another one
            self.slab_from_roi(use_roi=False)
        self.changed("layer")

    @property
    def selects_slab(self) -> bool:
        """Whether a plugin's selection is cut to the slab right now."""
        return self.select_in_slab and self.slab_shown and self.slab is not None

    def set_projection(self, projection: Projection) -> None:
        self.projection = projection
        self.changed("projection")

    # ------------------------------------------------------------- plugins
    def selection(self, layer: int = 0, without: Tuple[str, ...] = ()) -> Selection:
        """What a plugin looks at: the layer's filter, inside the ROI if any.

        An image layer has no localizations; the first locs layer stands in.
        ``without`` names filters of the layer to leave out.
        """
        if self.layers[layer].is_image:
            layer = self.first_locs_layer()
        return self._clip(self.layers[layer].selection(layer, without),
                          self.locs, layer)

    def shown_selection(self, layer: int = 0) -> Selection:
        """What the layer *draws* inside the ROI: its own table, grouped or not.

        For counting what is on screen.  `selection` is over the ungrouped
        table, since that is what plugins get, so on a grouped layer it counted
        frames where the count beside it counted blinks -- "29088" without an
        ROI and "68549 in ROI" with one drawn round nearly everything.  This
        never links: it counts whatever the layer is showing, stale or not,
        which is what the number next to the picture has to agree with.
        """
        if self.layers[layer].is_image:
            layer = self.first_locs_layer()
        lay = self.layers[layer]
        sel = Selection(lay.filter.mask, layer=layer, name=lay.name)
        return self._clip(sel, lay.locs, layer)

    def count_in_view(self, layer: int, x0: float, x1: float, y0: float,
                      y1: float) -> int:
        """How many of what the layer draws (`shown_selection`) lie in the
        box: the count beside a zoomed picture, which the filter's count is
        not.  On the usual axes the spatial index narrows it to a few cells,
        so it costs next to nothing after every zoom; on custom ones it is a
        pass over the table."""
        if self.layers[layer].is_image:
            return 0
        lay = self.layers[layer]
        locs = lay.locs
        if not len(locs):
            return 0
        mask = self.shown_selection(layer).mask
        axes = self.axes(layer)
        x, y = axes.coordinates(locs)
        if axes.is_default:
            idx = lay.state.index.query(x0, x1, y0, y1)
            x, y, mask = x[idx], y[idx], mask[idx]
        return int(np.count_nonzero(mask & (x >= x0) & (x < x1)
                                    & (y >= y0) & (y < y1)))

    def _clip(self, sel: Selection, locs: Localizations, layer: int) -> Selection:
        """Narrow a layer's selection to the ROI and, if asked, the slab."""
        axes = self.axes(layer)
        if self.roi is not None and len(locs):
            # an ROI is drawn on the picture, so it is in render coordinates:
            # a rectangle on a photons-against-frame view selects those
            # localizations, and the usual picture is unchanged
            x, y = axes.coordinates(locs)
            sel.mask = sel.mask & self.roi.mask(x, y)     # never in place: the
            # filter's cached mask is what `Selection` was handed
            sel.roi = self.roi
            sel.name += f", {self.roi}"
        if self.selects_slab and len(locs):
            x, y = axes.coordinates(locs)
            z = axes.depth(locs)
            sel.mask = sel.mask & self.slab.mask(x, y, z)
            sel.roi = self.slab
            sel.name += f", {self.slab}"
        return sel

    def table(self, layer: int = 0, grouping: Optional[str] = None,
              progress: Optional[Callable[[str, float], None]] = None
              ) -> Tuple[Localizations, Selection]:
        """A layer's table and selection, grouped or not: ``(locs, selection)``.

        ``grouping`` is "grouped", "ungrouped", or None for whatever the layer
        itself shows.  This is where the choice a chain or a plugin makes
        (`Context.table`) meets the session: the ungrouped answer is
        `self.locs` with `selection`, and the grouped one is the session's
        grouped table under the layer's bounds, cut to the same ROI and slab.

        Asking for a grouped table that is not current links it.  The
        rule elsewhere is to say what is missing rather than spend minutes
        behind the user's back, but here somebody has asked for exactly this
        table -- a chain step set to grouped -- and linking is the answer.
        What the layer *shows* is left as it was.
        """
        if self.layers[layer].is_image:
            layer = self.first_locs_layer()
        lay = self.layers[layer]
        if grouping is None:
            # what the layer shows: while its blinks are merged in the GUI's
            # worker that is the localizations, and a measurement measures
            # what is on screen rather than linking them here, again
            shown = lay.state.current
            if shown is lay.state.tables.ungrouped:
                return self.locs, self.selection(layer)
            locs, sel = lay.grouped_selection(layer, locset=shown)
            return locs, self._clip(sel, locs, layer)
        if grouping != "grouped":
            return self.locs, self.selection(layer)
        locs, sel = lay.grouped_selection(layer, progress=progress)
        return locs, self._clip(sel, locs, layer)

    def context(self, layer: int = 0,
                progress: Optional[Callable[[str], None]] = None,
                stream: Optional[Callable[[str, Any], None]] = None,
                **extra) -> Context:
        """What a plugin run against this session is given.

        Built on the caller's thread: the table and the selection are read
        here, so handing the context to a worker cannot race a live fit
        rebinding them.

        While a fit is still appending, reading them here is not enough: the
        live table is one object whose columns `append` replaces with longer
        ones, so a plugin holding it saw it grow under it -- a mask of 50 rows
        over a table of 100, or ``x_nm`` read before a block and ``y_nm``
        after it.  The context is then *pinned*: a snapshot of the table
        (`Localizations.snapshot`, no copy) and every layer's selection over
        exactly those rows, all taken now, on this thread.
        """
        if self._live and "locs" not in extra:
            pinned = self._pinned_tables()
            index = layer if not self.layers[layer].is_image else self.first_locs_layer()
            locs, selection = pinned[index]["ungrouped"]
            return Context(session=self, layer=layer, progress=progress,
                           stream=stream, locs=locs, selection=selection,
                           pinned=pinned, **extra)
        return Context(session=self, layer=layer, progress=progress,
                       stream=stream, **extra)

    def _pinned_tables(self) -> Dict[int, Dict[str, Tuple[Localizations, Selection]]]:
        """Every locs layer's ``(table, selection)``, as of now, for `context`.

        The ungrouped table is one snapshot shared by all layers.  A grouped
        table is offered only where the session already has one: it is a
        table of its own, never appended to, so it is safe to hand on -- but it
        does not include what arrived since it was linked, and linking a
        table that is still growing, on a worker, is exactly the race this
        avoids.
        """
        locs = self.locs.snapshot()
        n = len(locs)
        grouped = self.tables.peek_grouped()
        out: Dict[int, Dict[str, Tuple[Localizations, Selection]]] = {}
        for i, lay in enumerate(self.layers):
            if lay.is_image:
                continue
            sel = lay.selection(i)
            # the filter's mask may already be longer than the snapshot if a
            # block came in between; rows are only ever added at the end, so
            # its first n entries are the snapshot's
            sel.mask = sel.mask[:n].copy()
            tables = {"ungrouped": (locs, self._clip(sel, locs, i))}
            if grouped is not None:
                gsel = Selection(lay.state.filter_for(grouped).mask.copy(), layer=i,
                                 name=f"{lay.name}, blinks merged")
                tables["grouped"] = (grouped.locs, self._clip(gsel, grouped.locs, i))
            out[i] = tables
        return out

    def run(self, plugin: Plugin, settings=None, layer: int = 0,
            progress: Optional[Callable[[str], None]] = None,
            stream: Optional[Callable[[str, Any], None]] = None) -> Result:
        """Run a plugin on this session's table; apply what comes back.

        The plugin runs on whatever thread calls this; only `apply` touches
        the session, so a GUI can run the plugin in a worker and apply here.
        """
        ctx = self.context(layer, progress, stream, grouping=plugin.grouping)
        result = plugin.run(ctx, settings)
        refuse_grouped_table(ctx, result, plugin, plugin.path.rsplit("/", 1)[-1])
        self.apply(plugin, result)
        return result

    def apply(self, plugin: Plugin, result: Result, *,
              finishes_live: bool = False) -> None:
        """Put a result in place: its table, its files, its log line, its keep.

        While a fit is still appending (`live`), a result that changes the
        localizations is refused with `LiveTableError`, before anything is
        touched -- except the fit's own, ``finishes_live``, which is the
        finished form of the table it has been streaming, and one that
        carries a `ColumnRecipe`, which follows the live table (`_follow`).
        Applied, a drift correction replaced the growing table with one that
        could not take the next block, and the fit's result then replaced it
        in turn: the view stopped, the correction was lost, and the log kept
        a line for a correction the table did not have.
        """
        recipe = (result.recipe
                  if self._live and not finishes_live and not result.files else None)
        if (self._live and not finishes_live and recipe is None
                and (result.locs is not None or result.files)):
            name = plugin.path.rsplit("/", 1)[-1] or type(plugin).__name__
            raise LiveTableError(
                f"{name} changes the localizations, and the fit is still "
                f"adding to them: not applied")
        if finishes_live and self._recipes and result.locs is not None:
            result.locs = self._finish_recipes(result.locs)
        settings = result.settings
        # The log is the provenance of the localizations, so what goes in it
        # is what changed them -- a measurement can be repeated from the file
        # and would only record that somebody looked.  `Plugin.logged`
        # overrides the rule either way; see it for when that is right.
        changed = (result.locs is not None or bool(result.files)
                   or bool(result.placements))
        for n, (locs, info, grouped) in enumerate(result.files or ()):
            # the first replaces unless the plugin asked to append; the rest
            # always join, or opening three files would keep only the last
            self.add_file(locs, info, append=result.append or n > 0,
                          grouped=grouped)
        if result.locs is not None and not self.files and not len(self.locs) \
                and not self._live:
            # A table made from nothing -- a fit run where no file is open,
            # which is every fit in a batch -- is a file being opened, not a
            # correction: it gets its layers and its file entry, and its own
            # log, rather than an undo step back to an empty session.
            from .io.formats import FileInfo
            where = result.written
            name = Path(where).name if where else plugin.path.rsplit("/", 1)[-1]
            self.add_file(result.locs, FileInfo(name=name, path=str(where or ""),
                                                format=plugin.path, n=len(result.locs)))
            if not where:
                self.path = None         # nowhere to save back to until asked
        elif result.locs is not None and recipe is not None:
            self._follow(recipe, plugin, result)
        elif result.locs is not None:
            # the menu says what it is about to undo, so the step is named
            # after the plugin rather than after the fact that a table changed
            self.set_locs(result.locs, label=plugin.path.rsplit("/", 1)[-1],
                          text=result.text or "")
            self._follow_files()
        if result.locs is not None:
            # a correction is in no file yet; a fit that wrote its table is
            self.set_unsaved(not result.saved)
        if result.placements:
            self._place_images(plugin, result)
        for image in result.image_layers or ():
            self.add_image(image)
        for image in result.image_sources or ():
            self.add_source(image)
        for name, transformation in (result.transformations or {}).items():
            self.add_transformation(name, transformation)
        # The camera frames a fit kept (`smappy.rawframes`), with the table it
        # has just set.  A fit replaces the whole table, so these are all the
        # raw frames there are now -- including any just read back from the
        # file it wrote, which are the same frames.
        images = list(result.images or ())
        if images:
            self.images = [im for im in self.images if im.kind != "raw"]
            for im in images:
                im.metadata["filenumber"] = 0
                self.images.append(im)
            self.changed("images")
        # After the files: opening one starts the log again from the file's
        # own, and an entry written before it would be the one thing lost --
        # which is how a loader's settings never reached the history.
        if plugin.logged if plugin.logged is not None else changed:
            self.log(plugin.path, result.text,
                     settings=asdict(settings) if is_dataclass(settings) else settings,
                     changed=changed, **(result.log or {}))
        # A plugin that writes a column the user is meant to *filter* on says
        # so here, as `{field: (lo, hi)}`.  It cannot set the bound itself: a
        # filter belongs to the table it was built from, and the table the
        # column is in is the one that has just been set here.
        bounds = result.bounds or {}
        for field, (lo, hi) in bounds.items():
            for layer in self.layers:
                if not layer.is_image:
                    layer.set_bound(field, lo, hi)
        if bounds:
            self.changed("locs")
        # the layers a chain step set up -- bounds and grouping per layer, the
        # Render tab's work done by a plugin (`Analysis/Process/Layers`)
        if result.layers:
            self.set_layer_configs(result.layers)
        # the ROI manager's project as a chain left it: the ROIs its segmenter
        # found, the runs of its evaluators and the pipeline they make.  Into
        # the project that is there, which the ROI manager's window holds.
        if result.roi_project is not None:
            self.rois.from_dict(result.roi_project)
            self.changed("rois")
        if result.id_regions:
            from .chain import BLOCK_ROIS
            for column, tables in result.id_regions.items():
                if column == BLOCK_ROIS:         # the ROIs' own, a whole table
                    for label, table in tables.items():
                        self.rois.tables.pop(label, None)
                        self.rois.tables[label] = table     # newest last
                else:
                    self.id_regions.setdefault(column, {}).update(tables)
            self.changed("rois")
        if result.region_filter is not None:
            from .region_filter import Condition
            self.set_region_filter([Condition.from_dict(c) for c in result.region_filter])
        elif result.rois is not None:
            # a segmenter added ROIs to the project it was handed; the
            # manager draws them, and what a preview proposed is now either
            # among them or was not wanted
            self.rois.segment_preview = None
            self.changed("rois")
        # last: a plugin that opened a file has just cleared the session, this
        # one included, and what it worked out belongs to the file it opened
        self.remember(plugin, result)

    def _place_images(self, plugin: Plugin, result: Result) -> None:
        """Move the image layers a result moved, as one undo step -- with the
        table change, when the same result made one (`set_locs` pushed the
        step already, with the placements as they were)."""
        images = []
        for index in result.placements:
            layer = (self.layers[int(index)]
                     if 0 <= int(index) < len(self.layers) else None)
            if layer is None or not layer.is_image or layer.image.is_empty:
                raise ValueError(f"layer {int(index) + 1} shows no image")
            images.append(layer.image)
        if result.locs is None:
            self._push_undo(plugin.path.rsplit("/", 1)[-1], result.text or "")
        for image, placement in zip(images, result.placements.values()):
            image.placement = (None if placement is None
                               else np.array(placement, np.float64))
        self.changed("layer")
        self.set_unsaved(True)

    def show_preview(self, plugin: Plugin, result: Result) -> None:
        """What a preview has to show beyond its figures, put in place here,
        on the GUI thread, since a preview runs on a worker and changes
        nothing.  So far one thing: a segmenter's proposed ROIs
        (`SegmenterPlugin`), which replace the last preview's."""
        preview = result.roi_preview
        if preview is not None:
            self.rois.segment_preview = preview
            self.changed("roi-preview")

    def clear_preview(self) -> None:
        """Drop a segmenter's proposals without adding them."""
        if self._rois is not None and self._rois.segment_preview is not None:
            self._rois.segment_preview = None
            self.changed("roi-preview")

    def remember(self, plugin: Plugin, result: Result) -> None:
        """Keep what this tool worked out, so its figure can be drawn again.

        What is kept is the plugin's own business -- most keep nothing -- and
        it goes into the localization file, so that a drift curve is still
        there to look at when the corrected file is opened again.  A plugin
        that raises here has run successfully and is not going to be told
        otherwise over a figure, so the failure is logged and dropped.
        """
        try:
            saved = plugin.keep(result)
        except Exception as error:
            self.log(plugin.path, f"its result was not kept: {error}")
            return
        if not saved:
            return
        # the settings go with it: a figure a week later has to say what it
        # was made with, and for a measurement this is the only record there
        # is -- its run is not in the log, on purpose (see `apply`)
        self.results[plugin.path] = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "text": result.text, "data": saved,
            "settings": (asdict(result.settings) if is_dataclass(result.settings)
                         else result.settings)}
        self.changed("results")

    def restore_result(self, plugin: Plugin) -> Optional[Result]:
        """The result this tool left in the file, drawable again, or None."""
        saved = self.results.get(plugin.path)
        if not saved:
            return None
        try:
            return plugin.restore(saved.get("data") or {})
        except Exception:
            return None

    def log(self, what: str, text: str = "", **extra) -> None:
        if not self.recording:
            return
        self.history.append({"time": datetime.now().isoformat(timespec="seconds"),
                             "what": what, "text": text, **extra})
        self.changed("history")


def _in_memory(image: ImageData, path: Path) -> ImageData:
    """``image`` with its pixels read, if they are in the file at ``path``.

    Saving starts that file again, so an image still read from it plane by
    plane must be read in full first, and is kept in memory from then on.
    """
    data = image.data
    stored = getattr(data, "path", None)
    if stored is None or Path(stored).resolve() != Path(path).resolve():
        return image
    image.data = np.asarray(data)
    return image
