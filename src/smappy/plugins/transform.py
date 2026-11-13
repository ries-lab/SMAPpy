"""Transform: move a file's localizations, or an image layer.

SMAP's ApplyTransform, TransformLocsManually, Mirror and ChangeCamPixelsize
in one plugin (wave 5, step 5a; `docs/smap_porting/wave5_design.md`).  What
moves is the *target*: a file's localizations, optionally one `channel` --
their coordinates change, logged and undoable, never only the display, so
measurements, exports and ROIs agree with the picture -- or an image layer,
resampled into a new image, `<name>_T`, which the layers that showed the
original then show.  What it is moved by is the *source*: typed
values, a named transformation of the session, a move a loaded file records
(a two-channel fit's *Overlay channels*), a transformation file (`_2ct.h5`,
or this plugin's own `_T.h5`), or the drift of another file.  The maths is in `smappy.transformation`, and a
drift is applied by `Drift.apply`: nothing applies a transformation twice.

Departures from SMAP:

* **One order for the typed operations**: mirror, scale and rotate about a
  centre, then shift.  SMAP shifted first and then scaled and rotated about
  the view's centre; here the centre is the middle of the moved localizations,
  the origin or typed, since a plugin does not read the view.
* **An image is resampled, as in SMAP**, into a new image beside the
  original, `<name>_T` (and, on request, a TIFF, SMAP's `_T.tif`): a source,
  not a layer, and the layers that showed the original show it instead --
  only those, a layer of any other image is left alone.  It was once moved by
  an affine placement instead, lossless but no use for a bead map, which is
  projective: the image of a split chip could not follow its own
  localizations.  By default only the *secondary channel* moves, onto the
  main one, the half a two-channel fit's map moves; the main channel is cut
  out beside it.
* **Precisions and widths stay**, under a scale too, as in SMAP's
  ChangeCamPixelsize, which scaled positions only.  Mirror is about the
  centre, not about SMAP's fixed 256-pixel chip middle, and of the whole
  target, not of half a chip (*file* and *channel* choose the part).
* **Another file's drift** (SMAP's drift correction of one file applied to
  another) is for simultaneous acquisition only, and refused for files whose
  frames differ (`transformation.check_same_frames`).

Step 5b (*Register layers*) adds what it measured to the session's named
transformations through `Result.transformations` and opens this plugin on it
with `settings_for` and the GUI's ``"transform"`` window.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from . import (Context, FormState, Notice, Plot, Plugin, Result, param,
               register)
from ..locs import Localizations

PATH = "Analysis/Process/Transform"
SOURCES = (("typed", "typed"), ("named", "named transformation"),
           ("recorded", "recorded in a loaded file"),
           ("file", "transformation file"), ("drift", "drift of another file"))
# what of an image moves: a channel map takes the secondary half of the chip
# onto the main one, so by default only that half is read
PARTS = (("secondary", "secondary channel onto main"),
         ("all", "the whole image"))
# the move *recorded* takes on auto: the two halves of a two-channel fit put
# on top of each other (`channel_overlay.RECORD_NAME`)
OVERLAY = "overlay channels"
TRANSFORM_FILTER = "Transformation (*_T.h5 *_2ct.h5 *.h5 *.hdf5)"
DRIFT_FILTER = "Drift-corrected file (*.h5 *.hdf5)"
# the most points the Preview draws of each kind: enough to see a structure
# land on another, few enough to draw at once
PREVIEW_POINTS = 20000


@dataclass
class TypedSettings:
    shift_x_nm: float = param(0.0, label="shift x", unit="nm",
                              help="added last, after rotate, scale, mirror")
    shift_y_nm: float = param(0.0, label="shift y", unit="nm",
                              help="added last; y points down in the picture")
    shift_z_nm: float = param(0.0, label="shift z", unit="nm",
                              help="localizations only: an image has no z")
    angle_deg: float = param(0.0, label="rotate", unit="deg",
                             help="counterclockwise as the picture shows it")
    scale: float = param(1.0, label="scale", min=1e-6,
                         help="x and y multiplied about the centre")
    scale_y: Optional[float] = param(None, label="scale y", min=1e-6,
                                     help="auto: as scale", advanced=True)
    mirror: str = param("none", label="mirror",
                        choices=(("none", "none"), ("x", "left-right"),
                                 ("y", "up-down"), ("both", "both")),
                        help="turned over about the centre")
    centre: str = param("data", label="about",
                        choices=(("data", "middle of the target"),
                                 ("origin", "origin (0, 0)"),
                                 ("given", "the centre below")),
                        help="the point rotation, scale and mirror keep")
    centre_x_nm: float = param(0.0, label="centre x", unit="nm",
                               help="the centre, when typed")
    centre_y_nm: float = param(0.0, label="centre y", unit="nm",
                               help="the centre, when typed")


def _targets():
    return [(0, "localizations")]


def _files():
    return [(-1, "all files"), (0, "file 1")]


def _channels():
    return [(-1, "all")]


def _no_names():
    return [("", "-")]


def _no_records():
    return [("", "auto: the last overlay of channels")]


@dataclass
class TransformSettings:
    target: int = param(0, label="target", choices=_targets,
                        help="the localizations, or an image layer")
    file: int = param(0, label="file", choices=_files,
                      help="whose localizations move")
    channel: int = param(-1, label="channel", choices=_channels,
                         help="only the localizations of this channel")
    source: str = param("typed", label="source", choices=SOURCES,
                        help="what the target is moved by")
    typed: TypedSettings = param(default_factory=TypedSettings)
    named: str = param("", label="name", choices=_no_names,
                       help="one of the session's named transformations")
    recorded: str = param("", label="move", choices=_no_records,
                          help="a move the loaded table records, such as a "
                               "two-channel fit's overlay of its channels")
    path: str = param("", label="from file", kind="open_file",
                      file_filter=TRANSFORM_FILTER,
                      help="a _T.h5 this plugin wrote, or a _2ct.h5 of "
                           "Calibrate transform")
    invert: bool = param(False, label="invert",
                         help="apply the transformation backwards: a _2ct.h5 "
                              "maps the second channel onto the first")
    drift_path: str = param("", label="drift of", kind="open_file",
                            file_filter=DRIFT_FILTER,
                            help="a drift-corrected file of the same frames")
    part: str = param("secondary", label="what moves", choices=PARTS,
                      advanced=True,
                      help="of an image: its secondary channel, cut by the "
                           "chip regions the map or the fit records, or all")
    image_to: str = param("", label="save image", kind="save_file",
                          file_filter="TIFF (*.tif *.tiff)", advanced=True,
                          help="the moved image as a TIFF too (SMAP's _T.tif)")
    keep_as: str = param("", label="keep as",
                         help="a name: the transformation joins the "
                              "session's named ones")
    write_to: str = param("", label="write to", kind="save_file",
                          file_filter="Transformation (*_T.h5)",
                          help="a file to write the transformation to",
                          advanced=True)


def settings_for(name: str, target: int = 0, file: int = 0,
                 channel: int = -1) -> TransformSettings:
    """Transform set to apply the named transformation ``name``: what
    Register layers opens it with (the GUI's ``"transform"`` window)."""
    return TransformSettings(target=target, file=file, channel=channel,
                             source="named", named=name)


# ---------------------------------------------------------------- the pieces
def target_rows(locs: Localizations, file: int = -1, channel: int = -1) -> np.ndarray:
    """The mask of a file's localizations (``-1``: every file), of one
    `channel` (``-1``: all)."""
    rows = np.ones(len(locs), bool)
    if file >= 0 and "filenumber" in locs:
        rows &= np.asarray(locs["filenumber"]) == file
    elif file > 0:
        raise ValueError(f"the table has one file, not a file {file + 1}")
    if channel >= 0:
        if "channel" not in locs:
            raise ValueError("the table has no channel attribute")
        rows &= np.asarray(locs["channel"]) == channel
    return rows


def typed_transformation(typed: TypedSettings, locs: Optional[Localizations] = None,
                         rows=None):
    """The typed values as one `Transformation`; the centre *middle of the
    target* is the middle of the box around ``rows`` of ``locs``."""
    from ..transformation import typed as make
    if typed.centre == "data":
        if locs is None or not len(locs):
            centre = (0.0, 0.0)
        else:
            mask = np.ones(len(locs), bool) if rows is None else rows
            if not mask.any():
                raise ValueError("no localizations to find the middle of")
            x = np.asarray(locs["x_nm"], float)[mask]
            y = np.asarray(locs["y_nm"], float)[mask]
            centre = (0.5 * (np.nanmin(x) + np.nanmax(x)),
                      0.5 * (np.nanmin(y) + np.nanmax(y)))
    elif typed.centre == "origin":
        centre = (0.0, 0.0)
    else:
        centre = (typed.centre_x_nm, typed.centre_y_nm)
    return make((typed.shift_x_nm, typed.shift_y_nm, typed.shift_z_nm),
                typed.angle_deg, typed.scale, typed.scale_y, typed.mirror, centre)


def image_centre(image) -> tuple:
    x0, y0, x1, y1 = image.bounds
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def drift_of(path):
    """The drift curve a corrected file keeps: the ``/drift`` group of a
    ``_driftc.hdf5`` (`drift.save_drift_corrected`), or what COMET or RCC
    left in a file the session saved (`KeepsDrift`)."""
    import h5py
    from ..drift import Drift, load_drift
    from ..io.hdf5 import load_results
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no file {path}")
    try:
        with h5py.File(path, "r") as f:
            has_group = "drift" in f
    except OSError:
        raise ValueError(f"{path.name} is not an HDF5 file") from None
    if has_group:
        return load_drift(path)
    saved = load_results(path)
    for plugin in ("Analysis/Drift/COMET", "Analysis/Drift/RCC"):
        data = (saved.get(plugin) or {}).get("data") or {}
        if data.get("drift"):
            return Drift.from_dict(data["drift"])
    raise ValueError(f"{path.name} keeps no drift curve: correct its drift with "
                     "COMET or RCC and save it, or use its _driftc file")


def record_key(record: dict) -> str:
    """How a recorded move is named in the form and in saved settings: its
    name and its time, which no two moves of one table share."""
    return f"{record.get('name') or 'unnamed'} ({record.get('time', '')})"


def recorded_moves(metadata) -> list:
    """``[(key, record, file)]``: every move a table records that carries its
    transformation -- not a drift, which has none -- the table's own and
    each opened file's (``metadata["files"][n]["metadata"]``), once each."""
    metadata = metadata or {}
    found, seen = [], set()
    sources = [(metadata, -1)] + [
        ((f.get("metadata") or {}) if isinstance(f, dict) else {}, n)
        for n, f in enumerate(metadata.get("files") or [])]
    for meta, file in sources:
        for record in meta.get("transformations") or []:
            if not isinstance(record, dict) or not record.get("transformation"):
                continue
            key = record_key(record)
            if key not in seen:
                seen.add(key)
                found.append((key, record, file))
    return found


def recorded_transformation(metadata, key: str = ""):
    """``(Transformation, record, file)`` of the move ``key`` names, or on
    "" (auto) the last overlay of two channels."""
    from ..transformation import Transformation
    moves = recorded_moves(metadata)
    if key:
        chosen = [m for m in moves if m[0] == key]
    else:
        chosen = [m for m in moves if m[1].get("name") == OVERLAY][-1:]
    if not chosen:
        have = ", ".join(m[0] for m in moves) or "none"
        want = repr(key) if key else "an overlay of two channels"
        raise ValueError(f"the loaded table records no {want}; the moves it "
                         f"records: {have}.  Fit as two channels and overlay "
                         "them, or choose a transformation file")
    _, record, file = chosen[0]
    return Transformation.from_dict(record["transformation"]), record, file


def _pixel_sizes(ctx: Context, locs: Localizations, file: int):
    """The camera pixel a file's nm were made with, for a _2ct.h5."""
    from ..locs import table_pixel_sizes
    session = ctx.session
    if session is not None and 0 <= file < len(session.files):
        size = session.files[file].pixelsize_nm
        if size:
            return float(size), float(size)
    try:
        return table_pixel_sizes(locs)
    except ValueError:
        return None


# ---------------------------------------------------------------- the plugin
@register("Analysis/Process/Transform")
class Transform(Plugin):
    """Shift, rotate, scale or mirror a file's localizations or an image
    layer, or apply a named transformation, a move a loaded file records, a
    transformation file or another file's drift."""

    description = ("Move a file's localizations (or one channel of them), or "
                   "resample an image into a new one, <name>_T: typed shift, "
                   "rotation, scale and mirror, a named transformation, a move "
                   "a loaded file records (a two-channel fit's overlay), a "
                   "transformation file, or the drift of another file recorded "
                   "at the same time.")
    Settings = TransformSettings
    name = "Apply transform"
    changes_locs = True
    # every localization of the file moves, whatever the layer merges
    grouping = "ungrouped"
    version = "2"
    preview_help = ("draw the target before (pale) and after (strong) over "
                    "the rest, or the moved image under the localizations, "
                    "without moving anything")

    def changes_locs_for(self, settings) -> bool:
        return settings is None or int(settings.target) == 0

    # -------------------------------------------------------------- the form
    def describe(self, ctx: Optional[Context], settings: TransformSettings) -> FormState:
        image = int(settings.target) != 0
        source = settings.source
        typed = source == "typed"
        state = FormState(shown={
            "file": not image, "channel": not image,
            "typed": typed, "named": source == "named",
            "recorded": source == "recorded",
            "path": source == "file",
            "invert": source in ("named", "recorded", "file"),
            "drift_path": source == "drift",
            "keep_as": source in ("typed", "recorded", "file"),
            "write_to": source != "drift",
            "part": image and source in ("named", "recorded", "file"),
            "image_to": image,
            "typed.shift_z_nm": not image,
            "typed.centre_x_nm": typed and settings.typed.centre == "given",
            "typed.centre_y_nm": typed and settings.typed.centre == "given"})
        session = getattr(ctx, "session", None)
        if session is None:
            return state
        state.choices["target"] = [(0, "localizations")] + [
            (i + 1, f"image layer {i + 1}: {layer.name}")
            for i, layer in enumerate(session.layers)
            if layer.is_image and not layer.image.is_empty]
        if session.files:
            state.choices["file"] = [(-1, "all files")] + [
                (n, f"{n + 1}: {info.name}") for n, info in enumerate(session.files)]
        locs = session.locs
        if "channel" in locs and len(locs):
            state.choices["channel"] = [(-1, "all")] + [
                (int(c), str(int(c))) for c in np.unique(np.asarray(locs["channel"]))]
        moves = recorded_moves(session.locs.metadata)
        state.choices["recorded"] = _no_records() + [(k, k) for k, _, _ in moves]
        names = list(session.transformations)
        state.choices["named"] = [(n, n) for n in names] or _no_names()
        if source == "named" and not names:
            state.notice = Notice("the session has no named transformations: "
                                  "keep one here with *keep as*, or measure one "
                                  "with Register layers", kind="info")
        elif source == "recorded" and not moves:
            state.notice = Notice("the loaded table records no moves: fit as "
                                  "two channels and overlay them (Overlay "
                                  "channels), or choose a transformation file",
                                  kind="info")
        elif image and source == "drift":
            state.notice = Notice("an image has no frames to follow a drift: "
                                  "choose the localizations")
        return state

    # ------------------------------------------------------------ the work
    def run(self, ctx: Context, settings: TransformSettings) -> Result:
        return self._work(ctx, settings, apply=True)

    def preview(self, ctx: Context, settings: TransformSettings) -> Result:
        """The same move, drawn instead of made."""
        return self._work(ctx, settings, apply=False)

    def _work(self, ctx: Context, settings: TransformSettings, apply: bool) -> Result:
        if int(settings.target) != 0:
            return self._move_image(ctx, settings, apply)
        locs = ctx.locs
        rows = target_rows(locs, int(settings.file), int(settings.channel))
        if not rows.any():
            raise ValueError("the target has no localizations: " + self._which(ctx, settings))
        which = self._which(ctx, settings)
        if settings.source == "drift":
            moved, text, transformation = self._drift(locs, rows, settings, which)
        else:
            transformation = self._transformation(ctx, settings, locs, rows)
            moved = self._apply(locs, rows, transformation, settings, which)
            text = f"{which}: {int(rows.sum()):,} localizations, {transformation}"
        plot = _plot_locs(locs, moved, rows)
        if not apply:
            return Result(text=text, plot=plot, settings=settings,
                          data={"transformation": transformation})
        written = self._write(transformation, settings)
        if written:
            text += f"; written to {written}"
        kept = {}
        if settings.keep_as.strip() and settings.source in ("typed", "file"):
            kept[settings.keep_as.strip()] = transformation
        return Result(locs=moved, text=text, plot=plot, settings=settings,
                      data={"transformation": transformation},
                      transformations=kept, written=written)

    @staticmethod
    def _which(ctx: Context, settings: TransformSettings) -> str:
        session = ctx.session
        file = int(settings.file)
        if file < 0:
            what = "all files"
        elif session is not None and file < len(session.files):
            what = f"file {file + 1} ({session.files[file].name})"
        else:
            what = f"file {file + 1}"
        if int(settings.channel) >= 0:
            what += f", channel {int(settings.channel)}"
        return what

    def _transformation(self, ctx: Context, settings: TransformSettings,
                        locs: Optional[Localizations] = None, rows=None,
                        image=None):
        """The `Transformation` the settings name, inverted if asked."""
        from ..transformation import load_transformation
        source = settings.source
        if source == "typed":
            if image is not None:
                typed = settings.typed
                if typed.centre == "data":
                    cx, cy = image_centre(image)
                    typed = replace(typed, centre="given", centre_x_nm=cx,
                                    centre_y_nm=cy)
                return typed_transformation(typed)
            return typed_transformation(settings.typed, locs, rows)
        if source == "named":
            session = ctx.session
            named = getattr(session, "transformations", {}) or {}
            if not settings.named:
                raise ValueError("choose a named transformation")
            if settings.named not in named:
                raise ValueError(f"the session has no transformation called "
                                 f"{settings.named!r}; it has "
                                 + (", ".join(named) or "none"))
            t = named[settings.named]
        elif source == "recorded":
            metadata = None if ctx.session is None else ctx.session.locs.metadata
            if metadata is None and locs is not None:
                metadata = locs.metadata
            t, _, _ = recorded_transformation(metadata, settings.recorded)
        elif source == "file":
            if not settings.path:
                raise ValueError("choose a transformation file")
            sizes = None if locs is None else _pixel_sizes(ctx, locs, max(int(settings.file), 0))
            if sizes is None and image is not None:
                # no table to say the camera pixel: the image is the camera's
                sizes = image.pixelsize_xy
            t = load_transformation(settings.path, sizes)
        else:
            raise ValueError(f"unknown source {source!r}")
        return t.inverse() if settings.invert else t

    @staticmethod
    def _apply(locs, rows, transformation, settings, which):
        from ..transformation import transform_table
        name = settings.named if settings.source == "named" else settings.keep_as.strip()
        return transform_table(locs, transformation, rows, name=name, what=which)

    @staticmethod
    def _drift(locs, rows, settings: TransformSettings, which: str):
        """Another file's drift, frame by frame, on the target rows."""
        from datetime import datetime
        from ..transformation import check_same_frames
        if not settings.drift_path:
            raise ValueError("choose the drift-corrected file whose drift to apply")
        if "frame" not in locs:
            raise ValueError("the table has no frame attribute to apply a drift by")
        drift = drift_of(settings.drift_path)
        name = Path(settings.drift_path).name
        check_same_frames(len(drift), np.asarray(locs["frame"])[rows],
                          f"the drift of {name}", which)
        moved = drift.apply(locs, rows=rows)
        record = {"time": datetime.now().isoformat(timespec="seconds"),
                  "name": "", "rows": which, "n": int(rows.sum()),
                  "drift_of": str(settings.drift_path), "method": drift.method,
                  "n_frames": len(drift)}
        moved.metadata["transformations"] = list(
            moved.metadata.get("transformations") or []) + [record]
        span = drift.drift.max(axis=0) - drift.drift.min(axis=0)
        text = (f"{which}: {int(rows.sum()):,} localizations corrected by the "
                f"drift of {name} ({drift.method}, {len(drift)} frames, range "
                f"x {span[0]:.0f}, y {span[1]:.0f}, z {span[2]:.0f} nm"
                + ("" if drift.matrix is None
                   else f", and its {drift.transformation}") + ")")
        return moved, text, drift

    @staticmethod
    def _write(transformation, settings: TransformSettings) -> Optional[str]:
        from ..transformation import Transformation, save_transformation
        if not settings.write_to or not isinstance(transformation, Transformation):
            return None
        name = settings.named if settings.source == "named" else settings.keep_as.strip()
        return str(save_transformation(settings.write_to, transformation, name,
                                       overwrite=True))

    def _move_image(self, ctx: Context, settings: TransformSettings, apply: bool):
        from ..transformation import Transformation, transform_image
        session = ctx.session
        index = int(settings.target) - 1
        if session is None or not 0 <= index < len(session.layers) \
                or not session.layers[index].is_image:
            raise ValueError(f"layer {index + 1} is not an image layer")
        if settings.source == "drift":
            raise ValueError("an image has no frames to follow a drift: move "
                             "the localizations instead")
        layer = session.layers[index]
        image = layer.image
        locs = session.locs if len(session.locs) else None
        transformation = self._transformation(ctx, settings, locs, None, image)
        if transformation.moves_z:
            raise ValueError("an image has no z: this transformation moves "
                             "localizations in z")
        secondary = (settings.part == "secondary"
                     and settings.source in ("named", "recorded", "file"))
        made = []
        if secondary:
            main, other = channel_rects(ctx, transformation, image, locs)
            if _overlaps(main, image.bounds):
                # the main channel stays where it is, cut out beside the moved
                # one: the two channels as two layers, as the localizations are
                made.append(_image_like(image, transform_image(
                    image.data, image.grid, Transformation(), main),
                    _named(image, "_main"), None, "main"))
            moved = _image_like(image, transform_image(
                image.data, image.grid, transformation, other),
                _named(image, "_T"), transformation, "secondary")
            what = "secondary channel moved onto the main" + (
                ", the main cut beside it" if made else "")
        else:
            moved = _image_like(image, transform_image(
                image.data, image.grid, transformation),
                _named(image, "_T"), transformation, "all")
            what = "moved"
        made.append(moved)
        text = (f"image layer {index + 1} ({layer.name}): {what}, {transformation}; "
                + ", ".join(im.name for im in made) + " added to the images, "
                "the layers that showed the original now show " + moved.name)
        plot = _plot_image(made, locs)
        data = {"transformation": transformation}
        if not apply:
            return Result(text=text, plot=plot, settings=settings, data=data)
        written = self._write(transformation, settings)
        if written:
            text += f"; written to {written}"
        if settings.image_to:
            text += f"; the image saved as {write_tiff(settings.image_to, moved)}"
        kept = {}
        if settings.keep_as.strip() and settings.source in ("typed", "recorded", "file"):
            kept[settings.keep_as.strip()] = transformation
        # sources, not layers: the layers that showed the original show the
        # moved image now, and the rest pick it (or the main cut) as they like
        return Result(text=text, plot=plot, settings=settings, image_sources=made,
                      images_replaced=[(image, moved)],
                      transformations=kept, data=data, written=written)


# ------------------------------------------------------------------ images
def channel_rects(ctx: Optional[Context], transformation, image, locs=None):
    """``(main, secondary)``, each channel's rectangle on the chip in nm
    (x0, y0, x1, y1): from the map itself, which keeps the regions it was
    measured between (`transformation.from_channel_transform`), else from
    what the loaded fit records (``metadata["channels"]``), read with the
    camera pixel of the table, else of the image."""
    from ..transformation import channel_regions, region_nm
    regions = transformation.origin.get("channel_regions")
    metadata = {}
    if ctx is not None and ctx.session is not None:
        metadata = ctx.session.locs.metadata or {}
    elif locs is not None:
        metadata = locs.metadata or {}
    if not regions:
        found = [channel_regions(metadata.get("channels"))] + [
            channel_regions(((f.get("metadata") or {}) if isinstance(f, dict) else {})
                            .get("channels"))
            for f in metadata.get("files") or []]
        regions = next((r for r in found if r), None)
    if not regions:
        raise ValueError("neither the transformation nor the loaded table says "
                         "where the two channels lie on the chip: set *what "
                         "moves* to the whole image")
    pixel = transformation.pixelsize_nm
    if pixel is None and locs is not None and ctx is not None:
        pixel = _pixel_sizes(ctx, locs, 0)
    if pixel is None:
        pixel = image.pixelsize_xy
    return region_nm(regions[0], pixel), region_nm(regions[1], pixel)


def _overlaps(rect, bounds) -> bool:
    return (rect[0] < bounds[2] and bounds[0] < rect[2]
            and rect[1] < bounds[3] and bounds[1] < rect[3])


def _named(image, suffix: str) -> str:
    """``wf.tif`` -> ``wf_T.tif``: the name SMAP gave a moved image."""
    name = Path(image.name or "image")
    return f"{name.stem}{suffix}{name.suffix}"


def _image_like(image, data, name: str, transformation, part: str):
    """A new image on ``image``'s grid holding ``data``."""
    from ..images import ImageData
    metadata = dict(image.metadata)
    metadata["transformed"] = {
        "from": image.name, "part": part,
        "transformation": None if transformation is None else transformation.to_dict()}
    return ImageData(np.ascontiguousarray(data, np.float32), image.pixelsize,
                     image.x0, image.y0, name=name, frame=image.frame,
                     metadata=metadata, pixelsize_y=image.pixelsize_y,
                     frames=None if image.frames is None else np.array(image.frames))


def write_tiff(path, image) -> Path:
    """``image`` as a float TIFF, its pixel size in the resolution tags as
    SMAP writes it (pixels per centimetre, which ImageJ reads as µm)."""
    import tifffile
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        path = path.with_suffix(".tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    per_cm = 1e7 / float(image.pixelsize)
    tifffile.imwrite(str(path), np.asarray(image.data, np.float32),
                     resolution=(per_cm, per_cm), resolutionunit="CENTIMETER")
    return path


# ------------------------------------------------------------------ figures
def _sample(n: int, limit: int = PREVIEW_POINTS) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, limit, replace=False))


def _plot_locs(before: Localizations, after: Localizations, rows: np.ndarray) -> Plot:
    """The rest grey, the target pale where it was and strong where it goes."""
    others = np.flatnonzero(~rows)
    target = np.flatnonzero(rows)
    others = others[_sample(len(others))]
    target = target[_sample(len(target))]
    x0 = np.asarray(before["x_nm"], float)
    y0 = np.asarray(before["y_nm"], float)
    x1 = np.asarray(after["x_nm"], float)
    y1 = np.asarray(after["y_nm"], float)

    def draw(ax) -> None:
        if len(others):
            ax.plot(x0[others], y0[others], ",", color="0.6", label="the rest")
        ax.plot(x0[target], y0[target], ".", ms=1.5, color="#9ecae1", alpha=0.6,
                label="target before")
        ax.plot(x1[target], y1[target], ".", ms=1.5, color="#d62728", alpha=0.7,
                label="target after")
        ax.set(xlabel="x (nm)", ylabel="y (nm)", title="before and after")
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()                 # as the picture hangs: y down
        ax.legend(fontsize=7, loc="upper right", markerscale=6)
    return Plot(draw)


def _plot_image(images: list, locs: Optional[Localizations]) -> Plot:
    """The new images in colour -- the main channel green, the moved one
    magenta, so that what lies on its partner is white -- under a sample of
    the localizations: a good map puts them on the same structures."""
    colours = [(0.2, 0.9, 0.2), (1.0, 0.0, 1.0)] if len(images) == 2 else [(1.0, 1.0, 1.0)]
    first = images[0]
    ny, nx = first.shape
    px, py = first.pixelsize_xy
    rgb = np.zeros((ny, nx, 3), np.float32)
    for image, colour in zip(images, colours):
        plane = np.asarray(image.plane_at(image.frame), np.float32)
        lit = plane[plane > 0]
        top = float(np.percentile(lit, 99.5)) if lit.size else 1.0
        rgb += np.clip(plane / (top or 1.0), 0, 1)[..., None] * np.asarray(colour, np.float32)
    rgb = np.clip(rgb, 0, 1)
    extent = (first.x0, first.x0 + nx * px, first.y0 + ny * py, first.y0)
    sample = None if locs is None else _sample(len(locs))

    def draw(ax) -> None:
        ax.imshow(rgb, extent=extent, interpolation="nearest")
        if sample is not None and len(sample):
            ax.plot(np.asarray(locs["x_nm"])[sample], np.asarray(locs["y_nm"])[sample],
                    ",", color="#ffd92f", alpha=0.6, label="localizations")
            ax.legend(fontsize=7, loc="upper right", markerscale=6)
        ax.set(xlabel="x (nm)", ylabel="y (nm)",
               title="main green, moved magenta" if len(images) == 2 else "the moved image")
        ax.set_aspect("equal")
    return Plot(draw)
