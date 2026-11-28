"""Pixel size calibration: the camera's pixel in the sample, its angle and mirror, from beads
imaged at several stage positions.

The work is `smappy.pixel_size` (the beads associated over all frames first, the pixel after);
this reads the stack and the stage's position in each frame's tags, fits the beads in pixels
(Gaussian 2D, `api.fit` with a pixel of 1 nm, so the nm columns are pixels), says what was
measured and, on request, enters the pixel in the microscope database for the state the
microscope was in (`microscope_db.with_pixel`).  A run is remembered by file (`last`) so that the
form's notice can offer it to the microscope manager, through the GUI's ``pixel size entry``
window, where it is checked and saved by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import Context, FormState, Notice, Plot, Plugin, Result, param, register

TIFF_FILTER = "Image stacks (*.tif *.tiff *.ome.tif);;All files (*)"

#: the last run's numbers, by the stack's path: what the notice offers the microscope manager
_LAST: Dict[str, dict] = {}
#: what `describe` read of a stack, by (path, modification time): it is asked after every edit
_SEEN: Dict[Tuple[str, float], dict] = {}


@dataclass
class PixelSizeSettings:
    path: str = param("", label="beads", kind="open_file", file_filter=TIFF_FILTER,
                      help="a stack of beads near focus, the stage moved between frames: "
                           "steps of a third to two thirds of the field, in two directions")
    backlash: bool = param(True, label="fit backlash",
                           help="a term per stage axis for the direction it last moved in")
    enter: bool = param(False, label="enter in the microscope database",
                        help="the pixel becomes the microscope's for the state its devices "
                             "were in; else the notice's button shows it in the manager first")
    pixel_hint_nm: Optional[float] = param(
        None, label="expected pixel", unit="nm", min=0.0, advanced=True,
        help="auto: the microscope database's; only narrows the search, a wrong value "
             "costs time, not the answer; 0: none")
    x_tag: str = param("", label="stage x tag", advanced=True,
                       help="auto: XPositionUm, or a tag beside one named for y")
    y_tag: str = param("", label="stage y tag", advanced=True, beside=True,
                       help="auto: YPositionUm")
    stage_unit: str = param("um", label="stage unit", choices=[("um", "µm"), ("nm", "nm")],
                            advanced=True, help="the unit of the stage's tags")
    tolerance_px: float = param(0.5, label="tolerance", unit="px", min=0.05, advanced=True,
                                help="how far two beads' shifts may differ and agree")
    min_photons: float = param(1000.0, label="min photons", min=0.0, advanced=True,
                               help="spots dimmer than this are not beads")
    sigma_px: float = param(1.3, label="PSF width", unit="px", min=0.5, advanced=True,
                            help="the beads' Gaussian width near focus, for the detection")


@register("Localize/Pixel size calibration")
class PixelSizeCalibration(Plugin):
    """The camera's pixel size, angle and mirror against the stage, from beads at stage
    positions."""
    description = ("measure the camera's effective pixel size, its rotation and mirror against "
                   "the stage from beads imaged at several stage positions, and enter it for "
                   "the microscope")
    changes_locs = False
    Settings = PixelSizeSettings
    version = "1"

    def run(self, ctx: Context, settings: PixelSizeSettings) -> Result:
        from .. import pixel_size
        if not settings.path:
            raise ValueError("choose a stack of beads at several stage positions")
        found = read_stack(settings, ctx.report)
        hint = settings.pixel_hint_nm
        if hint is None and found["database_pixel"] is not None:
            hint = float(np.mean(found["database_pixel"]))
        frame, xy, shape = found["frame"], found["xy_px"], found["shape"]
        ctx.report(f"{len(frame)} beads in {found['n_frames']} frames")
        result = pixel_size.measure(frame, xy, found["stage_um"], shape, settings.tolerance_px,
                                    backlash=settings.backlash, pixel_hint_nm=hint or None,
                                    report=ctx.report)
        values = result.values
        text = [result.text()]
        if abs(values["skew_deg"]) > SKEW_WARNING_DEG:
            ctx.warn(f"the camera's axes are {values['skew_deg']:+.1f}° from square on the "
                     f"stage: a stage whose axes are not square, or frames placed wrongly")
        resolution = found["resolution"]
        if found["database_pixel"] is not None:
            x, y = found["database_pixel"]
            text.append(f"the microscope database's pixel: {x:g} x {y:g} nm "
                        f"({_percent(values['pixel_x_nm'], x)}, "
                        f"{_percent(values['pixel_y_nm'], y)})")
        _LAST[str(settings.path)] = {"pixel_nm": result.pixel_nm, "resolution": resolution,
                                     "values": dict(values)}
        if settings.enter:
            text.append(enter(resolution, result.pixel_nm, ctx.warn))
        positions = (frame, xy, tuple(shape))

        def draw(figure) -> None:
            draw_all(figure, result, *positions)

        return Result(text="\n".join(text), settings=settings,
                      data={k: v for k, v in values.items()},
                      plot=Plot(draw, panels=3, size=(14, 4.5)))

    def describe(self, ctx: Optional[Context], settings: PixelSizeSettings) -> FormState:
        """Which microscope and state the file is of, what the database says its pixel is, and
        -- after a run -- the button that shows the measured one in the microscope manager."""
        if not settings.path:
            return FormState()
        try:
            seen = _seen(settings.path)
        except Exception as error:                   # an unreadable file says so at Run
            return FormState(notice=Notice(f"{Path(settings.path).name}: {error}",
                                           kind="warning"))
        last = _LAST.get(str(settings.path))
        where = seen["where"]
        if not where:
            text = ("no microscope in the database recognises this file: the pixel can be "
                    "measured, and entered once one does")
            return FormState(notice=Notice(text, "Microscope manager...", "microscope manager",
                                           kind="info"))
        said = seen["pixel_text"]
        if last is None:
            return FormState(notice=Notice(f"{where}: {said}", "Microscope manager...",
                                           "microscope manager", kind="info"))
        x, y = last["pixel_nm"]
        return FormState(notice=Notice(
            f"{where}: measured {x:.1f} x {y:.1f} nm; {said}", "Show in microscope manager...",
            "pixel size entry", kind="info"))


#: a camera's axes further from square than this on the stage are worth a warning
SKEW_WARNING_DEG = 1.0


def _percent(measured: float, said: float) -> str:
    return f"{100.0 * (measured - said) / said:+.1f} %"


def read_stack(settings: PixelSizeSettings, report=None) -> dict:
    """The stack's beads in pixels, its stage positions and what the databases say about it."""
    from .. import pixel_size
    from ..api import fit
    from ..frametags import FrameTags
    from ..io.tiff import open_stack, resolve_camera
    say = report or (lambda text: None)
    source = open_stack(settings.path)
    source.tags = FrameTags()
    say(f"reading {source.n_frames} frames")
    images = np.concatenate([block for _, block in source.frames(chunk=100)])
    tags = source.tags.table()
    unit = 1e-3 if settings.stage_unit == "nm" else 1.0
    stage = pixel_size.stage_positions(tags, len(images), settings.x_tag, settings.y_tag, unit)
    resolution = resolve_camera(source)
    camera = resolution.camera_metadata()
    # positions in pixels: a pixel of 1 nm, and the image's own corner, not the chip's
    camera = replace(camera, pixelsize_um=0.001, roi=None)
    if camera.conversion is None or camera.offset is None:
        camera = replace(camera, conversion=camera.conversion or 1.0,
                         offset=camera.offset if camera.offset is not None
                         else float(np.percentile(images, 1)))
    say("fitting the beads")
    locs = fit(images, camera=camera, sigma=settings.sigma_px, cutoff_mode="dynamic",
               raw_frames=0)
    keep = np.asarray(locs["photons"], float) >= settings.min_photons
    frame = np.asarray(locs["frame"], np.int64)[keep]
    xy = np.column_stack([np.asarray(locs["x_nm"], float), np.asarray(locs["y_nm"], float)])[keep]
    scope = resolution.microscope
    return {"frame": frame, "xy_px": xy, "stage_um": stage, "shape": images.shape[1:],
            "n_frames": len(images), "resolution": scope,
            "database_pixel": scope.pixel_nm if scope is not None else None}


def _seen(path) -> dict:
    """The microscope and state of a stack, and the database's pixel for them, read once per
    version of the file."""
    from ..io.tiff import open_stack, resolve_camera
    key = (str(path), Path(path).stat().st_mtime)
    if key not in _SEEN:
        scope = resolve_camera(open_stack(path)).microscope
        where, pixel_text = "", ""
        if scope is not None and scope.microscope is not None:
            where = state_words(scope)
            pixel = scope.pixel_nm
            pixel_text = (f"the database says {pixel[0]:g} x {pixel[1]:g} nm" if pixel
                          else "the database has no pixel for it")
        _SEEN[key] = {"where": where, "pixel_text": pixel_text}
    return _SEEN[key]


def state_words(scope) -> str:
    """``M2 with 3DA on``: the microscope and the state of the device its pixel follows."""
    entry = scope.microscope
    raw = entry.parameters.get("pixel_nm")
    device = raw.get("device") if isinstance(raw, dict) and "device" in raw \
        else entry.lens_device()
    state = scope.state_names().get(device) if device else None
    return f"{entry.name} with {device} {state}" if state is not None else entry.name


def entry_with_pixel(resolution, pixel_nm):
    """(the microscope's entry with the measured pixel, what changed in words) for a file's
    `MicroscopeResolution`; raises a ValueError saying why not."""
    from .. import microscope_db
    if resolution is None or resolution.microscope is None:
        raise ValueError("no microscope in the database recognises these beads: add it in the "
                         "microscope manager, then run again")
    if resolution.version:
        raise ValueError(f"these beads were taken with {resolution.name}'s setup until "
                         f"{resolution.version}, not its current one")
    entry = microscope_db.database().get(resolution.name) or resolution.microscope
    return microscope_db.with_pixel(entry, resolution.state_names(), pixel_nm)


def enter(resolution, pixel_nm, warn=None) -> str:
    """Enter the measured pixel in the user's microscope database; what was done, in words."""
    from .. import microscope_db
    try:
        entry, words = entry_with_pixel(resolution, pixel_nm)
    except ValueError as error:
        if warn is not None:
            warn(str(error))
        return "not entered in the microscope database"
    path = microscope_db.save_microscope(entry)
    return f"entered as {words}, in {path}"


def last_entry(settings) -> tuple:
    """For the GUI's ``pixel size entry`` window: the entry with the pixel the last run on
    ``settings.path`` measured, and what changed; raises a ValueError when there is none."""
    last = _LAST.get(str(getattr(settings, "path", "")))
    if last is None:
        raise ValueError("run the calibration first: there is no pixel measured for this file")
    return entry_with_pixel(last["resolution"], last["pixel_nm"])


# ------------------------------------------------------------------ figures

def draw_all(figure, result, frame: np.ndarray, xy_px: np.ndarray,
             shape: Optional[Tuple[int, int]] = None) -> None:
    """The three panels of a run: offsets against the stage, residuals, the beads on one map."""
    left, middle, right = figure.subplots(1, 3)
    draw_offsets(left, result)
    draw_residuals(middle, result)
    draw_beads(right, result, frame, xy_px, shape)


def draw_offsets(ax, result, truth: Optional[np.ndarray] = None) -> None:
    """Each frame's camera offset (dots) and where the stage's fit puts it (circles), joined in the
    order of the scan; frames left out grey.  ``truth`` (n, 2), the true offsets, as crosses."""
    o = result.offsets_px
    fit = result.fit
    predicted = fit.predict(result.stage_um, fit.directions)
    used = fit.used
    ax.plot(o[:, 0], o[:, 1], "-", color="0.7", lw=0.8, zorder=1)
    ax.plot(o[used, 0], o[used, 1], "o", ms=4, label="measured")
    ax.plot(predicted[used, 0], predicted[used, 1], "o", ms=8, mfc="none", color="#c06000",
            label="fitted from the stage")
    left = np.isfinite(o[:, 0]) & ~used
    if left.any():
        ax.plot(o[left, 0], o[left, 1], "o", ms=4, color="0.6", label="left out")
    if truth is not None:
        ax.plot(truth[:, 0], truth[:, 1], "x", color="k", ms=5, label="true")
    ax.set(xlabel="camera offset x (px)", ylabel="camera offset y (px)",
           title="each frame's offset on the camera")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.2)                     # room for the legend beside the scan
    ax.invert_yaxis()
    ax.legend(fontsize=8)


def draw_residuals(ax, result) -> None:
    """What the stage's fit leaves of each frame's offset, in nm on the sample, by frame."""
    v = result.values
    pixel = np.array([v["pixel_x_nm"], v["pixel_y_nm"]])
    r = result.fit.residual_px * pixel
    frames = np.arange(len(r))
    used = result.fit.used
    for k, (axis, colour) in enumerate((("x", "C0"), ("y", "C1"))):
        ax.plot(frames[used], r[used, k], "o-", ms=3, lw=0.8, color=colour, label=axis)
        ax.plot(frames[~used], r[~used, k], "x", ms=5, color=colour)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set(xlabel="frame", ylabel="residual (nm)",
           title=f"residual {v['residual_nm']:.1f} nm rms")
    ax.legend(fontsize=8)


def draw_beads(ax, result, frame: np.ndarray, xy_px: np.ndarray,
               shape: Optional[Tuple[int, int]] = None) -> None:
    """The frames put back by their offsets (outlines, ``shape`` (height, width) in pixels) and
    the beads on one map: each where all the frames that saw it agree, larger the more frames saw
    it; the spots of no bead -- seen in one frame only -- small and grey."""
    from matplotlib.patches import Rectangle
    o = result.offsets_px
    frame = np.asarray(frame, np.int64)
    ok = np.isfinite(o[frame, 0])
    placed = xy_px[ok] - o[frame[ok]]
    bead = result.bead[ok]
    if shape is not None:
        height, width = shape
        for f in np.flatnonzero(np.isfinite(o[:, 0])):
            ax.add_patch(Rectangle((-o[f, 0], -o[f, 1]), width, height, fill=False, lw=0.6,
                                   color="0.75" if result.fit.used[f] else "#e0453a"))
    ax.plot(placed[bead < 0, 0], placed[bead < 0, 1], ".", ms=2, color="0.6")
    seen = np.bincount(bead[bead >= 0], minlength=len(result.bead_xy_px))
    p = result.bead_xy_px
    ax.scatter(p[:, 0], p[:, 1], s=4 + 3 * seen, c=np.arange(len(p)) % 10, cmap="tab10",
               vmin=0, vmax=9, linewidths=0)
    ax.set(xlabel="x (px)", ylabel="y (px)",
           title=f"{len(p)} beads from {int(np.isfinite(o[:, 0]).sum())} frames on one map")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.invert_yaxis()
