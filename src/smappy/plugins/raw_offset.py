"""*Raw channel offset*: how far the bead map of a split camera is off in the sample.

The thin plugin around `smappy.raw_offset`, whose docstring has the method: the crosstalk of
every molecule into the other half of the chip, far too dim to be detected, summed from the raw
frames at where the map puts it.  It needs a table fitted as *two channels* (``camera_channel``),
overlaid or not, the map (*Overlay channels*' record, or a ``_2ct.h5``) and the movie.

The movie is the one the table records (``metadata["source"]``); when it is not there any more
-- the table was copied, the share is mounted elsewhere -- a file of that name is looked for
beside the table, in its parent folder and one folder down (`find_raw`), and the form says what
it found.

It changes no coordinates.  What it measured is kept as a named transformation
(`Result.transformations`) that *Transform* applies to the overlaid table's channel 1 -- the
form's notice opens it -- and, on request, the map with the correction folded in is written as a
``_2ct.h5`` for *Overlay channels* to use on the next sample.  Applying it automatically was
tried for the crosstalk *pairs* and taken out (`channel_overlay`'s docstring): a correction is
the user's decision, made from the figures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import Context, FormState, Notice, Plot, Plugin, Result, param, register

TIFF_FILTER = "Image stacks (*.tif *.tiff *.ome.tif *.dcimg);;All files (*)"
TRANSFORM_FILTER = "Channel transformation (*_2ct.h5 *.h5);;All files (*)"
DEFAULT_NAME = "raw channel offset"
NEEDS = ("needs a table fitted as two channels (camera_channel): fit with channels > read: "
         "two channels")
DIRECTIONS = ("main -> secondary", "secondary -> main")


@dataclass
class RawOffsetSettings:
    raw: str = param("", label="raw frames", kind="open_file", file_filter=TIFF_FILTER,
                     help="the movie the table was fitted from; auto: the one it records, or "
                          "a file of that name near the table")
    registration: str = param("", label="channel map", kind="open_file",
                              file_filter=TRANSFORM_FILTER,
                              help="the bead map (_2ct.h5); auto: the one Overlay channels "
                                   "moved the table with")
    start: Optional[int] = param(None, label="first frame", min=0,
                                 help="auto: the middle of the fitted frames")
    frames: int = param(10000, label="frames", min=100,
                        help="how many frames are read: about 1.5 nm per 10,000 on a "
                             "typical movie")
    grid: int = param(3, label="regions", min=1, max=6,
                      help="the field divided into n x n regions, each measured on its own")
    model: str = param("shift", label="correction", choices=("shift", "affine"),
                       help="what is fitted to the regions' offsets")
    name: str = param("", label="name",
                      help="what the correction is kept as for Transform; auto: "
                           "raw channel offset")
    save: bool = param(False, label="save corrected map",
                       help="also write the map with the correction as <map>_raw_2ct.h5, "
                            "for Overlay channels")
    min_photons: float = param(300.0, label="min photons", min=0.0, advanced=True,
                               help="molecules dimmer than this are left out")
    lag: int = param(50, label="control", unit="frames later", min=1, advanced=True,
                     help="the control frame: what is still there this much later is "
                          "background")
    radius: int = param(5, label="patch", unit="px", min=3, max=10, advanced=True,
                        help="half the width of the patches summed")


# ------------------------------------------------------------------ finding
def _name_parts(source: str) -> Tuple[str, str]:
    """A recorded path's file name and its folder's, whatever system wrote it."""
    parts = [p for p in re.split(r"[\\/]", str(source)) if p]
    return (parts[-1] if parts else "", parts[-2] if len(parts) > 1 else "")


def find_raw(locs, table_path: Optional[str] = None) -> Optional[Path]:
    """The movie ``locs`` was fitted from: the path it records, else a file of that name beside
    the table, in the table's parent folder, in the recorded folder's name under either, or one
    folder further down.  None when there is none."""
    source = (locs.metadata.get("source")
              or (locs.metadata.get("provenance") or {}).get("source"))
    if not source:
        return None
    if Path(str(source)).exists():
        return Path(str(source))
    name, folder = _name_parts(source)
    if not name or not table_path:
        return None
    here = Path(table_path).parent
    for base in (here, here.parent):
        for candidate in (base / name, base / folder / name):
            if candidate.exists():
                return candidate
        if base.is_dir():
            found = sorted(base.glob(f"*/{name}"))
            if found:
                return found[0]
    return None


def table_path(ctx: Optional[Context], locs) -> Optional[str]:
    """Where the table's file is, from the session: the file its rows belong to."""
    session = getattr(ctx, "session", None)
    files = getattr(session, "files", None) or []
    if not files:
        return None
    number = 0
    if locs is not None and "filenumber" in locs and len(locs):
        number = int(np.asarray(locs["filenumber"])[0])
    if 0 <= number < len(files):
        return getattr(files[number], "path", None) or None
    return None


def overlay_record(locs) -> Optional[dict]:
    """*Overlay channels*' record of the move, when the table was overlaid."""
    from ..channel_overlay import RECORD_NAME
    for record in reversed(locs.metadata.get("transformations") or []):
        if record.get("name") == RECORD_NAME:
            return record
    return None


# ------------------------------------------------------------------ the map
@dataclass
class ChipMap:
    """The bead map in chip pixels, both ways, and where it is kept (for a corrected copy)."""
    to_reference: object
    to_secondary: object
    channel_transform: object = None      # the `ChannelTransform`, when it is read from a file
    path: Optional[Path] = None
    how: str = ""


def _from_nm(t, sizes) -> Tuple[object, object]:
    px, py = float(sizes[0]), float(sizes[1])
    back = t.inverse()

    def forward(xy):
        x, y, _ = t.apply(xy[:, 0] * px, xy[:, 1] * py)
        return np.c_[x / px, y / py]

    def inverse(xy):
        x, y, _ = back.apply(xy[:, 0] * px, xy[:, 1] * py)
        return np.c_[x / px, y / py]
    return forward, inverse


def chip_map(locs, settings: RawOffsetSettings, sizes) -> ChipMap:
    """The map ``settings`` names, else the one the table was overlaid with."""
    from ..calibrate.transform import load_transform
    from ..transformation import Transformation
    if settings.registration:
        t = load_transform(settings.registration)
        return ChipMap(t.to_reference, t.to_secondary, t, Path(settings.registration),
                       f"map {Path(settings.registration).name}")
    record = overlay_record(locs)
    if record is None:
        raise ValueError("choose the channel map (_2ct.h5): the table was not overlaid with "
                         "Overlay channels, so it does not say which map belongs to it")
    t = Transformation.from_dict(record["transformation"])
    forward, inverse = _from_nm(t, sizes)
    path = (t.origin or {}).get("file")
    kept = None
    if path and Path(path).exists():
        kept = load_transform(path)
    return ChipMap(forward, inverse, kept, Path(path) if path else None,
                   f"the map Overlay channels used{f' ({Path(path).name})' if path else ''}")


def chip_positions(locs, sizes) -> np.ndarray:
    """Each localization where its half of the chip saw it, in chip pixels: the secondary half's
    moved back when the table was overlaid."""
    from ..transformation import Transformation
    from ..channel_overlay import secondary_rows
    xy = np.c_[np.asarray(locs["x_nm"], float) / sizes[0],
               np.asarray(locs["y_nm"], float) / sizes[1]]
    record = overlay_record(locs)
    if record is not None:
        rows = secondary_rows(locs)
        _, inverse = _from_nm(Transformation.from_dict(record["transformation"]), sizes)
        xy[rows] = inverse(xy[rows])
    return xy


def _rect(region) -> Tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in region[:4])
    return x, y, x + w, y + h


def channel_rects(locs, xy) -> List[Tuple[float, float, float, float]]:
    """Each half's rectangle on the chip (x0, y0, x1, y1), as the fit recorded it; else the box
    of its localizations."""
    regions = (locs.metadata.get("channels") or {}).get("regions")
    if regions and len(regions) >= 2:
        return [_rect(regions[0]), _rect(regions[1])]
    cam = np.asarray(locs["camera_channel"])
    out = []
    for c in (0, 1):
        p = xy[cam == c]
        out.append((p[:, 0].min() - 0.5, p[:, 1].min() - 0.5,
                    p[:, 0].max() + 0.5, p[:, 1].max() + 0.5))
    return out


def roi_corner(source, rects) -> Tuple[int, int]:
    """The chip pixel of the frames' first pixel: the camera ROI the file records, else the
    corner of the channels' rectangles when they span the frame, else 0, 0."""
    try:
        from ..io.tiff import resolve_camera
        roi = resolve_camera(source).camera_metadata().roi
        if roi is not None:
            return int(roi[0]), int(roi[1])
    except Exception as error:          # no metadata at all: a plain TIFF
        print(f"raw channel offset: no camera ROI in the file ({error})")
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    h, w = source.shape[-2:]
    if round(x1 - x0) == w and round(y1 - y0) == h:
        return int(round(x0)), int(round(y0))
    return 0, 0


# ------------------------------------------------------------------ the plugin
@register("Localize/Raw channel offset")
class RawChannelOffset(Plugin):
    """Measure how far a split camera's bead map is off in the sample, from the crosstalk in the
    raw frames."""

    description = ("How far the bead map between a split camera's halves is off in the sample: "
                   "the crosstalk of every molecule into the other half, summed from the raw "
                   "frames where the map puts it; per region, with a shift or affine "
                   "correction for Transform.")
    Settings = RawOffsetSettings
    version = "1"
    changes_locs = False
    # each localization's own frame is read, whatever the layer groups
    grouping = "ungrouped"

    def describe(self, ctx: Optional[Context], settings: RawOffsetSettings) -> FormState:
        state = FormState()
        locs = getattr(ctx, "locs", None) if ctx is not None else None
        if locs is None or not len(locs):
            return state
        if "camera_channel" not in locs:
            state.notice = Notice(NEEDS)
            return state
        if not settings.raw:
            found = find_raw(locs, table_path(ctx, locs))
            if found is not None:
                state.hints["raw"] = str(found)
            else:
                state.notice = Notice("the raw frames are not where the table says, nor near "
                                      "it: choose the movie under raw frames", kind="warning")
                return state
        if not settings.registration:
            record = overlay_record(locs)
            if record is None:
                state.notice = Notice("the table was not overlaid: choose the channel map "
                                      "(_2ct.h5)", kind="warning")
                return state
            path = ((record.get("transformation") or {}).get("origin") or {}).get("file")
            if path:
                state.hints["registration"] = str(path)
        name = settings.name.strip() or DEFAULT_NAME
        state.hints["name"] = DEFAULT_NAME
        session = getattr(ctx, "session", None)
        kept = getattr(session, "transformations", {}).get(name) if session else None
        if kept is not None and overlay_record(locs) is not None:
            from .transform import settings_for
            file = int(np.asarray(locs["filenumber"])[0]) if "filenumber" in locs else 0
            state.notice = Notice(f"{name}: {kept}", action="Apply with Transform",
                                  window="transform",
                                  settings=settings_for(name, target=0, file=file, channel=1),
                                  kind="info")
        return state

    def run(self, ctx: Context, settings: RawOffsetSettings) -> Result:
        from .. import raw_offset as ro
        from ..io.tiff import open_stack
        from ..locs import table_pixel_sizes
        from ..transformation import Transformation
        locs, _ = ctx.ungrouped()
        if locs is None or not len(locs):
            raise ValueError("no localizations")
        if "camera_channel" not in locs:
            raise ValueError(NEEDS)
        if "filenumber" in locs and len(np.unique(np.asarray(locs["filenumber"]))) > 1:
            raise ValueError("the table holds several files: show one file in the layer, "
                             "since each was fitted from its own movie")
        raw = Path(settings.raw) if settings.raw else find_raw(locs, table_path(ctx, locs))
        if raw is None or not raw.exists():
            raise FileNotFoundError("the raw frames are not where the table says, nor near it: "
                                    "choose the movie under raw frames")
        sizes = table_pixel_sizes(locs)
        cmap = chip_map(locs, settings, sizes)
        xy = chip_positions(locs, sizes)
        rects = channel_rects(locs, xy)
        cam = np.asarray(locs["camera_channel"]).astype(int)
        frame = np.asarray(locs["frame"]).astype(int)
        photons = np.asarray(locs["photons"], float) if "photons" in locs else np.full(len(locs), np.inf)
        good = photons >= settings.min_photons
        if "sigma_nm" in locs:
            sigma = np.asarray(locs["sigma_nm"], float)
            for c in (0, 1):
                m = cam == c
                if m.any():
                    s = np.nanmedian(sigma[m])
                    good[m] &= (sigma[m] > 0.6 * s) & (sigma[m] < 1.6 * s)
            psf_px = float(np.nanmedian(sigma[good])) / float(np.mean(sizes)) if good.any() else 1.2
        else:
            psf_px = 1.2
        first_fit, last_fit = int(frame.min()), int(frame.max()) + 1
        start = (settings.start if settings.start is not None
                 else max(first_fit, (first_fit + last_fit) // 2 - settings.frames // 2))
        stop = min(start + settings.frames, last_fit)
        if stop - start < 4 * settings.lag:
            raise ValueError(f"frames {start} to {stop} are too few for a control "
                             f"{settings.lag} frames later")

        R = int(settings.radius)
        main_box = rects[0]
        directions, places = [], []
        for src, name in zip((0, 1), DIRECTIONS):
            rows = good & (cam == src) & (frame >= start) & (frame < stop)
            source_xy = xy[rows]
            target = cmap.to_secondary(source_xy) if src == 0 else cmap.to_reference(source_xy)
            place = source_xy if src == 0 else target
            x0, y0, x1, y1 = rects[1 - src]
            sx0, sy0, sx1, sy1 = rects[src]
            inside = ((target[:, 0] >= x0 + R + 1) & (target[:, 0] <= x1 - R - 2)
                      & (target[:, 1] >= y0 + R + 1) & (target[:, 1] <= y1 - R - 2)
                      & (source_xy[:, 0] >= sx0 + R + 1) & (source_xy[:, 0] <= sx1 - R - 2)
                      & (source_xy[:, 1] >= sy0 + R + 1) & (source_xy[:, 1] <= sy1 - R - 2))
            other = cam == 1 - src
            keep = inside & ro.exclude_seen(target, frame[rows], xy[other], frame[other],
                                            settings.lag, R + ro.EXCLUDE_MARGIN_PX)
            d = ro.Direction(name, source_xy[keep], target[keep], place[keep], frame[rows][keep],
                             ro.regions_of(place[keep], main_box, settings.grid))
            directions.append(d)
            ctx.report(f"{name}: {len(d.frame):,} molecules")
        source = open_stack(raw)
        corner = roi_corner(source, rects)
        ctx.report(f"reading frames {start} to {stop} of {raw.name}")
        blocks = source.frames(chunk=200, start=start, stop=min(stop + settings.lag,
                                                                 source.n_frames))
        sums = ro.sum_patches(blocks, directions, corner, R, settings.lag, start, stop,
                              report=ctx.report, stop_requested=lambda: ctx.stopped)

        # per direction, over the field and per region, as main-half residuals in px
        overall, per_region = [], []
        n_regions = settings.grid ** 2
        for d, s in zip(directions, sums):
            m = ro.measure(s, None, R, psf_px)
            overall.append(m)
            rows = []
            for k in range(n_regions):
                here = d.region == k
                rows.append((ro.measure(s, k, R, psf_px),
                             d.place[here].mean(axis=0) if here.any() else None))
            per_region.append(rows)

        def main_residual(m: "ro.Measured", place, name):
            r = np.array([[m.x, m.y]])
            e = np.array([m.x_err, m.y_err])
            to = ro.to_main(r, place, cmap.to_reference, cmap.to_secondary, name)[0]
            ex = ro.to_main(r + [e[0], 0], place, cmap.to_reference, cmap.to_secondary, name)[0]
            ey = ro.to_main(r + [0, e[1]], place, cmap.to_reference, cmap.to_secondary, name)[0]
            err = np.sqrt((ex - to) ** 2 + (ey - to) ** 2)
            return to, err

        scale = np.array(sizes, float)
        lines = [f"{raw.name}, frames {start} to {stop}; {cmap.how}"]
        centre = np.array([[(main_box[0] + main_box[2]) / 2, (main_box[1] + main_box[3]) / 2]])
        summary = {}
        for d, m in zip(directions, overall):
            if not m.measured:
                lines.append(f"{d.name}: {m.molecules:,} molecules, no measurable crosstalk "
                             f"({100 * m.fraction:.1f} +- {100 * m.fraction_err:.1f} % of "
                             "their light)")
                summary[d.name] = None
                continue
            r, e = main_residual(m, centre, d.name)
            summary[d.name] = {"residual_nm": (r * scale).tolist(), "error_nm": (e * scale).tolist(),
                               "fraction": m.fraction, "fraction_err": m.fraction_err,
                               "molecules": m.molecules, "sigma_nm": m.sigma * float(scale.mean())}
            lines.append(f"{d.name}: {m.molecules:,} molecules, {100 * m.fraction:.1f} +- "
                         f"{100 * m.fraction_err:.1f} % of their light leaks; the map is off by "
                         f"({r[0] * scale[0]:+.1f}, {r[1] * scale[1]:+.1f}) +- "
                         f"({e[0] * scale[0]:.1f}, {e[1] * scale[1]:.1f}) nm "
                         f"(width {m.sigma * scale.mean():.0f} nm)")

        # the regions: each direction measured there, combined by inverse variance
        table = []
        for k in range(n_regions):
            values, weights, place = [], [], None
            for d, rows in zip(directions, per_region):
                m, p = rows[k]
                if p is None or not m.measured or not np.all(np.isfinite([m.x_err, m.y_err])):
                    continue
                r, e = main_residual(m, p[None, :], d.name)
                values.append(r)
                weights.append(1.0 / np.maximum(e, 1e-6) ** 2)
                place = p if place is None else place
            if not values:
                continue
            w = np.array(weights)
            r = (w * np.array(values)).sum(axis=0) / w.sum(axis=0)
            table.append((k, place, r, 1.0 / np.sqrt(w.sum(axis=0))))
        result_data = {"directions": summary, "overall": overall, "regions": table,
                       "frames": (start, stop), "raw": str(raw)}
        transformations, plots = {}, {}
        spots = Plot(_spots_plot(directions, overall, scale), panels=len(directions),
                     size=(4.2 * len(directions), 4.0))
        if not table:
            lines.append("no region has measurable crosstalk: nothing to correct")
            return Result(text="\n".join(lines), plot=spots, settings=settings,
                          data=result_data)

        places = np.array([t[1] for t in table])
        res = np.array([t[2] for t in table])
        err = np.array([t[3] for t in table])
        matrix, after = ro.fit_correction(places, res, err, settings.model)
        name = settings.name.strip() or DEFAULT_NAME
        t = Transformation("affine", ro.correction_nm(matrix, sizes),
                           description=f"{settings.model} from the raw frames' crosstalk",
                           origin={"raw channel offset": {"raw": str(raw),
                                                          "frames": [start, stop]}})
        transformations[name] = t
        shift_nm = matrix[:2, 2] * scale if settings.model == "shift" else None
        if shift_nm is not None:
            lines.append(f"correction: a shift of ({shift_nm[0]:+.1f}, {shift_nm[1]:+.1f}) nm "
                         f"from {len(table)} regions")
        else:
            # about the field's centre: the translation about the chip's corner says little
            mid = np.array([(main_box[0] + main_box[2]) / 2, (main_box[1] + main_box[3]) / 2])
            at_mid = (matrix[:2, :2] @ mid + matrix[:2, 2] - mid) * scale
            a = matrix[:2, :2]
            turn = np.degrees(np.arctan2(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1]))
            lines.append(f"correction: an affine map from {len(table)} regions -- "
                         f"({at_mid[0]:+.1f}, {at_mid[1]:+.1f}) nm at the field's centre, "
                         f"scale {1e6 * (a[0, 0] - 1):+.0f} ppm in x and "
                         f"{1e6 * (a[1, 1] - 1):+.0f} ppm in y, turned {turn * 3600:+.0f} arcsec")
        rms = np.sqrt(np.mean(np.sum((after * scale) ** 2, axis=1)))
        lines.append(f"the regions about the correction: {rms:.1f} nm rms, their errors "
                     f"{np.sqrt(np.mean(np.sum((err * scale) ** 2, axis=1))):.1f} nm rms")
        lines.append(f"kept as {name!r}: Transform moves the overlaid channel 1 by it")
        plots["field"] = Plot(_field_plot(places, res, err, matrix, main_box, scale,
                                          settings.grid), size=(5.5, 5.0))
        if settings.save:
            lines.append(save_corrected(cmap, matrix, settings, start, stop, raw))
        result_data.update(matrix_px=matrix, transformation=t)
        return Result(text="\n".join(lines), plot=spots, plots=plots, settings=settings,
                      data=result_data, transformations=transformations)


def save_corrected(cmap: ChipMap, matrix: np.ndarray, settings: RawOffsetSettings, start: int,
                   stop: int, raw: Path) -> str:
    """Write the map with the correction folded in beside the map, and say where."""
    from ..calibrate.dual import ChannelTransform
    from ..calibrate.transform import save_channel_transform
    t = cmap.channel_transform
    if t is None or cmap.path is None:
        return ("not saved: the map is not in a file here -- choose it under channel map to "
                "save a corrected copy")
    if t.polynomial is not None:
        return ("not saved: the map is a polynomial, which a correction cannot be folded into; "
                "apply the kept transformation with Transform")
    h = np.asarray(matrix, float) @ np.asarray(t.transformation, float)
    parameters = dict(t.parameters)
    parameters["raw_channel_offset"] = {"from": str(cmap.path), "raw": str(raw),
                                        "frames": [int(start), int(stop)],
                                        "model": settings.model,
                                        "correction_px": np.asarray(matrix).tolist()}
    stem = cmap.path.name
    for suffix in ("_2ct.h5", ".h5", ".hdf5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    path = cmap.path.with_name(f"{stem}_raw_2ct.h5")
    save_channel_transform(path, ChannelTransform(h, t.geometry, parameters), overwrite=True)
    return f"corrected map saved as {path}"


# ------------------------------------------------------------------ figures
def _spots_plot(directions, overall, scale):
    """Each direction's summed crosstalk, with the fitted spot."""
    def draw(figure):
        axes = np.atleast_1d(figure.subplots(1, len(directions)))
        for ax, d, m in zip(axes, directions, overall):
            s = m.summed
            if s is None or s.molecules == 0:
                ax.set_axis_off()
                continue
            img = s.image()
            r = s.radius + 0.5
            ax.imshow(img, extent=[-r, r, r, -r], cmap="magma")
            ax.plot(0, 0, "+", color="w", ms=10, mew=1)
            if m.measured:
                t = np.linspace(0, 2 * np.pi, 100)
                for k in (1, 2):
                    ax.plot(m.x + k * m.sigma * np.cos(t), m.y + k * m.sigma * np.sin(t),
                            color="c", lw=0.8)
                ax.plot(m.x, m.y, "x", color="c", ms=8)
            state = (f"{100 * m.fraction:.1f} % leaks" if m.measured
                     else "no measurable crosstalk")
            ax.set_title(f"{d.name}\n{m.molecules:,} molecules, {state}", fontsize=9)
            ax.set_xlabel("x from the map's prediction (px)")
            ax.set_ylabel("y (px)")
    return draw


def _field_plot(places, res, err, matrix, box, scale, grid):
    """Each region's offset (arrow, nm, much enlarged) and the correction's (grey)."""
    def draw(ax):
        from ..calibrate.dual import map_points
        model = map_points(matrix, places) - places
        x0, y0, x1, y1 = box
        cell = min(x1 - x0, y1 - y0) / max(grid, 1)
        biggest = max(float(np.max(np.hypot(*(res * scale).T))), 1.0)
        k = 0.4 * cell / biggest                             # px drawn per nm
        for p, r, m, e in zip(places, res * scale, model * scale, err * scale):
            ax.annotate("", xy=(p[0] + k * r[0], p[1] + k * r[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", color="C0", lw=1.5))
            ax.annotate("", xy=(p[0] + k * m[0], p[1] + k * m[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", color="0.6", lw=1, ls="--"))
            ax.plot(p[0], p[1], ".", color="k", ms=3)
            ax.text(p[0], p[1] - 0.08 * cell, f"({r[0]:+.0f}, {r[1]:+.0f}) nm\n"
                    f"+-({e[0]:.0f}, {e[1]:.0f})", ha="center", va="bottom", fontsize=7)
        # a scale arrow of a round number of nm in the lower left corner
        bar = float(10 ** np.floor(np.log10(biggest)))
        sx, sy = x0 + 0.05 * (x1 - x0), y1 - 0.04 * (y1 - y0)
        ax.annotate("", xy=(sx + k * bar, sy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color="C0", lw=1.5))
        ax.text(sx, sy - 0.02 * (y1 - y0), f"{bar:g} nm", fontsize=7, va="bottom")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect("equal")
        ax.set_xlabel("x on the chip, main half (px)")
        ax.set_ylabel("y (px)")
        ax.set_title("the map's offset per region (blue) and the correction (grey)",
                     fontsize=9)
    return draw
