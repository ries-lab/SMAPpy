"""Image quality: the error map (SQUIRREL) and the local FRC map, per layer.

Two maps of where the super-resolved picture can be trusted, made the same
way and handed back the same way:

* the **error map** (`smappy.squirrel`): the localizations, blurred and
  scaled to look like a widefield image of the same field, against that
  image -- where structure is missing or invented.  NanoJ-SQUIRREL's (Culley
  et al. 2018), re-implemented;
* the **local FRC map** (`frc.local_frc`): the FRC resolution of each tile of
  the field -- where the picture resolves what and where it does not.  It
  was a part of Localization Precision until its version 7, which kept the
  whole field's FRC, a number beside NeNA and the CRLB, and handed the map
  here, beside the other map.

Each is measured once per visible localization layer -- two channels, or
two filter settings of one, are compared with the same reference in one run
-- and each is, on request, a column of the localizations (`error_map`,
`frc_local_nm`), an image layer, and a 32-bit TIFF beside the data file.
With two or more layers and both an RSP and an FRC, *QnR* ranks them as
eSRRF does (Laine et al. 2023, `squirrel.qnr`): the harmonic mean of the
fidelity and the resolution's rank in the set.

Reads the localizations, not merged blinks (``grouping = "ungrouped"``), as
Localization Precision does: the FRC splits the acquisition in time, which a
blink spanning both halves would defeat, and the error map counts what was
localized, which is what the reference saw.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import Context, FormState, Notice, Plot, Plugin, Result, param, register
from ..frc import DEFAULT_BLOCKS, LOCAL_TILE_NM, MIN_LOCS as MIN_FRC_LOCS, FRCMap
from ..images import find_image, shown_images
from ..locs import Localizations

ERROR_COLUMN = "error_map"
SIGNED_COLUMN = "error_map_signed"
LOCAL_COLUMN = "frc_local_nm"
LOCAL_ERROR_COLUMN = "frc_local_err_nm"


# ------------------------------------------------------------------ settings

@dataclass
class ErrorMapSettings:
    reference: str = param("", label="reference",
                           help="the widefield image of the same field; auto: "
                                "the one an image layer shows, else the "
                                "first image")
    sigma_nm: Optional[float] = param(None, label="blur", unit="nm", min=0.0,
                                      help="the Gaussian that takes the "
                                           "localizations to the widefield "
                                           "resolution; auto: fitted")
    register: bool = param(True, label="find shift",
                           help="the offset between the localizations and "
                                "the reference, found and taken out")
    signed: bool = param(False, label="signed",
                         help="also the map with its sign: positive where the "
                              "localizations show more than the reference")
    max_shift_nm: float = param(1000.0, label="max shift", unit="nm", min=1.0,
                                advanced=True,
                                help="the farthest the shift is looked for")
    magnification: int = param(4, label="finer by", min=1, max=10, advanced=True,
                               help="the localizations are counted on pixels "
                                    "this much finer than the reference's")


@dataclass
class LocalFRCSettings:
    tile_um: float = param(LOCAL_TILE_NM / 1000, label="tile size", unit="µm",
                           min=0.2, max=50.0,
                           help="the side of the squares the FRC is measured in")
    min_locs: int = param(MIN_FRC_LOCS, label="per tile at least", min=100,
                          help="localizations a tile needs for a value; fewer "
                               "is NaN")
    smooth: bool = param(False, label="smooth",
                         help="the attribute interpolated between the tile "
                              "centres, not constant in a tile")
    blocks: int = param(DEFAULT_BLOCKS, label="FRC blocks", min=2, advanced=True,
                        help="the acquisition is cut into this many stretches "
                             "of frames before the two halves are dealt")
    repeats: int = param(5, label="FRC splits", min=1, max=50, advanced=True,
                         help="the curves of this many random deals are "
                              "averaged; their spread is the error bar")


@dataclass
class ImageQualitySettings:
    error_map: bool = param(True, label="error map (SQUIRREL)",
                            help="the localizations against a widefield image "
                                 "of the same field: what is missing or added")
    squirrel: ErrorMapSettings = param(default_factory=ErrorMapSettings,
                                       label="error map")
    frc_map: bool = param(False, label="local FRC map",
                          help="the FRC resolution of each tile of the field: "
                               "where the picture is good and where not")
    frc_local: LocalFRCSettings = param(default_factory=LocalFRCSettings,
                                        label="local FRC map")
    qnr: bool = param(False, label="QnR of the layers",
                      help="two or more layers ranked by fidelity and "
                           "resolution together (eSRRF); needs the error map")
    write_column: bool = param(True, label="write to localizations",
                               help="each map's value on the localizations it "
                                    "was measured on, NaN elsewhere")
    image_layer: bool = param(True, label="add image layers",
                              help="each map as an image layer over the "
                                   "localizations")
    save_tiff: bool = param(False, label="save as TIFF",
                            help="each map as a 32-bit TIFF beside the "
                                 "localization file")


# ----------------------------------------------------------------- the layers

def measured_layers(ctx: Context) -> List[Tuple[str, Localizations, np.ndarray]]:
    """``(name, table, mask)`` of every visible localization layer -- the
    context's own table alone without a session."""
    session = ctx.session
    if session is None:
        return [(ctx.selection.name or "selection", ctx.locs, ctx.selection.mask)]
    out = []
    for i, layer in enumerate(session.layers):
        if layer.is_image or not layer.visible:
            continue
        locs, selection = ctx.table(i)
        out.append((layer.name, locs, np.asarray(selection.mask, bool)))
    if not out:
        raise ValueError("no visible layer of localizations to measure")
    return out


# -------------------------------------------------------- the maps, handed back

def with_columns(locs: Localizations, columns: Dict[str, np.ndarray]) -> Localizations:
    """A new table with ``columns`` added, each averaged per blink on a
    grouped layer, as a precision is."""
    from ..mathparse import remember

    out = Localizations({**locs.columns,
                         **{k: np.asarray(v, np.float32) for k, v in columns.items()}},
                        dict(locs.metadata))
    for name in columns:
        remember(out, name, "", grouped="mean")
    return out


def with_local_frc(locs: Localizations, mask, local: FRCMap,
                   smooth: bool = False) -> Localizations:
    """A new table with the local FRC on every measured localization: NaN
    outside ``mask`` and wherever the tile has no value."""
    value, error = local_frc_values(locs, mask, local, smooth)
    return with_columns(locs, {LOCAL_COLUMN: value, LOCAL_ERROR_COLUMN: error})


def local_frc_values(locs: Localizations, mask, local: FRCMap,
                     smooth: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    x = np.asarray(locs["x_nm"], dtype=float)[mask]
    y = np.asarray(locs["y_nm"], dtype=float)[mask]
    value = np.full(len(locs), np.nan, np.float32)
    error = np.full(len(locs), np.nan, np.float32)
    value[mask] = local.at(x, y, smooth=smooth)
    error[mask] = local.at(x, y, smooth=smooth, values=local.error)
    return value, error


def local_frc_image(local: FRCMap, name: str = "local FRC"):
    """The map as an image layer: one pixel per tile, placed on the table.

    A tile without a value is 0 rather than NaN, which an image layer's
    contrast would take for its maximum; read 0 as "not measured".
    """
    from ..images import ImageData

    data = np.nan_to_num(local.resolution, nan=0.0).astype(np.float32)
    return ImageData(data, pixelsize=float(local.tile_nm), x0=float(local.x0),
                     y0=float(local.y0), name=name)


def tiff_path(session, what: str, layer: str):
    """Where a map of ``layer`` is saved: beside the localization file,
    named after it.  None without a file."""
    from pathlib import Path
    where = getattr(session, "path", None)
    if not where:
        return None
    where = Path(where)
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{what}_{layer}").strip("_")
    return where.with_name(f"{where.stem}_{tag}.tif")


class _Columns:
    """The maps' values gathered over the layers into whole columns: each
    layer writes its own rows, and a row two layers show keeps the first
    one's value."""

    def __init__(self, n: int):
        self.n = n
        self.columns: Dict[str, np.ndarray] = {}
        self.overlap = False

    def put(self, name: str, mask: np.ndarray, values: np.ndarray) -> None:
        column = self.columns.setdefault(name, np.full(self.n, np.nan, np.float32))
        free = mask & np.isnan(column)
        self.overlap |= bool(np.any(mask & ~free & np.isfinite(values)))
        column[free] = values[free]


# ------------------------------------------------------------------ the plugin

@register("Analysis/Measure/Image quality")
class ImageQuality(Plugin):
    """Where the super-resolved picture can be trusted: an error map against
    a widefield image (SQUIRREL) and a local FRC map, for each layer."""

    description = ("Map where each layer's localizations disagree with a "
                   "widefield image of the same field (SQUIRREL's error map, "
                   "with RSE and RSP) and where the picture resolves what "
                   "(the local FRC); written to the localizations, as image "
                   "layers or as TIFFs, and ranked by QnR across layers.")
    Settings = ImageQualitySettings
    changes_locs = False
    grouping = "ungrouped"
    version = "1"

    def changes_locs_for(self, settings: ImageQualitySettings) -> bool:
        return bool(settings.write_column and (settings.error_map or settings.frc_map))

    def describe(self, ctx: Optional[Context], settings: ImageQualitySettings
                 ) -> FormState:
        state = FormState(shown={"squirrel": settings.error_map,
                                 "frc_local": settings.frc_map,
                                 "qnr": settings.error_map})
        session = getattr(ctx, "session", None)
        if session is None or not settings.error_map:
            return state
        images = shown_images(session)
        state.choices["squirrel.reference"] = [(im.name, im.name) for im in images]
        found = find_image(session, "") if not settings.squirrel.reference else None
        if found is not None:
            state.hints["squirrel.reference"] = found.name
        if not images:
            state.notice = Notice(
                "the error map needs a widefield image of the same field: open "
                "one (File > Open), or fit a movie, which keeps the average of "
                "its frames", kind="warning")
        return state

    def run(self, ctx: Context, settings: ImageQualitySettings) -> Result:
        from ..frc import frc_resolution, local_frc
        from ..squirrel import error_map, qnr, reference_plane

        if not (settings.error_map or settings.frc_map):
            raise ValueError("nothing to measure: tick the error map or the "
                             "local FRC map")
        layers = measured_layers(ctx)
        reference = None
        if settings.error_map:
            reference = find_image(ctx.session, settings.squirrel.reference)
            if reference is None:
                raise ValueError("the error map needs a widefield image: open "
                                 "one, or untick the error map")
            plane = reference_plane(reference)
            ctx.report(f"error maps against {reference.name}")

        maps: Dict[str, object] = {}
        frc_maps: Dict[str, FRCMap] = {}
        resolutions: Dict[str, float] = {}
        written = _Columns(len(ctx.locs))
        images, lines, saved = [], [], []
        for name, locs, mask in layers:
            shown = locs[mask]
            if not len(shown):
                ctx.warn(f"{name}: no localizations shown")
                continue
            same_rows = len(locs) == len(ctx.locs)
            x = np.asarray(shown["x_nm"], float)
            y = np.asarray(shown["y_nm"], float)
            if settings.error_map:
                s = settings.squirrel
                ctx.report(f"{name}: error map")
                found = error_map(plane, x, y, reference.x0, reference.y0,
                                  *reference.pixelsize_xy,
                                  magnification=s.magnification,
                                  sigma_nm=s.sigma_nm, register=s.register,
                                  max_shift_nm=s.max_shift_nm, report=ctx.report)
                maps[name] = found
                if found.message:
                    ctx.warn(f"{name}: {found.message}")
                if found.ok:
                    lines.append(_error_line(name, found))
                    if settings.write_column and same_rows:
                        all_x = np.asarray(locs["x_nm"], float)
                        all_y = np.asarray(locs["y_nm"], float)
                        written.put(ERROR_COLUMN, mask, found.at(all_x, all_y))
                        if s.signed:
                            written.put(SIGNED_COLUMN, mask,
                                        found.at(all_x, all_y, signed=True))
                    made = [found.image(f"error map of {name}")]
                    if s.signed:
                        made.append(found.image(f"signed error map of {name}",
                                                signed=True))
                    if settings.image_layer:
                        images.extend(made)
                    if settings.save_tiff:
                        saved += self._save(ctx, made, "error_map", name)
            if settings.frc_map or (settings.qnr and settings.error_map):
                if "frame" not in shown:
                    ctx.warn(f"{name}: no frame attribute, so no FRC")
                    continue
                frame = np.asarray(shown["frame"])
                f = settings.frc_local
                whole = frc_resolution(x, y, frame, n_blocks=f.blocks,
                                       repeats=f.repeats, report=ctx.report)
                resolutions[name] = whole.resolution if whole.ok else float("nan")
                if settings.frc_map:
                    ctx.report(f"{name}: local FRC")
                    local = local_frc(x, y, frame,
                                      pixelsize=whole.pixelsize if whole.ok else 0.0,
                                      tile_nm=f.tile_um * 1000, min_locs=f.min_locs,
                                      n_blocks=f.blocks, repeats=f.repeats,
                                      report=ctx.report)
                    frc_maps[name] = local
                    if local.message:
                        ctx.warn(f"{name}: local FRC: {local.message}")
                    if local.ok:
                        lines.append(_frc_line(name, local))
                        if settings.write_column and same_rows:
                            value, error = local_frc_values(locs, mask, local, f.smooth)
                            written.put(LOCAL_COLUMN, mask, value)
                            written.put(LOCAL_ERROR_COLUMN, mask, error)
                        made = [local_frc_image(local, f"local FRC of {name}")]
                        if settings.image_layer:
                            images.extend(made)
                        if settings.save_tiff:
                            saved += self._save(ctx, made, "local_frc", name)
            if settings.write_column and not same_rows:
                ctx.warn(f"{name}: its table is not the localizations', so "
                         "nothing was written to it")

        scores: Dict[str, float] = {}
        if settings.qnr and settings.error_map:
            named = [n for n in maps if maps[n].ok and n in resolutions]
            if len(named) < 2:
                ctx.warn("QnR ranks two or more layers against each other; "
                         f"{len(named)} had both an error map and an FRC")
            else:
                values = qnr([maps[n].rsp for n in named],
                             [resolutions[n] for n in named])
                scores = dict(zip(named, (float(v) for v in values)))
                best = max(scores, key=lambda n: np.nan_to_num(scores[n], nan=-1))
                lines.append("QnR: " + ", ".join(f"{n} {v:.2f}" for n, v in scores.items())
                             + f"; best: {best}")

        out_locs = None
        if written.columns:
            out_locs = with_columns(ctx.locs, written.columns)
            lines.append("written to the localizations: "
                         + ", ".join(written.columns))
            if written.overlap:
                ctx.warn("some localizations are in two layers; they keep the "
                         "first layer's value")
        if images:
            lines.append(f"{len(images)} image layer(s) added")
        if saved:
            lines.append("saved: " + ", ".join(str(p) for p in saved))
        if not lines:
            raise ValueError("no map could be measured: "
                             + "; ".join(m.message for m in maps.values() if m.message)
                             if maps else "no map could be measured")

        plots = {}
        for name in dict.fromkeys(list(maps) + list(frc_maps)):
            em, fm = maps.get(name), frc_maps.get(name)
            panels = (3 if em is not None and em.ok else 0) + (1 if fm is not None else 0)
            if panels:
                plots[name] = Plot(draw=_layer_figure(name, em, fm, settings),
                                   panels=panels, size=(3.2 * panels, 3.4))
        if scores:
            plots["QnR"] = Plot(draw=_qnr_figure(maps, resolutions, scores),
                                size=(5.5, 3.4))
        data = {"error_maps": maps, "frc_maps": frc_maps,
                "resolution": resolutions, "qnr": scores,
                "rse": {n: m.rse for n, m in maps.items()},
                "rsp": {n: m.rsp for n, m in maps.items()},
                "reference": None if reference is None else reference.name}
        return Result(text="\n".join(lines), data=data, plots=plots,
                      settings=settings, locs=out_locs, image_layers=images,
                      title="Image quality")

    @staticmethod
    def _save(ctx: Context, made, what: str, name: str) -> list:
        from ..images import save_tiff
        path = tiff_path(ctx.session, what, name)
        if path is None:
            ctx.warn("no TIFF saved: the localizations are in no file yet")
            return []
        out = []
        for image in made:
            signed = image.name.startswith("signed")
            target = path.with_name(path.stem.replace("error_map", "error_map_signed")
                                    + path.suffix) if signed else path
            out.append(save_tiff(image, target))
        return out

    def keep(self, result: Result):
        """The numbers of each map, without the maps: those are image layers
        or files when they were asked for."""
        data = result.data or {}
        return {"rse": {k: float(v) for k, v in (data.get("rse") or {}).items()},
                "rsp": {k: float(v) for k, v in (data.get("rsp") or {}).items()},
                "resolution": {k: float(v) for k, v
                               in (data.get("resolution") or {}).items()},
                "qnr": dict(data.get("qnr") or {}),
                "reference": data.get("reference")}


# ------------------------------------------------------------------- the text

def _error_line(name: str, found) -> str:
    sigma = (f"blur {found.sigma_nm:.0f} nm" + ("" if found.sigma_fitted else " (set)")
             + (" (at its bound)" if found.at_bound else ""))
    return (f"{name}: RSE {found.rse:.3g}, RSP {found.rsp:.3f}; {sigma}, "
            f"shift ({found.dx_nm:.0f}, {found.dy_nm:.0f}) nm, "
            f"scale {found.alpha:.3g}, offset {found.beta:.3g}; "
            f"{found.n_locs} localizations on {found.model.shape[1]} x "
            f"{found.model.shape[0]} pixels")


def _frc_line(name: str, local: FRCMap) -> str:
    values = local.resolution[np.isfinite(local.resolution)]
    ny, nx = local.resolution.shape
    return (f"{name}: local FRC {np.median(values):.1f} nm median over "
            f"{local.n_measured} of {nx}x{ny} tiles of {local.tile_nm / 1000:.3g} um "
            f"(10th to 90th percentile {np.percentile(values, 10):.1f} to "
            f"{np.percentile(values, 90):.1f} nm); {local.n_sparse} too sparse, "
            f"{local.n_uncrossed} never crossed 1/7")


# ---------------------------------------------------------------- the figures

def draw_error_map(axes, found, title: str = "") -> None:
    """The reference, the blurred and scaled localizations on the same
    scale, and the error map: three axes."""
    x0, y0, x1, y1 = found.bounds
    extent = (x0 / 1000, x1 / 1000, y1 / 1000, y0 / 1000)
    lo, hi = np.percentile(found.reference, [0.5, 99.5])
    for ax, picture, label in ((axes[0], found.reference, "reference"),
                               (axes[1], found.model, "localizations, blurred")):
        ax.imshow(picture, cmap="gray", vmin=lo, vmax=hi, extent=extent,
                  interpolation="nearest")
        ax.set_title(label, fontsize=7.5, color="0.25")
    error = found.error
    image = axes[2].imshow(error, cmap="inferno", extent=extent,
                           vmin=0, vmax=float(np.percentile(error, 99.5)) or None,
                           interpolation="nearest")
    axes[2].figure.colorbar(image, ax=axes[2], shrink=0.8, label="|error|")
    axes[2].set_title(title or f"error map: RSE {found.rse:.3g}, RSP {found.rsp:.3f}",
                      fontsize=7.5, color="0.25")
    for ax in axes:
        ax.set_xlabel("x (um)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (um)", fontsize=7)


def draw_frc_map(ax, local: FRCMap, name: str = "") -> None:
    """The resolution of each tile, with the tiles that have none in grey."""
    import matplotlib

    if local is None or not local.resolution.size:
        ax.set_axis_off()
        ax.set_title("no local FRC", fontsize=7.5, color="0.25")
        return
    ny, nx = local.resolution.shape
    extent = (local.x0 / 1000, (local.x0 + nx * local.tile_nm) / 1000,
              (local.y0 + ny * local.tile_nm) / 1000, local.y0 / 1000)
    # grey for a tile with localizations and no value, nothing for empty space
    cmap = matplotlib.colormaps["viridis"].with_extremes(
        bad="0.6" if np.any(local.n_locs > 0) else "white")
    shown = np.where(local.n_locs > 0, local.resolution, np.nan)
    image = ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, extent=extent,
                      interpolation="nearest")
    empty = np.where(local.n_locs > 0, np.nan, 1.0)
    ax.imshow(np.ma.masked_invalid(empty), cmap="Greys", vmin=0, vmax=10,
              extent=extent, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax, label="FRC resolution (nm)", shrink=0.8)
    ax.set_xlabel("x (um)", fontsize=7)
    ax.set_ylabel("y (um)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_aspect("equal")
    note = (f"{local.n_measured} tiles of {local.tile_nm / 1000:.3g} um"
            + (f", grey: {local.n_sparse + local.n_uncrossed} without a value"
               if local.n_sparse + local.n_uncrossed else ""))
    ax.set_title(f"local FRC{' of ' + name if name else ''}: {note}",
                 fontsize=7.5, color="0.25")


def _layer_figure(name, em, fm, settings):
    def draw(figure) -> None:
        n = (3 if em is not None and em.ok else 0) + (1 if fm is not None else 0)
        axes = np.atleast_1d(figure.subplots(1, n))
        i = 0
        if em is not None and em.ok:
            draw_error_map(axes[:3], em)
            i = 3
        if fm is not None:
            draw_frc_map(axes[i], fm)
    return draw


def _qnr_figure(maps, resolutions, scores):
    from ..squirrel import normalised_resolution

    names = list(scores)
    rsp = [maps[n].rsp for n in names]
    nfrc = normalised_resolution([resolutions[n] for n in names])
    q = [scores[n] for n in names]

    def draw(ax) -> None:
        at = np.arange(len(names))
        w = 0.27
        ax.bar(at - w, rsp, w, label="RSP (fidelity)", color="#1f77b4")
        ax.bar(at, nfrc, w, label="nFRC (resolution rank)", color="#2ca02c")
        ax.bar(at + w, q, w, label="QnR", color="#d62728")
        ax.set_xticks(at, names, fontsize=7, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6.5, frameon=False)
        ax.set_title("QnR: fidelity and resolution, ranked within these layers",
                     fontsize=7.5, color="0.25")
    return draw
