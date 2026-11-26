"""*Pixel equalisation*: a network's pull toward the camera's pixels taken out of the picture, and
put back on request.

The thin plugin around `smappy.pixel_equalise` (`fit_smooth`), whose docstring has the method, the
studies behind it and what it does not do: it flattens the histogram of where localizations sit
within their camera pixel -- the grid a network such as DECODE draws into a rendered image -- but
does not make them more precise.

**Correct and revert.**  Each localization's move is kept as two columns, `x_pixel_shift_nm` and
`y_pixel_shift_nm`, so *revert* is ``x - shift`` whatever came after: a drift correction or any other
translation moves a localization and leaves its shift as it was.  The maps themselves go into the
table's metadata (``pixel_equalisation``) for the record.  The Run button says which a click will do
(`FormState.run_label`): *Correct* on a table without the columns, *Revert* on one with them; a
chain or a script names it in *action*, and a correction of a corrected table is refused rather
than stacked.

**Before drift correction.**  The place within a pixel is the camera's only for positions as they
were fitted; a drift-corrected table's are moved by up to hundreds of nm, so a correction is
refused once the table records a drift correction.  What was corrected first stays revertible after.

Each file, channel and half of a split chip (``camera_channel``) is equalised on its own maps:
files can come from different cameras, models or z ranges, and a split chip's halves are fitted
on pixels of their own.  The fitters' *after the fit* runs it first, for a network's fit only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..locs import Localizations
from . import Context, FormState, Notice, Plugin, Result, param, register

SHIFTS = ("x_pixel_shift_nm", "y_pixel_shift_nm")
METADATA = "pixel_equalisation"
ACTIONS = (("auto", "correct, or revert a corrected table"), ("correct", "correct"),
           ("revert", "revert"))
DRIFTED = ("the table has been drift-corrected, so its positions are no longer where the camera's "
           "pixels saw them: equalise before drift correction (undo the drift correction, or load the "
           "fit as saved)")


@dataclass
class PixelEqualisationSettings:
    action: str = param("auto", label="action", choices=ACTIONS, advanced=True,
                        help="auto: correct a table that is not, revert one that is")
    z_splines: int = param(6, label="z splines", min=1, max=20, advanced=True,
                           help="how finely the correction may change with z")
    sigma_splines: int = param(4, label="uncertainty splines", min=1, max=12, advanced=True,
                               help="how finely it may change with the localization's uncertainty")
    harmonics: int = param(10, label="harmonics", min=1, max=30, advanced=True,
                           help="the sharpest structure within a pixel it may follow")
    significance: float = param(3.0, label="significance", min=0.0, advanced=True,
                                help="how far above counting noise a structure must stand to be "
                                     "corrected, in standard deviations")


def is_corrected(locs) -> bool:
    return locs is not None and all(c in locs for c in SHIFTS)


def action_of(locs, settings: PixelEqualisationSettings) -> str:
    """What a run does: the settings' action, or on *auto* what the table calls for."""
    if settings.action != "auto":
        return settings.action
    return "revert" if is_corrected(locs) else "correct"


def _column(locs, name, fallback=None):
    if name in locs:
        return np.asarray(locs[name], np.float64)
    return fallback


def _groups(locs):
    """Row indices per file, channel and half of a split chip: each its own camera pixels, or its
    own network.  A two-channel network fits each molecule once, in channel 0's pixels, so its
    table is one grid (`channel` is then the dye, written after this by *assign colours*)."""
    keys = [np.asarray(locs[c]) for c in ("filenumber", "channel", "camera_channel") if c in locs]
    if not keys:
        return {(): np.arange(len(locs))}
    table = np.stack(keys, 1)
    uniq, inverse = np.unique(table, axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    return {tuple(int(v) for v in u): np.flatnonzero(inverse == i) for i, u in enumerate(uniq)}


def equalise_table(locs: Localizations, settings: PixelEqualisationSettings, report=None):
    """The table with each file and channel's places equalised, its shifts kept as columns; the
    maps, per group, as dicts; the places before and after (pixels), for the figure."""
    from ..locs import in_pixels, table_pixel_sizes
    from ..pixel_equalise import fit_smooth
    px = np.asarray(table_pixel_sizes(locs), float)
    x, y = in_pixels(locs, "x_pix", "y_pix")
    n = len(locs)
    z = _column(locs, "z_nm", np.zeros(n))
    xy_err = _column(locs, "xy_err_nm", np.ones(n))
    x_err = _column(locs, "x_err_nm", xy_err)
    y_err = _column(locs, "y_err_nm", xy_err)
    new_x, new_y = x.copy(), y.copy()
    maps = {}
    options = dict(z_splines=settings.z_splines, sigma_splines=settings.sigma_splines,
                   harmonics=settings.harmonics, significance=settings.significance)
    for key, rows in _groups(locs).items():
        if report is not None:
            report(f"equalising {len(rows):,} localizations" + (f" of file and channel {key}" if key else ""))
        eq = fit_smooth(x[rows], y[rows], z[rows], x_err[rows], y_err[rows], **options)
        new_x[rows], new_y[rows] = eq(x[rows], y[rows], z[rows], x_err[rows], y_err[rows])
        maps[",".join(map(str, key)) or "all"] = dict(eq.to_dict(), n=len(rows),
                                                     harmonics_x=eq.x.harmonics_kept,
                                                     harmonics_y=eq.y.harmonics_kept)
    shift_x = (new_x - x) * px[0]
    shift_y = (new_y - y) * px[1]
    columns = dict(locs.columns)
    columns["x_nm"] = (np.asarray(locs["x_nm"], np.float64) + shift_x).astype(np.asarray(locs["x_nm"]).dtype)
    columns["y_nm"] = (np.asarray(locs["y_nm"], np.float64) + shift_y).astype(np.asarray(locs["y_nm"]).dtype)
    for name, shift, size in (("x_pix", shift_x, px[0]), ("y_pix", shift_y, px[1])):
        if name in columns:
            columns[name] = np.asarray(columns[name], np.float64) + shift / size
    columns[SHIFTS[0]] = shift_x.astype(np.float32)
    columns[SHIFTS[1]] = shift_y.astype(np.float32)
    metadata = dict(locs.metadata)
    metadata[METADATA] = {"maps": maps, "columns": list(SHIFTS)}
    return Localizations(columns, metadata), maps, (x, y), (new_x, new_y)


def revert_table(locs: Localizations) -> Localizations:
    """The table with the kept shifts taken off and their columns and record removed."""
    from ..locs import table_pixel_sizes
    columns = dict(locs.columns)
    sx, sy = (np.asarray(locs[c], np.float64) for c in SHIFTS)
    for name, shift in (("x_nm", sx), ("y_nm", sy)):
        columns[name] = (np.asarray(columns[name], np.float64) - shift).astype(np.asarray(columns[name]).dtype)
    if "x_pix" in columns or "y_pix" in columns:
        px = table_pixel_sizes(locs)
        for name, shift, size in (("x_pix", sx, px[0]), ("y_pix", sy, px[1])):
            if name in columns:
                columns[name] = np.asarray(columns[name], np.float64) - shift / size
    for c in SHIFTS:
        columns.pop(c)
    metadata = dict(locs.metadata)
    metadata.pop(METADATA, None)
    return Localizations(columns, metadata)


def draw_places(ax, before, after, bins: int = 40):
    """The places within the pixel, x and y pooled, before and after: flat is no grid."""
    from ..pixel_equalise import place
    edges = np.linspace(0, 1, bins + 1)
    for values, label, style in ((before, "before", dict(color="0.6", lw=1.5)),
                                 (after, "after", dict(color="C0", lw=1.5))):
        u = place(values[np.isfinite(values)])
        h, _ = np.histogram(u, edges)
        ax.stairs(h / max(h.mean(), 1e-12), edges, label=label, **style)
    ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_xlabel("place within the pixel (0, 1: its edges)")
    ax.set_ylabel("density (1: uniform)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)


@register("Analysis/After fit/Pixel equalisation")
class PixelEqualisation(Plugin):
    """Flattens where localizations sit within their camera pixel, the grid a network draws into
    the image; keeps each move, so a second click reverts it."""

    Settings = PixelEqualisationSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Context | None, settings: PixelEqualisationSettings) -> FormState:
        """Correct or Revert on the button, and a line where a correction would be refused."""
        locs = getattr(ctx, "locs", None) if ctx is not None else None
        action = action_of(locs, settings)
        state = FormState(run_label=action.capitalize())
        if locs is None or not len(locs):
            return state
        if action == "correct" and is_corrected(locs):
            state.notice = Notice("The table is already equalised: revert it first, or set action "
                                  "to auto")
        elif action == "revert" and not is_corrected(locs):
            state.notice = Notice("The table has not been equalised: nothing to revert")
        elif action == "correct" and "drift_correction" in locs.metadata:
            state.notice = Notice("Pixel equalisation: " + DRIFTED)
        return state

    def run(self, ctx: Context, settings: PixelEqualisationSettings) -> Result:
        locs = ctx.locs
        action = action_of(locs, settings)
        if action == "revert":
            if not is_corrected(locs):
                raise ValueError("the table has not been equalised: it has no "
                                 + " or ".join(SHIFTS))
            sx, sy = (np.asarray(locs[c], np.float64) for c in SHIFTS)
            moved = float(np.sqrt(np.nanmean(sx ** 2 + sy ** 2)))
            out = revert_table(locs)
            return Result(locs=out, settings=settings,
                          text=f"reverted the pixel equalisation of {len(locs):,} localizations "
                               f"(they had moved {moved:.2f} nm RMS)",
                          data={"action": "revert", "localizations": len(locs), "rms_move_nm": moved})
        if is_corrected(locs):
            raise ValueError("the table is already equalised: revert it first (action revert, or auto)")
        if "drift_correction" in locs.metadata:
            raise ValueError(DRIFTED)
        if not len(locs):
            raise ValueError("the table is empty")
        from ..training.reference import pixel_modulation
        out, maps, before, after = equalise_table(locs, settings, ctx.report)
        mod_before = pixel_modulation(np.r_[before[0], before[1]])
        mod_after = pixel_modulation(np.r_[after[0], after[1]])
        sx, sy = (np.asarray(out[c], np.float64) for c in SHIFTS)
        moved = float(np.sqrt(np.nanmean(sx ** 2 + sy ** 2)))
        kept = sum(m["harmonics_x"] + m["harmonics_y"] for m in maps.values())
        text = (f"equalised {len(locs):,} localizations in {len(maps)} file(s) and channel(s): "
                f"pixel modulation {mod_before:.3f} -> {mod_after:.3f}, moved {moved:.2f} nm RMS. "
                "This removes the camera's grid from the image; it does not make the localizations "
                "more precise.  Run again to revert.")
        if not kept:
            text = (f"nothing to equalise in {len(locs):,} localizations: where they sit within their "
                    f"pixels (modulation {mod_before:.3f}) is within counting noise of uniform")
        data = {"action": "correct", "localizations": len(locs), "groups": len(maps),
                "pixel_modulation_before": mod_before, "pixel_modulation_after": mod_after,
                "rms_move_nm": moved, "harmonics_kept": kept}

        def draw(ax):
            draw_places(ax, np.r_[before[0], before[1]], np.r_[after[0], after[1]])
        return Result(locs=out, text=text, data=data, settings=settings, plot=draw)
