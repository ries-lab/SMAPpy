"""*Field correction*: the bias of a fit with one spline over a field-dependent PSF, subtracted.

The thin plugin around `smappy.field_correction`, whose docstring has the method and its reasons.  It
runs where the localizations were fitted with the centre spline of a field-dependent calibration --
DECODE's network, or Spline 3D with this step on in its *after the fit*, which then fits with the centre
spline -- and moves `x_nm`, `y_nm` and `z_nm` of every localization, whatever the filter shows.

The calibration is the one the table says it was fitted with (the fit's record, or the DECODE model's),
drawn again for the sample's depth where the fit's was; typed, it is drawn as calibrated.  The
correction is measured at the table's median photons and background unless they are typed.

Run once: the table records the correction (``metadata["field_correction"]``), and a second run on it
is refused -- it would subtract the bias twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..locs import Localizations
from . import Context, FormState, Notice, Plot, Plugin, Result, param, register

CALIBRATION_FILTER = "Field-dependent calibrations (*.h5)"
#: where a table records the correction that moved it
RECORD = "field_correction"


@dataclass
class FieldCorrectionSettings:
    calibration: str = param("", label="calibration", kind="open_file", file_filter=CALIBRATION_FILTER,
                             help="a field-dependent calibration; empty: the one the table was fitted with")
    photons: float = param(0.0, label="photons", min=0.0, advanced=True,
                           help="the brightness the bias is measured at; 0: the table's median")
    background: float = param(0.0, label="background", min=0.0, advanced=True,
                              help="photons per pixel the bias is measured at; 0: the table's median")
    fit_box: int = param(0, label="fit box", unit="px", min=0, advanced=True,
                         help="the box the spots are fitted in; 0: the fit's, else 13")


def fitted_with(metadata: dict) -> tuple:
    """(calibration file, the sample-depth record it was drawn with or None, fit box or None) as the table's
    record says: Spline 3D's ``calibration``, else the DECODE model's."""
    record = metadata.get("calibration") or {}
    if record.get("file") and record.get("file") != "None":
        return str(record["file"]), record.get("sample_depth"), record.get("fit_box_px")
    decode = metadata.get("decode") or {}
    if decode.get("calibration"):
        depth = (decode.get("model_metadata") or {}).get("calibration_sample_depth")
        return str(decode["calibration"]), depth, None
    return None, None, None


def calibration_for(settings: FieldCorrectionSettings, metadata: dict):
    """The spline the table was fitted with, its field included: loaded, drawn at the record's depth."""
    from ..io.calibration import load_spline_calibration
    path, depth, _ = fitted_with(metadata)
    if settings.calibration:
        path, depth = settings.calibration, (depth if path and Path(path) == Path(settings.calibration) else None)
    if not path:
        raise ValueError("the table does not say which calibration it was fitted with: choose it")
    if not Path(path).is_file():
        raise ValueError(f"the calibration {path} is not there: choose it")
    cal = load_spline_calibration(path)
    if cal.field_splines is None:
        raise ValueError(f"{Path(path).name} has no field dependence: there is nothing to correct")
    if depth and depth.get("focus_nm") is not None:
        from ..calibrate.depth import at_depth
        cal = at_depth(cal, float(depth["focus_nm"]), depth.get("n_med"))
    return cal


def can_correct(metadata: dict) -> bool:
    """Whether the table was fitted with a field-dependent calibration that is here (for the form)."""
    from ..field_correction import has_field
    path, _, _ = fitted_with(metadata or {})
    return has_field(path)


@register("Analysis/After fit/Field correction")
class CorrectField(Plugin):
    """Subtracts the bias of a fit with one spline over a field-dependent PSF, measured from the calibration's
    field at the table's brightness."""

    Settings = FieldCorrectionSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: FieldCorrectionSettings) -> FormState:
        state = FormState()
        locs = getattr(ctx, "locs", None) if ctx is not None else None
        if locs is None or not len(locs):
            return state
        path, depth, _ = fitted_with(locs.metadata)
        if path:
            state.hints["calibration"] = Path(path).name
        if RECORD in locs.metadata:
            state.notice = Notice("the table is corrected already")
        elif not settings.calibration and not can_correct(locs.metadata):
            state.notice = Notice("the table was not fitted with a field-dependent calibration that is here: "
                                  "choose one")
        return state

    def run(self, ctx: Context, settings: FieldCorrectionSettings) -> Result:
        from ..field_correction import corrected, kept
        from ..locs import table_pixel_sizes
        locs = ctx.locs
        if RECORD in locs.metadata:
            raise ValueError("the table is corrected for the field already: a second correction would "
                             "subtract the bias twice")
        missing = [c for c in ("x_nm", "y_nm", "z_nm") if c not in locs]
        if missing:
            raise ValueError(f"Field correction needs x_nm, y_nm and z_nm; the table has "
                             f"{', '.join(sorted(locs.keys()))}")
        cal = calibration_for(settings, locs.metadata)
        _, _, box = fitted_with(locs.metadata)
        roi = int(settings.fit_box or box or 13)
        photons = float(settings.photons) or _median(locs, "photons", 1000.0)
        background = float(settings.background) or _median(locs, "background", 10.0)
        ctx.report(f"measuring the field's bias at {photons:.0f} photons, background {background:.1f}")
        correction = kept(cal, photons, background, roi)
        pixel = table_pixel_sizes(locs)
        x, y, z, outside = corrected(locs["x_nm"], locs["y_nm"], locs["z_nm"], correction, pixel)
        columns = dict(locs.columns)
        columns["x_nm"], columns["y_nm"], columns["z_nm"] = (np.asarray(a, np.float32) for a in (x, y, z))
        metadata = dict(locs.metadata)
        record = {**correction.summary(), "calibration": str(cal.source),
                  "sample_depth": (cal.parameters or {}).get("sample_depth"),
                  "outside": int(outside.sum())}
        metadata[RECORD] = record
        out = Localizations(columns, metadata)

        n = len(locs)
        moved = np.sqrt(np.nanmean((z - np.asarray(locs["z_nm"], float)) ** 2)) if n else 0.0
        text = (f"{n:,} localizations corrected for the field of {Path(str(cal.source)).name}, measured at "
                f"{photons:.0f} photons and background {background:.1f}: z moved by {moved:.1f} nm rms; the "
                f"correction follows the bias to {correction.residual_nm:.1f} nm rms")
        if outside.any():
            ctx.warn(f"field correction: {int(outside.sum()):,} of {n:,} localizations "
                     f"({100 * outside.mean():.1f} %) are outside the field or z range it was measured on, "
                     f"and took the edge's values")
        if correction.failed:
            ctx.warn(f"field correction: the field is too strong for one spline and a correction (the bias "
                     f"left {correction.residual_nm:.0f} nm rms, {100 * correction.lost:.0f} % of the "
                     f"measuring spots lost): fit with the field (Spline 3D without this step)")
        data = {"localizations": n, "outside": int(outside.sum()), "z_moved_rms_nm": float(moved),
                **correction.summary()}
        return Result(locs=out, text=text, data=data, settings=settings,
                      plot=Plot(lambda ax: draw_correction(ax, correction), size=(5, 3.6)))


def _median(locs: Localizations, name: str, default: float) -> float:
    if name not in locs:
        return default
    v = np.asarray(locs[name], float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else default


def draw_correction(ax, correction) -> None:
    """dz against the fitted z at the centre, the middle of each edge and the corners of the field."""
    lo, hi = correction.z_range
    z = np.linspace(lo, hi, 200)
    (x0, x1), (y0, y1) = correction.x_range, correction.y_range
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    places = [("centre", xm, ym, "k")] + \
        [(f"edge {i + 1}", x, y, "C0") for i, (x, y) in enumerate(((x0, ym), (x1, ym), (xm, y0), (xm, y1)))] + \
        [(f"corner {i + 1}", x, y, "C3") for i, (x, y) in enumerate(((x0, y0), (x1, y0), (x0, y1), (x1, y1)))]
    shown = set()
    for name, x, y, colour in places:
        d, _ = correction(np.full(len(z), x), np.full(len(z), y), z)
        label = name.split()[0]
        ax.plot(z, d[:, 2], color=colour, lw=1.2, label=None if label in shown else label)
        shown.add(label)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="fitted z (nm)", ylabel="z bias subtracted (nm)")
    ax.legend(frameon=False, fontsize="small")
