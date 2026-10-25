"""*Astigmatic z*: z from the widths of an elliptical Gaussian fit.

The thin plugin around `smappy.astigmatic_z`, whose docstring has the method
and its reasons.  It reads `sigma_x_nm` and `sigma_y_nm` -- what Gaussian 2D
writes with *elliptical* on -- and their errors where the fit wrote them,
and writes `z_nm` and `z_err_nm` onto every localization, whatever the
filter shows: z is a property of a localization, not of the view.

The calibration is a file from the PSF calibration window's *elliptical
Gaussian* method, or typed: one defocus curve per axis, the simulator's
model by default.  *z scale* is the refractive-index mismatch: beads on the
coverslip measure the focus as the objective moves, molecules in water above
it are compressed in z by about 0.7-0.8 with an oil objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..locs import Localizations
from . import Context, FormState, Notice, Plot, Plugin, Result, param, register

CALIBRATIONS = (("file", "from a file"), ("typed", "typed"))
CALIBRATION_FILTER = "Astigmatic z calibrations (*.h5)"
WIDTHS = ("sigma_x_nm", "sigma_y_nm")
WIDTH_ERRORS = ("sigma_x_err_nm", "sigma_y_err_nm")
#: chi^2 above this (with one degree of freedom, p = 0.003) counts as widths that match no z
NO_MATCH_CHI2 = 9.0
NEEDS_WIDTHS = ("needs the widths of an elliptical Gaussian fit, sigma_x_nm and sigma_y_nm: "
                "fit with Gaussian 2D and elliptical ticked")


@dataclass
class TypedCalibration:
    """One curve per axis, s(z) = s0 sqrt(1 + u^2 + A u^3 + B u^4), u = (z - focus) / depth."""
    sigma_x_nm: float = param(130.0, label="width x", unit="nm", min=1.0,
                              help="s0: the narrowest the spot gets in x")
    focus_x_nm: float = param(300.0, label="focus x", unit="nm",
                              help="the z where x is narrowest; z = 0 is the round spot")
    depth_x_nm: float = param(400.0, label="depth x", unit="nm", min=1.0,
                              help="how far from its focus the x width has grown by sqrt(2)")
    sigma_y_nm: float = param(130.0, label="width y", unit="nm", min=1.0,
                              help="s0: the narrowest the spot gets in y")
    focus_y_nm: float = param(-300.0, label="focus y", unit="nm",
                              help="the z where y is narrowest")
    depth_y_nm: float = param(400.0, label="depth y", unit="nm", min=1.0,
                              help="how far from its focus the y width has grown by sqrt(2)")
    a_x: float = param(0.0, label="A x", advanced=True, help="Huang's cubic term in x")
    b_x: float = param(0.0, label="B x", advanced=True, help="Huang's quartic term in x")
    a_y: float = param(0.0, label="A y", advanced=True, help="Huang's cubic term in y")
    b_y: float = param(0.0, label="B y", advanced=True, help="Huang's quartic term in y")
    z_range_nm: float = param(800.0, label="z range", unit="nm", min=10.0,
                              help="+- this: where the curves are trusted")

    def calibration(self):
        from ..astigmatic_z import AstigmaticCalibration, Defocus
        return AstigmaticCalibration(
            Defocus(self.sigma_x_nm, self.focus_x_nm, self.depth_x_nm, self.a_x, self.b_x),
            Defocus(self.sigma_y_nm, self.focus_y_nm, self.depth_y_nm, self.a_y, self.b_y),
            (-self.z_range_nm, self.z_range_nm), "typed")


@dataclass
class AstigmaticZSettings:
    calibration: str = param("file", label="calibration", choices=CALIBRATIONS,
                             help="a file from the PSF calibration window (elliptical Gaussian), "
                                  "or the curves typed")
    path: str = param("", label="file", kind="open_file", file_filter=CALIBRATION_FILTER,
                      help="an astigmatic z calibration (.h5)")
    typed: TypedCalibration = param(default_factory=TypedCalibration, label="curves")
    z_scale: float = param(1.0, label="z scale", min=0.01,
                           help="true z per apparent z: about 0.7-0.8 for an oil objective "
                                "imaging in water, 1 for none")


def calibration_of(settings: AstigmaticZSettings):
    """The calibration the settings name: the file's, or the typed one."""
    from ..astigmatic_z import AstigmaticCalibration
    if settings.calibration == "typed":
        return settings.typed.calibration()
    if not settings.path:
        raise ValueError("choose a calibration file -- the PSF calibration window makes one with the "
                         "method elliptical Gaussian -- or type the curves")
    return AstigmaticCalibration.load(settings.path)


def with_z(locs: Localizations, z_nm, z_err_nm) -> Localizations:
    """The table with `z_nm` and `z_err_nm`, replacing any it had."""
    columns = dict(locs.columns)
    columns["z_nm"] = np.asarray(z_nm, dtype=np.float32)
    columns["z_err_nm"] = np.asarray(z_err_nm, dtype=np.float32)
    return Localizations(columns, dict(locs.metadata))


def draw_widths(ax, z_nm, sx_nm, sy_nm, cal, z_scale: float = 1.0, truth=None, limit: int = 20000):
    """Each localization's widths at the z found, over the calibration's curves; a sample of
    ``limit`` points.  ``truth`` (true z) puts the widths at the true z instead, for a figure that
    shows the model fitting."""
    z = np.asarray(z_nm if truth is None else truth, dtype=float)
    sx, sy = np.asarray(sx_nm, dtype=float), np.asarray(sy_nm, dtype=float)
    keep = np.flatnonzero(np.isfinite(z) & np.isfinite(sx) & np.isfinite(sy))
    if len(keep) > limit:
        keep = np.random.default_rng(0).choice(keep, limit, replace=False)
    ax.scatter(z[keep], sx[keep], s=2, alpha=0.3, color="C0", lw=0, rasterized=True)
    ax.scatter(z[keep], sy[keep], s=2, alpha=0.3, color="C3", lw=0, rasterized=True)
    grid = np.linspace(*cal.z_range, 400)
    cx, cy = cal.widths(grid)
    ax.plot(grid * z_scale, cx, color="C0", lw=2, label="x")
    ax.plot(grid * z_scale, cy, color="C3", lw=2, label="y")
    ax.set_xlabel(("true " if truth is not None else "") + "z (nm)")
    ax.set_ylabel("width (nm)")
    top = np.nanpercentile(np.r_[sx[keep], sy[keep]], 99) if len(keep) else max(cx.max(), cy.max())
    ax.set_ylim(0, 1.1 * max(top, 1.0))
    ax.legend(loc="upper center", frameon=False)


def draw_z(ax, z_nm, z_err_nm, z_range_nm):
    """The z found, those with a usable error and those an edge or a mismatch threw out."""
    z = np.asarray(z_nm, dtype=float)
    err = np.asarray(z_err_nm, dtype=float)
    ok = np.isfinite(z)
    span = z_range_nm[1] - z_range_nm[0]
    good = ok & (err < 0.1 * span)
    bins = np.linspace(z_range_nm[0], z_range_nm[1], 81)
    ax.hist(z[good], bins, color="0.3", label="z error < 10 % of the range")
    ax.hist(z[ok & ~good], bins, color="C1", alpha=0.7, label="larger")
    ax.set_xlabel("z (nm)")
    ax.set_ylabel("localizations")
    ax.legend(frameon=False, fontsize="small")


@register("Analysis/After fit/Astigmatic z")
class AstigmaticZ(Plugin):
    """z and its error from the widths of an elliptical Gaussian fit, by a calibration of the widths in z."""

    Settings = AstigmaticZSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: AstigmaticZSettings) -> FormState:
        """The file or the typed curves, whichever is chosen; a line when the table has no widths."""
        state = FormState(shown={"path": settings.calibration == "file",
                                 "typed": settings.calibration == "typed"})
        locs = getattr(ctx, "locs", None) if ctx is not None else None
        if locs is not None and len(locs) and not all(c in locs for c in WIDTHS):
            state.notice = Notice("Astigmatic z " + NEEDS_WIDTHS)
        return state

    def run(self, ctx: Context, settings: AstigmaticZSettings) -> Result:
        from ..astigmatic_z import z_from_widths
        locs = ctx.locs
        missing = [c for c in WIDTHS if c not in locs]
        if missing:
            raise ValueError(f"Astigmatic z {NEEDS_WIDTHS}; the table has "
                             + ", ".join(sorted(locs.keys())))
        cal = calibration_of(settings)
        sx, sy = (np.asarray(locs[c], dtype=float) for c in WIDTHS)
        errors = [np.asarray(locs[c], dtype=float) if c in locs else None for c in WIDTH_ERRORS]
        weighed = all(e is not None for e in errors)
        ctx.report(f"z from the widths of {len(locs):,} localizations")
        found = z_from_widths(sx, sy, cal, *(errors if weighed else (None, None)))
        scale = float(settings.z_scale)
        z, z_err = found.z_nm * scale, found.z_err_nm * scale
        out = with_z(locs, z, z_err)

        n = len(locs)
        edge = int(found.edge.sum())
        mismatch = int((found.chi2 > NO_MATCH_CHI2).sum())
        span = (cal.z_range[0] * scale, cal.z_range[1] * scale)
        good = np.isfinite(z) & ~found.edge
        median_err = float(np.median(z_err[good])) if good.any() else float("nan")
        what = "the file's curves" if settings.calibration == "file" else "the typed curves"
        text = (f"z of {n:,} localizations from their widths, by {what} over "
                f"{span[0]:.0f} to {span[1]:.0f} nm; median error {median_err:.1f} nm; "
                f"{edge:,} on an edge of the range or without widths and {mismatch:,} whose widths "
                f"match no z (chi^2 > {NO_MATCH_CHI2:g}), all with a large z_err_nm")
        if not weighed:
            text += (f".  The table has no width errors: one for all, {found.width_err_nm:.1f} nm, "
                     "from the residuals")
        if scale != 1:
            text += f".  z scaled by {scale:g}"
        data = {"localizations": n, "on_edge": edge, "no_match": mismatch,
                "median_z_err_nm": median_err, "z_range_nm": list(span),
                "width_err_nm": found.width_err_nm}

        def draw(figure):
            left, right = figure.subplots(1, 2)
            draw_widths(left, z, sx, sy, cal, scale)
            draw_z(right, z, z_err, span)
        return Result(locs=out, text=text, data=data, settings=settings,
                      plot=Plot(draw, panels=2, size=(9, 3.6)))
