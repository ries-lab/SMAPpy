"""*4Pi z*: z from the interference phase, unwrapped by the envelope's z.

The thin plugin around `smappy.fourpi`'s `cavity_phase`, `unwrap` and
`refine_period`, whose comments have the method and its reasons.  It reads
what the 4Pi fit writes -- `phase_rad`, `phase_err_rad`, the envelope's
`z_env_nm` and `z_env_err_nm` -- and writes `z_nm` and `z_err_nm` from the
phase on every localization, with `fringe_probability` (that the fringe is
the right one) and `cavity_phase_rad` (the cavity's phase in its frame).  It
reads `z_env_nm` and never the `z_nm` it writes, so running it again with
other settings starts from the fit, not from its own answer.

The phase's slope in z, `phase_per_nm`, is the calibration's: the 4Pi fit
keeps it in the table's metadata (``metadata["fourpi"]``), or the model file
is named here.  Each file of the table has its own cavity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..locs import Localizations
from . import Context, FormState, Notice, Plot, Plugin, Result, param, register

NEEDS = ("phase_rad", "phase_err_rad")
SOURCES = (("fit", "as fitted"), ("file", "from a model file"))
CALIBRATION_FILTER = "4Pi models (*.h5)"


@dataclass
class FourPiZSettings:
    slope: str = param("fit", label="calibration", choices=SOURCES,
                       help="the phase's slope in z: kept with the table by the 4Pi fit, "
                            "or read from a 4Pi model file")
    path: str = param("", label="model", kind="open_file", file_filter=CALIBRATION_FILTER,
                      help="a 4Pi model (.h5): this program's or uiPSF v1's")
    window_frames: int = param(1000, label="time window", unit="frames", min=10,
                               help="the cavity's phase is measured over this many frames")
    smoothing: int = param(1, label="smoothing", unit="windows", min=1,
                           help="average each window with its neighbours, this many in all")
    refine_period: bool = param(False, label="refine period",
                                help="find the fringe period that the data agree with best")
    z_scale: float = param(1.0, label="z scale", min=0.01, advanced=True,
                           help="true z per apparent z: 1 when sample and immersion "
                                "index match")


def slope_of(locs: Localizations, settings: FourPiZSettings) -> float:
    """phase_per_nm, from the table's metadata or the named model file."""
    if settings.slope == "file":
        if not settings.path:
            raise ValueError("4Pi z needs a 4Pi model file, or the slope the fit kept with "
                             "the table (calibration: as fitted)")
        from ..fourpi import load_fourpi_calibration
        return float(load_fourpi_calibration(settings.path).phase_per_nm)
    kept = (locs.metadata or {}).get("fourpi") or {}
    if "phase_per_nm" not in kept:
        raise ValueError("this table does not say the phase's slope in z (it was not fitted "
                         "by the 4Pi fit): name the 4Pi model file instead")
    return float(kept["phase_per_nm"])


def envelope_of(locs: Localizations):
    """The envelope's z and error: `z_env_nm`, or `z_nm` for a table that
    has not been through this plugin yet."""
    name = "z_env_nm" if "z_env_nm" in locs else "z_nm"
    err = "z_env_err_nm" if "z_env_err_nm" in locs else "z_err_nm"
    if name not in locs or err not in locs:
        raise ValueError("4Pi z needs the envelope's z and its error (z_env_nm, z_env_err_nm, "
                         "or z_nm, z_err_nm); the table has " + ", ".join(sorted(locs.keys())))
    return np.asarray(locs[name], float), np.asarray(locs[err], float)


def draw_fringes(ax, phase, z_env_nm, phase0, phase_per_nm, limit: int = 50000):
    """The phase with the cavity's taken off against the envelope's z: the
    localizations lie on stripes of slope k, one per fringe, and the stripe
    each one is put on is the one its z_env is nearest."""
    rel = np.angle(np.exp(1j * (np.asarray(phase, float) - np.asarray(phase0, float))))
    z = np.asarray(z_env_nm, float)
    keep = np.flatnonzero(np.isfinite(rel) & np.isfinite(z))
    if len(keep) > limit:
        keep = np.random.default_rng(0).choice(keep, limit, replace=False)
    lo, hi = np.percentile(z[keep], [1, 99]) if len(keep) else (-500, 500)
    ax.hist2d(z[keep], rel[keep], bins=[120, 60], range=[[lo, hi], [-np.pi, np.pi]],
              cmap="Greys")
    grid = np.linspace(lo, hi, 400)
    line = np.angle(np.exp(1j * phase_per_nm * grid))
    line[np.abs(np.diff(line, prepend=line[0])) > np.pi] = np.nan
    ax.plot(grid, line, color="C3", lw=1)
    ax.set_xlabel("envelope z (nm)")
    ax.set_ylabel("phase - cavity phase (rad)")


def draw_cavity(ax, cavity, frame=None, truth=None):
    """The cavity's phase in each window, with its error; ``truth`` (per
    frame) drawn under it where it is known."""
    if truth is not None and frame is not None:
        order = np.argsort(frame)
        ax.plot(np.asarray(frame)[order], np.asarray(truth)[order], color="0.6", lw=3,
                label="true")
    ax.errorbar(cavity.centres, cavity.phase0, yerr=cavity.error, fmt="o", ms=3,
                color="C0", label="measured")
    ax.set_xlabel("frame")
    ax.set_ylabel("cavity phase (rad)")
    if truth is not None:
        ax.legend(frameon=False, fontsize="small")


@register("Analysis/After fit/4Pi z")
class FourPiZ(Plugin):
    """z from a 4Pi fit's interference phase: the cavity's phase followed in time, and each phase put on the fringe its envelope z makes most probable."""

    Settings = FourPiZSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: FourPiZSettings) -> FormState:
        state = FormState(shown={"path": settings.slope == "file"})
        locs = getattr(ctx, "locs", None) if ctx is not None else None
        if locs is not None and len(locs) and not all(c in locs for c in NEEDS):
            state.notice = Notice("4Pi z needs the interference phase of a 4Pi fit, "
                                  "phase_rad and phase_err_rad")
        return state

    def run(self, ctx: Context, settings: FourPiZSettings) -> Result:
        from ..fourpi import cavity_phase, refine_period, unwrap
        locs = ctx.locs
        missing = [c for c in NEEDS if c not in locs]
        if missing:
            raise ValueError("4Pi z needs the interference phase of a 4Pi fit (phase_rad, "
                             "phase_err_rad); the table has " + ", ".join(sorted(locs.keys())))
        k = slope_of(locs, settings)
        z_env, z_env_err = envelope_of(locs)
        phase = np.asarray(locs["phase_rad"], float)
        phase_err = np.asarray(locs["phase_err_rad"], float)
        frame = np.asarray(locs["frame"], float)
        files = (np.asarray(locs["filenumber"]) if "filenumber" in locs
                 else np.zeros(len(locs), int))

        if settings.refine_period:
            ctx.report("refining the fringe period")
            k_found = refine_period(phase, phase_err, z_env, z_env_err, frame, k,
                                    settings.window_frames)
        else:
            k_found = k
        phase0 = np.full(len(locs), np.nan)
        cavities = {}
        for f in np.unique(files):
            rows = files == f
            ctx.report(f"the cavity's phase in file {f}")
            cav = cavity_phase(phase[rows], phase_err[rows], z_env[rows], z_env_err[rows],
                               frame[rows], k_found, settings.window_frames,
                               settings.smoothing)
            phase0[rows] = cav.at
            cavities[int(f)] = cav
        found = unwrap(phase, phase_err, z_env, z_env_err, phase0, k_found)
        scale = float(settings.z_scale)

        columns = dict(locs.columns)
        columns["z_env_nm"] = z_env.astype(np.float32)
        columns["z_env_err_nm"] = z_env_err.astype(np.float32)
        columns["z_nm"] = (found.z_nm * scale).astype(np.float32)
        columns["z_err_nm"] = (found.z_err_nm * scale).astype(np.float32)
        columns["fringe_probability"] = found.probability.astype(np.float32)
        columns["cavity_phase_rad"] = np.angle(np.exp(1j * phase0)).astype(np.float32)
        metadata = dict(locs.metadata)
        metadata["fourpi"] = {**(metadata.get("fourpi") or {}), "phase_per_nm": k,
                              "phase_per_nm_used": k_found}
        out = Localizations(columns, metadata)

        ok = np.isfinite(found.probability)
        ghosts = float(np.mean(1 - found.probability[ok])) if ok.any() else float("nan")
        unsure = int(np.sum(found.probability[ok] < 0.9))
        period = 2 * np.pi / abs(k_found)
        text = (f"z from the phase of {int(ok.sum()):,} localizations, fringe period "
                f"{period:.1f} nm" + (f" (refined from {2 * np.pi / abs(k):.1f})"
                                       if k_found != k else "")
                + f"; median z error {np.nanmedian(found.z_err_nm) * scale:.1f} nm; "
                f"expected wrong fringes {100 * ghosts:.2f} %, {unsure:,} localizations "
                f"below 90 % sure; the envelope's z scatters {found.envelope_scale:.2f}x "
                "its error")
        data = {"localizations": int(ok.sum()), "period_nm": period,
                "expected_wrong_fringe": ghosts, "below_90_percent": unsure,
                "envelope_scale": found.envelope_scale,
                "cavity": {f: {"frame": c.centres.tolist(), "phase_rad": c.phase0.tolist(),
                               "error_rad": c.error.tolist()} for f, c in cavities.items()}}
        first = cavities[min(cavities)]

        def draw(figure):
            left, right = figure.subplots(1, 2)
            draw_fringes(left, phase, z_env, phase0, k_found)
            draw_cavity(right, first)
        return Result(locs=out, text=text, data=data, settings=settings,
                      plot=Plot(draw, panels=2, size=(9, 3.6)))
