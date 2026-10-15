"""Tracking analysis: how the tracks move -- the mean squared displacement
(MSD) and the jump distances, per track and over all.

The algorithms are `smappy.tracking.analysis` (its docstring has the
decisions); this is the wrapper.  SMAP's ``AnalyzeSPT`` is where it comes
from, rewritten rather than ported (`wave4_research.md`: partly broken):

* SMAP fitted the MSD over lags 2 to 4, D = slope / 4, with the frame time
  hard-coded at 30 ms.  Here the lags are the optimum for each track's
  length and localization error (Michalet 2010), the offset of the
  localization error is the precisions' (or fitted), and the frame time is
  the file's (step 18);
* SMAP fitted a lognormal to the histogram of D; here the jump distances
  are fitted with one to three diffusive states by maximum likelihood,
  which gives the states' D and fractions without a histogram's bins;
* the *track id* is any integer column -- `track_id` from Link tracks,
  MINFLUX's `tid` (on its own clock, `time_s`), a cluster's id -- and on
  *auto*, the default, the first of those the table has
  (`tracking.columns`).

``per_region``: in an ROI, or per track as a block over `track_id`, the
values go to the region table.  Per-track D and alpha can be written onto
the localizations (*write per track*), so the tracks can be coloured or
filtered by them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, Stopped, param, register
from ..locs import Localizations


def _id_choices():
    """Without a table: *auto* and the names it looks for."""
    from ..tracking.columns import AUTO, TRACK_COLUMNS
    return [(AUTO, "auto")] + [(c, c) for c in TRACK_COLUMNS]


@dataclass
class MSDFitSettings:
    offset: str = param("precisions", label="offset",
                        choices=(("precisions", "from the precisions"),
                                 ("fitted", "fitted")),
                        help="the localization error's offset: from xy_err_nm, "
                             "or fitted with the slope")
    lags: Optional[int] = param(None, label="lags", min=1,
                                help="auto: per track the number that gives the "
                                     "most precise D (Michalet 2010)")
    min_length: int = param(5, label="minimum length", min=2,
                            help="localizations a track needs for a D of its own")
    write_columns: bool = param(False, label="write per track",
                                help="track_D_um2_per_s and track_alpha onto the "
                                     "localizations")


@dataclass
class JumpFitSettings:
    lag: int = param(1, label="lag", unit="frames", min=1,
                     help="the jumps between localizations this many frames apart")
    states: int = param(2, label="states", min=1, max=3,
                        help="diffusive states fitted, 1 to 3")


@dataclass
class TrackingAnalysisSettings:
    msd: bool = param(True, label="MSD", help="the mean squared displacement over lag")
    jumps: bool = param(True, label="jump distances",
                        help="the distribution of jumps, as diffusive states")
    id_column: str = param("", label="track id", choices=_id_choices,
                           help="the column naming the tracks; auto: track_id, "
                                "else tid, else track_id_true")
    use_z: bool = param(True, label="in 3D", help="x, y and z when the table has z_nm")
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0.001, advanced=True,
                                           help="auto: the file's, else its exposure")
    msd_fit: MSDFitSettings = param(default_factory=MSDFitSettings, label="MSD")
    jump_fit: JumpFitSettings = param(default_factory=JumpFitSettings,
                                      label="jump distances")


@register("Analysis/Tracking/Tracking analysis")
class TrackingAnalysis(Plugin):
    """How the tracks move: the mean squared displacement, with D and alpha
    per track and over all, and the jump distances as diffusive states."""

    Settings = TrackingAnalysisSettings
    changes_locs = False
    per_region = True
    # tracks are of localizations, whatever the layer shows
    grouping = "ungrouped"
    version = "1"

    def changes_locs_for(self, settings: TrackingAnalysisSettings) -> bool:
        return bool(settings.msd and settings.msd_fit.write_columns)

    def describe(self, ctx: Optional[Context], settings: TrackingAnalysisSettings
                 ) -> FormState:
        state = FormState(shown={"msd_fit": settings.msd, "jump_fit": settings.jumps})
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            locs = ctx.locs
            from ..tracking.columns import track_column_choices
            state.choices["id_column"] = track_column_choices(locs)
            state.active["use_z"] = "z_nm" in locs
            if "time_s" in locs:
                state.hints["frame_time_ms"] = "time_s"
                state.active["frame_time_ms"] = False
            else:
                from ..metadata import table_frame_time
                ft = table_frame_time(locs)
                if ft:
                    state.hints["frame_time_ms"] = f"{ft:g}"
        return state

    def run(self, ctx: Context, settings: TrackingAnalysisSettings) -> Result:
        from ..metadata import table_frame_time
        from ..tracking.analysis import fit_jumps, gather, jumps, msd_analysis
        if not (settings.msd or settings.jumps):
            raise ValueError("tick MSD or jump distances")
        locs = ctx.locs
        from ..tracking.columns import track_column
        column = track_column(locs, settings.id_column)
        for c in ("x_nm", "y_nm", "frame"):
            if c not in locs:
                raise ValueError(f"tracking needs {c}: the table has "
                                 + ", ".join(sorted(locs.keys())))
        use = np.asarray(ctx.selection.mask, dtype=bool)
        clock = "time_s" in locs
        ft = None if clock else (settings.frame_time_ms or table_frame_time(locs))
        if not clock and not ft:
            raise ValueError("no frame time: the file records neither a frame time "
                             "nor an exposure, so set the frame time")
        ids = np.where(use, np.asarray(locs[column], np.int64), 0)
        in_3d = settings.use_z and "z_nm" in locs
        t = gather(ids, locs["frame"], locs["x_nm"], locs["y_nm"],
                   locs["z_nm"] if in_3d else None,
                   locs["xy_err_nm"] if "xy_err_nm" in locs else None,
                   locs["z_err_nm"] if in_3d and "z_err_nm" in locs else None,
                   frame_time_s=(ft or 1.0) / 1000.0,
                   time_s=locs["time_s"] if clock else None)
        if not t.n_tracks:
            if ctx.per_region:
                return Result(values={"n_tracks": 0}, settings=settings)
            raise ValueError(f"no tracks: no localization has {column} above 0")
        if ctx.stopped:
            raise Stopped("tracking analysis stopped: nothing measured")
        if "xy_err_nm" not in locs and settings.msd_fit.offset == "precisions":
            ctx.warn("no xy_err_nm: the offset is taken as 0; fit it instead")
        values: Dict[str, Any] = {"n_tracks": int(t.n_tracks)}
        arrays: Dict[str, np.ndarray] = {}
        lines = [f"{t.n_tracks} tracks of {len(t.rows)} localizations ({column}), "
                 f"a step of {t.dt_s * 1000:.4g} ms"
                 + (" (time_s)" if clock else "") + (", 3D" if in_3d else "")]
        panels = []
        msd = jump = None
        out_locs = None

        if settings.msd:
            m = settings.msd_fit
            ctx.report("tracking analysis: MSD")
            msd = msd_analysis(t, m.offset == "fitted", m.lags, m.min_length)
            p = msd.pooled
            per = msd.per_track
            have = np.isfinite(per.D_um2_per_s)
            values.update({
                "msd_D_um2_per_s": float(p.D_um2_per_s[0]),
                "msd_alpha": float(p.alpha[0]), "msd_lags": int(p.lags[0]),
                "msd_offset_nm2": float(p.offset_nm2[0]),
                "track_D_median_um2_per_s": (float(np.median(per.D_um2_per_s[have]))
                                             if have.any() else float("nan")),
                "track_alpha_median": (float(np.nanmedian(per.alpha[have]))
                                       if np.isfinite(per.alpha[have]).any()
                                       else float("nan")),
                "n_tracks_fitted": int(have.sum())})
            arrays.update({"lag_s": msd.lag_s, "msd_nm2": msd.msd_nm2,
                           "msd_pairs": msd.pairs})
            lines.append(
                f"MSD over all: D = {p.D_um2_per_s[0]:.4g} µm²/s, alpha = "
                f"{p.alpha[0]:.3g} ({int(p.lags[0])} lag{'s' if p.lags[0] > 1 else ''}, offset "
                f"{p.offset_nm2[0]:.0f} nm²); per track (median of {int(have.sum())} "
                f"with {m.min_length}+ localizations): D = "
                f"{values['track_D_median_um2_per_s']:.4g} µm²/s, alpha = "
                f"{values['track_alpha_median']:.3g}")
            if m.write_columns and not ctx.per_region:
                out_locs = _with_track_values(locs, t, per)
            panels += [lambda ax, msd=msd: draw_msd(ax, msd, t.dims),
                       lambda ax, msd=msd: draw_track_d(ax, msd)]

        if settings.jumps:
            j = settings.jump_fit
            ctx.report("tracking analysis: jump distances")
            d2, err = jumps(t, j.lag)
            lag_s = j.lag * t.dt_s
            try:
                jump = fit_jumps(d2, err, t.dims, lag_s, j.states)
            except ValueError as error:
                if not ctx.per_region:
                    raise
                ctx.warn(str(error))
            if jump is not None:
                for i, (D, f) in enumerate(zip(jump.D_um2_per_s, jump.fraction), 1):
                    values[f"jump_D{i}_um2_per_s"] = float(D)
                    values[f"jump_fraction{i}"] = float(f)
                values["jump_bic"] = float(jump.bic)
                values["n_jumps"] = int(jump.n_jumps)
                states = ", ".join(f"D = {D:.4g} µm²/s ({100 * f:.0f}%)"
                                   for D, f in zip(jump.D_um2_per_s, jump.fraction))
                fitted = ("" if jump.n_fitted == jump.n_jumps else
                          f" ({jump.n_fitted} fitted)")
                lines.append(f"jumps at {j.lag} frame{'s' if j.lag > 1 else ''}: "
                             f"{jump.n_jumps} jumps{fitted}, {states}; "
                             f"BIC {jump.bic:.1f}")
                r = np.sqrt(d2)
                err_typical = float(np.median(err)) if len(err) else 0.0
                panels.append(lambda ax, r=r, jump=jump: draw_jumps(
                    ax, r, jump, err_typical, t.dims, lag_s))

        def draw(figure):
            axes = np.atleast_1d(figure.subplots(1, len(panels)))
            for ax, panel in zip(axes, panels):
                panel(ax)

        plot = (Plot(draw, panels=len(panels), size=(4.0 * len(panels), 3.6))
                if panels else None)
        return Result(locs=out_locs, text="\n".join(lines), values=values,
                      arrays=arrays, plot=plot, settings=settings,
                      data={"msd": msd, "jumps": jump, "tracks": t},
                      title="tracking analysis")


def _with_track_values(locs: Localizations, t, per) -> Localizations:
    """``track_D_um2_per_s`` and ``track_alpha`` on the analysed rows, NaN
    elsewhere and for a track too short; recipes that average per blink."""
    from ..mathparse import remember
    n = len(locs)
    D = np.full(n, np.nan, np.float32)
    alpha = np.full(n, np.nan, np.float32)
    D[t.rows] = per.D_um2_per_s[t.track]
    alpha[t.rows] = per.alpha[t.track]
    out = Localizations({**locs.columns, "track_D_um2_per_s": D, "track_alpha": alpha},
                        dict(locs.metadata))
    remember(out, "track_D_um2_per_s", "", grouped="mean")
    remember(out, "track_alpha", "", grouped="mean")
    return out


def draw_msd(ax, msd, dims: int) -> None:
    """The pooled MSD against lag, the lags fitted and the fitted line."""
    p = msd.pooled
    k = int(p.lags[0])
    t, y = msd.lag_s, msd.msd_nm2
    ax.plot(t, y / 1e6, "o", ms=3, color="0.6")
    ax.plot(t[:k], y[:k] / 1e6, "o", ms=4, color="C0",
            label=f"fitted ({k} lag{'s' if k > 1 else ''})")
    line = np.linspace(0, t[min(len(t) - 1, max(3 * k, 5))], 50)
    D = p.D_um2_per_s[0]
    fit = (p.offset_nm2[0] / 1e6 + 2 * dims * D * line)
    ax.plot(line, fit, color="C3", lw=1, label=f"D = {D:.3g} µm²/s")
    ax.set_xlim(0, line[-1] * 1.05)
    ax.set_ylim(0, np.nanmax(np.r_[fit, y[:max(3 * k, 5)] / 1e6]) * 1.1)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("MSD (µm²)")
    ax.legend(fontsize=7)
    ax.set_title("MSD of all tracks", fontsize=9)


def draw_track_d(ax, msd) -> None:
    """The per-track D, on a log axis."""
    D = msd.per_track.D_um2_per_s
    D = D[np.isfinite(D) & (D > 0)]
    if len(D):
        bins = np.logspace(np.log10(D.min()), np.log10(D.max()) + 1e-9,
                           min(40, max(5, len(D) // 5)))
        ax.hist(D, bins=bins, color="0.6")
        ax.set_xscale("log")
    ax.set_xlabel("D per track (µm²/s)")
    ax.set_ylabel("tracks")
    ax.set_title(f"{len(D)} tracks", fontsize=9)


def draw_jumps(ax, r, jump, err, dims, lag_s) -> None:
    """The jump distances and the fitted states."""
    from ..tracking.analysis import jump_density
    top = np.quantile(r, 0.995) if len(r) else 1.0
    bins = np.linspace(0, top, 50)
    ax.hist(r, bins=bins, density=True, color="0.75")
    grid = np.linspace(0, top, 200)
    parts = jump_density(grid, jump, err, dims, lag_s)
    for i, part in enumerate(parts):
        ax.plot(grid, part, color=f"C{i}", lw=1,
                label=f"{jump.D_um2_per_s[i]:.3g} µm²/s, {100 * jump.fraction[i]:.0f}%")
    if len(parts) > 1:
        ax.plot(grid, parts.sum(0), color="k", lw=1)
    ax.set_xlabel("jump (nm)")
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.legend(fontsize=7)
    ax.set_title(f"{jump.n_jumps} jumps", fontsize=9)
