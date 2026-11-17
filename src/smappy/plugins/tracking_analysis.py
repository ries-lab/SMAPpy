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
  which gives the states' D and fractions without a histogram's bins --
  at several lags at once, and with an *axial range* corrected for fast
  particles leaving the focus (Spot-On's model, saSPT's correction:
  `tracking.analysis.in_focus`).  In 3D the correction fits the lateral
  jumps alone: the particles that stay in focus are those that moved
  less in z, so their z jumps are too short;
* two approaches SMAP did not have: the D spectrum, saSPT's state arrays
  (`tracking.spectrum`), and state switching, ExTrack's hidden Markov
  model with the rates between states (`tracking.hmm`);
* the *track id* is any integer column -- `track_id` from Link tracks,
  MINFLUX's `tid` (on its own clock, `time_s`), a cluster's id -- and on
  *auto*, the default, the first of those the table has
  (`tracking.columns`).

``per_region``: in an ROI, or per track as a block over `track_id`, the
values go to the region table.  Over the view the MSD's fit per track --
computed for every track at once anyway -- goes back as a region table of
the track id ("per track_id: Tracking analysis"), what a block over the
tracks would give in one run, for the summaries and the region filter;
only where the run saw the layer's own filter (no drawn ROI), since the
table is signed as a block's (`region_runs.column_table`).  Per-track D and
alpha can also be written onto the localizations (*write per track*), to
colour the tracks by them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from . import (NO_FRAME_TIME, Context, FormState, Plot, Plugin, Result, Stopped,
               frame_time_notice, param, register, settings_values)
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
    lags: int = param(4, label="lags", unit="frames", min=1, max=20,
                      help="the jumps 1, 2, ... this many frames apart, fitted together")
    states: int = param(2, label="states", min=1, max=3,
                        help="diffusive states fitted, 1 to 3")


@dataclass
class SwitchingSettings:
    states: int = param(2, label="states", min=1, max=3,
                        help="diffusive states that switch, 1 to 3")
    write_columns: bool = param(False, label="write per localization",
                                help="hmm_state and hmm_p1, hmm_p2, ...: each "
                                     "localization's state and its probabilities")


@dataclass
class SpectrumSettings:
    split: int = param(10, label="piece", unit="jumps", min=1, max=100,
                       help="tracks cut into pieces of at most this many jumps, "
                            "each given its own D")
    d_min: float = param(0.001, label="D from", unit="µm²/s", min=1e-6,
                         advanced=True, help="the smallest D of the grid")
    d_max: float = param(100.0, label="to", unit="µm²/s", min=1e-5, beside=True,
                         advanced=True, help="the largest D of the grid")
    write_column: bool = param(False, label="write per localization",
                               help="D_post_um2_per_s: each piece's D onto its "
                                    "localizations")


@dataclass
class TrackingAnalysisSettings:
    msd: bool = param(True, label="MSD", help="the mean squared displacement over lag")
    jumps: bool = param(True, label="jump distances",
                        help="the distribution of jumps, as diffusive states")
    spectrum: bool = param(False, label="D spectrum",
                           help="the jumps over a grid of D, no number of states "
                                "chosen, and each piece of a track's D (saSPT)")
    switching: bool = param(False, label="state switching",
                            help="states that switch in time: their D, the rates "
                                 "between them and each localization's state (HMM)")
    id_column: str = param("", label="track id", choices=_id_choices,
                           help="the attribute naming the tracks; auto: track_id, "
                                "else tid, else track_id_true")
    use_z: bool = param(True, label="in 3D", help="x, y and z when the table has z_nm")
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0.001, advanced=True,
                                           help="auto: the file's, else its exposure")
    axial_range_nm: float = param(0.0, label="axial range", unit="nm", min=0.0,
                                  help="the depth in which particles are seen, for "
                                       "the fast ones leaving it; 0: no correction")
    msd_fit: MSDFitSettings = param(default_factory=MSDFitSettings, label="MSD")
    jump_fit: JumpFitSettings = param(default_factory=JumpFitSettings,
                                      label="jump distances")
    spectrum_fit: SpectrumSettings = param(default_factory=SpectrumSettings,
                                           label="D spectrum")
    switching_fit: SwitchingSettings = param(default_factory=SwitchingSettings,
                                             label="state switching")


@register("Analysis/Tracking/Tracking analysis")
class TrackingAnalysis(Plugin):
    """How the tracks move: the mean squared displacement, with D and alpha
    per track and over all, and the jump distances as diffusive states."""

    Settings = TrackingAnalysisSettings
    changes_locs = False
    per_region = True
    # tracks are of localizations, whatever the layer shows
    grouping = "ungrouped"
    version = "3"

    def changes_locs_for(self, settings: TrackingAnalysisSettings) -> bool:
        return bool((settings.msd and settings.msd_fit.write_columns)
                    or (settings.spectrum and settings.spectrum_fit.write_column)
                    or (settings.switching and settings.switching_fit.write_columns))

    def describe(self, ctx: Optional[Context], settings: TrackingAnalysisSettings
                 ) -> FormState:
        state = FormState(shown={"msd_fit": settings.msd, "jump_fit": settings.jumps,
                                 "spectrum_fit": settings.spectrum,
                                 "switching_fit": settings.switching},
                          active={"axial_range_nm": settings.jumps or settings.spectrum})
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
                elif not settings.frame_time_ms:
                    state.notice = frame_time_notice()
        return state

    def run(self, ctx: Context, settings: TrackingAnalysisSettings) -> Result:
        from ..metadata import table_frame_time
        from ..tracking.analysis import fit_jumps, gather, jumps_over_lags, msd_analysis
        if not (settings.msd or settings.jumps or settings.spectrum or settings.switching):
            raise ValueError("tick MSD, jump distances, D spectrum or state switching")
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
            raise ValueError(NO_FRAME_TIME)
        ids = np.where(use, np.asarray(locs[column], np.int64), -1)   # -1: in none
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
            raise ValueError(f"no tracks: no localization has a {column} (-1 is none)")
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
        id_regions = {}

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
            if not ctx.per_region:
                from ..region_runs import column_table, layer_mask
                if np.array_equal(layer_mask(ctx), use):
                    step = {"label": self.name, "plugin": self.path,
                            "version": self.version,
                            "parameters": settings_values(settings)}
                    per_track = {"n_locs": msd.n_locs.astype(np.int64),
                                 "D_um2_per_s": per.D_um2_per_s, "alpha": per.alpha,
                                 "lags": per.lags.astype(float),
                                 "reduced_error": per.x}
                    id_regions = {column: {self.name: column_table(
                        locs, column, self.name, msd.ids, per_track, step,
                        layer=ctx.layer, session=ctx.session)}}
                    lines.append(f"kept per track as 'per {column}: {self.name}'")
            panels += [lambda ax, msd=msd: draw_msd(ax, msd, t.dims),
                       lambda ax, msd=msd: draw_track_d(ax, msd)]

        focus = float(settings.axial_range_nm or 0.0)
        tj = t
        if focus > 0 and t.dims == 3 and (settings.jumps or settings.spectrum
                                          or settings.switching):
            # what stays in focus moved less in z: the lateral jumps alone
            tj = gather(ids, locs["frame"], locs["x_nm"], locs["y_nm"], None,
                        locs["xy_err_nm"] if "xy_err_nm" in locs else None,
                        frame_time_s=(ft or 1.0) / 1000.0,
                        time_s=locs["time_s"] if clock else None)

        if settings.jumps:
            j = settings.jump_fit
            ctx.report("tracking analysis: jump distances")
            d2, err, lag = jumps_over_lags(tj, j.lags)
            try:
                jump = fit_jumps(d2, err, tj.dims, lag * t.dt_s, j.states,
                                 focal_range_nm=focus, step_s=t.dt_s)
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
                at = (f"1 to {j.lags} frames" if j.lags > 1 else "1 frame")
                where = (f", corrected for a {focus:g} nm focus"
                         + (" (lateral jumps)" if tj is not t else "")) if focus > 0 else ""
                lines.append(f"jumps at {at}: {jump.n_jumps} jumps{fitted}{where}, "
                             f"{states}; BIC {jump.bic:.1f}")
                if focus > 0:
                    values.update({f"jump_seen{i}": float(v)
                                   for i, v in enumerate(jump.seen[:, 0], 1)})
                    lines.append("of the jumps at 1 frame: " + ", ".join(
                        f"{100 * v:.0f}%" for v in jump.seen[:, 0]))
                one = lag == 1
                r = np.sqrt(d2[one])
                err_typical = float(np.median(err[one])) if one.any() else 0.0
                panels.append(lambda ax, r=r, jump=jump, dims=tj.dims: draw_jumps(
                    ax, r, jump, err_typical, dims, t.dt_s))
                if len(jump.lag_s) > 1:
                    seen = _seen_per_lag(d2, err, lag, jump, tj.dims, t.dt_s)
                    panels.append(lambda ax, jump=jump, seen=seen: draw_shares(
                        ax, jump, seen))

        spec = None
        if settings.spectrum:
            sp = settings.spectrum_fit
            ctx.report("tracking analysis: D spectrum")
            from ..tracking.spectrum import d_grid, peaks, per_localization, spectrum
            if not 0 < sp.d_min < sp.d_max:
                raise ValueError(f"the D grid runs from {sp.d_min:g} to {sp.d_max:g} "
                                 "µm²/s: the first must be the smaller")
            try:
                spec = spectrum(tj, d_grid(sp.d_min, sp.d_max), sp.split, focus)
            except ValueError as error:
                if not ctx.per_region:
                    raise
                ctx.warn(str(error))
            if spec is not None:
                found = peaks(spec)
                values["n_pieces"] = len(spec.pieces)
                for i, (D, f) in enumerate(found, 1):
                    values[f"spectrum_D{i}_um2_per_s"] = D
                    values[f"spectrum_fraction{i}"] = f
                arrays.update({"spectrum_D_um2_per_s": spec.D_um2_per_s,
                               "spectrum_occupation": spec.occupation})
                where = (f", corrected for a {focus:g} nm focus"
                         + (" (lateral jumps)" if tj is not t else "")) if focus > 0 else ""
                lines.append(
                    f"D spectrum: {len(spec.pieces)} pieces of up to {sp.split} jumps "
                    f"({spec.n_jumps} jumps){where}; peaks at "
                    + (", ".join(f"{D:.3g} µm²/s ({100 * f:.0f}%)" for D, f in found)
                       or "none"))
                if sp.write_column and not ctx.per_region:
                    out_locs = _with_column(out_locs if out_locs is not None else locs,
                                            "D_post_um2_per_s",
                                            per_localization(tj, spec, len(locs)))
                panels.append(lambda ax, spec=spec, found=found: draw_spectrum(
                    ax, spec, found))

        hmm = None
        if settings.switching:
            sw = settings.switching_fit
            ctx.report("tracking analysis: state switching")
            from ..tracking.hmm import (CROWDED, crowding, fit_hmm, nearest_spacing_nm,
                                        per_localization as hmm_states)
            try:
                hmm = fit_hmm(tj, sw.states)
            except ValueError as error:
                if not ctx.per_region:
                    raise
                ctx.warn(str(error))
            if hmm is not None:
                p = hmm.params
                fractions, dwell = hmm.fractions, p.dwell_s()
                for i in range(p.states):
                    values[f"hmm_D{i + 1}_um2_per_s"] = float(p.D_um2_per_s[i])
                    values[f"hmm_fraction{i + 1}"] = float(fractions[i])
                    values[f"hmm_dwell{i + 1}_s"] = float(dwell[i])
                    for k in range(p.states):
                        if k != i:
                            values[f"hmm_rate{i + 1}{k + 1}_per_s"] = float(p.Q_per_s[i, k])
                values["hmm_bic"] = float(hmm.bic)
                states = ", ".join(f"D = {D:.4g} µm²/s ({100 * f:.0f}%, {d:.3g} s)"
                                   for D, f, d in zip(p.D_um2_per_s, fractions, dwell))
                rates = ", ".join(f"{i + 1}→{k + 1} {p.Q_per_s[i, k]:.3g}/s"
                                  for i in range(p.states) for k in range(p.states)
                                  if i != k)
                lines.append(f"state switching: {hmm.n_jumps} jumps of {hmm.n_tracks} "
                             f"tracks; {states} (fraction, dwell)"
                             + (f"; rates {rates}" if rates else "")
                             + f"; BIC {hmm.bic:.1f}")
                spacing = nearest_spacing_nm(np.asarray(locs["x_nm"])[use], np.asarray(locs["y_nm"])[use],
                                             np.asarray(locs["frame"])[use])
                crowded = crowding(p, t.dt_s, spacing)
                values["hmm_crowding"] = crowded
                if crowded > CROWDED:
                    ctx.warn(f"state switching: the fastest state's jump is "
                             f"{crowded:.2f} of the {spacing:.0f} nm between "
                             "particles: wrong links look like switches, and in "
                             "simulations the rates came out 10-20% high from 0.15 "
                             "and twice as high at 0.4; image sparser or read the "
                             "rates as upper bounds")
                if focus > 0:
                    ctx.warn("state switching does not correct for the focus: fast "
                             "particles leave it, and in simulations the rate from "
                             "fast to slow came out high and the fast fraction low; "
                             "take the fractions from the jump distances or the D "
                             "spectrum")
                if sw.write_columns and not ctx.per_region:
                    prob, state = hmm_states(tj, p, len(locs))
                    base = out_locs if out_locs is not None else locs
                    out_locs = _with_states(base, prob, state)
                panels.append(lambda ax, hmm=hmm: draw_switching(ax, hmm))

        def draw(figure):
            axes = np.atleast_1d(figure.subplots(1, len(panels)))
            for ax, panel in zip(axes, panels):
                panel(ax)

        plot = (Plot(draw, panels=len(panels), size=(4.0 * len(panels), 3.6))
                if panels else None)
        return Result(locs=out_locs, text="\n".join(lines), values=values,
                      id_regions=id_regions,
                      arrays=arrays, plot=plot, settings=settings,
                      data={"msd": msd, "jumps": jump, "spectrum": spec,
                            "switching": hmm, "tracks": t},
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


def _with_column(locs: Localizations, name: str, values: np.ndarray) -> Localizations:
    """``name`` added (or replaced) as a column averaged per blink."""
    from ..mathparse import remember
    out = Localizations({**locs.columns, name: values.astype(np.float32)},
                        dict(locs.metadata))
    remember(out, name, "", grouped="mean")
    return out


def _with_states(locs: Localizations, prob: np.ndarray, state: np.ndarray
                 ) -> Localizations:
    """``hmm_state`` (-1 where none) and ``hmm_p1``, ``hmm_p2``, ...: a
    blink takes its localizations' mean probabilities, and the largest
    state among them."""
    from ..mathparse import remember
    columns = dict(locs.columns)
    columns["hmm_state"] = state.astype(np.int32)
    names = [f"hmm_p{i + 1}" for i in range(prob.shape[1])]
    for i, name in enumerate(names):
        columns[name] = prob[:, i].astype(np.float32)
    out = Localizations(columns, dict(locs.metadata))
    remember(out, "hmm_state", "", grouped="max")
    for name in names:
        remember(out, name, "", grouped="mean")
    return out


def _seen_per_lag(d2, err, lag, jump, dims, step_s) -> np.ndarray:
    """Each state's share of the jumps at each lag as the jumps say it, the
    mean of their memberships (states x lags): what `draw_shares` sets
    against the model's."""
    from scipy.special import logsumexp
    from ..tracking.analysis import _jump_logpdf
    lags = np.unique(lag)
    D = jump.D_um2_per_s * 1e6
    out = np.zeros((len(D), len(lags)))
    for i, k in enumerate(lags):
        here = lag == k
        s2 = 2 * D[None, :] * k * step_s + err[here, None]
        terms = (_jump_logpdf(d2[here, None], s2, dims)
                 + np.log(np.maximum(jump.seen[:, i], 1e-300))[None, :])
        member = np.exp(terms - logsumexp(terms, axis=1)[:, None])
        out[:, i] = member.mean(0)
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
    """The jump distances at the first lag and the fitted states, each at
    its share of those jumps."""
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
    ax.set_title(f"{len(r)} jumps at {lag_s * 1000:.3g} ms", fontsize=9)


def draw_shares(ax, jump, seen) -> None:
    """Each state's share of the jumps against lag: the model's (lines)
    and the jumps' own (dots), and, corrected, its share of the particles
    (dashed): fast states lose share with lag when they leave the focus."""
    lag = jump.lag_s
    for i, D in enumerate(jump.D_um2_per_s):
        ax.plot(lag, jump.seen[i], color=f"C{i}", lw=1,
                label=f"{D:.3g} µm²/s")
        ax.plot(lag, seen[i], "o", ms=3, color=f"C{i}")
        if jump.focal_range_nm:
            ax.axhline(jump.fraction[i], color=f"C{i}", lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("share of the jumps")
    ax.legend(fontsize=7)
    ax.set_title("per lag" + (" (dashed: of the particles)"
                              if jump.focal_range_nm else ""), fontsize=9)


def draw_spectrum(ax, spec, found) -> None:
    """The occupations over D: of the particles, and, corrected for the
    focus, of the jumps (dashed); the peaks marked with their fractions."""
    D = spec.D_um2_per_s
    ax.plot(D, spec.occupation, color="C0", lw=1.2, label="particles"
            if spec.focal_range_nm else None)
    if spec.focal_range_nm:
        ax.plot(D, spec.seen, color="0.5", lw=0.8, ls="--", label="jumps")
        ax.legend(fontsize=7)
    for d, f in found:
        ax.annotate(f"{100 * f:.0f}%", (d, spec.occupation[np.argmin(np.abs(D - d))]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.set_xlabel("D (µm²/s)")
    ax.set_ylabel("occupation")
    ax.set_title(f"D spectrum, {len(spec.pieces)} pieces", fontsize=9)


def draw_switching(ax, hmm) -> None:
    """The states as circles of area by their fraction, with their D, and
    an arrow per rate."""
    p = hmm.params
    S = p.states
    fractions = hmm.fractions
    angle = np.pi / 2 + 2 * np.pi * np.arange(S) / max(S, 1)
    xy = (np.column_stack([np.cos(angle), np.sin(angle)]) if S > 2 else
          np.column_stack([np.linspace(-1, 1, S) if S > 1 else [0.0], np.zeros(S)]))
    for i in range(S):
        for k in range(S):
            if i == k:
                continue
            a, b = xy[i], xy[k]
            normal = np.array([-(b - a)[1], (b - a)[0]])
            normal = 0.08 * normal / max(np.hypot(*normal), 1e-9)
            ax.annotate("", xy=b + normal - 0.3 * (b - a), xytext=a + normal + 0.3 * (b - a),
                        arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 0.8})
            mid = (a + b) / 2 + 2.2 * normal
            ax.text(*mid, f"{p.Q_per_s[i, k]:.3g}/s", fontsize=7, ha="center",
                    va="center", color="0.3")
    for i in range(S):
        ax.scatter(*xy[i], s=200 + 2500 * fractions[i], color=f"C{i}", alpha=0.6,
                   zorder=3)
        ax.text(xy[i][0], xy[i][1] - 0.32,
                f"{p.D_um2_per_s[i]:.3g} µm²/s\n{100 * fractions[i]:.0f}%",
                fontsize=7, ha="center", va="top")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("state switching", fontsize=9)
