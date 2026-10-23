"""Stability: how still the sample is, measured on beads.

A diagnostic with *approaches*, ticked on or off as in *Spatial statistics*
and *Tracking analysis*.  This step (wave 5, 5e) builds **vibrations**: the
jitter of a bead's position over time and its spectrum, SMAP's
``Analyze/measure/Vibrations``.  The second, **random scatter**, is the
noise that averages away: the Allan deviation of each trace's position,
which at one point is RMS(diff(x))/sqrt(2), set against the precision the
fit claims -- NeNA's question asked of a trace known to be one bead, at
every time scale.  It is off by default: a bead file is opened for its
vibrations first.  Long-term drift and the dual-channel drift
are to join as approaches of their own (`docs/smap_porting/wave5_design.md`,
topic 2).  It diagnoses and corrects nothing: correcting drift from beads is
*Fiducials*; the two share only the bead tracks.

The algorithms are `smappy.stability` (its docstring has where they depart
from SMAP's); this is the wrapper.  The beads are the localizations of an
*id column* -- ``track_id`` from *Link tracks* by default -- so it runs on
one bead in an ROI as SMAP's did, on every bead of the view, or once per
bead as a block over ``track_id``.  Time is the file's frame time
(`metadata.file_frame_time`) or the table's own clock (``time_s``).

``per_region``: per axis the SD, the precision, the vibration's RMS and the
largest peak's frequency and amplitude, pooled over the region's traces
(`stability.pooled`); the amplitude spectrum and the cumulative RMS of the
first trace are its arrays.

With *each bead*, over the view, the same numbers per bead go back as a
region table of the id column ("per track_id: Stability", `per_bead`):
what a block over the beads would give, in one run, for the summaries and
the region filter -- a bead that vibrates more than the others is found
there.  Only where the run saw the layer's own filter (no drawn ROI), since
the table is signed as a block's (`region_runs.column_table`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import (NO_FRAME_TIME, Context, FormState, Plot, Plugin, Result, frame_time_notice,
               param, register, settings_values)

# traces drawn over each other in a figure; more is a tangle
MAX_DRAWN = 8


def _id_choices():
    return [("track_id", "track_id"), ("tid", "tid")]


def _integer_columns(locs) -> List[str]:
    return [n for n in locs.keys() if np.asarray(locs[n]).ndim == 1
            and np.asarray(locs[n]).dtype.kind in "iu"]


@dataclass
class VibrationSettings:
    highpass_hz: float = param(1.0, label="high-pass", unit="Hz", min=0.0,
                               help="slower motion is drift, removed before the SD "
                                    "and the peaks; 0: only the mean")
    segment: int = param(1024, label="segment", unit="points", min=16,
                         help="the spectrum is averaged over segments this long")
    n_peaks: int = param(5, label="peaks", min=1, max=20,
                         help="how many of the largest peaks are listed")
    zoom: bool = param(True, label="trace, zoomed",
                       help="the first points of the trace, with a running mean")
    zoom_points: int = param(1000, label="zoom to", unit="points", min=10,
                             help="how many points the zoomed trace shows")
    smoothing: int = param(10, label="running mean", unit="points", min=1,
                           help="the overlay on the zoomed trace")
    whole: bool = param(True, label="whole trace",
                        help="all of it, high-passed, with its SD and the precision")
    amplitude: bool = param(True, label="amplitude spectrum",
                            help="the amplitude in nm against frequency")
    cumulative: bool = param(True, label="cumulative RMS",
                             help="the RMS of everything up to each frequency")
    peak_table: bool = param(True, label="table of peaks",
                             help="the largest peaks' frequencies and amplitudes")


@dataclass
class ScatterSettings:
    time_scale_ms: Optional[float] = param(None, label="time scale", unit="ms",
                                           min=0.001,
                                           help="also the deviation of means over "
                                                "this time; auto: one point only")
    max_gap: float = param(3.0, label="gap", unit="steps", min=1.0,
                           help="points further apart than this many typical "
                                "steps are not compared")
    longest_points: int = param(1000, label="longest", unit="points", min=2,
                                help="the curve averages up to this many points")
    allan: bool = param(True, label="Allan deviation",
                        help="the deviation against the time averaged, with "
                             "the precision's 1/sqrt(n)")
    steps: bool = param(True, label="steps",
                        help="histogram of diff(x)/sqrt(2) beside the precision's "
                             "Gaussian")


@dataclass
class StabilitySettings:
    vibrations: bool = param(True, label="vibrations",
                             help="the jitter of the beads and its spectrum")
    scatter: bool = param(False, label="random scatter",
                          help="the noise that averages away, from point-to-point "
                               "steps, against the precision")
    id_column: str = param("track_id", label="id column", choices=_id_choices,
                           help="which localizations are one bead")
    traces: str = param("each", label="traces",
                        choices=(("each", "each bead"),
                                 ("mean", "the mean over beads"),
                                 ("pair", "differential: minus the first bead"),
                                 ("others", "differential: minus the others")),
                        help="what is analysed; a differential trace cancels "
                             "what all beads share")
    use_z: bool = param(True, label="z", help="z too, when the table has z_nm")
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0.001, advanced=True,
                                           help="auto: the file's, else its exposure")
    vibration: VibrationSettings = param(default_factory=VibrationSettings,
                                         label="vibrations")
    random: ScatterSettings = param(default_factory=ScatterSettings,
                                    label="random scatter")


@register("Analysis/Measure/Stability")
class Stability(Plugin):
    """How still the sample is, from beads: the vibrations of each bead, their
    mean or their differences -- the trace, its SD beside the precision, the
    amplitude spectrum, the cumulative RMS and the main peaks -- and the
    random scatter, the Allan deviation of the position against the
    precision."""

    Settings = StabilitySettings
    changes_locs = False
    per_region = True
    # a bead is localized once per frame: its localizations, whatever the layer shows
    grouping = "ungrouped"
    version = "2"

    def describe(self, ctx: Optional[Context], settings: StabilitySettings) -> FormState:
        v = settings.vibration
        state = FormState(shown={"vibration": settings.vibrations,
                                 "vibration.zoom_points": v.zoom,
                                 "vibration.smoothing": v.zoom,
                                 "random": settings.scatter})
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            locs = ctx.locs
            columns = _integer_columns(locs)
            state.choices["id_column"] = [(c, c) for c in columns] or _id_choices()
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

    def run(self, ctx: Context, settings: StabilitySettings) -> Result:
        if not (settings.vibrations or settings.scatter):
            raise ValueError("tick an approach")
        from .. import stability as st
        from ..metadata import table_frame_time
        locs = ctx.locs
        column = settings.id_column
        if column not in locs:
            raise ValueError(f"no column {column!r}: link the beads first "
                             "(Analysis/Tracking/Link tracks), or choose one of "
                             + ", ".join(_integer_columns(locs)))
        for c in ("x_nm", "y_nm", "frame"):
            if c not in locs:
                raise ValueError(f"stability needs {c}: the table has "
                                 + ", ".join(sorted(locs.keys())))
        use = np.asarray(ctx.selection.mask, dtype=bool)
        ids = np.where(use, np.asarray(locs[column], np.int64), 0)
        clock = "time_s" in locs
        axes = ["x", "y"] + (["z"] if settings.use_z and "z_nm" in locs else [])
        xyz = np.column_stack([np.asarray(locs[f"{a}_nm"], float) for a in axes])
        err_cols = {"x": "xy_err_nm", "y": "xy_err_nm", "z": "z_err_nm"}
        err = np.column_stack([np.asarray(locs[err_cols[a]], float)
                               if err_cols[a] in locs else np.full(len(locs), np.nan)
                               for a in axes])
        if clock:
            t_s = np.asarray(locs["time_s"], float)
            dt_s = st.grid_step(t_s[ids > 0])
        else:
            ft = settings.frame_time_ms or table_frame_time(locs)
            if not ft:
                raise ValueError(NO_FRAME_TIME)
            dt_s = ft / 1000.0
            t_s = np.asarray(locs["frame"], float) * dt_s
        ctx.report("stability: traces")
        beads = st.bead_traces(ids, t_s, xyz, dt_s, err)
        if not beads:
            if ctx.per_region:
                return Result(values={"n_traces": 0}, settings=settings)
            raise ValueError(f"no beads: no localization has {column} above 0")
        traces = st.combine(beads, settings.traces)
        values: Dict[str, Any] = {}
        lines = [f"{len(beads)} bead{'s' if len(beads) != 1 else ''} ({column}), "
                 f"{len(traces)} trace{'s' if len(traces) != 1 else ''} "
                 f"({_traces_label(settings.traces)}); "
                 f"a step of {dt_s * 1000:.4g} ms" + (" (time_s, resampled)" if clock else "")]
        arrays: Dict[str, np.ndarray] = {}
        plots: Dict[str, Plot] = {}

        def row(draw_axis):
            def draw(figure):
                axs = np.atleast_1d(figure.subplots(1, len(axes)))
                for a, ax in enumerate(axs):
                    draw_axis(ax, a)
            return Plot(draw, panels=len(axes), size=(3.6 * len(axes), 3.2))

        results: list = []
        if settings.vibrations:
            results = self._vibrations(ctx, settings, traces, axes, dt_s, values,
                                       lines, arrays, plots, row)
        scattered: list = []
        if settings.scatter:
            scattered = self._scatter(ctx, settings, traces, axes, dt_s, values,
                                      lines, arrays, plots, row)
        if not results and not scattered:
            if ctx.per_region:
                return Result(values={"n_traces": 0}, settings=settings)
            raise ValueError("every trace is shorter than 4 points")
        values["n_traces"] = len(results) if results else len(scattered)
        main = None
        if plots:
            key = next(iter(plots))
            main = plots.pop(key)
            main = Plot(main.draw, name="", panels=main.panels, size=main.size)
        id_regions = {}
        if settings.traces == "each" and not ctx.per_region:
            from ..region_runs import column_table, layer_mask
            if np.array_equal(layer_mask(ctx), use):
                ids_, values_, groups = per_bead(results, axes, scattered)
                step = {"label": self.name or "Stability", "plugin": self.path,
                        "version": self.version, "parameters": settings_values(settings)}
                id_regions = {column: {step["label"]: column_table(
                    locs, column, step["label"], ids_, values_, step, layer=ctx.layer,
                    session=ctx.session, groups=groups)}}
                lines.append(f"kept per bead as 'per {column}: {step['label']}'")
        return Result(text="\n".join(lines), plot=main, plots=plots, values=values,
                      arrays=arrays, settings=settings, title="stability",
                      id_regions=id_regions,
                      data={"traces": traces, "results": results, "dt_s": dt_s,
                            "peaks": data_peaks(results), "scatter": scattered})

    def _vibrations(self, ctx, settings, traces, axes, dt_s, values, lines,
                    arrays, plots, row) -> list:
        """The vibrations approach into ``values``, ``lines``, ``arrays`` and
        ``plots``; its results per trace, [] where every trace is too short."""
        from .. import stability as st
        v = settings.vibration
        ctx.report("stability: spectra")
        results = st.analyse(traces, axes, highpass_hz=v.highpass_hz,
                             segment=v.segment, n_peaks=v.n_peaks)
        if not results:
            return []
        n_points = sum(r.run.stop - r.run.start for r in results)
        n_filled = sum(int(r.trace.filled[r.run].sum()) for r in results)
        values.update({"n_points": int(n_points), "n_filled": int(n_filled)})
        lines[0] += (f"; {n_points} points, {n_filled} filled "
                     f"({100 * n_filled / max(n_points, 1):.2g} %)")
        if n_filled > 0.2 * n_points:
            ctx.warn(f"{100 * n_filled / n_points:.0f} % of the points are filled gaps: "
                     "the spectrum above the gaps' rate is interpolation")
        for a, name in enumerate(axes):
            p = st.pooled(results, a, v.n_peaks, v.highpass_hz)
            for key, value in p.items():
                values[f"{name}_{key}"] = value
            lines.append(f"{name}: SD {p['sd_nm']:.3g} nm (above {v.highpass_hz:g} Hz), "
                         f"precision {p['precision_nm']:.3g} nm, vibration "
                         f"{p['vibration_nm']:.3g} nm RMS; largest peak "
                         f"{p['peak_nm']:.3g} nm at {p['peak_hz']:.4g} Hz")
        first = results[0]
        arrays["frequency_hz"] = first.axes[0].spectrum.f_hz
        for ax in first.axes:
            arrays[f"{ax.axis}_amplitude_nm"] = ax.spectrum.amplitude
            arrays[f"{ax.axis}_cumulative_rms_nm"] = st.cumulative_rms(ax.spectrum)

        peak_rows = data_peaks(results)
        if v.peak_table:
            lines.append("peaks (trace, axis, Hz, nm):")
            lines += [f"  {t}\t{a}\t{f:.4g}\t{amp:.3g}"
                      for t, a, f, amp in peak_rows[:MAX_DRAWN * len(axes) * v.n_peaks]]

        drawn = results[:MAX_DRAWN]
        more = (f" (first {MAX_DRAWN} of {len(results)})"
                if len(results) > MAX_DRAWN else "")
        if v.zoom:
            plots["trace, zoomed"] = row(lambda ax, a: draw_zoom(
                ax, drawn, a, v.zoom_points, v.smoothing, more))
        if v.whole:
            plots["whole trace"] = row(lambda ax, a: draw_whole(ax, drawn, a, more))
        if v.amplitude:
            plots["amplitude spectrum"] = row(lambda ax, a: draw_amplitude(
                ax, drawn, a, v.highpass_hz, more))
        if v.cumulative:
            plots["cumulative RMS"] = row(lambda ax, a: draw_cumulative(ax, drawn, a, more))
        if v.peak_table:
            plots["peaks"] = Plot(lambda ax: draw_peak_table(ax, peak_rows, MAX_DRAWN * 3),
                                  size=(4.8, 3.2))
        return results

    def _scatter(self, ctx, settings, traces, axes, dt_s, values, lines,
                 arrays, plots, row) -> list:
        """The random scatter: per trace ``(name, [stability.Scatter per
        axis])``, all on one set of m so they pool; [] where no trace has two
        measured points in a row."""
        from .. import stability as st
        r = settings.random
        ctx.report("stability: random scatter")
        # the time scale in points of the typical step, the same for every trace
        steps = [st.runs_of(t.t_s, np.isfinite(t.values).all(1) & ~t.filled,
                            r.max_gap)[1] for t in traces]
        step = float(np.nanmedian(steps)) if np.isfinite(steps).any() else dt_s
        also = []
        if r.time_scale_ms:
            also = [max(int(round(r.time_scale_ms / 1000.0 / step)), 1)]
        m = st.averaging_points(r.longest_points, also=also)
        scattered = [(t.name, st.scatter(t, axes, m, r.max_gap)) for t in traces]
        scattered = [(n, s) for n, s in scattered if s and s[0].count[0] > 0]
        if not scattered:
            return []
        lines.append(f"random scatter: diff/sqrt(2) of consecutive points, a typical "
                     f"step of {step * 1000:.4g} ms"
                     + (f"; at {also[0]} points = {also[0] * step * 1000:.4g} ms"
                        if also else ""))
        pooled = [st.pooled_scatter([s[a] for _, s in scattered])
                  for a in range(len(axes))]
        for p in pooled:
            ratio = p.scatter_nm / p.precision_nm if p.precision_nm > 0 else np.nan
            values[f"{p.axis}_scatter_nm"] = p.scatter_nm
            values[f"{p.axis}_scatter_robust_nm"] = p.robust_nm
            values[f"{p.axis}_scatter_ratio"] = float(ratio)
            values.setdefault(f"{p.axis}_precision_nm", p.precision_nm)
            text = (f"{p.axis}: random scatter {p.scatter_nm:.3g} nm (robust "
                    f"{p.robust_nm:.3g}), precision {p.precision_nm:.3g} nm, "
                    f"{ratio:.3g} x")
            if also:
                at = p.at(also[0])
                values[f"{p.axis}_scatter_at_nm"] = at
                text += (f"; over {also[0] * step * 1000:.4g} ms {at:.3g} nm, white "
                         f"noise {p.precision_nm / np.sqrt(also[0]):.3g}")
            lines.append(text)
            arrays[f"{p.axis}_allan_nm"] = p.allan_nm
        arrays["allan_tau_s"] = pooled[0].tau_s
        drawn = scattered[:MAX_DRAWN]
        more = (f" (first {MAX_DRAWN} of {len(scattered)})"
                if len(scattered) > MAX_DRAWN else "")
        mark = also[0] if also else None
        if r.allan:
            plots["Allan deviation"] = row(lambda ax, a: draw_allan(
                ax, [(n, s[a]) for n, s in drawn], pooled[a], mark, more))
        if r.steps:
            plots["steps"] = row(lambda ax, a: draw_steps(ax, pooled[a]))
        return scattered


def data_peaks(results) -> list:
    """``(trace, axis, Hz, nm)`` of every trace's peaks."""
    return [(r.trace.name, ax.axis, f, amp) for r in results
            for ax in r.axes for f, amp in ax.peaks]


def per_bead(results, axes, scattered=()):
    """``(bead ids, {value: one per bead}, groups)`` of the traces of *each
    bead*: per axis the SD, the precision, the vibration's RMS and the
    largest peak, named ``x_sd_nm`` ..., and the random scatter
    (``x_scatter_nm``, ``x_scatter_ratio``), grouped by axis.  A bead one
    approach has no number for is NaN there."""
    names = [r.trace.name for r in results]
    names += [n for n, _ in scattered if n not in names]
    ids = np.array([int(n) for n in names], dtype=np.int64)
    vib = {r.trace.name: r for r in results}
    sca = dict(scattered)
    n_points = [vib[n].run.stop - vib[n].run.start if n in vib
                else int(sca[n][0].count[0]) + 1 for n in names]
    values: Dict[str, np.ndarray] = {"n_points": np.array(n_points, np.int64)}
    groups: Dict[str, Dict[str, str]] = {}

    def column(get, n):
        try:
            return float(get(n))
        except (KeyError, IndexError):
            return np.nan

    for a, name in enumerate(axes):
        columns = {}
        if results:
            columns.update({
                "sd": ("sd_nm", lambda n: vib[n].axes[a].sd_nm),
                "precision": ("precision_nm", lambda n: vib[n].axes[a].precision_nm),
                "vibration": ("vibration_nm", lambda n: vib[n].axes[a].vibration_nm),
                "peak frequency": ("peak_hz", lambda n: vib[n].axes[a].peaks[0][0]),
                "peak amplitude": ("peak_nm", lambda n: vib[n].axes[a].peaks[0][1])})
        if scattered:
            if not results:
                columns["precision"] = ("precision_nm",
                                        lambda n: sca[n][a].precision_nm)
            columns.update({
                "random scatter": ("scatter_nm", lambda n: sca[n][a].scatter_nm),
                "scatter / precision": ("scatter_ratio", lambda n: (
                    sca[n][a].scatter_nm / sca[n][a].precision_nm))})
        groups[name] = {}
        for member, (suffix, get) in columns.items():
            key = f"{name}_{suffix}"
            with np.errstate(invalid="ignore", divide="ignore"):
                values[key] = np.array([column(get, n) for n in names], dtype=float)
            groups[name][member] = key
    return ids, values, groups


def _traces_label(key: str) -> str:
    return {"each": "each bead", "mean": "their mean",
            "pair": "each minus the first", "others": "each minus the others"}.get(key, key)


def _colour(i: int) -> str:
    return f"C{i % 10}"


def draw_zoom(ax, results, a: int, points: int, smoothing: int, more: str = "") -> None:
    """The first ``points`` of each trace, high-passed, and a running mean."""
    from ..stability import smooth
    for i, r in enumerate(results):
        x = r.axes[a].detrended[:points]
        t = (r.trace.t_s[r.run][:points] - r.trace.t_s[r.run][0]) * 1000.0
        ax.plot(t, x, lw=0.5, color=_colour(i), alpha=0.45)
        ax.plot(t, smooth(x, smoothing), lw=1.2, color=_colour(i), label=r.trace.name)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(f"{results[0].axes[a].axis} (nm)")
    ax.set_title(f"{results[0].axes[a].axis}, the first {points} points{more}",
                 fontsize=9)
    if 1 < len(results) <= MAX_DRAWN:
        ax.legend(fontsize=7)


def draw_whole(ax, results, a: int, more: str = "") -> None:
    """The whole of each trace, high-passed; the title gives the SD beside
    the precision."""
    for i, r in enumerate(results):
        ax.plot(r.trace.t_s[r.run], r.axes[a].detrended, lw=0.4, color=_colour(i),
                label=r.trace.name)
    x = results[0].axes[a]
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{x.axis} (nm)")
    ax.set_title(f"SD {x.sd_nm:.3g} nm, precision {x.precision_nm:.3g} nm{more}",
                 fontsize=9)


def draw_amplitude(ax, results, a: int, above_hz: float = 0.0, more: str = "") -> None:
    """The amplitude spectrum, its largest peak marked."""
    for i, r in enumerate(results):
        s = r.axes[a].spectrum
        ax.semilogy(s.f_hz[1:], s.amplitude[1:], lw=0.7, color=_colour(i),
                    label=r.trace.name)
    x = results[0].axes[a]
    if x.peaks:
        f, amp = x.peaks[0]
        ax.plot(f, amp, "x", color="k")
        title = f"{x.axis}: largest peak {amp:.3g} nm at {f:.4g} Hz"
    else:
        title = f"{x.axis}: no peak"
    if above_hz > 0:
        ax.axvline(above_hz, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude (nm)")
    ax.set_title(title + more, fontsize=9)


def draw_cumulative(ax, results, a: int, more: str = "") -> None:
    """The RMS of everything up to each frequency."""
    from ..stability import cumulative_rms
    for i, r in enumerate(results):
        s = r.axes[a].spectrum
        ax.plot(s.f_hz, cumulative_rms(s), lw=1, color=_colour(i), label=r.trace.name)
    x = results[0].axes[a]
    if np.isfinite(x.precision_nm):
        ax.axhline(x.precision_nm, color="0.5", lw=0.8, ls="--", label="precision")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("cumulative RMS (nm)")
    ax.set_ylim(0, None)
    ax.set_title(f"{x.axis}{more}", fontsize=9)
    ax.legend(fontsize=7)


def draw_peak_table(ax, rows, limit: int = 24) -> None:
    """The peaks as a table."""
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "no peaks above the floor", ha="center", va="center")
        return
    cells = [[t, a, f"{f:.4g}", f"{amp:.3g}"] for t, a, f, amp in rows[:limit]]
    table = ax.table(cellText=cells, colLabels=["trace", "axis", "Hz", "nm"],
                     loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    if len(rows) > limit:
        ax.set_title(f"the first {limit} of {len(rows)} peaks", fontsize=9)


def draw_allan(ax, traces, pooled, mark: Optional[int] = None, more: str = "") -> None:
    """The Allan deviation against the time averaged, per trace, with the
    white noise the precision predicts (precision / sqrt(n), dashed): where
    the curve leaves the line, something other than localization noise is
    there at that time scale."""
    for i, (name, s) in enumerate(traces):
        ok = s.count > 0
        ax.loglog(s.tau_s[ok], s.allan_nm[ok], ".-", lw=0.8, ms=3, color=_colour(i),
                  label=name)
    ok = pooled.count > 0
    if np.isfinite(pooled.precision_nm) and ok.any():
        ax.loglog(pooled.tau_s[ok], pooled.precision_nm / np.sqrt(pooled.m[ok]), "--",
                  color="0.4", lw=0.9, label="precision / \u221an")
    if mark is not None:
        ax.axvline(mark * pooled.step_s, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("time averaged (s)")
    ax.set_ylabel(f"Allan deviation, {pooled.axis} (nm)")
    ax.set_title(f"{pooled.axis}: {pooled.scatter_nm:.3g} nm at one point, precision "
                 f"{pooled.precision_nm:.3g}{more}", fontsize=9)
    if len(traces) <= MAX_DRAWN:
        ax.legend(fontsize=7)


def draw_steps(ax, pooled) -> None:
    """The steps diff(x)/sqrt(2) of every trace, with the Gaussian of the
    precision and of the measured scatter: a wider histogram than the
    precision's is noise the fit does not know about; heavy tails are jumps."""
    s = pooled.steps_nm
    if not len(s):
        return
    lim = 4 * max(pooled.robust_nm, pooled.precision_nm
                  if np.isfinite(pooled.precision_nm) else 0, 1e-9)
    counts, edges, _ = ax.hist(s, bins=80, range=(-lim, lim), color="0.7")
    centres = 0.5 * (edges[1:] + edges[:-1])
    area = len(s) * (edges[1] - edges[0])
    for sigma, style, label in ((pooled.precision_nm, "--", "precision"),
                                (pooled.scatter_nm, "-", "scatter")):
        if np.isfinite(sigma) and sigma > 0:
            ax.plot(centres, area * np.exp(-0.5 * (centres / sigma) ** 2)
                    / (sigma * np.sqrt(2 * np.pi)), style, lw=1, label=f"{label} {sigma:.3g} nm")
    ax.set_xlabel(f"diff({pooled.axis}) / \u221a2 (nm)")
    ax.set_ylabel("steps")
    ax.legend(fontsize=7)

