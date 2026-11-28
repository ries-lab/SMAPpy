"""*Sequence QC*: how a MINFLUX sequence ran, from every iteration of every
attempt -- abberior's ParaFlux plots, on a table read with *every
iteration*.  The maths is `smappy.minflux_qc`; this draws it and hands back
the numbers a batch compares.

It reads **every row of the files the layer shows**, not the layer's
filter: the layers of a table read with every iteration start filtered to
the localizations (``final`` = 1), and the attempts that failed have no
position for a filter or ROI to keep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Notice, Plot, Plugin, Result, param, register, settings_values

#: what a table needs to be read with every iteration
NEEDS = ("iteration", "final", "sta", "tid")
LABEL = "Sequence QC"
PARAMETERS = ("photons", "efo", "cfr", "fbg", "xy_err_nm", "ecc", "efc")


@dataclass
class SequenceQCSettings:
    parameter: str = param("photons", label="per iteration",
                           choices=[(p, p) for p in PARAMETERS],
                           help="the attribute shown per iteration and against the "
                                "localization's number in its trace")
    first_localizations: int = param(25, label="first localizations", min=2, max=500,
                                     help="how many of each trace's localizations "
                                          "the last panel follows")


@register("Analysis/MINFLUX/Sequence QC")
class SequenceQC(Plugin):
    """How the MINFLUX sequence ran: the attempts passed and lost per
    iteration and why, the background, the catching offsets, localizations
    and photons per burst, the time per iteration -- abberior's ParaFlux
    plots, from a table read with every iteration."""

    Settings = SequenceQCSettings
    changes_locs = False
    grouping = "ungrouped"
    version = "2"

    def describe(self, ctx: Optional[Context], settings: SequenceQCSettings) -> FormState:
        state = FormState()
        if ctx is None or ctx.session is None or not len(ctx.locs):
            return state
        missing = [c for c in NEEDS if c not in ctx.locs]
        if missing:
            state.notice = Notice("Sequence QC needs every iteration: open the MINFLUX "
                                  "file again with 'every iteration' ticked")
        return state

    def run(self, ctx: Context, settings: SequenceQCSettings) -> Result:
        from .. import minflux_qc as q
        locs = ctx.locs
        missing = [c for c in NEEDS if c not in locs]
        if missing:
            raise ValueError("Sequence QC needs a table read with every iteration "
                             "(File > Open, 'every iteration'): this one has no "
                             + ", ".join(missing))
        rows = shown_files(ctx)
        a = {c: np.asarray(locs[c])[rows] for c in locs.keys()
             if np.asarray(locs[c]).ndim == 1}
        get = lambda c: a.get(c)                                     # noqa: E731
        final = get("final").astype(bool)
        events = q.iteration_events(get("iteration"), get("sta"), final, get("vld"))
        background = q.per_iteration(get("iteration"), get("fbg")) if "fbg" in a else \
            {"iterations": np.zeros(0), "median": np.zeros(0)}
        times = q.time_per_iteration(get("iteration"), get("time_s")) if "time_s" in a \
            else {"iterations": np.zeros(0), "fraction": np.zeros(0), "seconds": np.zeros(0),
                  "scouting_s": np.zeros(0)}
        traces = q.traces(get("tid"), get("sta"), final, get("time_s"), get("thi"),
                          get("photons"), get("ecc"))
        axes = [c for c in ("x_nm", "y_nm", "z_nm") if c in a]
        offsets = q.catching_offsets(get("iteration"), final, get("tid"),
                                     get("trace_index"), np.column_stack([a[c] for c in axes]))
        parameter = settings.parameter if settings.parameter in a else "photons"
        values = q.per_iteration(get("iteration"), a[parameter]) if parameter in a else None
        summary = q.summary(events, traces, background, times)
        by_index = index_profile(get("trace_index"), final, a.get(parameter),
                                 settings.first_localizations)
        efo_cfr = (a["efo"][final], a["cfr"][final]) if "efo" in a and "cfr" in a else None

        lines = [f"{int(summary['scouting_events'])} attempts, {int(summary['localizations'])} "
                 f"localizations in {int(summary['bursts'])} bursts "
                 f"({summary['locs_per_burst']:.1f} a burst, median "
                 f"{summary['photons_per_burst']:.0f} photons)"]
        passed = events.passed_fraction()
        if passed:
            lines.append("passed, of the attempts: " + ", ".join(
                f"itr {k} {100 * f:.1f} %" for k, f in passed.items()))
        lines.append("bursts ended: " + ", ".join(
            f"{c} {100 * summary['ended_' + c.replace(' ', '_')]:.0f} %"
            for c in q.CATEGORIES if summary.get("ended_" + c.replace(" ", "_"), 0) > 0))
        if "scouting_time_fraction" in summary:
            lines.append(f"scouting took {100 * summary['scouting_time_fraction']:.0f} % "
                         "of the time")
        id_regions = self._trace_table(ctx, traces, settings)

        def draw(figure):
            grid = figure.subplots(3, 3).ravel()
            draw_events(grid[0], events)
            draw_background(grid[1], background)
            draw_offsets(grid[2], offsets, axes)
            draw_bursts(grid[3], traces)
            draw_photons(grid[4], traces)
            draw_parameter(grid[5], values, parameter)
            draw_efo_cfr(grid[6], efo_cfr)
            draw_times(grid[7], times)
            draw_profile(grid[8], by_index, parameter)

        return Result(text="\n".join(lines),
                      plot=Plot(draw, panels=9, size=(11.5, 9.5)),
                      data={"summary": summary, "events": events, "offsets": offsets},
                      id_regions=id_regions, settings=settings, title="sequence QC")

    def _trace_table(self, ctx: Context, traces, settings):
        """Per trace, as a region table of ``tid``: its rows, localizations,
        photons, duration and why it ended -- what the region filter and the
        summaries read."""
        from ..region_runs import column_table
        if not len(traces.ids):
            return {}
        values = {"n_rows": traces.rows.astype(float),
                  "n_locs": traces.localizations.astype(float),
                  "photon_budget": traces.photons,
                  "duration_s": traces.end_s - traces.start_s,
                  "end_sta": traces.sta.astype(float)}
        ids, first = np.unique(traces.ids, return_index=True)
        values = {k: v[first] for k, v in values.items()}
        step = {"label": LABEL, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        table = column_table(ctx.locs, "tid", LABEL, ids, values, step,
                             layer=ctx.layer, session=ctx.session)
        return {"tid": {LABEL: table}}


def shown_files(ctx: Context) -> np.ndarray:
    """Every row of the files (and channels) the layer shows, whatever its
    filter: a mask over ``ctx.locs``."""
    locs = ctx.locs
    n = len(locs)
    mask = np.asarray(ctx.selection.mask, bool) if ctx.selection is not None else np.ones(n, bool)
    if "filenumber" not in locs:
        return np.ones(n, bool)
    files = np.asarray(locs["filenumber"])
    shown = np.unique(files[mask]) if mask.any() else np.unique(files)
    return np.isin(files, shown)


def index_profile(index, final, values, first: int):
    """The median and quartiles of ``values`` over the localizations, by
    their number in the trace, 1 .. ``first``."""
    if values is None or index is None:
        return None
    index = np.asarray(index)
    final = np.asarray(final, bool)
    values = np.asarray(values, np.float64)
    out = {"index": np.arange(1, first + 1), "median": np.full(first, np.nan),
           "q25": np.full(first, np.nan), "q75": np.full(first, np.nan)}
    for j in range(first):
        v = values[final & (index == j + 1)]
        v = v[np.isfinite(v)]
        if len(v):
            out["median"][j] = np.median(v)
            out["q25"][j], out["q75"][j] = np.quantile(v, [0.25, 0.75])
    return out


def _whole():
    """Ticks on whole iterations only."""
    from matplotlib.ticker import MaxNLocator
    return MaxNLocator(integer=True)


COLOURS = {"passed": "#4c72b0", "cfr": "#55a868", "dark": "0.45",
           "out of range": "#c44e52", "other": "#dd8452"}


def draw_events(ax, events) -> None:
    """ParaFlux's iteration events: of the scouting events, the fraction
    that passed each iteration and why the rest ended."""
    from ..minflux_qc import CATEGORIES
    catching = np.isfinite(events.fraction["passed"])
    its = events.iterations[catching]
    # passed at the bottom, so its share reads off the axis, the losses above
    bottom = np.zeros(len(its))
    for c in CATEGORIES:
        f = 100 * events.fraction[c][catching]
        if np.any(f > 0):
            ax.bar(its, f, bottom=bottom, color=COLOURS[c], label=c, width=0.7)
            bottom += f
    for k, f in zip(its, 100 * events.fraction["passed"][catching]):
        ax.text(k, f, f"{f:.1f} %", ha="center", va="bottom", fontsize=7)
    if events.final is not None:
        k = list(events.iterations).index(events.final)
        ok = events.counts["passed"][k] / max(events.reached[k], 1)
        ax.set_title(f"attempts per iteration; final itr {events.final}: "
                     f"{100 * ok:.0f} % valid", fontsize=8)
    ax.xaxis.set_major_locator(_whole())
    ax.set_xlabel("iteration")
    ax.set_ylabel("% of the attempts")
    ax.legend(fontsize=6, frameon=False)


def draw_background(ax, background) -> None:
    its, med = background["iterations"], background["median"]
    if not len(its) or not np.any(np.nan_to_num(med) > 0):
        ax.text(0.5, 0.5, "no background measured\n(background sensing off)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
    else:
        ax.errorbar(its, med / 1e3, yerr=[(med - background["q25"]) / 1e3,
                                          (background["q75"] - med) / 1e3],
                    fmt="o", color="0.2", capsize=2)
    ax.xaxis.set_major_locator(_whole())
    ax.set_xlabel("iteration")
    ax.set_ylabel("fbg (kHz)")
    ax.set_title("background per iteration", fontsize=8)


def draw_offsets(ax, offsets, axes) -> None:
    its = offsets["iterations"]
    for j, name in enumerate(axes):
        if len(its):
            ax.errorbar(its + 0.1 * (j - 1), offsets["mean"][:, j], yerr=offsets["std"][:, j],
                        fmt="o", capsize=2, label=name.split("_")[0])
    ax.axhline(0, color="0.7", lw=0.6)
    ax.xaxis.set_major_locator(_whole())
    ax.set_xlabel("catching iteration")
    ax.set_ylabel("offset from the localization (nm)")
    ax.set_title("catching offsets, mean ± std", fontsize=8)
    if len(its):
        ax.legend(fontsize=6, frameon=False)


def draw_bursts(ax, traces) -> None:
    from ..minflux_qc import exponential_mean
    n = traces.localizations[traces.valid()]
    if len(n):
        hi = np.quantile(n, 0.98) if len(n) > 20 else n.max()
        ax.hist(n[n <= hi], bins=min(50, max(5, int(hi))), color="0.35")
    ax.set_xlabel("localizations per burst")
    ax.set_ylabel("bursts")
    ax.set_title(f"mean {exponential_mean(n):.1f}" if len(n) else "no bursts", fontsize=8)


def draw_photons(ax, traces) -> None:
    p = traces.photons[traces.valid()]
    p = p[p > 0]
    if len(p):
        ax.hist(np.log10(p), bins=40, color="0.35")
        ax.set_title(f"photons per burst, median {np.median(p):.0f}", fontsize=8)
    ax.set_xlabel("log10 photons per burst (every iteration)")
    ax.set_ylabel("bursts")


def draw_parameter(ax, values, name) -> None:
    if values is not None and len(values["iterations"]):
        its = values["iterations"]
        ax.errorbar(its, values["median"], yerr=[values["median"] - values["q25"],
                                                 values["q75"] - values["median"]],
                    fmt="o", color="0.2", capsize=2)
    ax.xaxis.set_major_locator(_whole())
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"{name}, median and quartiles")
    ax.set_title(f"{name} per iteration", fontsize=8)


def draw_efo_cfr(ax, efo_cfr) -> None:
    if efo_cfr is not None:
        efo, cfr = (np.asarray(v, np.float64) for v in efo_cfr)
        ok = np.isfinite(efo) & np.isfinite(cfr)
        if ok.sum() > 2000:
            ax.hist2d(cfr[ok], efo[ok] / 1e3, bins=60, cmap="Greys",
                      range=[[min(0, np.quantile(cfr[ok], 0.01)), np.quantile(cfr[ok], 0.99)],
                             [0, np.quantile(efo[ok], 0.99) / 1e3]])
        elif ok.any():          # cfr measured once a trace (2D sequences): a scatter
            ax.plot(cfr[ok], efo[ok] / 1e3, ".", color="0.3", ms=3)
        if ok.any():
            ax.set_title(f"efo against cfr ({ok.sum()} measured), cfr median "
                         f"{np.median(cfr[ok]):.2f}", fontsize=8)
    ax.set_xlabel("cfr")
    ax.set_ylabel("efo (kHz)")


def draw_times(ax, times) -> None:
    its = times["iterations"]
    if len(its):
        ax.bar(its, 100 * times["fraction"], color="0.35", width=0.7)
    ax.xaxis.set_major_locator(_whole())
    ax.set_xlabel("iteration")
    ax.set_ylabel("% of the time")
    ax.set_title("time per iteration", fontsize=8)


def draw_profile(ax, profile, name) -> None:
    if profile is not None:
        i = profile["index"]
        ax.fill_between(i, profile["q25"], profile["q75"], color="0.85", lw=0)
        ax.plot(i, profile["median"], "o-", color="0.2", ms=3)
    ax.set_xlabel("localization in its trace")
    ax.set_ylabel(name)
    ax.set_title(f"{name} along the trace's start", fontsize=8)
