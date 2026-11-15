"""*Quality filter*: MINFLUX's cfr and efo cuts found in the data and set
as ordinary filter bounds on the layers -- visible and editable in the
filter panel afterwards, recorded in a chain -- rather than a preset of
fixed numbers or one combined flag.  The maths is `smappy.quality`.

The thresholds are measured on the localizations the layer shows, without
the bounds on ``cfr`` and ``efo`` themselves: run again, it finds the same
cut rather than tightening its own.

**cfr measured once a trace.**  A 2D sequence measures cfr once a trace,
in an iteration before the localizations it then applies to, so most
localizations have none, and a filter range leaves out NaN (`filter`).
Where fewer than `MEASURED_FOR_BOUND` of the localizations have a cfr, the
cut is the trace's: each trace's cfr is the mean of its measured values,
NaN left out -- what merging a trace gives too -- written onto all its
localizations as ``trace_cfr``, the threshold found over the traces (one
value each, so a long trace does not count more), and the bound set on
``trace_cfr``.  A trace is then kept or dropped whole, on the localizations
as on the merged traces, with no rule about NaN in the filter.  A trace
whose cfr was never measured has none, and the bound drops it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register

#: the share of localizations with a measured cfr above which a bound is set
MEASURED_FOR_BOUND = 0.5


@dataclass
class QualityFilterSettings:
    cfr: bool = param(True, label="cfr", help="an upper bound at the median plus k robust "
                                              "sigmas")
    k: float = param(2.0, label="k", unit="sigma", min=0.5, max=10.0, beside=True,
                     help="robust standard deviations above the median")
    efo: bool = param(True, label="efo", help="an upper bound at the valley before the "
                                              "two-molecule peak")


@register("Analysis/MINFLUX/Quality filter")
class QualityFilter(Plugin):
    """Sets MINFLUX's cfr and efo cuts from the data: cfr below its median
    plus k robust sigmas, efo below the valley before the two-molecule peak,
    as ordinary filter bounds on the layers."""

    Settings = QualityFilterSettings
    # writes trace_cfr where cfr is measured once a trace
    changes_locs = True
    grouping = "ungrouped"
    # 2: cfr measured once a trace cut per trace, on trace_cfr
    version = "2"

    def changes_locs_for(self, settings: QualityFilterSettings) -> bool:
        return bool(settings.cfr)

    def describe(self, ctx: Optional[Context], settings: QualityFilterSettings) -> FormState:
        state = FormState()
        state.active["k"] = settings.cfr
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            state.active["cfr"] = "cfr" in ctx.locs
            state.active["efo"] = "efo" in ctx.locs
        return state

    def run(self, ctx: Context, settings: QualityFilterSettings) -> Result:
        from .. import quality as Q
        locs = ctx.locs
        wanted = [c for c, on in (("cfr", settings.cfr), ("efo", settings.efo))
                  if on and c in locs]
        if not wanted:
            raise ValueError("no cfr or efo to filter on: the table has "
                             + ", ".join(sorted(locs.keys())))
        mask = self._shown(ctx, wanted + ["trace_cfr"])
        bounds, lines, found = {}, [], {}
        out_locs = None
        if "cfr" in wanted:
            cfr = np.asarray(locs["cfr"], np.float64)[mask]
            upper = Q.robust_upper(cfr, settings.k)
            if np.isfinite(upper):
                kept = np.isfinite(cfr) & (cfr <= upper)
                measured = int(np.isfinite(cfr).sum())
                share = measured / max(len(cfr), 1)
                line = (f"cfr ≤ {upper:.3f} (median + {settings.k:g} robust sigmas): "
                        f"{100 * kept.sum() / max(measured, 1):.1f} % of the "
                        f"{measured} measured kept")
                if share >= MEASURED_FOR_BOUND or "tid" not in locs:
                    bounds["cfr"] = (None, upper)
                    lines.append(line)
                    found["cfr"] = (cfr, upper)
                else:
                    per_trace, values = trace_cfr(locs["tid"], locs["cfr"], mask)
                    upper = Q.robust_upper(values, settings.k)
                    from ..locs import Localizations
                    out_locs = Localizations({**locs.columns, "trace_cfr": per_trace},
                                             dict(locs.metadata))
                    bounds["trace_cfr"] = (None, upper)
                    kept = int((values <= upper).sum())
                    lines.append(f"cfr is measured for {100 * share:.0f} % of the "
                                 f"localizations (once a trace): cut per trace, trace_cfr ≤ "
                                 f"{upper:.3f} (median + {settings.k:g} robust sigmas over "
                                 f"{len(values)} traces): {kept} traces kept")
                    found["cfr"] = (values, upper)
        if "efo" in wanted:
            efo = np.asarray(locs["efo"], np.float64)[mask]
            v = Q.efo_valley(efo)
            if np.isfinite(v.bound):
                bounds["efo"] = (None, v.bound)
                kept = (efo <= v.bound).sum()
                how = ("the valley before the two-molecule peak" if v.valley
                       else "no second peak: where the tail falls to 5 %")
                lines.append(f"efo ≤ {v.bound / 1e3:.1f} kHz ({how}; one molecule at "
                             f"{v.peak / 1e3:.1f} kHz): {100 * kept / max(len(efo), 1):.1f} % kept")
                found["efo"] = v
        if not found:
            raise ValueError("too few values to find a threshold in")

        def draw(figure):
            axes = np.atleast_1d(figure.subplots(1, len(found)))
            for ax, (name, what) in zip(axes, found.items()):
                if name == "cfr":
                    v, upper = what
                    v = v[np.isfinite(v)]
                    ax.hist(v, bins=80, range=(min(0.0, v.min()), max(1.2, upper * 1.2)),
                            color="0.35")
                    ax.axvline(upper, color="tab:red")
                    ax.set_xlabel("cfr" if "cfr" in bounds else "cfr per trace")
                else:
                    ax.plot(what.centres / 1e3, what.counts, color="0.2")
                    ax.axvline(what.bound / 1e3, color="tab:red")
                    ax.axvline(what.peak / 1e3, color="0.6", ls=":")
                    ax.set_xlabel("efo (kHz)")
                ax.set_ylabel("localizations")

        return Result(locs=out_locs, text="\n".join(lines), bounds=bounds,
                      plot=Plot(draw, panels=len(found), size=(4.0 * len(found), 3.0)),
                      data={"bounds": bounds}, settings=settings, title="quality filter")

    @staticmethod
    def _shown(ctx: Context, fields) -> np.ndarray:
        fields = [f for f in fields if f in ctx.locs]
        return QualityFilter._mask(ctx, fields)

    @staticmethod
    def _mask(ctx: Context, fields) -> np.ndarray:
        """What the layer shows, without its bounds on ``fields``."""
        if ctx.session is not None:
            i = ctx.layer
            sel = ctx.session.layers[i].selection(i, tuple(fields))
            mask = np.asarray(sel.mask, bool)
            if len(mask) == len(ctx.locs):
                return mask
        return np.asarray(ctx.selection.mask, bool)


def trace_cfr(tid, cfr, mask=None):
    """Each localization's trace's cfr -- the mean of the trace's measured
    values, NaN left out, as merging the trace gives -- and one value per
    trace that has one, among the rows ``mask`` keeps."""
    tid = np.asarray(tid, np.int64)
    cfr = np.asarray(cfr, np.float64)
    use = np.isfinite(cfr) & (tid >= 0)
    if mask is not None:
        use &= np.asarray(mask, bool)
    ids, inverse = np.unique(tid, return_inverse=True)
    inverse = inverse.reshape(-1)
    total = np.bincount(inverse, weights=np.where(use, cfr, 0.0), minlength=len(ids))
    count = np.bincount(inverse, weights=use.astype(float), minlength=len(ids))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    per_row = np.where(tid >= 0, mean[inverse], np.nan).astype(np.float32)
    return per_row, mean[count > 0]
