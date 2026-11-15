"""*Quality filter*: MINFLUX's cfr and efo cuts found in the data and set
as ordinary filter bounds on the layers -- visible and editable in the
filter panel afterwards, recorded in a chain -- rather than a preset of
fixed numbers or one combined flag.  The maths is `smappy.quality`.

The thresholds are measured on the localizations the layer shows, without
the bounds on ``cfr`` and ``efo`` themselves: run again, it finds the same
cut rather than tightening its own.

**cfr measured once a trace.**  A filter range leaves out NaN (`filter`),
and a 2D sequence measures cfr once a trace, so most localizations have
none: a bound would drop them all.  Where fewer than `MEASURED_FOR_BOUND`
of the localizations have a cfr, the threshold is reported and no bound
set; the cut then belongs to the trace (*Trace statistics*' *cut after cfr
above*).
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
    changes_locs = False
    grouping = "ungrouped"
    version = "1"

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
        mask = self._shown(ctx, wanted)
        bounds, lines, found = {}, [], {}
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
                if share >= MEASURED_FOR_BOUND:
                    bounds["cfr"] = (None, upper)
                else:
                    line += (f"; not set as a bound: cfr is measured for "
                             f"{100 * share:.0f} % of the localizations (once a trace), "
                             "and a bound would drop the rest -- cut the traces with "
                             f"Trace statistics' 'cut after cfr above' {upper:.2f}")
                lines.append(line)
                found["cfr"] = (cfr, upper)
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
                    ax.set_xlabel("cfr")
                else:
                    ax.plot(what.centres / 1e3, what.counts, color="0.2")
                    ax.axvline(what.bound / 1e3, color="tab:red")
                    ax.axvline(what.peak / 1e3, color="0.6", ls=":")
                    ax.set_xlabel("efo (kHz)")
                ax.set_ylabel("localizations")

        return Result(text="\n".join(lines), bounds=bounds,
                      plot=Plot(draw, panels=len(found), size=(4.0 * len(found), 3.0)),
                      data={"bounds": bounds}, settings=settings, title="quality filter")

    @staticmethod
    def _shown(ctx: Context, fields) -> np.ndarray:
        """What the layer shows, without its bounds on ``fields``."""
        if ctx.session is not None:
            i = ctx.layer
            sel = ctx.session.layers[i].selection(i, tuple(fields))
            mask = np.asarray(sel.mask, bool)
            if len(mask) == len(ctx.locs):
                return mask
        return np.asarray(ctx.selection.mask, bool)
