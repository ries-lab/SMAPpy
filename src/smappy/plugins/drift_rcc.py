"""Redundant cross-correlation drift correction, the classic alternative to COMET."""
from __future__ import annotations

from ..rcc import RCCSettings, estimate_drift_rcc
from . import Context, ParamInfo, Plugin, Result, register
from .drift_result import KeepsDrift


@register("Analysis/Drift/RCC")
class RCCDrift(KeepsDrift, Plugin):
    description = ("Estimate drift by cross-correlating rendered time windows "
                   "(RCC) and subtract it from all localizations.")
    changes_locs = True
    # the correction moves every localization, and the estimate reads frames
    grouping = "ungrouped"
    Settings = RCCSettings

    def run(self, ctx: Context, settings: RCCSettings) -> Result:
        ctx.selection.require(5000, ctx.warn, "a drift estimate")
        ctx.report(f"correlating {ctx.selection}")
        # Stop ends it between time windows with `Stopped`: half the pairs
        # make no drift curve, so the table is left as it was
        drift = estimate_drift_rcc(ctx.locs, settings, select=ctx.selection.mask,
                                   stop=ctx.stop)
        return Result(locs=drift.apply(ctx.locs), text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
