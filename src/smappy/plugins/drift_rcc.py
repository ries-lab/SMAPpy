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
    Settings = RCCSettings

    def run(self, ctx: Context, settings: RCCSettings) -> Result:
        ctx.selection.require(5000, ctx.warn, "a drift estimate")
        ctx.report(f"correlating {ctx.selection}")
        drift = estimate_drift_rcc(ctx.locs, settings, select=ctx.selection.mask)
        return Result(locs=drift.apply(ctx.locs), text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
