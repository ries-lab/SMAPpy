"""Redundant cross-correlation drift correction, the classic alternative to COMET."""
from __future__ import annotations

from ..drift import merge_from_session, merge_values
from ..rcc import RCCSettings, estimate_drift_rcc
from . import Context, FormState, ParamInfo, Plugin, Result, register
from .drift_result import KeepsDrift


@register("Analysis/Drift/RCC")
class RCCDrift(KeepsDrift, Plugin):
    description = ("Estimate drift by cross-correlating rendered time windows "
                   "(RCC) and subtract it from all localizations.")
    changes_locs = True
    # the correction moves every localization, and the estimate reads frames
    grouping = "ungrouped"
    Settings = RCCSettings
    name = "RCC drift correction"

    def describe(self, ctx, settings) -> FormState:
        """What *within* and *dark frames* on auto take: the session's merge
        settings."""
        dx, dt = merge_values(ctx.session if ctx is not None else None)
        return FormState(hints={"group_dx_nm": dx, "group_dt": dt})

    def run(self, ctx: Context, settings: RCCSettings) -> Result:
        settings = merge_from_session(settings, ctx.session)
        ctx.selection.require(5000, ctx.warn, "a drift estimate")
        ctx.report(f"correlating {ctx.selection}")
        # Stop ends it between time windows with `Stopped`: half the pairs
        # make no drift curve, so the table is left as it was
        drift = estimate_drift_rcc(ctx.locs, settings, select=ctx.selection.mask,
                                   stop=ctx.stop)
        return Result(locs=drift.apply(ctx.locs), text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
