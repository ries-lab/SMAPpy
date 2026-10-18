"""COMET drift correction as a plugin.  The work is in :mod:`smappy.drift`."""
from __future__ import annotations

from typing import Optional

from dataclasses import replace

from ..drift import (DriftSettings, correct_drift, estimate_cost, merge_from_session,
                     merge_values)
from . import (Context, FormState, ParamInfo, Plugin, PreflightChoice,
               PreflightQuestion, Result, register)
from .drift_result import KeepsDrift


@register("Analysis/Drift/COMET")
class CometDrift(KeepsDrift, Plugin):
    description = ("Estimate drift from the selected localizations with COMET "
                   "and subtract it from all of them.")
    changes_locs = True
    # the correction moves every localization, and the estimate reads frames
    grouping = "ungrouped"
    Settings = DriftSettings
    name = "COMET drift correction"
    # 2: "RCC first" and "two stage" run under the spline fit (the default)
    # rather than being skipped by it.  3: a transformation besides the shift
    # (off by default: the shift alone gives the numbers it gave)
    version = "3"

    # Longer than this and the run is worth agreeing to first.  Five minutes
    # is where waiting stops being waiting and starts being a decision.
    slow_seconds = 300.0

    # What COMET is left to find once RCC has been over the data.  RCC lands
    # within a pixel of its rendering grid, which is 15 nm by default; 50 nm is
    # comfortably more than that and small enough that the pair count -- which
    # is what the wait is made of -- falls by more than an order of magnitude.
    prepass_max_drift_nm = 50.0

    def preflight(self, ctx: Context, settings: DriftSettings):
        """What this is going to cost, and a question if that is a lot.

        Time and memory both follow from the neighbour-pair count, which grows
        with the square of the localization count and steeply with the search
        radius -- so a dataset twice the size of the one the defaults were
        chosen on is not twice the wait, and the difference between a minute
        and an afternoon is not visible in the settings.

        When it is a lot, "no" is not the only useful answer, so the offer is
        made here rather than left to be found in the settings: RCC first over
        a few windows, which costs seconds and does not care how far it looks,
        and then COMET over the tens of nanometres it leaves behind.
        """
        settings = merge_from_session(settings, ctx.session)
        cost = estimate_cost(ctx.locs, settings, select=ctx.selection.mask)
        ctx.report(str(cost))
        question = cost.question(self.slow_seconds)
        if not question or settings.rcc_prepass:
            return question
        cheap = replace(settings, rcc_prepass=True,
                        rcc_prepass_max_drift_nm=settings.max_drift_nm,
                        max_drift_nm=self.prepass_max_drift_nm,
                        initial_sigma_nm=None)
        after = estimate_cost(ctx.locs, cheap, select=ctx.selection.mask)
        return PreflightQuestion(
            text=question,
            choices=[PreflightChoice(
                label=f"RCC first, then COMET within "
                      f"{self.prepass_max_drift_nm:g} nm",
                settings=cheap,
                help=f"RCC over {cheap.rcc_prepass_windows} time windows takes "
                     f"out the bulk of the drift in seconds; COMET then refines "
                     f"what is left, which is a search radius of "
                     f"{self.prepass_max_drift_nm:g} nm instead of "
                     f"{settings.max_drift_nm:g}.  Then: {after}")])

    def describe(self, ctx, settings) -> FormState:
        """What *within* and *dark frames* on auto take: the session's merge
        settings; and the transformation's knots are read only when there is
        a transformation, which only the spline fit has."""
        dx, dt = merge_values(ctx.session if ctx is not None else None)
        return FormState(
            hints={"group_dx_nm": dx, "group_dt": dt,
                   "transformation_knot_frames": settings.spline_knot_frames},
            active={"transformation": settings.spline,
                    "transformation_knot_frames":
                        settings.spline and settings.transformation != "shift"})

    def run(self, ctx: Context, settings: DriftSettings) -> Result:
        settings = merge_from_session(settings, ctx.session)
        ctx.selection.require(5000, ctx.warn, "a drift estimate")
        ctx.report(f"estimating drift from {ctx.selection}")
        corrected, drift = correct_drift(ctx.locs, settings,
                                         select=ctx.selection.mask,
                                         progress=ctx.report, stop=ctx.stop)
        return Result(locs=corrected, text=str(drift), plot=drift.figure(),
                      data={"drift": drift}, settings=settings)
