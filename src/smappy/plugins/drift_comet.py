"""COMET drift correction as a plugin.  The work is in :mod:`smappy.drift`."""
from __future__ import annotations

from typing import Optional

from dataclasses import replace

from ..drift import DriftSettings, correct_drift, estimate_cost
from . import (Context, ParamInfo, Plugin, PreflightChoice, PreflightQuestion,
               Result, register)
from .drift_result import KeepsDrift


@register("Analysis/Drift/COMET")
class CometDrift(KeepsDrift, Plugin):
    description = ("Estimate drift from the selected localizations with COMET "
                   "and subtract it from all of them.")
    Settings = DriftSettings
    main = ("segmentation_var", "max_drift_nm", "target_sigma_nm",
            "spline", "spline_knot_frames", "group", "use_z")
    params = {
        "segmentation_mode": ParamInfo(label="window unit",
                                       choices=((2, "frames"), (1, "localizations"),
                                                (0, "windows")),
                                       help="what window size counts; time-window "
                                            "fit only"),
        "segmentation_var": ParamInfo(label="window size", min=1,
                                      help="frames or localizations per time "
                                           "window, or the number of windows; "
                                           "time-window fit only"),
        "max_drift_nm": ParamInfo(label="max drift", unit="nm", min=1,
                                  help="the largest drift expected; also the "
                                       "radius within which localizations are "
                                       "paired"),
        "target_sigma_nm": ParamInfo(label="target sigma", unit="nm", min=0.1,
                                     help="the finest width of the overlap "
                                          "Gaussian, where refinement stops"),
        "initial_sigma_nm": ParamInfo(label="initial sigma", unit="nm",
                                      help="the width the refinement starts "
                                           "from; auto: a third of max drift"),
        "boxcar_width": ParamInfo(label="smoothing", unit="windows", min=1,
                                  help="running mean over this many windows "
                                       "between refinement steps; 1 is none; "
                                       "time-window fit only"),
        "interpolation": ParamInfo(choices=("cubic", "catmull-rom"),
                                   help="the curve from the window estimates to "
                                        "every frame; time-window fit only"),
        "max_locs_per_segment": ParamInfo(label="max locs / window", min=1,
                                          help="a random subset of at most this "
                                               "many per window; auto: all; "
                                               "time-window fit only"),
        "optimizer_ftol": ParamInfo(label="ftol",
                                    help="relative change of the cost at which "
                                         "the optimizer stops"),
        "optimizer_ftol_coarse": ParamInfo(label="ftol coarse",
                                           help="ftol for the steps above the "
                                                "target sigma; auto: ftol; "
                                                "time-window fit only"),
        "approximate_kernel": ParamInfo(label="approximate kernel",
                                        help="skip pairs more than 6 sigma apart; "
                                             "the spline fit always does"),
        "quality_control": ParamInfo(label="quality control",
                                     help="discard time windows whose fit does "
                                          "not beat no correction; needs fit "
                                          "spline off"),
        "min_lift": ParamInfo(label="min lift",
                              help="how much a window's fit must improve its "
                                   "overlap over no correction to be kept"),
        "group": ParamInfo(label="group blinks",
                           help="link the localizations of one blink into one "
                                "before estimating"),
        "group_dx_nm": ParamInfo(label="group radius", unit="nm", min=0,
                                 help="how close localizations in consecutive "
                                      "frames must be to be linked"),
        "group_dt": ParamInfo(label="group gap", unit="frames", min=0,
                              help="how many dark frames a blink may skip"),
        "two_stage": ParamInfo(label="two stage",
                               help="a grouped pass, then an ungrouped one "
                                    "within fine radius"),
        "two_stage_radius_nm": ParamInfo(label="fine radius", unit="nm", min=0,
                                         help="the max drift of the second, "
                                              "ungrouped pass"),
        "rcc_prepass": ParamInfo(label="RCC first",
                                 help="correlate rendered time windows to take "
                                      "out the bulk of the drift, then run "
                                      "COMET over a small max drift"),
        "rcc_prepass_windows": ParamInfo(label="RCC windows", min=2,
                                         help="the time windows RCC correlates"),
        "rcc_prepass_max_drift_nm": ParamInfo(label="RCC max drift", unit="nm",
                                              help="auto: RCC's own default"),
        "spline": ParamInfo(label="fit spline",
                            help="fit the drift as a smooth curve in time rather "
                                 "than one value per time window"),
        "spline_knot_frames": ParamInfo(label="knot spacing", unit="frames", min=10,
                                        help="frames between the spline's "
                                             "coefficients"),
        "spline_penalty": ParamInfo(label="spline penalty", min=0,
                                    help="extra penalty on the curve's bending; "
                                         "0 is none"),
        "use_z": ParamInfo(label="correct z", help="auto: if the table has z"),
        "backend": ParamInfo(choices=("cuda", "torch", "cpu"),
                             help="where the cost is computed; auto: the "
                                  "fastest available; time-window fit only"),
    }

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

    def run(self, ctx: Context, settings: DriftSettings) -> Result:
        ctx.selection.require(5000, ctx.report, "a drift estimate")
        ctx.report(f"estimating drift from {ctx.selection}")
        corrected, drift = correct_drift(ctx.locs, settings,
                                         select=ctx.selection.mask,
                                         progress=ctx.report)
        return Result(locs=corrected, text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
