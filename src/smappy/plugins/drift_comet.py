"""COMET drift correction as a plugin.  The work is in :mod:`smappy.drift`."""
from __future__ import annotations

from typing import Optional

from ..drift import DriftSettings, correct_drift, estimate_cost
from . import Context, ParamInfo, Plugin, Result, register


@register("Analysis/Drift/COMET")
class CometDrift(Plugin):
    description = ("Estimate drift from the selected localizations with COMET "
                   "and subtract it from all of them.")
    Settings = DriftSettings
    main = ("segmentation_var", "max_drift_nm", "target_sigma_nm",
            "spline", "spline_knot_frames", "group", "use_z")
    params = {
        "segmentation_mode": ParamInfo(label="window unit",
                                       choices=((2, "frames"), (1, "localizations"),
                                                (0, "windows"))),
        "segmentation_var": ParamInfo(label="window size", min=1),
        "max_drift_nm": ParamInfo(label="max drift", unit="nm", min=1),
        "target_sigma_nm": ParamInfo(label="target sigma", unit="nm", min=0.1),
        "initial_sigma_nm": ParamInfo(label="initial sigma", unit="nm"),
        "boxcar_width": ParamInfo(label="smoothing", unit="windows", min=1),
        "interpolation": ParamInfo(choices=("cubic", "catmull-rom")),
        "max_locs_per_segment": ParamInfo(label="max locs / window", min=1),
        "optimizer_ftol": ParamInfo(label="ftol"),
        "optimizer_ftol_coarse": ParamInfo(label="ftol coarse"),
        "approximate_kernel": ParamInfo(label="approximate kernel"),
        "quality_control": ParamInfo(label="quality control"),
        "min_lift": ParamInfo(label="min lift"),
        "group": ParamInfo(label="group blinks"),
        "group_dx_nm": ParamInfo(label="group radius", unit="nm", min=0),
        "group_dt": ParamInfo(label="group gap", unit="frames", min=0),
        "two_stage": ParamInfo(label="two stage"),
        "two_stage_radius_nm": ParamInfo(label="fine radius", unit="nm", min=0),
        "spline": ParamInfo(label="fit spline"),
        "spline_knot_frames": ParamInfo(label="knot spacing", unit="frames", min=10),
        "spline_penalty": ParamInfo(label="spline penalty", min=0),
        "use_z": ParamInfo(label="correct z", help="auto: if the table has z"),
        "backend": ParamInfo(choices=("cuda", "torch", "cpu")),
    }

    # Longer than this and the run is worth agreeing to first.  Five minutes
    # is where waiting stops being waiting and starts being a decision.
    slow_seconds = 300.0

    def preflight(self, ctx: Context, settings: DriftSettings) -> Optional[str]:
        """What this is going to cost, and a question if that is a lot.

        Time and memory both follow from the neighbour-pair count, which grows
        with the square of the localization count and steeply with the search
        radius -- so a dataset twice the size of the one the defaults were
        chosen on is not twice the wait, and the difference between a minute
        and an afternoon is not visible in the settings.
        """
        cost = estimate_cost(ctx.locs, settings, select=ctx.selection.mask)
        ctx.report(str(cost))
        return cost.question(self.slow_seconds)

    def run(self, ctx: Context, settings: DriftSettings) -> Result:
        ctx.selection.require(5000, ctx.report, "a drift estimate")
        ctx.report(f"estimating drift from {ctx.selection}")
        corrected, drift = correct_drift(ctx.locs, settings,
                                         select=ctx.selection.mask,
                                         progress=ctx.report)
        return Result(locs=corrected, text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
