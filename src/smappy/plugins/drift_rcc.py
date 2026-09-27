"""Redundant cross-correlation drift correction, the classic alternative to COMET."""
from __future__ import annotations

from ..rcc import RCCSettings, estimate_drift_rcc
from . import Context, ParamInfo, Plugin, Result, register
from .drift_result import KeepsDrift


@register("Analysis/Drift/RCC")
class RCCDrift(KeepsDrift, Plugin):
    description = ("Estimate drift by cross-correlating rendered time windows "
                   "(RCC) and subtract it from all localizations.")
    Settings = RCCSettings
    main = ("n_timepoints", "pixelsize_nm", "max_drift_nm", "use_z", "group")
    params = {
        "n_timepoints": ParamInfo(label="time windows", min=2,
                                  help="how many windows the acquisition is cut "
                                       "into: more follow faster drift, fewer are "
                                       "less noisy"),
        "pixelsize_nm": ParamInfo(label="pixel size", unit="nm", min=1,
                                  help="the pixel of the images that are "
                                       "correlated; about the localization precision"),
        "z_pixelsize_nm": ParamInfo(label="z bin", unit="nm", min=0.1,
                                    help="the bin width of the z histograms"),
        "max_drift_nm": ParamInfo(label="max drift", unit="nm", min=1,
                                  help="the largest drift between any two windows "
                                       "that can be found"),
        "fit_window": ParamInfo(label="peak fit half-width", unit="pix", min=1,
                                help="the patch the correlation peak is fitted "
                                     "over, for its sub-pixel position"),
        "max_pixels": ParamInfo(label="max image size", unit="pix", min=64,
                                help="a wider field of view is folded back onto "
                                     "itself, to keep the correlation fast"),
        "use_z": ParamInfo(label="correct z", help="auto: if the table has z"),
        "group": ParamInfo(label="group blinks",
                           help="link the localizations of one blink into one "
                                "before measuring"),
        "group_dx_nm": ParamInfo(label="group radius", unit="nm", min=0,
                                 help="how close localizations in consecutive "
                                      "frames must be to be one blink"),
        "group_dt": ParamInfo(label="group gap", unit="frames", min=0,
                              help="how many dark frames a blink may skip"),
        "tile_nm": ParamInfo(label="tile", unit="nm", min=1,
                             help="the axial pass correlates z histograms in "
                                  "tiles of the field of view this size"),
        "tile_y_nm": ParamInfo(label="tile y", unit="nm", min=1,
                               help="auto: square tiles"),
        "axial_max_drift_nm": ParamInfo(label="max axial drift", unit="nm", min=1,
                                        help="how far from zero the axial "
                                             "correlation peak is looked for"),
        "exclude_zero_lag": ParamInfo(label="skip zero shift",
                                      help="leave the zero-shift sample out of the "
                                           "axial peak search"),
    }

    def run(self, ctx: Context, settings: RCCSettings) -> Result:
        ctx.selection.require(5000, ctx.report, "a drift estimate")
        ctx.report(f"correlating {ctx.selection}")
        drift = estimate_drift_rcc(ctx.locs, settings, select=ctx.selection.mask)
        return Result(locs=drift.apply(ctx.locs), text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
