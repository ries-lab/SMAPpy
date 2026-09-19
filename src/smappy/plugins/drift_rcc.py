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
        "n_timepoints": ParamInfo(label="time windows", min=2),
        "pixelsize_nm": ParamInfo(label="pixel size", unit="nm", min=1),
        "z_pixelsize_nm": ParamInfo(label="z bin", unit="nm", min=0.1),
        "max_drift_nm": ParamInfo(label="max drift", unit="nm", min=1),
        "fit_window": ParamInfo(label="peak fit half-width", unit="pix", min=1),
        "max_pixels": ParamInfo(label="max image size", unit="pix", min=64),
        "use_z": ParamInfo(label="correct z", help="auto: if the table has z"),
        "group": ParamInfo(label="group blinks"),
        "group_dx_nm": ParamInfo(label="group radius", unit="nm", min=0),
        "group_dt": ParamInfo(label="group gap", unit="frames", min=0),
        "tile_nm": ParamInfo(label="tile", unit="nm", min=1),
        "tile_y_nm": ParamInfo(label="tile y", unit="nm", min=1),
    }

    def run(self, ctx: Context, settings: RCCSettings) -> Result:
        ctx.selection.require(5000, ctx.report, "a drift estimate")
        ctx.report(f"correlating {ctx.selection}")
        drift = estimate_drift_rcc(ctx.locs, settings, select=ctx.selection.mask)
        return Result(locs=drift.apply(ctx.locs), text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
