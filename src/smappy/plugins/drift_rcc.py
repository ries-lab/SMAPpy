"""Redundant cross-correlation drift correction, the classic alternative to COMET."""
from __future__ import annotations

from ..rcc import RCCSettings, estimate_drift_rcc
from . import ParamInfo, Plugin, Result, Selection, register


@register("Analysis/Drift/RCC")
class RCCDrift(Plugin):
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

    def run(self, locs, selection: Selection, settings: RCCSettings,
            progress=None, stream=None) -> Result:
        if progress:
            progress(f"correlating {selection}")
        drift = estimate_drift_rcc(locs, settings, select=selection.mask)
        return Result(locs=drift.apply(locs), text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
