"""COMET drift correction as a plugin.  The work is in :mod:`smappy.drift`."""
from __future__ import annotations

from ..drift import DriftSettings, correct_drift
from . import ParamInfo, Plugin, Result, Selection, register


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

    def run(self, locs, selection: Selection, settings: DriftSettings,
            progress=None, stream=None) -> Result:
        if progress:
            progress(f"estimating drift from {selection}")
        corrected, drift = correct_drift(locs, settings, select=selection.mask)
        return Result(locs=corrected, text=str(drift), plot=drift.plot,
                      data={"drift": drift}, settings=settings)
