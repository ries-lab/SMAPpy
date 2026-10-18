"""Fiducial drift correction: the tracks of beads, one spline.

The work is in :mod:`smappy.fiducials` (its docstring has the decisions).
Ported from SMAP's ``driftcorrection_beads``, which took the beads from
SMAP's grouping (``numberInGroup`` above a count), subtracted each bead's
mean and averaged them in a running window of frames.  Here:

* beads are tracks of *Link tracks*' simple linker over the localizations
  brighter than *min photons*, with a memory of 50 frames, and a bead must
  be seen in a fraction of the frames rather than a count;
* each bead's offset is fitted with the curve, not its mean subtracted, so
  a bead that starts late or is interrupted lands where it belongs;
* the curve is a cubic B-spline (COMET's representation), not a running
  mean, and is zero at the first frame;
* the beads are those of the layer's filter and the ROI, not a choice of
  "View/Roi" or the ROI manager's sites; one file at a time, as in SMAP.
"""
from __future__ import annotations

import numpy as np

from ..fiducials import (FiducialSettings, describe, draw_beads,
                         estimate_drift_fiducials)
from . import Context, Plot, Plugin, Result, register
from .drift_result import KeepsDrift


@register("Analysis/Drift/Fiducials")
class FiducialDrift(KeepsDrift, Plugin):
    description = ("Estimate drift from the tracks of fiducial beads and "
                   "subtract it from all localizations.")
    changes_locs = True
    # beads are tracks of localizations, and the correction moves every one
    grouping = "ungrouped"
    Settings = FiducialSettings
    name = "Fiducials drift correction"
    version = "1"

    def run(self, ctx: Context, settings: FiducialSettings) -> Result:
        locs = ctx.locs
        mask = np.asarray(ctx.selection.mask, bool)
        if "filenumber" in locs:
            files = np.unique(np.asarray(locs["filenumber"])[mask])
            if len(files) > 1:
                raise ValueError(
                    f"the selection holds {len(files)} files; drift is of one "
                    f"acquisition: show one file in the layer and run it per file")
        ctx.report(f"finding beads in {ctx.selection}")
        drift, beads, fit = estimate_drift_fiducials(locs, settings, select=mask)
        frame = np.asarray(locs["frame"], np.int64)[fit.rows]
        residuals, bead, curve = fit.residuals_nm, fit.bead, fit.drift_nm
        return Result(locs=drift.apply(locs), text=describe(drift, beads, fit),
                      plot=drift.plot, settings=settings,
                      plots={"beads": Plot(
                          lambda figure: draw_beads(figure, frame, residuals,
                                                    bead, curve),
                          name="beads", panels=residuals.shape[1],
                          size=(7, 1.8 * residuals.shape[1] + 0.6))},
                      data={"drift": drift, "beads": len(beads),
                            "rms_nm": fit.rms_nm.tolist()})
