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
  "View/Roi" or the ROI manager's sites; one file at a time, as in SMAP;
* a bead that disagrees with the others is left out (`beads.select_beads`,
  as MBM's), and any bead can be chosen by hand in the bead selector
  (`gui.bead_window`), where SMAP took them all.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..beads import BeadView, draw_bead_tracks, format_overrides, parse_overrides
from ..fiducials import (FiducialSettings, describe, estimate_drift_fiducials,
                         find_and_choose)
from . import Context, FormState, Notice, Plot, Plugin, Result, register
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
    # 2: the other files of the table keep their positions; 3: a bead that
    # disagrees with the others is left out, and beads can be chosen by hand
    version = "3"

    def run(self, ctx: Context, settings: FiducialSettings) -> Result:
        locs = ctx.locs
        mask = np.asarray(ctx.selection.mask, bool)
        rows = None
        if "filenumber" in locs:
            numbers = np.asarray(locs["filenumber"])
            files = np.unique(numbers[mask])
            if len(files) > 1:
                raise ValueError(
                    f"the selection holds {len(files)} files; drift is of one "
                    f"acquisition: show one file in the layer and run it per file")
            # the curve is that file's: the others keep their positions
            if len(files) and numbers.min() != numbers.max():
                rows = numbers == files[0]
        ctx.report(f"finding beads in {ctx.selection}")
        drift, beads, fit = estimate_drift_fiducials(locs, settings, select=mask)
        chosen = fit.chosen
        axes = ("x", "y", "z")[:fit.residuals_nm.shape[1]]
        return Result(locs=drift.apply(locs, rows=rows),
                      text=describe(drift, beads, fit),
                      plot=drift.plot, settings=settings,
                      plots={"beads": Plot(
                          lambda figure: draw_bead_tracks(figure, chosen, axes,
                                                          time_unit="frame"),
                          name="beads", panels=2 * len(axes),
                          size=(9, 2.0 * len(axes) + 0.8))},
                      data={"drift": drift, "beads": len(beads),
                            "rms_nm": fit.rms_nm.tolist()})

    def describe(self, ctx: Optional[Context], settings: FiducialSettings) -> FormState:
        """The button to the bead selector.  The beads are not counted here:
        finding them links every bright localization, too slow for each
        keystroke in the form."""
        if ctx is None or ctx.locs is None or not len(ctx.locs):
            return FormState()
        hand = parse_overrides(settings.beads) if settings.beads else {}
        text = (f"{len(hand)} bead(s) chosen by hand" if hand
                else "the beads are found and judged when run")
        return FormState(notice=Notice(text, "Select beads...", "bead selector",
                                       kind="info"))

    # -- the bead window's interface (`gui.bead_window`)
    def bead_view(self, ctx: Context, settings: FiducialSettings, source: int = 0):
        mask = np.asarray(ctx.selection.mask, bool)
        found = find_and_choose(ctx.locs, settings, select=mask)
        beads = found.beads
        return BeadView(found.chosen, found.axes, [str(ctx.selection)], 0,
                        {"frames": [f"{a}-{b}" for a, b in zip(beads.first, beads.last)]},
                        found.where_nm, title="fiducials", time_unit="frame")

    def with_overrides(self, settings: FiducialSettings, view,
                       overrides: Dict[str, bool]) -> FiducialSettings:
        from dataclasses import replace
        every = {k: v for k, v in parse_overrides(settings.beads).items()
                 if k not in view.beads.names}
        every.update(overrides)
        return replace(settings, beads=format_overrides(every))
