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
from ..fiducials import (SOURCES, FiducialSettings, describe, estimate_drift_fiducials,
                         find_and_choose)
from . import Context, FormState, Notice, Plot, Plugin, Result, register, settings_values
from .drift_result import KeepsDrift


def _rois(ctx: Context):
    """The ROI manager's project, or None."""
    project = getattr(ctx, "rois", None)
    if project is None and ctx.session is not None:
        project = getattr(ctx.session, "rois", None)
    return project


def roi_regions(ctx: Context, file: Optional[int] = None):
    """Each used ROI of file ``file`` (None: any) as ``("ROI n", rows)``,
    rows of ``ctx.locs`` -- the session's localizations, which is what the
    ROI manager's rows index."""
    project = _rois(ctx)
    if project is None or not project.rois:
        raise ValueError("beads from the ROI manager, and it has no ROIs: draw "
                         "one round each bead there, or let the plugin find them")
    if ctx.session is None or len(ctx.session.locs) != len(ctx.locs):
        raise ValueError("beads from the ROI manager need the session's table")
    out = []
    for i in project.ordered_ids():
        roi = project.rois[i]
        if not roi.use:
            continue
        if file is not None and project.sources[roi.file_id].number != file:
            continue
        out.append((f"ROI {roi.number}", np.asarray(project.session_rows(roi))))
    if not out:
        raise ValueError("no used ROI in this file: tick the bead ROIs' use, or "
                         "let the plugin find the beads")
    return out


def _file(ctx: Context, mask) -> Optional[int]:
    if "filenumber" not in ctx.locs:
        return None
    files = np.unique(np.asarray(ctx.locs["filenumber"])[mask])
    return int(files[0]) if len(files) == 1 else None


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
        regions = None
        if settings.source == SOURCES[1]:
            regions = roi_regions(ctx, _file(ctx, mask))
            ctx.report(f"beads in {len(regions)} ROIs")
        else:
            ctx.report(f"finding beads in {ctx.selection}")
        drift, beads, fit = estimate_drift_fiducials(locs, settings, select=mask,
                                                     regions=regions)
        if fit.empty:
            ctx.warn(f"no bead in {', '.join(fit.empty)}: fewer than two "
                     f"localizations brighter than {settings.min_photons:g} photons")
        chosen = fit.chosen
        axes = ("x", "y", "z")[:fit.residuals_nm.shape[1]]
        out = drift.apply(locs, rows=rows)
        # each bead's localizations named (`bead_id`), and what the selector
        # shows of each kept as a region table of them
        from ..beads import BEAD_ID, bead_values, with_bead_ids
        from ..region_runs import column_table
        mine = np.ones(len(locs), bool) if rows is None else np.asarray(rows, bool)
        out, offset = with_bead_ids(out, mine, beads.bead[mine] + 1)
        values, groups = bead_values(chosen, axes)
        step = {"label": self.name, "plugin": self.path, "version": self.version,
                "parameters": settings_values(settings)}
        table = column_table(out, BEAD_ID, self.name, np.asarray(chosen.ids) + offset,
                             values, step, layer=ctx.layer, session=ctx.session,
                             groups=groups)
        return Result(locs=out, id_regions={BEAD_ID: {self.name: table}},
                      text=describe(drift, beads, fit)
                      + f"\nkept per bead as 'per {BEAD_ID}: {self.name}'",
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
        from_rois = settings.source == SOURCES[1]
        # nothing is linked from ROIs: the linker's settings do not apply
        active = {"min_fraction": not from_rois, "max_jump_nm": not from_rois,
                  "memory": not from_rois}
        if ctx is None or ctx.locs is None or not len(ctx.locs):
            return FormState(active=active)
        hand = parse_overrides(settings.beads) if settings.beads else {}
        if from_rois:
            project = _rois(ctx)
            n = sum(1 for r in project.rois.values() if r.use) if project else 0
            text = f"a bead in each of the {n} used ROIs" if n else \
                "the ROI manager has no used ROI"
        else:
            text = "the beads are found and judged when run"
        if hand:
            text += f"; {len(hand)} chosen by hand"
        return FormState(active=active, notice=Notice(text, "Select beads...",
                                                      "bead selector", kind="info"))

    # -- the bead window's interface (`gui.bead_window`)
    def bead_view(self, ctx: Context, settings: FiducialSettings, source: int = 0):
        mask = np.asarray(ctx.selection.mask, bool)
        regions = (roi_regions(ctx, _file(ctx, mask)) if settings.source == SOURCES[1]
                   else None)
        found = find_and_choose(ctx.locs, settings, select=mask, regions=regions)
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
