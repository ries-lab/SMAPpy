"""Centre ROIs: SMAP's *centre all* (``ROIManager/Evaluate/CenterSites``),
moving every used ROI onto the centre of its structure.

A segmenter places an ROI where its filter peaked, a hand where the eye
landed; a measurement that assumes the structure in the middle -- a radial
profile, an average of many ROIs -- wants better.  This measures each ROI's
centre, by a fitted circle (`circle_fit.circle_fit`, the NPC segmenter's
fit) or the median of its localizations, and moves the ROI there.

A Regions plugin rather than a button of the ROI manager, so that a chain
can do it between finding the ROIs and measuring them, and a batch with it.
Moving is an edit of the ROIs: one step of the ROI undo, as Filter ROIs.

Departures from SMAP: SMAP's *mean* and *mask* (a ring template correlated
with the site image) are left out -- the mean is the median's less robust
twin, and the template is the ring segmenter's own job, done better there;
SMAP's *fitring* is a least-squares circle of fixed radius, here the
soft-L1 fit, with the radius fixed or free.  A centre that falls outside the
ROI is not taken; SMAP moves the site wherever the fit went.  z is not
centred (SMAP's *correct z*): an ROI has no z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import Context, FormState, Plugin, Result, param, register


@dataclass
class CentreROIsSettings:
    centre: str = param("circle", label="centre on",
                        choices=[("circle", "a fitted circle"),
                                 ("median", "the median of the localizations")],
                        help="a ring's centre is the circle's; a blob's, the median")
    radius_nm: float = param(0.0, label="fixed radius", unit="nm", min=0.0,
                             help="0: the circle's radius is fitted too")
    iterations: int = param(1, label="iterations", min=1, max=10,
                            help="measure and move again from the new place: "
                                 "an ROI far off its structure sees more of it "
                                 "after the first move")


@register("Analysis/Regions/Centre ROIs")
class CentreROIs(Plugin):
    """Moves every used ROI onto the centre of its localizations: a fitted
    circle's, or their median."""

    Settings = CentreROIsSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        return FormState(active={"radius_nm": settings.centre == "circle"})

    def run(self, ctx: Context, settings: CentreROIsSettings) -> Result:
        project = ctx.rois
        if project is None or not project.rois:
            raise ValueError("there are no ROIs to centre")
        used = [i for i in project.ordered_ids() if project.rois[i].use]
        if not used:
            raise ValueError("no ROI is used: tick use for the ROIs to centre")
        how = "circle fit" if settings.centre == "circle" else "median"
        moved, kept, shifts = 0, 0, []
        with project.edit(f"{self.name}: {how}"):
            for k, roi_id in enumerate(used):
                ctx.report(f"centring ROI {k + 1} of {len(used)}")
                start = np.asarray(project.rois[roi_id].center, float)
                for _ in range(int(settings.iterations)):
                    found = roi_centre(project, project.rois[roi_id], settings)
                    if found is None:
                        break
                    project.move_roi(roi_id, found)
                shift = float(np.hypot(*(np.asarray(project.rois[roi_id].center) - start)))
                if shift > 0:
                    moved += 1
                    shifts.append(shift)
                else:
                    kept += 1
        text = f"moved {moved} of {len(used)} ROIs onto the {how}"
        if shifts:
            text += f", by {np.median(shifts):.1f} nm (median), {max(shifts):.1f} nm at most"
        if kept:
            text += f"; {kept} left: too few localizations, or the centre outside the ROI"
        return Result(text=text, settings=settings, rois=[],
                      values={"moved": moved, "median_shift_nm":
                              float(np.median(shifts)) if shifts else 0.0})


def roi_centre(project, roi, settings: CentreROIsSettings):
    """Where one ROI's structure is centred, or None to leave it: its
    localizations as the ROI manager's layer shows them (filter, merged
    blinks), measured by `circle_fit.circle_fit` or their median.  A centre
    outside the ROI is refused: a circle fitted to a few background
    localizations can land anywhere, and an ROI walked off its structure is
    worse than one a little off-centre."""
    from .circle_fit import circle_fit
    ctx = project.region_context(roi)
    locs = ctx.selection.apply(ctx.locs)
    if not len(locs):
        return None
    x = np.asarray(locs["x_nm"], dtype=float)
    y = np.asarray(locs["y_nm"], dtype=float)
    if settings.centre == "median":
        centre = np.array([np.nanmedian(x), np.nanmedian(y)])
    else:
        fit = circle_fit(x, y, radius=settings.radius_nm or None)
        centre = np.array([fit["x_nm"], fit["y_nm"]])
    if not np.all(np.isfinite(centre)):
        return None
    if not bool(project.region(roi).mask(centre[:1], centre[1:])[0]):
        return None
    return centre.tolist()
