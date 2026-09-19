"""Drop -- or merely hide -- the localizations inside or outside the ROI.

SMAP's ``Process/Modify/RemoveLocs``: draw a region, and either throw away
what is in it (a fluorescent dirt particle, a cell that is out of focus) or
throw away everything else (one cell out of a field of twenty).

Two departures from SMAP.

**What counts as "inside".**  SMAP asks which files to act on and has a third
mode, "keep visible inside ROI", that quietly means *the layer's filter as
well*.  Here that is one setting, `region`: the drawn ROI alone, or the whole
selection -- the layer's filter, the ROI and the slab together, which is what
is on screen.  "Keep only what I can see" is then `region = selection` with
`which = outside`, and restricting to one file is what the layer's file
selector already does.

**Hiding rather than removing.**  SMAP's "set property: active" writes a flag
instead of deleting rows.  The same is here as `action = "hide"`, and the flag
is an ordinary column, so the filter, the renderer and every other plugin
treat it like any other number: hidden localizations are the ones with
``use = 0``, the run opens the filter at ``use >= 0.5``, and moving that
slider back shows them again.  The column is saved with the file, so the
decision survives a reload -- which the deletion, being undoable only within
the session, does not.

Hiding is also the safer half of the pair: a removal is one undo step and a
save away from being permanent, so a run that would leave nothing behind is
refused rather than performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..locs import Localizations
from ..mathparse import remember
from ..regions import Region
from ..render import positions
from . import Context, Plugin, Result, param, register

# What the flag means, and the bound a run opens the filter at.  0 and 1
# rather than True/False: the table is numeric throughout, and a range filter
# is what the GUI already knows how to show.
HIDDEN, VISIBLE = 0.0, 1.0
FILTER_BOUND: Tuple[Optional[float], Optional[float]] = (0.5, None)


# ----------------------------------------------------------------- the work

def region_mask(locs: Localizations, region: Region) -> np.ndarray:
    """Which localizations lie inside ``region``."""
    x, y = positions(locs)
    return np.asarray(region.mask(x, y), dtype=bool)


def current_roi(ctx: Context) -> Optional[Region]:
    """The region drawn in the render window, if there is one.

    From the session rather than from ``ctx.selection.roi``, which carries the
    slab instead when one is selecting: what a person means by "the ROI" is
    the shape they drew.
    """
    roi = getattr(ctx.session, "roi", None)
    if isinstance(roi, Region):
        return roi
    found = getattr(ctx.selection, "roi", None)
    return found if isinstance(found, Region) else None


def drop(locs: Localizations, keep: np.ndarray) -> Localizations:
    """The table with only ``keep``.

    A new table rather than columns edited in place: `Session.set_locs`
    remembers the one it replaced, and that is the whole of the undo.
    """
    return locs[np.asarray(keep, dtype=bool)]


def hide(locs: Localizations, keep: np.ndarray, field: str = "use") -> Localizations:
    """The table with ``field`` cleared where ``keep`` is false.

    Combined with whatever the column already said, never overwriting it: two
    runs that each hide a dirt particle hide both, which is what someone
    clicking around a field of view means.  The rows are untouched, so this is
    reversible by the filter alone.
    """
    keep = np.asarray(keep, dtype=bool)
    if field in locs:
        keep = keep & (np.asarray(locs[field]) > 0.5)
    columns = dict(locs.columns)
    columns[field] = np.where(keep, VISIBLE, HIDDEN).astype(np.float32)
    out = Localizations(columns, dict(locs.metadata))
    # Once the localizations are linked, a blink is visible only if all of its
    # localizations are: hiding a region and then grouping should not bring
    # half of it back as blinks that straddle the edge.
    remember(out, field, "", grouped="all")
    return out


def check_field(field: str) -> str:
    """The flag's name, refused now rather than half-written later."""
    name = (field or "").strip()
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not usable as a column name: a word of "
                         "letters, digits and underscores, not starting with "
                         "a digit")
    if name in ("group_id", "n_in_group"):
        raise ValueError(f"{name!r} is written by the grouping itself and "
                         "would be overwritten again")
    return name


# --------------------------------------------------------------- the plugin

@dataclass
class RemoveSettings:
    which: str = param("inside", label="remove",
                       choices=(("inside", "the localizations inside"),
                                ("outside", "the localizations outside")),
                       help="which side of the region goes")
    region: str = param("roi", label="region",
                        choices=(("roi", "the drawn ROI"),
                                 ("selection", "the selection (filter, ROI, slab)")),
                        help="the ROI on its own, or everything that decides "
                             "what is on screen")
    action: str = param("remove", label="action",
                        choices=(("remove", "remove them from the table"),
                                 ("hide", "keep them, flagged and filtered out")),
                        help="hiding writes a column and sets the filter on "
                             "it, so the decision is reversible and is saved "
                             "with the file")
    field: str = param("use", label="flag", advanced=True,
                       help="the column hiding writes: 1 visible, 0 hidden")


@register("Analysis/Process/Remove Localizations")
class RemoveLocalizations(Plugin):
    """Remove, or hide, the localizations inside or outside the ROI."""

    Settings = RemoveSettings
    version = "1"

    def active(self, settings: RemoveSettings):
        return {"field": settings.action == "hide"}

    def _keep(self, ctx: Context, settings: RemoveSettings) -> Tuple[np.ndarray, str]:
        """Which localizations survive, and how to say what was done."""
        locs = ctx.locs
        if not len(locs):
            raise ValueError("no localizations to remove")
        if settings.region == "selection":
            inside = np.asarray(ctx.selection.mask, dtype=bool)
            where = str(ctx.selection)
        else:
            roi = current_roi(ctx)
            if roi is None:
                raise ValueError("no ROI: draw one in the render window, or "
                                 "set the region to the selection")
            inside = region_mask(locs, roi)
            where = f"the {roi}"
        keep = ~inside if settings.which == "inside" else inside
        going = int((~keep).sum())
        if going == 0:
            raise ValueError(f"nothing is {settings.which} {where}")
        if going == len(locs):
            raise ValueError(f"every localization is {settings.which} {where}: "
                             "that would leave an empty table")
        return keep, where

    def preview(self, ctx: Context, settings: RemoveSettings) -> Result:
        """How many would go, before any of them does."""
        keep, where = self._keep(ctx, settings)
        going = int((~keep).sum())
        verb = "hide" if settings.action == "hide" else "remove"
        return Result(settings=settings,
                      text=f"would {verb} {going} of {len(ctx.locs)} "
                           f"localizations {settings.which} {where}")

    def run(self, ctx: Context, settings: RemoveSettings) -> Result:
        keep, where = self._keep(ctx, settings)
        going = int((~keep).sum())
        if settings.action == "hide":
            field = check_field(settings.field)
            locs = hide(ctx.locs, keep, field)
            hidden = int((np.asarray(locs[field]) < 0.5).sum())
            text = (f"hid {going} of {len(ctx.locs)} localizations "
                    f"{settings.which} {where}; {hidden} hidden in all, "
                    f"filtered out by {field} >= {FILTER_BOUND[0]}")
            # the bound belongs to the new table, so the session sets it once
            # that table is in place; see `Session.apply`
            data = {"bounds": {field: FILTER_BOUND}}
        else:
            locs = drop(ctx.locs, keep)
            text = (f"removed {going} of {len(ctx.locs)} localizations "
                    f"{settings.which} {where}; {len(locs)} left")
            data = {}
        return Result(locs=locs, text=text, data=data, settings=settings)
