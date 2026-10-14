"""Sort ROIs and Filter ROIs: SMAP's SortROIs and FilterSites, on the results
table's columns -- what the steps measured and the keywords ROIs were given.

The work is the project's (`ROIProject.sort`, `ROIProject.filter_rois`) and
`roi_manager.order`'s, which say why it is done as it is; these are the
plugins over them, so a chain can sort or filter as a step: find the pores,
measure them, keep those with six corners or more, sort by labelling.

Sorting changes the order and nothing else -- an ROI keeps its number, its
results and *use* -- and the ROI list, the results table and every summary
that reads it follow.  It is not an undo step: the order holds no work, and
*sort by* left empty puts it back by number.  Filtering sets *use*, which is
work, so it is one step of the ROI undo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from . import Context, FormState, Notice, Plugin, Result, param, register


def no_column():
    """Without a session there are no columns to offer: only *none*."""
    return [("", "none")]


@dataclass
class SortROIsSettings:
    by: str = param("", label="sort by", choices=no_column,
                    help="a measured column or a keyword list; none: by ROI number")
    descending: bool = param(True, label="largest first",
                             help="descending: the largest value, or a list's "
                                  "last keyword, first")
    then_by: str = param("", label="then by", choices=no_column,
                         help="breaks the ties of the first")
    then_descending: bool = param(True, label="largest first")
    last_by: str = param("", label="and then by", choices=no_column, advanced=True,
                         help="breaks the ties of the first two")
    last_descending: bool = param(True, label="largest first", advanced=True)


def _columns(ctx) -> List[Tuple[str, str]]:
    project = getattr(ctx, "rois", None) if ctx is not None else None
    names = project.sort_columns() if project is not None else []
    lists = project.annotation_lists if project is not None else {}
    return [(n, f"{n} (keywords)" if n in lists else n) for n in names]


@register("Analysis/Regions/Sort ROIs")
class SortROIs(Plugin):
    """Orders the ROIs by what was measured in them or the keywords they were
    given; the list, the results and the summaries follow."""

    Settings = SortROIsSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        found = _columns(ctx)
        choices = [("", "none")] + found
        return FormState(choices={"by": [("", "none (by number)")] + found,
                                  "then_by": choices, "last_by": choices})

    def run(self, ctx: Context, settings: SortROIsSettings) -> Result:
        project = ctx.rois
        if project is None or not project.rois:
            raise ValueError("there are no ROIs to sort")
        keys = [(c, d) for c, d in ((settings.by, settings.descending),
                                    (settings.then_by, settings.then_descending),
                                    (settings.last_by, settings.last_descending)) if c]
        if not keys:
            project.unsort()
            text = f"{len(project.rois)} ROIs in the order of their numbers"
        else:
            known = set(project.sort_columns())
            unknown = [c for c, _ in keys if c not in known]
            if unknown:
                raise ValueError(f"no ROI has {', '.join(unknown)}: measure the ROIs "
                                 "first, or choose another column")
            order = project.sort(keys)
            rows = project.sort_rows()
            lacking = sum(1 for i in order if rows[i].get(keys[0][0]) is None)
            text = (f"{len(order)} ROIs sorted by "
                    + ", then ".join(f"{c}{' (largest first)' if d else ''}"
                                     for c, d in keys)
                    + (f"; {lacking} without {keys[0][0]} last" if lacking else ""))
        return Result(text=text, settings=settings, rois=[])


@dataclass
class FilterROIsSettings:
    condition: str = param("", label="keep", kind="text",
                           help='which ROIs to use, e.g. n_corners >= 6 and '
                                'quality == "good"')
    mode: str = param("narrow", label="then",
                      choices=[("narrow", "exclude the others"),
                               ("set", "use exactly these")],
                      help="exclude the ROIs that fail, leaving those already "
                           "excluded so; or include every ROI that passes too")


@register("Analysis/Regions/Filter ROIs")
class FilterROIs(Plugin):
    """Includes or excludes ROIs by a condition on what was measured in them
    and the keywords they were given: their *use*."""

    Settings = FilterROIsSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        """How many ROIs the condition keeps, as it is typed."""
        project = getattr(ctx, "rois", None) if ctx is not None else None
        if project is None or not project.rois or not settings.condition.strip():
            names = ", ".join(n for n, _ in _columns(ctx)) if project is not None else ""
            return FormState(notice=Notice(f"columns: {names}", kind="info")
                             if names else None)
        try:
            kept = new_use(project, settings)
        except ValueError as error:
            return FormState(notice=Notice(str(error)))
        used = sum(kept.values())
        return FormState(notice=Notice(f"would use {used} of {len(kept)} ROIs "
                                       f"(now {sum(r.use for r in project.rois.values())})",
                                       kind="info"))

    def run(self, ctx: Context, settings: FilterROIsSettings) -> Result:
        project = ctx.rois
        if project is None or not project.rois:
            raise ValueError("there are no ROIs to filter")
        if not settings.condition.strip():
            raise ValueError("write a condition: which ROIs to keep")
        use = new_use(project, settings)
        changed = 0
        with project.edit(f"{self.name}: {settings.condition}"):
            for roi_id, on in use.items():
                roi = project.rois[roi_id]
                if roi.use != on:
                    roi.use = on
                    changed += 1
        used = sum(use.values())
        return Result(text=f"{used} of {len(use)} ROIs used ({changed} changed): "
                           f"{settings.condition}",
                      settings=settings, rois=[], values={"used": used})


def new_use(project, settings: FilterROIsSettings) -> dict:
    """``{roi id: use}`` after the filter, without applying it."""
    passed = project.filter_rois(settings.condition)
    if settings.mode == "set":
        return passed
    return {i: ok and project.rois[i].use for i, ok in passed.items()}
