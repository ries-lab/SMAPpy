"""Sort ROIs, Filter ROIs and Remove double ROIs: SMAP's SortROIs,
FilterSites and RemoveDoubleSites, on the results table's columns -- what the
steps measured, the keywords ROIs were given and where they are.

The work is the project's (`ROIProject.sort`, `ROIProject.filter_rois`) and
`roi_manager.order`'s, which say why it is done as it is; these are the
plugins over them, so a chain can sort or filter as a step: find the pores,
measure them, keep those with six corners or more, sort by labelling.

Sorting changes the order and nothing else -- an ROI keeps its number, its
results and *use* -- and the ROI list, the results table and every summary
that reads it follow.  It is not an undo step: the order holds no work, and
*sort by* left empty puts it back by number.  Filtering sets *use*, which is
work, so it is one step of the ROI undo.

Removing doubles is a plugin of its own rather than a mode of Filter ROIs:
a filter tests each ROI's own row, a double is a pair, and its one setting
(the distance) would be a field the filter shows and reads in one mode only.
It excludes rather than deletes, as Filter ROIs does: the one with fewer
localizations is not always the worse of the two, and *use* can be ticked
again.
"""
from __future__ import annotations

import re
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
    insert: str = param("", label="insert", choices=no_column,
                        help="a measured column or keyword list, added to the "
                             "end of the condition")
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
        """The columns to insert, and how many ROIs the condition keeps, as
        it is typed."""
        project = getattr(ctx, "rois", None) if ctx is not None else None
        found = _columns(ctx)
        choices = {"insert": [("", "a column..." if found else
                               "nothing measured yet")] + found}
        if project is None or not project.rois or not settings.condition.strip():
            return FormState(choices=choices)
        try:
            kept = new_use(project, settings)
        except ValueError as error:
            return FormState(choices=choices, notice=Notice(str(error)))
        used = sum(kept.values())
        return FormState(choices=choices,
                         notice=Notice(f"would use {used} of {len(kept)} ROIs "
                                       f"(now {sum(r.use for r in project.rois.values())})",
                                       kind="info"))

    def react(self, changed: str, settings):
        """A column chosen in *insert* goes onto the end of the condition --
        in backquotes when it is no plain name -- and the list goes back."""
        if changed != "insert" or not settings.insert:
            return None
        name = settings.insert
        token = name if re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", name) else f"`{name}`"
        text = settings.condition.rstrip()
        return {"condition": f"{text} {token}" if text else token, "insert": ""}

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


@dataclass
class RemoveDoubleSettings:
    distance_nm: float = param(100.0, label="closer than", unit="nm", min=0.0,
                               help="two used ROIs of one file whose centres are "
                                    "closer than this are doubles")


@register("Analysis/Regions/Remove double ROIs")
class RemoveDoubleROIs(Plugin):
    """Of two used ROIs closer than a distance, keeps the one with more
    localizations and excludes the other."""

    Settings = RemoveDoubleSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        """How many it would exclude, as the distance is typed: the count of
        localizations per ROI is the expensive part, a few ms per ROI."""
        project = getattr(ctx, "rois", None) if ctx is not None else None
        if project is None or not project.rois or len(project.rois) > 2000:
            return FormState()
        try:
            going = double_ids(project, settings.distance_nm)
        except Exception as error:                   # noqa: BLE001 -- a hint only
            return FormState(notice=Notice(str(error)))
        used = sum(r.use for r in project.rois.values())
        return FormState(notice=Notice(f"would exclude {len(going)} of {used} used ROIs",
                                       kind="info"))

    def run(self, ctx: Context, settings: RemoveDoubleSettings) -> Result:
        project = ctx.rois
        if project is None or not project.rois:
            raise ValueError("there are no ROIs")
        going = double_ids(project, settings.distance_nm)
        with project.edit(f"{self.name}: closer than {settings.distance_nm:g} nm"):
            for roi_id in going:
                project.rois[roi_id].use = False
        used = sum(r.use for r in project.rois.values())
        numbers = ", ".join(str(project.rois[i].number) for i in going[:20])
        more = f" and {len(going) - 20} more" if len(going) > 20 else ""
        return Result(text=(f"excluded {len(going)} double ROIs"
                            + (f" ({numbers}{more})" if going else "")
                            + f"; {used} used"),
                      settings=settings, rois=[],
                      values={"excluded": len(going), "used": used})


def double_ids(project, distance_nm: float) -> List[str]:
    """The used ROIs `order.doubles` would exclude, in the project's order.
    An ROI's size is the localizations it shows -- the layer's filter and
    grouping, as in the ROI manager -- so a double of a bright site is the
    dimmer one."""
    from ..roi_manager.order import doubles
    ids = [i for i in project.ordered_ids() if project.rois[i].use]
    if len(ids) < 2:
        return []
    rois = [project.rois[i] for i in ids]
    counts = [len(project.indices(r)) for r in rois]
    files = [project.file_number(r.file_id) for r in rois]
    double = doubles([r.center for r in rois], counts, distance_nm, files)
    return [i for i, d in zip(ids, double) if d]

