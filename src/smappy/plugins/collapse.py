"""One row per id: a track, a cluster or a merged blink instead of its
localizations.

SMAP's ``Process/Modify/GroupAny``, and ``Connect2Unconnect`` (one
channel's localizations replaced by its merged blinks): the table replaced
by one row per value of an id column, each column reduced by the rule
merging blinks uses (`group.combine`).  Its use is a smaller file -- a table of tracks, of
clusters, of blinks, to save or hand on -- and the localizations are gone
from it, which an undo brings back.  A form that writes the group's values
into every member's row is left out: it keeps the table's size, which is
the point here, and merged blinks on a layer already show it.

The work is `smappy.table_edit.collapse`.  Departures from SMAP: a row
with no id (0 or below) is kept as it is or dropped, by choice, where
GroupAny grouped a file's into one; every channel at once, where
Connect2Unconnect takes one; the grouping columns are renamed
``n_merged`` and dropped rather than left to be merged a second time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import Context, FormState, Plugin, Result, param, register


def _id_choices():
    return [("group_id", "group_id"), ("track_id", "track_id"),
            ("cluster_id", "cluster_id")]


@dataclass
class CollapseSettings:
    column: str = param("group_id", label="one row per", choices=_id_choices,
                        help="an id column: merged blinks (group_id), tracks, "
                             "clusters, or any whole-number column")
    unassigned: str = param("keep", label="rows without an id",
                            choices=[("keep", "keep, one row each"),
                                     ("remove", "remove")],
                            help="a row whose id is 0 or below: noise, a "
                                 "localization in no track")


@register("Analysis/Process/Collapse by id")
class Collapse(Plugin):
    """Replace the table with one row per id -- a blink, a track, a cluster --
    its columns combined as merged blinks are."""

    Settings = CollapseSettings
    changes_locs = True
    # the localizations are what is combined; a merged layer's rows are blinks
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: CollapseSettings) -> FormState:
        state = FormState()
        if ctx is not None and ctx.session is not None and len(ctx.locs):
            from ..table_edit import id_candidates
            found = id_candidates(ctx.locs)
            if found:
                state.choices["column"] = [(n, n) for n in found]
        return state

    def run(self, ctx: Context, settings: CollapseSettings) -> Result:
        from ..table_edit import MERGED, collapse
        locs = ctx.locs
        if settings.column == "group_id" and "group_id" not in locs:
            raise ValueError("no group_id: merge blinks on a layer once, and "
                             "each localization has the blink it is in")
        out, notes = collapse(locs, settings.column, settings.unassigned,
                              progress=lambda text, f: ctx.report(text))
        text = (f"{len(locs)} rows to {len(out)}, one per {settings.column} "
                f"({MERGED}: how many each)")
        if notes:
            text += "; " + "; ".join(notes)
        return Result(locs=out, settings=settings, text=text)
