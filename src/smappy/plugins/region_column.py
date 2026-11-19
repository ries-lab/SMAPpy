"""Between regions and columns: what was measured per region written onto
its localizations, and an ROI made of each region an id column names.

`docs/regions.md` ("Region values as columns") has the decision: rather than
a second table with a filter of its own, a region's results go onto its rows
as ordinary columns -- `n_corners`, a cluster's precision -- and every tool
that reads a column (the layer's filter, colour by a field, the statistics,
a block repeating over an id) works on them, per layer, with nothing new to
learn.  The cost is that a histogram of such a column counts localizations,
not regions (a large cluster weighs more): the distribution over regions is
*Summarize/Histograms*.

It is a snapshot, as any column written by a plugin.  A block writes the same
columns itself at the end of its run (*write values*), so a chain keeps them
current; this plugin is for results already there.

The ROIs' own numbers (`roi_id`, what *ROI to column* used to write) are one
of the values: of the ROIs, with or without an evaluation.  So are their
lines' angles and lengths (`direction_angle_deg`, ...), drawn by hand and no
evaluation's: they need none named in *results from*.

**Column to ROIs** -- now *Segment regions*' method *from an id column*
(`segment.py`), with `column_regions` here -- is the other direction: an ROI of the ROI manager for every
id of a column (a cluster, whoever found it) with enough localizations, to
look at, review and annotate.  It used to be part of DBSCAN; on its own, any
clusterer -- a plugin dropped in, or ids read from a file -- gets it.  By
default each ROI is the cluster's own outline, its convex hull; without it,
only the cluster's centre is placed and the ROI is the ROI manager's circle or
square, so clusters meant to be measured alike are measured in the same
window, and the shape can still be switched there afterwards.  A cluster with
no hull (its localizations on one line) gets the centre too, rather than no
ROI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import Context, FormState, Plugin, Result, param, register
from ..region_runs import VALUE_RULES
from .roi import auto_only
from .roi_results import ROIsFromResultsSettings

# what *results from* is when it is empty: the ROIs, without an evaluation
NUMBERS = "the ROIs' numbers"


@dataclass
class RegionColumnSettings:
    source: str = param("", label="results from", choices=auto_only,
                        help="what was measured on the ROIs, or per region of "
                             "an id attribute; empty: the ROIs, for their numbers")
    values: str = param("", label="values",
                        help="comma separated; empty: every number the "
                             "measurement has (no measurement: roi_id and the "
                             "keywords); roi_id: the ROI's number; "
                             "<line>_angle_deg, <line>_length_nm: its lines; "
                             "<list>_keyword: its keyword's number")
    prefix: str = param("", label="prefix", advanced=True,
                        help="put before each attribute's name")
    merge: str = param("last", label="overlaps", choices=VALUE_RULES, advanced=True,
                       help="a localization in two ROIs takes the later ROI's "
                            "value, the first's, or their mean")


def rows_values(rows: List[Dict[str, Any]], names: List[str]) -> Dict[str, Dict[str, float]]:
    """``{value: {region id: number}}`` from a site table's rows (``roi_id``
    and the values); ``names`` empty takes every numeric one.  A name no row
    has is an error naming the ones there are."""
    numeric = sorted({k for r in rows for k, v in r.items()
                      if k not in ("roi_id", "file_id") and isinstance(v, (int, float))})
    wanted = names or numeric
    missing = [k for k in wanted if k not in numeric]
    if missing:
        raise ValueError(f"no region has a number called {', '.join(missing)}; "
                         f"there is {', '.join(numeric) or 'none'}")
    return {k: {str(r["roi_id"]): r[k] for r in rows
                if isinstance(r.get(k), (int, float))} for k in wanted}


KEYWORD = "_keyword"     # a keyword list's column: ``<list>_keyword``


def keyword_columns(project, roi) -> Dict[str, int]:
    """An ROI's keywords as numbers, ``{<list>_keyword: k}``: its keyword's
    place in the list, from 1, so a layer filter can keep the localizations
    of the ROIs called *good* (``quality_keyword`` 1) -- a column holds
    numbers, not words.  A keyword its list no longer has is left out."""
    out = {}
    for name, keyword in roi.annotations.items():
        words = project.annotation_lists.get(name, [])
        if keyword in words and (name + KEYWORD).isidentifier():
            out[name + KEYWORD] = words.index(keyword) + 1
    return out


def own_rows(project) -> List[Dict[str, Any]]:
    """The numbers the included ROIs carry themselves, with no evaluation:
    their lines' angles and lengths (`core.line_columns`) and their
    keywords (`keyword_columns`)."""
    from ..roi_manager.core import line_columns
    out = []
    for roi_id in project.ordered_ids():
        roi = project.rois[roi_id]
        own = {**line_columns(roi, project.uses_angles), **keyword_columns(project, roi)} if roi.use else {}
        if own:
            out.append({"roi_id": roi.number, "file_id": roi.file_id, **own})
    return out


class RegionsToColumn(Plugin):
    """Writes what was measured per region onto the localizations in it.

    *Write results*' destination *the localizations* (`WriteResults`), not a
    plugin of the tree of its own any more; the class stays for the tests and
    scripts that call it, and a saved chain naming its old path is read as
    *Write results* (`plugins.MOVED`)."""

    # the path is the old one, which `MOVED` reads
    name = "Regions to attribute"
    path = "Analysis/Regions/Regions to attribute"
    Settings = RegionColumnSettings
    changes_locs = True
    # the values go onto the localizations
    grouping = "ungrouped"
    # 2: roi_id is the ROI's own number, kept when others go; 3: with nothing
    # named, the ROIs' keywords too (`keyword_columns`)
    version = "4"

    def describe(self, ctx, settings) -> FormState:
        """*results from*: the evaluations the session has."""
        if ctx is None:
            return FormState()
        from ..region_results import available
        return FormState(choices={"source": [("", NUMBERS)]
                                  + [(str(ev), str(ev)) for ev in available(ctx)]})

    def run(self, ctx: Context, settings: RegionColumnSettings) -> Result:
        from .. import region_results
        from ..chain import BLOCK_ROIS
        from ..region_runs import REGION_NUMBER, write_values
        names = [v.strip() for v in settings.values.split(",") if v.strip()]
        source = settings.source
        ev = region_results.Evaluation.parse(source)
        regions = ev.source
        if source and not region_results.is_current(ctx, ev):
            raise ValueError(region_results.missing(ctx, ev))
        if not ev.on_rois:
            numbers = False
            if REGION_NUMBER in names:
                raise ValueError(f"the regions of {regions} have no ROI numbers: "
                                 f"{regions} itself is theirs")
            values = rows_values(region_results.rows(ctx, ev), names)
        else:
            if ctx.rois is None:
                raise ValueError("there are no ROIs: find or draw some first")
            numbers = not source or REGION_NUMBER in names
            measured = [n for n in names if n != REGION_NUMBER]
            values = {}
            if source:
                if measured or not names:
                    values = rows_values(region_results.rows(ctx, ev), measured)
            elif not names:
                # the numbers, and what the user said of the ROIs: which are
                # used is a `roi_id` (-1: none), the keywords a column a list
                own = own_rows(ctx.rois)
                said = sorted({k for r in own for k in r if k.endswith(KEYWORD)})
                if said:
                    values = rows_values(own, said)
            elif measured:
                own = own_rows(ctx.rois)
                drawn = {k for r in own for k in r if k not in ("roi_id", "file_id")}
                unknown = [n for n in measured if n not in drawn]
                if unknown:
                    raise ValueError(f"{', '.join(unknown)}: name the measurement "
                                     "that measured them in *results from*"
                                     + (f"; the ROIs' lines give {', '.join(sorted(drawn))}"
                                        if drawn else ""))
                values = rows_values(own, measured)
        out, members = write_values(ctx, ctx.locs, regions, values, settings.merge,
                                    {"source": source or NUMBERS}, settings.prefix,
                                    numbers=numbers)
        n = len(ctx.locs)
        columns = [settings.prefix + k for k in values] + (
            [settings.prefix + REGION_NUMBER] if numbers else [])
        inside = np.zeros(n, dtype=bool)
        for _, rows in members:
            inside[rows] = True
        where = "ROIs" if regions == BLOCK_ROIS else f"{regions} regions"
        return Result(locs=out, settings=settings,
                      text=f"{', '.join(columns)}: {len(members)} {where} over "
                           f"{int(inside.sum())} localizations, "
                           f"{n - int(inside.sum())} in none",
                      values={"regions": len(members), "inside": int(inside.sum())})


# ------------------------------------------------------------ write results

DESTINATIONS = (("localizations", "the localizations, as attributes"),
                ("rois", "the ROIs, as their properties"))


@dataclass
class WriteResultsSettings:
    to: str = param("localizations", label="write to", choices=DESTINATIONS,
                    help="onto the localizations of each region, as attributes to "
                         "filter or colour by; or onto each ROI, as its angle, a "
                         "range, a line, a keyword or its centre")
    localizations: RegionColumnSettings = param(default_factory=RegionColumnSettings,
                                                label="to the localizations")
    rois: ROIsFromResultsSettings = param(default_factory=ROIsFromResultsSettings,
                                          label="to the ROIs")


def _prefixed(state: FormState, prefix: str) -> FormState:
    """A part's form state as the whole form's: its names under ``prefix``."""
    out = FormState(notice=state.notice)
    for name in ("hints", "active", "choices", "shown", "titles"):
        setattr(out, name, {f"{prefix}.{k}": v for k, v in getattr(state, name).items()})
    return out


@register("Analysis/Regions/Write results")
class WriteResults(Plugin):
    """Writes what was measured per region somewhere it is used: onto the
    localizations, as attributes, or onto the ROIs, as their properties."""

    Settings = WriteResultsSettings
    changes_locs = True
    # the attributes go onto the localizations
    grouping = "ungrouped"
    version = "1"

    def changes_locs_for(self, settings) -> bool:
        return settings.to == "localizations"

    def _part(self, settings):
        from .roi_results import ROIsFromResults
        if settings.to == "rois":
            return ROIsFromResults(), settings.rois
        return RegionsToColumn(), settings.localizations

    def describe(self, ctx, settings) -> FormState:
        """Only the chosen destination's part, with what its own form says."""
        plugin, part = self._part(settings)
        key = "rois" if settings.to == "rois" else "localizations"
        state = _prefixed(plugin.describe(ctx, part), key)
        state.shown.update({"localizations": key == "localizations", "rois": key == "rois"})
        return state

    def run(self, ctx: Context, settings: WriteResultsSettings) -> Result:
        if settings.to not in dict(DESTINATIONS):
            raise ValueError(f"write to {settings.to!r}: one of "
                             + ", ".join(dict(DESTINATIONS)))
        plugin, part = self._part(settings)
        result = plugin.run(ctx, part)
        result.settings = settings
        return result


# ------------------------------------------------------------ column to ROIs

#: offered without a session; with one, the table's whole-number attributes
#: (`id_attributes`), so a name is chosen rather than typed
COMMON_IDS = ("cluster_id", "track_id", "tid")


def common_ids():
    return [(c, c) for c in COMMON_IDS]


def id_attributes(locs) -> List[str]:
    """The table's whole-number attributes an ROI per id can be made of:
    the id columns first (`ids.is_id`: cluster_id, track_id, tid, ...), then
    any other integer column but the ones that only count or index rows."""
    from ..ids import is_id
    from ..tracking.columns import integer_columns
    skip = {"frame", "filenumber", "channel", "group_id", "n_in_group"}
    names = [n for n in integer_columns(locs) if n not in skip]
    return sorted(names, key=lambda n: (not is_id(n), n))


@dataclass
class ColumnROIsSettings:
    column: str = param("cluster_id", label="attribute", choices=common_ids,
                        help="an id attribute of the table; -1 is in no region")
    min_size: int = param(10, label="at least", unit="locs", min=1,
                          help="the localizations an id needs to become an ROI")
    outline: bool = param(True, label="the cluster's own outline",
                          help="each ROI the convex hull of its cluster; off: only "
                               "the centre, with the ROI manager's circle or square")
    membership: str = param("shape", label="an ROI holds",
                            choices=[("shape", "what is in its shape"),
                                     ("id", "its id alone")],
                            help="its shape: every localization inside, whatever its "
                                 "id; its id: that id's localizations only, as a "
                                 "track whose path others cross")
    draw: str = param("auto", label="draw it as",
                      choices=[("auto", "auto"), ("track", "a track"),
                               ("outline", "an outline")],
                      help="in the ROI manager: a line through its localizations "
                           "in frame order, or the hull round them; auto: a track "
                           "for a track attribute")
    keep: str = param("", label="keep", kind="text",
                      help="which ids become ROIs, by their trace's values, e.g. "
                           "eccentricity >= 3 and displacement_nm >= 300 for "
                           "motors; empty: all")
    both_colours: bool = param(False, label="only in both colours",
                               help="only the ids seen in both colours of a "
                                    "two-colour table (MINFLUX dual colour's pair_id)")
    angle: str = param("none", label="angle",
                       choices=[("none", "none"),
                                ("axis", "long axis, forward in time"),
                                ("ends", "first to last localization")],
                       help="the ROI's angle from its trace: for directed tracks "
                            "(motors); none for diffusion, where it means nothing")


def hull(xy: np.ndarray) -> Optional[List[List[float]]]:
    """The convex hull of a cluster's positions, as polygon vertices; None
    when there are fewer than three points or they lie on a line."""
    from scipy.spatial import ConvexHull, QhullError
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return None
    try:
        h = ConvexHull(xy)
    except (QhullError, ValueError):
        return None
    return xy[h.vertices].tolist()


#: what *keep* of *from an id attribute* can name: each id's localizations
#: as a trace (`traces.statistics`), in frame order (or ``time_s``)
TRACE_VALUES = ("n_locs", "displacement_nm", "std_long_nm", "std_short_nm",
                "eccentricity", "trend_nm", "angle_deg", "ends_angle_deg",
                "velocity_nm_per_s", "linearity", "n_colours")


def trace_values(locs, column: str) -> Dict[int, Dict[str, float]]:
    """Each id of ``column``'s localizations measured as a trace, in time
    order: `TRACE_VALUES`.  ``angle_deg`` is the long axis pointing from the
    trace's first quarter to its last (`traces.axes`), ``ends_angle_deg``
    the direction from its first localization to its last; both 0-360,
    anticlockwise from +x.  ``velocity_nm_per_s`` the displacement over the
    duration (per frame where there is no ``time_s``); ``linearity`` the
    squared correlation of the position along the long axis with time --
    near 1 for a motor walking on, near 0 for one stuck or diffusing back and
    forth along its track, which is as elongated; ``n_colours`` how many
    colours (`dual_colour.colour_column`) the id holds, 2 for a molecule seen
    in both.  All ids at once, no loop over them."""
    from .. import traces as T
    time = np.asarray(locs["time_s"] if "time_s" in locs else locs["frame"], np.float64)
    r = T.runs(locs[column], time)
    if not len(r.ids):
        return {}
    x = np.asarray(locs["x_nm"], np.float64)[r.order]
    y = np.asarray(locs["y_nm"], np.float64)[r.order]
    t = time[r.order]
    st = T.statistics(r, x, y, t)
    first, last = r.start, r.start + r.n - 1
    ends = np.degrees(np.arctan2(y[last] - y[first], x[last] - x[first])) % 360.0
    columns = {**{k: np.asarray(st[k], float) for k in TRACE_VALUES if k in st},
               "ends_angle_deg": ends,
               "velocity_nm_per_s": np.asarray(st["velocity"], float),
               "linearity": _linearity(r, x, y, t, np.asarray(st["angle_deg"], float)),
               "n_colours": _n_colours(locs, r)}
    return {int(k): {name: float(v[i]) for name, v in columns.items()}
            for i, k in enumerate(r.ids)}


def _linearity(r, x, y, t, angle_deg) -> np.ndarray:
    """Per run, r^2 of its position along its long axis against time, from
    sums over each run's rows (sorted run by run)."""
    run = np.repeat(np.arange(len(r.ids)), r.n)
    a = np.radians(np.nan_to_num(angle_deg))[run]
    s = x * np.cos(a) + y * np.sin(a)
    s = s - np.add.reduceat(s, r.start)[run] / r.n[run]
    tc = t - np.add.reduceat(t, r.start)[run] / r.n[run]
    sst = np.add.reduceat(s * tc, r.start)
    sss = np.add.reduceat(s * s, r.start)
    stt = np.add.reduceat(tc * tc, r.start)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sst ** 2 / (sss * stt)
    return np.where(r.n > 2, out, np.nan)


def _n_colours(locs, r) -> np.ndarray:
    """Per run, how many colours its localizations have."""
    from ..dual_colour import colour_column
    column = colour_column(locs)
    if column not in locs:
        return np.ones(len(r.ids))
    c = np.asarray(locs[column], np.int64)[r.order]
    run = np.repeat(np.arange(len(r.ids)), r.n)
    pairs = np.unique(np.column_stack([run, c]), axis=0)
    return np.bincount(pairs[:, 0], minlength=len(r.ids)).astype(float)


def column_regions(locs, column: str, min_size: int = 1, outline: bool = True
                   ) -> List[Dict[str, Any]]:
    """One entry per id of ``column`` with at least ``min_size`` rows:
    ``{"id", "rows", "center", "polygon", "file"}`` -- the centre the mean
    position, the polygon the convex hull (None when ``outline`` is off, or
    the points are fewer than three or on a line), the file that of its
    first row."""
    from ..region_runs import split_ids
    if column not in locs:
        raise ValueError(f"the table has no {column!r}: it has "
                         f"{', '.join(sorted(locs.keys()))}")
    values = np.asarray(locs[column])
    if values.dtype.kind not in "iub" and not (
            values.dtype.kind == "f" and np.all(np.mod(values[np.isfinite(values)], 1) == 0)):
        raise ValueError(f"{column} is not an id attribute: its values are not whole numbers")
    x = np.asarray(locs["x_nm"], dtype=float)
    y = np.asarray(locs["y_nm"], dtype=float)
    files = (np.asarray(locs["filenumber"]) if "filenumber" in locs
             else np.zeros(len(locs), dtype=np.int64))
    out = []
    ids, rows = split_ids(values)
    for k, members in zip(ids, rows):
        if len(members) < min_size:
            continue
        xy = np.column_stack((x[members], y[members]))
        out.append({"id": int(k), "rows": members,
                    "center": [float(xy[:, 0].mean()), float(xy[:, 1].mean())],
                    "polygon": hull(xy) if outline else None,
                    "file": int(files[members[0]])})
    return out
