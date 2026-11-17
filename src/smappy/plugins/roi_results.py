"""ROIs from results: a value measured in each ROI written onto the ROI --
its angle, a range of any attribute, a line, a keyword, or its centre.

What a structure is like is a measurement, and which measurement is best
depends on the structure: a motor's track points along its long axis
forward in time (the *Trace inspector*'s ``angle_deg`` in each ROI, or *Segment regions*'
when it made the ROIs), a tail ends at a frame, a filament lies along a
fitted line.  So nothing here measures anything: each target reads columns
of what the ROIs' steps or their segmenter measured, chosen by name, and
any plugin -- a user's own too -- can be the source.

A Regions plugin rather than a button of the ROI manager, as *Centre ROIs*
is, so that a chain does it between measuring the ROIs and measuring them
again in their new frame, range or category, and a batch with it.  One step
of the ROI undo.

Which ROIs it changes is a choice, since a property set by hand
(`ROI.set_by`) is worth more than one measured again: by default those are
left alone, so new ROIs can be given theirs without undoing corrections.
Each target is its own property there (``angle``, ``range:<attribute>``,
``line:<name>``, ``keyword:<list>``): a hand-drawn line does not protect the
angle.  The centre is never protected: placed by hand it is placed by eye.

Every target but the keyword changes what is measured in the ROI -- its
frame, its cut, its signature -- so those ROIs' results go out of date and
are measured again; a keyword is a note, and nothing is measured again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import Context, FormState, Notice, Plugin, Result, param, register

TARGETS = (("angle", "angle"), ("range", "a range"), ("line", "a line"),
           ("keyword", "a keyword"), ("centre", "centre"))
WHICH = (("not_hand", "all but those set by hand"),
         ("unset", "only those not set yet"),
         ("all", "all, those set by hand too"))
OTHERWISE_LEAVE, OTHERWISE_CLEAR = "", "-"


def no_column():
    """Without a session there are no measured columns to choose from."""
    return [("", "none")]


def open_column():
    return [("", "open")]


@dataclass
class AngleTarget:
    column: str = param("", label="angle from", choices=no_column,
                        help="a measured column in degrees, anticlockwise from +x: "
                             "the Trace inspector's or Segment regions' angle_deg, say")


def range_choices():
    """Without a session, the two attributes every project offers."""
    return [("frame", "frame"), ("z_nm", "z_nm")]


@dataclass
class RangeTarget:
    attribute: str = param("frame", label="attribute", choices=range_choices,
                           help="the attribute the range is on: frame, z_nm, time_s, ...")
    mode: str = param("columns", label="from",
                      choices=[("columns", "a low and a high column"),
                               ("centre", "a centre and a width")],
                      help="the range's two ends, or a centre plus and minus k widths")
    low: str = param("", label="low", choices=open_column,
                     help="a measured column with the range's low end; open: from the start")
    high: str = param("", label="high", choices=open_column,
                      help="the high end; open: to the end")
    centre: str = param("", label="centre", choices=no_column, help="the middle")
    width: str = param("", label="width", choices=no_column,
                       help="a spread, such as a standard deviation")
    k: float = param(2.0, label="k", min=0.0, help="widths on either side")


@dataclass
class LineTarget:
    name: str = param("line", label="line", help="the line's name; a new name joins "
                                                 "the project's line names")
    mode: str = param("axis", label="from",
                      choices=[("axis", "a centre, an angle and a length"),
                               ("ends", "its two ends")],
                      help="as the tracking plugins measure an axis, or two points")
    x: str = param("", label="centre x", choices=no_column, help="nm")
    y: str = param("", label="centre y", choices=no_column, help="nm")
    angle: str = param("", label="angle", choices=no_column,
                       help="degrees, anticlockwise from +x; the line points that way")
    length: str = param("", label="length", choices=no_column, help="nm")
    x0: str = param("", label="start x", choices=no_column, help="nm")
    y0: str = param("", label="start y", choices=no_column, help="nm")
    x1: str = param("", label="end x", choices=no_column, help="nm")
    y1: str = param("", label="end y", choices=no_column, help="nm")


@dataclass
class KeywordTarget:
    list_name: str = param("", label="list", choices=no_column,
                           help="the keyword list (the ROI manager's *lists*)")
    condition: str = param("", label="where", kind="text",
                           help="a condition on the measured columns, as Filter ROIs "
                                "writes it: eccentricity > 4")
    keyword: str = param("", label="keyword", choices=no_column,
                         help="given where the condition holds")
    otherwise: str = param(OTHERWISE_LEAVE, label="otherwise", choices=no_column,
                           help="where it does not: another keyword, none, or "
                                "left as it is")


@dataclass
class CentreTarget:
    x: str = param("", label="x", choices=no_column, help="the centre's x, nm")
    y: str = param("", label="y", choices=no_column, help="the centre's y, nm")


@dataclass
class ROIsFromResultsSettings:
    target: str = param("angle", label="set", choices=TARGETS,
                        help="which property of the ROIs is set from the results")
    angle: AngleTarget = param(default_factory=AngleTarget, label="angle")
    range: RangeTarget = param(default_factory=RangeTarget, label="a range")
    line: LineTarget = param(default_factory=LineTarget, label="a line")
    keyword: KeywordTarget = param(default_factory=KeywordTarget, label="a keyword")
    centre: CentreTarget = param(default_factory=CentreTarget, label="centre")
    which: str = param("not_hand", label="change", choices=WHICH,
                       help="which ROIs get the measured value")
    used_only: bool = param(True, label="used ROIs only",
                            help="leave the ROIs not ticked *use* as they are")


# ------------------------------------------------------------------ the values

def number(row: Dict[str, Any], column: str) -> float:
    """A row's value of ``column`` as a float; NaN when there is none."""
    if not column:
        return float("nan")
    try:
        return float(row.get(column))
    except (TypeError, ValueError):
        return float("nan")


def angle_value(row, target: AngleTarget) -> Optional[float]:
    from ..roi_manager.core import normal_angle
    a = number(row, target.column)
    return normal_angle(a) if np.isfinite(a) else None


def whole_attribute(column: str) -> bool:
    """Is the attribute counted in whole numbers -- frames, ids -- so a
    measured end is rounded to one?"""
    name = column.lower()
    return name == "frame" or name.endswith("_id") or name in ("tid", "id")


def range_value(row, target: RangeTarget) -> Optional[list]:
    """``[low, high]``, an open end None; None when the row has neither of
    the columns asked for, or the range is empty."""
    if target.mode == "centre":
        c, w = number(row, target.centre), number(row, target.width)
        if not (np.isfinite(c) and np.isfinite(w)):
            return None
        half = abs(w) * float(target.k)
        ends = [c - half, c + half]
    else:
        ends = [number(row, target.low), number(row, target.high)]
        if (target.low and not np.isfinite(ends[0])) or \
                (target.high and not np.isfinite(ends[1])):
            return None                     # asked for and not measured: leave it
    lo, hi = (float(v) if np.isfinite(v) else None for v in ends)
    if whole_attribute(target.attribute):
        lo, hi = (None if v is None else float(np.round(v)) for v in (lo, hi))
    if (lo is None and hi is None) or (lo is not None and hi is not None and hi < lo):
        return None
    return [lo, hi]


def line_value(row, target: LineTarget) -> Optional[list]:
    """``[[x0, y0], [x1, y1]]``: the two ends, or the centre less and plus
    half the length along the angle -- so the line points the way the
    angle does, its start behind the centre."""
    if target.mode == "ends":
        p = [number(row, c) for c in (target.x0, target.y0, target.x1, target.y1)]
        if not np.all(np.isfinite(p)) or (p[0], p[1]) == (p[2], p[3]):
            return None
        return [[p[0], p[1]], [p[2], p[3]]]
    cx, cy = number(row, target.x), number(row, target.y)
    a, length = number(row, target.angle), number(row, target.length)
    if not np.all(np.isfinite([cx, cy, a, length])) or length <= 0:
        return None
    u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) * length / 2
    return [[cx - u[0], cy - u[1]], [cx + u[0], cy + u[1]]]


def centre_value(row, target: CentreTarget) -> Optional[list]:
    x, y = number(row, target.x), number(row, target.y)
    return [x, y] if np.isfinite(x) and np.isfinite(y) else None


def keyword_values(project, target: KeywordTarget) -> Dict[str, Optional[str]]:
    """``{roi id: keyword}`` -- the keyword where the condition holds,
    *otherwise* where it does not (None: clear it); an ROI to be left as it
    is is not in it."""
    lists = project.annotation_lists
    if target.list_name not in lists:
        raise ValueError(f"no keyword list {target.list_name!r}: the ROI manager's "
                         f"lists are {', '.join(lists) or 'none'}")
    words = lists[target.list_name]
    for word in (target.keyword,) + ((target.otherwise,) if target.otherwise not in
                                     (OTHERWISE_LEAVE, OTHERWISE_CLEAR) else ()):
        if word not in words:
            raise ValueError(f"{word!r} is not in the list {target.list_name!r}: "
                             f"{', '.join(words)}")
    if not target.condition.strip():
        raise ValueError("write the condition the keyword is given on")
    passed = project.filter_rois(target.condition)
    out = {}
    for roi_id, ok in passed.items():
        if ok:
            out[roi_id] = target.keyword
        elif target.otherwise == OTHERWISE_CLEAR:
            out[roi_id] = None
        elif target.otherwise != OTHERWISE_LEAVE:
            out[roi_id] = target.otherwise
    return out


def prop_of(settings: ROIsFromResultsSettings) -> str:
    """The property of `ROI.set_by` the chosen target writes."""
    t = settings.target
    if t == "line":
        return f"line:{settings.line.name}"
    if t == "keyword":
        return f"keyword:{settings.keyword.list_name}"
    if t == "range":
        return f"range:{settings.range.attribute}"
    return t


def columns_read(settings: ROIsFromResultsSettings) -> List[str]:
    """The measured columns the chosen target reads."""
    t = settings.target
    part = getattr(settings, t)
    if t == "angle":
        names = [part.column]
    elif t == "range":
        names = [part.centre, part.width] if part.mode == "centre" else [part.low, part.high]
    elif t == "line":
        names = ([part.x0, part.y0, part.x1, part.y1] if part.mode == "ends"
                 else [part.x, part.y, part.angle, part.length])
    elif t == "centre":
        names = [part.x, part.y]
    else:
        names = []
    return [n for n in names if n]


def new_values(project, settings: ROIsFromResultsSettings) -> Tuple[dict, dict]:
    """``({roi id: value}, counts)``: what each ROI is to get, and how many
    were passed over and why.  Applies nothing."""
    t = settings.target
    if t not in dict(TARGETS):
        raise ValueError(f"set is one of {', '.join(dict(TARGETS))}, not {t!r}")
    if t == "range" and not settings.range.attribute.strip():
        raise ValueError("choose the attribute the range is on")
    if settings.which not in dict(WHICH):
        raise ValueError(f"change is one of {', '.join(dict(WHICH))}, not {settings.which!r}")
    prop = prop_of(settings)
    if t == "keyword":
        wanted = keyword_values(project, settings.keyword)
    else:
        needed = columns_read(settings)
        if not needed:
            raise ValueError(f"choose the measured columns the {dict(TARGETS)[t]} "
                             "is read from")
        rows = project.sort_rows()
        known = set(project.sort_columns())
        missing = [c for c in needed if c not in known]
        if missing:
            raise ValueError(f"no ROI has {', '.join(missing)}: measure the ROIs first, "
                             "or choose another column")
        read = {"angle": angle_value, "range": range_value,
                "line": line_value, "centre": centre_value}[t]
        part = getattr(settings, t)
        wanted = {roi_id: read(rows.get(roi_id) or {}, part) for roi_id in project.rois}
    out = {}
    counts = {"by hand": 0, "already set": 0, "not used": 0, "not measured": 0,
              "left as they were": 0}
    for roi_id, roi in project.rois.items():
        if settings.used_only and not roi.use:
            counts["not used"] += 1
            continue
        # a centre is never protected: placed by hand it is placed by eye,
        # roughly, and a measured one is always the better
        if t != "centre" and settings.which == "not_hand" and roi.source(prop) == "hand":
            counts["by hand"] += 1
            continue
        if t != "centre" and settings.which == "unset" and roi.source(prop):
            counts["already set"] += 1
            continue
        if t == "keyword" and roi_id not in wanted:
            counts["left as they were"] += 1      # *otherwise*: leave it
            continue
        if roi_id not in wanted or (wanted[roi_id] is None and t != "keyword"):
            counts["not measured"] += 1
            continue
        out[roi_id] = wanted[roi_id]
    return out, counts


def apply(project, settings: ROIsFromResultsSettings, values: dict, who: str) -> int:
    """Write ``values`` onto the ROIs (inside the caller's `ROIProject.edit`)
    and note ``who`` set them; how many changed."""
    from ..roi_manager.core import line_points
    t, prop = settings.target, prop_of(settings)
    changed = 0
    for roi_id, value in values.items():
        roi = project.rois[roi_id]
        if t == "angle":
            before, roi.angle = roi.angle, float(value)
        elif t == "range":
            column = settings.range.attribute
            before = roi.ranges.get(column)
            roi.set_range(column, value)          # a new dict: see `copy_rois`
            value = roi.ranges.get(column)
        elif t == "line":
            name = settings.line.name
            before = roi.lines.get(name)
            roi.lines = {**roi.lines, name: line_points(value)}     # replaced
            value = roi.lines[name]
        elif t == "keyword":
            name = settings.keyword.list_name
            before = roi.annotations.get(name)
            annotations = {k: v for k, v in roi.annotations.items() if k != name}
            if value:
                annotations[name] = value
            roi.annotations = annotations                          # replaced
        else:                                                       # centre
            before = list(roi.center)
            if roi.spatial:
                project.move_roi(roi_id, value)   # its polygon and lines with it
            else:
                roi.center = [float(value[0]), float(value[1])]
            value = list(roi.center)
        changed += before != value
        roi.set_source(prop, who)
    if t == "line" and settings.line.name not in project.line_names:
        project.set_line_names(list(project.line_names) + [settings.line.name])
    if t == "range" and settings.range.attribute not in project.range_columns:
        project.set_range_columns(list(project.range_columns) + [settings.range.attribute])
    return changed


# ------------------------------------------------------------------ the plugin

@register("Analysis/Regions/ROIs from results")
class ROIsFromResults(Plugin):
    """Sets a property of each ROI -- its angle, frame or z range, a line, a
    keyword or its centre -- from what was measured in it, leaving what was
    set by hand alone."""

    Settings = ROIsFromResultsSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        project = getattr(ctx, "rois", None) if ctx is not None else None
        names: List[str] = []
        lists: Dict[str, List[str]] = {}
        if project is not None:
            lists = dict(project.annotation_lists)
            names = [n for n in project.sort_columns() if n not in lists]
        angles = [n for n in names if "angle" in n.lower()]
        ordered = angles + [n for n in names if n not in angles]
        columns = [(n, n) for n in ordered] or no_column()
        opened = [("", "open")] + [(n, n) for n in ordered]
        state = FormState(shown={key: settings.target == key for key, _ in TARGETS})
        for part, fields in (("angle", ("column",)),
                             ("range", ("low", "high", "centre", "width")),
                             ("line", ("x", "y", "angle", "length", "x0", "y0", "x1", "y1")),
                             ("centre", ("x", "y"))):
            for f in fields:
                state.choices[f"{part}.{f}"] = (opened if (part, f) in (
                    ("range", "low"), ("range", "high")) else columns)
        state.choices["keyword.list_name"] = [(n, n) for n in lists] or no_column()
        words = lists.get(settings.keyword.list_name, [])
        state.choices["keyword.keyword"] = [(w, w) for w in words] or no_column()
        state.choices["keyword.otherwise"] = ([(OTHERWISE_LEAVE, "leave it"),
                                               (OTHERWISE_CLEAR, "none")]
                                              + [(w, w) for w in words])
        offered = list(project.range_columns) if project is not None else ["frame", "z_nm"]
        if settings.range.attribute and settings.range.attribute not in offered:
            offered.append(settings.range.attribute)
        state.choices["range.attribute"] = [(c, c) for c in offered]
        centred = settings.range.mode == "centre"
        for f in ("centre", "width", "k"):
            state.active[f"range.{f}"] = centred
        for f in ("low", "high"):
            state.active[f"range.{f}"] = not centred
        ends = settings.line.mode == "ends"
        for f in ("x0", "y0", "x1", "y1"):
            state.active[f"line.{f}"] = ends
        for f in ("x", "y", "angle", "length"):
            state.active[f"line.{f}"] = not ends
        if project is not None and not names:
            state.notice = Notice("no ROI has anything measured yet: run the "
                                  "per-region steps first", kind="info")
        return state

    def run(self, ctx: Context, settings: ROIsFromResultsSettings) -> Result:
        project = ctx.rois
        if project is None or not project.rois:
            raise ValueError("there are no ROIs to set")
        values, counts = new_values(project, settings)
        what = dict(TARGETS)[settings.target]
        if settings.target in ("line", "keyword", "range"):
            what += " " + {"line": settings.line.name, "keyword": settings.keyword.list_name,
                           "range": settings.range.attribute}[settings.target]
        with project.edit(f"{self.name}: {what}"):
            changed = apply(project, settings, values, self.name)
        passed = ", ".join(f"{n} {why}" for why, n in counts.items() if n)
        text = (f"{what} set on {len(values)} ROIs ({changed} changed)"
                + (f"; passed over: {passed}" if passed else ""))
        if changed and settings.target != "keyword":
            text += "; their results are out of date: measured in the new frame or cut"
        return Result(text=text, settings=settings, rois=[],
                      values={"set": len(values), "changed": changed})
