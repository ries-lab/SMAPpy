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
                       help="which of the ROIs selected in the ROI manager (all, "
                            "when none is) get the measured value")


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
    counts = {"by hand": 0, "already set": 0, "not measured": 0,
              "left as they were": 0}
    # the ROIs selected in the ROI manager, all when none is (a chain, a
    # batch): *use* is for the analysis, what is changed is chosen
    for roi_id in project.chosen():
        roi = project.rois[roi_id]
        why = may_change(roi, prop, settings.which)
        if why:
            counts[why] += 1
            continue
        if t == "keyword" and roi_id not in wanted:
            counts["left as they were"] += 1      # *otherwise*: leave it
            continue
        if roi_id not in wanted or (wanted[roi_id] is None and t != "keyword"):
            counts["not measured"] += 1
            continue
        out[roi_id] = wanted[roi_id]
    return out, counts


def write_property(project, roi_id, key: str, value) -> bool:
    """Write one property onto an ROI (inside the caller's
    `ROIProject.edit`), by its `ROI.set_by` key: ``angle``, ``centre``,
    ``range:<attribute>``, ``line:<name>``, ``keyword:<list>`` (None clears
    the keyword).  A new line name, range attribute or keyword joins the
    project's, so the ROI manager offers it.  Whether it changed."""
    from ..roi_manager.core import line_points
    roi = project.rois[roi_id]
    kind, _, name = key.partition(":")
    if key == "angle":
        before, roi.angle = roi.angle, float(value)
        after = roi.angle
    elif kind == "range":
        before = roi.ranges.get(name)
        roi.set_range(name, value)                # a new dict: see `copy_rois`
        after = roi.ranges.get(name)
        if name not in project.range_columns:
            project.set_range_columns(list(project.range_columns) + [name])
    elif kind == "line":
        before = roi.lines.get(name)
        roi.lines = {**roi.lines, name: line_points(value)}          # replaced
        after = roi.lines[name]
        if name not in project.line_names:
            project.set_line_names(list(project.line_names) + [name])
    elif kind == "keyword":
        before = roi.annotations.get(name)
        annotations = {k: v for k, v in roi.annotations.items() if k != name}
        if value:
            annotations[name] = str(value)
            words = project.annotation_lists.get(name, [])
            if value not in words:
                project.set_annotation_lists({**project.annotation_lists,
                                              name: list(words) + [str(value)]})
        roi.annotations = annotations                               # replaced
        after = roi.annotations.get(name)
    elif key == "centre":
        before = list(roi.center)
        if roi.spatial:
            project.move_roi(roi_id, value)       # its polygon and lines with it
        else:
            roi.center = [float(value[0]), float(value[1])]
        after = list(roi.center)
    else:
        raise ValueError(f"no ROI property {key!r}")
    return before != after


def may_change(roi, key: str, which: str) -> Optional[str]:
    """None when ``which`` lets a method change this property of the ROI,
    else why not: set by hand, or already set.  A centre always may: one
    placed by hand is placed by eye."""
    if key == "centre":
        return None
    if which == "not_hand" and roi.source(key) == "hand":
        return "by hand"
    if which == "unset" and roi.source(key):
        return "already set"
    return None


def apply(project, settings: ROIsFromResultsSettings, values: dict, who: str) -> int:
    """Write ``values`` onto the ROIs (inside the caller's `ROIProject.edit`)
    and note ``who`` set them; how many changed."""
    prop = prop_of(settings)
    changed = 0
    for roi_id, value in values.items():
        changed += write_property(project, roi_id, prop, value)
        project.rois[roi_id].set_source(prop, who)
    return changed


def proposal_steps(project) -> Dict[str, List[str]]:
    """``{step: [property, ...]}``: the steps whose results propose
    something for any ROI, and what."""
    out: Dict[str, List[str]] = {}
    for label, table in project.tables.items():
        keys: List[str] = []
        for proposals in getattr(table, "proposals", []) or []:
            for key in proposals or {}:
                if key not in keys:
                    keys.append(key)
        if keys:
            out[label] = keys
    return out


def apply_proposals(project, step: str, roi_ids, which: str = "not_hand",
                    keys=None, who: str = "") -> Tuple[int, int, Dict[str, int]]:
    """Take what ``step`` proposed for each of ``roi_ids`` -- each ROI its own
    proposals, as its results keep them -- the properties ``keys`` (None:
    all it proposes), leaving what ``which`` protects; one step of the ROI
    undo.  ``(ROIs changed, properties written, passed over by reason)``."""
    if which not in dict(WHICH):
        raise ValueError(f"change is one of {', '.join(dict(WHICH))}, not {which!r}")
    table = project.tables.get(step)
    if table is None:
        raise ValueError(f"no step {step!r} has measured the ROIs: "
                         + (", ".join(project.tables) or "none has"))
    who = who or step
    counts: Dict[str, int] = {"by hand": 0, "already set": 0, "nothing proposed": 0}
    rois_changed = written = 0
    with project.edit(f"Accept {step}'s proposals"):
        for roi_id in roi_ids:
            roi = project.rois.get(roi_id)
            entry = table.entry(roi_id) if roi is not None and roi_id in table else None
            proposals = dict((entry or {}).get("proposals") or {})
            if keys is not None:
                proposals = {k: v for k, v in proposals.items() if k in keys}
            if not proposals:
                counts["nothing proposed"] += 1
                continue
            touched = False
            for key, value in proposals.items():
                why = may_change(roi, key, which)
                if why:
                    counts[why] += 1
                    continue
                touched |= write_property(project, roi_id, key, value)
                roi.set_source(key, who)
                written += 1
            rois_changed += touched
    return rois_changed, written, counts


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
        chosen = len(project.chosen())
        scope = (f"of {chosen} selected" if chosen < len(project.rois)
                 else f"of all {chosen}")
        text = (f"{what} set on {len(values)} ROIs {scope} ({changed} changed)"
                + (f"; passed over: {passed}" if passed else ""))
        if changed and settings.target != "keyword":
            text += "; their results are out of date: measured in the new frame or cut"
        return Result(text=text, settings=settings, rois=[],
                      values={"set": len(values), "changed": changed})


# ------------------------------------------------------- accepting proposals

def no_step():
    """Without a session there are no steps that propose anything."""
    return [("", "none")]


def any_property():
    return [("", "all it proposes")]


@dataclass
class AcceptProposalsSettings:
    step: str = param("", label="step", choices=no_step,
                      help="the per-region step whose proposals are taken")
    property: str = param("", label="property", choices=any_property,
                          help="one of what it proposes, or all of it")
    which: str = param("not_hand", label="change", choices=WHICH,
                       help="which of the ROIs selected in the ROI manager (all, "
                            "when none is) take the proposals")


@register("Analysis/Regions/Accept proposals")
class AcceptProposals(Plugin):
    """Takes what a per-region step proposed each ROI should be -- its angle,
    a range, a line, a keyword, its centre -- as the Regions tab's *Apply*
    does, for a chain or a batch."""

    Settings = AcceptProposalsSettings
    changes_locs = False
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        project = getattr(ctx, "rois", None) if ctx is not None else None
        steps = proposal_steps(project) if project is not None else {}
        state = FormState(choices={
            "step": [(s, s) for s in steps] or no_step(),
            "property": any_property() + [(k, k) for k in steps.get(settings.step, [])]})
        if project is not None and not steps:
            state.notice = Notice("no step has proposed anything yet: run a per-region "
                                  "step that does (Trace inspector, Circle fit)",
                                  kind="info")
        return state

    def run(self, ctx: Context, settings: AcceptProposalsSettings) -> Result:
        project = ctx.rois
        if project is None or not project.rois:
            raise ValueError("there are no ROIs to change")
        steps = proposal_steps(project)
        if settings.step not in steps:
            raise ValueError(f"step {settings.step!r} has proposed nothing: "
                             + (", ".join(steps) or "no step has"))
        keys = [settings.property] if settings.property else None
        chosen = project.chosen()
        changed, written, counts = apply_proposals(project, settings.step, chosen,
                                                   settings.which, keys)
        passed = ", ".join(f"{n} {why}" for why, n in counts.items() if n)
        scope = (f"{len(chosen)} selected" if len(chosen) < len(project.rois)
                 else f"all {len(chosen)}")
        text = (f"{settings.step}'s proposals taken on {scope} ROIs: {written} "
                f"properties written, {changed} ROIs changed"
                + (f"; passed over: {passed}" if passed else ""))
        return Result(text=text, settings=settings, rois=[],
                      values={"written": written, "changed": changed})
