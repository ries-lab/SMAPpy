"""Average ROIs: every used ROI centred, turned to one orientation, and
superimposed into one table -- SMAP's AverageSites, with the centre and the
angle of its kinetochore variant (AverageSites_kinetochore) made choices.

SMAP has several averaging plugins, each one fixed choice: AverageSites
shifts each site by its centre and turns nothing; AverageSites_window and
_NPC read the site's own rotation (``xnmrot``) and a LocMoFit model's
parameters; AverageSites_kinetochore centres on a point of one annotation
line and turns by the angle of the other.  Here the *centre* is the ROI's,
a pair of measured values (a cluster's or a circle fit's centre) or a point
of a named line (`ROI.lines`), and the *orientation* none, a named line or
a measured angle: one plugin covers what they did, apart from LocMoFit's
model, which is a later port.

Two departures, both because the output is a table that is used further:

* **The frames of each ROI are moved apart.**  The new table is one file,
  and merging blinks links localizations in consecutive frames close
  together in space -- after superimposing, two ROIs' blinks in the same
  frames are close together, and would be merged into one.  Each ROI's
  frames are shifted by a multiple of the table's frame span, so linking
  never crosses ROIs; the frame in its file is kept in `frame_in_file`.
  SMAP regroups the averaged table as it is and merges them.
* **The average is centred on (0, 0)**, so `x_nm` is the distance from the
  centre along the orientation and `y_nm` across it.  SMAP moves it to the
  median of the data, to lie among the other files.

A line's angle and a measured angle are taken as SMAP's (``rotcoord``):
the ROI is turned by minus the angle, so the line ends pointing along +x.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import Context, FormState, Plot, Plugin, Result, param, register

CENTRES = [("roi", "the ROI's centre"), ("measured", "measured values"),
           ("line", "a point of a line")]
ORIENTATIONS = [("none", "as it is"), ("line", "a line"),
                ("measured", "a measured angle")]
LINE_POINTS = [("start", "its start"), ("middle", "its middle"), ("end", "its end")]
OUTPUTS = [("file", "a new file"), ("table", "the table")]


def line_choices():
    """Without a session, the one line a project starts with."""
    return [("direction", "direction")]


@dataclass
class AverageCentre:
    by: str = param("roi", label="centre on", choices=CENTRES,
                    help="the point of each ROI that goes to (0, 0)")
    x: str = param("centre_x_nm", label="x value",
                   help="the column of the results table with the centre's x, "
                        "in nm (Cluster shape: centre_x_nm; Circle fit: x_nm)")
    y: str = param("centre_y_nm", label="y value",
                   help="the same for y")
    line: str = param("direction", label="line", choices=line_choices,
                      help="the line whose point is the centre")
    point: str = param("middle", label="point", choices=LINE_POINTS,
                       help="which point of the line")


@dataclass
class AverageOrientation:
    by: str = param("none", label="turn by", choices=ORIENTATIONS,
                    help="what each ROI is turned by, so that it points along +x")
    line: str = param("direction", label="line", choices=line_choices,
                      help="the line that ends pointing along +x")
    angle: str = param("angle_deg", label="angle value",
                       help="the column of the results table with the angle, "
                            "in degrees from +x towards +y")


@dataclass
class AverageROIsSettings:
    centre: AverageCentre = field(default_factory=AverageCentre)
    orientation: AverageOrientation = field(default_factory=AverageOrientation)
    filtered: bool = param(False, label="only what the layers show",
                           help="only the localizations that pass a layer's "
                                "filter; off: every localization in the ROI")
    output: str = param("file", label="write to", choices=OUTPUTS,
                        help="a file added to the session, beside the data; or "
                             "in place of the table (undoable)")


# ------------------------------------------------------------- the algorithm

def superimpose(x, y, centre, angle_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Positions moved so that ``centre`` is at (0, 0) and turned by
    ``-angle_deg``: a direction at that angle ends along +x (SMAP's
    ``rotcoord``)."""
    a = np.radians(angle_deg)
    dx = np.asarray(x, dtype=float) - centre[0]
    dy = np.asarray(y, dtype=float) - centre[1]
    c, s = np.cos(a), np.sin(a)
    return c * dx + s * dy, -s * dx + c * dy


def line_point(line, which: str) -> np.ndarray:
    """The start, middle or end of a line ``[[x0, y0], [x1, y1]]``."""
    a, b = np.asarray(line, dtype=float)
    return {"start": a, "end": b}.get(which, (a + b) / 2)


def placement(roi, row: Dict[str, Any], settings: AverageROIsSettings
              ) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    """``(centre, angle, why)`` of one ROI: where it goes and how far it
    turns, from the ROI, its results row and its lines; None and why not
    when it lacks what the settings ask for."""
    from ..roi_manager.core import line_angle_length
    c, o = settings.centre, settings.orientation
    if c.by == "measured":
        x, y = row.get(c.x), row.get(c.y)
        if not _finite(x) or not _finite(y):
            return None, None, f"no {c.x if not _finite(x) else c.y}"
        centre = np.array([x, y], dtype=float)
    elif c.by == "line":
        if c.line not in roi.lines:
            return None, None, f"no line {c.line}"
        centre = line_point(roi.lines[c.line], c.point)
    else:
        centre = np.asarray(roi.center, dtype=float)
    if o.by == "line":
        if o.line not in roi.lines:
            return None, None, f"no line {o.line}"
        angle = line_angle_length(roi.lines[o.line])[0]
    elif o.by == "measured":
        angle = row.get(o.angle)
        if not _finite(angle):
            return None, None, f"no {o.angle}"
        angle = float(angle)
    else:
        angle = 0.0
    return centre, angle, ""


def _finite(value) -> bool:
    return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value))


def average(parts: List[Tuple[Any, np.ndarray, float, int]]):
    """One table of ``parts``, ``(locs, centre, angle, roi number)`` each:
    the positions superimposed, ``roi_id`` the ROI's number, and the frames
    of each ROI moved past the last one's (see the module docstring), the
    frame in its file kept as ``frame_in_file``."""
    from ..locs import Localizations, concat
    frames = [np.asarray(p[0]["frame"]) for p in parts if "frame" in p[0] and len(p[0])]
    span = int(max((f.max() for f in frames), default=0)) + 1
    tables = []
    for k, (locs, centre, angle, number) in enumerate(parts):
        columns = dict(locs.columns)
        columns["x_nm"], columns["y_nm"] = superimpose(locs["x_nm"], locs["y_nm"],
                                                       centre, angle)
        columns["roi_id"] = np.full(len(locs), number, dtype=np.int64)
        if "frame" in columns:
            frame = np.asarray(columns["frame"])
            columns["frame_in_file"] = frame
            columns["frame"] = frame + k * span
        tables.append(Localizations(columns, dict(locs.metadata)))
    return concat(tables)


# ---------------------------------------------------------------- the plugin

@register("Analysis/Regions/Average ROIs")
class AverageROIs(Plugin):
    """Centres every used ROI, turns it to one orientation and superimposes
    them all into one table."""

    Settings = AverageROIsSettings
    changes_locs = True
    # the localizations of each ROI, whatever the layer shows: a new table
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx, settings) -> FormState:
        c, o = settings.centre, settings.orientation
        shown = {"centre.x": c.by == "measured", "centre.y": c.by == "measured",
                 "centre.line": c.by == "line", "centre.point": c.by == "line",
                 "orientation.line": o.by == "line",
                 "orientation.angle": o.by == "measured"}
        project = getattr(ctx, "rois", None) if ctx is not None else None
        if project is None:
            return FormState(shown=shown)
        names = list(project.line_names)
        names += sorted({n for r in project.rois.values() for n in r.lines
                         if n not in names})
        lines = [(n, n) for n in names] or line_choices()
        return FormState(shown=shown, choices={"centre.line": lines,
                                               "orientation.line": lines})

    def run(self, ctx: Context, settings: AverageROIsSettings) -> Result:
        project = ctx.rois
        if project is None or not project.rois:
            raise ValueError("there are no ROIs: find or draw some first")
        from ..roi_manager.core import annotation_columns
        measured = {r["roi_id"]: r for r in project.results()}
        parts, skipped = [], {}
        used = [project.rois[i] for i in project.ordered_ids() if project.rois[i].use]
        for k, roi in enumerate(used):
            row = measured.get(roi.number) or annotation_columns(roi, {})
            centre, angle, why = placement(roi, row, settings)
            if centre is None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            rctx = project.region_context(roi)
            locs, _ = rctx.ungrouped()          # not the blinks a layer may show
            if settings.filtered:
                keep = np.zeros(len(locs), dtype=bool)
                for i, lay in enumerate(ctx.session.layers if ctx.session else ()):
                    if not lay.is_image:
                        keep |= np.asarray(rctx.layer_selection(i).mask, dtype=bool)
                locs = locs[keep]
            parts.append((locs, centre, angle, roi.number))
            ctx.report(f"ROI {k + 1} of {len(used)}")
        if not parts:
            raise ValueError(f"none of the {len(used)} used ROIs could be placed: "
                             + _skipped(skipped))
        out = average(parts)
        out.metadata["average_rois"] = {
            "rois": [p[3] for p in parts], "centre": settings.centre.by,
            "orientation": settings.orientation.by}
        text = (f"{len(out)} localizations of {len(parts)} ROIs superimposed"
                + (f"; left out: {_skipped(skipped)}" if skipped else ""))
        x, y = np.asarray(out["x_nm"]), np.asarray(out["y_nm"])
        reach = float(np.percentile(np.hypot(x, y), 99)) if len(x) else 1.0
        values = {"n_rois": len(parts), "n_skipped": int(sum(skipped.values())),
                  "n_localizations": len(out)}
        result = Result(text=text, settings=settings, values=values,
                        data={"rois": [p[3] for p in parts]},
                        plot=Plot(lambda fig: draw_average(fig.subplots(1, 2), x, y, reach),
                                  panels=2, size=(8, 3.6)),
                        title=f"{len(parts)} ROIs")
        if settings.output == "file" and ctx.session is not None and not ctx.session.files:
            # a session made from a table (`Session(locs)`) has no file for
            # the average to join: adding one would replace the table
            raise ValueError("the table is in no file to add the average beside: "
                             "write to the table instead")
        if settings.output == "table":
            result.locs = out
        else:
            from ..io.formats import FileInfo
            name = f"average of {len(parts)} ROIs"
            result.files = [(out, FileInfo(name=name, path=name, format="average",
                                           n=len(out)), None)]
            result.append = True
        return result


def _skipped(skipped: Dict[str, int]) -> str:
    return ", ".join(f"{n} with {why}" for why, n in skipped.items())


def draw_average(axes, x, y, reach: float, bin_nm: Optional[float] = None) -> None:
    """The average as a 2D histogram, and its profiles along x and y."""
    reach = max(reach, 1.0)
    bin_nm = bin_nm or max(reach / 60, 1.0)
    edges = np.arange(-reach, reach + bin_nm, bin_nm)
    image, _, _ = np.histogram2d(y, x, bins=(edges, edges))
    a, b = axes
    a.imshow(image, origin="upper", cmap="hot", interpolation="nearest",
             extent=(edges[0], edges[-1], edges[-1], edges[0]))
    a.set_xlabel("x (nm), along the orientation")
    a.set_ylabel("y (nm)")
    a.set_title("average", fontsize=9)
    centres = (edges[:-1] + edges[1:]) / 2
    b.plot(centres, image.sum(axis=0), label="along x")
    b.plot(centres, image.sum(axis=1), label="along y")
    b.set_xlabel("position (nm)")
    b.set_ylabel("localizations")
    b.legend(fontsize=8)
    b.set_title("profiles", fontsize=9)
