"""Segment regions: ROIs for the ROI manager, by one of several methods.

`docs/smap_porting/wave2_design.md`, step 11, made three plugins that each
made ROIs into one, with a *method* choice and a settings part per method,
only the chosen one shown; `wave6_design.md` (decided with Jonas,
2026-10-25) made it simpler and gave it more structures.  A method is a
model of the structure:

* **circle template** -- a ring or a disc, at one radius or over a range of
  radii (`segmentation.find_circles`): nuclear pores, clathrin-coated pits,
  any round structure.  It replaced the ring template and density peaks; a
  disc at a small radius does density peaks' job, losing only its Gaussian
  blur's tolerance for blobs of no shape in particular, which was accepted;
* **beads** -- the bright, long-lived tracks of `fiducials.find_beads`
  (`segmentation.measure_beads`), for stability and drift: *Fiducials*
  takes each used ROI as a bead, and *Stability* runs per ROI;
* **from an id column** -- an ROI per id of `cluster_id` or any whole-number
  column, the cluster's hull or the project's shape
  (`region_column.column_regions`);
* **positions from a file** -- an ROI at each x, y of a csv
  (`segmentation.read_positions`), such as *Export table* writes.

Every candidate of the first two is measured -- the shared values and its
model's own, `segmentation.VALUES` -- and **keep**, one condition in *Filter
ROIs*' syntax, decides which become ROIs (`segmentation.judge`): one field
in place of the ring template's five checks, and the structure's knowledge
in the preset rather than in the plugin.  The values go with each ROI
(``origin["values"]``) and are columns of the results table
(`roi_manager.core.segment_columns`), so *Filter ROIs*, *Sort ROIs* and the
summaries read them like anything measured.  They are kept in the origin
rather than as a "segment" evaluation because they describe the candidate
as it was found, not the ROI as it is now: an evaluation would go stale --
and its values vanish from the table -- the moment the ROI is moved or the
filter changed, and it would need a step, a signature and a table for
numbers that are never computed again.  No aliases for the old settings,
as in wave 2: chains and presets name the new ones.

A **preset** is a small YAML file of values (``NPC``: a ring of 55 nm with
the pore's checks as *keep*; ``CME``: discs of 30 to 150 nm; ``Circles``:
rings of 100 to 500 nm), shipped in ``smappy/data/segment_presets`` and the
user's under the config directory, which win over a shipped one of the same
name, as `io/conversions.py` keeps csv conversions.  Choosing one fills the
settings (`SegmentRegions.react`); the settings, not the preset's name, are
what a run uses and what the ROIs record.  A preset is fixed text and cannot
know the table, so ``NPC`` leaves SMAP's in-focus check (``psf_nm <= 150``)
for the user to add where the table has ``sigma_nm``.

Each ROI's origin names the method that made it (``origin["segment"]``)
beside the plugin, so *replace* removes the ROIs the same method made before
and leaves another method's: beads found after the pores do not delete them.

The circle template and beads search the file the ROI manager shows, as the
manager filters it -- a z or frame filter there limits the search, so
neither has a setting for it; from an id column reads the localizations of
every file, each ROI going to its own file, and positions from a file puts
each position on the file it names.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import (Context, FormState, Notice, Plot, Result, param, register,
               settings_values)
from .. import segmentation
from .region_column import ColumnROIsSettings, column_regions
from .roi import SegmenterPlugin, current_file, replace_param

PATH = "Analysis/Regions/Segment regions"

METHODS = (("circle", "circle template"), ("beads", "beads"),
           ("ids", "from an id column"), ("file", "positions from a file"))
#: the methods that measure each candidate and judge it by *keep*
MEASURED = ("circle", "beads")

SHIPPED = Path(__file__).resolve().parent.parent / "data" / "segment_presets"
SUFFIX = ".yaml"


# ------------------------------------------------------------------ presets

def directories() -> List[Path]:
    """Where presets are looked for: the user's first, then the shipped ones."""
    from ..config import config_dir
    return [config_dir() / "segment_presets", SHIPPED]


def _flat(values: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def presets() -> Dict[str, Dict[str, Any]]:
    """Every preset by name, as dotted setting names and values; the user's
    win over a shipped one of the same name.  A file that cannot be read is
    left out, costing only itself."""
    import yaml
    out: Dict[str, Dict[str, Any]] = {}
    for d in directories():
        for path in sorted(d.glob(f"*{SUFFIX}")) if d.is_dir() else ():
            name = path.name[:-len(SUFFIX)]
            if name in out:
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(data, dict):
                out[name] = _flat(data)
    return out


def preset_values(name: str) -> Dict[str, Any]:
    """A preset's values, only those that are settings of this plugin (a
    preset written for a later version may name more)."""
    found = presets().get(name)
    if found is None:
        raise ValueError(f"no segmentation preset {name!r}: there are "
                         + (", ".join(sorted(presets())) or "none"))
    known = {k for k in settings_values(SegmentSettings())}
    return {k: v for k, v in found.items() if k in known and k != "preset"}


def preset_choices() -> List[tuple]:
    return [("", "-")] + [(name, name) for name in sorted(presets())]


# ------------------------------------------------------------------ settings

@dataclass
class CircleSettings:
    shape: str = param("disc", label="shape", choices=(("ring", "ring"), ("disc", "disc")),
                       help="a ring of localizations, or a filled disc")
    min_radius_nm: float = param(30.0, label="min radius", unit="nm", min=1.0,
                                 help="the smallest radius looked for; equal to the "
                                      "max for one radius")
    max_radius_nm: float = param(150.0, label="max radius", unit="nm", min=1.0,
                                 help="the largest radius looked for")
    ring_width_nm: float = param(15.0, label="ring width", unit="nm", min=1.0,
                                 help="half the width of the ring's band, and the "
                                      "width of the filter's ring")
    precision_nm: float = param(20.0, label="judge on precision", unit="nm", min=0.1,
                                advanced=True,
                                help="the fits and the far test use only "
                                     "localizations more precise than this")
    bin_nm: float = param(10.0, label="bin", unit="nm", min=0.1, advanced=True,
                          help="pixel size of the density image that is filtered")


@dataclass
class BeadSegmentSettings:
    min_photons: float = param(5000.0, label="min photons", min=0,
                               help="localizations dimmer than this are not beads")
    min_fraction: float = param(0.25, label="min length", min=0.0, max=1.0,
                                help="a bead is seen in at least this fraction "
                                     "of the frames")
    max_jump_nm: float = param(150.0, label="maximum jump", unit="nm", min=1.0,
                               advanced=True,
                               help="the farthest a bead moves between the frames "
                                    "it is seen in")
    memory: int = param(50, label="memory", unit="frames", min=0, advanced=True,
                        help="frames a bead may be missing from its track")


@dataclass
class PositionsSettings:
    path: str = param("", label="file", kind="open_file",
                      file_filter="Text (*.csv *.tsv *.txt);;All files (*)",
                      help="a table with x_nm and y_nm, and optionally the file "
                           "each position belongs to")


@dataclass
class SegmentSettings:
    preset: str = param("", label="preset", choices=preset_choices,
                        help="a structure's values, which fill the settings; "
                             "-: the settings as they stand")
    method: str = param("circle", label="method", choices=METHODS,
                        help="how the regions are found")
    circle: CircleSettings = param(default_factory=CircleSettings,
                                   label="circle template")
    beads: BeadSegmentSettings = param(default_factory=BeadSegmentSettings,
                                       label="beads")
    ids: ColumnROIsSettings = param(default_factory=ColumnROIsSettings,
                                    label="from an id column")
    file: PositionsSettings = param(default_factory=PositionsSettings,
                                    label="positions from a file")
    keep: str = param("n_locs >= 10", label="keep", kind="text",
                      help='which candidates become ROIs, e.g. n_ring >= 4 and '
                           '40 <= radius_nm <= 70; empty: all')
    keep_rejected: bool = param(False, label="keep rejected",
                                help="add the candidates keep rejects as ROIs "
                                     "that are not used, to look at them")
    replace: bool = replace_param()


def draw_as(column: str) -> str:
    """How an ROI of this id column alone is drawn on *auto*: a track for a
    column that names tracks, an outline for any other (a cluster)."""
    from ..tracking.columns import TRACK_COLUMNS
    return "track" if column in TRACK_COLUMNS or "track" in column else "outline"


def made_by(roi, method: str) -> bool:
    """Whether Segment regions made this ROI with this method."""
    origin = roi.origin or {}
    return origin.get("method") == PATH and origin.get("segment") == method


def provided(settings: SegmentSettings) -> tuple:
    """The values the chosen method measures on each candidate."""
    key = settings.circle.shape if settings.method == "circle" else settings.method
    return segmentation.VALUES.get(key, ())


unavailable = segmentation.unavailable


def candidates(locs, settings: SegmentSettings, existing: Sequence = ()
               ) -> List[Dict[str, Any]]:
    """Every candidate of the circle template or beads on a table, measured
    and judged by *keep* (each with ``use`` and ``failed``), strongest
    first.  *keep* is checked before the search, which can take a while."""
    if settings.method in MEASURED:
        segmentation.keep_test(settings.keep, provided(settings), unavailable(locs))
    if settings.method == "circle":
        c = settings.circle
        if c.shape not in segmentation.SHAPES:
            raise ValueError(f"no shape {c.shape!r}: one of "
                             + ", ".join(segmentation.SHAPES))
        sites = segmentation.find_circles(locs, c.shape, c.min_radius_nm,
                                          c.max_radius_nm, c.ring_width_nm,
                                          c.bin_nm, c.precision_nm, existing)
    elif settings.method == "beads":
        b = settings.beads
        n_frames = (locs.metadata or {}).get("n_frames") if hasattr(locs, "metadata") \
            else None
        sites = segmentation.measure_beads(locs, b.min_photons, b.min_fraction,
                                           b.max_jump_nm, b.memory,
                                           n_frames=int(n_frames) if n_frames else None)
        if existing:
            # a bead within its spread of an ROI already there is that ROI's
            ex = np.asarray(existing, dtype=float).reshape(-1, 2)
            sites = [s for s in sites if not np.any(
                np.hypot(*(ex - np.asarray(s["center"])).T)
                < max(3 * s["values"]["spread_nm"], 1.0))]
    else:
        raise ValueError(f"{settings.method} has no candidates to measure")
    segmentation.judge(sites, settings.keep, provided(settings), unavailable(locs))
    return sites


# ------------------------------------------------------------------ the plugin

# a literal: discovery reads the path from the file without importing it
@register("Analysis/Regions/Segment regions")
class SegmentRegions(SegmenterPlugin):
    """Finds regions and makes ROIs of them: rings or discs by a circle
    template, beads, an id column such as cluster_id, or positions from a
    file."""

    Settings = SegmentSettings
    changes_locs = False
    # from an id column reads the localizations, whose ids they are (the
    # circle template and beads read the ROI manager's own table)
    grouping = "ungrouped"
    # 2: the circle template (ring or disc, over a range of radii) and keep
    # replace the ring template and density peaks; beads; positions
    version = "2"

    def describe(self, ctx: Optional[Context], settings: SegmentSettings) -> FormState:
        """Only the chosen method's part; the presets there are; the values
        *keep* can name, or what is wrong with it."""
        state = FormState(shown={key: settings.method == key for key, _ in METHODS})
        state.choices["preset"] = preset_choices()
        by_id = settings.ids.membership == "id"
        state.active["ids.outline"] = not by_id
        state.active["ids.draw"] = by_id
        if by_id and settings.ids.draw == "auto":
            state.hints["ids.draw"] = draw_as(settings.ids.column.strip())
        measured = settings.method in MEASURED
        state.shown["keep"] = measured
        state.shown["keep_rejected"] = measured
        state.active["circle.ring_width_nm"] = settings.circle.shape == "ring"
        if measured:
            names = provided(settings)
            locs = None
            project = getattr(ctx, "rois", None) if ctx is not None else None
            if project is not None:
                file_id = current_file(project)
                if file_id is not None:
                    try:
                        locs = project.state(file_id).locs
                    except Exception:
                        locs = None
            try:
                segmentation.keep_test(settings.keep, names, unavailable(locs))
            except ValueError as error:
                state.notice = Notice(str(error))
            else:
                state.notice = Notice("keep can name " + ", ".join(names), kind="info")
        return state

    def react(self, changed: str, settings: SegmentSettings) -> Optional[Dict[str, Any]]:
        """Choosing a preset fills the settings with its values."""
        if changed == "preset" and settings.preset:
            return preset_values(settings.preset)
        return None

    def replaced(self, project, file_id, settings: SegmentSettings) -> List[str]:
        """With *replace*: the ROIs the same method made on the file before."""
        if not settings.replace:
            return []
        return [roi.id for roi in project.rois.values()
                if roi.file_id == file_id and made_by(roi, settings.method)]

    def propose(self, locs, settings: SegmentSettings, existing: Sequence = ()
                ) -> List[Dict[str, Any]]:
        """The sites for `ROIProject.find`, each naming its method in the
        origin the ROI keeps, with the values measured on it."""
        if settings.method not in MEASURED:
            raise ValueError(f"{dict(METHODS).get(settings.method, settings.method)} "
                             "makes its ROIs of the whole table, not one file's: "
                             "run the plugin")
        sites = candidates(locs, settings, existing)
        # every candidate, for the run's text and figures: `find` keeps only
        # the ROIs it made, and the rejected ones are what *keep* is tuned on
        self.last = sites
        kept = [s for s in sites if s["use"] or settings.keep_rejected]
        return [{"center": s["center"], "use": s["use"],
                 "origin": {"segment": settings.method, "values": s["values"],
                            "failed": s["failed"]}}
                for s in kept]

    def segment(self, ctx: Context, settings: SegmentSettings, commit: bool) -> Result:
        if settings.method not in dict(METHODS):
            raise ValueError(f"no method {settings.method!r}: one of "
                             + ", ".join(k for k, _ in METHODS))
        if settings.method in MEASURED:
            return self._measured(ctx, settings, commit)
        if settings.method == "file":
            return self._file(ctx, settings, commit)
        return self._ids(ctx, settings, commit)

    # -- circle template and beads: candidates measured, judged by keep

    def _measured(self, ctx: Context, settings: SegmentSettings, commit: bool) -> Result:
        project = ctx.rois
        effect: Dict[str, Any] = {}
        if project is None:
            locs = ctx.locs
            sites = candidates(locs, settings)
            data = {"sites": sites,
                    "centers": [s["center"] for s in sites if s["use"]]}
        else:
            file_id = current_file(project)
            if file_id is None:
                raise ValueError("no file to search: load localizations first")
            if settings.method == "beads" and project.grouped:
                raise ValueError("beads are tracks of single localizations: show "
                                 "the localizations, not merged blinks, in the "
                                 "layer the ROI manager follows")
            state = project.state(file_id)
            locs = state.locs[state.filter.indices]
            self.last = []
            proposed, effect = self.found(project, file_id, settings, commit)
            sites = self.last
            data = {"sites": sites,
                    "centers": [p["center"] for p in proposed if p["use"]]}
        kept = sum(s["use"] for s in sites)
        what = "beads" if settings.method == "beads" else "structures"
        text = f"{kept} {what} of {len(sites)} candidates"
        if sites and kept < len(sites):
            reasons: Dict[str, int] = {}
            for s in sites:
                for f in s["failed"]:
                    reasons[f] = reasons.get(f, 0) + 1
            text += "; failing " + ", ".join(f"{k} ({v})" for k, v in
                                             sorted(reasons.items()))
        if not commit:
            text = "preview: " + text
        xy = np.column_stack((np.asarray(locs["x_nm"], dtype=float),
                              np.asarray(locs["y_nm"], dtype=float)))
        names = segmentation.keep_test(settings.keep, provided(settings),
                                       unavailable(locs))[1] or ["radius_nm", "n_locs"]
        limits = segmentation.bounds(settings.keep)
        plots = {}
        if sites:
            plots["keep"] = Plot(lambda f: draw_values(f, sites, names, limits),
                                 panels=len(names), size=(2.7 * min(len(names), 4),
                                                          2.6 * -(-len(names) // 4)))
        return Result(text=text, settings=settings, data=data, **effect,
                      plot=(lambda ax: draw_sites(ax, xy, sites)) if sites else None,
                      plots=plots)

    # -- positions from a file

    def _file(self, ctx: Context, settings: SegmentSettings, commit: bool) -> Result:
        if ctx.per_region:
            raise ValueError("ROIs cannot be made inside a block")
        project = ctx.rois
        if project is None:
            raise ValueError("there is no ROI manager project to add ROIs to")
        if not project.sources:
            raise ValueError("the ROI manager has no file to add ROIs to")
        if not settings.file.path:
            raise ValueError("choose the file of positions")
        read = segmentation.read_positions(settings.file.path)
        current = current_file(project)
        file_ids, unknown = place_positions(project, read, current)
        placed = [(fid, (float(x), float(y))) for fid, x, y in
                  zip(file_ids, read["x_nm"], read["y_nm"]) if fid is not None]

        def ours(r) -> bool:
            return made_by(r, "file")

        targets = {fid for fid, _ in placed}
        note = f"; {unknown} name a file the ROI manager does not have" if unknown else ""
        if not commit:
            replaced = ([r.id for r in project.rois.values()
                         if r.file_id == current and ours(r)] if settings.replace else [])
            sites = [{"center": list(c), "use": True} for fid, c in placed if fid == current]
            return Result(text=f"preview: {len(sites)} ROIs on this file of "
                               f"{len(read['x_nm'])} positions" + note,
                          settings=settings, data={"centers": [s["center"] for s in sites]},
                          roi_preview={"file_id": current, "method": self.path,
                                       "sites": sites, "replaces": replaced})
        made = []
        with project.edit(f"{self.name}: ROIs") as step:
            replaced = []
            if settings.replace:
                replaced = [r.id for r in project.rois.values()
                            if ours(r) and r.file_id in targets]
                for roi_id in replaced:
                    project.rois.pop(roi_id, None)
            origin = {"method": self.path, "segment": "file",
                      "parameters": settings_values(settings),
                      "source": str(settings.file.path)}
            for fid, centre in placed:
                made.append(project.add_roi(fid, centre, origin=origin))
            step.label = f"{self.name}: {len(made)} ROIs" + (
                f" for {len(replaced)}" if replaced else "")
        text = f"{len(made)} ROIs from {Path(settings.file.path).name}"
        if replaced:
            text += f", replacing {len(replaced)}"
        return Result(text=text + note, settings=settings, rois=made,
                      values={"rois": len(made)})

    # -- from an id column (was Analysis/Regions/Column to ROIs)

    def _ids(self, ctx: Context, settings: SegmentSettings, commit: bool) -> Result:
        if ctx.per_region:
            raise ValueError("ROIs cannot be made inside a block: make them of "
                             "the column after it")
        project = ctx.rois
        if project is None:
            raise ValueError("there is no ROI manager project to add ROIs to")
        sources = {getattr(s, "number", 0): s.id for s in project.sources.values()}
        if not sources:
            raise ValueError("the ROI manager has no file to add ROIs to")
        ids = settings.ids
        column = ids.column.strip()
        by_id = ids.membership == "id"
        if ids.membership not in ("shape", "id"):
            raise ValueError(f"an ROI holds 'shape' or 'id', not {ids.membership!r}")
        draw = (ids.draw if ids.draw in ("track", "outline") else draw_as(column))
        # an ROI of its id alone is drawn from its own localizations, so it
        # needs no hull of its own
        outline = ids.outline and not by_id
        found = column_regions(ctx.locs, column, ids.min_size, outline)

        def ours(r) -> bool:
            return made_by(r, "ids") and (r.origin or {}).get("column") == column

        if not commit:
            # the preview, as the other methods': the centres on the file shown
            file_id = current_file(project)
            first = next(iter(sources.values()))
            replaced = ([r.id for r in project.rois.values()
                         if r.file_id == file_id and ours(r)] if settings.replace else [])
            sites = [{"center": r["center"], "use": True} for r in found
                     if sources.get(r["file"], first) == file_id]
            return Result(text=f"preview: {len(sites)} ROIs of {column} on this file",
                          settings=settings, data={"centers": [s["center"] for s in sites]},
                          roi_preview={"file_id": file_id, "method": self.path,
                                       "sites": sites, "replaces": replaced})
        made, flat = [], 0
        with project.edit(f"{self.name}: ROIs") as step:
            replaced = []
            if settings.replace:
                replaced = [r.id for r in project.rois.values() if ours(r)]
                for roi_id in replaced:
                    project.rois.pop(roi_id, None)
            for region in found:
                if outline and region["polygon"] is None:
                    flat += 1                   # no hull: the centre, as without
                file_id = sources.get(region["file"], next(iter(sources.values())))
                origin = {"method": self.path, "segment": "ids",
                          "parameters": settings_values(settings), "column": column,
                          "cluster": region["id"],
                          "localizations": int(len(region["rows"]))}
                if by_id:
                    origin["draw"] = draw
                    made.append(project.add_roi(
                        file_id, region["center"], origin=origin,
                        ids={column: region["id"]}, spatial=False))
                else:
                    made.append(project.add_roi(file_id, region["center"],
                                                polygon=region["polygon"], origin=origin))
            step.label = f"{self.name}: {len(made)} ROIs" + (
                f" for {len(replaced)}" if replaced else "")
        text = f"{len(made)} ROIs of {column}, from {ids.min_size} localizations"
        if replaced:
            text += f", replacing {len(replaced)}"
        if flat:
            text += f"; {flat} with no outline (on a line) got the ROI manager's shape"
        return Result(text=text, settings=settings, rois=made,
                      values={"rois": len(made)})


def place_positions(project, read: Dict[str, Any], current) -> tuple:
    """``(file ids, unknown)``: the file each position goes to -- the one its
    ``file_id`` names (an *Export table* of this project), else its
    ``file`` (a file's name) or ``filenumber``, else the file shown -- and
    how many named a file the project does not have (None in their place)."""
    by_name = {s.name: s.id for s in project.sources.values()}
    by_number = {int(s.number): s.id for s in project.sources.values()}
    out, unknown = [], 0
    for k in range(len(read["x_nm"])):
        fid = current
        given = [(read.get("file_id"), lambda v: v if v in project.sources else None),
                 (read.get("file"), by_name.get),
                 (read.get("filenumber"), lambda v: by_number.get(int(v)))]
        named = False
        for column, look in given:
            if column is None or column[k] in ("", None):
                continue
            named = True
            found = look(column[k])
            if found is not None:
                fid = found
                break
        else:
            if named:
                fid = None
                unknown += 1
        out.append(fid)
    return out, unknown


# ------------------------------------------------------------------ figures

def draw_values(figure, sites: Sequence[Dict[str, Any]], names: Sequence[str],
                limits: Dict[str, List[float]]) -> None:
    """A histogram of each value *keep* reads, kept and rejected stacked,
    the numbers the condition compares it with drawn in: where the cut falls
    against the distribution."""
    used = np.array([s["use"] for s in sites], dtype=bool)
    across = min(len(names), 4)
    rows = -(-len(names) // across)
    axes = np.ravel(figure.subplots(rows, across, squeeze=False,
                                    gridspec_kw={"wspace": 0.45, "hspace": 0.6}))
    for ax, key in zip(axes, names):
        values = np.array([s["values"].get(key, np.nan) for s in sites], dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            edges = np.histogram_bin_edges(values[finite], bins=20)
            ax.hist([values[finite & used], values[finite & ~used]], bins=edges,
                    stacked=True, color=["#2f7fd0", "0.75"], label=["kept", "rejected"])
        for limit in limits.get(key, []):
            ax.axvline(limit, color="#d62728", lw=1)
        ax.set_xlabel(key, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    axes[0].set_ylabel("candidates", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)


def draw_sites(ax, xy: np.ndarray, sites: Sequence[Dict[str, Any]]) -> None:
    """The localizations with each candidate's circle -- its measured radius,
    else the template's: kept in blue, rejected in grey."""
    from matplotlib.patches import Circle
    ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="k", linewidths=0, alpha=0.5)
    for site in sites:
        radius = site["values"].get("radius_nm", np.nan)
        if not np.isfinite(radius) or radius <= 0:
            radius = site.get("template_nm", 50.0)
        ax.add_patch(Circle(site["center"], radius, fill=False, lw=0.8,
                            color="#2f7fd0" if site["use"] else "0.6"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
