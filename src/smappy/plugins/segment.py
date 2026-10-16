"""Segment regions: ROIs for the ROI manager, by one of several methods.

`docs/smap_porting/wave2_design.md`, step 11, has the decision.  Three
plugins that each made ROIs -- *Find/NPC*, *Find/Density Peaks* and *Column
to ROIs* -- became one, with a *method* choice and a settings part per
method, only the chosen one shown:

* **ring template** -- a ring filter, a circle fitted to each candidate and
  checks on it (`npc.segment_npcs`), for nuclear pores and other rings;
* **density peaks** -- maxima of a blurred density image
  (`roi.density_peaks`), for any compact structure;
* **from an id column** -- an ROI per id of `cluster_id` or any whole-number
  column, the cluster's hull or the project's shape
  (`region_column.column_regions`).

The algorithms stayed where they were, as module-level functions with their
settings classes; what moved is each plugin's `run`, unchanged in what it
does.  One plugin rather than three, because they answer one question --
where are the regions? -- and a structure is then a choice of values rather
than of plugin: a **preset** is a small YAML file of those values (``NPC``:
the ring template at 55 nm), shipped in ``smappy/data/segment_presets`` and
the user's under the config directory, which win over a shipped one of the
same name, as `io/conversions.py` keeps csv conversions.  Choosing one fills
the settings (`SegmentRegions.react`); the settings, not the preset's name,
are what a run uses and what the ROIs record.

Each ROI's origin names the method that made it (``origin["segment"]``)
beside the plugin, so *replace* removes the ROIs the same method made before
and leaves another method's: density peaks run after the ring template does
not delete the pores.

The ring template and density peaks search the file the ROI manager shows,
as the manager filters it; from an id column reads the localizations of
every file, each ROI going to its own file -- as the plugins did.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import (Context, FormState, Plot, Result, param, register, settings_values)
from .npc import NPCSegmentSettings, draw_quality, draw_sites, segment_npcs
from .region_column import ColumnROIsSettings, column_regions
from .roi import (DensityPeakSettings, SegmenterPlugin, current_file, density_peaks,
                  replace_param)

PATH = "Analysis/Regions/Segment regions"

METHODS = (("ring", "ring template"), ("peaks", "density peaks"),
           ("ids", "from an id column"))

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


# ------------------------------------------------------------------ the plugin

@dataclass
class SegmentSettings:
    preset: str = param("", label="preset", choices=preset_choices,
                        help="a structure's values, which fill the settings; "
                             "-: the settings as they stand")
    method: str = param("peaks", label="method", choices=METHODS,
                        help="how the regions are found")
    ring: NPCSegmentSettings = param(default_factory=NPCSegmentSettings,
                                     label="ring template")
    peaks: DensityPeakSettings = param(default_factory=DensityPeakSettings,
                                       label="density peaks")
    ids: ColumnROIsSettings = param(default_factory=ColumnROIsSettings,
                                    label="from an id column")
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


# a literal: discovery reads the path from the file without importing it
@register("Analysis/Regions/Segment regions")
class SegmentRegions(SegmenterPlugin):
    """Finds regions and makes ROIs of them: by a ring template, density
    peaks, or an id column such as cluster_id."""

    Settings = SegmentSettings
    changes_locs = False
    # from an id column reads the localizations, whose ids they are (the
    # ring template and density peaks read the ROI manager's own table)
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: SegmentSettings) -> FormState:
        """Only the chosen method's part; the presets there are."""
        state = FormState(shown={key: settings.method == key for key, _ in METHODS})
        state.choices["preset"] = preset_choices()
        by_id = settings.ids.membership == "id"
        state.active["ids.outline"] = not by_id
        state.active["ids.draw"] = by_id
        if by_id and settings.ids.draw == "auto":
            state.hints["ids.draw"] = draw_as(settings.ids.column.strip())
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
        origin the ROI keeps."""
        if settings.method == "ring":
            sites = segment_npcs(locs, settings.ring, existing)
            # every candidate, for the run's text and figures: `find` keeps
            # only the ROIs it made, and the rejected ones are what the
            # checks are for
            self.last = sites
            keep = [s for s in sites if s["use"] or settings.ring.keep_rejected]
            return [{"center": s["center"], "use": s["use"],
                     "origin": {"segment": "ring",
                                **{k: s[k] for k in ("radius_nm", "n_ring", "n_precise",
                                                     "far_fraction", "failed")}}}
                    for s in keep]
        if settings.method == "peaks":
            return [{"center": c, "use": True, "origin": {"segment": "peaks"}}
                    for c in density_peaks(locs, settings.peaks, existing)]
        raise ValueError("from an id column makes its ROIs of the whole table, "
                         "not one file's: run the plugin")

    def segment(self, ctx: Context, settings: SegmentSettings, commit: bool) -> Result:
        if settings.method not in dict(METHODS):
            raise ValueError(f"no method {settings.method!r}: one of "
                             + ", ".join(k for k, _ in METHODS))
        if settings.method == "ring":
            return self._ring(ctx, settings, commit)
        if settings.method == "peaks":
            return self._peaks(ctx, settings, commit)
        return self._ids(ctx, settings, commit)

    # -- ring template (was Analysis/Regions/Find/NPC)

    def _ring(self, ctx: Context, settings: SegmentSettings, commit: bool) -> Result:
        from .npc import precision_column
        ring = settings.ring
        project = ctx.rois
        effect: Dict[str, Any] = {}
        if project is None:
            locs = ctx.locs
            sites = segment_npcs(locs, ring)
            data = {"sites": sites,
                    "centers": [s["center"] for s in sites if s["use"]]}
        else:
            file_id = current_file(project)
            if file_id is None:
                raise ValueError("no file to search: load localizations first")
            state = project.state(file_id)
            locs = state.locs[state.filter.indices]
            self.last = []
            proposed, effect = self.found(project, file_id, settings, commit)
            sites = self.last
            data = {"sites": sites,
                    "centers": [p["center"] for p in proposed if p["use"]]}
        kept = sum(s["use"] for s in sites)
        text = f"{kept} pores of {len(sites)} candidates"
        if precision_column(locs) is None:
            text += " (no precision column: the far test is off)"
        if sites and kept < len(sites):
            reasons: Dict[str, int] = {}
            for s in sites:
                for f in s["failed"]:
                    reasons[f] = reasons.get(f, 0) + 1
            text += "; rejected for " + ", ".join(f"{k} ({v})" for k, v in
                                                  sorted(reasons.items()))
        if not commit:
            text = "preview: " + text
        xy = np.column_stack((np.asarray(locs["x_nm"], dtype=float),
                              np.asarray(locs["y_nm"], dtype=float)))
        return Result(text=text, settings=settings, data=data, **effect,
                      plot=(lambda ax: draw_sites(ax, xy, sites, ring)) if sites else None,
                      plots={"checks": Plot(lambda f: draw_quality(f, sites, ring),
                                            panels=3, size=(8, 2.8))} if sites else {})

    # -- density peaks (was Analysis/Regions/Find/Density Peaks)

    def _peaks(self, ctx: Context, settings: SegmentSettings, commit: bool) -> Result:
        project = ctx.rois
        if project is None:                 # no session: just report the centres
            centers = density_peaks(ctx.locs, settings.peaks)
            return Result(text=f"{len(centers)} candidates",
                          data={"centers": centers}, settings=settings)
        file_id = current_file(project)
        if file_id is None:
            raise ValueError("no file to search: load localizations first")
        sites, effect = self.found(project, file_id, settings, commit)
        text = f"{len(sites)} candidates " + ("found" if commit else "previewed")
        return Result(text=text, settings=settings,
                      data={"centers": [s["center"] for s in sites]}, **effect)

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
