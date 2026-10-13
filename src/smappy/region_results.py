"""What was measured per region, read one way whichever store holds it.

Two stores keep `region_table.RegionTable`s, and both stay: `ROIProject.tables`
(per ROI) and `Session.id_regions` (per region of an id column, a cluster).
What differs is ownership, not what is kept: an ROI is edited alone, so its
row is checked alone (`ROIProject.evaluations`); a column's regions change
together, so its table is current or not as a whole -- the digest of the
column's inputs (`region_runs.column_inputs`) against the one each row was
measured on.  Every reader used to branch on the store and parse
"per cluster_id: label" itself, and the per-cluster branch never checked: three
clusters relabelled as one, and Histograms still counted three.

An evaluation is an `Evaluation(source, label)`; its string form is what
*results from* holds and a chain file saves -- "Corners" on the ROIs,
"per cluster_id: Corners" per cluster -- so both still read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .chain import BLOCK_ROIS as ROIS

PER = "per "


@dataclass(frozen=True)
class Evaluation:
    source: str      # ROIS, or the id column that names the regions
    label: str

    @property
    def on_rois(self) -> bool:
        return self.source == ROIS

    def __str__(self) -> str:
        return self.label if self.on_rois else f"{PER}{self.source}: {self.label}"

    @classmethod
    def parse(cls, text: str) -> "Evaluation":
        text = str(text or "")
        if text.startswith(PER) and ": " in text:
            column, _, label = text[len(PER):].partition(": ")
            return cls(column, label)
        return cls(ROIS, text)


def available(ctx, plugin: Optional[str] = None) -> List[Evaluation]:
    """The current evaluations, those by ``plugin`` (a path) when given: the
    ROIs' by name, then each id column's.  A stale per-cluster table is not
    offered, as an ROI evaluation that no longer describes any ROI is not."""
    project = ctx.rois
    out = [Evaluation(ROIS, label) for label in
           sorted(project.labels(plugin) if project is not None else [])]
    if ctx.session is not None:
        now: Dict[str, set] = {}          # a column's digests, once, not per table
        for column, tables in sorted(ctx.session.id_regions.items()):
            for label, found in tables.items():
                made = next((found.step(r) for r in found.ids), {})
                if (plugin is None or made.get("plugin") == plugin) \
                        and is_current(ctx, Evaluation(column, label), now):
                    out.append(Evaluation(column, label))
    return out


def table(ctx, ev: Evaluation):
    """The `RegionTable` of ``ev``, current or not; None when there is none."""
    if ev.on_rois:
        return None if ctx.rois is None else ctx.rois.tables.get(ev.label)
    if ctx.session is None:
        return None
    return (ctx.session.id_regions.get(ev.source) or {}).get(ev.label)


def is_current(ctx, ev: Evaluation, _now: Optional[Dict[str, set]] = None) -> bool:
    """Whether ``ev`` still describes its regions: for the ROIs, some included
    ROI's row does; for an id column, the table does, as a whole.  A table of
    regions inside regions (``step/inner``) is as current as its outer step."""
    if ev.on_rois:
        return ctx.rois is not None and ev.label in ctx.rois.labels()
    found = table(ctx, ev)
    if found is None or not len(found):
        return False
    outer, _, inner = ev.label.partition("/")
    if inner and outer in ctx.session.id_regions[ev.source]:
        return is_current(ctx, Evaluation(ev.source, outer), _now)
    if ev.source not in ctx.session.locs:
        return False
    now = {} if _now is None else _now
    if ev.source not in now:
        now[ev.source] = _digests(ctx.session, ev.source)
    return all(d in now[ev.source] for d in found.data)


def _digests(session, column: str) -> set:
    """The data signature a run of ``column``'s regions would give now, for
    whichever layer it followed (the layer is part of the inputs)."""
    from .region_runs import column_inputs
    from .roi_manager.core import digest, json_text
    inputs = json.loads(json_text(column_inputs(session.locs, column, 0, session)))
    return {digest({**inputs, "layer": i}) for i in range(len(session.layers))}


def rows(ctx, ev: Evaluation) -> List[Dict[str, Any]]:
    """The site table of ``ev``, current rows only: a region's id as
    ``roi_id`` and its values beside it.  For the ROIs, the included ones
    whose row still describes them (`ROIProject.results`); for an id column,
    every region where the step did not fail, or none when the table is out
    of date."""
    if ev.on_rois:
        return [] if ctx.rois is None else ctx.rois.results(labels=[ev.label])
    if not is_current(ctx, ev):
        return []
    found = table(ctx, ev)
    out = []
    for region_id in found.ids:
        entry = found.entry(region_id)
        if entry and "values" in entry:
            out.append({"roi_id": region_id, "file_id": "", **entry["values"]})
    return out


def steps(ctx, ev: Evaluation, ids=None) -> List[Dict[str, Any]]:
    """The distinct step records (plugin, version, parameters) that made the
    current rows of ``ev`` -- of the regions ``ids``, else all of them: one,
    unless some regions were measured again with other settings."""
    if ev.on_rois:
        return [] if ctx.rois is None else ctx.rois.evaluation_steps(ev.label, ids)
    if not is_current(ctx, ev):
        return []
    found = table(ctx, ev)
    out = []
    for region_id in (found.ids if ids is None else ids):
        made = found.step(region_id)
        if made and made not in out:
            out.append(made)
    return out


def missing(ctx, ev: Evaluation) -> str:
    """Why ``ev`` cannot be read: out of date, or not there at all."""
    if table(ctx, ev) is not None:
        what = ("the ROIs, their localizations or a layer's filter changed"
                if ev.on_rois else
                f"the table, {ev.source} or a layer's filter changed")
        return (f"the results {str(ev)!r} are out of date: {what} since they "
                "were measured; run them again")
    if ev.on_rois:
        there = available(ctx)
        return (f"no results on the ROIs are called {ev.label!r}"
                + (f"; there is {', '.join(map(str, there))}" if there else ""))
    return f"there are no results {str(ev)!r} (any more)"


def without_results(metadata: Dict[str, Any], results: bool = True,
                    figures: bool = True) -> Dict[str, Any]:
    """The file's metadata with both stores' results left out (``results``
    False), or only what their regions kept to draw figures (``figures``
    False): what `Session.save` writes when asked to leave them out."""
    metadata = dict(metadata)
    if not results:
        if isinstance(metadata.get("roi_project"), dict):
            metadata["roi_project"] = {k: v for k, v in metadata["roi_project"].items()
                                       if k != "tables"}
        metadata.pop("id_regions", None)
    elif not figures:
        def unkept(doc):
            return {k: v for k, v in doc.items() if k != "kept"}
        if isinstance(metadata.get("roi_project"), dict):
            project = dict(metadata["roi_project"])
            project["tables"] = [unkept(t) for t in project.get("tables") or []]
            metadata["roi_project"] = project
        if metadata.get("id_regions"):
            metadata["id_regions"] = {c: [unkept(t) for t in ts]
                                      for c, ts in metadata["id_regions"].items()}
    return metadata
