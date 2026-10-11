"""A block of a chain: its steps run once in each region.

`docs/regions.md` ("Evaluation is a block of a chain") has the decisions; a
block is the chain's own `Block` (its file form), and `block_class` makes it a
plugin as `chain_class` makes a chain one, so the GUI draws a block as a part
holding its steps and a chain runs it as a step like any other.

What a run of a block does:

* **The regions** are the included ROIs of the ROI manager, or the regions an
  id column names (`region_runs.ColumnRegions`).  Inside another block's
  region they are the regions of that region's table: blocks nest, and a
  region narrows.
* **In each region the steps run in one `region_runs.RegionScratch`**: a table
  a step hands back is the next step's table, and all of it is thrown away
  afterwards.  What stays is each step's numbers -- with the ROIs
  (`ROIProject.tables`) or with the column (`Session.id_regions`), under the
  step's name, as any evaluation's -- and the columns the block writes back.
* **Write-back** is opt-in, by name: the columns are merged into the real
  table by the rows they came from (`region_runs.WriteBack`), and a column a
  step added or changed that is not written back is warned about once, after
  the first region that has one.  A block that writes back changes the table,
  one undo step and one log line; one that does not changes nothing.
* **A step added to a block starts with its slow options off**
  (`param(slow=True)`, `plugins.fast_values`) unless its values say otherwise.
* **A segmenter is refused** inside a block: ROIs found in each region have
  nowhere to go yet.
* **A block inside a block** makes a table per region, not a row: its tables
  are kept beside the outer step's, a row per inner region named
  ``outer:inner``.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import make_dataclass
from typing import Any, Dict, List, Optional, Tuple

from .chain import (BLOCK_ROIS, DEFAULT_GROUP, KEEP_FIGURES, Block, ChainError,
                    ChainPlugin, Step, _flat_infos, plugin_settings, step_plugin,
                    step_settings_class)
from .plugins import Context, ParamInfo, Result, fast_values, param, settings_values
from .region_runs import WriteBack, changed_columns

# the block's own fields, before its steps', in its settings
OWN = ("regions", "write_back", "merge", "keep_figures")


class BlockPlugin(ChainPlugin):
    """A block, as a plugin.  Made by `block_class`; not registered."""
    block: Block = None
    per_region = False
    logged = True

    def changes_locs_for(self, settings) -> Optional[bool]:
        return bool(_columns(settings.write_back))

    def active(self, settings):
        """Its steps' own, and no grouping to choose: in a region every layer
        is cut as it is shown."""
        from .chain import GROUPING_FIELD
        out = dict(super().active(settings) or {})
        for key, step, cls in self.steps:
            if hasattr(getattr(settings, key), GROUPING_FIELD):
                out[f"{key}.{GROUPING_FIELD}"] = False
        return out or None

    @classmethod
    def evaluations(cls, settings) -> List[Tuple[str, str]]:
        """``(name, plugin path)`` of what it leaves on the ROIs: its steps',
        when it runs over the ROIs."""
        if settings.regions != BLOCK_ROIS:
            return []
        return [(step.title(), step_cls.path) for key, step, step_cls in cls.steps
                if not isinstance(step, Block)]

    def block_from(self, settings) -> Block:
        """The block as its file would say it, with ``settings``."""
        block = copy.deepcopy(self.block)
        block.regions = settings.regions
        block.write_back = _columns(settings.write_back)
        block.merge = settings.merge
        block.keep_figures = settings.keep_figures
        inner = {key: (step, cls) for key, step, cls in self.steps}
        steps = []
        for key, step in zip(self.block.keys(), self.block.steps):
            step = copy.deepcopy(step)
            if key in inner:
                sub = getattr(settings, key)
                cls = inner[key][1]
                if isinstance(step, Block):
                    step = cls().block_from(sub)
                else:
                    step.values = settings_values(plugin_settings(cls, sub))
                    step.version = cls.version
            steps.append(step)
        block.steps = steps
        return block

    def pipeline(self, settings) -> list:
        """The steps as `roi_manager.pipeline.Step`s, ready to run."""
        from .roi_manager.pipeline import Step as Run
        keep = {"yes": True, "no": False}.get(settings.keep_figures,
                                              settings.regions == BLOCK_ROIS)
        out = []
        for key, step, cls in self.steps:
            sub = getattr(settings, key)
            own = sub if isinstance(step, Block) else plugin_settings(cls, sub)
            out.append(Run(label=step.title(), path=cls.path, plugin=cls(),
                           settings=own, version=str(cls.version), keep=keep))
        return out

    def run(self, ctx: Context, settings) -> Result:
        steps = self.pipeline(settings)
        if not steps:
            raise ValueError("the block has no steps")
        columns = _columns(settings.write_back)
        back = WriteBack(columns, settings.merge)
        warned: List[str] = []
        children: Dict[str, Any] = {}

        def after(region_id, scratch, original):
            if back:
                back.add(scratch)
            if not warned:
                lost = [c for c in changed_columns(original, scratch)
                        if c not in columns]
                if lost:
                    warned.extend(lost)
                    ctx.warn(f"{self.block.title()}: the steps changed "
                             f"{', '.join(lost)}, which the block does not write "
                             "back -- they are thrown away with each region")
            for label, tables in scratch.children.items():
                for name, table in tables.items():
                    children.setdefault(f"{label}/{name}", []).append((region_id, table))

        report = ctx.report
        if ctx.per_region:
            tables, n, where = self._run_within(ctx, settings, steps, after)
        elif settings.regions == BLOCK_ROIS:
            tables, n, where = self._run_rois(ctx, steps, after, bool(back), report)
        else:
            tables, n, where = self._run_column(ctx, settings, steps, after,
                                                bool(back), report)
        merged = _children(children)
        if merged and not ctx.per_region:
            # the inner blocks' tables, beside the steps' own
            if settings.regions == BLOCK_ROIS and ctx.rois is not None:
                ctx.rois.tables.update(merged)
            elif ctx.session is not None:
                ctx.session.id_regions.setdefault(settings.regions, {}).update(merged)
        failed = sum(1 for t in tables.values() for e in t.errors if e)
        text = f"{n} {where}, {len(steps)} step(s)"
        if failed:
            text += f", {failed} failed"
        locs = None
        if back and n:
            if back.missing:
                ctx.warn(f"{self.block.title()}: no step wrote "
                         f"{', '.join(sorted(back.missing))} in "
                         f"{max(back.missing.values())} region(s)")
            locs = back.merge(ctx.locs)
            text += f"; wrote back {', '.join(columns)}"
        return Result(locs=locs, text=text, settings=settings,
                      values={"regions": n},
                      data={"tables": tables, "region_tables": {**tables, **merged},
                            "evaluations": self.evaluations(settings)
                            if not ctx.per_region else [],
                            "regions": n, "failed": failed})

    # ------------------------------------------------------------ the three
    def _run_rois(self, ctx, steps, after, writes, report):
        project = ctx.rois
        if project is None:
            raise ValueError("there is no ROI manager project to repeat over")
        ids = [i for i, roi in project.rois.items() if roi.reviewed and roi.use]
        if not ids:
            raise ValueError("there are no ROIs to evaluate: find or draw some first")
        run = project.evaluate(steps=steps, roi_ids=ids, reuse=not writes,
                               chain=f"block {self.block.title()}", report=report,
                               stop=ctx.stop, after=after, tag=writes)
        tables = {s.label: project.tables[s.label] for s in steps
                  if s.label in project.tables}
        return tables, len(run["records"]), "ROIs"

    def _run_column(self, ctx, settings, steps, after, writes, report):
        session = ctx.session
        if session is None:
            from .region_runs import ColumnRegions, evaluate_regions
            source = ColumnRegions(locs=ctx.locs, column=settings.regions)
            tables = evaluate_regions(source, steps, report=report, stop=ctx.stop,
                                      after=after, tag=writes,
                                      keep_figures=steps[0].keep, reuse=not writes)
        else:
            tables = session.evaluate_regions(
                settings.regions, steps, layer=ctx.layer, report=report,
                stop=ctx.stop, reuse=not writes, keep_figures=steps[0].keep,
                after=after, tag=writes)
        n = max((len(t) for t in tables.values()), default=0)
        return {s.label: tables[s.label] for s in steps}, n, f"{settings.regions} regions"

    def _run_within(self, ctx, settings, steps, after):
        """Inside another block's region: the regions of this region's table."""
        from .region_runs import ContextRegions, evaluate_regions
        if settings.regions == BLOCK_ROIS:
            raise ValueError("a block over the ROIs cannot run inside another "
                             "block's region: repeat over an id column there")
        source = ContextRegions(ctx, settings.regions)
        tables = evaluate_regions(source, steps, stop=ctx.stop, after=after,
                                  tag=False, keep_figures=False, reuse=False)
        n = max((len(t) for t in tables.values()), default=0)
        return tables, n, f"{settings.regions} regions"


def block_class(block: Block) -> type:
    """A `BlockPlugin` subclass that runs ``block``.

    Its settings are the block's own fields, then one part per step, as a
    chain's are; every step's plugin is imported here, so a block naming one
    that is not installed fails now.
    """
    from .plugins.roi import SegmenterPlugin
    keys = block.keys()
    steps: List[tuple] = []
    columns: List[tuple] = [
        ("regions", str, param(
            block.regions, label="for each",
            help="rois: every included ROI of the ROI manager; or the name of "
                 "an id column, such as cluster_id")),
        ("write_back", str, param(
            ", ".join(block.write_back), label="write back",
            help="columns the steps make that go into the real table, comma "
                 "separated; empty: nothing changes the table")),
        ("merge", str, param(
            block.merge, label="overlaps", choices=WriteBack.RULES, advanced=True,
            help="a row in two regions takes the last region's value, the "
                 "first's, their mean, or any (a flag)")),
        ("keep_figures", str, param(
            block.keep_figures, label="keep figure data", choices=KEEP_FIGURES,
            advanced=True,
            help="what each region keeps to draw its figures again; auto: "
                 "for ROIs, not for an id column")),
    ]
    extras: Dict[str, ParamInfo] = {}
    for key, step in zip(keys, block.steps):
        if not step.enabled:
            continue
        if isinstance(step, Block):
            cls = block_class(step)
            columns.append((key, cls.Settings, param(
                default_factory=cls.Settings, label=step.title(), collapsed=True,
                help=cls.description)))
            extras.update(_flat_infos(cls.specs(), f"{key}."))
            steps.append((key, step, cls))
            continue
        cls = step_plugin(step)
        if issubclass(cls, SegmenterPlugin):
            raise ChainError(f"{step.title()!r} finds ROIs, which a block cannot "
                             "keep: put it before the block")
        # a step added to a block starts with its slow options off
        fast = {k: v for k, v in fast_values(cls.Settings).items()
                if k not in step.values}
        if fast:
            step = dataclasses.replace(step, values={**step.values, **fast})
        step_cls = step_settings_class(key, cls, step)
        columns.append((key, step_cls, param(
            default_factory=step_cls, label=step.title(), collapsed=True,
            help=cls.description or step.plugin)))
        extras.update(_flat_infos(cls.specs(), f"{key}."))
        steps.append((key, step, cls))
    settings_cls = make_dataclass("BlockSettings", columns,
                                  namespace={"__module__": __name__})
    name = block.title()
    attrs = {"block": block, "spec": None, "steps": steps, "Settings": settings_cls,
             "params": extras, "name": name, "path": "Chain/Block",
             "description": name + ": " + " -> ".join(s.title() for _, s, _ in steps),
             "version": "1"}
    return type("Block_" + "".join(ch for ch in name if ch.isalnum()), (BlockPlugin,),
                attrs)


def _columns(text) -> List[str]:
    if isinstance(text, (list, tuple)):
        return [str(c).strip() for c in text if str(c).strip()]
    return [c.strip() for c in str(text or "").split(",") if c.strip()]


def _children(children: Dict[str, list]) -> Dict[str, Any]:
    """Inner blocks' tables, one per name, a row per inner region named
    ``outer:inner`` (`docs/regions.md`, nesting)."""
    from .region_table import RegionTable
    out = {}
    for name, parts in children.items():
        merged = RegionTable(name)
        for outer, table in parts:
            for inner in table.ids:
                merged.set(f"{outer}:{inner}", table.entry(inner), table.step(inner),
                           table.data_signature(inner))
        out[name] = merged
    return out
