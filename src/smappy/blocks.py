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
import time
from dataclasses import dataclass, make_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .chain import (BLOCK_ROIS, DEFAULT_GROUP, KEEP_FIGURES, Block, ChainError,
                    ChainPlugin, Step, _flat_infos, plugin_settings, step_plugin,
                    step_settings_class)
from .plugins import (Context, ParamInfo, PreflightChoice, PreflightQuestion, Result,
                      fast_values, param, settings_values)
from .region_runs import WriteBack, changed_columns

# the block's own fields, before its steps', in its settings
OWN = ("regions", "write_back", "merge", "keep_figures")
# how a block's time is estimated (docs/regions.md, "Necessary slow work is
# timed"): the steps run in a few regions spread over the source, at most
# this many or this long, and the median per step is multiplied out
SAMPLE_REGIONS = 5
SAMPLE_SECONDS = 1.0
# a run estimated to take longer than this asks before it starts
ASK_ABOVE_SECONDS = 60.0


@dataclass
class Estimate:
    """What a block will cost: per step and region, and over all of them."""
    regions: int
    per_region: Dict[str, float]          # step label -> seconds, the median
    sampled: int
    # columns the steps added or changed in the sampled regions that the
    # block does not write back: what the editor's check warns about
    unwritten: List[str] = dataclasses.field(default_factory=list)

    @property
    def total(self) -> float:
        return self.regions * sum(self.per_region.values())

    def text(self) -> str:
        from .region_runs import _duration
        steps = ", ".join(f"{label} {_ms(t)}" for label, t in self.per_region.items())
        total = _duration(self.total)
        text = (f"{'' if total.startswith('under') else 'about '}{total} "
                f"over {self.regions} regions "
                f"({steps} a region, timed on {self.sampled})")
        if self.unwritten:
            text += (f"; the steps change {', '.join(self.unwritten)}, which the "
                     "block does not write back")
        return text


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.2g} ms" if seconds < 1 else f"{seconds:.2g} s"


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

    # ------------------------------------------------------------- timing
    def estimate(self, ctx: Context, settings) -> Optional[Estimate]:
        """Time the block's steps in a few of its regions, nothing kept
        (`SAMPLE_REGIONS`, `SAMPLE_SECONDS`), and multiply out.  None when
        there is nothing to time -- no regions, or inside another block."""
        from .region_runs import ColumnRegions, RegionScratch, changed_columns
        if ctx.per_region:
            return None
        steps = self.pipeline(settings)
        written = set(_columns(settings.write_back))
        unwritten: List[str] = []
        if settings.regions == BLOCK_ROIS:
            project = ctx.rois
            if project is None:
                return None
            ids = [i for i, roi in project.rois.items() if roi.reviewed and roi.use]

            def cut(k):
                roi = project.rois[ids[k]]
                region = project.region(roi)
                return RegionScratch(project._region_tables(roi, region), region, roi.id,
                                     tag=True)
            n = len(ids)
        else:
            if settings.regions not in ctx.locs:
                return None
            source = ColumnRegions(ctx.session, settings.regions, ctx.layer,
                                   locs=ctx.locs)
            n = len(source)

            def cut(k):
                return RegionScratch(source.tables(k), None, source.region_ids()[k],
                                     tag=True)
        if not n or not steps:
            return None
        picks = np.unique(np.linspace(0, n - 1, min(n, SAMPLE_REGIONS)).astype(int))
        times: Dict[str, list] = {s.label: [] for s in steps}
        started = time.perf_counter()
        sampled = 0
        for k in picks:
            scratch = cut(int(k))
            original = scratch.locs
            for step in steps:
                t0 = time.perf_counter()
                try:
                    result = step.plugin.run(scratch.context(), step.settings)
                    scratch.apply(result, step.label)
                except Exception:
                    pass                  # a failure costs what it cost
                times[step.label].append(time.perf_counter() - t0)
            unwritten += [c for c in changed_columns(original, scratch)
                          if c not in written and c not in unwritten]
            sampled += 1
            if time.perf_counter() - started > SAMPLE_SECONDS:
                break
        return Estimate(n, {label: float(np.median(t)) for label, t in times.items()},
                        sampled, unwritten)

    def preflight(self, ctx: Context, settings):
        """Its steps' own questions, and the time it will take when that is
        long -- offering to run with the slow options off when there are any."""
        asked = super().preflight(ctx, settings)
        try:
            estimate = self.estimate(ctx, settings)
        except Exception:
            estimate = None               # an estimate is a courtesy
        if estimate is None or estimate.total < ASK_ABOVE_SECONDS:
            return asked
        text = f"{self.block.title()} will take {estimate.text()}."
        if asked:
            text += "\n\n" + (asked.text if isinstance(asked, PreflightQuestion)
                               else str(asked))
        fast = self.without_slow(settings)
        choices = ([PreflightChoice(label="Slow options off", settings=fast,
                                    help="run every step without the options "
                                         "marked slow")]
                   if fast is not None else [])
        return PreflightQuestion(text=text, choices=choices)

    def without_slow(self, settings):
        """These settings with every slow option of its steps off, or None
        when none is on."""
        out = copy.deepcopy(settings)
        changed = False
        for key, step, cls in self.steps:
            sub = getattr(out, key)
            if isinstance(step, Block):
                inner = cls().without_slow(sub)
                if inner is not None:
                    setattr(out, key, inner)
                    changed = True
                continue
            for name in fast_values(cls.Settings):
                if "." not in name and getattr(sub, name, False):
                    setattr(sub, name, False)
                    changed = True
        return out if changed else None

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
