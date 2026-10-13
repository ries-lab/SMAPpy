"""A block of a chain: its steps run once in each region.

`docs/regions.md` ("Evaluation is a block of a chain") has the decisions; a
block is the chain's own `Block` (its file form), and `block_class` makes it a
plugin as `chain_class` makes a chain one, so the GUI draws a block as a part
holding its steps and a chain runs it as a step like any other.

What a run of a block does:

* **The regions** are the included ROIs of the ROI manager, or the regions an
  id column names; inside another block's region they are the regions of
  that region's table: blocks nest, and a region narrows.  One adapter each
  (`region_runs.regions_of`), with one interface, so running, timing and
  writing per region are written once.
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
* **Write values** puts numbers the steps measured onto the table, each
  region's onto its rows (`region_runs.write_values`; `docs/regions.md`,
  "Region values as columns"), so a filter or a colour can use them; the
  columns are refused when the table has one of the name that was not
  written from regions.
* **A segmenter is refused** inside a block: ROIs found in each region have
  nowhere to go yet.
* **A block inside a block** makes a table per region, not a row: its tables
  are kept beside the outer step's, a row per inner region named
  ``outer:inner``.  What it changes stays in the region's copy, and what was
  asked to be kept of it is warned about (`nested_losses`).
* **Stop** (`Context.stop`) ends the run between regions, and what the
  regions so far measured and wrote back is the result, which says
  ``stopped`` (`Result.stopped`).  A region that has begun finishes: its
  steps are not told to stop, since half a region's numbers would be kept as
  if they were all of them.
"""
from __future__ import annotations

import copy
import dataclasses
import time
from dataclasses import dataclass, make_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .chain import (BLOCK_ROIS, DEFAULT_GROUP, Block, ChainError,
                    ChainPlugin, Step, _flat_infos, plugin_settings, step_plugin,
                    step_settings_class)
from .plugins import (Context, FormState, ParamInfo, PreflightQuestion,
                      Result, param, settings_values)
from .region_runs import WriteBack, changed_columns

# how a block's time is estimated (docs/regions.md, "Necessary slow work is
# timed"): the steps run in a few regions spread over the source, at most
# this many or this long, and the median per step is multiplied out
SAMPLE_REGIONS = 5
SAMPLE_SECONDS = 1.0
# a run estimated to take longer than this asks before it starts
ASK_ABOVE_SECONDS = 60.0
# why *overlaps* any is refused beside *write values*: a flag's rule, which
# means nothing for a number measured over a region
ANY_FOR_VALUES = ("overlaps: any applies to write back only; written values "
                  "take last, first or mean")
# why what a block inside a block's region does to the localizations is lost
NESTED_LOST = ("inside a nested block, changes to the localizations stay in "
               "that region's copy")


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
    # its regions are an id column of the localizations, and what it writes
    # back goes onto them; each region's steps see the layers as shown
    grouping = "ungrouped"
    # whether it writes columns back: `block_class` answers for the block's
    # own values, `changes_locs_for` for the form's
    changes_locs = True

    def changes_locs_for(self, settings) -> bool:
        return bool(_columns(settings.write_back) or _columns(settings.write_values))

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        """Its steps' own, and no grouping to choose: in a region every layer
        is cut as it is shown.  *overlaps* is read only when something is
        written, and *any* not for written values (`ANY_FOR_VALUES`)."""
        from .chain import GROUPING_FIELD
        from .plugins import Notice
        from .region_runs import VALUE_RULES
        state = super().describe(ctx, settings)
        for key, step, cls in self.steps:
            if hasattr(getattr(settings, key), GROUPING_FIELD):
                state.active[f"{key}.{GROUPING_FIELD}"] = False
        state.active["merge"] = bool(_columns(settings.write_back)
                                     or _columns(settings.write_values))
        said = []
        if _columns(settings.write_values) and settings.merge not in VALUE_RULES:
            said.append(ANY_FOR_VALUES)
        try:
            said += nested_losses(self.block_from(settings))
        except Exception:
            pass                           # best effort, like the steps' own
        if said:
            # this block's word replaces an inner block's, which says less
            state.notice = Notice("; ".join(said), kind="warning")
        return state

    @classmethod
    def evaluations(cls, settings) -> List[Tuple[str, str]]:
        """``(name, plugin path)`` of what it leaves, a pair per step: on the
        ROIs, or per region of its id column -- the name then in the form
        *results from* reads (`region_results.Evaluation`), so an analysis
        after the block takes the block's own without asking."""
        from .region_results import Evaluation
        return [(str(Evaluation(settings.regions, label)), step_cls.path)
                for label, (key, step, step_cls) in zip(cls.labels(), cls.steps)
                if not isinstance(step, Block)]

    @classmethod
    def labels(cls) -> List[str]:
        """Each step's name, its results' and its columns': the title, and
        ``title 2`` for a second step of the same title, so two steps never
        write into one another's."""
        out: List[str] = []
        seen: Dict[str, int] = {}
        for key, step, step_cls in cls.steps:
            label = step.title()
            seen[label] = seen.get(label, 0) + 1
            out.append(label if seen[label] == 1 else f"{label} {seen[label]}")
        return out

    # ------------------------------------------------------------- timing
    def estimate(self, ctx: Context, settings) -> Optional[Estimate]:
        """Time the block's steps in a few of its regions, nothing kept
        (`SAMPLE_REGIONS`, `SAMPLE_SECONDS`), and multiply out.  None when
        there is nothing to time -- no regions, or inside another block."""
        from .region_runs import RegionScratch, changed_columns, regions_of
        if ctx.per_region:
            return None
        steps = self.pipeline(settings)
        written = set(_columns(settings.write_back))
        unwritten: List[str] = []
        try:
            source = regions_of(ctx, settings.regions)
        except ValueError:
            return None                   # no ROI project, or no such column
        n = len(source)

        def cut(k):
            return RegionScratch(source.tables(k), source.region(k),
                                 source.region_ids()[k], tag=True)
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
                    result = step.plugin.run(
                        scratch.context(grouping=step.plugin.grouping), step.settings)
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
        long (`ASK_ABOVE_SECONDS`)."""
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
        return text

    def block_from(self, settings) -> Block:
        """The block as its file would say it, with ``settings``."""
        block = copy.deepcopy(self.block)
        block.regions = settings.regions
        block.write_back = _columns(settings.write_back)
        block.write_values = _columns(settings.write_values)
        block.merge = settings.merge
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
        out = []
        for label, (key, step, cls) in zip(self.labels(), self.steps):
            sub = getattr(settings, key)
            own = sub if isinstance(step, Block) else plugin_settings(cls, sub)
            out.append(Run(label=label, path=cls.path, plugin=cls(),
                           settings=own, version=str(cls.version)))
        return out

    def run(self, ctx: Context, settings) -> Result:
        steps = self.pipeline(settings)
        if not steps:
            raise ValueError("the block has no steps")
        columns = _columns(settings.write_back)
        back = WriteBack(columns, settings.merge)
        written = _columns(settings.write_values)
        if written:
            from .region_runs import VALUE_RULES, check_writable
            if settings.merge not in VALUE_RULES:
                raise ValueError(ANY_FOR_VALUES)
            check_writable(ctx.locs, written)
        warned: List[str] = []
        children: Dict[str, Any] = {}
        if not ctx.per_region:
            # once, from the outermost block, rather than once per region
            for lost in nested_losses(self.block_from(settings)):
                ctx.warn(f"{self.block.title()}: {lost}")

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

        from .region_runs import regions_of
        source = regions_of(ctx, settings.regions)
        tables, n = source.evaluate(steps, report=ctx.report, stop=ctx.stop,
                                    after=after, tag=bool(back), reuse=not back,
                                    chain=f"block {self.block.title()}")
        tables = {s.label: tables[s.label] for s in steps if s.label in tables}
        total, where = len(source), source.where
        stopped = ctx.stopped and n < total
        merged = _children(children)
        if merged:
            source.keep(merged)       # the inner blocks' tables, beside the steps' own
        failed = sum(1 for t in tables.values() for e in t.errors if e)
        text = (f"stopped after {n} of {total} {where}" if stopped else f"{n} {where}"
                ) + f", {len(steps)} step(s)"
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
        if written and n:
            locs = self._write_values(ctx, settings, tables, written,
                                      locs if locs is not None else ctx.locs)
            text += f"; wrote {', '.join(written)}"
        return Result(locs=locs, text=text, settings=settings,
                      values={"regions": n},
                      data={"tables": tables, "regions": n, "failed": failed},
                      region_tables={**tables, **merged},
                      evaluations=(self.evaluations(settings)
                                   if not ctx.per_region else []),
                      stopped=stopped)

    def _write_values(self, ctx, settings, tables, names, locs):
        """The steps' numbers ``names`` onto the rows of their regions
        (`region_runs.write_values`, as *Regions to column* writes them); a
        number two steps return is the later step's."""
        from .region_runs import table_values, write_values
        values: Dict[str, Dict[str, float]] = {}
        owner: Dict[str, str] = {}
        for label, table in tables.items():
            for name in names:
                try:
                    found = table_values(table, [name])[name]
                except ValueError:
                    continue
                if name in owner:
                    ctx.warn(f"{self.block.title()}: {name} is measured by "
                             f"{owner[name]} and {label}; {label}'s is written")
                values[name], owner[name] = found, label
        missing = [k for k in names if k not in values]
        if missing:
            raise ValueError(f"{self.block.title()}: no step measured "
                             f"{', '.join(missing)}, which it is to write")
        if settings.regions == BLOCK_ROIS:
            # the steps' tables know an ROI by its id; a written value, as a
            # results row, by its number
            number = ctx.rois.numbers()
            values = {name: {str(number[r]): v for r, v in per.items() if r in number}
                      for name, per in values.items()}
        out, _ = write_values(ctx, locs, settings.regions, values, settings.merge,
                              {"source": f"block {self.block.title()}"})
        return out



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
        ("write_values", str, param(
            ", ".join(block.write_values), label="write values",
            help="numbers the steps measure, comma separated, each region's "
                 "written onto its localizations")),
        ("merge", str, param(
            block.merge, label="overlaps", choices=WriteBack.RULES, advanced=True,
            help="a row in two regions takes the last region's value, the "
                 "first's, their mean, or any (a flag written back)")),
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
             "version": "1",
             # as its own settings start; `changes_locs_for` reads the form's
             "changes_locs": bool(_columns(block.write_back)
                                  or _columns(block.write_values))}
    return type("Block_" + "".join(ch for ch in name if ch.isalnum()), (BlockPlugin,),
                attrs)


def nested_losses(block: Block) -> List[str]:
    """What the blocks inside ``block`` (at any depth) would change in the
    localizations and then throw away, a sentence each.

    Inside a block's region a block runs on that region's copy of the table,
    and nothing it writes reaches the real one (`ContextRegions` runs without
    the rows a write-back needs).  A step that changes the copy for the step
    after it is the normal use -- a filter before a measurement -- and its
    effect is meant to go with the region.  What is reported is what was
    asked to be kept: an inner block's *write back*; its *write values*,
    unless the block around it writes those columns back; and a last step
    that changes the table, whose change nothing reads.
    """
    out: List[str] = []
    for inner in block.steps:
        if not isinstance(inner, Block) or not inner.enabled:
            continue
        name = inner.title()
        if inner.write_back:
            out.append(f"{name} writes back {', '.join(inner.write_back)}, which is "
                       f"discarded: {NESTED_LOST}")
        lost = [c for c in inner.write_values if c not in block.write_back]
        if lost:
            out.append(f"{name} writes the values {', '.join(lost)}, which "
                       f"{block.title()} does not write back, so they are discarded: "
                       f"{NESTED_LOST}")
        steps = [s for s in inner.steps if s.enabled]
        if steps and not isinstance(steps[-1], Block) and _changes_table(steps[-1]):
            out.append(f"{steps[-1].title()}, the last step of {name}, changes the "
                       f"localizations and nothing reads them after it, so the "
                       f"change is discarded: {NESTED_LOST}")
        out += nested_losses(inner)
    return out


def _changes_table(step) -> bool:
    """Whether a chain step hands back a new table with its values."""
    from .plugins import settings_from
    try:
        cls = step_plugin(step)
        return bool(cls().changes_locs_for(settings_from(cls.Settings, step.values)))
    except Exception:
        return False


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
