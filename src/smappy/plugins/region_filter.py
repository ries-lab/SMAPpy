"""Filter regions: keep the localizations of the regions whose results pass.

The filter is `smappy.region_filter` (its docstring has the rules: per
region, OR within a source, AND across sources, out where there is no
region); this is the step that sets it, for a chain and a batch -- the
Regions tab's *Region filter* section edits the same conditions by hand.
One run adds, replaces or clears; each condition is one value of one
evaluation and a range, or the texts it accepts.

What it changes is not the table but its conditions and the flag they give
(`region_filter`), so it is no undo step; it is logged, since what a later
measurement saw depends on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import Context, FormState, Notice, Plugin, Result, param, register

MODES = (("add", "add to the conditions"), ("replace", "replace the conditions"),
         ("clear", "clear every condition"))


def _evaluations(ctx) -> List[str]:
    """Every evaluation a condition can read, as *results from* spells it,
    the ROIs' annotations among them."""
    from ..region_filter import ANNOTATIONS, sources
    from ..region_results import Evaluation
    from ..chain import BLOCK_ROIS
    if ctx is None or ctx.session is None:
        return []
    out = []
    for source, labels in sources(ctx.session).items():
        for label in labels:
            out.append(str(Evaluation(source, label)) if label != ANNOTATIONS or
                       source != BLOCK_ROIS else ANNOTATIONS)
    return out


def condition_of(settings) -> "Condition":
    """The condition the settings describe; ValueError for an incomplete one."""
    from ..region_filter import ANNOTATIONS, Condition
    from ..region_results import Evaluation
    from ..chain import BLOCK_ROIS
    if not settings.evaluation:
        raise ValueError("choose the results to filter on")
    if not settings.value:
        raise ValueError("choose the value to filter on")
    ev = (Evaluation(BLOCK_ROIS, ANNOTATIONS) if settings.evaluation == ANNOTATIONS
          else Evaluation.parse(settings.evaluation))
    accept = [a.strip() for a in settings.accept.split(",") if a.strip()]
    if accept:
        return Condition(ev.source, ev.label, settings.value, accept=accept)
    if settings.low is None and settings.high is None:
        raise ValueError("give a range (low, high or both) or the texts to accept")
    return Condition(ev.source, ev.label, settings.value, settings.low, settings.high)


@dataclass
class FilterRegionsSettings:
    mode: str = param("add", label="then", choices=MODES,
                      help="add this condition to the others (all must hold), "
                           "replace them by it, or clear them all")
    evaluation: str = param("", label="results from", choices=lambda: [("", "choose...")],
                            help="the evaluation, as the summaries name it; "
                                 "annotations: an ROI's use and keywords")
    value: str = param("", label="value", choices=lambda: [("", "choose...")],
                       help="the value of each region to test")
    low: Optional[float] = param(None, label="at least", help="empty: no lower bound")
    high: Optional[float] = param(None, label="at most", help="empty: no upper bound")
    accept: str = param("", label="or one of",
                        help="texts to accept, separated by commas, for a keyword "
                             "or a flag (true, false); replaces the range")


@register("Analysis/Regions/Filter regions")
class FilterRegions(Plugin):
    """Keeps the localizations of the regions -- ROIs, clusters, traces --
    whose results pass a condition, by one flag every layer honours."""

    Settings = FilterRegionsSettings
    changes_locs = False
    logged = True
    version = "1"

    def describe(self, ctx: Optional[Context], settings: FilterRegionsSettings
                 ) -> FormState:
        state = FormState()
        clear = settings.mode == "clear"
        for name in ("evaluation", "value", "low", "high", "accept"):
            state.active[name] = not clear
        if ctx is None or ctx.session is None:
            return state
        found = _evaluations(ctx)
        state.choices["evaluation"] = [("", "choose..." if found else
                                        "nothing measured yet")] + [(e, e) for e in found]
        values = self._values(ctx, settings)
        state.choices["value"] = [("", "choose...")] + [(v, v) for v in values]
        current = ctx.session.region_filter
        if clear:
            state.notice = Notice(f"would clear {len(current)} condition"
                                  f"{'' if len(current) == 1 else 's'}", kind="info")
            return state
        try:
            condition = condition_of(settings)
        except ValueError:
            return state
        from ..region_filter import evaluate
        outcome = evaluate(ctx.session, [condition], members=False)
        kept, total = outcome.counts[0] if outcome.counts else (0, 0)
        state.notice = Notice(f"keeps {kept} of {total} regions"
                              + ("; " + outcome.warnings[0] if outcome.warnings else ""),
                              kind="info")
        return state

    @staticmethod
    def _values(ctx, settings) -> List[str]:
        from ..region_filter import ANNOTATIONS, region_values
        from ..region_results import Evaluation
        from ..chain import BLOCK_ROIS
        if not settings.evaluation:
            return []
        ev = (Evaluation(BLOCK_ROIS, ANNOTATIONS) if settings.evaluation == ANNOTATIONS
              else Evaluation.parse(settings.evaluation))
        return sorted(region_values(ctx.session, ev.source, ev.label)[1])

    def run(self, ctx: Context, settings: FilterRegionsSettings) -> Result:
        from ..region_filter import evaluate
        if ctx.session is None:
            raise ValueError("the region filter needs a session: its regions' "
                             "results are kept there")
        current = list(ctx.session.region_filter)
        if settings.mode == "clear":
            return Result(text=f"cleared {len(current)} region filter conditions",
                          region_filter=[], settings=settings)
        condition = condition_of(settings)
        if not self._values(ctx, settings):
            raise ValueError(f"there are no results {settings.evaluation!r} to filter on")
        found = [condition] if settings.mode == "replace" else current + [condition]
        outcome = evaluate(ctx.session, found)
        lines = [f"{c.text()}: {k} of {n} regions"
                 for c, (k, n) in zip(found, outcome.counts)]
        if outcome.flag is not None:
            lines.append(f"{int(outcome.flag.sum())} of {len(outcome.flag)} "
                         "localizations in regions that pass")
        for warning in outcome.warnings:
            ctx.warn(warning)
        return Result(text="\n".join(lines), region_filter=[c.to_dict() for c in found],
                      settings=settings, values={"conditions": len(found)})
