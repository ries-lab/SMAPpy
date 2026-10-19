"""A new field from an expression in the existing ones.

SMAP's ``Process/Modify/MathParser``: type ``on_time_ms = n_in_group * 20``,
get a column called ``on_time_ms`` that the filter, the renderer and every
other plugin then treat like any other.  It is the escape hatch for everything
the program does not have a button for -- a refractive-index correction on z, a
flag for the localizations a filter cannot express, a ratio between two fitted
quantities -- and the reason it earns its place is that the result is a
*column*, so the rest of the program works on it unchanged.

`smappy.mathparse` is the expression half: what may be written, what it means,
and why it is parsed rather than ``eval``ed.  What is here is the rest:

* **where it is written.**  The whole table, or only the current selection --
  SMAP's "this file / all", generalised to the layer's filter and the ROI.
* **what it means once the localizations are grouped**, which SMAP answers with
  a "regroup" checkbox and a full relink.  Here the field is a recipe
  (`mathparse`), and the recipe says either *recompute* -- evaluate the
  expression again on the grouped table, which is what an expression in
  ``n_in_group`` means -- or a rule to reduce the localizations' values with,
  which is what a per-localization measurement means.  Either way the field
  needs no relinking of its own: the rule goes with the column, and the
  grouped table is derived from the ungrouped one whenever it is built --
  which, for a layer shown grouped, is after every run that changes the
  table, this one included.
* **the history**, SMAP's most-used feature here.  The last twenty
  ``field = expression`` pairs are kept in the settings directory and offered
  in a dropdown, because the expression someone wants today is nearly always
  one they wrote before.

Departures from SMAP, beyond the parser: no ``.*`` (numpy is elementwise
already), the reductions are available (``z_nm - median(z_nm)``), and the
result is float32 like the rest of the table rather than double.

**On regions** (*on*: the ROIs, or an id column) the expression is over
what was measured per region, one row per region, and the result goes into
the source's ``maths`` region table as a recipe computed again when its
inputs are measured again (`smappy.region_maths`, which has the names: an
evaluation by its alias, ``npc_corners.n_corners``); *also onto the
localizations* writes it onto each region's rows as *Regions to column*
does.  SMAP had no such thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..locs import Localizations
from ..mathparse import (CONSTANTS, FUNCTIONS, GROUPED_CHOICES, RECOMPUTE,
                         ExpressionError, evaluate, names_in, recipes, reduces,
                         remember)
from ..region_runs import VALUE_RULES
from ..render import LATERAL_FIELDS
from . import Context, ColumnRecipe, FormState, Notice, Plugin, Result, param, register

#: *on*: the localizations
LOCS = ""

MAX_HISTORY = 20


# ------------------------------------------------------------------ history

def history_file() -> Path:
    """Where the remembered expressions live.

    Beside the configuration rather than in it: this is written on every run
    and would otherwise rewrite a file the user edits by hand.
    """
    from .. import config
    return config.config_dir() / "math_parser.yaml"


def history() -> List[Dict[str, str]]:
    """The remembered expressions, most recent first.

    A history that cannot be read is an empty history, never an error: it is a
    convenience, and the plugin has to work on a machine where the file is
    corrupt, owned by someone else or not there at all.
    """
    import yaml
    path = history_file()
    try:
        entries = yaml.safe_load(path.read_text()) or []
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(entries, list):
        return []
    return [dict(e) for e in entries
            if isinstance(e, dict) and e.get("field") and e.get("expression")]


def remember_expression(field: str, expression: str,
                        grouped: str = RECOMPUTE) -> List[Dict[str, str]]:
    """Put this one at the top of the history, once.  Returns the new list.

    Deduplicated on the pair, not on the expression alone: the same formula
    written into two different fields is two things worth keeping, and SMAP's
    dedup on the expression loses the second.
    """
    entry = {"field": field, "expression": expression.strip(), "grouped": grouped,
             "time": datetime.now().isoformat(timespec="seconds")}
    kept = [e for e in history()
            if (e.get("field"), e.get("expression")) != (entry["field"],
                                                         entry["expression"])]
    entries = [entry] + kept[:MAX_HISTORY - 1]
    import yaml
    try:
        path = history_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(entries, sort_keys=False))
    except OSError:
        pass                    # a history that cannot be written is not a failure
    return entries


def _label(entry: Dict[str, str]) -> str:
    return f"{entry['field']} = {entry['expression']}"


def _recall_choices():
    """The history as dropdown entries.  The value *is* the label, so that a
    settings file saved with one history reads back against another."""
    return [("", "")] + [(_label(e), _label(e)) for e in history()]


# ----------------------------------------------------------------- the work

def write_column(locs: Localizations, field: str, values: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Localizations:
    """A copy of the table with ``field`` set -- everywhere, or under ``mask``.

    A copy rather than a column added in place, because this is what the undo
    keeps: `Session.set_locs` puts the table it replaced on the undo stack,
    and a table edited in place would have nothing to go back to.  The arrays
    themselves are shared, so the copy costs a dictionary -- and so a step of
    the history costs one column rather than a whole table (`smappy.undo`).

    Where a mask leaves a localization out, the field keeps whatever it had
    before, or NaN if the field is new: those rows have no value rather than a
    value of zero, and every summary downstream knows what to do with NaN.
    """
    if mask is None:
        column = values
    else:
        if field in locs:
            column = np.asarray(locs[field]).astype(np.float32, copy=True)
        else:
            column = np.full(len(locs), np.nan, np.float32)
        column[mask] = np.asarray(values, np.float32)[mask]
    columns = dict(locs.columns)
    columns[field] = column
    return Localizations(columns, dict(locs.metadata))


def check_field(field: str, locs: Localizations) -> str:
    """The new field's name, refused now rather than half-written later."""
    name = (field or "").strip()
    if not name:
        raise ValueError("name the field the result goes into")
    if not name.isidentifier():
        raise ValueError(f"{name!r} cannot be used in a later expression: a "
                         "field name is a word of letters, digits and "
                         "underscores, not starting with a digit")
    if name in FUNCTIONS:
        raise ValueError(f"{name!r} is the name of a function an expression "
                         "can call; choose another")
    if name in CONSTANTS:
        raise ValueError(f"{name!r} is a constant in an expression, and a "
                         "column of that name would change what it means; "
                         "choose another")
    if name in ("group_id", "n_in_group"):
        raise ValueError(f"{name!r} is written by merge blinks itself and "
                         "would be overwritten again")
    return name


def describe(field: str, values: np.ndarray) -> str:
    """What was written, in the one line the log keeps."""
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return f"{field}: no finite values"
    return (f"{field}: {finite.size} values, "
            f"{finite.min():.4g} to {finite.max():.4g}, "
            f"median {np.median(finite):.4g}")


def _source_choices():
    return [(LOCS, "localizations")]


@dataclass
class MathSettings:
    source: str = param(LOCS, label="on", choices=_source_choices,
                        help="the localizations, or the regions -- ROIs or an id "
                             "column -- whose measured values the expression reads")
    field: str = param("within", label="new field",
                       help="the column the result is written to")
    expression: str = param("(xy_err_nm < 25) & (sigma_nm > 100)",
                            label="=", kind="text",
                            help="an expression in the column names; `&` and "
                                 "`|` combine comparisons, each in brackets")
    where: str = param("all", label="apply to",
                       choices=[("all", "all localizations"),
                                ("selection", "the selection")],
                       help="the selection is the layer's filter, the ROI and "
                            "the slab together; the rest of the field is NaN")
    grouped: str = param(RECOMPUTE, label="merged blinks",
                         choices=[(RECOMPUTE, "recompute the expression"),
                                  ("mean", "mean of the localizations"),
                                  ("sum", "sum of the localizations"),
                                  ("min", "minimum"), ("max", "maximum"),
                                  ("any", "any localization (flag)"),
                                  ("all", "all localizations (flag)"),
                                  ("none", "leave it off")],
                         help="what the field means once the blinks "
                              "are merged: the expression again, or a rule "
                              "to combine their values")
    recall: str = param("", label="history", choices=_recall_choices,
                        help="an expression used before; choosing one fills "
                             "the field, the expression and the merged-blinks rule")
    to_locs: bool = param(False, label="also onto the localizations",
                          help="on regions: write the value onto each region's "
                               "localizations too, NaN outside")
    overlap: str = param("last", label="where ROIs overlap", choices=VALUE_RULES,
                         advanced=True,
                         help="on ROIs, with also onto the localizations: the value "
                              "of a localization in two ROIs")


@register("Analysis/Process/Math Parser")
class MathParser(Plugin):
    """A new localization field from an expression in the existing ones."""

    Settings = MathSettings
    changes_locs = True
    # a column per localization; the grouped table takes it by the recipe's rule
    grouping = "ungrouped"
    version = "1"

    def react(self, changed: str, settings: MathSettings):
        """Picking from the history fills the field and the expression."""
        if changed != "recall" or not settings.recall:
            return None
        for entry in history():
            if _label(entry) == settings.recall:
                return {"field": entry["field"],
                        "expression": entry["expression"],
                        "grouped": entry.get("grouped", RECOMPUTE)}
        return None

    def describe(self, ctx: Optional[Context], settings: MathSettings) -> FormState:
        """The sources there are, which fields each reads, and on regions
        the aliases the expression can name the evaluations by."""
        on_regions = bool(settings.source)
        state = FormState(active={"where": not on_regions, "grouped": not on_regions,
                                  "to_locs": on_regions,
                                  "overlap": on_regions and settings.to_locs})
        if ctx is None or ctx.session is None:
            return state
        from ..chain import BLOCK_ROIS
        from ..region_filter import sources
        found = sources(ctx.session)
        state.choices["source"] = [(LOCS, "localizations")] + [
            (s, "ROIs" if s == BLOCK_ROIS else f"per {s}") for s in found]
        if on_regions:
            from ..region_maths import aliases
            named = aliases(ctx.session, settings.source)
            if named:
                state.notice = Notice("names: " + "; ".join(
                    f"{word}.<value> ({label})" for word, label in named.items()),
                    kind="info")
            else:
                state.notice = Notice("nothing has been measured on these regions")
        return state

    def changes_locs_for(self, settings: MathSettings) -> bool:
        return not settings.source or bool(settings.to_locs)

    def _regions(self, ctx: Context, settings: MathSettings):
        """On regions: the field, ``{field: expression}`` the maths table
        will hold, and that table, computed."""
        from ..region_maths import compute, recipes
        if ctx.session is None:
            raise ValueError("maths on regions needs a session: the results are "
                             "kept there")
        field = check_field(settings.field, ctx.locs)
        expression = settings.expression.strip()
        # a recipe reads only the recipes before it (`compute`), so one that
        # reads itself is refused there, as a name not yet there
        found = recipes(ctx.session, settings.source)
        found[field] = expression           # a redefinition keeps its place
        return field, found, compute(ctx.session, settings.source, found)

    def _compute(self, ctx: Context, settings: MathSettings):
        """The field's name, its values over the whole table, and the mask."""
        locs = ctx.locs
        field = check_field(settings.field, locs)
        missing = sorted(names_in(settings.expression) - set(locs.keys()))
        if missing:
            hint = ""
            if "n_in_group" in missing or "group_id" in missing:
                hint = ("; on-time and group id are written by merge blinks -- "
                        "tick merge blinks on a layer once and they are there")
            raise ExpressionError(
                f"no column {', '.join(missing)}; the table has: "
                f"{', '.join(sorted(locs.keys()))}{hint}")
        mask = None
        if settings.where == "selection":
            ctx.selection.require(1, ctx.warn, "this")
            mask = np.asarray(ctx.selection.mask, bool)
            # evaluated on the selected rows only, so that a reduction --
            # median(z_nm) -- is the selection's, as in SMAP, and not the
            # whole table's; the rows outside get NaN (see `write_column`)
            values = np.full(len(locs), np.nan)
            values[mask] = np.asarray(evaluate(locs[mask], settings.expression), float)
        else:
            values = evaluate(locs, settings.expression)
        return field, values, mask

    def preview(self, ctx: Context, settings: MathSettings) -> Result:
        """Run the expression and say what it gives, without writing it.

        The expression is the whole plugin, and getting it wrong is cheap to
        do and easy to miss -- a precedence mistake gives numbers, not an
        error.  So there is a way to see the numbers before the table has
        them.
        """
        if settings.source:
            field, _, table = self._regions(ctx, settings)
            values = table.columns()[field]
            return Result(text=f"would keep {describe(field, values)} over "
                               f"{len(values)} regions", settings=settings)
        field, values, mask = self._compute(ctx, settings)
        shown = values if mask is None else np.asarray(values)[mask]
        where = "" if mask is None else f" over the selection ({len(shown)})"
        return Result(text=f"would write {describe(field, shown)}{where}",
                      settings=settings)

    def _run_regions(self, ctx: Context, settings: MathSettings) -> Result:
        from ..chain import BLOCK_ROIS
        from ..region_maths import MATHS
        from ..region_runs import write_values
        field, _, table = self._regions(ctx, settings)
        values = table.columns()[field]
        where = "the ROIs" if settings.source == BLOCK_ROIS else f"per {settings.source}"
        text = (f"{describe(field, values)} over {len(values)} regions ({where}), "
                f"kept as {MATHS}.{field}")
        locs = None
        if settings.to_locs:
            ids = list(table.ids)
            if settings.source == BLOCK_ROIS:      # written by the ROIs' numbers
                rois = ctx.session.rois.rois
                ids = [str(rois[i].number) for i in ids]
            locs, _ = write_values(ctx, ctx.locs, settings.source,
                                   {field: dict(zip(ids, values.tolist()))},
                                   rule=settings.overlap,
                                   origin={"evaluation": MATHS})
            text += "; written onto the localizations"
        remember_expression(field, settings.expression, RECOMPUTE)
        return Result(locs=locs, settings=settings, text=text,
                      id_regions={settings.source: {MATHS: table}})

    def run(self, ctx: Context, settings: MathSettings) -> Result:
        if settings.source:
            return self._run_regions(ctx, settings)
        field, values, mask = self._compute(ctx, settings)
        locs = write_column(ctx.locs, field, values, mask)

        # How the field reaches the grouped table.  A field computed over part
        # of the table only is not a function of the table, so its expression
        # cannot be re-evaluated anywhere; it can still be *reduced*, which is
        # what the localizations that do have a value say about their blink.
        rule = settings.grouped if settings.grouped in GROUPED_CHOICES else RECOMPUTE
        note = ""
        if mask is not None and rule == RECOMPUTE:
            rule = "mean"
            note = ("; on merged blinks it is the mean of the "
                    "localizations, since the expression cannot be "
                    "recomputed for a selection")
        if field in names_in(settings.expression):
            # ``photons = photons * 2`` is a change made once, not a recipe:
            # re-evaluated -- and every grouping re-evaluates recipes on the
            # ungrouped table -- it compounds (the photons came back x4 after
            # one grouping) and the grouped table, which has no photons of
            # its own to read, lost the column.  So no expression is kept.  A
            # measured column goes on being reduced by its own rule (photons
            # add up per blink); a derived one keeps only its reduction.
            previous = next((r for r in recipes(ctx.locs) if r["field"] == field),
                            None)
            if previous is not None:
                kept = previous.get("grouped", "mean")
                remember(locs, field, "", grouped="mean" if kept == RECOMPUTE else kept)
            note += "; applied once, since it reads the field it writes"
        else:
            remember(locs, field, settings.expression, grouped=rule,
                     where="selection" if mask is not None else None)
        remember_expression(field, settings.expression, rule)

        shown = values if mask is None else np.asarray(values)[mask]
        where = "" if mask is None else f" (selection of {len(ctx.locs)})"
        recipe = (MathRecipe(field, settings.expression)
                  if self.column_recipe_for(settings) else None)
        return Result(locs=locs, settings=settings, recipe=recipe,
                      text=f"{describe(field, shown)}{where}{note}")

    def column_recipe_for(self, settings: MathSettings) -> bool:
        """A field over the whole table, from other columns, and not a
        position: that can follow a live fit (`MathRecipe`).  Over the
        selection it cannot -- which rows a block adds to the selection is a
        question about filters and an ROI that may change meanwhile -- and a
        field that reads itself is a change made once, not a recipe."""
        field = (settings.field or "").strip()
        if settings.source or settings.where != "all" or field in LATERAL_FIELDS:
            return False
        try:
            return field not in names_in(settings.expression)
        except ExpressionError:
            return False


class MathRecipe(ColumnRecipe):
    """The expression, evaluated for each block a live fit appends -- and
    over the whole table again when it reduces a column (`reduces`), since
    then every row's value moves with the table."""

    def __init__(self, field: str, expression: str):
        self.field, self.expression = field, expression
        self.whole = reduces(expression)

    def columns(self, locs: Localizations) -> dict:
        return {self.field: evaluate(locs, self.expression)}
