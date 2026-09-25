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
  which is what a per-localization measurement means.  Either way nothing is
  relinked: the grouped table is derived from the ungrouped one when it is
  next built.
* **the history**, SMAP's most-used feature here.  The last twenty
  ``field = expression`` pairs are kept in the settings directory and offered
  in a dropdown, because the expression someone wants today is nearly always
  one they wrote before.

Departures from SMAP, beyond the parser: no ``.*`` (numpy is elementwise
already), the reductions are available (``z_nm - median(z_nm)``), and the
result is float32 like the rest of the table rather than double.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..columns import current
from ..locs import Localizations
from ..mathparse import (FUNCTIONS, GROUPED_CHOICES, RECOMPUTE,
                         ExpressionError, evaluate, names_in, remember)
from . import Context, Plugin, Result, param, register

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
        entries = current(yaml.safe_load(path.read_text()) or [])
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
    if name in ("group_id", "n_in_group"):
        raise ValueError(f"{name!r} is written by the grouping itself and "
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


@dataclass
class MathSettings:
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
    grouped: str = param(RECOMPUTE, label="grouped",
                         choices=[(RECOMPUTE, "recompute the expression"),
                                  ("mean", "mean of the localizations"),
                                  ("sum", "sum of the localizations"),
                                  ("min", "minimum"), ("max", "maximum"),
                                  ("any", "any localization (flag)"),
                                  ("all", "all localizations (flag)"),
                                  ("none", "leave it off")],
                         help="what the field means once the localizations "
                              "are grouped; nothing is relinked either way")
    recall: str = param("", label="history", choices=_recall_choices,
                        help="an expression used before; choosing one fills "
                             "the two fields above")


@register("Analysis/Process/Math Parser")
class MathParser(Plugin):
    """A new localization field from an expression in the existing ones."""

    Settings = MathSettings
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

    def _compute(self, ctx: Context, settings: MathSettings):
        """The field's name, its values over the whole table, and the mask."""
        locs = ctx.locs
        field = check_field(settings.field, locs)
        missing = sorted(names_in(settings.expression) - set(locs.keys()))
        if missing:
            hint = ""
            if "n_in_group" in missing or "group_id" in missing:
                hint = ("; on-time and group id are written by the grouping -- "
                        "switch a layer to grouped once and they are there")
            raise ExpressionError(
                f"no column {', '.join(missing)}; the table has: "
                f"{', '.join(sorted(locs.keys()))}{hint}")
        values = evaluate(locs, settings.expression)
        mask = None
        if settings.where == "selection":
            ctx.selection.require(1, ctx.report, "this")
            mask = ctx.selection.mask
        return field, values, mask

    def preview(self, ctx: Context, settings: MathSettings) -> Result:
        """Run the expression and say what it gives, without writing it.

        The expression is the whole plugin, and getting it wrong is cheap to
        do and easy to miss -- a precedence mistake gives numbers, not an
        error.  So there is a way to see the numbers before the table has
        them.
        """
        field, values, mask = self._compute(ctx, settings)
        shown = values if mask is None else np.asarray(values)[mask]
        where = "" if mask is None else f" over the selection ({len(shown)})"
        return Result(text=f"would write {describe(field, shown)}{where}",
                      settings=settings)

    def run(self, ctx: Context, settings: MathSettings) -> Result:
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
            note = ("; on the grouped table it is the mean of the "
                    "localizations, since the expression cannot be "
                    "recomputed for a selection")
        remember(locs, field, settings.expression, grouped=rule,
                 where="selection" if mask is not None else None)
        remember_expression(field, settings.expression, rule)

        shown = values if mask is None else np.asarray(values)[mask]
        where = "" if mask is None else f" (selection of {len(ctx.locs)})"
        return Result(locs=locs, settings=settings,
                      text=f"{describe(field, shown)}{where}{note}")
