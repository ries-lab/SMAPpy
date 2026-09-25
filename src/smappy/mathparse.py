"""Expressions over localization columns, and the fields they define.

A user who wants a column the fitter did not produce -- ``z`` corrected for a
refractive index mismatch, an on-time in milliseconds, a flag marking the
localizations a filter cannot express -- writes it as an expression in the
column names:

    on_time_ms = n_in_group * 20
    within     = (xy_err_nm < 25) & (sigma_nm > 100)

This is SMAP's ``Process/Modify/MathParser``, with two departures.  MATLAB
needs ``.*`` and ``./`` to stay elementwise and numpy does not, so the dots are
gone; and where SMAP ``eval``s the string, this parses it and walks the tree.
That is not caution for its own sake: an expression is saved in a workspace and
in a pipeline file, so it travels between people, and ``eval`` of a string that
arrived in a file is a shell.  Walking the tree also gives the two things the
plugin needs anyway -- which columns an expression reads, so a missing one is
named before anything is computed, and a real position in the string to point
at when something is wrong.

**A derived field is a recipe, not data.**  The expression is kept in the
table's metadata (`recipes`) and the grouped table gets the field too, one of
two ways, because both are right for different fields:

``recompute``    evaluate the expression again on the grouped table.  This is
                 what ``on_time_ms = n_in_group * 20`` means -- it is a
                 statement about a blink, and on the grouped table it is the
                 blink's own on-time rather than an average of anything.
``mean``, ...    reduce the localizations' values with that rule, the way
                 grouping reduces every other column.  This is what
                 ``photons_kept = photons * (logl_rel > -1)`` means -- it is a
                 measurement per localization, and the group's value is the
                 sum or the mean of the measurements.

The recipe is metadata, so both survive a save and a reload: the field is still
there next week, and still says what it was computed from.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from .locs import Localizations

# Where the recipes live in `Localizations.metadata`.  A list, not a dict:
# a recipe may use a field an earlier one defined, so the order is part of it.
RECIPES = "derived"

# What a recipe says to do on the grouped table.  `recompute` evaluates the
# expression there; the rest reduce the localizations' values with that rule,
# and `none` leaves the field off the grouped table altogether.  `any` and
# `all` are for a flag: was any localization of this blink kept, were they all.
RECOMPUTE = "recompute"
COMBINE_RULES = ("mean", "sum", "min", "max", "any", "all")
GROUPED_CHOICES = (RECOMPUTE,) + COMBINE_RULES + ("none",)

# What an expression may call.  Only what is defined for a whole column at
# once, plus the reductions -- `z_nm - median(z_nm)` is the most asked-for
# expression there is, and MATLAB gives it away for free.
FUNCTIONS: Dict[str, Callable] = {
    # elementwise
    "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp,
    "log": np.log, "log2": np.log2, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "arctan2": np.arctan2, "hypot": np.hypot,
    "floor": np.floor, "ceil": np.ceil, "round": np.round, "sign": np.sign,
    "mod": np.mod, "rem": np.remainder,
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip,
    "where": np.where,
    "isfinite": np.isfinite, "isnan": np.isnan,
    "float": lambda v: np.asarray(v, np.float64),
    "int": lambda v: np.asarray(v, np.int64),
    # reductions: one number from a column
    "mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd,
    "sum": np.nansum, "min": np.nanmin, "max": np.nanmax,
    "percentile": np.nanpercentile, "count": np.size,
}

CONSTANTS: Dict[str, Any] = {"pi": np.pi, "e": np.e, "nan": np.nan,
                             "inf": np.inf, "True": True, "False": False}

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow,
           ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
           ast.BitXor: operator.xor}

_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg,
          ast.Invert: operator.invert}

_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
            ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}

_BITWISE = (ast.BitAnd, ast.BitOr, ast.BitXor)


class ExpressionError(ValueError):
    """An expression that will not run, said in terms of what was typed."""


def _where(node: ast.AST, expression: str) -> str:
    """The piece of the expression a node came from, for the message."""
    piece = ast.get_source_segment(expression, node)
    return f" in `{piece}`" if piece else ""


def _bracketed(node: ast.AST, text: str) -> bool:
    """Whether the user put this piece in brackets.

    The tree cannot say -- it is the same tree either way -- so the text is
    asked instead.  It is the difference between `a < (b & c)`, which someone
    meant, and `a < b & c`, which nobody does.
    """
    if "\n" in text:
        return True                  # not worth guessing across lines
    i = node.col_offset - 1
    while i >= 0 and text[i] == " ":
        i -= 1
    return i >= 0 and text[i] == "("


def parse(expression: str) -> ast.Expression:
    """The expression as a tree, with the mistakes that read as Python named.

    Three of them are worth catching by hand, because numpy's own answer is
    either a crash far from the cause or -- worse -- a wrong number:

    * ``a and b``, which asks a whole column for a truth value;
    * ``0 < x < 5``, which is the same question in disguise;
    * ``a < 25 & b > 100``, which Python reads as ``a < (25 & b) > 100``
      because ``&`` binds tighter than ``<``.  This is the classic numpy trap
      and the one SMAP users hit first, since MATLAB's precedence is the
      other way round.
    """
    text = (expression or "").strip()
    if not text:
        raise ExpressionError("no expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"cannot read the expression: {error.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            raise ExpressionError(
                "`and` / `or` ask a whole column whether it is true; write "
                "`&` and `|` between the comparisons, each in brackets"
                + _where(node, text))
        if isinstance(node, ast.Compare):
            for part in node.comparators:
                if (isinstance(part, ast.BinOp) and isinstance(part.op, _BITWISE)
                        and not _bracketed(part, text)):
                    raise ExpressionError(
                        "`&` and `|` bind tighter than a comparison, so this "
                        "reads as `a < (b & c)`: put each comparison in "
                        "brackets" + _where(node, text))
            if len(node.ops) > 1:
                raise ExpressionError(
                    "a chained comparison asks a column whether it is true; "
                    "write `(0 < x) & (x < 5)`" + _where(node, text))
    return tree


def names_in(expression: str) -> Set[str]:
    """Every column name the expression reads.

    Function names and constants are not columns, so they are left out: what
    comes back is exactly what the table has to provide.
    """
    tree = parse(expression)
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} \
        - called - set(CONSTANTS)


def _evaluate_node(node: ast.AST, names: Dict[str, Any], expression: str) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, names, expression)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, complex)):
            return node.value
        raise ExpressionError(f"{node.value!r} is not a number")
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if node.id in FUNCTIONS:
            raise ExpressionError(f"`{node.id}` is a function; call it, "
                                  f"`{node.id}(...)`")
        known = ", ".join(sorted(names)) or "none"
        raise ExpressionError(f"no column `{node.id}`; the table has: {known}")
    if isinstance(node, ast.BinOp):
        op = _BINARY.get(type(node.op))
        if op is None:
            raise ExpressionError(f"operator not allowed{_where(node, expression)}")
        left = _evaluate_node(node.left, names, expression)
        right = _evaluate_node(node.right, names, expression)
        try:
            return op(left, right)
        except TypeError as error:
            # `&` on a float column is the one that catches people out: a
            # field this plugin wrote is 0/1 floats like every other column
            # (see `as_column`), so a flag has to be compared before it can be
            # combined.  Coercing it silently would turn `x_nm & 3` -- a real
            # mistake -- into a plausible wrong answer, so it says so instead.
            if isinstance(node.op, _BITWISE):
                raise ExpressionError(
                    "`&`, `|` and `^` work on comparisons and whole numbers, "
                    "not on measured values; a flag column is 0/1, so compare "
                    "it first: `(flag > 0) & (other > 0)`"
                    + _where(node, expression)) from None
            raise ExpressionError(f"{error}{_where(node, expression)}") from None
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise ExpressionError(
                "`not` asks a whole column whether it is true; write `~`"
                + _where(node, expression))
        return op(_evaluate_node(node.operand, names, expression))
    if isinstance(node, ast.Compare):
        op = _COMPARE.get(type(node.ops[0]))
        if op is None:
            raise ExpressionError(f"comparison not allowed{_where(node, expression)}")
        return op(_evaluate_node(node.left, names, expression),
                  _evaluate_node(node.comparators[0], names, expression))
    if isinstance(node, ast.IfExp):     # `a if cond else b`, elementwise
        return np.where(_evaluate_node(node.test, names, expression),
                        _evaluate_node(node.body, names, expression),
                        _evaluate_node(node.orelse, names, expression))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            piece = ast.get_source_segment(expression, node.func) or "that"
            raise ExpressionError(
                f"`{piece}` is not one of the functions an expression may "
                f"call: {', '.join(sorted(FUNCTIONS))}")
        if node.keywords:
            raise ExpressionError(f"`{node.func.id}` takes its arguments in "
                                  "order, without names")
        args = [_evaluate_node(a, names, expression) for a in node.args]
        try:
            return FUNCTIONS[node.func.id](*args)
        except TypeError as error:
            raise ExpressionError(f"{node.func.id}: {error}") from None
    # Everything else -- an attribute, a subscript, a lambda, a comprehension,
    # a walrus -- is refused rather than supported.  None of them means
    # anything as arithmetic over columns, and each is a way into the
    # interpreter for an expression that arrived in someone else's file.
    piece = ast.get_source_segment(expression, node) or type(node).__name__
    raise ExpressionError(f"`{piece}` is not allowed in an expression")


def evaluate(locs: Localizations, expression: str,
             extra: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """The expression over the table's columns, as a column.

    A result that is one number -- ``median(z_nm)``, or a constant -- is
    broadcast, so ``offset = 0`` defines a field rather than failing.
    """
    names: Dict[str, Any] = dict(locs.columns)
    names.update(extra or {})
    values = _evaluate_node(parse(expression), names, expression.strip())
    return as_column(values, len(locs), expression)


def as_column(values: Any, n: int, expression: str = "") -> np.ndarray:
    """A result as a column of ``n``: broadcast a scalar, refuse a wrong length.

    Booleans are kept as 0/1 floats rather than as a bool column.  A flag is
    the most common thing to compute here and the rest of the program treats
    columns as numbers -- the filter takes a range of them, the renderer
    colours by them, grouping reduces them -- so a bool column would be a
    column that works everywhere except where it is used.
    """
    array = np.asarray(values)
    if array.dtype == bool:
        array = array.astype(np.float32)
    elif array.dtype.kind not in "iuf":
        raise ExpressionError(f"the result is {array.dtype}, not a number"
                              + (f": `{expression}`" if expression else ""))
    if array.ndim == 0:
        array = np.full(n, array, dtype=array.dtype)
    elif array.shape != (n,):
        raise ExpressionError(
            f"the result has {array.shape[0]} values, the table has {n}"
            + (f": `{expression}`" if expression else ""))
    # float32 like the rest of the table; an integer result keeps its width,
    # since a frame number or a label is exact and float32 stops being so at
    # 2**24
    return array if array.dtype.kind in "iu" else array.astype(np.float32)


# --------------------------------------------------------------- the recipes

def recipes(locs: Localizations) -> List[Dict[str, str]]:
    """The derived fields of this table, in the order they were defined.

    A recipe is how a field survives being grouped.  Usually that is an
    expression to evaluate again on the blinks, but a field that was
    *measured* per localization and only needs a rule -- a flag written by
    `plugins.remove_locs`, say -- is a recipe with a rule and no expression:
    it cannot be recomputed and must not be averaged like a measurement.

    Read tolerantly: the recipes come out of a file that another version wrote,
    so an entry that is not one is dropped rather than raising, and one that
    does not say what to do on the grouped table gets the default.
    """
    found = locs.metadata.get(RECIPES) or []
    out = []
    for recipe in found:
        if not isinstance(recipe, dict):
            continue
        if not recipe.get("field"):
            continue
        if not recipe.get("expression") and recipe.get("grouped") not in COMBINE_RULES:
            continue          # nothing to recompute and no rule: not a recipe
        entry = dict(recipe)
        if entry.get("grouped") not in GROUPED_CHOICES:
            entry["grouped"] = RECOMPUTE
        out.append(entry)
    return out


def derived_names(locs: Localizations) -> Set[str]:
    """The fields that are defined by an expression rather than measured."""
    return {r["field"] for r in recipes(locs)}


def remember(locs: Localizations, field: str, expression: str,
             grouped: str = RECOMPUTE, where: Optional[str] = None) -> None:
    """Record how ``field`` is computed, replacing an earlier recipe for it.

    A redefinition keeps its place in the order rather than moving to the end:
    a field defined from it would otherwise be computed from the old value
    once and the new one after a reload.
    """
    found = recipes(locs)
    entry = {"field": field, "expression": expression.strip(),
             "grouped": grouped if grouped in GROUPED_CHOICES else RECOMPUTE}
    if where:
        entry["where"] = where
    for i, recipe in enumerate(found):
        if recipe["field"] == field:
            found[i] = entry
            break
    else:
        found.append(entry)
    locs.metadata[RECIPES] = found


def forget(locs: Localizations, field: str) -> None:
    """Drop the recipe for ``field``; the column that is already there stays."""
    locs.metadata[RECIPES] = [r for r in recipes(locs) if r["field"] != field]


def apply_recipes(locs: Localizations, only: Optional[Sequence[str]] = None,
                  report: Optional[Callable[[str], None]] = None) -> List[str]:
    """Recompute the derived fields of a table.  Returns the ones that ran.

    Called wherever a table is built from another one -- grouping, above all.
    A recipe that cannot run (its column is not in *this* table: ``n_in_group``
    before anything has been linked) is reported and skipped, never raised:
    a field that cannot be recomputed is one missing column, and the table it
    belongs to is still the table the user asked for.
    """
    done = []
    for recipe in recipes(locs):
        field, expression = recipe["field"], recipe.get("expression", "")
        if only is not None and field not in only:
            continue
        if not expression:
            continue          # a rule, not an expression: nothing to run here
        if recipe.get("where") == "selection":
            # computed for part of the table only, so the expression alone
            # does not say what the field is; the values that are there stand
            continue
        try:
            locs.columns[field] = evaluate(locs, expression)
            done.append(field)
        except ExpressionError as error:
            if report:
                report(f"{field} = {expression}: {error}")
    return done
