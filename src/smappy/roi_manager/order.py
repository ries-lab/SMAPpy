"""Sorting and filtering ROIs by what was measured in them and what they were
called: SMAP's SortROIs and FilterSites, on the columns of the results table.

SMAP walks a site's struct for a field (``evaluation.NPCcorners.n``,
``annotation.list3.value``) and `eval`s what it is given.  Here the names are
the results table's own -- a measured column, or a keyword list -- so what a
sort or a filter reads is what a summary reads, and a filter is parsed, never
evaluated as Python.

**Sorting** orders the ROIs, nothing else: an ROI keeps its number, its
results and its *use*.  The order is the project's (`ROIProject.order`), and
the ROI list, the results table and whatever reads that table follow it.  A
keyword sorts by its place in its list -- *good, bad, unsure* in that order
-- since the list is how its writer ranked them; an ROI with no value goes
last whichever way, and ties fall to the next key, then to the ROI number.
An ROI whose result is out of date sorts by what is stored: the list would
otherwise reshuffle under the user while it is being measured again.

**Filtering** sets *use*.  Keywords are compared as text (``quality ==
"good"``), not by their place: a list reordered after the filter was written
must not change which ROIs it keeps -- the trap in SMAP's ``list3.value==1``.
A name an ROI has no value for makes its comparisons false, so an unmeasured
ROI is left out by ``n_corners >= 6`` and kept by ``not n_corners >= 6``.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

MISSING = None


def sort_ids(rows: Dict[str, Dict[str, Any]], keys: Sequence[Tuple[str, bool]],
             lists: Optional[Dict[str, List[str]]] = None,
             numbers: Optional[Dict[str, int]] = None) -> List[str]:
    """The ids of ``rows`` (``{roi id: {column: value}}``) sorted by ``keys``,
    ``[(column, descending)]``.  ``lists`` are the keyword lists: a column of
    one sorts by the keyword's place in it, a keyword it lacks after those
    it has, by its text.  ``numbers`` breaks the last tie."""
    lists = lists or {}
    numbers = numbers or {}

    def rank(column: str, value) -> Optional[tuple]:
        if value is None:
            return None
        if isinstance(value, str):
            words = lists.get(column) or lists.get(column.partition(".")[2]) or []
            return (0, words.index(value), "") if value in words else (1, 0, value)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(number) else (0, number, "")

    ids = sorted(rows, key=lambda i: numbers.get(i, 0))
    # one stable sort per key, the last key first: the first key then decides
    # and the later ones break its ties
    for column, descending in reversed(list(keys)):
        if not column:
            continue
        ranked = {i: rank(column, rows[i].get(column)) for i in ids}
        have = [i for i in ids if ranked[i] is not None]
        lack = [i for i in ids if ranked[i] is None]
        have.sort(key=lambda i: ranked[i], reverse=descending)
        ids = have + lack                         # no value: last, either way
    return ids


_COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
            ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def condition(text: str) -> Tuple[Callable[[Dict[str, Any]], bool], List[str]]:
    """``n_corners >= 6 and quality == "good"`` as a test of a row, and the
    names it reads.  Comparisons (chained too), ``and``, ``or``, ``not``,
    arithmetic, numbers, quoted text and ``in [...]``; a column name with
    spaces in backquotes.  Nothing else parses,
    so nothing in it is run.  Raises ValueError, saying what it cannot read."""
    # a column whose name is no Python name -- "NPC Corners.n_corners", when
    # two steps write n_corners -- is written in backquotes
    quoted: Dict[str, str] = {}

    def stand_in(match) -> str:
        quoted[f"_q{len(quoted)}"] = match.group(1)
        return f"_q{len(quoted) - 1}"

    text = re.sub(r"`([^`]+)`", stand_in, text)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"cannot read the condition: {error.msg}") from None
    names: List[str] = []

    def check(node) -> None:
        allowed = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
                   ast.USub, ast.UAdd, ast.Compare, ast.BinOp, ast.Name, ast.Load,
                   ast.Constant, ast.List, ast.Tuple, ast.In, ast.NotIn,
                   ast.Attribute, *_COMPARE, *_ARITHMETIC)
        if not isinstance(node, allowed):
            raise ValueError(f"the condition cannot contain {type(node).__name__}: "
                             "compare columns with numbers or quoted keywords")
        if isinstance(node, (ast.Name, ast.Attribute)):
            names.append(quoted.get(_name(node), _name(node)))
            return
        for child in ast.iter_child_nodes(node):
            check(child)

    check(tree)

    def value(node, row):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Name, ast.Attribute)):
            found = row.get(quoted.get(_name(node), _name(node)))
            if isinstance(found, float) and math.isnan(found):
                return MISSING
            return found
        if isinstance(node, (ast.List, ast.Tuple)):
            return [value(e, row) for e in node.elts]
        if isinstance(node, ast.BoolOp):
            parts = (truth(v, row) for v in node.values)
            return all(parts) if isinstance(node.op, ast.And) else any(parts)
        if isinstance(node, ast.UnaryOp):
            inner = value(node.operand, row)
            if isinstance(node.op, ast.Not):
                return not truth(node.operand, row)
            if inner is MISSING:
                return MISSING
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            a, b = value(node.left, row), value(node.right, row)
            if a is MISSING or b is MISSING:
                return MISSING
            try:
                return _ARITHMETIC[type(node.op)](a, b)
            except (TypeError, ZeroDivisionError):
                return MISSING
        if isinstance(node, ast.Compare):
            left = value(node.left, row)
            for op, right_node in zip(node.ops, node.comparators):
                right = value(right_node, row)
                if left is MISSING or right is MISSING:
                    return False
                try:
                    if isinstance(op, (ast.In, ast.NotIn)):
                        ok = (left in right) == isinstance(op, ast.In)
                    else:
                        ok = _COMPARE[type(op)](left, right)
                except TypeError:          # text against a number: not so
                    return False
                if not ok:
                    return False
                left = right
            return True
        raise ValueError(type(node).__name__)

    def truth(node, row) -> bool:
        found = value(node, row)
        return bool(found) if found is not MISSING else False

    return (lambda row: truth(tree.body, row)), names


def _name(node) -> str:
    """``a`` or ``a.b``: a column qualified with its step's label."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    raise ValueError(type(node).__name__)
