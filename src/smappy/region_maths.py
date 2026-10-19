"""Expressions over what was measured per region: *Math parser* on a region source.

`docs/region-filter.md` ("Maths on region values").  The localizations'
expressions (`mathparse`) are evaluated here over a source's regions -- the
ROIs, or the regions of an id column -- with one row per region instead of
one per localization: ``density = corners.n_locs / (pi * stats.rg_nm ** 2)``.

**Names.**  Every evaluation of the source is reached by an **alias**, its
label made a word (`alias`: *NPC Corners* is ``npc_corners``, *Id
statistics* ``id_statistics``), and a value as ``alias.value``; a value only
one evaluation has can be written bare.  The ROIs' annotations are
``annotations.use`` and so on, a flag as 0/1; a keyword (text) is no
number and is left out.  Evaluations meet by region id: a region one of
them did not measure has NaN there.  A reduction reduces over the regions --
``n_locs / median(n_locs)``.

**Where the result goes.**  Into the source's ``maths`` table, a region
table like any other, so the summaries, the region filter and *Regions to
column* read it; the expression is kept with it as a **recipe**
(`RegionTable.recipes`), and the table is computed again from all its
recipes, in order, whenever an evaluation it reads is measured again
(`refresh`) -- a later recipe may read an earlier one, ``maths.density``.
A region whose inputs are out of date gets no current row: the table is
signed only where they are current, as a block's would be.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chain import BLOCK_ROIS as ROIS

#: the evaluation the results go into, per source
MATHS = "maths"
#: the plugin whose results these are (`Step.as_record`)
PLUGIN = "Analysis/Process/Math Parser"
VERSION = "1"

_DOTTED = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\.([A-Za-z_]\w*)")
_SEP = "__"


def alias(label: str) -> str:
    """An evaluation's name as a word an expression can use."""
    word = re.sub(r"\W+", "_", str(label).strip().lower()).strip("_")
    if not word or word[0].isdigit():
        word = "e_" + word
    return word


def aliases(session, source: str) -> Dict[str, str]:
    """``{alias: label}`` of every evaluation of ``source``, a second of the
    same word numbered (``corners_2``)."""
    from .region_filter import sources
    out: Dict[str, str] = {}
    for label in (sources(session).get(source) or {}):
        word, k = alias(label), 2
        while word in out:
            word, k = f"{alias(label)}_{k}", k + 1
        out[word] = label
    return out


def _numbers(column) -> Optional[np.ndarray]:
    """A column of a region table as floats, flags as 0/1; None for text."""
    present = [v for v in column if v is not None]
    if any(isinstance(v, str) for v in present):
        return None
    return np.array([np.nan if v is None else float(v) for v in column], dtype=float)


def region_names(session, source: str, skip: Tuple[str, ...] = ()
                 ) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, str]]:
    """``(region ids, {name: a value per region}, {alias: label})``: every
    number of every evaluation of ``source`` (but those in ``skip``) under
    ``alias__value``, and bare where only one evaluation has the name."""
    from .region_filter import region_values
    found = aliases(session, source)
    per: Dict[str, Tuple[List[str], Dict[str, list]]] = {
        word: region_values(session, source, label) for word, label in found.items()
        if label not in skip}
    ids: List[str] = []
    seen = set()
    for got, _ in per.values():
        for r in got:
            if r not in seen:
                seen.add(r)
                ids.append(r)
    if source != ROIS:
        ids.sort(key=lambda r: (len(r), r))         # numerically, ids are digits
    at = {r: k for k, r in enumerate(ids)}
    names: Dict[str, np.ndarray] = {}
    owners: Dict[str, List[str]] = {}
    for word, (got, values) in per.items():
        rows = np.array([at[r] for r in got], dtype=np.int64)
        for name, column in values.items():
            numbers = _numbers(column)
            if numbers is None:
                continue
            full = np.full(len(ids), np.nan)
            full[rows] = numbers
            names[word + _SEP + name] = full
            owners.setdefault(name, []).append(word + _SEP + name)
    for name, keys in owners.items():
        if len(keys) == 1 and name not in names:
            names[name] = names[keys[0]]
    return ids, names, found


def _plain(expression: str) -> str:
    """``alias.value`` written as the one word the parser reads."""
    return _DOTTED.sub(lambda m: m.group(1) + _SEP + m.group(2), expression)


def _dotted(text: str) -> str:
    return str(text).replace(_SEP, ".")


def evaluate(session, source: str, expression: str, extra: Optional[Dict] = None,
             skip: Tuple[str, ...] = ()) -> Tuple[List[str], np.ndarray]:
    """``(region ids, a value per region)`` of ``expression`` over the
    regions of ``source``.  Raises `mathparse.ExpressionError`, naming the
    values there are, for a name none has."""
    from .locs import Localizations
    from .mathparse import ExpressionError, evaluate as evaluate_locs, names_in
    ids, names, _ = region_names(session, source, skip)
    names.update(extra or {})
    text = _plain(expression)
    missing = sorted(names_in(text) - set(names))
    if missing:
        offered = sorted(_dotted(n) for n in names)
        raise ExpressionError(
            f"no value {', '.join(_dotted(m) for m in missing)} on the regions of "
            f"{'the ROIs' if source == ROIS else source}; there is: "
            + (", ".join(offered) or "nothing measured yet"))
    if not ids:
        raise ExpressionError("no region has been measured yet")
    table = Localizations(names, {})
    try:
        values = evaluate_locs(table, text)
    except ExpressionError as error:
        raise ExpressionError(_dotted(error)) from None
    return ids, np.asarray(values, dtype=float)


# ------------------------------------------------------------ the table

def recipes(session, source: str) -> Dict[str, str]:
    """``{field: expression}`` of the source's maths table, in order."""
    table = _table(session, source)
    return dict(table.recipes) if table is not None else {}


def _table(session, source: str):
    if source == ROIS:
        return session.rois.tables.get(MATHS)
    return (session.id_regions.get(source) or {}).get(MATHS)


def compute(session, source: str, found: Dict[str, str]):
    """The maths table of ``source`` from the recipes ``found``, in order,
    each able to read those before it as ``maths.<field>``; signed per region
    where every evaluation it reads is current (see the module docstring)."""
    values: Dict[str, np.ndarray] = {}
    ids: List[str] = []
    for field, expression in found.items():
        extra = {MATHS + _SEP + f: v for f, v in values.items()}
        extra.update({f: v for f, v in values.items()})
        ids, column = evaluate(session, source, expression, extra=extra, skip=(MATHS,))
        values[field] = column
    step = {"label": MATHS, "plugin": PLUGIN, "version": VERSION,
            "parameters": {"recipes": dict(found)}}
    if source != ROIS:
        from .region_runs import column_table
        current = _inputs_current(session, source)
        table = column_table(session.locs, source, MATHS, ids, values, step,
                             session=session)
        if not current:                # out of date with what it was computed from
            table.data = [""] * len(table.ids)
            table.signatures = [""] * len(table.ids)
    else:
        table = _roi_table(session, ids, values, step)
    table.recipes = dict(found)
    return table


def _inputs_current(session, source: str) -> bool:
    from .region_results import Evaluation, is_current
    ctx = session.context()
    return all(is_current(ctx, Evaluation(source, label))
               for label in (session.id_regions.get(source) or {}) if label != MATHS)


def _roi_table(session, ids, values, step):
    """A row per ROI, signed as the ROI project signs a step's row where all
    the ROI's evaluations are current, unsigned where one is not."""
    import json
    from .region_table import RegionTable
    from .roi_manager.core import digest, json_text
    project = session.rois
    table = RegionTable(MATHS)
    identity = {k: v for k, v in step.items() if k != "label"}
    for k, roi_id in enumerate(ids):
        roi = project.rois.get(roi_id)
        if roi is None:
            continue
        entry = {"values": {f: float(v[k]) for f, v in values.items()}}
        stored = [label for label, t in project.tables.items()
                  if label != MATHS and roi_id in t]
        current = project.evaluations(roi_id)
        if all(label in current for label in stored):
            inputs = json.loads(json_text(project.inputs(roi)))
            entry["signature"] = digest({"inputs": inputs, "step": identity})
        table.set(roi_id, entry, step)
    return table


def store(session, source: str, table) -> None:
    """Put ``table`` where ``source``'s results live."""
    if source == ROIS:
        project = session.rois
        project.tables.pop(MATHS, None)
        project.tables[MATHS] = table
    else:
        session.id_regions.setdefault(source, {})[MATHS] = table


def _key(session, source: str):
    """What a source's maths table depends on: its other tables, as they are."""
    if source == ROIS:
        project = session.rois
        return tuple((label, id(t), tuple(t.signatures)) for label, t in
                     project.tables.items() if label != MATHS) + \
            (tuple(project.rois), id(session.locs))
    tables = session.id_regions.get(source) or {}
    return tuple((label, id(t), len(t), tuple(t.data[:1])) for label, t in
                 tables.items() if label != MATHS) + (id(session.locs),)


def refresh(session) -> None:
    """Compute every maths table again whose inputs moved: what keeps a
    recipe a recipe (`Session.changed`)."""
    from .mathparse import ExpressionError
    cache = session.__dict__.setdefault("_region_maths_keys", {})
    sources = []
    if session._rois is not None and MATHS in session.rois.tables:
        sources.append(ROIS)
    if session._id_regions is not None:
        sources += [s for s, tables in session._id_regions.items() if MATHS in tables]
    for source in sources:
        found = recipes(session, source)
        if not found:
            continue
        key = _key(session, source)
        if cache.get(source) == key:
            continue
        try:
            store(session, source, compute(session, source, found))
        except ExpressionError:
            pass        # a value its recipe reads is gone: the old numbers stand, stale
        cache[source] = _key(session, source)
