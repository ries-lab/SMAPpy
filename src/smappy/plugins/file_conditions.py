"""*File conditions*: what differs between the files opened together -- a
laser power, a buffer, a sample -- written onto their localizations as
attributes, so that filters, colours and *Compare groups* can tell the
conditions apart.  The MFX scripts made such "independent variables" from
the file names by hand; here a pattern does it, or a line per file.

A condition that is a number becomes a numeric attribute; one that is a
word becomes a whole-number code, the words kept in the table's metadata
(``metadata["conditions"]``) so the summaries can name the groups.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import Context, FormState, Notice, Plugin, Result, param, register

#: where the words behind a coded condition are kept
CONDITIONS = "conditions"


@dataclass
class FileConditionsSettings:
    pattern: str = param("", label="from the names",
                         help="a regular expression with named groups, e.g. "
                              "(?P<power>\\d+)uW: each group a condition, from each "
                              "file's name")
    assign: str = param("", label="typed",
                        help="lines 'part of a file name: key=value, key=value'; "
                             "after the pattern, and over it")


def conditions_of(names: List[str], pattern: str = "", assign: str = ""
                  ) -> Dict[str, List[Optional[str]]]:
    """``{condition: [value per file]}`` from the file ``names``: the named
    groups of ``pattern`` searched in each name, then each line of
    ``assign`` -- ``part of a name: key=value, ...`` -- for the files whose
    name holds that part.  None where a file has no value."""
    out: Dict[str, List[Optional[str]]] = {}
    if pattern.strip():
        rx = re.compile(pattern)
        for i, name in enumerate(names):
            m = rx.search(name)
            for key in rx.groupindex:
                out.setdefault(key, [None] * len(names))[i] = (m.group(key) if m else None)
    for line in assign.splitlines():
        if ":" not in line:
            continue
        part, _, pairs = line.rpartition(":")
        part = part.strip()
        for pair in pairs.split(","):
            if "=" not in pair:
                continue
            key, _, value = pair.partition("=")
            key, value = key.strip(), value.strip()
            if not key.isidentifier():
                raise ValueError(f"{key!r} is not usable as an attribute name")
            column = out.setdefault(key, [None] * len(names))
            for i, name in enumerate(names):
                if part and part in name:
                    column[i] = value
    return out


def as_column(values: List[Optional[str]]):
    """A condition's values per file as numbers -- or, where one is a word,
    as codes 0, 1, ... with the words in their order of first appearance;
    -1 (or NaN) where a file has none.  ``(per_file, words or None)``."""
    def number(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    nums = [number(v) for v in values]
    if all(n is not None or v is None for n, v in zip(nums, values)):
        return np.array([np.nan if n is None else n for n in nums]), None
    words: List[str] = []
    for v in values:
        if v is not None and v not in words:
            words.append(v)
    return np.array([words.index(v) if v is not None else -1 for v in values]), words


@register("Analysis/Process/File conditions")
class FileConditions(Plugin):
    """Writes what differs between the open files -- from their names by a
    pattern, or typed per file -- onto their localizations, one attribute
    per condition."""

    Settings = FileConditionsSettings
    changes_locs = True
    grouping = "ungrouped"
    version = "1"

    def describe(self, ctx: Optional[Context], settings: FileConditionsSettings) -> FormState:
        state = FormState()
        if ctx is not None and ctx.session is not None and ctx.session.files:
            names = [f.name for f in ctx.session.files]
            try:
                found = conditions_of(names, settings.pattern, settings.assign)
            except (re.error, ValueError) as e:
                state.notice = Notice(f"not understood: {e}")
                return state
            if found:
                state.notice = Notice("; ".join(f"{k}: " + ", ".join(str(v) for v in vals)
                                                for k, vals in found.items()), kind="info")
        return state

    def run(self, ctx: Context, settings: FileConditionsSettings) -> Result:
        from ..locs import Localizations
        locs = ctx.locs
        if ctx.session is not None and ctx.session.files:
            names = [f.name for f in ctx.session.files]
        else:
            names = [str(locs.metadata.get("source", "") or "file")]
        try:
            found = conditions_of(names, settings.pattern, settings.assign)
        except re.error as e:
            raise ValueError(f"the pattern is not a regular expression: {e}")
        if not found:
            raise ValueError("no conditions: give a pattern with named groups, e.g. "
                             "(?P<power>\\d+)uW, or lines 'name: key=value'")
        files = (np.asarray(locs["filenumber"], np.int64) if "filenumber" in locs
                 else np.zeros(len(locs), np.int64))
        columns = dict(locs.columns)
        metadata = dict(locs.metadata)
        kept = dict(metadata.get(CONDITIONS) or {})
        lines = []
        for key, values in found.items():
            per_file, words = as_column(values)
            idx = np.clip(files, 0, len(per_file) - 1)
            columns[key] = (per_file[idx].astype(np.int32) if words is not None
                            else per_file[idx].astype(np.float32))
            if words is not None:
                kept[key] = words
            lines.append(f"{key}: " + ", ".join(f"{n} = {v}" for n, v in zip(names, values)))
        metadata[CONDITIONS] = kept
        return Result(locs=Localizations(columns, metadata), text="\n".join(lines),
                      settings=settings)
