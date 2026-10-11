"""The results of one evaluation: a row per region, kept in columns.

`docs/regions.md` ("What a region keeps", "The region table is a table") has
the decisions.  What this module is for: an evaluation used to be kept as one
JSON record per region and step, about 1.3 kB a region for one step -- 2.5 MB
for 2000 ROIs, and 130 MB for the 10^5 clusters a clusterer finds.  Here the
same results are a column per number, so they cost what the numbers do, and
the file holds them as HDF5 datasets rather than as text.

A table is one evaluation, by its name: the rows are the regions it measured,
each with the values its step returned, the arrays it kept, the step that made
it (plugin, version, parameters -- a row re-measured with other settings says
so), the signature its freshness is checked by, and the error that took its
place when the step failed on that region.  A row is set whole; a value a row
did not return is missing (NaN in the column), not zero.

Arrays are a list per row in memory -- a row's own length or a fixed one --
and are written as one array of all their values end to end and the offsets
where each row's begin, so a hundred thousand profiles are two datasets.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# the kind of each value column, for writing it as one dtype and reading the
# values back as what they were
KINDS = ("float", "int", "bool", "str")


def _kind(values: Iterable[Any]) -> str:
    seen = [v for v in values if v is not None]
    if seen and all(isinstance(v, str) for v in seen):
        return "str"
    if seen and all(isinstance(v, bool) for v in seen):
        return "bool"
    if seen and all(isinstance(v, int) and not isinstance(v, bool) for v in seen):
        return "int"
    return "float"


class RegionTable:
    """One evaluation's results, a row per region (see the module docstring)."""

    def __init__(self, label: str = ""):
        self.label = label
        self.ids: List[str] = []
        self._index: Dict[str, int] = {}
        self.steps: List[Dict[str, Any]] = []   # the step records rows refer to
        self.step_of: List[int] = []             # per row: index into `steps`, -1 none
        self.signatures: List[str] = []          # per row: data and step, "" none
        self.data: List[str] = []                # per row: the data alone, "" none
        self.errors: List[str] = []              # per row: "" or why it failed
        self.values: Dict[str, List[Any]] = {}   # per column: a value per row or None
        self.arrays: Dict[str, List[Optional[np.ndarray]]] = {}
        # per row: what the step kept to draw its figure again (plain data),
        # or None (`Plugin.keep`)
        self.kept: List[Optional[Dict[str, Any]]] = []

    # ---------------------------------------------------------------- rows
    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, region_id) -> bool:
        return str(region_id) in self._index

    def _row(self, region_id: str) -> int:
        """The row of this region, made (empty) when it has none."""
        i = self._index.get(region_id)
        if i is not None:
            return i
        i = len(self.ids)
        self.ids.append(region_id)
        self._index[region_id] = i
        self.step_of.append(-1)
        self.signatures.append("")
        self.data.append("")
        self.errors.append("")
        self.kept.append(None)
        for column in self.values.values():
            column.append(None)
        for column in self.arrays.values():
            column.append(None)
        return i

    def set(self, region_id, entry: Dict[str, Any], step: Optional[Dict[str, Any]] = None,
            data: str = "") -> None:
        """This region's row, whole: ``entry`` is what a run stored
        (`roi_manager.core.region_entry`: ``values``, ``arrays``, and a
        ``signature``, or an ``error``), ``step`` the record of the step that
        made it, ``data`` the signature of the data alone."""
        i = self._row(str(region_id))
        if step is not None:
            try:
                k = self.steps.index(step)
            except ValueError:
                k = len(self.steps)
                self.steps.append(dict(step))
            self.step_of[i] = k
        else:
            self.step_of[i] = -1
        self.signatures[i] = str(entry.get("signature") or "")
        self.data[i] = str(data or "")
        self.errors[i] = str(entry.get("error") or "")
        self.kept[i] = entry.get("kept") or None
        values = entry.get("values") or {}
        for name in values:
            if name not in self.values:
                self.values[name] = [None] * len(self.ids)
        for name, column in self.values.items():
            column[i] = values.get(name)
        arrays = entry.get("arrays") or {}
        for name in arrays:
            if name not in self.arrays:
                self.arrays[name] = [None] * len(self.ids)
        for name, column in self.arrays.items():
            a = arrays.get(name)
            column[i] = None if a is None else np.asarray(a)

    def entry(self, region_id) -> Optional[Dict[str, Any]]:
        """The row as a run stored it -- ``values`` (the ones it has),
        ``arrays``, ``signature``, or ``error`` -- or None when there is none."""
        i = self._index.get(str(region_id))
        if i is None:
            return None
        if self.errors[i]:
            out: Dict[str, Any] = {"error": self.errors[i]}
        else:
            out = {"values": {name: column[i] for name, column in self.values.items()
                              if column[i] is not None}}
            arrays = {name: column[i].tolist() for name, column in self.arrays.items()
                      if column[i] is not None}
            if arrays:
                out["arrays"] = arrays
            if self.kept[i] is not None:
                out["kept"] = self.kept[i]
        if self.signatures[i]:
            out["signature"] = self.signatures[i]
        return out

    def step(self, region_id) -> Dict[str, Any]:
        """The record of the step that made this row; {} when unknown."""
        i = self._index.get(str(region_id))
        if i is None or self.step_of[i] < 0:
            return {}
        return self.steps[self.step_of[i]]

    def data_signature(self, region_id) -> str:
        i = self._index.get(str(region_id))
        return "" if i is None else self.data[i]

    def keep_only(self, region_ids) -> None:
        """Drop the rows of regions not in ``region_ids`` (ROIs that are gone)."""
        keep = [i for i, r in enumerate(self.ids) if r in set(map(str, region_ids))]
        if len(keep) == len(self.ids):
            return
        self.ids = [self.ids[i] for i in keep]
        self._index = {r: n for n, r in enumerate(self.ids)}
        for name in ("step_of", "signatures", "data", "errors", "kept"):
            setattr(self, name, [getattr(self, name)[i] for i in keep])
        self.values = {k: [c[i] for i in keep] for k, c in self.values.items()}
        self.arrays = {k: [c[i] for i in keep] for k, c in self.arrays.items()}
        used = sorted({k for k in self.step_of if k >= 0})
        remap = {old: new for new, old in enumerate(used)}
        self.steps = [self.steps[k] for k in used]
        self.step_of = [remap.get(k, -1) for k in self.step_of]

    def columns(self) -> Dict[str, np.ndarray]:
        """The value columns as arrays: numbers as floats with NaN where a
        row has none, ints as ints while none is missing, text as text."""
        out = {}
        for name, column in self.values.items():
            kind = _kind(column)
            if kind == "str":
                out[name] = np.array(["" if v is None else v for v in column], dtype=object)
            elif kind in ("int", "bool") and all(v is not None for v in column):
                out[name] = np.array(column, dtype=np.int64 if kind == "int" else bool)
            else:
                out[name] = np.array([np.nan if v is None else float(v) for v in column],
                                     dtype=float)
        return out

    # ------------------------------------------------------------- as data
    def to_dict(self, region_ids=None) -> Dict[str, Any]:
        """Columnar plain data: what travels inside the ROI project's JSON
        (a chain's scratch copy, a handoff).  ``region_ids`` keeps only those
        rows, without changing the table."""
        if region_ids is None:
            rows = range(len(self.ids))
        else:
            wanted = set(map(str, region_ids))
            rows = [i for i, r in enumerate(self.ids) if r in wanted]
        return {"label": self.label, "ids": [self.ids[i] for i in rows],
                "steps": self.steps,
                "step_of": [self.step_of[i] for i in rows],
                "signatures": [self.signatures[i] for i in rows],
                "data": [self.data[i] for i in rows],
                "errors": [self.errors[i] for i in rows],
                "kept": [self.kept[i] for i in rows],
                "values": {k: [_plain(c[i]) for i in rows] for k, c in self.values.items()},
                "arrays": {k: [None if c[i] is None else c[i].tolist() for i in rows]
                           for k, c in self.arrays.items()}}

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "RegionTable":
        table = cls(str(doc.get("label") or ""))
        table.ids = [str(i) for i in doc.get("ids") or []]
        table._index = {r: n for n, r in enumerate(table.ids)}
        n = len(table.ids)

        def column(name, default):
            got = list(doc.get(name) or [])
            return got if len(got) == n else [default] * n
        table.steps = [dict(s) for s in doc.get("steps") or []]
        table.step_of = [int(k) for k in column("step_of", -1)]
        table.signatures = [str(s or "") for s in column("signatures", "")]
        table.data = [str(s or "") for s in column("data", "")]
        table.errors = [str(s or "") for s in column("errors", "")]
        table.kept = [k or None for k in column("kept", None)]
        table.values = {k: list(c) for k, c in (doc.get("values") or {}).items()
                        if len(c) == n}
        table.arrays = {k: [None if a is None else np.asarray(a) for a in c]
                        for k, c in (doc.get("arrays") or {}).items() if len(c) == n}
        return table

    # ------------------------------------------------------------- as HDF5
    def to_hdf5(self, group) -> None:
        """Write into an (empty) h5py group: a dataset per column."""
        import h5py
        text = h5py.string_dtype("utf-8")
        group.attrs["label"] = self.label
        group.attrs["steps"] = json.dumps(self.steps, default=_plain)
        # a variable-length string costs ~40 bytes on disk, so ids that are
        # numbers (a cluster's) are written as numbers, and a column of text
        # that is empty throughout (no errors, unsigned rows) is not written
        if all(r.isdigit() for r in self.ids):
            group.create_dataset("ids", data=np.array([int(r) for r in self.ids],
                                                      dtype=np.int64))
        else:
            group.create_dataset("ids", data=np.array(self.ids, dtype=object), dtype=text)
        group.create_dataset("step_of", data=np.asarray(self.step_of, dtype=np.int32))
        for name in ("signatures", "data", "errors"):
            column = getattr(self, name)
            if any(column):
                group.create_dataset(name, data=np.array(column, dtype=object),
                                     dtype=text)
        if any(k is not None for k in self.kept):
            group.create_dataset("kept", data=np.array(
                ["" if k is None else json.dumps(k, default=_plain) for k in self.kept],
                dtype=object), dtype=text)
        values = group.create_group("values")
        # which rows have no value, where any has none: a dataset, since an
        # attribute is bounded by the object header (~64 kB)
        missing = group.create_group("missing")
        for name, column in self.values.items():
            kind = _kind(column)
            if kind == "str":
                ds = values.create_dataset(
                    name, data=np.array(["" if v is None else v for v in column],
                                        dtype=object), dtype=text)
            else:
                ds = values.create_dataset(
                    name, data=np.array([np.nan if v is None else float(v)
                                         for v in column], dtype=float))
            ds.attrs["kind"] = kind
            gone = np.array([v is None for v in column], dtype=bool)
            if gone.any():
                missing.create_dataset(name, data=gone)
        arrays = group.create_group("arrays")
        for name, column in self.arrays.items():
            flat = [np.asarray(a, dtype=float).ravel() if a is not None else np.zeros(0)
                    for a in column]
            offsets = np.concatenate([[0], np.cumsum([len(f) for f in flat])]).astype(np.int64)
            sub = arrays.create_group(name)
            sub.create_dataset("values", data=np.concatenate(flat) if flat else np.zeros(0))
            sub.create_dataset("offsets", data=offsets)
            sub.create_dataset("present", data=np.array([a is not None for a in column]))

    @classmethod
    def from_hdf5(cls, group) -> "RegionTable":
        table = cls(str(group.attrs.get("label", "")))

        def strings(name):
            if name not in group:
                return []
            ds = group[name]
            if ds.dtype.kind in "iu":
                return [str(int(v)) for v in ds[()]]
            return [str(s) for s in ds.asstr()[()]]
        table.ids = strings("ids")
        table._index = {r: n for n, r in enumerate(table.ids)}
        n = len(table.ids)
        table.steps = json.loads(group.attrs.get("steps", "[]"))
        table.step_of = [int(k) for k in group["step_of"][()]] if "step_of" in group \
            else [-1] * n
        for name in ("signatures", "data", "errors"):
            got = strings(name)
            setattr(table, name, got if len(got) == n else [""] * n)
        kept = strings("kept")
        table.kept = ([json.loads(k) if k else None for k in kept] if len(kept) == n
                      else [None] * n)
        for name, ds in group.get("values", {}).items():
            kind = str(ds.attrs.get("kind", "float"))
            missing = (group["missing"][name][()] if "missing" in group
                       and name in group["missing"] else np.zeros(n, bool))
            raw = ds.asstr()[()] if kind == "str" else ds[()]
            cast = {"int": int, "bool": bool, "str": str}.get(kind, float)
            table.values[name] = [None if gone else cast(v)
                                  for v, gone in zip(raw, missing)]
        for name, sub in group.get("arrays", {}).items():
            values, offsets = sub["values"][()], sub["offsets"][()]
            present = sub["present"][()]
            table.arrays[name] = [values[offsets[i]:offsets[i + 1]].copy() if present[i]
                                  else None for i in range(n)]
        return table


def _plain(value):
    """A JSON value: numpy scalars as Python ones."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
