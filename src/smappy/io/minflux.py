"""Abberior MINFLUX records into a localization table, from any source.

Imspector has written two record layouts, and every source -- the `.npy`,
`.json` and `.zip` exports, and the zarr array inside an `.msr` -- holds one
of them:

* **m2205, nested**: a row per localization attempt, with ``itr[k]`` a
  sub-record per iteration (``loc``, ``lnc``, ``eco``, ``efo``, ``cfr``, ...)
  and per row ``tim``, ``tid``, ``vld``, ``gri``, ``sqi``.  The localization
  is the last iteration.
* **m2410, flat**: a row per *iteration*, the fields side by side (``vld``,
  ``fnl``, ``bot``, ``eot``, ``sta``, ``tim``, ``tid``, ``gri``, ``thi``,
  ``sqi``, ``itr``, ``loc``, ``lnc``, ``eco``, ``ecc``, ``efo``, ``efc``,
  ``fbg``, ``cfr``, ``dcr[2]``).  A localization is a row with ``fnl``
  (final iteration) set.  Told from m2205 by a scalar ``itr``.

What is the same for both:

* **cfr is not measured in the last iteration.**  Imspector measures the
  centre-frequency ratio in one iteration only -- the one before the last in
  every file looked at -- and writes 0 into the others; that iteration is
  found as the last one whose cfr varies over the measurement (pyMINFLUX
  does the same).  In m2205 it is a column of the nested record.  In m2410
  it is an earlier row of the same localization -- after the trace's
  previous final iteration -- joined here (`_joined`).  How often it is
  measured depends on the sequence: for every localization in the 3D bead
  sequences looked at, once at the start of a trace in the 2D ones.  A
  localization whose own iterations did not measure it has a cfr of NaN, and
  is not given its trace's: a value it was never measured with would read as
  one, and every statistic here leaves NaN out.
* ``loc`` is the position as corrected by beam monitoring (MBM) during the
  acquisition, ``lnc`` as measured.  Both are kept where they differ.
* ``dcr`` in m2410 has two entries, ch1/(ch1+ch2) and ch2/(ch1+ch2): the
  first is ``dcr``, as in m2205, and the second ``dcr2``.
* ``sta`` is why an iteration ended -- the abort criterion, coded
  differently in the two versions -- and is kept as it is.
* Positions are metres, written in nm.  z is multiplied by the *z scale*,
  which corrects for the refractive index mismatch; the default 1.0 keeps
  Imspector's numbers.

There are no camera frames: ``frame`` is the rank in time (``time_s``,
Imspector's ``tim``), the trace is ``tid``, and merging blinks merges a trace
(`group.by_trace`).  The photons are the final iteration's ``eco`` and the
precision SMAP's stand-in, 150 nm / sqrt(N): the records give none.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..locs import Localizations

MINFLUX_PSF_NM = 150.0     # for the precision proxy, SMAP's number


# ------------------------------------------------------------------ sources

def records(path: Path) -> np.ndarray:
    """The structured array Abberior's export holds: .npy, or a .zip of one/json."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            npy = [n for n in names if n.lower().endswith(".npy")]
            if npy:
                return np.load(io.BytesIO(z.read(npy[0])), allow_pickle=True)
            js = [n for n in names if n.lower().endswith(".json")]
            if js:
                return from_json(json.loads(z.read(js[0])))
        raise ValueError(f"{path.name}: no .npy or .json inside")
    if path.suffix.lower() == ".json":
        return from_json(json.loads(path.read_text()))
    raise ValueError(f"not a MINFLUX export: {path.name}")


def from_json(records: list) -> np.ndarray:
    """The json export: a list of dicts, nested as the m2205 npy, or flat
    (m2410), whose fields become the record's."""
    if records and not isinstance(records[0].get("itr"), (list, dict)):
        return _flat_from_json(records)
    n = len(records)
    itr = [r["itr"][-1] if isinstance(r["itr"], list) else r["itr"] for r in records]
    dt_itr = [("loc", "f8", (3,)), ("efo", "f8"), ("cfr", "f8"), ("dcr", "f8"),
              ("eco", "f8"), ("itr", "i4")]
    out = np.zeros(n, dtype=[("itr", dt_itr), ("tim", "f8"), ("tid", "i8"),
                             ("vld", "?")])
    for i, (r, it) in enumerate(zip(records, itr)):
        out["itr"][i]["loc"] = it.get("loc", [np.nan] * 3)
        for k in ("efo", "cfr", "dcr", "eco", "itr"):
            if k in it:
                out["itr"][i][k] = it[k]
        out["tim"][i], out["tid"][i], out["vld"][i] = r.get("tim", 0), r.get("tid", 0), r.get("vld", True)
    return out


def _flat_from_json(records: list) -> np.ndarray:
    fields = [("vld", "?"), ("fnl", "?"), ("itr", "i4"), ("tim", "f8"), ("tid", "i8"),
              ("loc", "f8", (3,)), ("lnc", "f8", (3,)), ("eco", "f8"), ("ecc", "f8"),
              ("efo", "f8"), ("efc", "f8"), ("fbg", "f8"), ("cfr", "f8"),
              ("dcr", "f8", (2,)), ("sta", "i4"), ("gri", "i4"), ("thi", "i4"),
              ("sqi", "i4")]
    have = [f for f in fields if f[0] in records[0]]
    out = np.zeros(len(records), dtype=have)
    for name in out.dtype.names:
        out[name] = [r[name] for r in records]
    return out


def is_flat(rec: np.ndarray) -> bool:
    """m2410's one row per iteration, against m2205's nested iterations."""
    names = rec.dtype.names or ()
    return "itr" in names and rec.dtype["itr"].base.names is None


# -------------------------------------------------------------------- table

def _cfr_iteration(itr: np.ndarray, cfr: np.ndarray) -> Optional[int]:
    """The iteration whose cfr is measured: the last one where it varies."""
    best = None
    for k in np.unique(itr):
        values = cfr[(itr == k) & np.isfinite(cfr)]
        if len(values) > 1 and np.std(values) > 0:
            best = int(k)
    return best


def _joined(rec: np.ndarray, rows: np.ndarray, source: np.ndarray,
            value: np.ndarray) -> np.ndarray:
    """For each row in ``rows`` (final iterations), ``value`` of the nearest
    earlier row of the same trace where ``source`` is set, with no final
    iteration between them: the same localization's earlier iteration.  NaN
    where that localization has none."""
    n = len(rec)
    thi = rec["thi"] if "thi" in rec.dtype.names else np.zeros(n, np.int8)
    order = np.lexsort((np.arange(n), rec["tid"], thi))
    tid, th = rec["tid"][order], thi[order]
    final = (rec["fnl"].astype(bool) if "fnl" in rec.dtype.names
             else np.zeros(n, bool))
    final[rows] = True
    final = final[order]
    start = np.ones(n, bool)
    start[1:] = (tid[1:] != tid[:-1]) | (th[1:] != th[:-1]) | final[:-1]
    segment = np.cumsum(start)
    at = np.where(source[order], np.arange(n), -1)
    last = np.maximum.accumulate(at)
    ok = (last >= 0) & (segment[np.maximum(last, 0)] == segment)
    out = np.full(n, np.nan, np.float32)
    out[ok] = value[order][last[ok]]
    back = np.empty(n, np.float32)
    back[order] = out
    return back[rows]


def _fields(rec: np.ndarray, valid_only: bool) -> Tuple[Dict[str, np.ndarray], int]:
    """The final iteration of every localization, as plain arrays, from
    either layout; and how many attempts there were."""
    names = rec.dtype.names or ()
    if is_flat(rec):
        final = rec["fnl"].astype(bool) if "fnl" in names else np.ones(len(rec), bool)
        if valid_only and "vld" in names:
            final &= rec["vld"].astype(bool)
        rows = np.flatnonzero(final)
        out = {k: rec[k][rows] for k in names if k not in ("fnl", "bot", "eot")}
        if "cfr" in names:
            cfr = np.asarray(rec["cfr"], np.float32)
            valid = rec["vld"].astype(bool) if "vld" in names else np.ones(len(rec), bool)
            k = _cfr_iteration(rec["itr"][valid], cfr[valid])
            last = int(rec["itr"][rows].max()) if len(rows) else None
            if k is not None and k != last:
                out["cfr"] = _joined(rec, rows, rec["itr"] == k, cfr)
        return out, int(rec["fnl"].sum()) if "fnl" in names else len(rec)
    itr = rec["itr"]
    last = itr[:, -1] if itr.ndim == 2 else itr
    keep = np.ones(len(rec), bool)
    if valid_only and "vld" in names:
        keep &= rec["vld"].astype(bool)
    out = {k: last[k][keep] for k in last.dtype.names}
    for k in names:
        if k != "itr":
            out[k] = rec[k][keep]
    if itr.ndim == 2 and "cfr" in itr.dtype.names:
        cfr = np.asarray(itr["cfr"][keep], np.float32)
        spread = [np.nanstd(cfr[:, j]) if len(cfr) > 1 else 0 for j in range(cfr.shape[1])]
        varying = [j for j, s in enumerate(spread) if s > 0]
        if varying:
            out["cfr"] = cfr[:, varying[-1]]
    return out, len(rec)


def table(rec: np.ndarray, valid_only: bool = True, z_scale: float = 1.0,
          dimensionality: Optional[int] = None, source: str = ""
          ) -> Tuple[Localizations, Dict[str, int]]:
    """The localizations of a MINFLUX record array, either layout, sorted by
    time; and the counts (attempts, localizations).

    ``dimensionality`` (2 or 3, from the measurement's metadata) decides
    whether there is a z; without it, a z that is not zero everywhere.
    """
    f, attempts = _fields(rec, valid_only)
    loc = np.asarray(f["loc"], np.float64) * 1e9
    keep = np.isfinite(loc[:, 0]) & np.isfinite(loc[:, 1])
    if "tim" in f:
        order = np.flatnonzero(keep)[np.argsort(np.asarray(f["tim"])[keep], kind="stable")]
    else:
        order = np.flatnonzero(keep)
    f = {k: np.asarray(v)[order] for k, v in f.items()}
    loc = loc[order]
    n = len(loc)
    with_z = dimensionality == 3 if dimensionality else bool(np.any(np.abs(loc[:, 2]) > 0))
    columns: Dict[str, np.ndarray] = {
        "x_nm": loc[:, 0].astype(np.float32), "y_nm": loc[:, 1].astype(np.float32),
        "frame": np.arange(n, dtype=np.int64)}
    if with_z:
        columns["z_nm"] = (loc[:, 2] * z_scale).astype(np.float32)
    if "lnc" in f:
        lnc = np.asarray(f["lnc"], np.float64) * 1e9
        if np.any(np.abs(np.nan_to_num(lnc - loc)) > 1e-3):   # MBM corrected it
            columns["x_nc_nm"] = lnc[:, 0].astype(np.float32)
            columns["y_nc_nm"] = lnc[:, 1].astype(np.float32)
            if with_z:
                columns["z_nc_nm"] = (lnc[:, 2] * z_scale).astype(np.float32)
    photons = (np.asarray(f["eco"], np.float32) if "eco" in f
               else np.full(n, np.nan, np.float32))
    columns["photons"] = photons
    columns["xy_err_nm"] = (MINFLUX_PSF_NM / np.sqrt(np.maximum(photons, 1))).astype(np.float32)
    columns["sigma_nm"] = np.full(n, MINFLUX_PSF_NM, np.float32)
    for name in ("efo", "cfr", "efc", "ecc", "fbg"):
        if name in f:
            columns[name] = np.asarray(f[name], np.float32)
    if "dcr" in f:
        dcr = np.asarray(f["dcr"], np.float32)
        columns["dcr"] = dcr[:, 0] if dcr.ndim == 2 else dcr
        if dcr.ndim == 2 and dcr.shape[1] > 1:
            columns["dcr2"] = dcr[:, 1]
    if "itr" in f:
        columns["iteration"] = np.asarray(f["itr"], np.int32)
    if "tim" in f:
        columns["time_s"] = np.asarray(f["tim"], np.float64)
    if "tid" in f:
        columns["tid"] = np.asarray(f["tid"], np.int64)
    for name in ("sta", "gri", "thi", "sqi"):
        if name in f:
            columns[name] = np.asarray(f[name], np.int32)
    if not valid_only and "vld" in f:
        columns["vld"] = np.asarray(f["vld"], np.int8)
    locs = Localizations(columns, {"units": "nm", "source": source, "minflux": True})
    return locs, {"attempts": attempts, "localizations": n}


def load(path: Path, valid_only: bool = True, z_scale: float = 1.0):
    """An export (`.npy`, `.json`, `.zip`) as a table and a `FileInfo`."""
    from .formats import FileInfo
    path = Path(path)
    locs, counts = table(records(path), valid_only, z_scale, source=str(path))
    return locs, FileInfo(path.name, str(path), "MINFLUX", pixelsize_nm=100.0,
                          metadata={"n_raw": counts["attempts"],
                                    "n_valid": counts["localizations"]})


# ----------------------------------------------------------------- metadata

def _flat(d: Any, prefix: str = "", out: Optional[dict] = None) -> dict:
    """Nested parameters as ``a.b`` keys; long lists (coefficient tables) left out."""
    out = {} if out is None else out
    if isinstance(d, dict):
        for k, v in d.items():
            _flat(v, f"{prefix}{k}.", out)
    elif isinstance(d, list):
        if len(d) <= 4 and all(not isinstance(v, (list, dict)) for v in d):
            out[prefix[:-1]] = d
    else:
        out[prefix[:-1]] = d
    return out


def _sequence_node(attrs: dict) -> Optional[dict]:
    """The sequence in either version's attributes: m2410
    ``measurement.threads[0].sequences[0]``, m2205 ``seq`` (or its own top)."""
    try:
        return attrs["measurement"]["threads"][0]["sequences"][0]
    except (KeyError, IndexError, TypeError):
        pass
    for key in ("seq", "sequence"):
        if isinstance(attrs.get(key), dict):
            return attrs[key]
    return attrs if "Itr" in attrs else None


def sequence(attrs: dict) -> Dict[str, Any]:
    """The MINFLUX sequence: a row of parameters per iteration and the
    sequence's global ones, flattened (``Mode.pattern``, ``_excitation.power``)."""
    node = _sequence_node(attrs) or {}
    iterations = [_flat(it) for it in node.get("Itr", [])]
    globals_ = _flat({k: v for k, v in node.items() if k != "Itr"})
    return {"iterations": iterations, "globals": globals_}


def instrument(attrs: dict) -> Dict[str, Any]:
    """Lasers and detectors of the first thread (m2410), as names and settings."""
    try:
        thread = attrs["measurement"]["threads"][0]
    except (KeyError, IndexError, TypeError):
        return {}
    lasers = [{k: l.get(k) for k in ("id", "calibration", "wavelength", "purpose")}
              for l in thread.get("lasers", [])]
    detectors = [{k: d.get(k) for k in ("id", "expcontrol_name")}
                 for d in thread.get("detectors", [])]
    return {"lasers": lasers, "detectors": detectors}


def iteration_table(seq: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    """The sequence's most read parameters, one line per iteration, as text."""
    keys = keys or ["Mode.pattern", "_excitation.laser", "_excitation.power",
                    "patGeoFactor", "patDwellTime", "phtLimit", "ccrLimit",
                    "bgcThreshold"]
    rows = seq.get("iterations", [])
    keys = [k for k in keys if any(k in r for r in rows)]
    lines = ["itr  " + "  ".join(keys)]
    for i, r in enumerate(rows):
        lines.append(f"{i:<4} " + "  ".join(str(r.get(k, "")) for k in keys))
    return "\n".join(lines)
