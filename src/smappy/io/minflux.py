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
(`group.by_trace`).  The photons are the final iteration's ``eco``.

The records give no precision, so ``xy_err_nm`` is the Cramér-Rao bound at
the centre of the beam pattern -- a number from the photons, as a camera
fitter's CRLB is, and not the spread that is measured, which is the
precision plugins' job.  In the quadratic approximation of the doughnut's
zero, with Ns photons from the molecule and Nb from background counted at
the pattern's K >= 3 offsets on a circle of diameter L (`crlb_xy`),

    sigma_xy = L / (2 sqrt(2 Ns)) * sqrt(1 + Nb / Ns)

whatever K, and with or without a centre exposure, which at the centre adds
no information.  Without background it is Balzarotti et al. 2017's Eq. S27
(Marin & Ries 2024, Eq. 45); with it, the 2D form of Marin & Ries's Eq. 33,
derived the same way (the multinomial Fisher information of the counts), and
equal to their Eq. 44 for K = 3 with a centre, which is written with the
total photons and the SBR.  Along z, two offsets on a line, it is
L / (4 sqrt(Ns)) * sqrt(1 + Nb / Ns) (`crlb_z`; Eq. 33 with only the
offsets' background).  It is the bound for a molecule at the centre; one off
it is localized worse, by 1 + (2r/L)^2 along a line, and the records do not
say where it was.

L is the pattern size of the sequence's last lateral iteration,
``patGeoFactor`` x 360 nm (`pattern_sizes`; Abberior's guide pairs 0.21,
0.28 and 0.42 with 75, 100 and 150 nm), and of its last axial (``zline``)
one for z.  A 3D sequence alternates the two, so the lateral photons are the
last lateral iteration's of the same localization, joined as the cfr is.
The photons ``eco`` are split into Ns and Nb by ``fbg``/``efo`` (`photon_split`),
Imspector's background rate over its photon rate -- *assuming* ``efo``
includes the background and ``fbg`` is a rate at the offsets in the same
units, which nothing Abberior publishes says.  With background sensing off,
as in every file looked at, ``fbg`` is 0 and Nb is 0.  An export without its
sequence (`.npy`, `.json`) gets L = 100 nm, the 2D default, and no z
precision.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..locs import Localizations

MINFLUX_PSF_NM = 150.0      # sigma_nm, SMAP's number
PATTERN_UNIT_NM = 360.0     # L for a patGeoFactor of 1
DEFAULT_PATTERN_NM = 100.0  # L when the sequence is not known: Abberior's 2D final
MAX_BACKGROUND_FRACTION = 0.9  # beyond it the photons are background, not a bound


# --------------------------------------------------------------- precision

def _with_background(n_signal, n_background):
    ns = np.maximum(np.asarray(n_signal, np.float64), 1.0)
    return ns, np.sqrt(1.0 + np.maximum(np.nan_to_num(np.asarray(n_background, np.float64)),
                                        0.0) / ns)


def crlb_xy(pattern_nm, n_signal, n_background=0.0) -> np.ndarray:
    """The lateral Cramér-Rao bound at the pattern's centre, per axis:
    L / (2 sqrt(2 Ns)) sqrt(1 + Nb/Ns), for Ns signal and Nb background
    photons counted at the offsets."""
    ns, factor = _with_background(n_signal, n_background)
    return np.asarray(pattern_nm, np.float64) / (2.0 * np.sqrt(2.0 * ns)) * factor


def crlb_z(pattern_nm, n_signal, n_background=0.0) -> np.ndarray:
    """The axial bound at the centre of a line of two offsets:
    L / (4 sqrt(Ns)) sqrt(1 + Nb/Ns)."""
    ns, factor = _with_background(n_signal, n_background)
    return np.asarray(pattern_nm, np.float64) / (4.0 * np.sqrt(ns)) * factor


def photon_split(eco, efo, fbg) -> Tuple[np.ndarray, np.ndarray]:
    """The counts ``eco`` as (signal, background), in the ratio of
    Imspector's background rate ``fbg`` to its photon rate ``efo`` -- if
    ``efo`` includes the background and ``fbg`` is in its units, which is
    assumed, not documented.  No background where ``fbg`` is 0 or missing."""
    eco = np.nan_to_num(np.asarray(eco, np.float64))
    efo = np.asarray(efo, np.float64)
    fbg = np.nan_to_num(np.asarray(fbg, np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = fbg / efo
    b = np.clip(np.where(np.isfinite(b), b, 0.0), 0.0, MAX_BACKGROUND_FRACTION)
    return eco * (1.0 - b), eco * b


def pattern_sizes(seq: Optional[Dict[str, Any]]) -> Dict[str, Tuple[int, float]]:
    """The last lateral and the last axial iteration of a `sequence`, each
    with its pattern size L in nm: ``{"xy": (k, L), "z": (k, L)}``, a key
    left out where the sequence has no such iteration."""
    out: Dict[str, Tuple[int, float]] = {}
    for k, it in enumerate((seq or {}).get("iterations") or []):
        geo = it.get("patGeoFactor")
        if isinstance(geo, (list, tuple)):      # a coarse zline's two: never the last
            geo = geo[0] if geo else None
        if geo is None or "Mode.pattern" not in it:
            continue
        axial = str(it["Mode.pattern"]).lower().startswith("z")
        out["z" if axial else "xy"] = (k, float(geo) * PATTERN_UNIT_NM)
    return out


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


PER_ITERATION = ("eco", "efo", "fbg")   # what the precision reads, of its iteration


def _fields(rec: np.ndarray, valid_only: bool, iterations: Tuple[int, ...] = ()
            ) -> Tuple[Dict[str, np.ndarray], int]:
    """The final iteration of every localization, as plain arrays, from
    either layout; and how many attempts there were.  For each of
    ``iterations``, also ``PER_ITERATION`` of that iteration of the same
    localization, as ``"eco@k"``: NaN where it has none."""
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
        for k in iterations:
            for name in PER_ITERATION:
                if name in names:
                    out[f"{name}@{k}"] = _joined(rec, rows, rec["itr"] == k,
                                                 np.asarray(rec[name], np.float32))
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
    if itr.ndim == 2:
        for k in iterations:
            for name in PER_ITERATION:
                if name in itr.dtype.names and k < itr.shape[1]:
                    out[f"{name}@{k}"] = np.asarray(itr[name][keep, k], np.float32)
    return out, len(rec)


def _at_iteration(f: Dict[str, np.ndarray], name: str, k: Optional[int]) -> np.ndarray:
    """``name`` of iteration ``k`` of each localization; the final
    iteration's where there is no k, or the localization has none."""
    n = len(f["loc"])
    final = np.asarray(f[name], np.float64) if name in f else np.full(n, np.nan)
    if k is None or f"{name}@{k}" not in f:
        return final
    value = np.asarray(f[f"{name}@{k}"], np.float64)
    return np.where(np.isfinite(value), value, final)


def _bound(f: Dict[str, np.ndarray], k: Optional[int], pattern_nm: float, crlb) -> np.ndarray:
    signal, background = photon_split(*(_at_iteration(f, name, k) for name in PER_ITERATION))
    return crlb(pattern_nm, signal, background).astype(np.float32)


def table(rec: np.ndarray, valid_only: bool = True, z_scale: float = 1.0,
          dimensionality: Optional[int] = None, source: str = "",
          sequence: Optional[Dict[str, Any]] = None
          ) -> Tuple[Localizations, Dict[str, int]]:
    """The localizations of a MINFLUX record array, either layout, sorted by
    time; and the counts (attempts, localizations).

    ``dimensionality`` (2 or 3, from the measurement's metadata) decides
    whether there is a z; without it, a z that is not zero everywhere.
    ``sequence`` (`sequence`) gives the pattern sizes of the precision.
    """
    sizes = pattern_sizes(sequence)
    f, attempts = _fields(rec, valid_only, tuple(k for k, _ in sizes.values()))
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
    k_xy, pattern_xy = sizes.get("xy", (None, DEFAULT_PATTERN_NM))
    columns["xy_err_nm"] = _bound(f, k_xy, pattern_xy, crlb_xy)
    if with_z and "z" in sizes:
        k_z, pattern_z = sizes["z"]
        columns["z_err_nm"] = _bound(f, k_z, pattern_z * z_scale, crlb_z)
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
    names = ["itr"] + [k.split(".")[-1].lstrip("_") for k in keys]
    cells = [[str(i)] + [str(r.get(k, "")) for k in keys] for i, r in enumerate(rows)]
    widths = [max(len(c[j]) for c in [names] + cells) for j in range(len(names))]
    return "\n".join("  ".join(c[j].ljust(widths[j]) for j in range(len(names))).rstrip()
                     for c in [names] + cells)
