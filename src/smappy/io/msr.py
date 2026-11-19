"""Abberior Imspector `.msr` files: MINFLUX measurements, images, beads.

An `.msr` is an OBF container (`io.obf`) whose stacks are of three kinds:

* a **MINFLUX measurement** -- a stack of bytes whose ``minflux`` tag is
  ``{"type": "data", "label", "did", ...}``: inflated, an MFXDTA archive of a
  zarr tree (`io.zarr_lite`) holding the records (``zarr/mfx``, `io.minflux`),
  the sequence and the instrument (``zarr/mfx/.zattrs``), the ROI it was
  taken in (``zarr/.zattrs``), and the beam-monitoring beads
  (``zarr/grd/mbm/points``, ``zarr/mbm/.zattrs``);
* **images** -- confocal and STED channels (``Ch1 {11}``), placed in the
  sample in metres by each stack's offset and length.  MINFLUX positions are
  in the same frame -- the renders Imspector keeps beside a measurement have
  exactly the offset of the confocal they were drawn on -- so the image layer
  and the localizations line up without a registration;
* **renders** -- what Imspector drew of a measurement (``MF(label)/density/loc``,
  ``.../trace/tid``): images with a ``minflux`` tag of another type.  They are
  listed, and not offered by default: the localizations draw them again.

`contents` lists a file -- it reads the stack headers and inflates each
measurement's archive, a few tens of ms per measurement, but decodes no
records -- and `load` reads measurements into one table.  The GUI asks which
(`gui.dialogs.MsrImportDialog`); headless, `load(path)` reads all of them,
one after another in time (`merge`).

**The beads.**  Imspector keeps each MBM bead's track, a point every couple
of seconds, in ``zarr/grd/mbm/points``, and corrects the localizations by
the mean displacement of the used beads: ``loc = lnc - mean_b (bead_b(t) -
bead_b(t_0))``, interpolated to each localization's time.  Checked on a 3D
measurement of four beads (2026-10): that reconstruction leaves 1.3 / 1.6 /
0.08 nm rms in x / y / z of a correction that itself varies by 6 / 11 / 2.6
nm.  Some files keep only the first points, before the measurement starts
(seen on a file of 2026-06), and then ``loc - lnc`` is a constant.  So a
bad bead moves every localization by its share of the mean, and correcting
again needs the tracks: `bead_table` makes them a table of their own, a
second file beside the measurement's (``load(..., beads=True)``), with no
column the localizations do not have -- a bead's id is ``tid``, its
brightness ``photons``, its precision ``xy_err_nm`` -- so that opening it
beside them pads nothing.  The two are paired by `mbm_key`.  A two-colour
measurement may keep a grid per colour (``grd/mbm_0``, ``grd/mbm_1``, each
measured by a thread of its own and correcting the thread
``zarr/mbm/<grid>.correcting_thi`` names): each bead is then two tracks,
one per colour, a bias apart that differs from bead to bead (`_beads`).

**Colours.**  A localization's ``channel`` is its thread's MINFLUX laser,
numbered over the lasers of every measurement in the file, longest first
(`minflux.colours`): a bead measurement that ran 560 nm as thread 0 and a
tracking measurement that ran 640 nm as thread 0 -- the pair looked at
(2026-11) -- then name each colour alike, and so do their MBM grids.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations, concat
from . import minflux, obf, zarr_lite


@dataclass
class Measurement:
    """One MINFLUX measurement of the file, as listed."""
    number: int                       # its place among the file's measurements
    label: str
    did: str
    stack: obf.StackInfo
    layout: str = ""                  # "m2205" (nested) or "m2410" (flat)
    dimensionality: Optional[int] = None
    rows: int = 0                     # records: attempts, or iterations (m2410)
    date: str = ""                    # ISO, as Imspector wrote it
    n_beads: int = 0                  # MBM beads used (one colour), recorded (two: `_n_beads`)
    attrs: Dict[str, Any] = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        dim = f"{self.dimensionality}D" if self.dimensionality else "?D"
        beads = f", MBM {self.n_beads} beads" if self.n_beads else ""
        return f"{self.label}: {dim}, {self.rows} records, {self.layout}{beads}"


@dataclass
class Image:
    """An image stack of the file: a confocal channel or a render."""
    name: str
    stack: obf.StackInfo
    render: bool = False

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.stack.shape

    @property
    def pixelsize_nm(self) -> float:
        s = self.stack
        return s.length_m[0] / max(s.res[0], 1) * 1e9

    @property
    def pixelsize_y_nm(self) -> float:
        s = self.stack
        return s.length_m[1] / max(s.res[1], 1) * 1e9 if len(s.res) > 1 else self.pixelsize_nm

    def summary(self) -> str:
        size = " x ".join(str(n) for n in self.shape[::-1])
        field_um = " x ".join(f"{l * 1e6:.3g}" for l in self.stack.length_m[:2])
        return f"{self.name}: {size} px, {self.pixelsize_nm:.3g} nm, {field_um} µm"


@dataclass
class Contents:
    path: str
    measurements: List[Measurement]
    images: List[Image]

    @property
    def renders(self) -> List[Image]:
        return [im for im in self.images if im.render]


def _minflux_tag(stack: obf.StackInfo) -> Dict[str, Any]:
    text = stack.tag("minflux").strip()
    if not text.startswith("{"):
        return {}
    try:
        return json.loads(text)
    except ValueError:
        return {}


def _archive(path, stack: obf.StackInfo) -> Dict[str, bytes]:
    return zarr_lite.archive_files(obf.read_bytes(path, stack))


def contents(path) -> Contents:
    """What the file holds; no records are decoded."""
    measurements, images = [], []
    for stack in obf.stacks(path):
        tag = _minflux_tag(stack)
        if tag.get("type") == "data" and stack.dtype_code == 0x01:
            blob = obf.read_bytes(path, stack)
            m = _describe(len(measurements), stack, tag, zarr_lite.archive_files(blob))
            if not m.date and zarr_lite.archive_time(blob):
                m.date = datetime.fromtimestamp(zarr_lite.archive_time(blob)).astimezone().isoformat()
            measurements.append(m)
        elif stack.dtype is not None and int(np.prod(stack.res)) > 1:
            images.append(Image(stack.name or f"stack {stack.index}", stack,
                                render=bool(tag)))
    return Contents(str(path), measurements, images)


def _describe(number: int, stack, tag: dict, files: Dict[str, bytes]) -> Measurement:
    attrs = zarr_lite.attrs(files, "zarr/mfx")
    m = Measurement(number, tag.get("label") or stack.description or f"measurement {number + 1}",
                    tag.get("did", ""), stack, attrs=attrs)
    if zarr_lite.has_array(files, "zarr/mfx"):
        info = zarr_lite.array_info(files, "zarr/mfx")
        m.rows = int(info["shape"][0])
        names = [f[0] for f in info["dtype"]] if isinstance(info["dtype"], list) else []
        m.layout = "m2410" if "fnl" in names else "m2205"
    m.dimensionality = _dimensionality(attrs)
    m.date = str(attrs.get("acquisition_date") or "")
    m.n_beads = _n_beads(files)
    return m


def _dimensionality(attrs: dict) -> Optional[int]:
    for node in (attrs.get("measurement") or {}, attrs):
        if isinstance(node, dict) and node.get("dimensionality") in (2, 3):
            return int(node["dimensionality"])
    seq = minflux.sequence(attrs).get("iterations") or []
    dims = [r.get("Mode.dim") for r in seq if r.get("Mode.dim") in (2, 3)]
    return int(max(dims)) if dims else None


# --------------------------------------------------------------------- read

def _grid_names(files: Dict[str, bytes]) -> List[str]:
    """The MBM grids of a measurement: ``mbm`` alone (one colour), or one
    per colour (``mbm_0``, ``mbm_1``: ``grd/<name>`` of type ``mbm``)."""
    names = sorted({k.split("/")[2] for k in files
                    if k.startswith("zarr/grd/") and k.count("/") >= 4
                    and k.split("/")[3] == "points"})
    return [g for g in names
            if g == "mbm" or zarr_lite.attrs(files, f"zarr/grd/{g}").get("type") == "mbm"]


def _grid(files: Dict[str, bytes], node: str) -> Dict[str, Any]:
    """One grid's bead points, in nm and s, and Imspector's record of each
    bead."""
    out: Dict[str, Any] = {}
    if not zarr_lite.has_array(files, node):
        return out
    points = zarr_lite.read_array(files, node)
    names = points.dtype.names or ()
    table: Dict[str, Any] = {}
    if "gri" in names:
        table["gri"] = points["gri"].astype(int).tolist()
    if "xyz" in names:
        xyz = np.asarray(points["xyz"], np.float64) * 1e9
        for i, axis in enumerate("xyz"):
            table[f"{axis}_nm"] = np.round(xyz[:, i], 3).tolist()
    for name, key in (("tim", "time_s"), ("str", "strength")):
        if name in names:
            table[key] = np.asarray(points[name], np.float64).tolist()
    out["points"] = table
    by_gri = zarr_lite.attrs(files, node).get("points_by_gri") or {}
    out["beads"] = [{"name": b.get("name"), "gri": b.get("gri"),
                     "used": bool(b.get("used")),
                     "position_nm": [round(v * 1e9, 3) for v in b.get("position") or []],
                     # Imspector's own view of the bead: its precision
                     # (m), whether it is tracked, and how often it was lost
                     "sigpos_nm": _nm(b.get("sigpos")),
                     "state": b.get("state"), "times_failed": b.get("times_failed")}
                    for b in by_gri.values()]
    return out


def _beads(files: Dict[str, bytes], attrs: Optional[dict] = None,
           channel_of: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """The MBM beads: which were used, where they were, and their positions
    as Imspector recorded them, in nm and s.

    One colour: ``grd/mbm/points``, as ``{"used", "points", "beads"}``.  Two
    colours: a grid per colour (``grd/mbm_0``, ``grd/mbm_1``), each measured
    by a thread of its own (``watching_thi``, with its laser) and correcting
    the thread ``zarr/mbm/<grid>.correcting_thi`` names -- the same beads in
    both, as ``{"used", "grids": [...]}``, each grid with its ``thread``,
    ``corrects``, ``excitation_nm``, ``channel`` (as the localizations'),
    ``used``, ``points`` and ``beads``."""
    grids = _grid_names(files)
    if grids in ([], ["mbm"]):
        return {"used": list(zarr_lite.attrs(files, "zarr/mbm").get("used") or []),
                **_grid(files, "zarr/grd/mbm/points")}
    correcting = {}
    for k in {k.split("/")[2] for k in files if k.startswith("zarr/mbm/") and k.count("/") >= 3}:
        a = zarr_lite.attrs(files, f"zarr/mbm/{k}")
        if a.get("grid"):
            correcting[a["grid"]] = a
    out: Dict[str, Any] = {"used": [], "grids": []}
    for g in grids:
        ga = zarr_lite.attrs(files, f"zarr/grd/{g}")
        thread = ga.get("watching_thi", ga.get("controlling_thi"))
        ma = correcting.get(g, {})
        nm = minflux.excitation_nm(attrs, int(thread)) if attrs and thread is not None else None
        nm = round(float(nm), 1) if nm is not None else None
        used = list(ma.get("used") or [])
        out["used"] += [u for u in used if u not in out["used"]]
        out["grids"].append({
            "name": g, "thread": thread, "corrects": list(ma.get("correcting_thi") or []),
            "excitation_nm": nm,
            "channel": (channel_of or {}).get(int(thread)) if thread is not None else None,
            "used": used, **_grid(files, f"zarr/grd/{g}/points")})
    return out


def _grids_of(mbm: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A measurement's MBM grids as `_beads` keeps them: the list of a
    two-colour one, or the one grid of a one-colour one (and of files read
    before there were two)."""
    if mbm.get("grids"):
        return list(mbm["grids"])
    return [mbm] if mbm.get("points") else []


def _n_beads(files: Dict[str, bytes]) -> int:
    """The beads a measurement lists, without decoding their points: the
    used ones of one colour, or every bead a grid of two colours records
    (whose ``used`` lists stayed empty in the file looked at)."""
    grids = _grid_names(files)
    if grids in ([], ["mbm"]):
        return len(zarr_lite.attrs(files, "zarr/mbm").get("used") or [])
    slots = set()
    for g in grids:
        by_gri = zarr_lite.attrs(files, f"zarr/grd/{g}/points").get("points_by_gri") or {}
        slots |= {str(k) for k in by_gri}
    return len(slots)


def _nm(metres) -> Optional[float]:
    try:
        return round(float(metres) * 1e9, 4)
    except (TypeError, ValueError):
        return None


def _roi(files: Dict[str, bytes]) -> List[Dict[str, Any]]:
    rois = zarr_lite.attrs(files, "zarr").get("rois") or []
    return [{"name": r.get("name", ""), "type": r.get("type"),
             "corners_nm": [[round(v * 1e9, 3) for v in c] for c in r.get("corners") or []]}
            for r in rois]


def read_measurement(path, m: Measurement, valid_only: bool = True,
                     z_scale: float = 1.0, every: bool = False,
                     colours: Optional[Sequence[float]] = None
                     ) -> Tuple[Localizations, Dict[str, Any]]:
    """One measurement's table, and what is known about it.  ``every``:
    every iteration row (`minflux.table`).  ``colours``: the lasers (nm)
    that number ``channel``, longest first -- the file's (`minflux.colours`
    over its measurements); by default this measurement's own."""
    files = _archive(path, m.stack)
    rec = zarr_lite.read_array(files, "zarr/mfx")
    threads = minflux.sequences(m.attrs)
    seq = threads[0]
    locs, counts = minflux.table(rec, valid_only, z_scale, m.dimensionality,
                                 source=str(path), sequence=seq, every=every,
                                 threads=threads if len(threads) > 1 else None)
    order = list(colours) if colours is not None else minflux.colours([m.attrs])
    channel_of = minflux.channel_of_threads(m.attrs, order)
    thi = (np.asarray(locs["thi"], np.int64) if "thi" in locs
           else np.zeros(len(locs), np.int64))
    if channel_of and all(int(t) in channel_of for t in np.unique(thi)):
        lut = np.full(max(channel_of) + 1, -1, np.int16)
        for t, c in channel_of.items():
            lut[t] = c
        locs.columns["channel"] = lut[thi]
    meta = {"label": m.label, "did": m.did, "file": Path(path).name,
            "acquisition_date": m.date, "layout": m.layout,
            "dimensionality": m.dimensionality, "zarr_version": m.attrs.get("version"),
            "z_scale": z_scale, "valid_only": valid_only, "every_iteration": every,
            "records": m.rows,
            **counts,
            "duration_s": (float(np.ptp(locs["time_s"])) if len(locs) and "time_s" in locs
                           else 0.0),
            "sequence": seq, "instrument": minflux.instrument(m.attrs),
            **({"sequences": threads,
                "instruments": [minflux.instrument(m.attrs, i) for i in range(len(threads))]}
               if len(threads) > 1 else {}),
            **({"channels": _channels(m.attrs, channel_of, order)} if "channel" in locs
               else {}),
            "roi": _roi(files), "mbm": _beads(files, m.attrs, channel_of)}
    return locs, meta


def _channels(attrs: dict, channel_of: Dict[int, int], order: Sequence[float]
              ) -> List[Dict[str, Any]]:
    """What each ``channel`` is: its laser, the threads that localize in it
    and their detectors -- what a figure or a layer names a colour by."""
    out = []
    skip = minflux.mbm_threads(attrs)
    for c, nm in enumerate(order):
        mine = [t for t, k in sorted(channel_of.items()) if k == c and t not in skip]
        detectors = [f"{d.get('id')} {d.get('expcontrol_name') or ''}".strip()
                     for t in mine for d in minflux.instrument(attrs, t).get("detectors", [])]
        out.append({"channel": c, "excitation_nm": float(nm), "name": f"{nm:g} nm",
                    "threads": mine, "detectors": detectors})
    return out


def _start(date: str) -> Optional[float]:
    try:
        return datetime.fromisoformat(date).timestamp()
    except (TypeError, ValueError):
        return None


def merge(tables: Sequence[Localizations], metas: Sequence[Dict[str, Any]]) -> Localizations:
    """Measurements into one table, one after another in time.

    Each one's ``time_s`` is moved by when it started after the first one
    did (``acquisition_date``, to the second), and never back into the one
    before it; its traces are renumbered after the last one's, since a
    trace id is only unique within its measurement, Imspector's own number
    kept in ``tid_file`` (`file_ids`), and a ``measurement`` column says
    which it was.  ``frame`` is the rank in the joined time.
    """
    parts, offset, tid_offset, end = [], 0.0, 0, None
    first = _start(metas[0].get("acquisition_date", "")) if metas else None
    for k, (locs, meta) in enumerate(zip(tables, metas)):
        columns = dict(locs.columns)
        start = _start(meta.get("acquisition_date", ""))
        offset = start - first if start is not None and first is not None else 0.0
        if end is not None and "time_s" in columns and len(locs):
            offset = max(offset, end - float(np.min(columns["time_s"])) + 1e-3)
        if "time_s" in columns:
            columns["time_s"] = np.asarray(columns["time_s"], np.float64) + offset
            if len(locs):
                end = float(np.max(columns["time_s"]))
        meta["time_offset_s"] = offset
        if "tid" in columns:
            if len(tables) > 1:          # Imspector's own number, `file_ids`
                columns["tid_file"] = np.asarray(columns["tid"], np.int64)
            columns["tid"] = np.asarray(columns["tid"], np.int64) + tid_offset
            if len(locs):
                tid_offset = int(np.max(columns["tid"])) + 1
        columns["measurement"] = np.full(len(locs), k, np.int16)
        parts.append(Localizations(columns, dict(locs.metadata)))
    out = concat(parts) if len(parts) > 1 else parts[0]
    if "time_s" in out and len(out):
        order = np.argsort(out["time_s"], kind="stable")
        out = Localizations({c: np.asarray(v)[order] for c, v in out.columns.items()},
                            dict(out.metadata))
    out.columns["frame"] = np.arange(len(out), dtype=np.int64)
    return out


def bead_table(metas: Sequence[Dict[str, Any]]) -> Optional[Localizations]:
    """The MBM beads of measurements read together, a row per recorded point.

    On the localizations' clock: each measurement's ``time_offset_s`` (`merge`)
    is added, so a bead and the localizations it corrected share ``time_s``.
    z gets the measurement's *z scale*, as the localizations do, and is left
    out of a 2D measurement, whose beads' z Imspector does not track.  A
    bead's id (``tid``, from 1) is renumbered per measurement: Imspector's
    ``gri`` is a slot -- R0 is gri 61 in every measurement, and in two
    measurements of one file it was two beads 20 um apart -- so one ``gri``
    is not one bead.  ``frame`` is the rank in time, as in the localizations.
    Each bead's name, slot, use and Imspector's numbers go in
    ``metadata["mbm_beads"]``.  None if no measurement recorded a bead.

    **Two colours**: a measurement with a grid per colour (`_beads`) gives
    each bead once per grid, an id of its own each -- the two colours of a
    bead are two tracks, a bias apart -- with ``channel`` the grid's colour,
    numbered as the localizations' are, where every grid's is known.  Its
    entry adds the ``grid``, the ``channel``, the laser (``excitation_nm``)
    and the threads it ``corrects``, and ``bead`` names it uniquely in its
    measurement (``R0@638``); one grid's beads keep ``bead`` = the name.
    """
    parts, beads, next_id = [], [], 1
    three_d = any(m.get("dimensionality") == 3 for m in metas)
    # a channel column only where every measurement's beads have a colour:
    # the localizations then have one too, and a bead table adds no column
    colour_grids = [g for meta in metas for g in (meta.get("mbm") or {}).get("grids") or []]
    one_colour = any(not (meta.get("mbm") or {}).get("grids")
                     and (meta.get("mbm") or {}).get("points") for meta in metas)
    with_channel = (bool(colour_grids) and not one_colour
                    and all(g.get("channel") is not None for g in colour_grids))
    for k, meta in enumerate(metas):
        mbm = meta.get("mbm") or {}
        grids = _grids_of(mbm)
        several = len(grids) > 1
        for grid in grids:
            points = grid.get("points") or {}
            if not points.get("gri"):
                continue
            gri = np.asarray(points["gri"], np.int64)
            known = {int(b["gri"]): b for b in grid.get("beads") or []
                     if b.get("gri") is not None}
            used = set(grid.get("used") or mbm.get("used") or [])
            slots = np.unique(gri)
            ids = next_id + np.searchsorted(slots, gri)
            columns = {"x_nm": np.asarray(points["x_nm"], np.float64),
                       "y_nm": np.asarray(points["y_nm"], np.float64),
                       "time_s": (np.asarray(points["time_s"], np.float64)
                                  + float(meta.get("time_offset_s", 0.0))),
                       "tid": ids.astype(np.int64),
                       "photons": np.asarray(points.get("strength", np.full(len(gri), np.nan)),
                                             np.float64)}
            if three_d:
                columns["z_nm"] = (np.asarray(points["z_nm"], np.float64)
                                   * float(meta.get("z_scale", 1.0)))
            err = np.full(len(gri), np.nan)
            nm = grid.get("excitation_nm")
            colour = (f"{nm:g}" if nm else str(grid.get("name", ""))) if several else ""
            for j, slot in enumerate(slots):
                b = known.get(int(slot), {})
                if b.get("sigpos_nm"):
                    err[gri == slot] = b["sigpos_nm"]
                name = b.get("name")
                entry = {"tid": int(next_id + j), "name": name, "gri": int(slot),
                         "bead": (f"{name or slot}@{colour}" if colour else name),
                         "used": (bool(b["used"]) if "used" in b
                                  else (name in used if name else None)),
                         "measurement": k, "label": meta.get("label"),
                         "position_nm": b.get("position_nm"),
                         "sigpos_nm": b.get("sigpos_nm"), "state": b.get("state"),
                         "times_failed": b.get("times_failed")}
                if several:
                    entry.update(grid=grid.get("name"), channel=grid.get("channel"),
                                 excitation_nm=nm, corrects=list(grid.get("corrects") or []))
                beads.append(entry)
            columns["xy_err_nm"] = err
            if with_channel:
                columns["channel"] = np.full(len(gri), grid.get("channel", -1), np.int16)
            if len(metas) > 1:
                columns["measurement"] = np.full(len(gri), k, np.int16)
            parts.append(Localizations(columns, {}))
            next_id += len(slots)
    if not parts:
        return None
    out = concat(parts) if len(parts) > 1 else parts[0]
    order = np.argsort(out["time_s"], kind="stable")
    columns = {c: np.asarray(v)[order] for c, v in out.columns.items()}
    columns["frame"] = np.arange(len(order), dtype=np.int64)
    return Localizations(columns, {"units": "nm", "mbm_beads": beads})


def mbm_key(info) -> Optional[Tuple[str, Tuple[str, ...], bool]]:
    """What pairs a measurement's localizations with its beads: the file's
    path, the measurements' ids (``did``), and whether it is the beads.

    Never the file's number or its name, which change as files are added,
    removed or renamed; the path and the ids are what the two were read from.
    None for a file that is neither.
    """
    meta = getattr(info, "metadata", None) or {}
    of = meta.get("mbm_of")
    if of:
        return str(Path(of["path"])), tuple(of["measurements"]), True
    measurements = (meta.get("minflux") or {}).get("measurements") \
        if isinstance(meta.get("minflux"), dict) else None
    if measurements:
        return (str(Path(info.path)), tuple(m.get("did", "") for m in measurements), False)
    return None


def mbm_partner(files: Sequence[Any], number: int) -> Optional[int]:
    """The number of the file paired with file ``number``: a measurement's
    bead file, or a bead file's measurement; None if it is not open."""
    key = mbm_key(files[number]) if 0 <= number < len(files) else None
    if key is None:
        return None
    for k, other in enumerate(files):
        theirs = mbm_key(other)
        if k != number and theirs is not None and theirs[:2] == key[:2] \
                and theirs[2] != key[2]:
            return k
    return None


# What `_read` made of the last few loads: a bead file asked for after its
# measurement (the GUI's queue opens them one after the other) takes the
# offsets from here rather than decoding the records a second time.
_METAS: "OrderedDict[tuple, List[Dict[str, Any]]]" = OrderedDict()


def _metas_key(path: Path, chosen: Sequence[Measurement], valid_only: bool,
               z_scale: float, every: bool = False) -> tuple:
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = None
    return (str(path), stamp, tuple(m.did or m.number for m in chosen), valid_only,
            float(z_scale), bool(every))


def select(found: Contents, measurements=None) -> List[Measurement]:
    """The measurements named: None or "" all; otherwise numbers (from 1),
    labels or dids, or a regular expression matched against the labels --
    a list, or one string separated by commas."""
    every = found.measurements
    if measurements is None or measurements == "" or measurements == "all":
        return list(every)
    if isinstance(measurements, str):
        measurements = [p.strip() for p in measurements.split(",") if p.strip()]
    out: List[Measurement] = []
    for want in measurements:
        if isinstance(want, Measurement):
            hit = [want]
        elif isinstance(want, int) or str(want).isdigit():
            n = int(want) - 1
            hit = [every[n]] if 0 <= n < len(every) else []
        else:
            hit = [m for m in every if want in (m.label, m.did)] or \
                  [m for m in every if re.search(str(want), m.label)]
        if not hit:
            raise ValueError(f"no measurement {want!r} in {Path(found.path).name}; "
                             "it has " + ", ".join(m.label for m in every))
        out += [m for m in hit if m not in out]
    return out


def _read(path: Path, found: Contents, chosen: Sequence[Measurement], valid_only: bool,
          z_scale: float, progress=None, beads: bool = False, every: bool = False):
    from .formats import FileInfo
    if not chosen:
        raise ValueError(f"{path.name} holds no MINFLUX measurement"
                         + (f" (it has {len(found.images)} images: File > Open image)"
                            if found.images else ""))
    key = _metas_key(path, chosen, valid_only, z_scale, every)
    if len(chosen) == 1:
        name = f"{path.stem}: {chosen[0].label}" if len(found.measurements) > 1 else path.name
    elif len(chosen) == len(found.measurements):
        name = path.name
    else:
        name = f"{path.stem}: {len(chosen)} measurements"
    if beads and key in _METAS:
        return _bead_file(path, name, _METAS[key])
    tables, metas = [], []
    # every measurement of the file numbers the colours, so that one read
    # alone names them as it does read with the others
    colours = minflux.colours([m.attrs for m in found.measurements])
    for m in chosen:
        if progress:
            progress(f"reading {m.label}")
        locs, meta = read_measurement(path, m, valid_only, z_scale, every, colours)
        tables.append(locs)
        metas.append(meta)
    locs = merge(tables, metas) if len(tables) > 1 else tables[0]
    _METAS[key] = metas
    while len(_METAS) > 8:
        _METAS.popitem(last=False)
    if beads:
        return _bead_file(path, name, metas)
    locs.metadata = {**locs.metadata, "minflux": {"measurements": metas}}
    return locs, FileInfo(name, str(path), "Abberior MSR", pixelsize_nm=100.0,
                          metadata={"minflux": {"measurements": metas}})


def _bead_file(path: Path, name: str, metas: Sequence[Dict[str, Any]]):
    from .formats import FileInfo
    table = bead_table(metas)
    if table is None:
        raise ValueError(f"{name}: no MBM beads recorded")
    of = {"path": str(path), "measurements": [m.get("did", "") for m in metas],
          "labels": [m.get("label") for m in metas]}
    table.metadata.update({"source": str(path), "mbm_of": of})
    return table, FileInfo(f"{name} (MBM beads)", str(path), "Abberior MSR",
                           pixelsize_nm=100.0,
                           metadata={"mbm_of": of, "mbm_beads": table.metadata["mbm_beads"]})


def load(path, measurements=None, valid_only: bool = True, z_scale: float = 1.0,
         progress=None, beads: bool = False, every: bool = False):
    """The measurements chosen (`select`; all by default) as one table, and
    its `FileInfo`.  Several are joined by `merge`.  ``beads``: their MBM
    beads instead (`bead_table`), as the file paired with that table."""
    path = Path(path)
    found = contents(path)
    return _read(path, found, select(found, measurements), valid_only, z_scale, progress,
                 beads, every)


def read_files(path, measurements=None, merged: bool = False, valid_only: bool = True,
               z_scale: float = 1.0, progress=None, found: Optional[Contents] = None,
               beads: bool = False, every: bool = False):
    """The measurements chosen as ``(locs, info)`` pairs: one per measurement,
    or one for all of them when ``merged``.  ``beads``: each followed by its
    MBM beads, where it recorded any."""
    path = Path(path)
    found = found or contents(path)
    chosen = select(found, measurements)
    groups = [chosen] if merged or len(chosen) <= 1 else [[m] for m in chosen]
    out = []
    for group in groups:
        out.append(_read(path, found, group, valid_only, z_scale, progress, every=every))
        if beads and any(m.n_beads for m in group):
            try:
                out.append(_read(path, found, group, valid_only, z_scale, progress, True,
                                 every))
            except ValueError:                # the beads were named but never recorded
                pass
    return out


def image(path, im: Image):
    """An image of the file as an `ImageData`, placed in nm."""
    from ..images import ImageData
    data = obf.read_stack(path, im.stack)
    s = im.stack
    meta: Dict[str, Any] = {"source": Path(path).name, "stack": s.name}
    if data.ndim == 3 and len(s.res) > 2:
        # a z-stack (`ImageData.axis`): the offset is the outer edge of the
        # first plane, as it is of the first pixel, and z0 that plane's centre
        step = s.length_m[2] / max(s.res[2], 1) * 1e9
        meta.update(axis="z", z_step_nm=step, z0_nm=s.offset_m[2] * 1e9 + 0.5 * step)
    py = im.pixelsize_y_nm
    return ImageData(np.ascontiguousarray(data), im.pixelsize_nm,
                     x0=s.offset_m[0] * 1e9, y0=s.offset_m[1] * 1e9 if len(s.offset_m) > 1 else 0.0,
                     name=f"{Path(path).stem}: {im.name}", path=str(path), metadata=meta,
                     pixelsize_y=None if abs(py - im.pixelsize_nm) < 1e-6 * im.pixelsize_nm else py)
