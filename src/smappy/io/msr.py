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
"""
from __future__ import annotations

import json
import re
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
    n_beads: int = 0                  # MBM beads used
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
    m.n_beads = len(zarr_lite.attrs(files, "zarr/mbm").get("used") or [])
    return m


def _dimensionality(attrs: dict) -> Optional[int]:
    for node in (attrs.get("measurement") or {}, attrs):
        if isinstance(node, dict) and node.get("dimensionality") in (2, 3):
            return int(node["dimensionality"])
    seq = minflux.sequence(attrs).get("iterations") or []
    dims = [r.get("Mode.dim") for r in seq if r.get("Mode.dim") in (2, 3)]
    return int(max(dims)) if dims else None


# --------------------------------------------------------------------- read

def _beads(files: Dict[str, bytes]) -> Dict[str, Any]:
    """The MBM beads: which were used, where they were, and their positions
    as Imspector recorded them (``grd/mbm/points``), in nm and s."""
    out: Dict[str, Any] = {"used": list(zarr_lite.attrs(files, "zarr/mbm").get("used") or [])}
    node = "zarr/grd/mbm/points"
    if zarr_lite.has_array(files, node):
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
                         "position_nm": [round(v * 1e9, 3) for v in b.get("position") or []]}
                        for b in by_gri.values()]
    return out


def _roi(files: Dict[str, bytes]) -> List[Dict[str, Any]]:
    rois = zarr_lite.attrs(files, "zarr").get("rois") or []
    return [{"name": r.get("name", ""), "type": r.get("type"),
             "corners_nm": [[round(v * 1e9, 3) for v in c] for c in r.get("corners") or []]}
            for r in rois]


def read_measurement(path, m: Measurement, valid_only: bool = True,
                     z_scale: float = 1.0) -> Tuple[Localizations, Dict[str, Any]]:
    """One measurement's table, and what is known about it."""
    files = _archive(path, m.stack)
    rec = zarr_lite.read_array(files, "zarr/mfx")
    seq = minflux.sequence(m.attrs)
    locs, counts = minflux.table(rec, valid_only, z_scale, m.dimensionality,
                                 source=str(path), sequence=seq)
    meta = {"label": m.label, "did": m.did, "file": Path(path).name,
            "acquisition_date": m.date, "layout": m.layout,
            "dimensionality": m.dimensionality, "zarr_version": m.attrs.get("version"),
            "z_scale": z_scale, "valid_only": valid_only, "records": m.rows,
            **counts,
            "duration_s": (float(np.ptp(locs["time_s"])) if len(locs) and "time_s" in locs
                           else 0.0),
            "sequence": seq, "instrument": minflux.instrument(m.attrs),
            "roi": _roi(files), "mbm": _beads(files)}
    return locs, meta


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
    trace id is only unique within its measurement, and a ``measurement``
    column says which it was.  ``frame`` is the rank in the joined time.
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
          z_scale: float, progress=None):
    from .formats import FileInfo
    if not chosen:
        raise ValueError(f"{path.name} holds no MINFLUX measurement"
                         + (f" (it has {len(found.images)} images: File > Open image)"
                            if found.images else ""))
    tables, metas = [], []
    for m in chosen:
        if progress:
            progress(f"reading {m.label}")
        locs, meta = read_measurement(path, m, valid_only, z_scale)
        tables.append(locs)
        metas.append(meta)
    locs = merge(tables, metas) if len(tables) > 1 else tables[0]
    locs.metadata = {**locs.metadata, "minflux": {"measurements": metas}}
    if len(chosen) == 1:
        name = f"{path.stem}: {chosen[0].label}" if len(found.measurements) > 1 else path.name
    elif len(chosen) == len(found.measurements):
        name = path.name
    else:
        name = f"{path.stem}: {len(chosen)} measurements"
    return locs, FileInfo(name, str(path), "Abberior MSR", pixelsize_nm=100.0,
                          metadata={"minflux": {"measurements": metas}})


def load(path, measurements=None, valid_only: bool = True, z_scale: float = 1.0,
         progress=None):
    """The measurements chosen (`select`; all by default) as one table, and
    its `FileInfo`.  Several are joined by `merge`."""
    path = Path(path)
    found = contents(path)
    return _read(path, found, select(found, measurements), valid_only, z_scale, progress)


def read_files(path, measurements=None, merged: bool = False, valid_only: bool = True,
               z_scale: float = 1.0, progress=None, found: Optional[Contents] = None):
    """The measurements chosen as ``(locs, info)`` pairs: one per measurement,
    or one for all of them when ``merged``."""
    path = Path(path)
    found = found or contents(path)
    chosen = select(found, measurements)
    if merged or len(chosen) <= 1:
        return [_read(path, found, chosen, valid_only, z_scale, progress)]
    return [_read(path, found, [m], valid_only, z_scale, progress) for m in chosen]


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
