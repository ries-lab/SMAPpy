"""Localization file formats: one registry, one ``load`` for all of them.

Every reader turns a file into smappy's column set -- ``x_nm``, ``y_nm``,
``z_nm``, ``frame``, ``photons``, ``loc_precision_nm``, ``sigma_nm``,
``logl_rel``, ... -- keeping any extra column under its own name, and
returns it with a :class:`FileInfo`.  Positions are in nm; the pixel size,
where the format records one, is in the info.

    from smappy.io.formats import load
    locs, info = load("run_sml.mat")
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..locs import Localizations


@dataclass
class FileInfo:
    name: str
    path: str
    format: str
    n: int = 0
    pixelsize_nm: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"name": self.name, "path": self.path, "format": self.format, "n": self.n,
                "pixelsize_nm": self.pixelsize_nm}


@dataclass
class Reader:
    name: str
    suffixes: Tuple[str, ...]
    load: Callable[..., Tuple[Localizations, FileInfo]]
    sniff: Optional[Callable[[Path], bool]] = None   # beyond the suffix
    needs: Tuple[str, ...] = ()       # extra arguments a GUI must ask for


READERS: List[Reader] = []


def register(reader: Reader) -> Reader:
    READERS.append(reader)
    return reader


def reader_for(path) -> Reader:
    path = Path(path)
    name = path.name.lower()
    candidates = [r for r in READERS if any(name.endswith(s) for s in r.suffixes)]
    for r in candidates:
        if r.sniff is None or r.sniff(path):
            return r
    raise ValueError(f"no reader for {path.name}; known: "
                     + ", ".join(f"{r.name} ({' '.join(r.suffixes)})" for r in READERS))


def load(path, **kwargs) -> Tuple[Localizations, FileInfo]:
    """Read any known format.  ``kwargs`` go to the reader (a csv mapping, say)."""
    reader = reader_for(path)
    locs, info = reader.load(Path(path), **kwargs)
    info.n = len(locs)
    return locs, info


def name_filter() -> str:
    """A Qt file-dialog filter over every reader."""
    every = " ".join(f"*{s}" for r in READERS for s in r.suffixes)
    parts = [f"Localizations ({every})"]
    parts += [f"{r.name} ({' '.join('*' + s for s in r.suffixes)})" for r in READERS]
    return ";;".join(parts + ["All files (*)"])


# ------------------------------------------------------------------ smappy
def _load_smappy(path: Path) -> Tuple[Localizations, FileInfo]:
    from .hdf5 import load_localizations
    locs = load_localizations(path)
    cam = (locs.metadata.get("camera") or {})
    px = cam.get("pixelsize_um")
    return locs, FileInfo(path.name, str(path), "smappy",
                          pixelsize_nm=px * 1000 if px else None, metadata=locs.metadata)


def _is_smappy(path: Path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return "locs" in f
    except OSError:
        return False


register(Reader("smappy HDF5", (".hdf5", ".h5"), _load_smappy, _is_smappy))


# --------------------------------------------------------------- SMAP sml
SML_COLUMNS = {
    "xnm": "x_nm", "ynm": "y_nm", "znm": "z_nm", "frame": "frame",
    "phot": "photons", "bg": "background", "locprecnm": "loc_precision_nm",
    "locprecznm": "loc_precision_z_nm", "PSFxnm": "sigma_nm", "PSFynm": "sigma_y_nm",
    "LLrel": "logl_rel", "logLikelihood": "logl", "channel": "channel",
    "xnmerr": "x_err_nm", "ynmerr": "y_err_nm", "zerr": "z_err_nm",
    "photerr": "photons_err", "iterations": "iterations",
    "xpix": "x_pix", "ypix": "y_pix", "PSFxpix": "sigma_pix",
}
SML_DROP = ("filenumber", "xnmf", "ynmf", "xpixf", "ypixf", "xpixerr", "ypixerr",
            "PSFypix")


def _h5_scalar(group, name):
    """A number or a MATLAB char array stored by ``save -v7.3``."""
    if name not in group:
        return None
    v = np.array(group[name])
    if v.dtype.kind == "u" and v.dtype.itemsize == 2:      # a char array
        return bytes(v.ravel().astype("uint8")).decode("latin-1").strip("\\x00")
    return v.ravel()


def _load_sml(path: Path) -> Tuple[Localizations, FileInfo]:
    import h5py
    columns, info = {}, {}
    try:
        f = h5py.File(path, "r")
    except OSError:
        f = None
    if f is not None:
        with f:
            loc = f["saveloc"]["loc"]
            for name in loc:
                if name in SML_DROP:
                    continue
                columns[SML_COLUMNS.get(name, name)] = np.array(loc[name]).ravel()
            file = f["saveloc"].get("file")
            if file is not None and "info" in file:
                # `file` may be a struct array: the first file's info then
                inf = file["info"]
                px = _h5_scalar(inf, "cam_pixelsize_um")
                if px is not None and px.dtype.kind == "f":
                    info["pixelsize_um"] = float(px[0])
                for k in ("numberOfFrames", "conversion", "offset", "emgain"):
                    v = _h5_scalar(inf, k)
                    if v is not None and v.dtype.kind == "f":
                        info[k] = float(v[0])
    else:                                                    # MATLAB v5-v7
        import scipy.io as sio
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        loc = m["saveloc"].loc
        for name in loc._fieldnames:
            if name in SML_DROP:
                continue
            columns[SML_COLUMNS.get(name, name)] = np.atleast_1d(getattr(loc, name))
        try:
            files = m["saveloc"].file
            first = files[0] if isinstance(files, np.ndarray) else files
            px = np.atleast_1d(first.info.cam_pixelsize_um)
            info["pixelsize_um"] = float(px[0])
        except AttributeError:
            pass
    if "frame" in columns:
        columns["frame"] = (np.asarray(columns["frame"]) - 1).astype(np.int64)  # MATLAB is 1-based
    for name in ("channel", "iterations"):
        if name in columns:
            columns[name] = np.asarray(columns[name]).astype(np.int32)
    for name, values in columns.items():
        if values.dtype == np.float64:
            columns[name] = values.astype(np.float32)
    px = info.get("pixelsize_um")
    locs = Localizations(columns, {"units": "nm", "source": str(path), "smap": info})
    return locs, FileInfo(path.name, str(path), "SMAP",
                          pixelsize_nm=px * 1000 if px else None, metadata=info)


register(Reader("SMAP", ("_sml.mat", ".mat"), _load_sml,
                lambda p: p.name.endswith("_sml.mat") or _has_saveloc(p)))


def _has_saveloc(path: Path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return "saveloc" in f
    except OSError:
        try:
            import scipy.io as sio
            return "saveloc" in sio.whosmat(path).__iter__().__next__()
        except Exception:
            return False


# ---------------------------------------------------------------- MINFLUX
MINFLUX_PSF_NM = 150.0     # for the precision proxy, SMAP's number


def _minflux_records(path: Path) -> np.ndarray:
    """The structured array Abberior's export holds: .npy, or a .zip of one/json."""
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
                return _minflux_from_json(json.loads(z.read(js[0])))
        raise ValueError(f"{path.name}: no .npy or .json inside")
    if path.suffix.lower() == ".json":
        return _minflux_from_json(json.loads(path.read_text()))
    raise ValueError(f"not a MINFLUX export: {path.name}")


def _minflux_from_json(records: list) -> np.ndarray:
    """The json export is a list of dicts with the same nesting as the npy."""
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


def _load_minflux(path: Path, valid_only: bool = True) -> Tuple[Localizations, FileInfo]:
    """Last iteration of every valid localization; positions m -> nm.

    ``frame`` is the localization's rank in time, since MINFLUX has no
    frames; ``time_s`` keeps the clock, ``tid`` the trace.  Photons are the
    last iteration's ``eco``; the precision is SMAP's proxy 150 / sqrt(N).
    """
    rec = _minflux_records(path)
    itr = rec["itr"]
    last = itr[:, -1] if itr.ndim == 2 else itr
    keep = np.ones(len(rec), dtype=bool)
    if valid_only and "vld" in rec.dtype.names:
        keep &= rec["vld"].astype(bool)
    loc = np.asarray(last["loc"], dtype=np.float64) * 1e9
    keep &= np.isfinite(loc[:, 0]) & np.isfinite(loc[:, 1])
    loc, last, rec = loc[keep], last[keep], rec[keep]
    order = np.argsort(rec["tim"], kind="stable") if "tim" in rec.dtype.names else np.arange(len(rec))
    loc, last, rec = loc[order], last[order], rec[order]
    n = len(rec)
    columns = {"x_nm": loc[:, 0].astype(np.float32), "y_nm": loc[:, 1].astype(np.float32),
               "frame": np.arange(n, dtype=np.int64)}
    if np.any(np.abs(loc[:, 2]) > 0):
        columns["z_nm"] = loc[:, 2].astype(np.float32)
    photons = (np.asarray(last["eco"], np.float32) if "eco" in last.dtype.names
               else np.full(n, np.nan, np.float32))
    columns["photons"] = photons
    columns["loc_precision_nm"] = (MINFLUX_PSF_NM / np.sqrt(np.maximum(photons, 1))).astype(np.float32)
    columns["sigma_nm"] = np.full(n, MINFLUX_PSF_NM, np.float32)
    for name in ("efo", "cfr", "dcr", "efc", "ecc", "fbg"):
        if name in last.dtype.names:
            columns[name] = np.asarray(last[name], np.float32)
    if "itr" in last.dtype.names:
        columns["iteration"] = np.asarray(last["itr"], np.int32)
    for name, dtype in (("tim", np.float64), ("tid", np.int64)):
        if name in rec.dtype.names:
            columns["time_s" if name == "tim" else name] = np.asarray(rec[name], dtype)
    locs = Localizations(columns, {"units": "nm", "source": str(path), "minflux": True})
    return locs, FileInfo(path.name, str(path), "MINFLUX", pixelsize_nm=100.0,
                          metadata={"n_raw": len(keep), "n_valid": n})


register(Reader("MINFLUX", (".npy", ".zip", ".json"), _load_minflux))


# -------------------------------------------------------------------- csv
# header names we recognise (lower-cased, units stripped) -> smappy column
CSV_NAMES = {
    "x": "x_nm", "y": "y_nm", "z": "z_nm", "xnm": "x_nm", "ynm": "y_nm", "znm": "z_nm",
    "x_nm": "x_nm", "y_nm": "y_nm", "z_nm": "z_nm", "frame": "frame", "t": "frame",
    "intensity": "photons", "photons": "photons", "phot": "photons", "n": "photons",
    "uncertainty": "loc_precision_nm", "uncertainty_xy": "loc_precision_nm",
    "locprecnm": "loc_precision_nm", "loc_precision_nm": "loc_precision_nm",
    "uncertainty_z": "loc_precision_z_nm", "locprecznm": "loc_precision_z_nm",
    "sigma": "sigma_nm", "sigma1": "sigma_nm", "sigma2": "sigma_y_nm", "psfxnm": "sigma_nm",
    "sigma_nm": "sigma_nm", "offset": "background", "bkgstd": "background_std",
    "bg": "background", "background": "background", "channel": "channel",
    "llrel": "logl_rel", "logl_rel": "logl_rel", "id": "id",
}
CSV_REQUIRED = ("x_nm", "y_nm")


def _clean(header: str) -> str:
    h = header.strip().strip('"').lower()
    unit = ""
    if "[" in h:
        h, unit = h.split("[", 1)
        unit = unit.rstrip("]").strip()
    return h.strip().replace(" ", "_"), unit


def csv_columns(path: Path) -> Tuple[List[str], List[str], bool]:
    """(headers, first data row, has_header) -- what a mapping dialog shows."""
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().rstrip("\\n")
        second = f.readline().rstrip("\\n")
    delimiter = ";" if first.count(";") > first.count(",") else ","
    cells = [c.strip() for c in first.split(delimiter)]
    has_header = not all(_is_number(c) for c in cells)
    row = [c.strip() for c in (second if has_header else first).split(delimiter)]
    headers = cells if has_header else [f"column {i + 1}" for i in range(len(cells))]
    return headers, row, has_header


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def guess_csv_mapping(headers: Sequence[str]) -> Dict[str, str]:
    """header -> smappy column, for the headers we recognise."""
    mapping = {}
    for h in headers:
        name, unit = _clean(h)
        target = CSV_NAMES.get(name)
        if target and target not in mapping.values():
            mapping[h] = target
    return mapping


def _load_csv(path: Path, mapping: Optional[Dict[str, str]] = None,
              units: str = "nm", pixelsize_nm: Optional[float] = None
              ) -> Tuple[Localizations, FileInfo]:
    """A delimited text table.  ``mapping`` is header -> smappy column; the
    recognised names (ThunderSTORM, SMAP exports, x/y/z/frame) need none.
    ``units`` "px" scales positions by ``pixelsize_nm``."""
    headers, _, has_header = csv_columns(path)
    delimiter = ";" if open(path, encoding="utf-8-sig").readline().count(";") > 0 else ","
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=1 if has_header else 0,
                         dtype=np.float64, encoding="utf-8-sig", invalid_raise=False)
    data = np.atleast_2d(data)
    if data.shape[1] != len(headers):
        data = data.reshape(-1, len(headers))
    mapping = dict(mapping or guess_csv_mapping(headers))
    missing = [c for c in CSV_REQUIRED if c not in mapping.values()]
    if missing:
        raise ValueError(f"{path.name}: no column for {', '.join(missing)}; "
                         f"headers are {headers}. Pass mapping={{header: column}}.")
    columns = {}
    for i, h in enumerate(headers):
        target = mapping.get(h)
        if target is None:
            continue
        values = data[:, i]
        if target == "frame":
            columns[target] = np.nan_to_num(values).astype(np.int64)
        elif target in ("channel", "id"):
            columns[target] = np.nan_to_num(values).astype(np.int32)
        else:
            columns[target] = values.astype(np.float32)
    if units == "px":
        if not pixelsize_nm:
            raise ValueError("positions in pixels need pixelsize_nm")
        for name in ("x_nm", "y_nm", "loc_precision_nm", "sigma_nm", "sigma_y_nm"):
            if name in columns:
                columns[name] = columns[name] * np.float32(pixelsize_nm)
    if "frame" not in columns:
        columns["frame"] = np.zeros(len(data), np.int64)
    ok = np.isfinite(columns["x_nm"]) & np.isfinite(columns["y_nm"])
    columns = {k: v[ok] for k, v in columns.items()}
    locs = Localizations(columns, {"units": "nm", "source": str(path)})
    return locs, FileInfo(path.name, str(path), "csv", pixelsize_nm=pixelsize_nm,
                          metadata={"mapping": mapping, "headers": headers})


register(Reader("csv", (".csv", ".txt", ".tsv"), _load_csv, needs=("mapping",)))
