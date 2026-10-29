"""Localization file formats: one registry, one ``load`` for all of them.

Every reader turns a file into smappy's column set -- ``x_nm``, ``y_nm``,
``z_nm``, ``frame``, ``photons``, ``xy_err_nm``, ``sigma_nm``,
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
from ..metadata import pixel_sizes


# What a table's metadata says about the session rather than about the
# acquisition of one file: a file's own metadata is everything else
# (`own_metadata`).  Kept out of `FileInfo.metadata`, which would otherwise
# hold a saved session's history and file list inside each of its files.
SESSION_KEYS = ("files", "history", "roi", "roi_project", "id_regions", "session",
                "named_transformations", "gui", "bounds", "derived", "grouped")


def own_metadata(metadata: Optional[Dict]) -> Dict:
    return {k: v for k, v in (metadata or {}).items()
            if k not in SESSION_KEYS and not str(k).startswith("_")}


@dataclass
class FileInfo:
    name: str
    path: str
    format: str
    n: int = 0
    pixelsize_nm: Optional[float] = None
    # The file's own metadata -- its camera, its fit, a MINFLUX measurement's
    # sequence -- which `Session.add_file` fills from the file's table.  The
    # table's metadata is the first file's once several are open together;
    # this is where the second file's is kept, saved with the session
    # (``metadata["files"]``) and restored by the undo with the file list.
    metadata: Dict = field(default_factory=dict)
    # the file's frame time in ms (`metadata.frame_time_ms`), kept here and
    # not only in the table's camera metadata, which is the first file's
    # once several are open together
    frame_time_ms: Optional[float] = None
    # the columns the file had when it was read (`Session.add_file`).  Files
    # opened together are one table, in which a column one of them lacks is
    # padding in its rows -- NaN, or 0 for an integer column, which for an id
    # reads "in none" -- so this is what tells "file 2 has no z_nm" from
    # "file 2's z is not a number".  None: not known (a session saved before
    # it was kept).
    columns: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        out = {"name": self.name, "path": self.path, "format": self.format, "n": self.n,
               "pixelsize_nm": self.pixelsize_nm, "frame_time_ms": self.frame_time_ms}
        if self.columns is not None:
            out["columns"] = list(self.columns)
        own = own_metadata(self.metadata)
        if own:
            out["metadata"] = own
        return out

    @classmethod
    def from_dict(cls, d: Dict) -> "FileInfo":
        """Back from `to_dict`; keys an older or newer version wrote that
        this one does not have are left out rather than refused."""
        known = {"name", "path", "format", "pixelsize_nm", "metadata", "frame_time_ms",
                 "columns"}
        out = cls(**{k: v for k, v in d.items() if k in known})
        if out.columns is not None:
            out.columns = [str(c) for c in out.columns]
        return out


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


def load(path, reader: Optional[Reader] = None, **kwargs) -> Tuple[Localizations, FileInfo]:
    """Read any known format.  ``kwargs`` go to the reader (a csv mapping, say).

    ``reader`` forces one instead of choosing by the file name, which is how a
    `File/Load/<format>` plugin means what it says: without it a `.hdf5` opened
    as SMAP would quietly be read as smappy.
    """
    reader = reader or reader_for(path)
    locs, info = reader.load(Path(path), **kwargs)
    info.n = len(locs)
    return with_nm(locs, info.pixelsize_nm), info


def with_nm(locs: Localizations, pixelsize_nm: Optional[float] = None
            ) -> Localizations:
    """A table in pixels only, given nm columns as a fit's output has them.

    Everything downstream reads nm, so a file that has only `x_pix` / `y_pix`
    -- a SMAP file without ``xnm``, a csv in pixels -- is converted on the
    way in, with the file's pixel size or, failing that, the default one,
    which the table then records as assumed.
    """
    if "x_nm" in locs or "x_pix" not in locs:
        return locs
    from ..locs import to_nm
    from ..metadata import DEFAULT_PIXELSIZE_NM
    size = pixelsize_nm or locs.metadata.get("pixelsize_nm")
    return to_nm(locs, size or DEFAULT_PIXELSIZE_NM, assumed=not size)


@dataclass
class Writer:
    name: str
    suffixes: Tuple[str, ...]          # the first is the default
    save: Callable[..., Path]
    needs: Tuple[str, ...] = ()        # extra arguments a GUI must ask for


WRITERS: List[Writer] = []


def register_writer(writer: Writer) -> Writer:
    WRITERS.append(writer)
    return writer


def writer_for(path) -> Writer:
    name = Path(path).name.lower()
    for w in WRITERS:
        if any(name.endswith(s) for s in w.suffixes):
            return w
    raise ValueError(f"no writer for {Path(path).name}; known: "
                     + ", ".join(f"{w.name} ({' '.join(w.suffixes)})" for w in WRITERS))


def writer_named(name: str) -> Optional[Writer]:
    """A writer by its name: a suffix alone cannot say Picasso from smappy."""
    return next((w for w in WRITERS if w.name == name), None)


def writer_names() -> List[str]:
    return [w.name for w in WRITERS]


def save(locs, path, metadata=None, **kwargs) -> Path:
    """Write a table in whichever format the name asks for."""
    return writer_for(path).save(locs, Path(path), metadata=metadata, **kwargs)


def name_filter() -> str:
    """A Qt file-dialog filter over every reader."""
    every = " ".join(f"*{s}" for r in READERS for s in r.suffixes)
    parts = [f"Localizations ({every})"]
    parts += [f"{r.name} ({' '.join('*' + s for s in r.suffixes)})" for r in READERS]
    return ";;".join(parts + ["All files (*)"])


def save_filter() -> str:
    """The same, over every writer."""
    parts = [f"{w.name} ({' '.join('*' + s for s in w.suffixes)})" for w in WRITERS]
    return ";;".join(parts + ["All files (*)"])


def reader_names() -> List[str]:
    """For a format dropdown; the order the readers were registered in."""
    return [r.name for r in READERS]


def reader_named(name: str) -> Optional[Reader]:
    return next((r for r in READERS if r.name == name), None)


# ------------------------------------------------------------------ smappy
def _load_smappy(path: Path) -> Tuple[Localizations, FileInfo]:
    from .hdf5 import load_localizations
    locs = load_localizations(path)
    cam = (locs.metadata.get("camera") or {})
    px = pixel_sizes(cam.get("pixelsize_um"))
    px = px and px[0]
    return locs, FileInfo(path.name, str(path), "smappy",
                          pixelsize_nm=px * 1000 if px else None, metadata=locs.metadata)


def _is_smappy(path: Path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            # a group of columns; Picasso's /locs is one compound dataset
            return isinstance(f.get("locs"), h5py.Group)
    except OSError:
        return False


register(Reader("smappy HDF5", (".hdf5", ".h5"), _load_smappy, _is_smappy))


def _save_smappy(locs, path: Path, metadata=None) -> Path:
    from .hdf5 import save_localizations
    return save_localizations(path, locs, metadata)


register_writer(Writer("smappy HDF5", (".hdf5", ".h5"), _save_smappy))


# --------------------------------------------------------------- SMAP sml
SML_COLUMNS = {
    "xnm": "x_nm", "ynm": "y_nm", "znm": "z_nm", "frame": "frame",
    "phot": "photons", "bg": "background", "locprecnm": "xy_err_nm",
    "locprecznm": "z_err_nm", "PSFxnm": "sigma_nm", "PSFynm": "sigma_y_nm",
    "LLrel": "logl_rel", "logLikelihood": "logl", "channel": "channel",
    "xnmerr": "x_err_nm", "ynmerr": "y_err_nm", "zerr": "z_err_nm",
    "photerr": "photons_err", "iterations": "iterations",
    "xpix": "x_pix", "ypix": "y_pix", "PSFxpix": "sigma_pix",
}
# Columns to skip.  This is the wrong shape and is kept until the thing that
# replaces it exists: a per-format set of columns to *keep*, chosen in a
# settings dialog and remembered, rather than a hard-coded list of names to
# drop.  It matters more than it looks -- of the 30 columns read from a real
# `_sml.mat`, 13 are read by anything downstream, and opening a 57 M
# localization one peaks at 20 GB, which is more memory than a microscope PC
# has.  See "Loading only the columns that are used" in NOTES.md.
SML_DROP = ("filenumber", "xnmf", "ynmf", "xpixf", "ypixf", "xpixerr", "ypixerr",
            "PSFypix")


def _h5_scalar(group, name):
    """A number or a MATLAB char array stored by ``save -v7.3``."""
    if name not in group:
        return None
    v = np.array(group[name])
    if v.dtype.kind == "u" and v.dtype.itemsize == 2:      # a char array
        return bytes(v.ravel().astype("uint8")).decode("latin-1").strip("\x00")
    return v.ravel()


def _superseded(name: str, names) -> bool:
    """Whether another field of the same file wins the column ``name`` maps to.

    ``zerr`` is the fit's axial CRLB and ``locprecznm`` the axial precision
    SMAP derives from it; both map to `z_err_nm`, and read in file order the
    later one silently replaced the other.  ``locprecznm`` is the one SMAP's
    analyses read, so it wins.
    """
    return name == "zerr" and "locprecznm" in names


def _load_sml(path: Path) -> Tuple[Localizations, FileInfo]:
    import h5py
    columns, info, sites = {}, {}, None
    try:
        f = h5py.File(path, "r")
    except OSError:
        f = None
    if f is not None:
        with f:
            loc = f["saveloc"]["loc"]
            for name in loc:
                if name in SML_DROP or _superseded(name, loc):
                    continue
                columns[SML_COLUMNS.get(name, name)] = np.array(loc[name]).ravel()
            sites = _sml_sites(f)
            file = f["saveloc"].get("file")
            if file is not None and "info" in file:
                # `file` may be a struct array: the first file's info then
                inf = file["info"]
                info = _sml_info(lambda k: _h5_scalar(inf, k))
    else:                                                    # MATLAB v5-v7
        import scipy.io as sio
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        loc = m["saveloc"].loc
        for name in loc._fieldnames:
            if name in SML_DROP or _superseded(name, loc._fieldnames):
                continue
            columns[SML_COLUMNS.get(name, name)] = np.atleast_1d(getattr(loc, name))
        try:
            files = m["saveloc"].file
            first = files[0] if isinstance(files, np.ndarray) else files
            info = _sml_info(lambda k: np.atleast_1d(getattr(first.info, k))
                             if hasattr(first.info, k) else None)
        except (AttributeError, TypeError, ValueError):
            pass
    if "frame" in columns:
        columns["frame"] = (np.asarray(columns["frame"]) - 1).astype(np.int64)  # MATLAB is 1-based
    for name in ("channel", "iterations"):
        if name in columns:
            columns[name] = np.asarray(columns[name]).astype(np.int32)
    for name, values in columns.items():
        if values.dtype == np.float64:
            columns[name] = values.astype(np.float32)
    px = pixel_sizes(info.get("pixelsize_um"))
    px = px and px[0]                   # the file's one number: its x pixel
    metadata = {"units": "nm", "source": str(path), "smap": info}
    camera = _sml_camera(info)
    if camera:
        metadata["camera"] = camera
    if sites:
        metadata["roi_project"] = sites
    locs = Localizations(columns, metadata)
    return locs, FileInfo(path.name, str(path), "SMAP",
                          pixelsize_nm=px * 1000 if px else None, metadata=info)


def _sml_info(get: Callable) -> Dict:
    """The camera's numbers of a SMAP file's ``file.info``; ``get(name)`` is
    the field as an array, or None (or a string) where it is missing."""
    info = {}

    def numbers(name):
        v = get(name)
        if isinstance(v, np.ndarray) and v.size and v.dtype.kind in "fiub":
            return v.ravel()
        return None

    px = numbers("cam_pixelsize_um")
    if px is not None:
        info["pixelsize_um"] = _sml_pixel(px)
    for k in ("numberOfFrames", "conversion", "offset", "emgain",
              "exposure", "timediff"):
        v = numbers(k)
        if v is not None:
            info[k] = float(v[0])
    em = numbers("EMon")
    if em is not None:
        info["EMon"] = bool(em[0])
    roi = numbers("roi")
    if roi is not None and roi.size == 4:
        info["roi"] = [int(v) for v in roi]     # x, y on the chip, w, h
    return info


def _sml_pixel(px: np.ndarray):
    """SMAP's ``cam_pixelsize_um``: one number, or x and y where they differ
    -- they do on some setups (117 x 127 nm), and SMAP scaled `ynm` by the
    second."""
    values = [float(v) for v in np.ravel(px)[:2]]
    return values[0] if len(values) == 1 or values[0] == values[1] else values


def _sml_camera(info: Dict) -> Dict:
    """A SMAP file's camera as smappy keeps it: where a fit of smappy's
    records it, so that what moves `x_pix` as well as `x_nm` (drift) finds
    the pixel size, *File properties* shows the conversion, and the chip
    region places the image (SMAP's `xnm` is ``(xpix + roi x) * pixel``, as
    smappy's).

    SMAP's ``timediff`` is the frame time it measured from the timestamps of
    the first and last frames -- or 1, a placeholder some of its loaders
    write, or the exposure.  A frame cannot be shorter than its exposure, so
    a ``timediff`` below it is the placeholder and is dropped.
    """
    camera = {}
    pixel = pixel_sizes(info.get("pixelsize_um"))
    if pixel is not None and min(pixel) > 0:
        camera["pixelsize_um"] = info["pixelsize_um"]
    for smap, ours in (("conversion", "conversion"), ("offset", "offset"),
                       ("emgain", "emgain"), ("EMon", "em_on"), ("roi", "roi")):
        if info.get(smap) is not None:
            camera[ours] = info[smap]
    exposure = info.get("exposure")
    if exposure is not None and exposure > 0:
        camera["exposure_ms"] = float(exposure)
    cycle = info.get("timediff")
    if cycle is not None and cycle > 0 and cycle >= camera.get("exposure_ms", np.inf):
        camera["frame_time_ms"] = float(cycle)
    return camera


# SMAP's site lines, by the names an ROI's lines take here: ``rotationpos``
# is the line the site image is turned along, which is what `direction` is
SML_LINES = (("rotationpos", "direction"), ("line1", "line1"), ("line2", "line2"))
SML_LISTS = ("list1", "list2", "list3", "list4")
DIRECTION_LINE = "direction"            # `roi_manager.core.DIRECTION`


def _sml_sites(f) -> Optional[dict]:
    """The ROI manager's sites of a v7.3 `_sml.mat`, as an ROI project
    (`ROIProject.from_dict`), or None.  SMAP keeps them as objects, which
    `io.mcos` reads; a v5-v7 file keeps them in a store that is not read."""
    from .mcos import read_objects
    sites = read_objects(f, "saveloc/siteexplorer/sites")
    return sml_sites_project(sites) if sites else None


def sml_sites_project(sites: List[dict]) -> Optional[dict]:
    """SMAP's sites -- each a dict of an ``SEsites`` object's properties --
    as an ROI project of one file.

    A site is an ROI at its ``pos`` (nm), numbered by its SMAP ID, with
    *use* and the comment.  Its lines (µm in SMAP) become named lines in nm:
    the orientation line ``rotationpos`` is `direction`, ``line1`` and
    ``line2`` keep their names; a line never drawn is all zeros and is left
    out.  ``line3``, the shape drawn by hand inside the site, is the ROI's
    polygon: a polygon or free-hand outline as drawn, a rectangle
    (``[x y w h]``) by its corners, an ellipse (``[cx cy a b angle]``) as
    64 vertices; a polyline is not a shape and is left out.  SMAP stores
    that shape as it was drawn on the site image, which it turns along the
    orientation line when *rotate* is on (a setting the file does not keep),
    so for such a site the polygon is turned by that angle.  The keyword
    lists ``list1``-``list4`` become annotations with the keywords SMAP
    offered; a list of nothing but ``empty`` is not one.  All sites go to
    the first file, as the localizations do.  The ROI size is not in the
    file (SMAP keeps it in its GUI's settings), so the project's stays.
    """
    rois, lists, numbers = [], {}, []
    for site in sites:
        pos = _numbers(site.get("pos"))
        if pos is None or len(pos) < 2:
            continue
        note = site.get("annotation") or {}
        lines = {}
        for smap, name in SML_LINES:
            line = _numbers((note.get(smap) or {}).get("pos"))
            if line is not None and line.size == 4 and np.any(line):
                lines[name] = (line.reshape(2, 2) * 1000).tolist()
        annotations = {}
        for name in SML_LISTS:
            entry = note.get(name) or {}
            words = entry.get("string")
            words = [words] if isinstance(words, str) else list(words or [])
            words = [str(w).strip() for w in words]
            if not words or words == ["empty"]:
                continue
            lists.setdefault(name, words)
            value = _numbers(entry.get("value"))
            if value is not None and value.size and 1 <= int(value[0]) <= len(words):
                annotations[name] = words[int(value[0]) - 1]
        ident = _numbers(site.get("ID"))
        number = int(ident[0]) if ident is not None and ident.size else len(rois) + 1
        numbers.append(number)
        use = _numbers(note.get("use"))
        comment = note.get("comments")
        rois.append({"file_id": "file0", "center": [float(pos[0]), float(pos[1])],
                     "number": number, "polygon": _sml_shape(note.get("line3")),
                     "lines": lines, "use": bool(use[0]) if use is not None and use.size else True,
                     "comment": comment if isinstance(comment, str) else "",
                     "origin": {"method": "SMAP", "name": str(site.get("name") or "")},
                     "annotations": annotations})
    if not rois:
        return None
    names = [DIRECTION_LINE] + [n for _, n in SML_LINES[1:]
                                if any(n in r["lines"] for r in rois)]
    project = {"format_version": 1, "sources": [{"id": "file0", "number": 0, "name": ""}],
               "rois": rois, "next_number": max(numbers) + 1, "line_names": names}
    if lists:
        project["annotation_lists"] = lists
    return project


def _numbers(value) -> Optional[np.ndarray]:
    """A MATLAB number read by `io.mcos` as a flat float array, or None."""
    if value is None or isinstance(value, (str, dict, list)):
        return None
    try:
        return np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None


def _sml_shape(entry) -> Optional[list]:
    """SMAP's ``line3`` as polygon vertices in nm, or None (see
    `sml_sites_project`)."""
    if not isinstance(entry, dict) or entry.get("pos") is None:
        return None
    pos = np.asarray(entry["pos"], dtype=float) * 1000
    if pos.ndim == 2 and pos.shape[1] == 2 and len(pos) >= 3:
        if "polyline" in _roi_function(entry):
            return None
        return pos.tolist()
    flat = pos.ravel()
    if flat.size == 4:                                     # x, y, width, height
        x, y, w, h = flat
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    if flat.size == 5:                                     # centre, semi-axes, angle
        cx, cy, a, b = flat[:4]
        turn = np.deg2rad(flat[4] / 1000)                  # the angle was not in µm
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        x, y = a * np.cos(t), b * np.sin(t)
        return np.column_stack([cx + x * np.cos(turn) - y * np.sin(turn),
                                cy + x * np.sin(turn) + y * np.cos(turn)]).tolist()
    return None


def _roi_function(entry) -> str:
    """The name of the MATLAB function that drew a ``line3``, or ''."""
    found = entry.get("roifun")
    for item in found if isinstance(found, list) else [found]:
        if isinstance(item, dict):
            handle = item.get("function_handle") or {}
            if isinstance(handle, dict) and isinstance(handle.get("function"), str):
                return handle["function"]
    return ""


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
# the records and both of Imspector's layouts: `io.minflux`; the `.msr`
# container with its images and beads: `io.msr`

def _load_minflux(path: Path, valid_only: bool = True, z_scale: float = 1.0):
    from .minflux import load as load_minflux
    return load_minflux(path, valid_only, z_scale)


def _load_msr(path: Path, measurements=None, valid_only: bool = True,
              z_scale: float = 1.0, beads: bool = False):
    from .msr import load as load_msr
    return load_msr(path, measurements, valid_only, z_scale, beads=beads)


register(Reader("MINFLUX", (".npy", ".zip", ".json"), _load_minflux))
# `needs`: the GUI asks which measurements and images (`MsrImportDialog`)
register(Reader("Abberior MSR", (".msr",), _load_msr, needs=("measurements",)))


# -------------------------------------------------------------------- csv
# header names we recognise (lower-cased, units stripped) -> smappy column
CSV_NAMES = {
    "x": "x_nm", "y": "y_nm", "z": "z_nm", "xnm": "x_nm", "ynm": "y_nm", "znm": "z_nm",
    "x_nm": "x_nm", "y_nm": "y_nm", "z_nm": "z_nm", "frame": "frame", "t": "frame",
    "intensity": "photons", "photons": "photons", "phot": "photons", "n": "photons",
    "uncertainty": "xy_err_nm", "uncertainty_xy": "xy_err_nm",
    "locprecnm": "xy_err_nm", "xy_err_nm": "xy_err_nm",
    "uncertainty_z": "z_err_nm", "locprecznm": "z_err_nm", "z_err_nm": "z_err_nm",
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


def csv_delimiter(line: str) -> str:
    """The separator of a delimited line: tab, semicolon or comma, the most
    frequent.  One rule for the reader and the mapping dialog, which used to
    disagree -- and neither split a .tsv."""
    counts = {d: line.count(d) for d in ("\t", ";", ",")}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] else ","


def csv_columns(path: Path) -> Tuple[List[str], List[str], bool]:
    """(headers, first data row, has_header) -- what a mapping dialog shows."""
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().rstrip("\r\n")
        second = f.readline().rstrip("\r\n")
    delimiter = csv_delimiter(first)
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


#: length units a delimited file may be written in, and what one is in nm;
#: "px" is the camera pixel, given separately
LENGTH_UNITS = {"nm": 1.0, "um": 1e3, "mm": 1e6, "px": None}
#: the lengths a mapping can name: lateral ones are in pixels when the file
#: is, axial ones never are (a pixel has no depth; Picasso and DECODE write z
#: in nm whatever x and y are in)
LATERAL_LENGTHS = ("x_nm", "y_nm", "xy_err_nm", "x_err_nm", "y_err_nm",
                   "sigma_nm", "sigma_y_nm")
AXIAL_LENGTHS = ("z_nm", "z_err_nm")
INTEGER_COLUMNS = ("channel", "id")


def column_name(header: str, index: int = 0) -> str:
    """A header as a column name: units and anything not a word character
    dropped, ``"chi2"`` from ``"chi2"`` and ``"bkgstd"`` from ``"bkgstd [photon]"``.
    """
    import re
    name, _ = _clean(header)
    name = re.sub(r"\W+", "_", name).strip("_")
    return name or f"column_{index + 1}"


def _whole(values: np.ndarray) -> bool:
    """Whether every value is a whole number that fits a 32-bit integer."""
    return bool(len(values)) and bool(np.all(np.isfinite(values))) \
        and bool(np.all(values == np.round(values))) and bool(np.all(np.abs(values) < 2**31))


def _load_csv(path: Path, mapping: Optional[Dict[str, str]] = None,
              units: str = "nm", pixelsize_nm: Optional[float] = None,
              first_frame: Optional[int] = None, conversion: Optional[str] = None
              ) -> Tuple[Localizations, FileInfo]:
    """A delimited text table, every numeric column of it.

    ``mapping`` is header -> smappy column for the ones that mean something
    to smappy (positions, frame, photons, ...); the recognised names
    (ThunderSTORM, SMAP exports, smappy's own) need none.  Every other column
    is read too, under its header as a name (`column_name`): what a column
    means is the user's business, and an export read back must have what it
    was written with.

    ``units`` is what the mapped lengths are in -- nm, um, mm, or px, which
    needs ``pixelsize_nm`` and scales the lateral ones only.  ``first_frame``
    is the number of the first frame in the file (ThunderSTORM and SMAP count
    from 1, smappy from 0).  Without them, a saved `conversion` whose headers
    the file has supplies them (`io.conversions`), and ``conversion`` names
    one to use whatever the headers.
    """
    from .conversions import conversion_for, conversion_named
    headers, _, has_header = csv_columns(path)
    conv = conversion_named(conversion) if conversion else conversion_for(headers)
    if conv is not None:
        mapping = mapping if mapping is not None else (dict(conv.mapping) or None)
        if units == "nm" and conv.units != "nm":
            units = conv.units
        pixelsize_nm = pixelsize_nm or conv.pixelsize_nm
        first_frame = conv.first_frame if first_frame is None else first_frame
    with open(path, encoding="utf-8-sig") as f:
        delimiter = csv_delimiter(f.readline())
    data = np.genfromtxt(path, delimiter=delimiter, skip_header=1 if has_header else 0,
                         dtype=np.float64, encoding="utf-8-sig", invalid_raise=False)
    data = np.atleast_2d(data)
    if data.shape[1] != len(headers):
        data = data.reshape(-1, len(headers))
    mapping = dict(guess_csv_mapping(headers), **(mapping or {}))
    mapping = {h: t for h, t in mapping.items() if t}
    missing = [c for c in CSV_REQUIRED if c not in mapping.values()]
    if missing:
        raise ValueError(f"{path.name}: no column for {', '.join(missing)}; "
                         f"headers are {headers}. Pass mapping={{header: column}}.")
    if units not in LENGTH_UNITS:
        raise ValueError(f"units must be one of {', '.join(LENGTH_UNITS)}, not {units!r}")
    if units == "px" and not pixelsize_nm:
        raise ValueError("positions in pixels need pixelsize_nm")
    lateral = pixelsize_nm if units == "px" else LENGTH_UNITS[units]
    axial = 1.0 if units == "px" else LENGTH_UNITS[units]
    columns = {}
    for i, h in enumerate(headers):
        values = data[:, i]
        target = mapping.get(h)
        if target is None:
            if not np.isfinite(values).any():
                continue                         # text, or empty
            target = column_name(h, i)
            while target in columns or target in mapping.values():
                target += "_"
        if target == "frame":
            columns[target] = (np.nan_to_num(values) - (first_frame or 0)).astype(np.int64)
        elif target in INTEGER_COLUMNS or (h not in mapping and _whole(values)):
            # a column of whole numbers is an id, a count or a flag -- an id
            # column is what a region is (`docs/regions.md`), and one read
            # back as floats would not be one
            columns[target] = np.nan_to_num(values).astype(np.int32)
        else:
            scale = (lateral if target in LATERAL_LENGTHS
                     else axial if target in AXIAL_LENGTHS else 1.0)
            columns[target] = (values * scale).astype(np.float32)
    if "frame" not in columns:
        columns["frame"] = np.zeros(len(data), np.int64)
    ok = np.isfinite(columns["x_nm"]) & np.isfinite(columns["y_nm"])
    columns = {k: v[ok] for k, v in columns.items()}
    locs = Localizations(columns, {"units": "nm", "source": str(path)})
    return locs, FileInfo(path.name, str(path), "csv",
                          pixelsize_nm=pixelsize_nm if units == "px" else None,
                          metadata={"mapping": mapping, "headers": headers,
                                    "units": units, "first_frame": first_frame or 0,
                                    "conversion": conv.name if conv else None})


#: smappy's columns under ThunderSTORM's headers, which more programs read
#: than any other text format; ThunderSTORM counts frames from 1
THUNDERSTORM_NAMES = {
    "frame": "frame", "x_nm": "x [nm]", "y_nm": "y [nm]", "z_nm": "z [nm]",
    "sigma_nm": "sigma [nm]", "photons": "intensity [photon]",
    "background": "offset [photon]", "xy_err_nm": "uncertainty_xy [nm]",
    "z_err_nm": "uncertainty_z [nm]",
}


def _save_csv(locs, path: Path, metadata=None, columns: Optional[Sequence[str]] = None,
              names: str = "smappy") -> Path:
    """Every column, or ``columns``, as delimited text: a comma for .csv, a
    tab for .tsv and .txt.  ``names`` "smappy" writes smappy's own names and
    frames from 0; "ThunderSTORM" writes its headers and frames from 1, and
    the other columns under their own names."""
    columns = [c for c in (columns or locs.keys()) if c in locs]
    if not columns:
        raise ValueError("no columns to write")
    thunder = names.lower() == "thunderstorm"
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    headers, arrays, formats = [], [], []
    for name in columns:
        values = np.asarray(locs[name])
        if values.ndim != 1:
            continue                              # an array per row: not a column
        if thunder:
            headers.append(THUNDERSTORM_NAMES.get(name, name))
            if name == "frame":
                values = values + 1
        else:
            headers.append(name)
        integer = values.dtype.kind in "iub"
        arrays.append(values.astype(np.int64) if integer else values.astype(np.float64))
        formats.append("%d" if integer else "%.9g")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(delimiter.join(headers) + "\n")
        step = 200_000                            # bounded memory for any size
        for start in range(0, len(locs), step):
            # float64 throughout: "%d" prints an integer stored as a float
            # exactly up to 2^53, which no frame or id comes near
            block = np.column_stack([a[start:start + step].astype(np.float64)
                                     for a in arrays])
            np.savetxt(f, block, fmt=formats, delimiter=delimiter)
    return path


register(Reader("csv", (".csv", ".txt", ".tsv"), _load_csv, needs=("mapping",)))
register_writer(Writer("csv", (".csv", ".tsv", ".txt"), _save_csv))


# ------------------------------------------------- Picasso and DECODE (HDF5)

def _load_picasso(path: Path):
    from .picasso import load as load_picasso
    return load_picasso(path)


def _is_picasso(path: Path) -> bool:
    from .picasso import is_picasso
    return is_picasso(path)


def _save_picasso(locs, path: Path, metadata=None, **kwargs) -> Path:
    from .picasso import save as save_picasso
    return save_picasso(locs, path, metadata, **kwargs)


def _load_decode(path: Path):
    from .decode import load as load_decode
    return load_decode(path)


def _is_decode(path: Path) -> bool:
    from .decode import is_decode
    return is_decode(path)


register(Reader("Picasso", (".hdf5", ".h5"), _load_picasso, _is_picasso))
register(Reader("DECODE", (".h5", ".hdf5"), _load_decode, _is_decode))
register_writer(Writer("Picasso", (".hdf5", ".h5"), _save_picasso))
