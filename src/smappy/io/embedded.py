"""The calibration a table was fitted with, kept inside its localization file.

A fitted file used to record only the calibration's path, which pointed nowhere once the calibration was
moved, renamed or never copied with the data.  Now the file carries the calibration itself, in a group
`GROUP` laid out exactly as the calibration's own file (the same ``format`` attribute and datasets), so
that one reader serves both: `io.calibration.load_spline_calibration`,
`calibrate.dual.load_dual_color_calibration` and `fourpi.load_fourpi_calibration` take a localization file
and read its copy, and a fitter's *calibration* field takes a fitted file as the calibration of a new data
set (docs/insitu-fitting-plan.md §1).

What is copied is small, and exact:

* **the spline's knots, not its coefficients**, where the knots rebuild them bit for bit
  (`io.calibration.knots_rebuild`): 0.2 MB in place of 13 MB for a 26 × 26 × 80 grid.  A spline fitted
  otherwise -- SMAP's, whose ``psf`` is its bead stack -- keeps its coefficients;
* **the pupil model** the spline was drawn from (`calibrate.pupil.MODEL_GROUP`, ``uipsf``), as it is, so
  that a focus in the sample is drawn from the copy as from the original (`calibrate.depth.model_file`);
* a 4Pi model's knots, maps and layout, which is all its file holds anyway.

In memory the copy is an HDF5 image in bytes under the table's metadata key `KEY`: the table carries it
through whatever happens after the fit, and `io.hdf5.LocalizationWriter` writes it back as `GROUP`.  It is
never put in the JSON metadata.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

#: the localization file's group that holds the calibration
GROUP = "calibration"
#: the table's metadata key that holds it in memory, an HDF5 image in bytes
KEY = "calibration_h5"
#: the pupil model's group inside a calibration (`calibrate.pupil.MODEL_GROUP`, uipsf's ``EMBEDDED``)
MODEL = "uipsf"
#: uipsf's model-file format attribute (`uipsf.export.model_file.FORMAT`), read without importing uipsf
MODEL_FORMAT = "uipsf-model"
LOCALIZATIONS = "smappy-localizations"


def is_localizations(f) -> bool:
    """Whether the open HDF5 file ``f`` is a localization file."""
    return f.attrs.get("format") == LOCALIZATIONS


def calibration_group(f, path=None):
    """The calibration group of the open localization file ``f``, or a ValueError that says what the file
    recorded instead (a file fitted before calibrations were kept in it)."""
    import h5py
    group = f.get(GROUP)
    if isinstance(group, h5py.Group):
        return group
    import json
    name = Path(path).name if path is not None else "this localization file"
    recorded = None
    try:
        recorded = (json.loads(f.attrs["metadata"]).get("calibration") or {}).get("file")
    except Exception:
        pass
    raise ValueError(f"{name} carries no calibration" +
                     (f": it was fitted with {recorded}" if recorded else ""))


def carried_format(path) -> Optional[str]:
    """The format of the calibration the localization file at ``path`` carries (``smappy-spline-
    calibration``, ``smappy-dual-color-calibration``, ``smappy-4pi-calibration``), or None: not a
    localization file, or one without.  Read with h5py alone."""
    import h5py
    try:
        with h5py.File(Path(path), "r") as f:
            if not is_localizations(f):
                return None
            group = f.get(GROUP)
            return str(group.attrs.get("format")) if isinstance(group, h5py.Group) else None
    except (OSError, ValueError):
        return None


def model_group(path) -> Optional[str]:
    """Where in the HDF5 file at ``path`` a pupil model is: ``"uipsf"`` in a calibration that carries one,
    ``"calibration/uipsf"`` in a localization file's copy of one, None otherwise (a model file itself is
    read at its root, `uipsf.load_model` without a group).  Read with h5py alone."""
    import h5py
    try:
        with h5py.File(Path(path), "r") as f:
            for name in (MODEL, f"{GROUP}/{MODEL}"):
                group = f.get(name)
                if isinstance(group, h5py.Group) and group.attrs.get("format") == MODEL_FORMAT:
                    return name
    except (OSError, ValueError):
        pass
    return None


# ------------------------------------------------------------------ the copy, as bytes

def _copy_into(source, target) -> None:
    """Everything in the open HDF5 group ``source``, its attributes too, into the group ``target``."""
    target.attrs.update(dict(source.attrs))
    for name in source:
        source.copy(source[name], target, name=name)


#: the native calibration formats, copied group by group (`_copy_native`)
NATIVE = ("smappy-spline-calibration", "smappy-dual-color-calibration", "smappy-4pi-calibration")
#: what a native calibration holds that a fit does not need: the calibration run's own record
LEFT_OUT = ("diagnostics", MODEL)


def image(path) -> Optional[bytes]:
    """The calibration at ``path`` as the HDF5 image a localization file keeps (module docstring): a native
    spline, dual-colour or 4Pi calibration, a SMAP ``_3dcal.mat``, a uiPSF v1 4Pi model, or a localization
    file's own copy.  None for an empty path; an error for what is no calibration."""
    if not path:
        return None
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if _MADE.get("key") != key:
        _MADE.update(key=key, data=_made(path))
    return _MADE["data"]


#: the last image made, by file and time: a batch fits many data sets with one calibration, and a field-
#: dependent one takes seconds to read
_MADE: dict = {}


def _made(path: Path) -> bytes:
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            if is_localizations(f):
                return _image(lambda out: _copy_into(calibration_group(f, path), out))
            if f.attrs.get("format") in NATIVE:
                # copied as it is, never read through the loaders: a fit reads its calibration once
                return _image(lambda out: (_copy_native(f, out), _copy_model(path, out)))
        from ..fourpi import is_fourpi_calibration
        if is_fourpi_calibration(path):           # uiPSF v1's, made native
            from ..fourpi import load_fourpi_calibration, write_fourpi
            cal = load_fourpi_calibration(path)
            return _image(lambda out: write_fourpi(out, cal))
    from .calibration import load_spline_calibration, write_native
    cal = load_spline_calibration(path)            # a SMAP _3dcal.mat, turned to the camera's orientation
    return _image(lambda out: write_native(out, cal, knots_only=True))


def _copy_native(source, out) -> None:
    """A native calibration file's root ``source`` into ``out``, without what a fit does not need
    (`LEFT_OUT`), and each spline's coefficients replaced by its knots where they rebuild it exactly."""
    from .calibration import knots_rebuild
    out.attrs.update(dict(source.attrs))
    spline = "coeff" in source
    for name in source:
        if name in LEFT_OUT or spline and name == "coeff":
            continue
        if name in ("main", "secondary") and "coeff" in source[name]:
            _copy_native(source[name], out.create_group(name))
            continue
        if name == "field" and "coeff" in source[name]:
            _copy_field(source[name], out.create_group(name))
            continue
        source.copy(source[name], out, name=name)
    if spline:
        coeff = source["coeff"][...]
        psf = source["psf"][...] if "psf" in source else None
        if knots_rebuild(coeff, psf):
            out.attrs["coeff_from"] = "psf"
        else:
            out.create_dataset("coeff", data=coeff, compression="gzip")


#: a field-dependent calibration's node splines are on disk in half precision where their size allows
#: (`io.calibration.write_field_splines`), which moves the spline's values by ~4e-4 of its peak: the knots'
#: spline is checked by its values, at each cell's centre, against that rounding
HALF_TOLERANCE = 2e-3


def _centre_values(coeff):
    """A spline's values at the centre of each of its cells, from its coefficients ``(64, nz, ny, nx)``."""
    import numpy as np
    powers = np.array([i // 16 + (i // 4) % 4 + i % 4 for i in range(64)])
    return np.tensordot(0.5 ** powers, np.asarray(coeff, np.float64), axes=(0, 0))


def _copy_field(source, out) -> None:
    """A field-dependent calibration's node splines (``field``) into ``out``: each node's knots where they
    spline back to it -- its coefficients within float32 rounding, or for coefficients kept in half precision
    its values within that rounding -- 14 MB in place of 370 MB for a 7 × 7 grid; else the coefficients as
    they are."""
    import numpy as np

    from ..calibrate.core import spline_coefficients
    from .calibration import knots_rebuild, spline_knots
    for name in source:
        if name != "coeff":
            source.copy(source[name], out, name=name)
    out.attrs.update(dict(source.attrs))
    stored = source["coeff"]
    half = stored.dtype == np.float16
    knots = []
    for node in range(stored.shape[0]):
        coeff = stored[node].astype(np.float32)
        k = spline_knots(coeff)
        if half:
            scale = float(np.abs(k).max())
            good = scale == 0 or np.abs(_centre_values(spline_coefficients(k))
                                        - _centre_values(coeff)).max() <= HALF_TOLERANCE * scale
        else:
            good = knots_rebuild(coeff, k)
        if not good:
            source.copy(stored, out, name="coeff")
            return
        knots.append(k)
    out.create_dataset("knots", data=np.stack(knots), compression="gzip")


def _image(fill) -> bytes:
    import h5py
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as out:
        fill(out)
    return buffer.getvalue()


def _copy_model(path, out) -> None:
    """The pupil model of the calibration at ``path`` into ``out``'s `MODEL`: the one it carries, or the
    model file an older save put beside it (`calibrate.pupil.model_path`)."""
    import h5py
    path = Path(path)
    if not h5py.is_hdf5(path):
        return
    with h5py.File(path, "r") as f:
        inner = f.get(MODEL)
        if isinstance(inner, h5py.Group) and inner.attrs.get("format") == MODEL_FORMAT:
            f.copy(inner, out, name=MODEL)
            return
    from ..calibrate.pupil import model_path
    beside = model_path(path)
    if beside.is_file() and h5py.is_hdf5(beside):
        with h5py.File(beside, "r") as f:
            if f.attrs.get("format") == MODEL_FORMAT:
                _copy_into(f, out.create_group(MODEL))


def write(f, data: Optional[bytes]) -> None:
    """The image ``data`` (`image`) into the open localization file ``f`` as `GROUP`, replacing one there;
    nothing for None."""
    import h5py
    if GROUP in f:
        del f[GROUP]
    if not data:
        return
    with h5py.File(io.BytesIO(data), "r") as source:
        _copy_into(source, f.create_group(GROUP))


def read(f) -> Optional[bytes]:
    """The open localization file ``f``'s calibration as an image (`image`), or None."""
    import h5py
    group = f.get(GROUP)
    if not isinstance(group, h5py.Group):
        return None
    return _image(lambda out: _copy_into(group, out))


def describe(data: Optional[bytes]) -> Optional[str]:
    """One line on the image ``data``: what kind of calibration, and whether its pupil model is in it."""
    if not data:
        return None
    import h5py
    with h5py.File(io.BytesIO(data), "r") as f:
        kind = {"smappy-spline-calibration": "spline", "smappy-dual-color-calibration": "dual-colour",
                "smappy-4pi-calibration": "4Pi"}.get(str(f.attrs.get("format")), "unknown")
        model = MODEL in f
    return f"{kind} calibration{' with its pupil model' if model else ''}"


# ------------------------------------------------------------------ in JSON

def plain(metadata: Optional[dict]) -> Optional[dict]:
    """``metadata`` with its calibration (bytes) as base64, for JSON: a second file's metadata in the
    table's file list (`io.formats.FileInfo.to_dict`), a handed-over table (`handoff`)."""
    import base64
    data = (metadata or {}).get(KEY)
    if not isinstance(data, (bytes, bytearray)):
        return metadata
    out = dict(metadata)
    out[KEY] = {"base64": base64.b64encode(bytes(data)).decode("ascii")}
    return out


def unplain(metadata: Optional[dict]) -> Optional[dict]:
    """`plain` undone, in place."""
    import base64
    data = (metadata or {}).get(KEY)
    if isinstance(data, dict) and "base64" in data:
        metadata[KEY] = base64.b64decode(data["base64"])
    return metadata
