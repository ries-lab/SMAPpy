"""SMAP's two-channel calibrations read as SMAPpy's dual calibration (`dual.DualColorCalibration`).

SMAP's global calibration (``calibrate_globalworkflow.m``, *global 2 channel*) writes a
``_3dcal.mat`` with a spline per channel in ``SXY(1:2)`` and keeps the registration as a MATLAB
object (``interfaces.LocTransformN``) that scipy cannot read; DECODE-Plex's users export it beside
the calibration as ``_trafo.mat`` -- ``trafo_raw`` and ``trafo_inv_raw``, each a ``ref`` / ``tar``
pair of 3x3 matrices.  This reads the two, so that what SMAP calibrated -- DECODE-Plex's published
data among it -- can be fitted (Spline 3D 2C) and simulated (`simulate.camera.split_layout`) here
(docs/decode/plex-plan.md §6).

**The splines.**  Each ``SXY`` entry is a spline as SMAPpy's single reader reads one
(`io.calibration.smap_spline`): its ``coeff`` is the channel as the camera saw it (the global
calibration's ``coeffrawref`` / ``coeffrawtar``, the secondary channel not flipped back), un-mirrored
in x where SMAP built it from x-mirrored beads (``EMon``).  So the main one is ``main`` and the other
``secondary``, as `dual.build_dual_calibration` keeps them.  SMAP's averaged bead stacks are not the
splines' knots, which SMAPpy's format holds, so the knots are drawn from the splines.

**The registration.**  ``trafo_raw.tar`` maps a position in the secondary channel to the main one
(``trafo_inv_raw.tar`` the other way), as MATLAB's ``affine2d`` / ``projective2d`` do: a row vector
``[x y 1]`` times the matrix, then divided by its third element, with ``x`` along the columns.  Its
coordinates are SMAP's: pixels from 1, in units of 1000 pixels (DECODE-Plex reads it so, and its
offset maps then agree with its movies' own shift between the channels: plex-plan §9), in the bead
image as SMAP loaded it -- un-mirrored in x when it was taken with EM gain.  SMAPpy keeps the camera's
orientation and the chip's pixels from 0, so the matrix is moved through those: from 1 to 0, the
units, the EM mirror about the bead image's width, and the bead image's camera ROI (``roi``, x, y,
width, height) onto the chip.  It stays projective (SMAP's ``Tform``; DECODE-Plex drops the division,
which moves a point by 0.04 px at the centre of Fig. 2a's chip).

**The split** is SMAP's ``Tmode`` (``up-down mirror``, ``right-left mirror``, without the mirror) at
``Tsplitpos``, the main channel the half of ``SXY(1)``'s range; `dual.split_geometry` writes it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SCALE = 1e-3              # SMAP's registration works on pixels / 1000


def load_smap_dual_calibration(path, trafo=None):
    """``path``, a SMAP global two-channel ``_3dcal.mat``, with its registration ``trafo`` (default:
    the ``_trafo.mat`` beside it) as a `dual.DualColorCalibration` in SMAPpy's conventions."""
    import scipy.io

    from ..io.calibration import _as_struct, _smap_flipped, _to_scalar, smap_spline
    from .dual import DualColorCalibration, split_geometry

    path = Path(path)
    trafo = Path(trafo) if trafo is not None else _trafo_beside(path)
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    if "SXY" not in mat or "parameters_g" not in mat:
        raise ValueError(f"{path.name}: not a SMAP global two-channel calibration (no SXY and parameters_g)")
    entries = np.atleast_1d(mat["SXY"]).ravel()
    if len(entries) != 2:
        raise ValueError(f"{path.name}: {len(entries)} splines; a two-channel calibration has 2 "
                         "(a spatially resolved one is not read)")
    par_g = _fields(_as_struct(mat["parameters_g"]))
    pars = [_fields(_as_struct(mat[k])) if k in mat else par_g for k in ("parameters1", "parameters2")]

    splines, em = [], []
    for entry, par in zip(entries, pars, strict=True):
        em_on = _to_scalar(getattr(entry, "EMon", None))
        cal = smap_spline(_as_struct(entry.cspline), getattr(entry, "PSF", None), em_on, par, path)
        splines.append(_with_knots(cal, path))
        em.append(_smap_flipped(_to_scalar(par.get("emmirror")), em_on, path))
    if em[0] != em[1]:
        raise ValueError(f"{path.name}: one channel's beads were x-mirrored and the other's not")
    main, secondary = splines

    x0, y0, width, height = _roi(par_g, path)
    h = chip_transformation(_trafo_matrix(trafo), roi=(x0, y0, width, height), em_mirrored=em[0])

    mode = str(par_g.get("Tmode", "up-down mirror"))
    across = mode.startswith("right-left")
    if not (across or mode.startswith("up-down")):
        raise ValueError(f"{path.name}: Tmode {mode!r} (two cameras) is not read: one chip, split, is")
    first = entries[0]
    lower_bound = float(np.asarray(first.Xrange if across else first.Yrange, float).ravel()[0])
    main_channel = ("left" if across else "upper") if math.isinf(lower_bound) else ("right" if across else "lower")
    split = _to_scalar(par_g.get("Tsplitpos"))
    if split is not None:
        split = float(split)
        if across and em[0]:
            split = width - split            # SMAP's split position is in the un-mirrored image
        split += x0 if across else y0
    layout = ("right-left" if across else "up-down") + (" mirrored" if "mirror" in mode else "")
    geometry = split_geometry(layout, main_channel, (height, width), roi=(x0, y0, width, height),
                              split_position=split)
    geometry["sources"][0]["path"] = str(path)
    parameters = {
        "source": str(path), "trafo": str(trafo),
        "smappy_conversion": "SMAP global two-channel calibration read by calibrate.smap_dual",
        "smap": {"Tmode": mode, "Tform": str(par_g.get("Tform", "")), "Tsplitpos": _to_scalar(par_g.get("Tsplitpos")),
                 "emmirror": _scalar(par_g.get("emmirror")), "EMon": [bool(e) for e in em],
                 "roi": [x0, y0, width, height], "pixelsize_um": _list(par_g.get("pixelsize")),
                 "normf": [_scalar(getattr(_as_struct(e.cspline), "normf", None)) for e in entries],
                 "dz": _scalar(par_g.get("dz")), "modality": str(par_g.get("modality", ""))},
    }
    return DualColorCalibration(main, secondary, h, geometry, parameters)


def chip_transformation(trafo_raw: np.ndarray, roi, em_mirrored: bool) -> np.ndarray:
    """SMAP's ``trafo_raw.tar`` (row-vector, pixels from 1 / 1000, the un-mirrored bead image) as
    SMAPpy's 3x3: column vectors on the chip's pixels from 0, secondary to main, normalised."""
    x0, y0, width, _ = (float(v) for v in roi)
    to_smap = np.array([[SCALE, 0, SCALE], [0, SCALE, SCALE], [0, 0, 1]])   # pixel from 0 -> SMAP's
    h = np.linalg.inv(to_smap) @ np.asarray(trafo_raw, float).T @ to_smap
    if em_mirrored:
        mirror = np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], float)
        h = mirror @ h @ mirror
    corner = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
    h = corner @ h @ np.linalg.inv(corner)
    return h / h[2, 2]


def spline_knots(coeff: np.ndarray) -> np.ndarray:
    """The spline's values at its knots, (nz+1, ny+1, nx+1): each cell's cubic at its corners, the
    last knot along an axis from the last cell's far end -- what SMAPpy's format keeps as ``psf``."""
    c = np.asarray(coeff, np.float64)
    _, nz, ny, nx = c.shape
    c = c.reshape(4, 4, 4, nz, ny, nx)                       # pz, py, px, z, y, x

    def ends(n):
        cell = np.minimum(np.arange(n + 1), n - 1)
        return cell, (np.arange(n + 1) - cell).astype(float)  # 0 at a cell's start, 1 at the last end

    (cz, tz), (cy, ty), (cx, tx) = ends(nz), ends(ny), ends(nx)
    g = c[:, :, :, cz][:, :, :, :, cy][:, :, :, :, :, cx]     # pz, py, px, Z, Y, X
    pz = tz[None, :] ** np.arange(4)[:, None]
    py = ty[None, :] ** np.arange(4)[:, None]
    px = tx[None, :] ** np.arange(4)[:, None]
    return np.einsum("abcZYX,aZ,bY,cX->ZYX", g, pz, py, px).astype(np.float32)


def _with_knots(cal, path):
    from dataclasses import replace

    return replace(cal, psf=spline_knots(cal.coeff),
                   parameters={"source": str(path), "smappy_conversion": "an SXY entry of a SMAP global "
                               "two-channel calibration; psf: the spline's knots"})


def _trafo_beside(path: Path) -> Path:
    name = path.name
    for suffix in ("_3dcal.mat", "_3Dcal.mat"):
        if name.endswith(suffix):
            candidate = path.with_name(name[: -len(suffix)] + "_trafo.mat")
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"{path.name}: no _trafo.mat beside it; SMAP keeps the registration as a MATLAB "
                            "object, which has to be exported as trafo_raw / trafo_inv_raw (ref, tar)")


def _trafo_matrix(trafo: Path) -> np.ndarray:
    import scipy.io

    t = scipy.io.loadmat(trafo, struct_as_record=False, squeeze_me=True)
    raw = t["trafo_raw"]
    ref, tar = np.asarray(raw.ref, float), np.asarray(raw.tar, float)
    if not np.allclose(ref, np.eye(3)):
        raise ValueError(f"{trafo.name}: trafo_raw.ref is not the identity; only the main channel as "
                         "reference is read")
    if "trafo_inv_raw" in t:   # the other direction, fitted separately by SMAP: a check of the reading
        inv = np.asarray(t["trafo_inv_raw"].tar, float)
        product = tar @ inv
        if np.abs(product / product[2, 2] - np.eye(3)).max() > 0.01:
            raise ValueError(f"{trafo.name}: trafo_raw and trafo_inv_raw are not each other's inverse")
    return tar


def _roi(par_g: dict, path: Path):
    roi = par_g.get("roi")
    if roi is None:
        rois = []
    elif all(np.ndim(r) == 0 for r in np.atleast_1d(roi)):   # one bead file: the 4-vector itself
        rois = [np.asarray(roi, float).ravel()]
    else:                                                     # a cell array, a 4-vector per file
        rois = [np.asarray(r, float).ravel() for r in np.atleast_1d(roi)]
    rois = [r for r in rois if r.size == 4]
    if not rois:
        raise ValueError(f"{path.name}: the bead images' camera ROI (parameters_g.roi) is missing")
    if any(not np.array_equal(r, rois[0]) for r in rois):
        raise ValueError(f"{path.name}: the bead files have different camera ROIs")
    return tuple(int(v) for v in rois[0])


def _fields(struct) -> dict:
    return {name: getattr(struct, name) for name in getattr(struct, "_fieldnames", [])}


def _scalar(v):
    try:
        return float(np.asarray(v, float).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _list(v):
    try:
        a = np.asarray(v[0] if np.asarray(v).dtype == object else v, float).ravel()
        return a.tolist()
    except (TypeError, ValueError, IndexError):
        return None
