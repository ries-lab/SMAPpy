"""Native HDF5 and SMAP ``*_3Dcal.mat`` cubic-spline PSF calibration files.

This is the one module that has to understand MATLAB's array conventions.
Everything downstream works in a single, explicit convention:

* images are ``(y, x)`` C-contiguous, as they come out of a TIFF reader
* the spline coefficient array is stored as ``(64, nz, ny, nx)`` C-contiguous,
  so the flat index used by the fitter is ``((i*nz + z)*ny + y)*nx + x``
* fit parameters are ordered ``(x, y, photons, background, z)``

In the MATLAB fitter the equivalent array is ``(n1, n2, nz, 64)`` in
column-major order and the first axis -- called ``spline_xsize`` in the C
sources -- is the *first image axis*, i.e. rows, i.e. ``y``.  That is why
MATLAB's ``P(:,1)`` is y and ``P(:,2)`` is x.  We swap the two spatial axes
once, here, so that the rest of the code can call x "x".

The 64 coefficients per grid point are ordered ``i = 16*pz + 4*py + px``
(powers of the z, y, x fractional offsets), matching ``kernel_computeDelta3D``
in ``CPUsplineLib.cpp``; that ordering is preserved by the axis swap because
we swap the corresponding coefficient axes too.

SMAP builds a spline from bead images it has flipped in x when they were
taken with EM gain (``readbeadimages``; its loader flips the data the same
way).  SMAPpy never flips the data, so such a spline is turned into the
camera's orientation here, on load (`_unmirror_x`), and nothing downstream --
the fitter, the simulator, a saved ``.h5`` -- knows that mirroring exists.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from math import comb
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io


@dataclass
class SplineCalibration:
    """A cubic-spline PSF model in the layout consumed by the SMAPpy fitter."""

    coeff: np.ndarray  # (64, nz, ny, nx) float32, C-contiguous
    dz: float  # z spacing of the calibration stack, nm
    z0: float  # index of the focal plane in the spline z grid
    x0: Optional[float] = None  # lateral centre of the calibration ROI, pixels
    psf: Optional[np.ndarray] = None  # (nz, ny, nx) averaged bead stack, if present
    em_on: Optional[bool] = None  # EM gain used for the bead stack
    pixelsize_um: Optional[np.ndarray] = None
    source: Optional[Path] = None
    parameters: dict = field(default_factory=dict)  # raw calibration parameters

    @property
    def shape(self) -> tuple:
        """Spatial shape of the spline grid, ``(nz, ny, nx)``."""
        return self.coeff.shape[1:]

    def z_index_to_nm(self, z_index: np.ndarray) -> np.ndarray:
        """Convert a spline z index to nm relative to the focal plane."""
        return -(np.asarray(z_index) - self.z0) * self.dz

    def z_nm_to_index(self, z_nm: np.ndarray) -> np.ndarray:
        return -np.asarray(z_nm) / self.dz + self.z0


def _as_struct(obj):
    """Unwrap the 0-d object arrays scipy hands back for MATLAB structs."""
    while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        obj = obj.item()
    return obj


def _first(obj):
    """First element of a MATLAB struct array (SXY may be 2-D for spatial cal)."""
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size > 1:
        raise NotImplementedError(
            f"spatially varying calibration with {obj.shape} entries is not "
            "supported; only single-channel, single-region calibrations are"
        )
    return _as_struct(obj)


def _to_scalar(v, default=None):
    if v is None:
        return default
    v = np.asarray(v)
    return default if v.size == 0 else v.reshape(-1)[0]


def load_spline_calibration(path) -> SplineCalibration:
    """Load a native SMAPpy HDF5 calibration or SMAP MATLAB v7 calibration.

    What comes back is always in the camera's orientation: a SMAP spline of
    x-mirrored beads is converted (`_unmirror_x`).  Converting a ``_3dcal.mat``
    to ``.h5`` is therefore ``save_spline_calibration(h5,
    load_spline_calibration(mat))``.
    """
    path = Path(path)
    import h5py
    if h5py.is_hdf5(path):
        return _load_native_calibration(path)
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)

    params = mat.get("parameters")
    if "SXY" in mat:
        entry = _first(mat["SXY"])
        cspline = _as_struct(entry.cspline)
        psf = getattr(entry, "PSF", None)
        em_on = _to_scalar(getattr(entry, "EMon", None))
    elif "cspline" in mat:
        # older flat layout written by calibrate3D_g for single-region fits
        entry = None
        cspline = _as_struct(mat["cspline"])
        psf = mat.get("PSF")
        em_on = None
    else:
        raise ValueError(f"{path.name}: no 'SXY' or 'cspline' found; not a 3D "
                         "calibration file")

    coeff = _as_struct(cspline.coeff)
    if isinstance(coeff, np.ndarray) and coeff.dtype == object:
        coeff = coeff.flat[0]  # cell array: single channel -> first cell
    coeff = np.asarray(coeff, dtype=np.float32)
    if coeff.ndim != 4 or coeff.shape[3] != 64:
        raise ValueError(f"{path.name}: unexpected coefficient shape "
                         f"{coeff.shape}, expected (n1, n2, nz, 64)")

    coeff = _matlab_coeff_to_c(coeff)

    if psf is not None:
        psf = np.asarray(psf, dtype=np.float32)
        if psf.ndim == 3:
            psf = np.ascontiguousarray(psf.transpose(2, 0, 1))  # -> (nz, ny, nx)
        else:
            psf = None

    par = {}
    if params is not None:
        params = _as_struct(params)
        for name in getattr(params, "_fieldnames", []):
            par[name] = getattr(params, name)

    flipped = _smap_flipped(_to_scalar(par.get("emmirror")), em_on, path)
    pixelsize = par.get("pixelsize")
    if pixelsize is not None:
        pixelsize = np.asarray(_as_struct(np.asarray(pixelsize).reshape(-1)[0]
                                          if np.asarray(pixelsize).dtype == object
                                          else pixelsize), dtype=float).reshape(-1)

    cal = SplineCalibration(
        coeff=coeff,
        dz=float(_to_scalar(cspline.dz)),
        z0=float(_to_scalar(cspline.z0)),
        x0=_maybe_float(_to_scalar(getattr(cspline, "x0", None))),
        psf=psf,
        em_on=None if em_on is None else bool(em_on),
        pixelsize_um=pixelsize,
        source=path,
        parameters=par,
    )
    return _unmirror_x(cal) if flipped else cal


def _smap_flipped(emmirror, em_on, path) -> bool:
    """Whether SMAP flipped the bead images in x before building the spline.

    OWNER: CHECK ON A REAL FILE (tests/test_calibration.py,
    ``test_smaps_mirror_flag_and_its_record_of_the_flip_agree``, runs with
    SMAPPY_TEST_CAL).  This is read from SMAP's code; no real file was at hand.

    ``parameters.emmirror`` is the calibrator's popup, not the flip:
    1 "EM gain and mirror from metadata", 2 "no mirror", 3 "EM gain used
    (mirrored)", and 0 when the field was never set.  ``readbeadimages``
    flips for 3 always and for 1 only if the file's metadata says EM on; 0
    and 2 do nothing.  What it did comes back as ``isem``, which
    ``calibrate3D_g`` saves as ``SXY.EMon`` -- so ``EMon`` is SMAP's own
    record of the flip, and it decides.  (Until this was read, ``emmirror``
    was taken as a bool, so a file saved with "no mirror", 2, was flipped.)
    Without ``EMon`` (the older flat layout) only ``emmirror`` is left: 1 was
    the old checkbox "EM gain used (mirrored)" before the popup, so 1 and 3
    flip.  Without ``emmirror`` the file did not come from SMAP's bead
    calibrator, and nothing says it was flipped.
    """
    if emmirror is None:
        return False
    emmirror = int(emmirror)
    if em_on is None:
        return emmirror in (1, 3)
    flipped = bool(em_on)
    if (emmirror in (0, 2) and flipped) or (emmirror == 3 and not flipped):
        warnings.warn(
            f"{Path(path).name}: emmirror={emmirror} but EMon={int(flipped)}; "
            f"taking EMon, SMAP's record of the flip, so the spline is "
            f"{'' if flipped else 'not '}x-flipped", stacklevel=3)
    return flipped


# T[q, p] = (-1)^q C(p, q): sum_p a_p (1 - t)^p = sum_q (T a)_q t^q
_REFLECT = np.array([[(-1) ** q * comb(p, q) for p in range(4)] for q in range(4)])


def _unmirror_x(cal: SplineCalibration) -> SplineCalibration:
    """The same calibration in the camera's orientation: S'(u) = S(nx - u).

    The fitter samples ROI pixel j of an emitter at x at u = j - x + c + off
    on the spline's x axis, with c = (R-1)/2 and off = floor((nx+1-R)/2).  A
    spline of mirrored beads was fitted to the ROI flipped (j -> R-1-j,
    x -> R-1-x), i.e. sampled at x - j + c + off, which is nx - u whenever
    nx + 1 - R is even.  So S' is S reflected about the middle of its grid,
    nx/2: x cell k of S' is
    cell nx-1-k of S run backwards, t -> 1 - t, and each cell's cubic in x
    is re-expanded, b_q = (-1)^q sum_{p>=q} C(p, q) a_p on the px digit of
    16*pz + 4*py + px (z and y are untouched).  Flipping the signs of the odd
    terms alone is not this: (1 - t)^p is not (-t)^p.

    The ROI is always odd, so an odd nx makes nx + 1 - R odd and the fitter's
    offset one short; one more cell, a copy of the new last one, gives it
    floor((nx+2-R)/2) = off + 1, the one needed, and a sample clamped past
    that edge still meets the cell the flipped path clamped to.  Measured on
    synthetic splines: the model equals the flipped path to float32 rounding,
    and fits to the size of a one-ulp change in the coefficients
    (tests/test_unmirror.py).
    """
    _, nz, ny, nx = cal.coeff.shape
    c = cal.coeff.astype(np.float64).reshape(4, 4, 4, nz, ny, nx)  # pz,py,px,z,y,x
    c = np.einsum('qp,abpzyx->abqzyx', _REFLECT, c)[..., ::-1]
    # a native file's PSF is the spline's knots, nx+1 across; a SMAP file's is
    # its averaged bead stack, mirrored like the spline but of its own size
    psf = None if cal.psf is None else cal.psf[..., ::-1]
    if nx % 2:
        c = np.concatenate([c, c[..., -1:]], axis=-1)
        if psf is not None and psf.shape[-1] == nx + 1:
            psf = np.concatenate([psf, psf[..., -1:]], axis=-1)
    return replace(
        cal, coeff=np.ascontiguousarray(c.reshape(64, nz, ny, -1), np.float32),
        psf=None if psf is None else np.ascontiguousarray(psf),
        x0=None if cal.x0 is None else nx - cal.x0,
        parameters={**cal.parameters, "smappy_conversion":
                    "x-flipped on load: SMAP built this spline from x-mirrored beads"})


def _maybe_float(v):
    return None if v is None else float(v)


def _matlab_coeff_to_c(coeff: np.ndarray) -> np.ndarray:
    """``(n1, n2, nz, 64)`` MATLAB order -> ``(64, nz, ny, nx)`` C order.

    MATLAB's first spatial axis is the image row axis (y), so the two spatial
    axes are swapped.  The 64 coefficients are indexed ``16*pz + 4*p1 + p2``
    where ``p1`` belongs to the first spatial axis; swapping the spatial axes
    therefore requires swapping the ``p1``/``p2`` digits of that index as well.
    """
    n1, n2, nz, _ = coeff.shape
    # (n1, n2, nz, 64) -> (64, nz, n1, n2), with n1 = y, n2 = x
    out = np.ascontiguousarray(coeff.transpose(3, 2, 0, 1))
    # reorder the coefficient axis: i = 16*pz + 4*py + px, with py from axis n1
    idx = np.arange(64)
    pz, p1, p2 = idx // 16, (idx // 4) % 4, idx % 4
    out = out[16 * pz + 4 * p2 + p1]
    return np.ascontiguousarray(out, dtype=np.float32)


def compute_delta3d(dx: float, dy: float, dz: float) -> np.ndarray:
    """The 64 monomials ``dz**pz * dy**py * dx**px``, ordered as the coefficients.

    Mirrors ``kernel_computeDelta3D``; ``px`` is the fastest-varying digit.
    """
    powers = np.arange(4)
    cz = np.asarray(dz, dtype=np.float64) ** powers
    cy = np.asarray(dy, dtype=np.float64) ** powers
    cx = np.asarray(dx, dtype=np.float64) ** powers
    return (cz[:, None, None] * cy[None, :, None] * cx[None, None, :]).reshape(64)


def evaluate_spline(cal: SplineCalibration, x: float, y: float, z: float,
                    roisize: int) -> np.ndarray:
    """Render the normalized PSF model into a ``(roisize, roisize)`` image.

    ``x``/``y`` are the emitter position in ROI pixel coordinates and ``z`` is
    in spline index units.  This reproduces the model evaluation of
    ``kernel_splineMLEFit_z_EMCCD`` and exists so the loaded coefficients can be
    checked against the bead stack stored in the same file.
    """
    _, nz, ny, nx = cal.coeff.shape

    xc = -((x - roisize / 2) + 0.5)
    yc = -((y - roisize / 2) + 0.5)
    xstart, ystart, zstart = int(np.floor(xc)), int(np.floor(yc)), int(np.floor(z))
    delta = compute_delta3d(xc - xstart, yc - ystart, z - zstart)

    # SMAP centres the (smaller) ROI inside the (larger) spline grid
    off_x = int(np.floor((nx + 1 - roisize) / 2))
    off_y = int(np.floor((ny + 1 - roisize) / 2))

    ix = np.clip(np.arange(roisize) + xstart + off_x, 0, nx - 1)
    iy = np.clip(np.arange(roisize) + ystart + off_y, 0, ny - 1)
    iz = int(np.clip(zstart, 0, nz - 1))

    block = cal.coeff[:, iz][:, iy[:, None], ix[None, :]]  # (64, roisize, roisize)
    return np.tensordot(delta.astype(np.float32), block, axes=(0, 0))


def warn_on_em_mismatch(cal: SplineCalibration, data_em_on: Optional[bool]) -> None:
    """Warn if bead calibration and data were taken with different EM settings."""
    if cal.em_on is None or data_em_on is None:
        return
    if bool(cal.em_on) != bool(data_em_on):
        warnings.warn(
            f"EM gain mismatch: calibration was acquired with EM "
            f"{'on' if cal.em_on else 'off'} but the data with EM "
            f"{'on' if data_em_on else 'off'}. The PSF model may be mirrored "
            "relative to the data. Acquire both with the same EM setting.",
            stacklevel=2,
        )


def _validate_native_calibration(cal):
    c = np.asarray(cal.coeff)
    if c.ndim != 4 or c.shape[0] != 64 or min(c.shape[1:]) < 3 or not np.isfinite(c).all():
        raise ValueError('invalid spline coefficient array: expected finite (64,nz,ny,nx)')
    if not np.isfinite(cal.dz) or cal.dz <= 0:
        raise ValueError('calibration dz must be positive and finite')
    if not np.isfinite(cal.z0) or not 0 <= cal.z0 <= c.shape[1]:
        raise ValueError('calibration z0 is outside the spline grid')
    if cal.psf is not None:
        if cal.psf.shape != tuple(n+1 for n in c.shape[1:]) or not np.isfinite(cal.psf).all():
            raise ValueError('native PSF must contain the finite spline knots, shape (nz+1,ny+1,nx+1)')


def save_spline_calibration(path, cal, diagnostics=None, overwrite=False):
    """Save native HDF5, atomically; default refuses an existing destination.

    Arrays stay in fitter order. JSON records carry settings and acquisition
    provenance without Python pickles or MATLAB-specific structures.
    """
    import json
    import os
    import tempfile
    import h5py
    path = Path(path)
    if cal.psf is not None and cal.psf.shape != tuple(n + 1 for n in cal.coeff.shape[1:]):
        # a SMAP file's PSF is its averaged bead stack, not this spline's
        # knots, and the format holds only the knots
        cal = replace(cal, psf=None)
    _validate_native_calibration(cal)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(tmp, 'w') as f:
            f.attrs.update(format='smappy-spline-calibration', version=1,
                           coefficient_order='64,z,y,x;16*pz+4*py+px',
                           lateral_unit='pixel', axial_unit='nm',
                           dz=cal.dz, z0=cal.z0)
            if cal.x0 is not None:
                f.attrs['x0'] = cal.x0
            if cal.em_on is not None:
                f.attrs['em_on'] = bool(cal.em_on)
            f.create_dataset('coeff', data=cal.coeff, compression='gzip')
            if cal.psf is not None:
                f.create_dataset('psf', data=cal.psf, compression='gzip')
            f.create_dataset('parameters_json',
                             data=json.dumps(cal.parameters, default=_jsonable))
            g = f.create_group('diagnostics')
            for key, value in (diagnostics or {}).items():
                if key.endswith('_json'):
                    g.create_dataset(key, data=json.dumps(value))
                else:
                    a = np.asarray(value)
                    g.create_dataset(key, data=a, compression='gzip' if a.ndim else None)
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)  # no overwrite race with another writer
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _jsonable(value):
    """What ``json`` cannot write of a SMAP file's parameters: arrays, structs."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "_fieldnames"):  # a MATLAB struct, as scipy reads it
        return {name: getattr(value, name) for name in value._fieldnames}
    return str(value)


def _load_native_calibration(path):
    import json
    import h5py
    with h5py.File(path, 'r') as f:
        if f.attrs.get('format') == 'smappy-dual-color-calibration':
            raise ValueError('Dual-color calibration: use load_dual_color_calibration() and select .main or .secondary')
        if f.attrs.get('format') != 'smappy-spline-calibration' or f.attrs.get('version') != 1:
            raise ValueError(f'{path}: unsupported HDF5 calibration format/version')
        if f.attrs.get('coefficient_order') != '64,z,y,x;16*pz+4*py+px':
            raise ValueError('unsupported calibration coefficient convention')
        if f.attrs.get('lateral_unit') != 'pixel' or f.attrs.get('axial_unit') != 'nm':
            raise ValueError('unsupported calibration coordinate units')
        cal = SplineCalibration(
            coeff=np.ascontiguousarray(f['coeff'][...], dtype=np.float32),
            dz=float(f.attrs['dz']), z0=float(f.attrs['z0']),
            x0=f.attrs.get('x0'), psf=f['psf'][...] if 'psf' in f else None,
            source=Path(path),
            em_on=bool(f.attrs['em_on']) if 'em_on' in f.attrs else None,
            parameters=json.loads(f['parameters_json'][()]))
        # written by versions that kept a mirrored spline as it was
        mirrored = bool(f.attrs.get('em_mirror', False))
    _validate_native_calibration(cal)
    return _unmirror_x(cal) if mirrored else cal
