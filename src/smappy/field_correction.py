"""A field-dependent PSF fitted with one spline, its bias corrected afterwards: dx, dy, dz(x, y, z_fit).

A field-dependent calibration (`field_splines`) holds a spline per node of a grid over the chip, and
Spline 3D blends them per spot.  Anything that fits with one PSF for the whole chip -- the centre spline
-- has a bias that varies over the field: mostly a z that is scaled and bent differently at each place.
That is DECODE's network (trained on the centre spline), Spline 3D told to fit with the centre spline,
and any fitter without the field.  The simulation study (`docs/uipsf/field-dependence.md` §11) found that
the bias can be subtracted afterwards, as a smooth function of where the spot was fitted:

* **From the model, not from beads.**  The correction is measured here on noise-free spots drawn with the
  node splines -- which are the learnt pupil and its field, drawn for the sample's depth where the fit's
  spline was (`calibrate.depth.depth_spline` redraws the nodes too) -- and fitted with the centre spline.
  The study's model correction left 0.3–0.6 nm rms on M5's field, a correction from 40 beads 2–3 nm; and
  beads bring their size, their SAF and their own positions, which the model has not.
* **At the data's brightness.**  The bias of a mismatched PSF moves a little with photons and background
  (the Poisson weights shift between the spot's core and its tails): a correction measured at 20 000
  photons applied at 1000 left 3–6 nm, one measured at the data's 0.2–2 nm (§11).  So the spots are drawn
  at the table's median photons and background.
* **A function of what was fitted**: x, y on the chip and z_fit, never the truth, which the data does not
  have.  A polynomial of order `UV_ORDER` in the chip position times Legendre polynomials of z to
  `Z_DEGREE`: the z bias wiggles along z, and degree 6 left 21 nm where 14 left 2.7 (§11).
* **Never extrapolated.**  Outside the grid of node splines, or outside the z range it was measured on, a
  localization takes the value at the edge (`FieldCorrection.__call__` clamps); the caller says how many.
* **Precision is not given back.**  The mismatched PSF lost information; the correction moves the mean.
  From about twice the study's synthetic field even the mean cannot be recovered (a fitted z that is no
  longer a smooth function of the true one): `FieldCorrection.residual_nm` measures how well the
  polynomial followed the bias, and `LIMIT_NM` is where it is said to have failed.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: degree in z of the Legendre polynomials (§11: 6 left 21 nm of wiggle, 14 left 2.7)
Z_DEGREE = 14
#: order in the chip position: the model correction at order 4 left 0.3–0.6 nm on mild fields (§11)
UV_ORDER = 4
#: positions per side of the measuring grid over the nodes' field, and its z step, nm
GRID, DZ_NM = 11, 25.0
#: the z range measured: the spline's, less this share of it at each end, where a fit runs to the edge
Z_MARGIN = 0.1
#: a fit counts as converged within this many nm of the truth in z
LOST_NM = 300.0
#: the polynomial's own rms residual in z above which the field is too strong for a correction (§11)
LIMIT_NM = 5.0


@dataclass
class FieldCorrection:
    """dx, dy (camera pixels) and dz (nm) as functions of the fitted chip position (pixels) and z (nm)."""
    coef: np.ndarray                    # (3, M) the polynomial's coefficients, x and y in pixels, z in nm
    x_range: tuple                      # (lo, hi) chip pixels the correction was measured over
    y_range: tuple
    z_range: tuple                      # (lo, hi) nm
    residual_nm: float = float("nan")   # rms of z bias less the polynomial on the measuring grid
    residual_max_nm: float = float("nan")
    lost: float = 0.0                   # the share of measuring spots whose fit did not converge
    photons: float = float("nan")
    background: float = float("nan")
    record: dict = field(default_factory=dict)

    def coordinates(self, x_px, y_px, z_nm):
        """(u, v, t) in [−1, 1] and whether each point was outside the measured ranges."""
        def scaled(a, lo, hi):
            a = np.asarray(a, float)
            mid, half = (lo + hi) / 2, max((hi - lo) / 2, 1e-9)
            s = (a - mid) / half
            return np.clip(s, -1.0, 1.0), ~(np.abs(s) <= 1.0 + 1e-9)
        u, ou = scaled(x_px, *self.x_range)
        v, ov = scaled(y_px, *self.y_range)
        t, ot = scaled(z_nm, *self.z_range)
        return u, v, t, ou | ov | ot

    def __call__(self, x_px, y_px, z_nm):
        """((n, 3) dx px, dy px, dz nm, outside (n,) bool): the bias at each fitted position, the edge's
        value where it is outside what was measured (never extrapolated).  Subtract it."""
        u, v, t, outside = self.coordinates(x_px, y_px, z_nm)
        B = basis(u, v, t)
        return B @ self.coef.T, outside

    def summary(self) -> dict:
        """What a table records of the correction that moved it (its metadata's ``field_correction``)."""
        return {**self.record, "x_range_px": list(map(float, self.x_range)),
                "y_range_px": list(map(float, self.y_range)), "z_range_nm": list(map(float, self.z_range)),
                "photons": float(self.photons), "background": float(self.background),
                "residual_rms_nm": float(self.residual_nm), "residual_max_nm": float(self.residual_max_nm),
                "lost": float(self.lost), "uv_order": UV_ORDER, "z_degree": Z_DEGREE}

    @property
    def failed(self) -> bool:
        """Whether the field is too strong for one spline and a correction (module docstring)."""
        return not (self.residual_nm <= LIMIT_NM) or self.lost > 0.02


def basis(u, v, t) -> np.ndarray:
    """(n, M): the monomials u^i v^j, i + j ≤ `UV_ORDER`, times Legendre P0..P`Z_DEGREE` of t, all in [−1, 1]."""
    from numpy.polynomial import legendre
    u, v = np.asarray(u, float).reshape(-1), np.asarray(v, float).reshape(-1)
    uv = np.stack([u ** (n - j) * v ** j for n in range(UV_ORDER + 1) for j in range(n + 1)], 1)
    pz = legendre.legvander(np.asarray(t, float).reshape(-1), Z_DEGREE)
    return (uv[:, :, None] * pz[:, None, :]).reshape(len(u), -1)


def _render(coeff: np.ndarray, z_index: np.ndarray, roi: int) -> np.ndarray:
    """(nz, roi, roi): the spline ``coeff`` (64, nz, ny, nx) of a spot on the ROI's central pixel at each z
    index, as `io.calibration.evaluate_spline` draws it."""
    from .io.calibration import compute_delta3d
    _, nz, ny, nx = coeff.shape
    c = (roi - 1) / 2
    xc = -((c - roi / 2) + 0.5)
    xs = int(np.floor(xc))
    ox, oy = int(np.floor((nx + 1 - roi) / 2)), int(np.floor((ny + 1 - roi) / 2))
    ix = np.clip(np.arange(roi) + xs + ox, 0, nx - 1)
    iy = np.clip(np.arange(roi) + xs + oy, 0, ny - 1)
    out = np.empty((len(z_index), roi, roi), np.float64)
    for k, z in enumerate(np.asarray(z_index, float)):
        zs = int(np.floor(z))
        delta = compute_delta3d(xc - xs, xc - xs, z - zs)
        block = coeff[:, int(np.clip(zs, 0, nz - 1))][:, iy[:, None], ix[None, :]]
        out[k] = np.tensordot(delta, block, axes=(0, 0))
    return out


def measure(cal, photons: float, background: float, roi: int = 13, grid: int = GRID,
            n_threads: int = 0, iterations: int = 50) -> FieldCorrection:
    """The correction of ``cal``'s centre spline for its field (``cal.field_splines``), at ``photons`` and
    ``background`` per pixel, on a ``roi``-pixel fit box: noise-free spots drawn with the node splines on a
    ``grid`` × ``grid`` raster over the nodes' field and every `DZ_NM` in z, fitted with the centre spline."""
    from .calibrate.depth import in_focus, z_bounds
    from .field_splines import bilinear
    from .psf import SplinePSF
    fs = getattr(cal, "field_splines", None)
    if fs is None:
        raise ValueError("the calibration has no field dependence: there is nothing to correct")
    lateral = min(cal.shape[1:])
    if roi > lateral - 2:
        raise ValueError(f"the fit box ({roi} px) must be at least 2 px smaller than the calibration "
                         f"({lateral} px across)")
    lo, hi = z_bounds(cal)
    margin = Z_MARGIN * (hi - lo)
    z_lo, z_hi = lo + margin, hi - margin
    z_true = np.arange(z_lo, z_hi + 1e-6, DZ_NM)
    z_index = cal.z_nm_to_index(z_true)
    nodes = [_render(fs.coeff[g], z_index, roi) for g in range(fs.coeff.shape[0])]
    xs = np.linspace(fs.xs[0], fs.xs[-1], grid if len(fs.xs) > 1 else 1)
    ys = np.linspace(fs.ys[0], fs.ys[-1], grid if len(fs.ys) > 1 else 1)
    X, Y = (a.reshape(-1) for a in np.meshgrid(xs, ys, indexing="ij"))
    idx, w = bilinear(fs, np.c_[X, Y])
    images = np.stack([sum(float(w[p, k]) * nodes[idx[p, k]] for k in range(4)) for p in range(len(X))])
    spots = (float(photons) * np.maximum(images, 0) + float(background)).astype(np.float32)
    spots = spots.reshape(-1, roi, roi)
    centre = copy.copy(cal)
    centre.field_splines = None
    fitter = SplinePSF(centre, z_start_nm=in_focus(cal))
    out = fitter.unpack(fitter.fit(spots, iterations, n_threads))
    c = (roi - 1) / 2
    n = len(z_true)
    x_fit = np.repeat(X, n) + (out["x_roi"] - c)
    y_fit = np.repeat(Y, n) + (out["y_roi"] - c)
    z_fit = out["z_nm"]
    err = np.c_[out["x_roi"] - c, out["y_roi"] - c, z_fit - np.tile(z_true, len(X))]
    converged = np.isfinite(err).all(1) & (np.abs(err[:, 2]) < LOST_NM)
    # a fit that went past the measured range is not lost, only not where the polynomial is fitted
    good = converged & (z_fit >= z_lo) & (z_fit <= z_hi)
    x_range = (float(xs[0]), float(xs[-1])) if len(xs) > 1 else (float(xs[0]) - 1, float(xs[0]) + 1)
    y_range = (float(ys[0]), float(ys[-1])) if len(ys) > 1 else (float(ys[0]) - 1, float(ys[0]) + 1)
    result = FieldCorrection(np.zeros((3, basis([0], [0], [0]).shape[1])), x_range, y_range, (z_lo, z_hi),
                             photons=float(photons), background=float(background))
    u, v, t, _ = result.coordinates(x_fit[good], y_fit[good], z_fit[good])
    B = basis(u, v, t)
    result.coef = np.stack([np.linalg.lstsq(B, err[good, a], rcond=None)[0] for a in range(3)])
    left = err[good, 2] - B @ result.coef[2]
    result.residual_nm = float(np.sqrt(np.mean(left ** 2))) if len(left) else float("nan")
    result.residual_max_nm = float(np.abs(left).max()) if len(left) else float("nan")
    result.lost = float(1 - converged.mean())
    result.record = {"grid": [len(fs.ys), len(fs.xs)], "fit_box_px": int(roi),
                     "uncorrected_rms_nm": float(np.sqrt(np.mean(err[good, 2] ** 2))) if good.any() else None}
    return result


_KEPT: dict = {}


def kept(cal, photons: float, background: float, roi: int = 13, n_threads: int = 0) -> FieldCorrection:
    """`measure`, kept for the last few (calibration, depth, brightness, box): a batch asks for the same one
    with every file.  The brightness is rounded to 5 %: closer is within the noise of what it moves (§11)."""
    source = getattr(cal, "source", None)
    stamp = Path(source).stat().st_mtime if source and Path(source).is_file() else id(cal)
    depth = json.dumps((cal.parameters or {}).get("sample_depth"), sort_keys=True, default=str)
    key = (str(source), stamp, depth, round(float(np.log(max(photons, 1.0)) / 0.05)),
           round(float(np.log(max(background, 0.1)) / 0.05)), int(roi))
    if key not in _KEPT:
        if len(_KEPT) >= 4:
            _KEPT.pop(next(iter(_KEPT)))
        _KEPT[key] = measure(cal, photons, background, roi, n_threads=n_threads)
    return _KEPT[key]


def has_field(path) -> bool:
    """Whether the calibration file at ``path`` is field-dependent, without loading its splines."""
    if not path:
        return False
    p = Path(path)
    if p.suffix.lower() != ".h5" or not p.is_file():
        return False
    try:
        import h5py
        with h5py.File(p, "r") as f:
            return "field" in f
    except Exception:
        return False


def corrected(x_nm, y_nm, z_nm, correction: FieldCorrection, pixel_nm) -> tuple:
    """(x, y, z) nm less the bias at each, and the mask of those outside what was measured.  ``x_nm``,
    ``y_nm`` are on the camera chip (a fit's), ``pixel_nm`` its (x, y) pixel size."""
    px, py = pixel_nm
    d, outside = correction(np.asarray(x_nm, float) / px, np.asarray(y_nm, float) / py, z_nm)
    return (np.asarray(x_nm, float) - d[:, 0] * px, np.asarray(y_nm, float) - d[:, 1] * py,
            np.asarray(z_nm, float) - d[:, 2], outside)
