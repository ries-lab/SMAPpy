"""z from the widths of an elliptical Gaussian fit: astigmatic 3D without a spline.

Behind a cylindrical lens a spot is wide in x above focus, wide in y below and
round in between (Huang et al., Science 2008).  The elliptical Gaussian fitter
measures both widths and writes no z; this turns them into one, given how the
widths change with z -- a calibration of one curve per axis,

    s(z) = s0 sqrt(1 + u^2 + A u^3 + B u^4),   u = (z - c) / d,

the defocus curve of a Gaussian beam (A = B = 0) with Huang's two terms for
what a real objective adds.  The calibration is typed (the simulator's own
model is one) or fitted to beads stepped through focus (`fit_calibration`).
Its z is 0 where the spot is round, so a calibration from beads and a typed
one agree on what z = 0 means.

**z per localization** is the least-squares z of both widths at once,

    chi^2(z) = ((sx - Sx(z)) / dsx)^2 + ((sy - Sy(z)) / dsy)^2,

weighted by the fit's own errors of the widths (their CRLB).  Huang 2008
minimised the distance between square roots of the widths, a stand-in for
the weights the fit now reports.  SMAP's other way, a lookup of sx^2 - sy^2
against z, uses only the difference of the widths: it finds a z for a spot
too wide in both axes as readily as for a good one, and has no error.  Here
it is only the start: within the stretch where sx^2 - sy^2 is one-to-one
with z (all of it for the pure model, where it is linear in z) it gives each
localization a z in one interpolation, and six Gauss-Newton steps on chi^2
refine it.  A grid search starts the few outside that stretch.  A million
localizations take about 0.6 s; a grid search to start all of them took 3.5 s.

**The error** is the CRLB's, carried through the curves' slopes,
1/dz^2 = sum (S'(z) / ds)^2, so it is large where both widths are flat in z.
With two widths and one z, chi^2 at the minimum is chi-squared with one
degree of freedom when the model fits; a larger one means the widths agree on
no z -- an overlap of two spots, a spot from out of the calibrated range --
and the error is multiplied by sqrt(chi^2) (when it exceeds 1: Birge's
ratio), so a filter on `z_err_nm` takes such a spot out without a flag
column.  On simulated frames, fitted, that leaves the error about a fifth
larger than the scatter of z about the truth: a little conservative.  A z that ends on an edge of the calibrated range is not a
measurement of z at all, and gets the whole range as its error.

A fit with no width errors (a file from elsewhere) gets one error for every
width, estimated from the residuals: the median of chi^2 with one degree of
freedom is 0.455.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

#: the median of chi-squared with one degree of freedom, for a width error from the residuals
CHI2_1_MEDIAN = 0.4549364
#: Gauss-Newton steps from the lookup's start: 4 reach float precision on the pure model
ITERATIONS = 6
#: the grid the lookup and the grid search are tabulated on
GRID_POINTS = 2001
COARSE_POINTS = 201
#: HDF5 group the calibration is saved under
GROUP = "astigmatic_z"


@dataclass(frozen=True)
class Defocus:
    """One axis' width against z: ``s0 sqrt(1 + u^2 + a u^3 + b u^4)``, ``u = (z - c) / d``."""
    s0: float
    c: float
    d: float
    a: float = 0.0
    b: float = 0.0

    def _poly(self, z):
        """u and the polynomial under the root, by Horner: numpy's powers cost 20 times as much."""
        u = (np.asarray(z, dtype=np.float64) - self.c) / self.d
        if self.a or self.b:
            poly = 1 + u * u * (1 + u * (self.a + self.b * u))
        else:
            poly = 1 + u * u
        # the higher terms can take the polynomial below zero far from focus; a width there is
        # meaningless, and a floor keeps the square root and its derivative finite
        return u, np.maximum(poly, 1e-6)

    def width(self, z):
        return self.s0 * np.sqrt(self._poly(z)[1])

    def width_and_slope(self, z):
        """S(z) and dS/dz, sharing the work."""
        u, poly = self._poly(z)
        root = np.sqrt(poly)
        if self.a or self.b:
            dpoly = u * (2 + u * (3 * self.a + 4 * self.b * u))
        else:
            dpoly = 2 * u
        return self.s0 * root, self.s0 * dpoly / (2 * self.d * root)

    def slope(self, z):
        """dS/dz."""
        return self.width_and_slope(z)[1]

    def shifted(self, dz: float) -> "Defocus":
        return Defocus(self.s0, self.c - dz, self.d, self.a, self.b)


@dataclass(frozen=True)
class AstigmaticCalibration:
    """The widths' curves in x and y, and the z range they were measured over (nm, z = 0 round)."""
    x: Defocus
    y: Defocus
    z_range: Tuple[float, float]
    source: str = ""
    #: what a fit to beads leaves: the rms residual of each axis, nm
    residual_nm: Tuple[float, float] = (float("nan"), float("nan"))
    beads: int = 0

    def widths(self, z):
        return self.x.width(z), self.y.width(z)

    def save(self, path) -> Path:
        import h5py
        path = Path(path)
        with h5py.File(path, "w") as f:
            g = f.create_group(GROUP)
            for axis in ("x", "y"):
                for name, value in asdict(getattr(self, axis)).items():
                    g.attrs[f"{axis}_{name}"] = value
            g.attrs["z_range"] = np.asarray(self.z_range, dtype=np.float64)
            g.attrs["residual_nm"] = np.asarray(self.residual_nm, dtype=np.float64)
            g.attrs["beads"] = int(self.beads)
            g.attrs["source"] = self.source
        return path

    @classmethod
    def load(cls, path) -> "AstigmaticCalibration":
        import h5py
        with h5py.File(path, "r") as f:
            if GROUP not in f:
                raise ValueError(f"{Path(path).name} is not an astigmatic z calibration: make one with the "
                                 "PSF calibration window's method elliptical Gaussian")
            g = f[GROUP].attrs
            axes = {axis: Defocus(*(float(g[f"{axis}_{n}"]) for n in ("s0", "c", "d", "a", "b")))
                    for axis in ("x", "y")}
            return cls(axes["x"], axes["y"], tuple(float(v) for v in g["z_range"]), str(g.get("source", "")),
                       tuple(float(v) for v in g.get("residual_nm", (np.nan, np.nan))), int(g.get("beads", 0)))


def is_calibration(path) -> bool:
    """Whether ``path`` is a file `AstigmaticCalibration.save` wrote."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return GROUP in f
    except (OSError, ValueError):
        return False


def simulator_calibration(sigma_nm: float = 130.0, focal_offset_nm: float = 300.0, depth_nm: float = 400.0,
                          z_range_nm: float = 800.0) -> AstigmaticCalibration:
    """`simulate.camera.astigmatic_sigmas` as a calibration: x in focus at +offset, y at -offset."""
    return AstigmaticCalibration(Defocus(sigma_nm, focal_offset_nm, depth_nm),
                                 Defocus(sigma_nm, -focal_offset_nm, depth_nm),
                                 (-z_range_nm, z_range_nm), "simulator's model")


def _crossing(x: Defocus, y: Defocus, lo: float, hi: float) -> float:
    """The z in [lo, hi] where the spot is round, nearest the middle of the range."""
    z = np.linspace(lo, hi, GRID_POINTS)
    diff = x.width(z) - y.width(z)
    flips = np.flatnonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) <= 0)
    if not len(flips):
        raise ValueError("the widths in x and y never cross: the beads are not astigmatic, or the stack "
                         "does not reach the round spot between the two foci")
    i = flips[np.argmin(np.abs(z[flips] - 0.5 * (lo + hi)))]
    # linear between the two samples that bracket it
    d0, d1 = diff[i], diff[i + 1]
    return float(z[i] + (z[i + 1] - z[i]) * (d0 / (d0 - d1) if d0 != d1 else 0.5))


def _fit_axis(z, s, err, higher_orders: bool) -> Defocus:
    from scipy.optimize import least_squares
    order = np.argsort(s)
    s0 = float(np.median(s[order[:max(3, len(s) // 20)]]))
    c = float(np.median(z[order[:max(3, len(s) // 20)]]))
    # the depth where the width has grown by sqrt(2): from the points between 1.3 and 1.5 s0
    near = np.abs(s / s0 - np.sqrt(2)) < 0.1
    d = float(np.median(np.abs(z[near] - c))) if near.any() else 0.25 * float(np.ptp(z))
    start = [s0, c, max(d, 50.0)] + ([0.0, 0.0] if higher_orders else [])

    def residuals(p):
        axis = Defocus(*p) if higher_orders else Defocus(p[0], p[1], p[2])
        return (s - axis.width(z)) / err

    # soft_l1: a bead that touched another, or a plane where a fit failed, is not let pull the curve
    fit = least_squares(residuals, start, loss="soft_l1", f_scale=3.0,
                        bounds=([1.0, -np.inf, 1.0] + [-np.inf] * 2 * higher_orders,
                                [np.inf] * (5 if higher_orders else 3)))
    values = [float(v) for v in fit.x]
    return Defocus(*values) if higher_orders else Defocus(*values[:3])


def fit_calibration(z_nm, sx_nm, sy_nm, sx_err_nm=None, sy_err_nm=None, higher_orders: bool = True,
                    source: str = "", beads: int = 0) -> AstigmaticCalibration:
    """The curves of both widths from beads through focus: ``z_nm`` each measurement's z in the
    sample (the objective's position turned around: raised by dz, it leaves a bead dz below focus),
    the beads' widths and, if known, their errors.  z is then moved so that 0 is where the spot is
    round, and the range is the beads'.  ``higher_orders`` fits Huang's A and B as well."""
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    sx = np.asarray(sx_nm, dtype=np.float64).ravel()
    sy = np.asarray(sy_nm, dtype=np.float64).ravel()
    ex = np.ones_like(sx) if sx_err_nm is None else np.asarray(sx_err_nm, dtype=np.float64).ravel()
    ey = np.ones_like(sy) if sy_err_nm is None else np.asarray(sy_err_nm, dtype=np.float64).ravel()
    ok = np.isfinite(z) & np.isfinite(sx) & np.isfinite(sy) & (ex > 0) & (ey > 0) & (sx > 0) & (sy > 0)
    if ok.sum() < (10 if higher_orders else 6):
        raise ValueError(f"{int(ok.sum())} widths to calibrate from: too few")
    z, sx, sy, ex, ey = z[ok], sx[ok], sy[ok], ex[ok], ey[ok]
    x = _fit_axis(z, sx, ex, higher_orders)
    y = _fit_axis(z, sy, ey, higher_orders)
    lo, hi = float(z.min()), float(z.max())
    zero = _crossing(x, y, lo, hi)
    x, y = x.shifted(zero), y.shifted(zero)
    rx = float(np.sqrt(np.mean((sx - x.width(z - zero)) ** 2)))
    ry = float(np.sqrt(np.mean((sy - y.width(z - zero)) ** 2)))
    return AstigmaticCalibration(x, y, (lo - zero, hi - zero), source, (rx, ry), beads)


def _one_to_one(z, d):
    """The stretch of ``d(z)`` through the round spot where it is monotonic, increasing: (z, d)."""
    i0 = int(np.argmin(np.abs(z)))
    i0 = min(max(i0, 1), len(z) - 2)
    sign = np.sign(d[i0 + 1] - d[i0 - 1]) or 1.0
    step = sign * np.diff(d)
    left = i0
    while left > 0 and step[left - 1] > 0:
        left -= 1
    right = i0
    while right < len(step) and step[right] > 0:
        right += 1
    zs, ds = z[left:right + 1], d[left:right + 1]
    return (zs, ds) if sign > 0 else (zs[::-1], ds[::-1])


def _grid_search(sx, sy, wx, wy, cal: AstigmaticCalibration, chunk: int = 50_000):
    z = np.linspace(*cal.z_range, COARSE_POINTS)
    gx, gy = cal.widths(z)
    out = np.empty(len(sx))
    for i in range(0, len(sx), chunk):
        part = slice(i, i + chunk)
        chi = (wx[part, None] * (sx[part, None] - gx) ** 2 + wy[part, None] * (sy[part, None] - gy) ** 2)
        out[part] = z[np.argmin(chi, axis=1)]
    return out


def _refine(z, sx, sy, wx, wy, cal: AstigmaticCalibration, iterations: int = ITERATIONS):
    lo, hi = cal.z_range
    for _ in range(iterations):
        (wdx, jx), (wdy, jy) = cal.x.width_and_slope(z), cal.y.width_and_slope(z)
        rx, ry = sx - wdx, sy - wdy
        curvature = wx * jx * jx + wy * jy * jy
        step = (wx * rx * jx + wy * ry * jy) / np.maximum(curvature, 1e-30)
        # no step longer than a tenth of the range: from a poor start a flat stretch of both curves
        # would otherwise throw z across the range
        limit = 0.1 * (hi - lo)
        z = np.clip(z + np.clip(step, -limit, limit), lo, hi)
    return z


@dataclass
class ZFromWidths:
    """What `z_from_widths` found: z and its error (nm), chi^2 at the minimum, which rows ended on an
    edge of the range (or had no widths), and the width error used where none was given."""
    z_nm: np.ndarray
    z_err_nm: np.ndarray
    chi2: np.ndarray
    edge: np.ndarray
    width_err_nm: Optional[float] = None
    extra: dict = field(default_factory=dict)


def z_from_widths(sx_nm, sy_nm, cal: AstigmaticCalibration, sx_err_nm=None, sy_err_nm=None) -> ZFromWidths:
    """Each localization's z from its widths: least squares on both, started from the lookup of
    sx^2 - sy^2 (see the module's docstring).  Errors in, or one estimated from the residuals."""
    sx = np.asarray(sx_nm, dtype=np.float64)
    sy = np.asarray(sy_nm, dtype=np.float64)
    lo, hi = cal.z_range
    given = sx_err_nm is not None and sy_err_nm is not None
    if given:
        ex = np.asarray(sx_err_nm, dtype=np.float64)
        ey = np.asarray(sy_err_nm, dtype=np.float64)
        bad_err = ~(np.isfinite(ex) & np.isfinite(ey) & (ex > 0) & (ey > 0))
    else:
        ex = ey = np.ones_like(sx)
        bad_err = np.zeros(len(sx), dtype=bool)
    ok = np.isfinite(sx) & np.isfinite(sy) & ~bad_err
    sx_ok, sy_ok = sx[ok], sy[ok]
    wx, wy = 1.0 / ex[ok] ** 2, 1.0 / ey[ok] ** 2

    grid = np.linspace(lo, hi, GRID_POINTS)
    gx, gy = cal.widths(grid)
    zs, ds = _one_to_one(grid, gx * gx - gy * gy)
    d = sx_ok * sx_ok - sy_ok * sy_ok
    z = np.interp(d, ds, zs)
    outside = (d < ds[0]) | (d > ds[-1])
    if outside.any():
        z[outside] = _grid_search(sx_ok[outside], sy_ok[outside], wx[outside], wy[outside], cal)
    z = _refine(z, sx_ok, sy_ok, wx, wy, cal)

    (wdx, jx), (wdy, jy) = cal.x.width_and_slope(z), cal.y.width_and_slope(z)
    rx, ry = sx_ok - wdx, sy_ok - wdy
    width_err = None
    if not given:
        # one error for every width, from the residuals: chi^2 with one degree of freedom
        sq = rx * rx + ry * ry
        width_err = float(np.sqrt(np.median(sq) / CHI2_1_MEDIAN)) if len(sq) else float("nan")
        width_err = width_err if width_err > 0 else 1.0
        wx = wy = np.full(len(z), 1.0 / width_err ** 2)
    chi2 = wx * rx * rx + wy * ry * ry
    err = 1.0 / np.sqrt(np.maximum(wx * jx * jx + wy * jy * jy, 1e-30))
    err *= np.sqrt(np.maximum(chi2, 1.0))
    # an edge is not a measurement: the true z may be anywhere beyond it
    tol = 1e-6 * (hi - lo) + 1e-3
    on_edge = (z <= lo + tol) | (z >= hi - tol)
    err[on_edge] = hi - lo

    n = len(sx)
    out = ZFromWidths(np.full(n, np.nan), np.full(n, hi - lo), np.full(n, np.nan), ~ok, width_err)
    out.z_nm[ok], out.z_err_nm[ok], out.chi2[ok] = z, np.minimum(err, hi - lo), chi2
    out.edge[ok] = on_edge
    return out
