"""Where an astigmatic PSF is narrowest in x and where in y: the two focal lines and their separation.

A cylindrical lens splits the focus in two: above it the PSF is narrow in one direction, below it in the
other.  How far apart the two lines lie sets the z range and the z precision of a 3D calibration, and it is a
property of the instrument alone -- the lens and the camera's distance from it -- not of the beads or the
sample.  So it is the number to compare between calibrations of one microscope: when it moves, the lens or
the camera has moved (users computed it by hand from saved splines for exactly that, November 2026).

**How.**  The spline is sampled at its planes, as the fitter evaluates it (`io.calibration.evaluate_spline`,
the fitter's box, `crlb.ROISIZE`); at each the PSF is summed onto x and onto y and a Gaussian with an offset
fitted to each profile -- the offset takes the defocused rings' and the wings' light, which a second moment
would count as width.  The narrowest plane of each width is refined by a parabola through it and its two
neighbours, sampled at a fifth of the spline's step, so the answer does not snap to the spline's planes.
z is the spline's (the emitter's height above the plane in focus, `SplineCalibration.z_index_to_nm`).

A width whose least lies at the end of the sampled range has no minimum inside the calibration: then no
focal plane is given for it (None), rather than the edge of the stack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: the refinement's step, in spline planes
REFINE = 0.2


@dataclass
class FocalPlanes:
    """The z (nm, the spline's) at which the PSF is narrowest in x and in y, and the widths there (pixels)."""
    x_nm: float | None
    y_nm: float | None
    sigma_x_px: float = float("nan")
    sigma_y_px: float = float("nan")

    @property
    def separation_nm(self) -> float | None:
        """|z_y − z_x|; None where either has no minimum inside the spline."""
        if self.x_nm is None or self.y_nm is None:
            return None
        return abs(self.y_nm - self.x_nm)

    def text(self, short: bool = False) -> str:
        """*focal separation 412 nm (x narrowest at z −198 nm, y at +214 nm)*; a short form for a title."""
        def at(z):
            return "beyond the spline" if z is None else f"{z:+.0f} nm"
        sep = self.separation_nm
        head = "focal separation " + ("not found" if sep is None else f"{sep:.0f} nm")
        if short:
            return head + f" (x {at(self.x_nm)}, y {at(self.y_nm)})"
        return head + f" (x narrowest at z {at(self.x_nm)}, y at {at(self.y_nm)})"

    def as_dict(self) -> dict:
        return {"x_nm": self.x_nm, "y_nm": self.y_nm, "separation_nm": self.separation_nm,
                "sigma_x_px": self.sigma_x_px, "sigma_y_px": self.sigma_y_px}


def gaussian_sigma(profile) -> float:
    """σ (samples) of a Gaussian with an offset fitted to ``profile`` by least squares; NaN where the fit
    fails (a flat profile)."""
    from scipy.optimize import least_squares
    p = np.asarray(profile, float)
    if not np.isfinite(p).all() or p.max() <= p.min():
        return float("nan")
    i = np.arange(len(p), dtype=float)
    base = float(p.min())
    w = np.clip(p - base, 0, None)
    centre = float(np.sum(i * w) / np.sum(w))
    width = float(np.clip(np.sqrt(np.sum(w * (i - centre) ** 2) / np.sum(w)), 0.5, len(p) / 2))

    def residual(q):
        a, c, s, b = q
        return a * np.exp(-0.5 * ((i - c) / s) ** 2) + b - p
    start = (float(p.max() - base), centre, width, base)
    try:
        fit = least_squares(residual, start, bounds=([0, -1, 0.1, -np.inf], [np.inf, len(p), len(p), np.inf]))
    except ValueError:
        return float("nan")
    return float(fit.x[2]) if fit.success else float("nan")


def widths_px(cal, z_index, roisize: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(σ_x, σ_y) in pixels of the spline ``cal`` at each plane ``z_index`` (index units), each a
    Gaussian fitted to the PSF summed onto that axis (module docstring)."""
    from ..io.calibration import evaluate_spline
    from .crlb import ROISIZE
    roisize = int(roisize or ROISIZE)
    c = (roisize - 1) / 2
    sx, sy = [], []
    for z in np.atleast_1d(np.asarray(z_index, float)):
        image = np.asarray(evaluate_spline(cal, c, c, float(z), roisize), float)       # [y, x]
        sx.append(gaussian_sigma(image.sum(0)))
        sy.append(gaussian_sigma(image.sum(1)))
    return np.asarray(sx), np.asarray(sy)


def _narrowest(cal, z_index: np.ndarray, sigma: np.ndarray, which: int, roisize) -> tuple[float | None, float]:
    """(z index, σ) of the least of ``sigma`` over ``z_index``, refined (module docstring); (None, σ there)
    at an end of the range, (None, NaN) with no finite width."""
    if not np.isfinite(sigma).any():
        return None, float("nan")
    k = int(np.nanargmin(sigma))
    if k == 0 or k == len(sigma) - 1:
        return None, float(sigma[k])
    fine = np.arange(z_index[k - 1], z_index[k + 1] + 1e-9, REFINE)
    s = widths_px(cal, fine, roisize)[which]
    j = int(np.nanargmin(s))
    if 0 < j < len(s) - 1 and np.isfinite(s[j - 1:j + 2]).all():
        a, b, _ = np.polyfit(fine[j - 1:j + 2] - fine[j], s[j - 1:j + 2], 2)
        shift = -b / (2 * a) if a > 0 else 0.0
        return float(fine[j] + np.clip(shift, -REFINE, REFINE)), float(s[j])
    return float(fine[j]), float(s[j])


def focal_planes(cal, roisize: int | None = None) -> FocalPlanes:
    """`FocalPlanes` of the spline calibration ``cal`` (a `SplineCalibration`): where its PSF is narrowest
    in x and in y, over every plane but the outermost (whose cells extrapolate, `crlb.z_range_nm`)."""
    nz = cal.coeff.shape[1]
    z_index = np.arange(1, nz - 1, dtype=float)
    sx, sy = widths_px(cal, z_index, roisize)
    (zx, wx), (zy, wy) = (_narrowest(cal, z_index, s, which, roisize) for which, s in ((0, sx), (1, sy)))

    def nm(z):
        return None if z is None else float(cal.z_index_to_nm(z))
    return FocalPlanes(nm(zx), nm(zy), wx, wy)


def of_calibration(cal) -> list:
    """[(channel name, `FocalPlanes`)] of a spline calibration (one, named "") or of a dual one (main and
    secondary as the channel layer fits them, `DualColorCalibration.layer_splines`).  Kept on ``cal``
    (``_focal_planes``): the CRLB page, the summary and the saved record all ask, and a dual calibration's
    takes a second or so."""
    kept = getattr(cal, "_focal_planes", None)
    if kept is not None:
        return kept
    if hasattr(cal, "layer_splines"):
        main, secondary = cal.layer_splines()
        found = [("main", focal_planes(main)), ("secondary", focal_planes(secondary))]
    else:
        found = [("", focal_planes(cal))]
    try:
        cal._focal_planes = found
    except AttributeError:                  # a frozen calibration: found again next time
        pass
    return found


def summary_text(found: list, short: bool = False) -> str:
    """The `FocalPlanes` of `of_calibration` in a line, each channel named where there are two."""
    return "; ".join((f"{name}: " if name else "") + planes.text(short) for name, planes in found)


def record(found: list) -> dict:
    """What a saved run keeps of `of_calibration` (`saved_run.record_of`): {channel name or "psf": values}."""
    return {name or "psf": planes.as_dict() for name, planes in found}
